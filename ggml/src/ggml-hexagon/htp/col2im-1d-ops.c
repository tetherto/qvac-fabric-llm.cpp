#include <HAP_farf.h>

#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-dma.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"

// Output positions handled per (band, channel) pass. A band's columns stay
// cache resident while every channel gathers from them.
#define COL2IM_1D_BAND 256

struct htp_col2im_1d_context {
    struct htp_ops_context * octx;
    uint32_t                 stride;
    uint32_t                 channels;
    uint32_t                 kernel;
    uint32_t                 crop;
    uint32_t                 bands_per_thread;
    uint32_t                 n_bands;
};

static inline float col2im_1d_load(const struct htp_tensor * cols, uint32_t row, uint32_t col) {
    const uint8_t * p = (const uint8_t *) cols->data + row * cols->nb[0] + col * cols->nb[1];
    return cols->type == HTP_TYPE_F16 ? (float) *(const __fp16 *) p : *(const float *) p;
}

// Sum of every (column, tap) pair landing on one output position, walking taps
// k, k + stride, ... from the last contributing column backwards.
static float col2im_1d_gather(const struct htp_col2im_1d_context * cctx, uint32_t row0, int64_t col, uint32_t k) {
    const struct htp_tensor * cols   = cctx->octx->src[0];
    const uint32_t            n_cols = cols->ne[1];

    float sum = 0.0f;
    for (; k < cctx->kernel && col >= 0; k += cctx->stride, col--) {
        if ((uint64_t) col < n_cols) {
            sum += col2im_1d_load(cols, row0 + k, (uint32_t) col);
        }
    }
    return sum;
}

static inline void col2im_1d_store(const struct htp_tensor * dst, uint8_t * row, uint32_t t, float v) {
    if (dst->type == HTP_TYPE_F16) {
        *(__fp16 *) (row + t * dst->nb[0]) = (__fp16) v;
    } else {
        *(float *) (row + t * dst->nb[0]) = v;
    }
}

// The last contributing column and its tap advance by one position at a time,
// so only the band start needs a division (Hexagon has no integer divider).
static void col2im_1d_band_channel(const struct htp_col2im_1d_context * cctx, uint32_t oc, uint32_t t0, uint32_t t1) {
    const struct htp_tensor * dst  = cctx->octx->dst;
    uint8_t *                 row  = (uint8_t *) dst->data + oc * dst->nb[1];
    const uint32_t            row0 = oc * cctx->kernel;

    const uint32_t t_abs = t0 + cctx->crop;
    int64_t        col   = t_abs / cctx->stride;
    uint32_t       k     = t_abs - (uint32_t) col * cctx->stride;
    for (uint32_t t = t0; t < t1; t++) {
        col2im_1d_store(dst, row, t, col2im_1d_gather(cctx, row0, col, k));
        if (++k == cctx->stride) {
            k = 0;
            col++;
        }
    }
}

static void col2im_1d_band(const struct htp_col2im_1d_context * cctx, uint32_t band) {
    const uint32_t t_out = cctx->octx->dst->ne[0];
    const uint32_t t0    = band * COL2IM_1D_BAND;
    const uint32_t t1    = MIN(t0 + COL2IM_1D_BAND, t_out);
    for (uint32_t oc = 0; oc < cctx->channels; oc++) {
        col2im_1d_band_channel(cctx, oc, t0, t1);
    }
}

static void col2im_1d_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_col2im_1d_context * cctx = (const struct htp_col2im_1d_context *) data;

    const uint32_t start = cctx->bands_per_thread * ith;
    const uint32_t end   = MIN(start + cctx->bands_per_thread, cctx->n_bands);
    for (uint32_t band = start; band < end; band++) {
        col2im_1d_band(cctx, band);
    }
}

// F32 overlap-add through VTCM for kernels that are a whole number m of
// strides (every transposed-conv upsampler). Output position a = q * s + r
// sums taps k = r + j * s of columns q - j, j < m. Thirty-two consecutive
// outputs therefore read one fixed pattern of column offsets per phase r,
// so each tap group is one word gather from the staged column block, which
// a single DMA brings in with zero rows for columns outside the signal.
#define COL2IM_1D_VTCM_MAX_BAND 256
#define COL2IM_1D_VTCM_MIN_BAND 32
#define COL2IM_1D_MAX_STRIDE    32

struct htp_col2im_1d_vtcm {
    struct htp_col2im_1d_context base;
    uint32_t                     taps_per_output;
    uint32_t                     band;
    uint32_t                     off_tmp;
    uint32_t                     off_out;
    uint32_t                     bytes_per_thread;
    uint32_t                     plan_bytes;
    HVX_Vector *                 phase_offsets;
};

static uint32_t col2im_1d_band_cols(const struct htp_col2im_1d_vtcm * v, uint32_t band) {
    return band / v->base.stride + v->taps_per_output + 1;
}

static bool col2im_1d_vtcm_fit(struct htp_col2im_1d_vtcm * v, uint32_t n_threads, size_t vtcm_size) {
    const uint32_t k_oc = v->base.kernel * v->base.channels;
    v->plan_bytes       = v->base.stride * VLEN;
    for (uint32_t band = COL2IM_1D_VTCM_MAX_BAND; band >= COL2IM_1D_VTCM_MIN_BAND; band /= 2) {
        v->band             = band;
        v->off_tmp          = hex_round_up(col2im_1d_band_cols(v, band) * k_oc * sizeof(float), VLEN);
        v->off_out          = v->off_tmp + v->taps_per_output * VLEN;
        v->bytes_per_thread = v->off_out + hex_round_up(v->base.channels * band * sizeof(float), VLEN);
        if (v->plan_bytes + (size_t) v->bytes_per_thread * n_threads <= vtcm_size) {
            return true;
        }
    }
    return false;
}

static void col2im_1d_vtcm_phase(const struct htp_col2im_1d_vtcm * v, uint32_t r) {
    const uint32_t k_oc = v->base.kernel * v->base.channels;
    int32_t        lanes[VLEN_FP32] __attribute__((aligned(VLEN)));
    for (uint32_t p = 0; p < VLEN_FP32; p++) {
        lanes[p] = (int32_t) ((((r + p) / v->base.stride) * k_oc + (r + p) % v->base.stride) * sizeof(float));
    }
    v->phase_offsets[r] = *(const HVX_Vector *) lanes;
}

static void col2im_1d_vtcm_plan(struct htp_col2im_1d_vtcm * v) {
    v->phase_offsets = (HVX_Vector *) v->base.octx->ctx->vtcm_base;
    for (uint32_t r = 0; r < v->base.stride; r++) {
        col2im_1d_vtcm_phase(v, r);
    }
}

// Columns [col_first, col_end) of the input; rows outside the signal are zero.
static void col2im_1d_vtcm_stage(const struct htp_col2im_1d_vtcm * v, dma_queue * q, float * cols, int64_t col_first,
                                 int64_t col_end) {
    const struct htp_tensor * src   = v->base.octx->src[0];
    const uint32_t            k_oc  = v->base.kernel * v->base.channels;
    const int64_t             n_col = src->ne[1];
    const int64_t             lo    = MAX(col_first, 0);
    const int64_t             hi    = MIN(col_end, n_col);
    if (lo > col_first || hi < col_end) {
        hvx_splat_f32_a(cols, 0.0f, (uint32_t) (col_end - col_first) * k_oc);
    }
    if (hi > lo) {
        dma_queue_copy_rows(q, dma_make_ptr(cols + (lo - col_first) * k_oc, (const uint8_t *) src->data + lo * src->nb[1]),
                            k_oc * sizeof(float), src->nb[1], k_oc * sizeof(float), (size_t) (hi - lo));
    }
}

// Thirty-two outputs of one channel starting at absolute position a.
static HVX_Vector col2im_1d_vtcm_chunk(const struct htp_col2im_1d_vtcm * v, HVX_Vector * tmp, const float * cols,
                                       const float * cols_end, int64_t col_first, uint32_t oc, int64_t a) {
    const uint32_t k_oc  = v->base.kernel * v->base.channels;
    const int64_t  q0    = a / v->base.stride;
    const uint32_t r     = (uint32_t) (a - q0 * v->base.stride);
    for (uint32_t j = 0; j < v->taps_per_output; j++) {
        const float *  rt     = cols + (q0 - j - col_first) * k_oc + oc * v->base.kernel + j * v->base.stride;
        const uint32_t region = (uint32_t) ((const uint8_t *) cols_end - (const uint8_t *) rt) - 1;
        Q6_vgather_ARMVw(&tmp[j], (size_t) rt, region, v->phase_offsets[r]);
    }
    HVX_Vector sum = tmp[0];
    for (uint32_t j = 1; j < v->taps_per_output; j++) {
        sum = hvx_vec_add_f32_f32(sum, tmp[j]);
    }
    return sum;
}

static void col2im_1d_vtcm_channel(const struct htp_col2im_1d_vtcm * v, HVX_Vector * tmp, float * out_row,
                                   const float * cols, const float * cols_end, int64_t col_first, uint32_t oc,
                                   int64_t a0, uint32_t n) {
    for (uint32_t i = 0; i < n; i += VLEN_FP32) {
        const HVX_Vector sum = col2im_1d_vtcm_chunk(v, tmp, cols, cols_end, col_first, oc, a0 + i);
        hvx_vec_store_u(out_row + i, MIN(VLEN_FP32, n - i) * sizeof(float), sum);
    }
}

static void col2im_1d_vtcm_band(const struct htp_col2im_1d_vtcm * v, unsigned int ith, uint32_t band) {
    const struct htp_ops_context * octx = v->base.octx;
    const struct htp_tensor *      dst  = octx->dst;
    const uint32_t                 k_oc = v->base.kernel * v->base.channels;
    const uint32_t                 t0   = band * v->band;
    const uint32_t                 n    = MIN(v->band, dst->ne[0] - t0);

    const int64_t a0        = (int64_t) t0 + v->base.crop;
    const int64_t col_first = a0 / v->base.stride - (int64_t) v->taps_per_output + 1;
    const int64_t col_end   = (a0 + hex_round_up(n, VLEN_FP32) - 1) / v->base.stride + 1;

    uint8_t *    base = octx->ctx->vtcm_base + v->plan_bytes + (size_t) ith * v->bytes_per_thread;
    float *      cols = (float *) base;
    HVX_Vector * tmp  = (HVX_Vector *) (base + v->off_tmp);
    float *      out  = (float *) (base + v->off_out);
    dma_queue *  q    = octx->ctx->dma[ith];

    const float * cols_end = cols + (col_end - col_first) * k_oc;
    col2im_1d_vtcm_stage(v, q, cols, col_first, col_end);
    for (uint32_t oc = 0; oc < v->base.channels; oc++) {
        col2im_1d_vtcm_channel(v, tmp, out + oc * v->band, cols, cols_end, col_first, oc, a0, n);
    }
    dma_queue_copy_rows(q, dma_make_ptr((uint8_t *) dst->data + t0 * sizeof(float), out), dst->nb[1], v->band * sizeof(float),
                        n * sizeof(float), v->base.channels);
}

static void col2im_1d_vtcm_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_col2im_1d_vtcm * v = (const struct htp_col2im_1d_vtcm *) data;

    const uint32_t start = v->base.bands_per_thread * ith;
    const uint32_t end   = MIN(start + v->base.bands_per_thread, v->base.n_bands);
    for (uint32_t band = start; band < end; band++) {
        col2im_1d_vtcm_band(v, ith, band);
    }
}

static bool col2im_1d_run_vtcm(struct htp_ops_context * octx, const struct htp_col2im_1d_context * cctx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    if (src->type != HTP_TYPE_F32 || src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        cctx->kernel % cctx->stride != 0 || cctx->stride > COL2IM_1D_MAX_STRIDE) {
        return false;
    }
    struct htp_col2im_1d_vtcm v = { .base = *cctx, .taps_per_output = cctx->kernel / cctx->stride };
    if (!col2im_1d_vtcm_fit(&v, octx->n_threads, octx->ctx->vtcm_size)) {
        return false;
    }
    col2im_1d_vtcm_plan(&v);
    v.base.n_bands          = (dst->ne[0] + v.band - 1) / v.band;
    const uint32_t threads  = MIN(octx->n_threads, v.base.n_bands);
    v.base.bands_per_thread = (v.base.n_bands + threads - 1) / threads;
    worker_pool_run_func(octx->ctx->worker_pool, col2im_1d_vtcm_thread, &v, threads);
    return true;
}

int op_col2im_1d(struct htp_ops_context * octx) {
    const struct htp_tensor * cols = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if ((cols->type != HTP_TYPE_F32 && cols->type != HTP_TYPE_F16) || cols->type != dst->type) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t stride   = octx->op_params[0];
    const int32_t channels = octx->op_params[1];
    const int32_t crop     = octx->op_params[2];
    if (stride <= 0 || channels <= 0 || crop < 0 || cols->ne[0] % (uint32_t) channels != 0 ||
        dst->ne[1] != (uint32_t) channels || cols->ne[2] != 1 || cols->ne[3] != 1) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_bands = (dst->ne[0] + COL2IM_1D_BAND - 1) / COL2IM_1D_BAND;
    const uint32_t threads = MIN(octx->n_threads, n_bands);
    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || threads == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_col2im_1d_context cctx = {
        .octx             = octx,
        .stride           = (uint32_t) stride,
        .channels         = (uint32_t) channels,
        .kernel           = cols->ne[0] / (uint32_t) channels,
        .crop             = (uint32_t) crop,
        .bands_per_thread = (n_bands + threads - 1) / threads,
        .n_bands          = n_bands,
    };
    if (col2im_1d_run_vtcm(octx, &cctx)) {
        return HTP_STATUS_OK;
    }
    worker_pool_run_func(octx->ctx->worker_pool, col2im_1d_thread, &cctx, threads);
    return HTP_STATUS_OK;
}
