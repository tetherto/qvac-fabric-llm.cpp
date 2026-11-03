#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-dma-reader.h"
#include "hvx-utils.h"

struct htp_conv_dw_context {
    struct htp_ops_context *octx;
    uint32_t rows;
    uint32_t rows_per_thread;
    bool channels_contiguous;
    bool planar_stride2_3x3;
    size_t reader_offset;
};

struct conv_dw_access {
    const struct htp_ops_context * octx;
    struct htp_dma_reader weights;
    struct htp_dma_reader input;
};

// All addresses use the original tensor strides, including the permuted
// [KW, KH, 1, C] weight view used by Parakeet's channel-contiguous conformer.
static inline float conv_dw_weight(struct conv_dw_access * access, uint32_t x, uint32_t y, uint32_t c) {
    const struct htp_tensor * k = access->weights.tensor;
    const void * p = htp_dma_reader_view(&access->weights, x * k->nb[0] + y * k->nb[1] + c * k->nb[3],
        k->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float));
    return k->type == HTP_TYPE_F16 ? (float)*(const __fp16 *)p : *(const float *)p;
}

static inline const float *conv_dw_input(struct conv_dw_access * access, uint32_t w, uint32_t h, uint32_t c, uint32_t n, uint32_t count) {
    const struct htp_tensor * x = access->input.tensor;
    return htp_dma_reader_view(&access->input, w * x->nb[0] + h * x->nb[1] + c * x->nb[2] + n * x->nb[3],
        count * sizeof(float));
}

static inline float *conv_dw_output(const struct htp_tensor *y, uint32_t w, uint32_t h, uint32_t c, uint32_t n) {
    return (float *)((uint8_t *)y->data + w * y->nb[0] + h * y->nb[1] + c * y->nb[2] + n * y->nb[3]);
}

static float conv_dw_scalar(struct conv_dw_access * access, uint32_t ox, uint32_t oy, uint32_t c, uint32_t n) {
    const struct htp_ops_context * octx = access->octx;
    const struct htp_tensor *k = octx->src[0];
    const struct htp_tensor *x = octx->src[1];
    const int32_t *p = octx->op_params;
    float sum = 0.0f;
    for (uint32_t ky = 0; ky < k->ne[1]; ++ky) {
        const int64_t iy = (int64_t)oy * p[1] - p[3] + (int64_t)ky * p[5];
        if (iy < 0 || iy >= x->ne[1]) {
            continue;
        }
        for (uint32_t kx = 0; kx < k->ne[0]; ++kx) {
            const int64_t ix = (int64_t)ox * p[0] - p[2] + (int64_t)kx * p[4];
            if (ix >= 0 && ix < x->ne[0]) {
                sum += conv_dw_weight(access, kx, ky, c) * *conv_dw_input(access, ix, iy, c, n, 1);
            }
        }
    }
    return sum;
}

// Channel-vectorized FP32 accumulation avoids activation transposes and the
// large im2col temporary for the conformer's [T, 1, C, N] depthwise
// convolution.
static void conv_dw_channels_point(struct conv_dw_access * access, uint32_t ox, uint32_t oy, uint32_t n) {
    const struct htp_ops_context * octx = access->octx;
    const struct htp_tensor *k = octx->src[0];
    const struct htp_tensor *x = octx->src[1];
    const struct htp_tensor *y = octx->dst;
    const int32_t *p = octx->op_params;
    uint32_t c = 0;
    for (; c + VLEN_FP32 <= x->ne[2]; c += VLEN_FP32) {
        HVX_Vector sum = Q6_V_vzero();
        for (uint32_t ky = 0; ky < k->ne[1]; ++ky) {
            const int64_t iy = (int64_t)oy * p[1] - p[3] + (int64_t)ky * p[5];
            if (iy < 0 || iy >= x->ne[1]) {
                continue;
            }
            for (uint32_t kx = 0; kx < k->ne[0]; ++kx) {
                const int64_t ix = (int64_t)ox * p[0] - p[2] + (int64_t)kx * p[4];
                if (ix < 0 || ix >= x->ne[0]) {
                    continue;
                }
                const HVX_Vector xv = *(const HVX_UVector *)conv_dw_input(access, ix, iy, c, n, VLEN_FP32);
                const void * kp = htp_dma_reader_view(&access->weights, kx * k->nb[0] + ky * k->nb[1] + c * k->nb[3],
                    VLEN_FP32 * (k->type == HTP_TYPE_F32 ? sizeof(float) : sizeof(__fp16)));
                HVX_Vector kv;
                if (k->type == HTP_TYPE_F32) {
                    kv = *(const HVX_UVector *)kp;
                } else {
                    // Copy exactly 32 halfs: loading a whole HVX vector
                    // could read beyond the final channel block of a view.
                    __fp16 halfs[VLEN_FP16] __attribute__((aligned(128))) = {0};
                    memcpy(halfs, kp, VLEN_FP32 * sizeof(__fp16));
                    kv = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *)halfs));
                }
                sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(xv, kv));
            }
        }
        *(HVX_UVector *)conv_dw_output(y, ox, oy, c, n) = sum;
    }
    for (; c < x->ne[2]; ++c) {
        *conv_dw_output(y, ox, oy, c, n) = conv_dw_scalar(access, ox, oy, c, n);
    }
}

// WHCN: vectorize independent output positions. Strided samples and padded
// edges are staged in a bounded stack tile; interior stride-1 loads use HVX
// directly for normal-address inputs.
static void conv_dw_planar_row(struct conv_dw_access * access, uint32_t oy, uint32_t c, uint32_t n) {
    const struct htp_ops_context * octx = access->octx;
    const struct htp_tensor *k = octx->src[0];
    const struct htp_tensor *x = octx->src[1];
    const struct htp_tensor *y = octx->dst;
    const int32_t *p = octx->op_params;
    uint32_t ox = 0;
    for (; ox + VLEN_FP32 <= y->ne[0]; ox += VLEN_FP32) {
        HVX_Vector sum = Q6_V_vzero();
        for (uint32_t ky = 0; ky < k->ne[1]; ++ky) {
            const int64_t iy = (int64_t)oy * p[1] - p[3] + (int64_t)ky * p[5];
            if (iy < 0 || iy >= x->ne[1]) {
                continue;
            }
            for (uint32_t kx = 0; kx < k->ne[0]; ++kx) {
                const int64_t ix = (int64_t)ox * p[0] - p[2] + (int64_t)kx * p[4];
                HVX_Vector xv;
                if (p[0] == 1 && ix >= 0 && ix + VLEN_FP32 <= x->ne[0]) {
                    xv = *(const HVX_UVector *)conv_dw_input(access, ix, iy, c, n, VLEN_FP32);
                } else {
                    float tile[VLEN_FP32] __attribute__((aligned(128)));
                    for (uint32_t lane = 0; lane < VLEN_FP32; ++lane) {
                        const int64_t sx = ix + (int64_t)lane * p[0];
                        tile[lane] = sx >= 0 && sx < x->ne[0] ? *conv_dw_input(access, sx, iy, c, n, 1) : 0.0f;
                    }
                    xv = *(const HVX_Vector *)tile;
                }
                const HVX_Vector kv = hvx_vec_splat_f32(conv_dw_weight(access, kx, ky, c));
                sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(xv, kv));
            }
        }
        *(HVX_UVector *)conv_dw_output(y, ox, oy, c, n) = sum;
    }
    for (; ox < y->ne[0]; ++ox) {
        *conv_dw_output(y, ox, oy, c, n) = conv_dw_scalar(access, ox, oy, c, n);
    }
}

// The subsampler has narrow planar rows (often only 10 or 20 outputs). Deal
// contiguous input words into stride-2 samples instead of gathering each lane
// for each tap. Partial output tiles use the same vector arithmetic as full
// tiles; bounded staging handles row ends and implicit horizontal padding.
static void conv_dw_planar_stride2_3x3_row(struct conv_dw_access * access, uint32_t oy, uint32_t c, uint32_t n) {
    const struct htp_ops_context * octx = access->octx;
    const struct htp_tensor *x = octx->src[1];
    const struct htp_tensor *y = octx->dst;
    const int32_t *p = octx->op_params;
    float tile[2 * VLEN_FP32] __attribute__((aligned(128)));

    for (uint32_t ox = 0; ox < y->ne[0]; ox += VLEN_FP32) {
        const uint32_t lanes = MIN(VLEN_FP32, y->ne[0] - ox);
        const int64_t ix = (int64_t)ox * 2 - p[2];
        HVX_Vector sum = Q6_V_vzero();
        for (uint32_t ky = 0; ky < 3; ++ky) {
            const int64_t iy = (int64_t)oy * 2 - p[3] + ky;
            if (iy < 0 || iy >= x->ne[1]) {
                continue;
            }

            HVX_Vector lo, hi;
            if (ix >= 0 && ix + 2 * VLEN_FP32 <= x->ne[0]) {
                const float *row = conv_dw_input(access, ix, iy, c, n, 2 * VLEN_FP32);
                lo = *(const HVX_UVector *)row;
                hi = *(const HVX_UVector *)(row + VLEN_FP32);
            } else {
                // Copy only samples belonging to this row. Two zeroed vectors
                // also define unused output lanes without an out-of-bounds
                // vector read at the end of the tensor.
                ((HVX_Vector *)tile)[0] = Q6_V_vzero();
                ((HVX_Vector *)tile)[1] = Q6_V_vzero();
                const int64_t begin = MAX(ix, 0);
                const int64_t end = MIN(ix + MIN(2 * lanes + 1, 2 * VLEN_FP32), x->ne[0]);
                if (end > begin) {
                    memcpy(tile + (begin - ix), conv_dw_input(access, begin, iy, c, n, end - begin),
                           (end - begin) * sizeof(float));
                }
                lo = ((const HVX_Vector *)tile)[0];
                hi = ((const HVX_Vector *)tile)[1];
            }

            // vdeal(..., -4): lo = x[0,2,...,62], hi = x[1,3,...,63].
            // The third tap is the even vector shifted by one float, with
            // x[64] supplied separately only for a full output tile.
            const HVX_VectorPair samples = Q6_W_vdeal_VVR(hi, lo, -4);
            const HVX_Vector even = Q6_V_lo_W(samples);
            const HVX_Vector odd = Q6_V_hi_W(samples);
            const int64_t last_ix = ix + 2 * VLEN_FP32;
            const float last = lanes == VLEN_FP32 && last_ix >= 0 && last_ix < x->ne[0]
                                   ? *conv_dw_input(access, last_ix, iy, c, n, 1) : 0.0f;
            const HVX_Vector next = Q6_V_valign_VVR(hvx_vec_splat_f32(last), even, sizeof(float));
            const HVX_Vector w0 = hvx_vec_splat_f32(conv_dw_weight(access, 0, ky, c));
            const HVX_Vector w1 = hvx_vec_splat_f32(conv_dw_weight(access, 1, ky, c));
            const HVX_Vector w2 = hvx_vec_splat_f32(conv_dw_weight(access, 2, ky, c));
            // Keep the accumulator in IEEE FP32 between taps. Chaining qf32
            // accumulations across consecutive zero products can produce a
            // spurious large exponent at padded edges with zero inputs/weights.
            sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(even, w0));
            sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(odd, w1));
            sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(next, w2));
        }
        hvx_vec_store_u(conv_dw_output(y, ox, oy, c, n), lanes * sizeof(float), sum);
    }
}

static void conv_dw_thread(unsigned int nth, unsigned int ith, void *data) {
    struct htp_conv_dw_context *dc = (struct htp_conv_dw_context *)data;
    struct htp_ops_context *octx = dc->octx;
    struct htp_thread_trace *tr = &octx->ctx->trace[ith];
    const uint32_t start = ith * dc->rows_per_thread;
    const uint32_t end = MIN(start + dc->rows_per_thread, dc->rows);
    const uint32_t oh = octx->dst->ne[1];
    const uint32_t ow = octx->dst->ne[0];
    const uint32_t channels = octx->dst->ne[2];
    uint8_t * scratch = octx->ctx->vtcm_base + dc->reader_offset + ith * 2 * HTP_DMA_READER_BYTES;
    struct conv_dw_access access = {octx,
        htp_dma_reader_init(octx->src[0], octx->ctx->dma[ith], scratch),
        htp_dma_reader_init(octx->src[1], octx->ctx->dma[ith], scratch + HTP_DMA_READER_BYTES)};
    (void)nth;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start);
    for (uint32_t row = start; row < end; ++row) {
        if (dc->channels_contiguous) {
            conv_dw_channels_point(&access, row % ow, (row / ow) % oh, row / (ow * oh));
        } else if (dc->planar_stride2_3x3) {
            conv_dw_planar_stride2_3x3_row(&access, row % oh, (row / oh) % channels, row / (oh * channels));
        } else {
            conv_dw_planar_row(&access, row % oh, (row / oh) % channels, row / (oh * channels));
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start);
}

static void conv_dw_convert_weights(float * dst, const __fp16 * src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (float) src[i];
    }
}

static void conv_dw_convert_weights_dma(float * dst, struct htp_dma_reader * reader, size_t count) {
    for (size_t i = 0; i < count; i += VLEN_FP16) {
        const uint32_t n = MIN(VLEN_FP16, count - i);
        const __fp16 * src = htp_dma_reader_view(reader, i * sizeof(__fp16), n * sizeof(__fp16));
        conv_dw_convert_weights(dst + i, src, n);
    }
}

static size_t conv_dw_prepare_weights(struct htp_ops_context * local, struct htp_tensor * weights, uint32_t threads) {
    const struct htp_tensor * src = local->src[0];
    const size_t count = (size_t) src->ne[0] * src->ne[1] * src->ne[3];
    const size_t reader_bytes = (size_t) threads * 2 * HTP_DMA_READER_BYTES;
    if (src->type != HTP_TYPE_F16 || reader_bytes > local->ctx->vtcm_size ||
        count > (local->ctx->vtcm_size - reader_bytes) / sizeof(float)) {
        return 0;
    }
    const size_t offset = hex_round_up(count * sizeof(float), VLEN);
    if (offset > local->ctx->vtcm_size - reader_bytes) {
        return 0;
    }
    *weights = *src;
    weights->type = HTP_TYPE_F32;
    weights->data = (uintptr_t) local->ctx->vtcm_base;
    weights->size = count * sizeof(float);
    for (uint32_t i = 0; i < HTP_OP_MAX_DIMS; ++i) {
        weights->nb[i] = src->nb[i] * (sizeof(float) / sizeof(__fp16));
    }
    struct htp_dma_reader reader = htp_dma_reader_init(src, local->ctx->dma[0], local->ctx->vtcm_base + offset);
    conv_dw_convert_weights_dma((float *) local->ctx->vtcm_base, &reader, count);
    local->src[0] = weights;
    return offset;
}

int op_conv_2d_dw(struct htp_ops_context *octx) {
    const struct htp_tensor *k = octx->src[0];
    const struct htp_tensor *x = octx->src[1];
    const struct htp_tensor *y = octx->dst;
    if ((k->type != HTP_TYPE_F32 && k->type != HTP_TYPE_F16) || x->type != HTP_TYPE_F32 || y->type != HTP_TYPE_F32) {
        FARF(ERROR, "conv_2d_dw: requires F32 input/output and F32/F16 kernel");
        return HTP_STATUS_NO_SUPPORT;
    }
    const int32_t *p = octx->op_params;
    if (p[0] <= 0 || p[1] <= 0 || p[2] < 0 || p[3] < 0 || p[4] <= 0 || p[5] <= 0 || k->ne[2] != 1 ||
        k->ne[3] != x->ne[2] || x->ne[2] != y->ne[2] || x->ne[3] != y->ne[3]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    // When C=1 both layouts are equivalent. Choose planar as the host does.
    const bool channels_contiguous = x->nb[0] != sizeof(float) && x->nb[2] == sizeof(float);
    const bool planar_stride2_3x3 = !channels_contiguous && x->nb[0] == sizeof(float) && y->nb[0] == sizeof(float) &&
                                   k->ne[0] == 3 && k->ne[1] == 3 && p[0] == 2 && p[1] == 2 && p[4] == 1 && p[5] == 1;
    const uint32_t rows = y->ne[1] * y->ne[3] * (channels_contiguous ? y->ne[0] : y->ne[2]);
    const uint32_t threads = MIN(octx->n_threads, rows);
    if (threads == 0) {
        return HTP_STATUS_OK;
    }
    struct htp_ops_context local = *octx;
    struct htp_tensor weights;
    if ((size_t) threads * 2 * HTP_DMA_READER_BYTES > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    const size_t reader_offset = conv_dw_prepare_weights(&local, &weights, threads);
    struct htp_conv_dw_context dc = {
        &local, rows, (rows + threads - 1) / threads, channels_contiguous, planar_stride2_3x3, reader_offset};
    work_queue_run(octx->ctx->work_queue, conv_dw_thread, &dc, threads);
    return HTP_STATUS_OK;
}
