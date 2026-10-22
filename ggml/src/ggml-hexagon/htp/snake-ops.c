#include <HAP_farf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-dma-reader.h"
#include "hvx-utils.h"
#include "hvx-sin-cos.h"

struct htp_snake_context {
    struct htp_ops_context * octx;
    uint32_t                 channels_per_thread;
};

static inline HVX_Vector snake_vec(HVX_Vector x, HVX_Vector alpha, HVX_Vector inv_beta) {
    const HVX_Vector s = hvx_vec_sin_f32(hvx_vec_mul_f32_f32(alpha, x));
    return hvx_vec_add_f32_f32(x, hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(s, s), inv_beta));
}

static void snake_row(float * restrict y, const float * restrict x, uint32_t n, float alpha, float inv_beta) {
    const HVX_Vector valpha    = hvx_vec_splat_f32(alpha);
    const HVX_Vector vinv_beta = hvx_vec_splat_f32(inv_beta);

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec; i++) {
        hvx_vmemu(y + i * VLEN_FP32) = snake_vec(hvx_vmemu(x + i * VLEN_FP32), valpha, vinv_beta);
    }
    if (nloe) {
        const HVX_Vector v = snake_vec(hvx_vmemu(x + nvec * VLEN_FP32), valpha, vinv_beta);
        hvx_vec_store_u(y + nvec * VLEN_FP32, nloe * sizeof(float), v);
    }
}

static float snake_channel_param(struct htp_dma_reader * reader, uint32_t c) {
    return *(const float *) htp_dma_reader_view(reader, c * sizeof(float), sizeof(float));
}

static void snake_dma_row(float * y, struct htp_dma_reader * reader, uint32_t offset, uint32_t n,
                          float alpha, float inv_beta) {
    float tile[VLEN_FP32] __attribute__((aligned(VLEN)));
    for (uint32_t i = 0; i < n; i += VLEN_FP32) {
        const uint32_t count = MIN(VLEN_FP32, n - i);
        memset(tile, 0, sizeof(tile));
        memcpy(tile, htp_dma_reader_view(reader, offset + i * sizeof(float), count * sizeof(float)), count * sizeof(float));
        snake_row(y + i, tile, count, alpha, inv_beta);
    }
}

static inline bool snake_channel_params_valid(const struct htp_tensor * p, uint32_t channels) {
    return p->nb[0] == sizeof(float) && p->ne[0] * p->ne[1] * p->ne[2] * p->ne[3] == channels;
}

static void snake_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_snake_context * sctx = (const struct htp_snake_context *) data;
    const struct htp_ops_context *   octx = sctx->octx;

    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * alpha    = octx->src[1];
    const struct htp_tensor * inv_beta = octx->src[2];
    const struct htp_tensor * y        = octx->dst;

    const uint32_t channels = x->ne[1];
    const uint32_t start    = sctx->channels_per_thread * ith;
    const uint32_t end      = MIN(start + sctx->channels_per_thread, channels);

    uint8_t * scratch = octx->ctx->vtcm_base + ith * 3 * HTP_DMA_READER_BYTES;
    struct htp_dma_reader xr = htp_dma_reader_init(x, octx->ctx->dma[ith], scratch);
    struct htp_dma_reader ar = htp_dma_reader_init(alpha, octx->ctx->dma[ith], scratch + HTP_DMA_READER_BYTES);
    struct htp_dma_reader br = htp_dma_reader_init(inv_beta, octx->ctx->dma[ith], scratch + 2 * HTP_DMA_READER_BYTES);
    for (uint32_t c = start; c < end; c++) {
        float * out = (float *) ((uint8_t *) y->data + c * y->nb[1]);
        const float a = snake_channel_param(&ar, c);
        const float b = snake_channel_param(&br, c);
        if (htp_tensor_is_extended(x)) {
            snake_dma_row(out, &xr, c * x->nb[1], x->ne[0], a, b);
        } else {
            snake_row(out, (const float *) ((const uint8_t *) x->data + c * x->nb[1]), x->ne[0], a, b);
        }
    }
}

int op_snake(struct htp_ops_context * octx) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * alpha    = octx->src[1];
    const struct htp_tensor * inv_beta = octx->src[2];
    const struct htp_tensor * y        = octx->dst;

    if (x->type != HTP_TYPE_F32 || y->type != HTP_TYPE_F32 || alpha->type != HTP_TYPE_F32 ||
        inv_beta->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (x->nb[0] != sizeof(float) || y->nb[0] != sizeof(float) || x->ne[2] != 1 || x->ne[3] != 1 ||
        y->ne[0] != x->ne[0] || y->ne[1] != x->ne[1] || !snake_channel_params_valid(alpha, x->ne[1]) ||
        !snake_channel_params_valid(inv_beta, x->ne[1])) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t channels = x->ne[1];
    const uint32_t threads  = MIN(octx->n_threads, channels);
    if (threads == 0) {
        return HTP_STATUS_OK;
    }

    if ((size_t) threads * 3 * HTP_DMA_READER_BYTES > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    struct htp_snake_context sctx = {
        .octx                = octx,
        .channels_per_thread = (channels + threads - 1) / threads,
    };
    work_queue_run(octx->ctx->work_queue, snake_thread, &sctx, threads);
    return HTP_STATUS_OK;
}
