#include <HAP_farf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
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

// y = x + sin(alpha * x)^2 * inv_beta over one channel's contiguous time row.
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

static inline float snake_channel_param(const struct htp_tensor * p, uint32_t c) {
    return *(const float *) ((const uint8_t *) p->data + c * p->nb[1]);
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

    for (uint32_t c = start; c < end; c++) {
        snake_row((float *) ((uint8_t *) y->data + c * y->nb[1]),
                  (const float *) ((const uint8_t *) x->data + c * x->nb[1]), x->ne[0],
                  snake_channel_param(alpha, c), snake_channel_param(inv_beta, c));
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
        y->ne[0] != x->ne[0] || y->ne[1] != x->ne[1] || alpha->ne[1] != x->ne[1] || inv_beta->ne[1] != x->ne[1]) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t channels = x->ne[1];
    const uint32_t threads  = MIN(octx->n_threads, channels);
    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || threads == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_snake_context sctx = {
        .octx                = octx,
        .channels_per_thread = (channels + threads - 1) / threads,
    };
    worker_pool_run_func(octx->ctx->worker_pool, snake_thread, &sctx, threads);
    return HTP_STATUS_OK;
}
