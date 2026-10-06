#include <HAP_farf.h>

#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-dma.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"

// Fused 1D depthwise convolution: IM2COL(kernel, x) followed by
// MUL_MAT(columns, w) with one input channel per output channel, computed
// directly as dst[c][t] = sum_k x[c][t + k*d - p] * w[c][k]. The allocator may
// place dst over x, so every channel is staged in VTCM before any is written.
struct htp_dw1d_context {
    struct htp_ops_context *  octx;
    const struct htp_tensor * x;
    const struct htp_tensor * w;
    uint32_t                  taps;
    uint32_t                  dilation;
    uint32_t                  pad;
    uint32_t                  in_pitch;
    uint32_t                  out_pitch;
    uint32_t                  channels_per_thread;
    float *                   in;
    float *                   out;
};

static void dw1d_channel_range(const struct htp_dw1d_context * c, unsigned int ith, uint32_t * first, uint32_t * count) {
    const uint32_t C = c->octx->dst->ne[2];
    *first           = MIN(ith * c->channels_per_thread, C);
    *count           = MIN(c->channels_per_thread, C - *first);
}

static void dw1d_stage_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_dw1d_context * c = (const struct htp_dw1d_context *) data;
    uint32_t                        c0, nc;
    dw1d_channel_range(c, ith, &c0, &nc);
    if (nc == 0) {
        return;
    }
    float *         in  = c->in + (size_t) c0 * c->in_pitch;
    const uint8_t * src = (const uint8_t *) c->x->data + (size_t) c0 * c->x->nb[2];
    dma_queue *     q   = c->octx->ctx->dma[ith];
    hvx_splat_f32_a(in, 0.0f, nc * c->in_pitch);
    dma_queue_copy_rows(q, dma_make_ptr(in + c->pad, src), c->in_pitch * sizeof(float), c->x->nb[2],
                        c->x->ne[0] * sizeof(float), nc);
}

static void dw1d_row(const struct htp_dw1d_context * c, float * out, const float * in, const float * w,
                     uint32_t out_len) {
    for (uint32_t t = 0; t < out_len; t += VLEN_FP32) {
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t k = 0; k < c->taps; k++) {
            const HVX_Vector v = hvx_vmemu(in + t + k * c->dilation);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(w[k])));
        }
        *(HVX_Vector *) (out + t) = acc;
    }
}

static void dw1d_compute_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_dw1d_context * c   = (const struct htp_dw1d_context *) data;
    const struct htp_tensor *       dst = c->octx->dst;
    uint32_t                        c0, nc;
    dw1d_channel_range(c, ith, &c0, &nc);
    if (nc == 0) {
        return;
    }
    for (uint32_t i = c0; i < c0 + nc; i++) {
        const float * w = (const float *) ((const uint8_t *) c->w->data + (size_t) i * c->w->nb[2]);
        dw1d_row(c, c->out + (size_t) i * c->out_pitch, c->in + (size_t) i * c->in_pitch, w, dst->ne[0]);
    }
    dma_queue * q        = c->octx->ctx->dma[ith];
    uint8_t *   dst_rows = (uint8_t *) dst->data + (size_t) c0 * dst->nb[2];
    dma_queue_copy_rows(q, dma_make_ptr(dst_rows, c->out + (size_t) c0 * c->out_pitch), dst->nb[2],
                        c->out_pitch * sizeof(float), dst->ne[0] * sizeof(float), nc);
}

int op_depthwise_conv_1d(struct htp_ops_context * octx) {
    const struct htp_tensor * x   = octx->src[1];
    const struct htp_tensor * w   = octx->src[2] ? octx->src[2] : octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct htp_dw1d_context c = {
        .octx     = octx,
        .x        = x,
        .w        = w,
        .taps     = w->ne[0],
        .dilation = octx->op_params[4],
        .pad      = octx->op_params[2],
    };
    c.in_pitch  = htp_dw1d_in_pitch(dst->ne[0], c.taps, c.dilation, c.pad, x->ne[0]);
    c.out_pitch = hex_round_up(dst->ne[0], VLEN_FP32);

    const uint32_t C = dst->ne[2];
    if (htp_dw1d_vtcm_bytes(C, c.in_pitch, c.out_pitch) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    c.in  = (float *) octx->ctx->vtcm_base;
    c.out = c.in + (size_t) C * c.in_pitch;

    const uint32_t n_threads = MIN(octx->n_threads, C);
    c.channels_per_thread    = (C + n_threads - 1) / n_threads;
    work_queue_run(octx->ctx->work_queue, dw1d_stage_thread, &c, n_threads);
    work_queue_run(octx->ctx->work_queue, dw1d_compute_thread, &c, n_threads);
    return HTP_STATUS_OK;
}
