#include "hvx-utils.h"
#include "htp-ctx.h"
#include "conv-taps-validate.h"
#include <string.h>

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

static inline HVX_VectorPair panel_f16_weights(const uint8_t * row, uint32_t remaining) {
    HVX_Vector packed;
    if (remaining >= VLEN_FP16) {
        packed = hvx_vmemu(row);
    } else {
        packed = Q6_V_vror_VR(hvx_vmemu(row - VLEN / 2), VLEN / 2);
    }
    return hvx_vec_f16_to_f32(packed); // upper F32 vector is unused for a half-vector tail
}

static inline void vec_store_f32_residual(float * dst, uint32_t cells, HVX_Vector product,
                                         const float * residual, struct htp_thread_trace * tr) {
    if (residual) {
        // Qualcomm can fold QF32 -> SF into the following ADD. The standalone
        // product store rounds here; retain that boundary in an HVX register.
        __asm__ volatile("" : "+v"(product));
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, cells);
        HVX_Vector z = hvx_vec_splat_f32(residual[0]);
        if (cells == 2) {
            z = Q6_V_valign_VVR(hvx_vec_splat_f32(residual[1]), z, VLEN - sizeof(float));
        }
        product = hvx_vec_add_f32_f32(product, z);
        hvx_vec_store_u(dst, cells * sizeof(float), product);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, cells);
    } else {
        hvx_vec_store_u(dst, cells * sizeof(float), product);
    }
}

static inline void vec_dot_f16_f32_panel_1x1_residual(const uint32_t n, float * s,
                                           const void * restrict vx, const void * restrict vy,
                                           const float * z, struct htp_thread_trace * tr) {
    const uint8_t * x = vx;
    const uint8_t * y = vy;
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t k = 0; k < n; k += VLEN_FP16) {
        const HVX_VectorPair w = panel_f16_weights(x + k * 2, n - k);
        sum = HVX_OP_ADD_F32(sum, HVX_OP_MUL_F32(Q6_V_lo_W(w), hvx_vmemu(y + k * 4)));
        if (n - k >= VLEN_FP16) {
            sum = HVX_OP_ADD_F32(sum, HVX_OP_MUL_F32(Q6_V_hi_W(w), hvx_vmemu(y + (k + VLEN_FP32) * 4)));
        }
    }
    vec_store_f32_residual(s, 1, hvx_vec_reduce_sum_f32(sum), z, tr);
}

static inline void vec_dot_f16_f32_uu_4x2_residual(const uint32_t n, float * s0, float * s1,
                                        const void * restrict vx0, const void * restrict vx1,
                                        const void * restrict vx2, const void * restrict vx3,
                                        const void * restrict vy0, const void * restrict vy1,
                                        const float * z0, const float * z1, struct htp_thread_trace * tr) {
    const uint8_t * x0 = vx0;
    const uint8_t * x1 = vx1;
    const uint8_t * x2 = vx2;
    const uint8_t * x3 = vx3;
    const uint8_t * y0 = vy0;
    const uint8_t * y1 = vy1;
    HVX_Vector sum00 = Q6_V_vzero(), sum01 = Q6_V_vzero();
    HVX_Vector sum10 = Q6_V_vzero(), sum11 = Q6_V_vzero();
    HVX_Vector sum20 = Q6_V_vzero(), sum21 = Q6_V_vzero();
    HVX_Vector sum30 = Q6_V_vzero(), sum31 = Q6_V_vzero();
    for (uint32_t k = 0; k < n; k += VLEN_FP16) {
        const HVX_VectorPair w0 = panel_f16_weights(x0 + k * 2, n - k);
        const HVX_VectorPair w1 = panel_f16_weights(x1 + k * 2, n - k);
        const HVX_VectorPair w2 = panel_f16_weights(x2 + k * 2, n - k);
        const HVX_VectorPair w3 = panel_f16_weights(x3 + k * 2, n - k);
        HVX_Vector a0 = hvx_vmemu(y0 + k * 4);
        HVX_Vector a1 = hvx_vmemu(y1 + k * 4);
        sum00 = HVX_OP_ADD_F32(sum00, HVX_OP_MUL_F32(Q6_V_lo_W(w0), a0));
        sum01 = HVX_OP_ADD_F32(sum01, HVX_OP_MUL_F32(Q6_V_lo_W(w0), a1));
        sum10 = HVX_OP_ADD_F32(sum10, HVX_OP_MUL_F32(Q6_V_lo_W(w1), a0));
        sum11 = HVX_OP_ADD_F32(sum11, HVX_OP_MUL_F32(Q6_V_lo_W(w1), a1));
        sum20 = HVX_OP_ADD_F32(sum20, HVX_OP_MUL_F32(Q6_V_lo_W(w2), a0));
        sum21 = HVX_OP_ADD_F32(sum21, HVX_OP_MUL_F32(Q6_V_lo_W(w2), a1));
        sum30 = HVX_OP_ADD_F32(sum30, HVX_OP_MUL_F32(Q6_V_lo_W(w3), a0));
        sum31 = HVX_OP_ADD_F32(sum31, HVX_OP_MUL_F32(Q6_V_lo_W(w3), a1));
        if (n - k >= VLEN_FP16) {
            a0 = hvx_vmemu(y0 + (k + VLEN_FP32) * 4);
            a1 = hvx_vmemu(y1 + (k + VLEN_FP32) * 4);
            sum00 = HVX_OP_ADD_F32(sum00, HVX_OP_MUL_F32(Q6_V_hi_W(w0), a0));
            sum01 = HVX_OP_ADD_F32(sum01, HVX_OP_MUL_F32(Q6_V_hi_W(w0), a1));
            sum10 = HVX_OP_ADD_F32(sum10, HVX_OP_MUL_F32(Q6_V_hi_W(w1), a0));
            sum11 = HVX_OP_ADD_F32(sum11, HVX_OP_MUL_F32(Q6_V_hi_W(w1), a1));
            sum20 = HVX_OP_ADD_F32(sum20, HVX_OP_MUL_F32(Q6_V_hi_W(w2), a0));
            sum21 = HVX_OP_ADD_F32(sum21, HVX_OP_MUL_F32(Q6_V_hi_W(w2), a1));
            sum30 = HVX_OP_ADD_F32(sum30, HVX_OP_MUL_F32(Q6_V_hi_W(w3), a0));
            sum31 = HVX_OP_ADD_F32(sum31, HVX_OP_MUL_F32(Q6_V_hi_W(w3), a1));
        }
    }
    vec_store_f32_residual(s0, 2, hvx_vec_reduce_sum_f32x2(sum00, sum10), z0, tr);
    vec_store_f32_residual(s0 + 2, 2, hvx_vec_reduce_sum_f32x2(sum20, sum30), z0 ? z0 + 2 : NULL, tr);
    vec_store_f32_residual(s1, 2, hvx_vec_reduce_sum_f32x2(sum01, sum11), z1, tr);
    vec_store_f32_residual(s1 + 2, 2, hvx_vec_reduce_sum_f32x2(sum21, sum31), z1 ? z1 + 2 : NULL, tr);
}

static void hvx_conv_taps_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ops_context * octx = data;
    const struct htp_conv_taps_params * p = (const struct htp_conv_taps_params *) octx->kernel_params;
    const struct htp_tensor * w = octx->src[0];
    const struct htp_tensor * x = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t channels = dst->ne[0];
    const uint32_t rows = dst->ne[1];
    const uint32_t blocks = 1 + (rows - 1) / p->tile_rows;
    float * tile = (float *) (octx->ctx->vtcm_base + (size_t) ith * p->worker_bytes);
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    for (uint32_t block = ith; block < blocks; block += nth) {
        const uint32_t first = block * p->tile_rows;
        const uint32_t count = MIN(p->tile_rows, rows - first);
        for (uint32_t tap = 0; tap < w->ne[2]; ++tap) {
            const uint8_t * weights = (const uint8_t *) w->data + (size_t) tap * w->nb[2];
            for (uint32_t r = 0; r < count; r += 2) {
                const int64_t pos = (int64_t) p->first_row + first + r + (int64_t) tap * p->tap_step;
                const uint8_t * a0 = (const uint8_t *) x->data + (size_t) pos * x->nb[1];
                float * d0 = tile + (size_t) r * channels;
                if (r + 1 < count) {
                    const uint8_t * a1 = a0 + x->nb[1];
                    float * d1 = d0 + channels;
                    for (uint32_t c = 0; c < channels; c += 4) {
                        const uint8_t * w0 = weights + (size_t) c * w->nb[1];
                        vec_dot_f16_f32_uu_4x2_residual(w->ne[0], d0 + c, d1 + c,
                            w0, w0 + w->nb[1], w0 + 2 * w->nb[1], w0 + 3 * w->nb[1],
                            a0, a1, tap ? d0 + c : NULL, tap ? d1 + c : NULL, tr);
                    }
                } else {
                    for (uint32_t c = 0; c < channels; ++c) {
                        vec_dot_f16_f32_panel_1x1_residual(w->ne[0], d0 + c,
                            weights + (size_t) c * w->nb[1], a0, tap ? d0 + c : NULL, tr);
                    }
                }
            }
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, count);
        memcpy((uint8_t *) dst->data + (size_t) first * dst->nb[1], tile,
               (size_t) count * channels * sizeof(float));
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, count);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

int op_conv_taps_f32(struct htp_ops_context * octx) {
    const struct htp_conv_taps_params * p = (const struct htp_conv_taps_params *) octx->kernel_params;
    const struct htp_tensor * w = octx->src[0];
    const struct htp_tensor * x = octx->src[1];
    const struct htp_tensor * d = octx->dst;
    const int status = htp_conv_taps_validate(w, x, d, p, octx->n_threads, octx->ctx->vtcm_size);
    if (status != HTP_STATUS_OK) return status;
    if (!work_queue_run(octx->ctx->work_queue, hvx_conv_taps_thread, octx, octx->n_threads)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    return HTP_STATUS_OK;
}
