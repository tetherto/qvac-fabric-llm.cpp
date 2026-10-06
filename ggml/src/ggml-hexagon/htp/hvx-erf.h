#ifndef HVX_ERF_H
#define HVX_ERF_H

#include "hvx-base.h"
#include "hvx-inverse.h"

// erf(x) = x * P(x^2) / Q(x^2) on [-4, 4] (erf is +-1 in f32 outside it).
// Both polynomials are negated from the usual form so Q stays positive for
// the reciprocal.
#define HVX_ERF_CLAMP 4.0f

#define HVX_ERF_P1  1.60960333262415e-02f
#define HVX_ERF_P3  2.95459980854025e-03f
#define HVX_ERF_P5  7.34990630326855e-04f
#define HVX_ERF_P7  5.69250639462346e-05f
#define HVX_ERF_P9  2.10102402082508e-06f
#define HVX_ERF_P11 -2.77068142495902e-08f
#define HVX_ERF_P13 2.72614225801306e-10f

#define HVX_ERF_Q0 1.42647390514189e-02f
#define HVX_ERF_Q2 7.37332916720468e-03f
#define HVX_ERF_Q4 1.68282697438203e-03f
#define HVX_ERF_Q6 2.13374055278905e-04f
#define HVX_ERF_Q8 1.45660718464996e-05f

#define HVX_GELU_ERF_SQRT_2_INV 0.70710678118654752440f

static inline HVX_Vector hvx_vec_madd_f32(HVX_Vector a, HVX_Vector b, float c) {
    return hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(a, b), hvx_vec_splat_f32(c));
}

static inline HVX_Vector hvx_vec_erf_numerator_f32(HVX_Vector x, HVX_Vector x2) {
    HVX_Vector p = hvx_vec_madd_f32(x2, hvx_vec_splat_f32(HVX_ERF_P13), HVX_ERF_P11);
    p = hvx_vec_madd_f32(x2, p, HVX_ERF_P9);
    p = hvx_vec_madd_f32(x2, p, HVX_ERF_P7);
    p = hvx_vec_madd_f32(x2, p, HVX_ERF_P5);
    p = hvx_vec_madd_f32(x2, p, HVX_ERF_P3);
    p = hvx_vec_madd_f32(x2, p, HVX_ERF_P1);
    return hvx_vec_mul_f32_f32(x, p);
}

static inline HVX_Vector hvx_vec_erf_denominator_f32(HVX_Vector x2) {
    HVX_Vector q = hvx_vec_madd_f32(x2, hvx_vec_splat_f32(HVX_ERF_Q8), HVX_ERF_Q6);
    q = hvx_vec_madd_f32(x2, q, HVX_ERF_Q4);
    q = hvx_vec_madd_f32(x2, q, HVX_ERF_Q2);
    return hvx_vec_madd_f32(x2, q, HVX_ERF_Q0);
}

static inline HVX_Vector hvx_vec_erf_f32(HVX_Vector v) {
    const HVX_Vector hi = hvx_vec_splat_f32(HVX_ERF_CLAMP);
    const HVX_Vector lo = hvx_vec_splat_f32(-HVX_ERF_CLAMP);
    const HVX_Vector x  = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v, hi), lo);
    const HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
    const HVX_Vector p  = hvx_vec_erf_numerator_f32(x, x2);
    const HVX_Vector q  = hvx_vec_erf_denominator_f32(x2);
    return hvx_vec_mul_f32_f32(p, hvx_vec_inverse_f32(q));
}

static inline HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
    const HVX_Vector erf_v = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(HVX_GELU_ERF_SQRT_2_INV)));
    const HVX_Vector one_plus_erf = hvx_vec_add_f32_f32(erf_v, hvx_vec_splat_f32(1.0f));
    const HVX_Vector half_x = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(0.5f));
    return hvx_vec_mul_f32_f32(half_x, one_plus_erf);
}

static inline void hvx_gelu_erf_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_erf_f32(vsrc[i]));
    }
}

#endif  // HVX_ERF_H
