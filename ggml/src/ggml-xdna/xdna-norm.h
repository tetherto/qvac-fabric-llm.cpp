#pragma once

// RMS_NORM on the host the way ggml's CPU op rounds it, vectorized: the
// backend builds for plain x86-64, and XDNA ships only with Zen 4/5, so AVX2
// is there. ggml's own op is scalar with a double accumulator, and a prompt's
// norms were a fifth of the host's time (each busy core costs the package
// ~10-16 W).

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>

// A row's scale: the squares in f32 summed in double, the mean back in f32.
__attribute__((target("avx2"))) static inline float xdna_rms_scale(const float * x, int64_t n, float eps) {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 v  = _mm256_loadu_ps(x + j);
        const __m256 sq = _mm256_mul_ps(v, v);
        s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
        s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
    }
    double t[4];
    _mm256_storeu_pd(t, _mm256_add_pd(s0, s1));
    double sum = t[0] + t[1] + t[2] + t[3];
    for (; j < n; j++) {
        sum += (double) (x[j] * x[j]);
    }
    const float mean = (float) (sum / n);
    return 1.0f / sqrtf(mean + eps);
}

// xdna_rms_scale of a row held as two runs of n (n a multiple of 8): the
// same accumulators over both, so the same bits as one run of 2n.
__attribute__((target("avx2"))) static inline float xdna_rms_scale2(const float * x0, const float * x1, int64_t n,
                                                                   float eps) {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    for (const float * x : { x0, x1 }) {
        for (int64_t j = 0; j < n; j += 8) {
            const __m256 v  = _mm256_loadu_ps(x + j);
            const __m256 sq = _mm256_mul_ps(v, v);
            s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
            s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
        }
    }
    double t[4];
    _mm256_storeu_pd(t, _mm256_add_pd(s0, s1));
    const float mean = (float) ((t[0] + t[1] + t[2] + t[3]) / (2 * n));
    return 1.0f / sqrtf(mean + eps);
}

// y = (x * s) * w, each product rounded to f32 in the order ggml's RMS_NORM
// and MUL round them.
__attribute__((target("avx2"))) static inline void xdna_norm_mul_row(const float * x, const float * w, float s,
                                                                    float * y, int64_t n) {
    const __m256 vs = _mm256_set1_ps(s);
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + j), vs), _mm256_loadu_ps(w + j)));
    }
    for (; j < n; j++) {
        y[j] = (x[j] * s) * w[j];
    }
}

// ggml's silu, x / (1 + exp(-x)) with its vector expf, in the variant its
// CPU backend runs: -march=native builds on Zen 4/5 take the AVX-512 one,
// others the AVX2 one (the two differ in the last bit).
__attribute__((target("avx512f,avx512dq"))) static inline __m512 xdna_v_expf512(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) {
        return res;
    }
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

__attribute__((target("avx2,fma"))) static inline __m256 xdna_v_expf256(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                       _mm256_set1_epi32(0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
                         _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                      _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// y = ((x * s) * w) * silu(z): a gated norm's row (the recurrent layers'
// output), each step rounded as ggml's RMS_NORM, MUL, SILU and MUL are
__attribute__((target("avx512f,avx512dq"))) static inline void xdna_gated_row512(const float * x, const float * w,
                                                                                float s, const float * z, float * y,
                                                                                int64_t n) {
    const __m512 vs = _mm512_set1_ps(s), one = _mm512_set1_ps(1), zero = _mm512_setzero_ps();
    for (int64_t j = 0; j < n; j += 16) {
        const __m512 m  = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + j), vs), _mm512_loadu_ps(w + j));
        const __m512 zz = _mm512_loadu_ps(z + j);
        const __m512 sl = _mm512_div_ps(zz, _mm512_add_ps(one, xdna_v_expf512(_mm512_sub_ps(zero, zz))));
        _mm512_storeu_ps(y + j, _mm512_mul_ps(m, sl));
    }
}

__attribute__((target("avx2,fma"))) static inline void xdna_gated_row256(const float * x, const float * w, float s,
                                                                        const float * z, float * y, int64_t n) {
    const __m256 vs = _mm256_set1_ps(s), one = _mm256_set1_ps(1), zero = _mm256_setzero_ps();
    for (int64_t j = 0; j < n; j += 8) {
        const __m256 m  = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + j), vs), _mm256_loadu_ps(w + j));
        const __m256 zz = _mm256_loadu_ps(z + j);
        const __m256 sl = _mm256_div_ps(zz, _mm256_add_ps(one, xdna_v_expf256(_mm256_sub_ps(zero, zz))));
        _mm256_storeu_ps(y + j, _mm256_mul_ps(m, sl));
    }
}

// n a multiple of 16
static inline void xdna_gated_row(const float * x, const float * w, float s, const float * z, float * y, int64_t n) {
    static const bool avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    if (avx512) {
        xdna_gated_row512(x, w, s, z, y, n);
    } else {
        xdna_gated_row256(x, w, s, z, y, n);
    }
}

// f32 bits to bf16, rounded to nearest even as xdna_bf16 (16 at a time)
__attribute__((target("avx2"))) static inline __m256i xdna_bf16x16(__m256i a, __m256i b) {
    const __m256i bias = _mm256_set1_epi32(0x7FFF), one = _mm256_set1_epi32(1);
    a = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, bias), _mm256_and_si256(_mm256_srli_epi32(a, 16), one)), 16);
    b = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(b, bias), _mm256_and_si256(_mm256_srli_epi32(b, 16), one)), 16);
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
}

// n f16 (a multiple of 16) to bf16: xdna_bf16(ggml_fp16_to_fp32(x)), exactly
__attribute__((target("avx2,f16c"))) static inline void xdna_f16_to_bf16_row(const uint16_t * src, uint16_t * dst,
                                                                            int64_t n) {
    for (int64_t j = 0; j < n; j += 16) {
        const __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (src + j)));
        const __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (src + j + 8)));
        _mm256_storeu_si256((__m256i *) (dst + j), xdna_bf16x16(_mm256_castps_si256(a), _mm256_castps_si256(b)));
    }
}

// n f32 (a multiple of 16) times s to bf16: xdna_bf16(x * s)
__attribute__((target("avx2"))) static inline void xdna_scaled_bf16_row(const float * src, float s, uint16_t * dst,
                                                                       int64_t n) {
    const __m256 vs = _mm256_set1_ps(s);
    for (int64_t j = 0; j < n; j += 16) {
        const __m256 a = _mm256_mul_ps(_mm256_loadu_ps(src + j), vs);
        const __m256 b = _mm256_mul_ps(_mm256_loadu_ps(src + j + 8), vs);
        _mm256_storeu_si256((__m256i *) (dst + j), xdna_bf16x16(_mm256_castps_si256(a), _mm256_castps_si256(b)));
    }
}

// ggml-xdna.cpp's xdna_fast_sigmoid, 8 at a time: the same f32 operations in
// the same order (no FMA, as the scalar one is built), so the same bits.
__attribute__((target("avx2"))) static inline __m256 xdna_fast_sigmoid8(__m256 x) {
    __m256 t = _mm256_mul_ps(_mm256_xor_ps(x, _mm256_set1_ps(-0.0f)), _mm256_set1_ps(1.44269504f));
    t = _mm256_max_ps(_mm256_min_ps(t, _mm256_set1_ps(126.0f)), _mm256_set1_ps(-126.0f));
    const __m256 big = _mm256_set1_ps(12582912.0f);
    const __m256 n = _mm256_sub_ps(_mm256_add_ps(t, big), big);
    const __m256 f = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(1.535336188e-4f);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(1.339887440e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(9.618437357e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(5.550332471e-2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(2.402264791e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(6.931472028e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(1.0f));
    const __m256i bits = _mm256_add_epi32(_mm256_castps_si256(p), _mm256_slli_epi32(_mm256_cvttps_epi32(n), 23));
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, _mm256_castsi256_ps(bits)));
}

// y = a * sigmoid(g) for n (a multiple of 8) values: an attention output
// times its gate, as ggml's MUL of the two rounds it
__attribute__((target("avx2"))) static inline void xdna_gate_row(const float * a, const float * g, float * y, int64_t n) {
    for (int64_t j = 0; j < n; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_loadu_ps(a + j), xdna_fast_sigmoid8(_mm256_loadu_ps(g + j))));
    }
}

// y = a + b for n f32 (a multiple of 8), as ggml's ADD
__attribute__((target("avx2"))) static inline void xdna_add_row(const float * a, const float * b, float * y, int64_t n) {
    for (int64_t j = 0; j < n; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_add_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
    }
}
