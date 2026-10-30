#pragma once

// The arithmetic of the host attention kernel (xdna-host-attn.cpp) as portable_ops and avx512_ops, one interface;
// avx512_ops is picked at run time because this backend is built without -march.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#    define XDNA_HOST_ATTN_AVX512 1
#    include <immintrin.h>
#endif

namespace xdna_host_attn {

constexpr int KB          = 64;  // keys a block
constexpr int LANES       = 16;  // f32 lanes of a vector, and the head size's granule
constexpr int TILE_ROWS   = 4;   // rows a register tile
constexpr int TILE_KEYS   = 4;   // keys a score tile
constexpr int TILE_CHUNKS = 4;   // 16-float chunks of V a P V tile

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// The portable kernels.
struct portable_ops {
    static float widen(uint16_t x) { return ggml_fp16_to_fp32((ggml_fp16_t) x); }
    static float widen(float x) { return x; }

    template <typename T>
    static void qk(int rt, int jt, const float * const * q, const char * k0, size_t kstride, int dk, float * s, int ss) {
        for (int i = 0; i < rt; i++) {
            for (int j = 0; j < jt; j++) {
                s[i * ss + j] = dot<T>(q[i], (const T *) (k0 + (size_t) j * kstride), dk);
            }
        }
    }

    template <typename T>
    static void pv(int rt, int ct, const float * p, int ps, const char * v0, size_t vstride, int nj, float * o, int os,
                   const float * rs) {
        for (int i = 0; i < rt; i++) {
            scale_row(o + i * os, ct * LANES, rs[i]);
            for (int j = 0; j < nj; j++) {
                add_scaled<T>(o + i * os, (const T *) (v0 + (size_t) j * vstride), ct * LANES, p[i * ps + j]);
            }
        }
    }

    // The lanes outside [lane_lo, lane_hi] and past nk are hidden; `mask` (f16, or null) is added to the rest.
    // Returns the factor the row's earlier o and sum were scaled by.
    static float softmax_row(float * s, int lane_lo, int lane_hi, int nk, float scale, const uint16_t * mask, float * M,
                             float * L) {
        float row_max = NEG_INF;
        for (int j = 0; j < KB; j++) {
            const bool hidden = j < lane_lo || j > lane_hi || j >= nk;
            s[j]              = hidden ? NEG_INF : s[j] * scale + (mask ? widen(mask[j]) : 0.0f);
            row_max           = std::max(row_max, s[j]);
        }
        if (row_max == NEG_INF) {
            std::fill(s, s + KB, 0.0f);
            return 1.0f;
        }
        const float m_new = std::max(*M, row_max);
        const float rs    = *M == NEG_INF ? 0.0f : std::exp(*M - m_new);
        *L                = *L * rs + weights(s, m_new);
        *M                = m_new;
        return rs;
    }

    static void finalize_row(const float * o, float l, float * dst, int dv) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int d = 0; d < dv; d++) {
            dst[d] = o[d] * inv;
        }
    }

  private:
    template <typename T>
    static float dot(const float * q, const T * k, int n) {
        float sum = 0.0f;
        for (int d = 0; d < n; d++) {
            sum += q[d] * widen(k[d]);
        }
        return sum;
    }

    static void scale_row(float * o, int n, float f) {
        for (int d = 0; d < n; d++) {
            o[d] *= f;
        }
    }

    template <typename T>
    static void add_scaled(float * o, const T * v, int n, float f) {
        for (int d = 0; d < n; d++) {
            o[d] += f * widen(v[d]);
        }
    }

    // exp(s - m) in place, hidden scores to 0; returns their sum
    static float weights(float * s, float m) {
        float sum = 0.0f;
        for (int j = 0; j < KB; j++) {
            s[j] = s[j] == NEG_INF ? 0.0f : std::exp(s[j] - m);
            sum += s[j];
        }
        return sum;
    }
};

#ifdef XDNA_HOST_ATTN_AVX512

#    define XDNA_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")))
#    define XDNA_UNROLL _Pragma("GCC unroll 16")

// 2^f on |f| <= 1/2 as xdna_exp2_poly (xdna-util.h), highest degree first
constexpr float EXP2_POLY[7] = { 1.535336188e-4f, 1.339887440e-3f, 9.618437357e-3f, 5.550332471e-2f,
                                 2.402264791e-1f, 6.931472028e-1f, 1.0f };
constexpr float LOG2E        = 1.44269504088896341f;
constexpr float EXP2_FLOOR   = -126.0f;  // below it 2^x leaves the f32 normal range: a weight of 0

// _mm512_shuffle_f32x4 selectors: the 128-bit lanes 0 and 2, 1 and 3, of each source
constexpr int LANES_EVEN = 0x88;
constexpr int LANES_ODD  = 0xdd;

struct avx512_ops {
    XDNA_AVX512 static inline __m512 load_kv(const uint16_t * p) {
        return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) p));
    }
    XDNA_AVX512 static inline __m512 load_kv(const float * p) { return _mm512_loadu_ps(p); }

    // Lane j of the result is the sum of the 16 lanes of a[j]
    XDNA_AVX512 static inline __m512 reduce16(const __m512 * a) {
        __m512 t[8];
        XDNA_UNROLL
        for (int i = 0; i < 8; i++) {
            t[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2 * i], a[2 * i + 1]), _mm512_unpackhi_ps(a[2 * i], a[2 * i + 1]));
        }
        __m512 u[4];
        XDNA_UNROLL
        for (int i = 0; i < 4; i++) {
            const __m512d lo = _mm512_unpacklo_pd(_mm512_castps_pd(t[2 * i]), _mm512_castps_pd(t[2 * i + 1]));
            const __m512d hi = _mm512_unpackhi_pd(_mm512_castps_pd(t[2 * i]), _mm512_castps_pd(t[2 * i + 1]));
            u[i]             = _mm512_add_ps(_mm512_castpd_ps(lo), _mm512_castpd_ps(hi));
        }
        const __m512 x0 = fold_lanes(u[0], u[1]);
        const __m512 x1 = fold_lanes(u[2], u[3]);
        return fold_lanes(x0, x1);
    }

    // Sums the 128-bit lanes of a in pairs, then those of b: [a0+a1, a2+a3, b0+b1, b2+b3]
    XDNA_AVX512 static inline __m512 fold_lanes(__m512 a, __m512 b) {
        return _mm512_add_ps(_mm512_shuffle_f32x4(a, b, LANES_EVEN), _mm512_shuffle_f32x4(a, b, LANES_ODD));
    }

    template <typename T, int RT, int JT>
    XDNA_AVX512 static void qk_tile(const float * const * q, const char * k0, size_t kstride, int dk, float * s, int ss) {
        static_assert(RT * JT <= 16, "a tile reduces at most 16 accumulators");
        __m512    acc[RT][JT];
        const T * kp[JT];
        XDNA_UNROLL
        for (int j = 0; j < JT; j++) {
            kp[j] = (const T *) (k0 + (size_t) j * kstride);
            XDNA_UNROLL
            for (int i = 0; i < RT; i++) {
                acc[i][j] = _mm512_setzero_ps();
            }
        }
        for (int d = 0; d < dk; d += LANES) {
            __m512 kv[JT];
            XDNA_UNROLL
            for (int j = 0; j < JT; j++) {
                kv[j] = load_kv(kp[j] + d);
            }
            XDNA_UNROLL
            for (int i = 0; i < RT; i++) {
                const __m512 qv = _mm512_loadu_ps(q[i] + d);
                XDNA_UNROLL
                for (int j = 0; j < JT; j++) {
                    acc[i][j] = _mm512_fmadd_ps(qv, kv[j], acc[i][j]);
                }
            }
        }
        store_dots<RT, JT>(acc, s, ss);
    }

    // The RT x JT accumulators' lane sums into s[i][j]
    template <int RT, int JT>
    XDNA_AVX512 static void store_dots(const __m512 (&acc)[RT][JT], float * s, int ss) {
        __m512 a[16];
        XDNA_UNROLL
        for (int n = 0; n < 16; n++) {
            a[n] = n < RT * JT ? acc[(n / JT) % RT][n % JT] : _mm512_setzero_ps();
        }
        alignas(64) float dots[16];
        _mm512_store_ps(dots, reduce16(a));
        XDNA_UNROLL
        for (int i = 0; i < RT; i++) {
            XDNA_UNROLL
            for (int j = 0; j < JT; j++) {
                s[i * ss + j] = dots[i * JT + j];
            }
        }
    }

    template <typename T, int RT>
    XDNA_AVX512 static void qk_keys(int jt, const float * const * q, const char * k0, size_t ks, int dk, float * s, int ss) {
        switch (jt) {
            case 1:  qk_tile<T, RT, 1>(q, k0, ks, dk, s, ss); break;
            case 2:  qk_tile<T, RT, 2>(q, k0, ks, dk, s, ss); break;
            case 3:  qk_tile<T, RT, 3>(q, k0, ks, dk, s, ss); break;
            default: qk_tile<T, RT, 4>(q, k0, ks, dk, s, ss); break;
        }
    }

    template <typename T>
    XDNA_AVX512 static void qk(int rt, int jt, const float * const * q, const char * k0, size_t ks, int dk, float * s, int ss) {
        switch (rt) {
            case 1:  qk_keys<T, 1>(jt, q, k0, ks, dk, s, ss); break;
            case 2:  qk_keys<T, 2>(jt, q, k0, ks, dk, s, ss); break;
            case 3:  qk_keys<T, 3>(jt, q, k0, ks, dk, s, ss); break;
            default: qk_keys<T, 4>(jt, q, k0, ks, dk, s, ss); break;
        }
    }

    template <typename T, int RT, int CT>
    XDNA_AVX512 static void pv_tile(const float * p, int ps, const char * v0, size_t vstride, int nj, float * o, int os,
                                    const float * rs) {
        __m512 acc[RT][CT];
        XDNA_UNROLL
        for (int i = 0; i < RT; i++) {
            const __m512 r = _mm512_set1_ps(rs[i]);
            XDNA_UNROLL
            for (int c = 0; c < CT; c++) {
                acc[i][c] = _mm512_mul_ps(_mm512_loadu_ps(o + i * os + c * LANES), r);
            }
        }
        for (int j = 0; j < nj; j++) {
            const T * vp = (const T *) (v0 + (size_t) j * vstride);
            __m512    v[CT];
            XDNA_UNROLL
            for (int c = 0; c < CT; c++) {
                v[c] = load_kv(vp + c * LANES);
            }
            XDNA_UNROLL
            for (int i = 0; i < RT; i++) {
                const __m512 pb = _mm512_set1_ps(p[i * ps + j]);
                XDNA_UNROLL
                for (int c = 0; c < CT; c++) {
                    acc[i][c] = _mm512_fmadd_ps(pb, v[c], acc[i][c]);
                }
            }
        }
        XDNA_UNROLL
        for (int i = 0; i < RT; i++) {
            XDNA_UNROLL
            for (int c = 0; c < CT; c++) {
                _mm512_storeu_ps(o + i * os + c * LANES, acc[i][c]);
            }
        }
    }

    template <typename T, int RT>
    XDNA_AVX512 static void pv_chunks(int ct, const float * p, int ps, const char * v0, size_t vs, int nj, float * o, int os,
                                      const float * rs) {
        switch (ct) {
            case 1:  pv_tile<T, RT, 1>(p, ps, v0, vs, nj, o, os, rs); break;
            case 2:  pv_tile<T, RT, 2>(p, ps, v0, vs, nj, o, os, rs); break;
            case 3:  pv_tile<T, RT, 3>(p, ps, v0, vs, nj, o, os, rs); break;
            default: pv_tile<T, RT, 4>(p, ps, v0, vs, nj, o, os, rs); break;
        }
    }

    template <typename T>
    XDNA_AVX512 static void pv(int rt, int ct, const float * p, int ps, const char * v0, size_t vs, int nj, float * o, int os,
                               const float * rs) {
        switch (rt) {
            case 1:  pv_chunks<T, 1>(ct, p, ps, v0, vs, nj, o, os, rs); break;
            case 2:  pv_chunks<T, 2>(ct, p, ps, v0, vs, nj, o, os, rs); break;
            case 3:  pv_chunks<T, 3>(ct, p, ps, v0, vs, nj, o, os, rs); break;
            default: pv_chunks<T, 4>(ct, p, ps, v0, vs, nj, o, os, rs); break;
        }
    }

    // e^x for x <= 0, to ~2e-7 relative; 0 where 2^x leaves the f32 normal range (and for -inf)
    XDNA_AVX512 static inline __m512 exp_ps(__m512 x) {
        const __m512    t    = _mm512_mul_ps(x, _mm512_set1_ps(LOG2E));
        const __mmask16 live = _mm512_cmp_ps_mask(t, _mm512_set1_ps(EXP2_FLOOR), _CMP_GE_OQ);
        const __m512    n    = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m512    f    = _mm512_sub_ps(t, n);
        __m512          p    = _mm512_set1_ps(EXP2_POLY[0]);
        XDNA_UNROLL
        for (int i = 1; i < 7; i++) {
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(EXP2_POLY[i]));
        }
        return _mm512_maskz_scalef_ps(live, p, n);
    }

    // Bit i of the result for lo <= i <= hi, i in [0, 16)
    XDNA_AVX512 static inline __mmask16 lanes_between(int lo, int hi) {
        lo = std::max(lo, 0);
        hi = std::min(hi, LANES - 1);
        if (lo > hi) {
            return 0;
        }
        return (__mmask16) ((0xFFFFu >> (LANES - 1 - hi)) & (0xFFFFu << lo));
    }

    XDNA_AVX512 static float softmax_row(float * s, int lane_lo, int lane_hi, int nk, float scale, const uint16_t * mask,
                                         float * M, float * L) {
        constexpr int NV = KB / LANES;
        __m512        v[NV];
        __m512        row_max = _mm512_set1_ps(NEG_INF);
        XDNA_UNROLL
        for (int i = 0; i < NV; i++) {
            v[i] = _mm512_loadu_ps(s + i * LANES);
            if (scale != 1.0f) {
                v[i] = _mm512_mul_ps(v[i], _mm512_set1_ps(scale));
            }
            if (mask) {
                const __m256i mv = _mm256_maskz_loadu_epi16(lanes_between(0, nk - 1 - i * LANES), mask + i * LANES);
                v[i]             = _mm512_add_ps(v[i], _mm512_cvtph_ps(mv));
            }
            const __mmask16 shown = lanes_between(lane_lo - i * LANES, lane_hi - i * LANES);
            v[i]                  = _mm512_mask_mov_ps(_mm512_set1_ps(NEG_INF), shown, v[i]);
            row_max               = _mm512_max_ps(row_max, v[i]);
        }
        const float m_row = _mm512_reduce_max_ps(row_max);
        if (m_row == NEG_INF) {
            XDNA_UNROLL
            for (int i = 0; i < NV; i++) {
                _mm512_storeu_ps(s + i * LANES, _mm512_setzero_ps());
            }
            return 1.0f;
        }
        const float  m_new = std::max(*M, m_row);
        const float  rs    = *M == NEG_INF ? 0.0f : _mm512_cvtss_f32(exp_ps(_mm512_set1_ps(*M - m_new)));
        const __m512 mn    = _mm512_set1_ps(m_new);
        __m512       sum   = _mm512_setzero_ps();
        XDNA_UNROLL
        for (int i = 0; i < NV; i++) {
            v[i] = exp_ps(_mm512_sub_ps(v[i], mn));
            sum  = _mm512_add_ps(sum, v[i]);
            _mm512_storeu_ps(s + i * LANES, v[i]);
        }
        *L = *L * rs + _mm512_reduce_add_ps(sum);
        *M = m_new;
        return rs;
    }

    XDNA_AVX512 static void finalize_row(const float * o, float l, float * dst, int dv) {
        const __m512 inv = _mm512_set1_ps(l > 0.0f ? 1.0f / l : 0.0f);
        for (int d = 0; d < dv; d += LANES) {
            _mm512_storeu_ps(dst + d, _mm512_mul_ps(_mm512_loadu_ps(o + d), inv));
        }
    }
};

#endif  // XDNA_HOST_ATTN_AVX512

}  // namespace xdna_host_attn
