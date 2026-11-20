// ggml-vector-index-turbovec-avx512.cpp - AVX-512 TurboVec rotation
// kernels, ported from upstream turbovec 1.1.2 rotation.rs. Built with
// AVX-512 F and AVX2 enabled and called only after a runtime check for both
// (the sub-16 tails use AVX2 instructions, as upstream's do). The kernels
// perform the scalar reference's IEEE operations on the same operands in the
// same order, so their output is bit-identical.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

// Upstream rotation.rs `permute_gather_avx512`.
template <int MODE>
void permute_gather_avx512_impl(
        const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n) {
    const __m512 invv = _mm512_set1_ps(inv);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512i idx = _mm512_loadu_si512(static_cast<const void *>(perm + i));
        __m512 v = _mm512_i32gather_ps(idx, src, 4);
        if (MODE == 2) {
            v = _mm512_mul_ps(v, invv);
        }
        if (MODE != 0) {
            v = _mm512_mul_ps(v, _mm512_loadu_ps(signs + i));
        }
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(perm + i));
        __m256 v = _mm256_i32gather_ps(src, idx, 4);
        if (MODE == 2) {
            v = _mm256_mul_ps(v, _mm256_set1_ps(inv));
        }
        if (MODE != 0) {
            v = _mm256_mul_ps(v, _mm256_loadu_ps(signs + i));
        }
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) {
        const float v = src[perm[i]];
        dst[i] = MODE == 0 ? v : (MODE == 1 ? v * signs[i] : (v * inv) * signs[i]);
    }
}

} // namespace

void turbovec_permute_gather_avx512(
        int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n) {
    if (mode == 0) {
        permute_gather_avx512_impl<0>(src, perm, signs, inv, dst, n);
    } else if (mode == 1) {
        permute_gather_avx512_impl<1>(src, perm, signs, inv, dst, n);
    } else {
        permute_gather_avx512_impl<2>(src, perm, signs, inv, dst, n);
    }
}

// Upstream rotation.rs `wht_block_avx512`: the AVX2 schedule at 16 lanes,
// with stages len = 1, 2, 4, 8 resolved inside one register. A block of 8
// has no room for a 16-lane register and takes the AVX2 kernel.
void turbovec_wht_block_avx512(float * p, int block, float inv_sqrt_block) {
    if (block < 16) {
        turbovec_wht_block_avx2(p, block, inv_sqrt_block);
        return;
    }
    for (int j = 0; j < block; j += 16) {
        const __m512 v = _mm512_loadu_ps(p + j);
        // len = 1: pairs (k, k+1) inside each 128-bit lane.
        __m512 a = _mm512_shuffle_ps(v, v, 0xA0);  // 0b10_10_00_00
        __m512 b = _mm512_shuffle_ps(v, v, 0xF5);  // 0b11_11_01_01
        const __m512 r1 = _mm512_mask_blend_ps(0xAAAA, _mm512_add_ps(a, b), _mm512_sub_ps(a, b));
        // len = 2: pairs (k, k+2) inside each 128-bit lane.
        a = _mm512_shuffle_ps(r1, r1, 0x44);  // 0b01_00_01_00
        b = _mm512_shuffle_ps(r1, r1, 0xEE);  // 0b11_10_11_10
        const __m512 r2 = _mm512_mask_blend_ps(0xCCCC, _mm512_add_ps(a, b), _mm512_sub_ps(a, b));
        // len = 4: pairs across adjacent 128-bit lanes (0 and 1, 2 and 3).
        a = _mm512_shuffle_f32x4(r2, r2, 0xA0);  // 0b10_10_00_00
        b = _mm512_shuffle_f32x4(r2, r2, 0xF5);  // 0b11_11_01_01
        const __m512 r4 = _mm512_mask_blend_ps(0xF0F0, _mm512_add_ps(a, b), _mm512_sub_ps(a, b));
        // len = 8: pairs across the two 256-bit halves (lanes 0 and 2, 1 and 3).
        a = _mm512_shuffle_f32x4(r4, r4, 0x44);  // 0b01_00_01_00
        b = _mm512_shuffle_f32x4(r4, r4, 0xEE);  // 0b11_10_11_10
        const __m512 r8 = _mm512_mask_blend_ps(0xFF00, _mm512_add_ps(a, b), _mm512_sub_ps(a, b));
        _mm512_storeu_ps(p + j, r8);
    }

    int len = 16;
    for (; 4 * len < block; len <<= 3) {
        for (int i = 0; i < block; i += 8 * len) {
            for (int j = i; j < i + len; j += 16) {
                const __m512 a = _mm512_loadu_ps(p + j);
                const __m512 b = _mm512_loadu_ps(p + j + len);
                const __m512 c = _mm512_loadu_ps(p + j + 2 * len);
                const __m512 d = _mm512_loadu_ps(p + j + 3 * len);
                const __m512 e = _mm512_loadu_ps(p + j + 4 * len);
                const __m512 f = _mm512_loadu_ps(p + j + 5 * len);
                const __m512 g = _mm512_loadu_ps(p + j + 6 * len);
                const __m512 h = _mm512_loadu_ps(p + j + 7 * len);
                const __m512 apb = _mm512_add_ps(a, b);
                const __m512 amb = _mm512_sub_ps(a, b);
                const __m512 cpd = _mm512_add_ps(c, d);
                const __m512 cmd = _mm512_sub_ps(c, d);
                const __m512 epf = _mm512_add_ps(e, f);
                const __m512 emf = _mm512_sub_ps(e, f);
                const __m512 gph = _mm512_add_ps(g, h);
                const __m512 gmh = _mm512_sub_ps(g, h);
                const __m512 s0 = _mm512_add_ps(apb, cpd);
                const __m512 s1 = _mm512_add_ps(amb, cmd);
                const __m512 s2 = _mm512_sub_ps(apb, cpd);
                const __m512 s3 = _mm512_sub_ps(amb, cmd);
                const __m512 s4 = _mm512_add_ps(epf, gph);
                const __m512 s5 = _mm512_add_ps(emf, gmh);
                const __m512 s6 = _mm512_sub_ps(epf, gph);
                const __m512 s7 = _mm512_sub_ps(emf, gmh);
                _mm512_storeu_ps(p + j, _mm512_add_ps(s0, s4));
                _mm512_storeu_ps(p + j + len, _mm512_add_ps(s1, s5));
                _mm512_storeu_ps(p + j + 2 * len, _mm512_add_ps(s2, s6));
                _mm512_storeu_ps(p + j + 3 * len, _mm512_add_ps(s3, s7));
                _mm512_storeu_ps(p + j + 4 * len, _mm512_sub_ps(s0, s4));
                _mm512_storeu_ps(p + j + 5 * len, _mm512_sub_ps(s1, s5));
                _mm512_storeu_ps(p + j + 6 * len, _mm512_sub_ps(s2, s6));
                _mm512_storeu_ps(p + j + 7 * len, _mm512_sub_ps(s3, s7));
            }
        }
    }
    if (2 * len < block) {
        for (int i = 0; i < block; i += 4 * len) {
            for (int j = i; j < i + len; j += 16) {
                const __m512 a = _mm512_loadu_ps(p + j);
                const __m512 b = _mm512_loadu_ps(p + j + len);
                const __m512 c = _mm512_loadu_ps(p + j + 2 * len);
                const __m512 d = _mm512_loadu_ps(p + j + 3 * len);
                const __m512 apb = _mm512_add_ps(a, b);
                const __m512 amb = _mm512_sub_ps(a, b);
                const __m512 cpd = _mm512_add_ps(c, d);
                const __m512 cmd = _mm512_sub_ps(c, d);
                _mm512_storeu_ps(p + j, _mm512_add_ps(apb, cpd));
                _mm512_storeu_ps(p + j + len, _mm512_add_ps(amb, cmd));
                _mm512_storeu_ps(p + j + 2 * len, _mm512_sub_ps(apb, cpd));
                _mm512_storeu_ps(p + j + 3 * len, _mm512_sub_ps(amb, cmd));
            }
        }
        len <<= 2;
    }
    if (len < block) {
        for (int i = 0; i < block; i += 2 * len) {
            for (int j = i; j < i + len; j += 16) {
                const __m512 a = _mm512_loadu_ps(p + j);
                const __m512 b = _mm512_loadu_ps(p + j + len);
                _mm512_storeu_ps(p + j, _mm512_add_ps(a, b));
                _mm512_storeu_ps(p + j + len, _mm512_sub_ps(a, b));
            }
        }
    }

    const __m512 sv = _mm512_set1_ps(inv_sqrt_block);
    for (int j = 0; j < block; j += 16) {
        _mm512_storeu_ps(p + j, _mm512_mul_ps(_mm512_loadu_ps(p + j), sv));
    }
}

} // namespace ggml_vec_index_detail
