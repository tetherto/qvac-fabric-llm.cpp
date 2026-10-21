// ggml-vector-index-turbovec-avx512bw.cpp - AVX-512 BW TurboVec search
// kernels, ported from upstream turbovec 1.1.2 search.rs. Built with AVX-512
// F and BW (plus AVX2 and FMA, which the flush epilogue uses, as upstream's
// does) and called only after a runtime check for all of them. Each kernel
// performs its scalar reference's operations on the same operands in the
// same order, so its output is bit-identical.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

constexpr size_t kBlock = 32;
constexpr size_t kFlushEvery = 256;

// Upstream `avx2_batch_flush_to_fa`: one block's u16 accumulators for one
// query through the SUB trick and the lane combine, converted and added
// into the f32 accumulators with fmadd(scale, partial, fa).
inline void batch_flush_to_fa(const __m256i accus[4], __m256 v_scale, __m256 fa[4]) {
    const __m256i a0 = _mm256_sub_epi16(accus[0], _mm256_slli_epi16(accus[1], 8));
    const __m256i a1 = accus[1];
    const __m256i a2 = _mm256_sub_epi16(accus[2], _mm256_slli_epi16(accus[3], 8));
    const __m256i a3 = accus[3];
    const __m256i dis0 = _mm256_add_epi16(_mm256_permute2x128_si256(a0, a1, 0x21), _mm256_blend_epi32(a0, a1, 0xf0));
    const __m256i dis1 = _mm256_add_epi16(_mm256_permute2x128_si256(a2, a3, 0x21), _mm256_blend_epi32(a2, a3, 0xf0));
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
    const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
    const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));
    fa[0] = _mm256_fmadd_ps(v_scale, f0, fa[0]);
    fa[1] = _mm256_fmadd_ps(v_scale, f1, fa[1]);
    fa[2] = _mm256_fmadd_ps(v_scale, f2, fa[2]);
    fa[3] = _mm256_fmadd_ps(v_scale, f3, fa[3]);
}

// `REV8[b]` is `b` with its bits reversed: a 16-lane compare mask has
// coordinate i in bit i, a plane byte has it in bit 7 - i.
struct Rev8 {
    uint8_t t[256];
    Rev8() : t() {
        for (int b = 0; b < 256; ++b) {
            int r = 0;
            for (int i = 0; i < 8; ++i) {
                r |= ((b >> i) & 1) << (7 - i);
            }
            t[b] = static_cast<uint8_t>(r);
        }
    }
};

} // namespace

// Upstream `search_multi_query_avx512bw`: four queries over a pair of
// consecutive blocks, the two blocks' 32-byte code groups side by side in
// one zmm so a single `vpshufb` looks both up. Two groups per inner step,
// with the odd group of a batch taken alone; FLUSH_EVERY batches as the
// AVX2 and NEON kernels have, so every flush, and hence every rounding,
// falls where theirs do. Writes fa[(block * 4 + query) * 32 + lane] =
// bias + sum over flushes of fmadd(scale, partial).
void turbovec_lut_pair_avx512bw(
        const uint8_t * const luts[4],
        const float scales[4],
        const float biases[4],
        const uint8_t * block0,
        size_t n_byte_groups,
        float * fa_out) {
    const uint8_t * block1 = block0 + n_byte_groups * kBlock;
    const __m512i mask512 = _mm512_set1_epi8(0x0f);
    __m256 v_scales[4];
    __m256 fa_b0[4][4];
    __m256 fa_b1[4][4];
    for (int q = 0; q < 4; ++q) {
        v_scales[q] = _mm256_set1_ps(scales[q]);
        for (int i = 0; i < 4; ++i) {
            fa_b0[q][i] = _mm256_set1_ps(biases[q]);
            fa_b1[q][i] = fa_b0[q][i];
        }
    }
    const size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
    for (size_t batch = 0; batch < n_batches; ++batch) {
        const size_t g_start = batch * kFlushEvery;
        const size_t g_end = g_start + kFlushEvery < n_byte_groups ? g_start + kFlushEvery : n_byte_groups;
        __m512i accus[4][4];
        for (auto & qa : accus) {
            for (__m512i & a : qa) {
                a = _mm512_setzero_si512();
            }
        }
        auto codes_at = [&](size_t g) {
            return _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(block0 + g * kBlock))),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block1 + g * kBlock)), 1);
        };
        size_t g_pair = g_start;
        while (g_pair + 1 < g_end) {
            const size_t g0 = g_pair;
            const size_t g1 = g0 + 1;
            const __m512i codes_a = codes_at(g0);
            const __m512i codes_b = codes_at(g1);
            const __m512i clo_a = _mm512_and_si512(codes_a, mask512);
            const __m512i chi_a = _mm512_and_si512(_mm512_srli_epi16(codes_a, 4), mask512);
            const __m512i clo_b = _mm512_and_si512(codes_b, mask512);
            const __m512i chi_b = _mm512_and_si512(_mm512_srli_epi16(codes_b, 4), mask512);
            for (int q = 0; q < 4; ++q) {
                const __m512i lut_a = _mm512_broadcast_i64x4(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(luts[q] + g0 * 32)));
                const __m512i lut_b = _mm512_broadcast_i64x4(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(luts[q] + g1 * 32)));
                const __m512i res0_a = _mm512_shuffle_epi8(lut_a, clo_a);
                const __m512i res1_a = _mm512_shuffle_epi8(lut_a, chi_a);
                const __m512i res0_b = _mm512_shuffle_epi8(lut_b, clo_b);
                const __m512i res1_b = _mm512_shuffle_epi8(lut_b, chi_b);
                accus[q][0] = _mm512_add_epi16(accus[q][0], _mm512_add_epi16(res0_a, res0_b));
                accus[q][1] = _mm512_add_epi16(
                    accus[q][1], _mm512_add_epi16(_mm512_srli_epi16(res0_a, 8), _mm512_srli_epi16(res0_b, 8)));
                accus[q][2] = _mm512_add_epi16(accus[q][2], _mm512_add_epi16(res1_a, res1_b));
                accus[q][3] = _mm512_add_epi16(
                    accus[q][3], _mm512_add_epi16(_mm512_srli_epi16(res1_a, 8), _mm512_srli_epi16(res1_b, 8)));
            }
            g_pair += 2;
        }
        for (size_t g = g_pair; g < g_end; ++g) {
            const __m512i codes_v = codes_at(g);
            const __m512i clo = _mm512_and_si512(codes_v, mask512);
            const __m512i chi = _mm512_and_si512(_mm512_srli_epi16(codes_v, 4), mask512);
            for (int q = 0; q < 4; ++q) {
                const __m512i lut = _mm512_broadcast_i64x4(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(luts[q] + g * 32)));
                const __m512i res0 = _mm512_shuffle_epi8(lut, clo);
                const __m512i res1 = _mm512_shuffle_epi8(lut, chi);
                accus[q][0] = _mm512_add_epi16(accus[q][0], res0);
                accus[q][1] = _mm512_add_epi16(accus[q][1], _mm512_srli_epi16(res0, 8));
                accus[q][2] = _mm512_add_epi16(accus[q][2], res1);
                accus[q][3] = _mm512_add_epi16(accus[q][3], _mm512_srli_epi16(res1, 8));
            }
        }
        for (int q = 0; q < 4; ++q) {
            const __m256i b0[4] = {
                _mm512_castsi512_si256(accus[q][0]),
                _mm512_castsi512_si256(accus[q][1]),
                _mm512_castsi512_si256(accus[q][2]),
                _mm512_castsi512_si256(accus[q][3]),
            };
            batch_flush_to_fa(b0, v_scales[q], fa_b0[q]);
            const __m256i b1[4] = {
                _mm512_extracti64x4_epi64(accus[q][0], 1),
                _mm512_extracti64x4_epi64(accus[q][1], 1),
                _mm512_extracti64x4_epi64(accus[q][2], 1),
                _mm512_extracti64x4_epi64(accus[q][3], 1),
            };
            batch_flush_to_fa(b1, v_scales[q], fa_b1[q]);
        }
    }
    for (int q = 0; q < 4; ++q) {
        for (int i = 0; i < 4; ++i) {
            _mm256_storeu_ps(fa_out + (static_cast<size_t>(q) * 32 + static_cast<size_t>(i) * 8), fa_b0[q][i]);
            _mm256_storeu_ps(fa_out + ((4 + static_cast<size_t>(q)) * 32 + static_cast<size_t>(i) * 8), fa_b1[q][i]);
        }
    }
}

// Upstream `build_low_masks_avx512`: sixteen coordinates (two plane bytes)
// per step, each weight trunc(|q| * inv + 0.5) capped at `top`, as the
// scalar build computes it lane for lane. Returns the signed weight sum.
// `dim` is a multiple of 16.
int32_t turbovec_build_low_masks_avx512(const float * q, size_t dim, float inv, uint32_t top, uint8_t * masks) {
    static const Rev8 rev8;
    constexpr size_t kLowBits = 6;
    constexpr size_t kLowChunk = 64;
    const __m512 invv = _mm512_set1_ps(inv);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512i topv = _mm512_set1_epi32(static_cast<int>(top));
    const __m512i zero = _mm512_setzero_si512();
    const __m512 zerof = _mm512_setzero_ps();
    // |x| by clearing the sign bit: the integer AND-NOT is AVX-512 F, its
    // float form (upstream's) would need DQ.
    const __m512i sign_bit = _mm512_castps_si512(_mm512_set1_ps(-0.0f));
    __m512i sum = _mm512_setzero_si512();
    for (size_t c = 0; c * 8 < dim; c += 2) {
        const __m512 x = _mm512_loadu_ps(q + c * 8);
        const __m512 a = _mm512_castsi512_ps(_mm512_andnot_si512(sign_bit, _mm512_castps_si512(x)));
        const __m512i w = _mm512_min_epi32(_mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(a, invv), half)), topv);
        const __mmask16 neg = _mm512_cmp_ps_mask(x, zerof, _CMP_LT_OQ);
        sum = _mm512_add_epi32(sum, _mm512_mask_sub_epi32(w, neg, zero, w));
        const size_t base = (c / kLowChunk) * 2 * kLowBits * kLowChunk + c % kLowChunk;
        for (size_t b = 0; b < kLowBits; ++b) {
            const __mmask16 on = _mm512_test_epi32_mask(w, _mm512_set1_epi32(1 << b));
            const uint16_t pos = static_cast<uint16_t>(on & static_cast<__mmask16>(~neg));
            const uint16_t ng = static_cast<uint16_t>(on & neg);
            const size_t p = base + b * kLowChunk;
            const size_t n = base + (kLowBits + b) * kLowChunk;
            masks[p] = rev8.t[pos & 0xff];
            masks[p + 1] = rev8.t[pos >> 8];
            masks[n] = rev8.t[ng & 0xff];
            masks[n + 1] = rev8.t[ng >> 8];
        }
    }
    return _mm512_reduce_add_epi32(sum);
}

} // namespace ggml_vec_index_detail
