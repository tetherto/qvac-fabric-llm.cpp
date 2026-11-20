#pragma once

// Upstream turbovec 1.1.2 search.rs `define_permute_dot!`: the x86
// permute-dot scan of 4-bit codes in the vector-major layout, shared by the
// AVX-512 VNNI object and its GFNI twin, which differ only in how a code
// byte's high nibble is extracted. Included by those two objects alone;
// everything here has internal linkage, so neither object can end up
// calling code compiled for the other's instruction set.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ggml_vec_index_detail {
namespace {

// Integer sums sum_d weight[d] * level[code[d]] of NQ queries over up to BLK
// consecutive blocks (n_blocks of them), into sums[(sub * NQ + q) * 32 +
// lane]. `weights[q]` is upstream's `QueryPermuteDot::weights` (per four
// byte groups, the four low-nibble weights then the four high-nibble ones)
// and `zeros[q]` its `zero` (-128 times the weight sum), which cancels the
// +128 that puts the signed levels into `vpdpbusd`'s unsigned operand.
// `bytes_after` bounds the prefetches: the cache bytes from `blocks` on.
template <int NQ, int BLK, typename HiNibble>
void permute_dot_sums_x86(
        const int8_t * levels16,
        const int8_t * const * weights,
        const int32_t * zeros,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums,
        HiNibble hi_nibble) {
    constexpr size_t kBlock = 32;
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const size_t quads = n_byte_groups / 4;
    const size_t block_bytes = n_byte_groups * kBlock;
    const size_t nb = n_blocks < static_cast<size_t>(BLK) ? n_blocks : static_cast<size_t>(BLK);
    // H62: a single sweep prefetches 8 quads ahead, a batch 32.
    constexpr size_t pf_quads = NQ == 1 ? 8 : 32;
    const __m512i levels = _mm512_xor_si512(
        _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(levels16))),
        _mm512_set1_epi8(static_cast<char>(0x80)));

    // Zeroed in full, as upstream does: only the first nb are seeded.
    __m512i acc[BLK][NQ][2] = {};
    for (size_t sub = 0; sub < nb; ++sub) {
        for (int q = 0; q < NQ; ++q) {
            const __m512i z = _mm512_set1_epi32(zeros[q]);
            acc[sub][q][0] = z;
            acc[sub][q][1] = z;
        }
    }
    for (size_t q4 = 0; q4 < quads; ++q4) {
        for (size_t h = 0; h < 2; ++h) {
            for (size_t sub = 0; sub < nb; ++sub) {
                const size_t base = sub * block_bytes;
                const size_t pf = base + (q4 + pf_quads) * 128 + h * 64;
                if (pf + 64 <= bytes_after) {
                    _mm_prefetch(reinterpret_cast<const char *>(blocks + pf), _MM_HINT_T0);
                }
                const __m512i c = _mm512_loadu_si512(reinterpret_cast<const void *>(blocks + base + q4 * 128 + h * 64));
                // One shared nibble -> level lookup for every query.
                const __m512i vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
                const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibble(c, m0f));
                for (int q = 0; q < NQ; ++q) {
                    // Four dimensions of query weights, broadcast.
                    int32_t wlo_bits;
                    int32_t whi_bits;
                    std::memcpy(&wlo_bits, weights[q] + q4 * 8, 4);
                    std::memcpy(&whi_bits, weights[q] + q4 * 8 + 4, 4);
                    acc[sub][q][h] = _mm512_dpbusd_epi32(acc[sub][q][h], vlo, _mm512_set1_epi32(wlo_bits));
                    acc[sub][q][h] = _mm512_dpbusd_epi32(acc[sub][q][h], vhi, _mm512_set1_epi32(whi_bits));
                }
            }
        }
    }
    for (size_t sub = 0; sub < nb; ++sub) {
        for (int q = 0; q < NQ; ++q) {
            int32_t * out = sums + (sub * NQ + static_cast<size_t>(q)) * kBlock;
            _mm512_storeu_si512(reinterpret_cast<void *>(out), acc[sub][q][0]);
            _mm512_storeu_si512(reinterpret_cast<void *>(out + 16), acc[sub][q][1]);
        }
    }
}

// The entry point both objects export: nq is 1 (up to eight blocks
// interleaved, upstream's single-query instantiation), 8 or 10 (one block,
// upstream's batch widths); any other nq runs as single queries.
template <typename HiNibble>
void permute_dot_sums_dispatch(
        const int8_t * levels16,
        const int8_t * const * weights,
        const int32_t * zeros,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums,
        HiNibble hi_nibble) {
    constexpr size_t kBlock = 32;
    if (nq == 10 && n_blocks == 1) {
        permute_dot_sums_x86<10, 1>(levels16, weights, zeros, blocks, 1, n_byte_groups, bytes_after, sums, hi_nibble);
        return;
    }
    if (nq == 8 && n_blocks == 1) {
        permute_dot_sums_x86<8, 1>(levels16, weights, zeros, blocks, 1, n_byte_groups, bytes_after, sums, hi_nibble);
        return;
    }
    if (nq == 1) {
        const size_t block_bytes = n_byte_groups * kBlock;
        for (size_t b = 0; b < n_blocks; b += 8) {
            const size_t nb = n_blocks - b < 8 ? n_blocks - b : 8;
            permute_dot_sums_x86<1, 8>(
                levels16, weights, zeros, blocks + b * block_bytes, nb, n_byte_groups, bytes_after - b * block_bytes,
                sums + b * kBlock, hi_nibble);
        }
        return;
    }
    const size_t block_bytes = n_byte_groups * kBlock;
    for (size_t b = 0; b < n_blocks; ++b) {
        for (int q = 0; q < nq; ++q) {
            permute_dot_sums_x86<1, 1>(
                levels16, weights + q, zeros + q, blocks + b * block_bytes, 1, n_byte_groups,
                bytes_after - b * block_bytes, sums + (b * static_cast<size_t>(nq) + static_cast<size_t>(q)) * kBlock,
                hi_nibble);
        }
    }
}

} // namespace
} // namespace ggml_vec_index_detail
