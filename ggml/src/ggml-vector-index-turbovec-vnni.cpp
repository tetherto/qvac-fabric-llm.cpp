// ggml-vector-index-turbovec-vnni.cpp - AVX-512 VBMI/VNNI TurboVec search
// kernels over upstream's vector-major layout, ported from upstream turbovec
// 1.1.2 search.rs. Built with AVX-512 F, BW, VBMI and VNNI enabled and
// called only after a runtime check for all four (turbovec_cpu_dot_product).
// Every kernel returns exact integer sums, equal to its scalar model's in
// ggml-vector-index-turbovec.cpp whatever the order of accumulation; the
// float epilogue is shared with that model.

#include "ggml-vector-index-x86.h"
#include "ggml-vector-index-turbovec-pd-x86.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

constexpr size_t kBlock = 32;

// Upstream `search_multi_query_vnni::<PF, NQ>`: per four byte groups,
// `vpermb` picks each byte's entry from the four groups' 16-entry tables
// (the index is (j << 4) | code for byte j of a vector's dword) and
// `vpdpbusd` adds the four entries into that vector's u32 lane.
template <int NQ, bool PF>
void vnni_lut_sums_nq(
        const uint8_t * const * split_luts,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const __m512i kpos = _mm512_set1_epi32(0x30201000);
    const __m512i ones = _mm512_set1_epi8(1);
    const size_t quads = n_byte_groups / 4;
    __m512i acc[NQ][2];
    for (int q = 0; q < NQ; ++q) {
        acc[q][0] = _mm512_setzero_si512();
        acc[q][1] = _mm512_setzero_si512();
    }
    for (size_t q4 = 0; q4 < quads; ++q4) {
        for (size_t h = 0; h < 2; ++h) {
            if (PF) {
                const size_t pf = (q4 + 8) * 128 + h * 64;
                if (pf + 64 <= bytes_after) {
                    _mm_prefetch(reinterpret_cast<const char *>(block + pf), _MM_HINT_T0);
                }
            }
            const __m512i c = _mm512_loadu_si512(reinterpret_cast<const void *>(block + q4 * 128 + h * 64));
            const __m512i ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
            const __m512i ihi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
            for (int q = 0; q < NQ; ++q) {
                const uint8_t * tp = split_luts[q] + q4 * 128;
                const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const void *>(tp));
                const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const void *>(tp + 64));
                acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], _mm512_permutexvar_epi8(ilo, tlo), ones);
                acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], _mm512_permutexvar_epi8(ihi, thi), ones);
            }
        }
    }
    for (int q = 0; q < NQ; ++q) {
        _mm512_storeu_si512(reinterpret_cast<void *>(sums + static_cast<size_t>(q) * kBlock), acc[q][0]);
        _mm512_storeu_si512(reinterpret_cast<void *>(sums + static_cast<size_t>(q) * kBlock + 16), acc[q][1]);
    }
}

// Upstream `search_single_query_vnni_blk2`: one query over two blocks in
// flight, sharing each quad's table loads; prefetches the first block's
// stream eight quads ahead.
void vnni_lut_sums_blk2(
        const uint8_t * split_lut,
        const uint8_t * block0,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const __m512i kpos = _mm512_set1_epi32(0x30201000);
    const __m512i ones = _mm512_set1_epi8(1);
    const size_t quads = n_byte_groups / 4;
    const uint8_t * block1 = block0 + n_byte_groups * kBlock;
    __m512i a0[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
    __m512i a1[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
    for (size_t q4 = 0; q4 < quads; ++q4) {
        for (size_t h = 0; h < 2; ++h) {
            const size_t pf = (q4 + 8) * 128 + h * 64;
            if (pf + 64 <= bytes_after) {
                _mm_prefetch(reinterpret_cast<const char *>(block0 + pf), _MM_HINT_T0);
            }
            const uint8_t * tp = split_lut + q4 * 128;
            const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const void *>(tp));
            const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const void *>(tp + 64));
            const __m512i c0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block0 + q4 * 128 + h * 64));
            const __m512i c1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block1 + q4 * 128 + h * 64));
            const __m512i i0lo = _mm512_or_si512(_mm512_and_si512(c0, m0f), kpos);
            const __m512i i0hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c0, 4), m0f), kpos);
            const __m512i i1lo = _mm512_or_si512(_mm512_and_si512(c1, m0f), kpos);
            const __m512i i1hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c1, 4), m0f), kpos);
            a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0lo, tlo), ones);
            a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0hi, thi), ones);
            a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1lo, tlo), ones);
            a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1hi, thi), ones);
        }
    }
    _mm512_storeu_si512(reinterpret_cast<void *>(sums), a0[0]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + 16), a0[1]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + kBlock), a1[0]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + kBlock + 16), a1[1]);
}

struct HiNibbleShift {
    __m512i operator()(__m512i c, __m512i m0f) const {
        return _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
    }
};

} // namespace

void turbovec_vnni_lut_sums(
        const uint8_t * const * split_luts,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    const size_t block_bytes = n_byte_groups * kBlock;
    if (nq == 1) {
        size_t b = 0;
        for (; b + 2 <= n_blocks; b += 2) {
            vnni_lut_sums_blk2(split_luts[0], blocks + b * block_bytes, n_byte_groups, bytes_after - b * block_bytes,
                sums + b * kBlock);
        }
        if (b < n_blocks) {
            // The odd tail block, single stream, without prefetch.
            vnni_lut_sums_nq<1, false>(split_luts, blocks + b * block_bytes, n_byte_groups, 0, sums + b * kBlock);
        }
        return;
    }
    // H53: one instantiation per batch width, so the accumulators stay in
    // registers; the batch re-reads the codes and does not prefetch.
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t * block = blocks + b * block_bytes;
        int32_t * out = sums + b * static_cast<size_t>(nq) * kBlock;
        switch (nq) {
            case 2: vnni_lut_sums_nq<2, false>(split_luts, block, n_byte_groups, 0, out); break;
            case 3: vnni_lut_sums_nq<3, false>(split_luts, block, n_byte_groups, 0, out); break;
            case 4: vnni_lut_sums_nq<4, false>(split_luts, block, n_byte_groups, 0, out); break;
            case 5: vnni_lut_sums_nq<5, false>(split_luts, block, n_byte_groups, 0, out); break;
            case 6: vnni_lut_sums_nq<6, false>(split_luts, block, n_byte_groups, 0, out); break;
            case 7: vnni_lut_sums_nq<7, false>(split_luts, block, n_byte_groups, 0, out); break;
            default:
                // Callers batch at most 8 (the scan's width is 6); more would
                // leave the extra queries' sums unwritten, as upstream would.
                assert(nq <= 8);
                vnni_lut_sums_nq<8, false>(split_luts, block, n_byte_groups, 0, out);
                break;
        }
    }
}

void turbovec_permute_dot_sums_vnni(
        const int8_t * levels16,
        const int8_t * const * weights,
        const int32_t * zeros,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    permute_dot_sums_dispatch(
        levels16, weights, zeros, nq, blocks, n_blocks, n_byte_groups, bytes_after, sums, HiNibbleShift{});
}

// Upstream `exact4_sum_avx512`: one vector's 4-bit permute-dot sum read
// straight from a planes cache, its sign plane from the vector-major sign
// region and its three low planes from the vector's packed row. `w` is
// laid out the way a plane byte is read (position 8c + b is dim
// 8c + 7 - b) and zero-padded to whole 64s.
int32_t turbovec_exact4_sum_avx512(
        const int8_t * w,
        const int8_t * levels16,
        int32_t zero,
        const uint8_t * sign_block,
        const uint8_t * low_row,
        size_t nsg,
        size_t lane) {
    auto quad = [&](size_t q) -> uint32_t {
        uint32_t v;
        std::memcpy(&v, sign_block + q * 128 + (lane / 16) * 64 + (lane % 16) * 4, 4);
        return v;
    };
    auto rd = [](const uint8_t * p, bool wide) -> uint64_t {
        if (wide) {
            uint64_t v;
            std::memcpy(&v, p, 8);
            return v;
        }
        uint32_t v;
        std::memcpy(&v, p, 4);
        return v;
    };
    const __m512i levels = _mm512_xor_si512(
        _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(levels16))),
        _mm512_set1_epi8(static_cast<char>(-128)));
    const __m512i b8 = _mm512_set1_epi8(8);
    const __m512i b4 = _mm512_set1_epi8(4);
    const __m512i b2 = _mm512_set1_epi8(2);
    const __m512i b1 = _mm512_set1_epi8(1);
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < nsg; c += 8) {
        // Eight bytes of each plane, or the last four.
        const bool wide = c + 8 <= nsg;
        const uint64_t ks = wide ? (static_cast<uint64_t>(quad(c / 4)) | (static_cast<uint64_t>(quad(c / 4 + 1)) << 32)) :
                                   static_cast<uint64_t>(quad(c / 4));
        const uint64_t k2 = rd(low_row + 2 * nsg + c, wide);
        const uint64_t k1 = rd(low_row + nsg + c, wide);
        const uint64_t k0 = rd(low_row + c, wide);
        const __m512i codes = _mm512_or_si512(
            _mm512_or_si512(_mm512_maskz_mov_epi8(ks, b8), _mm512_maskz_mov_epi8(k2, b4)),
            _mm512_or_si512(_mm512_maskz_mov_epi8(k1, b2), _mm512_maskz_mov_epi8(k0, b1)));
        const __m512i lv = _mm512_shuffle_epi8(levels, codes);
        const __m512i wv = _mm512_loadu_si512(reinterpret_cast<const void *>(w + c * 8));
        acc = _mm512_dpbusd_epi32(acc, lv, wv);
    }
    return static_cast<int32_t>(static_cast<uint32_t>(_mm512_reduce_add_epi32(acc)) + static_cast<uint32_t>(zero));
}

} // namespace ggml_vec_index_detail
