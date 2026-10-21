// ggml-vector-index-turbovec-vpopcnt.cpp - the AVX-512 VPOPCNTDQ form of
// the staged search's bit-mask dot product (upstream turbovec 1.1.2
// search.rs `low_dot_avx512`). Built with AVX-512 F, BW and VPOPCNTDQ
// enabled and called only after a runtime check for all three; it counts
// the same bits as the scalar `low_dot`, so the result is identical.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ggml_vec_index_detail {

// sum_b 2^b * (popcount(row & pos[b]) - popcount(row & neg[b])) over a low
// row of `row_len` bytes; `masks` holds, per 64 row bytes, pos[0..6] then
// neg[0..6], 64 bytes each. A masked load reads only a short tail's live
// bytes.
int64_t turbovec_low_dot_avx512(const uint8_t * masks, const uint8_t * row, size_t row_len) {
    constexpr size_t kLowBits = 6;
    constexpr size_t kLowChunk = 64;
    __m512i acc = _mm512_setzero_si512();
    const uint8_t * mp = masks;
    for (size_t off = 0; off < row_len; off += kLowChunk) {
        const size_t n = row_len - off < kLowChunk ? row_len - off : kLowChunk;
        const __mmask64 live = n == kLowChunk ? ~static_cast<__mmask64>(0) : (static_cast<__mmask64>(1) << n) - 1;
        const __m512i l = _mm512_maskz_loadu_epi8(live, row + off);
        for (size_t b = 0; b < kLowBits; ++b) {
            const __m512i pos = _mm512_loadu_si512(reinterpret_cast<const void *>(mp + b * kLowChunk));
            const __m512i neg = _mm512_loadu_si512(reinterpret_cast<const void *>(mp + (kLowBits + b) * kLowChunk));
            const __m512i d = _mm512_sub_epi64(
                _mm512_popcnt_epi64(_mm512_and_si512(l, pos)), _mm512_popcnt_epi64(_mm512_and_si512(l, neg)));
            acc = _mm512_add_epi64(acc, _mm512_sll_epi64(d, _mm_cvtsi32_si128(static_cast<int>(b))));
        }
        mp += 2 * kLowBits * kLowChunk;
    }
    return _mm512_reduce_add_epi64(acc);
}

} // namespace ggml_vec_index_detail
