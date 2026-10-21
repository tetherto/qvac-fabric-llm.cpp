// ggml-vector-index-sse42.cpp - SSSE3 TurboVec layout kernels (upstream
// turbovec 1.1.2 pack.rs) and the SSE4.2 CRC-32C used by every checksum.
// Built with SSSE3 and SSE4.2 enabled and called only after a runtime check.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

constexpr size_t kGroupBytes = 32;
// x86 search layout nibble order (upstream pack.rs PERM0) and its inverse.
alignas(16) constexpr uint8_t kPerm0[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
alignas(16) constexpr uint8_t kInvPerm0[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };

} // namespace

// Upstream pack.rs `interleave_chunk_ssse3`, in place over `len` bytes (a
// multiple of 32): for each 32-byte group of sequential lane bytes s,
// out[j] = (s[perm0[j]] >> 4) | (s[perm0[j] + 16] & 0xf0) and
// out[16 + j] = (s[perm0[j]] & 0x0f) | ((s[perm0[j] + 16] & 0x0f) << 4).
// Both halves of a group are loaded before either store.
void turbovec_interleave_ssse3(uint8_t * buf, size_t len) {
    const __m128i permv = _mm_load_si128(reinterpret_cast<const __m128i *>(kPerm0));
    const __m128i lo_mask = _mm_set1_epi8(0x0f);
    const __m128i hi_mask = _mm_set1_epi8(static_cast<char>(0xf0));
    const size_t n = len / kGroupBytes;
    for (size_t i = 0; i < n; ++i) {
        uint8_t * p = buf + i * kGroupBytes;
        const __m128i a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)), permv);
        const __m128i b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16)), permv);
        const __m128i a_hi = _mm_and_si128(_mm_srli_epi16(a, 4), lo_mask);
        const __m128i out_hi = _mm_or_si128(a_hi, _mm_and_si128(b, hi_mask));
        const __m128i b_lo4 = _mm_and_si128(b, lo_mask);
        const __m128i out_lo = _mm_or_si128(_mm_and_si128(a, lo_mask), _mm_slli_epi16(b_lo4, 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p), out_hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 16), out_lo);
    }
}

// Upstream pack.rs `deinterleave_chunk_ssse3`, the inverse of the interleave:
// ba = ((hi & 0x0f) << 4) | (lo & 0x0f) and bb = (hi & 0xf0) | (lo >> 4),
// scattered back to lane order through the inverse permutation. `blocked`
// and `out` do not overlap.
void turbovec_deinterleave_ssse3(const uint8_t * blocked, uint8_t * out, size_t len) {
    const __m128i invv = _mm_load_si128(reinterpret_cast<const __m128i *>(kInvPerm0));
    const __m128i lo_mask = _mm_set1_epi8(0x0f);
    const __m128i hi_mask = _mm_set1_epi8(static_cast<char>(0xf0));
    const size_t n = len / kGroupBytes;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t * b = blocked + i * kGroupBytes;
        uint8_t * o = out + i * kGroupBytes;
        const __m128i hi_plane = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));
        const __m128i lo_plane = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 16));
        const __m128i ba = _mm_or_si128(
            _mm_slli_epi16(_mm_and_si128(hi_plane, lo_mask), 4), _mm_and_si128(lo_plane, lo_mask));
        const __m128i bb = _mm_or_si128(
            _mm_and_si128(hi_plane, hi_mask), _mm_and_si128(_mm_srli_epi16(lo_plane, 4), lo_mask));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o), _mm_shuffle_epi8(ba, invv));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o + 16), _mm_shuffle_epi8(bb, invv));
    }
}

// CRC-32C with the SSE4.2 crc32 instruction (what upstream's crc32c crate
// uses on x86): each byte step equals the reflected table step, so the
// result matches the table loop for any input. No initial value or final
// inversion is applied here, as in crc32c_update.
uint32_t crc32c_update_sse42(uint32_t crc, const uint8_t * data, size_t size) {
#if defined(__x86_64__) || defined(_M_X64)
    while (size != 0 && (reinterpret_cast<uintptr_t>(data) & 7u) != 0) {
        crc = _mm_crc32_u8(crc, *data++);
        --size;
    }
    uint64_t crc64 = crc;
    for (; size >= 8; size -= 8, data += 8) {
        uint64_t word = 0;
        std::memcpy(&word, data, sizeof(word));
        crc64 = _mm_crc32_u64(crc64, word);
    }
    crc = static_cast<uint32_t>(crc64);
#else
    for (; size >= 4; size -= 4, data += 4) {
        uint32_t word = 0;
        std::memcpy(&word, data, sizeof(word));
        crc = _mm_crc32_u32(crc, word);
    }
#endif
    while (size != 0) {
        crc = _mm_crc32_u8(crc, *data++);
        --size;
    }
    return crc;
}

} // namespace ggml_vec_index_detail
