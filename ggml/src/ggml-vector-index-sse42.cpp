// ggml-vector-index-sse42.cpp - the SSE4.2 CRC-32C used by every checksum.
// Built with SSE4.2 enabled and called only after a runtime check.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ggml_vec_index_detail {

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
