#pragma once

// x86 forms of the TurboVec rotation, encode, layout and checksum kernels,
// ported from upstream turbovec 1.1.2. Each object is built with its own ISA
// flags and runs only after the matching runtime feature check
// (vec_index_cpu_features); every kernel is bit-identical to its scalar
// reference.

#include <cstddef>
#include <cstdint>

namespace ggml_vec_index_detail {

// ggml-vector-index-turbovec-avx2.cpp (AVX2 + FMA object).
void turbovec_wht_block_avx2(float * blk, int block, float inv_sqrt_block);
void turbovec_permute_gather_avx2(
    int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n);
float turbovec_norm_sq_avx(const float * row, int dim);
double turbovec_quantize_row_avx2(
    const float * rotated,
    const float * shift,
    const float * scale_tq,
    const float * inv_scale_tq,
    const float * boundaries,
    const float * centroids,
    int bits,
    int dim,
    uint8_t * packed_row);
size_t first_invalid_abs_avx2(const float * values, size_t n, float max_abs);
void turbovec_interleave_avx2(uint8_t * buf, size_t len);

// ggml-vector-index-turbovec-avx512.cpp (AVX-512 F + AVX2 object).
void turbovec_wht_block_avx512(float * blk, int block, float inv_sqrt_block);
void turbovec_permute_gather_avx512(
    int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n);

// ggml-vector-index-sse42.cpp (SSSE3 + SSE4.2 object).
void turbovec_interleave_ssse3(uint8_t * buf, size_t len);
void turbovec_deinterleave_ssse3(const uint8_t * blocked, uint8_t * out, size_t len);
uint32_t crc32c_update_sse42(uint32_t crc, const uint8_t * data, size_t size);

} // namespace ggml_vec_index_detail
