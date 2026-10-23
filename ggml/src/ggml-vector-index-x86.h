#pragma once

// x86 forms of the TurboVec rotation, encode, layout, checksum and search
// kernels, ported from upstream turbovec 1.1.2. Each object is built with
// its own ISA flags and runs only after the matching runtime feature check
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
// A permute-dot block's scores: fma(sum, scale, bias) * vector_scales[lane]
// for the first `valid` of 32 lanes, -inf after. The hardware fused
// multiply-add rounds once, as std::fma does, without a libm call.
void permute_dot_finish_avx2(
    const int32_t * sums, float scale, float bias, const float * vector_scales, size_t valid, float * out_scores);
void turbovec_interleave_avx2(uint8_t * buf, size_t len);

// ggml-vector-index-turbovec-avx512.cpp (AVX-512 F + AVX2 object).
void turbovec_wht_block_avx512(float * blk, int block, float inv_sqrt_block);
void turbovec_permute_gather_avx512(
    int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n);

// ggml-vector-index-turbovec-avx512bw.cpp (AVX-512 F + BW, AVX2 + FMA
// object): the four-query LUT scan over a pair of blocks, fa[(block * 4 +
// query) * 32 + lane] before the vector scales, and the staged search's
// AVX-512 bit-mask build (dim a multiple of 16).
void turbovec_lut_pair_avx512bw(
    const uint8_t * const luts[4],
    const float scales[4],
    const float biases[4],
    const uint8_t * block0,
    size_t n_byte_groups,
    float * fa_out);
int32_t turbovec_build_low_masks_avx512(const float * q, size_t dim, float inv, uint32_t top, uint8_t * masks);

// ggml-vector-index-turbovec-vnni.cpp (AVX-512 F + BW + VBMI + VNNI
// object). Exact integer sums over vector-major blocks, laid out
// sums[(block * nq + query) * 32 + lane]; `bytes_after` (the cache bytes
// from `blocks` on) bounds the prefetches.
// The `vpermb` scan of 2- and 3-bit codes over split tables (nq 1 to 8; a
// lone query walks two blocks at a time).
void turbovec_vnni_lut_sums(
    const uint8_t * const * split_luts,
    int nq,
    const uint8_t * blocks,
    size_t n_blocks,
    size_t n_byte_groups,
    size_t bytes_after,
    int32_t * sums);
// The 4-bit permute-dot scan: upstream's weight layout and zero.
void turbovec_permute_dot_sums_vnni(
    const int8_t * levels16,
    const int8_t * const * weights,
    const int32_t * zeros,
    int nq,
    const uint8_t * blocks,
    size_t n_blocks,
    size_t n_byte_groups,
    size_t bytes_after,
    int32_t * sums);
// One vector's 4-bit permute-dot sum from the vector-major sign region and
// its packed row's three low planes.
int32_t turbovec_exact4_sum_avx512(
    const int8_t * w,
    const int8_t * levels16,
    int32_t zero,
    const uint8_t * sign_block,
    const uint8_t * low_row,
    size_t nsg,
    size_t lane);

// ggml-vector-index-turbovec-gfni.cpp (AVX-512 F + BW + VNNI + GFNI
// object): the permute-dot scan with the GFNI nibble shift.
void turbovec_permute_dot_sums_gfni(
    const int8_t * levels16,
    const int8_t * const * weights,
    const int32_t * zeros,
    int nq,
    const uint8_t * blocks,
    size_t n_blocks,
    size_t n_byte_groups,
    size_t bytes_after,
    int32_t * sums);

// ggml-vector-index-turbovec-vpopcnt.cpp (AVX-512 F + BW + VPOPCNTDQ
// object): the staged search's bit-mask dot product.
int64_t turbovec_low_dot_avx512(const uint8_t * masks, const uint8_t * row, size_t row_len);

// ggml-vector-index-ssse3.cpp (SSSE3 object).
void turbovec_interleave_ssse3(uint8_t * buf, size_t len);
void turbovec_deinterleave_ssse3(const uint8_t * blocked, uint8_t * out, size_t len);
// ggml-vector-index-sse42.cpp (SSE4.2 object).
uint32_t crc32c_update_sse42(uint32_t crc, const uint8_t * data, size_t size);

} // namespace ggml_vec_index_detail
