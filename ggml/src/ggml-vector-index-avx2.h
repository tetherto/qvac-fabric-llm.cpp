#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_vec_index_detail {

double dot_q8_avx2(const float * query, const int8_t * codes, float scale, int dim);
double dot_q4_avx2(const float * query, const uint8_t * codes, float scale, int dim);
void score_turbovec_lut_block_avx2(
    const uint8_t * lut,
    float lut_scale,
    float lut_bias,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t block_index,
    size_t n_byte_groups,
    size_t n_vectors,
    float * out_scores);
void score_turbovec_lut_block4_avx2(
    const uint8_t * const luts[4],
    const float scales[4],
    const float biases[4],
    const uint8_t * block,
    size_t n_byte_groups,
    float * fa_out);

} // namespace ggml_vec_index_detail
