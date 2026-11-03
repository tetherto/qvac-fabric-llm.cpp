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

} // namespace ggml_vec_index_detail
