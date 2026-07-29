#include "ggml-vector-index-avx2.h"

#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <limits>

namespace ggml_vec_index_detail {
namespace {

inline double horizontal_sum(__m256d v) {
    const __m128d lo  = _mm256_castpd256_pd128(v);
    const __m128d hi  = _mm256_extractf128_pd(v, 1);
    const __m128d sum = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    return _mm_cvtsd_f64(sum);
}

inline __m256 cvtepi8_epi32_ps(__m128i bytes) {
    const __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(bytes, bytes), 8);
    const __m128i q32_lo = _mm_srai_epi32(_mm_unpacklo_epi16(q16, q16), 16);
    const __m128i q32_hi = _mm_srai_epi32(_mm_unpackhi_epi16(q16, q16), 16);
    const __m128 lo = _mm_cvtepi32_ps(q32_lo);
    const __m128 hi = _mm_cvtepi32_ps(q32_hi);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

inline void dot_q8_avx2_accum(
        const float * query,
        __m128i bytes,
        __m256d scale,
        __m256d & acc0,
        __m256d & acc1) {
    const __m256 q = cvtepi8_epi32_ps(bytes);
    const __m256d q0 = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(q)), scale);
    const __m256d q1 = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(q, 1)), scale);
    acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query)), q0));
    acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query + 4)), q1));
}

} // namespace

double dot_q8_avx2(const float * query, const int8_t * codes, float scale, int dim) {
    const __m256d scale_v = _mm256_set1_pd(static_cast<double>(scale));
    __m256d       acc0    = _mm256_setzero_pd();
    __m256d       acc1    = _mm256_setzero_pd();

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + i));
        dot_q8_avx2_accum(query + i, q8, scale_v, acc0, acc1);
    }

    double acc = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; i < dim; ++i) {
        const double value = static_cast<double>(codes[i]) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

double dot_q4_avx2(const float * query, const uint8_t * codes, float scale, int dim) {
    const __m128i low_mask   = _mm_set1_epi8(0x0f);
    const __m128i zero_point = _mm_set1_epi8(8);
    const __m256d scale_v    = _mm256_set1_pd(static_cast<double>(scale));
    __m256d       acc0       = _mm256_setzero_pd();
    __m256d       acc1       = _mm256_setzero_pd();

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + static_cast<size_t>(i) / 2));
        const __m128i low = _mm_and_si128(packed, low_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        const __m128i nibbles = _mm_unpacklo_epi8(low, high);
        const __m128i qbytes = _mm_sub_epi8(nibbles, zero_point);

        dot_q8_avx2_accum(query + i, qbytes, scale_v, acc0, acc1);
        dot_q8_avx2_accum(query + i + 8, _mm_srli_si128(qbytes, 8), scale_v, acc0, acc1);
    }

    double acc = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const double value = static_cast<double>(static_cast<int>(nibble) - 8) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

void score_turbovec_lut_block_avx2(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_byte_groups,
        size_t n_vectors,
        float * out_scores) {
    constexpr size_t block_size = 32;
    constexpr size_t flush_every = 256;
    const size_t base_vector = block_index * block_size;
    const size_t block_offset = block_index * n_byte_groups * block_size;
    const size_t valid_lanes = std::min(block_size, n_vectors - base_vector);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256 scale = _mm256_set1_ps(lut_scale);
    __m256 float_accum[4] = {
        _mm256_set1_ps(lut_bias),
        _mm256_set1_ps(lut_bias),
        _mm256_set1_ps(lut_bias),
        _mm256_set1_ps(lut_bias),
    };

    const size_t n_batches = (n_byte_groups + flush_every - 1) / flush_every;
    for (size_t batch = 0; batch < n_batches; ++batch) {
        const size_t group_begin = batch * flush_every;
        const size_t group_end = std::min(group_begin + flush_every, n_byte_groups);
        __m256i accum[4] = {
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        };
        for (size_t group = group_begin; group < group_end; ++group) {
            const __m256i codes = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(
                    blocked_codes + block_offset + group * block_size));
            const __m256i low = _mm256_and_si256(codes, nibble_mask);
            const __m256i high = _mm256_and_si256(
                _mm256_srli_epi16(codes, 4),
                nibble_mask);
            const __m256i table = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(lut + group * block_size));
            const __m256i low_values = _mm256_shuffle_epi8(table, low);
            const __m256i high_values = _mm256_shuffle_epi8(table, high);
            accum[0] = _mm256_add_epi16(accum[0], low_values);
            accum[1] = _mm256_add_epi16(accum[1], _mm256_srli_epi16(low_values, 8));
            accum[2] = _mm256_add_epi16(accum[2], high_values);
            accum[3] = _mm256_add_epi16(accum[3], _mm256_srli_epi16(high_values, 8));
        }

        accum[0] = _mm256_sub_epi16(
            accum[0],
            _mm256_slli_epi16(accum[1], 8));
        accum[2] = _mm256_sub_epi16(
            accum[2],
            _mm256_slli_epi16(accum[3], 8));
        const __m256i decoded0 = _mm256_add_epi16(
            _mm256_permute2x128_si256(accum[0], accum[1], 0x21),
            _mm256_blend_epi32(accum[0], accum[1], 0xf0));
        const __m256i decoded1 = _mm256_add_epi16(
            _mm256_permute2x128_si256(accum[2], accum[3], 0x21),
            _mm256_blend_epi32(accum[2], accum[3], 0xf0));
        const __m256 values[4] = {
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(decoded0))),
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(decoded0, 1))),
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(decoded1))),
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(decoded1, 1))),
        };
        for (int i = 0; i < 4; ++i) {
            float_accum[i] = _mm256_fmadd_ps(scale, values[i], float_accum[i]);
        }
    }

    if (valid_lanes == block_size) {
        for (int i = 0; i < 4; ++i) {
            const __m256 vector_scale =
                _mm256_loadu_ps(vector_scales + base_vector + static_cast<size_t>(i) * 8);
            _mm256_storeu_ps(
                out_scores + static_cast<size_t>(i) * 8,
                _mm256_mul_ps(float_accum[i], vector_scale));
        }
    } else {
        alignas(32) float decoded[block_size];
        for (int i = 0; i < 4; ++i) {
            _mm256_store_ps(decoded + static_cast<size_t>(i) * 8, float_accum[i]);
        }
        for (size_t lane = 0; lane < block_size; ++lane) {
            out_scores[lane] = lane < valid_lanes ?
                decoded[lane] * vector_scales[base_vector + lane] :
                -std::numeric_limits<float>::infinity();
        }
    }
}

} // namespace ggml_vec_index_detail
