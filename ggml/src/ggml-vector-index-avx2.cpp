#include "ggml-vector-index-avx2.h"

#include <immintrin.h>

#include <cstddef>

namespace ggml_vec_index_detail {
namespace {

inline double horizontal_sum(__m256d v) {
    const __m128d lo  = _mm256_castpd256_pd128(v);
    const __m128d hi  = _mm256_extractf128_pd(v, 1);
    const __m128d sum = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    return _mm_cvtsd_f64(sum);
}

inline __m256 cvtepi8_epi32_ps(__m128i bytes) {
    const __m128i q16    = _mm_srai_epi16(_mm_unpacklo_epi8(bytes, bytes), 8);
    const __m128i q32_lo = _mm_srai_epi32(_mm_unpacklo_epi16(q16, q16), 16);
    const __m128i q32_hi = _mm_srai_epi32(_mm_unpackhi_epi16(q16, q16), 16);
    const __m128  lo     = _mm_cvtepi32_ps(q32_lo);
    const __m128  hi     = _mm_cvtepi32_ps(q32_hi);
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

}  // namespace

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
        const __m128i packed  = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + static_cast<size_t>(i) / 2));
        const __m128i low     = _mm_and_si128(packed, low_mask);
        const __m128i high    = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        const __m128i nibbles = _mm_unpacklo_epi8(low, high);
        const __m128i qbytes  = _mm_sub_epi8(nibbles, zero_point);

        dot_q8_avx2_accum(query + i, qbytes, scale_v, acc0, acc1);
        dot_q8_avx2_accum(query + i + 8, _mm_srli_si128(qbytes, 8), scale_v, acc0, acc1);
    }

    double acc = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; i < dim; ++i) {
        const uint8_t byte   = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ? static_cast<uint8_t>(byte & 0x0f) : static_cast<uint8_t>(byte >> 4);
        const double  value  = static_cast<double>(static_cast<int>(nibble) - 8) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

}  // namespace ggml_vec_index_detail
