// ggml-vector-index-search.cpp - search and IVF implementation.

#include "ggml-vector-index-impl.h"

#if GGML_VEC_INDEX_USE_NEON
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT 1
#else
#define GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT 0
#endif

#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
#include "ggml-vector-index-avx2.h"
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

namespace {

bool search_buffers_addressable(size_t n_q, size_t k, size_t dim) {
    if ((dim != 0 && n_q > std::numeric_limits<size_t>::max() / dim) ||
        n_q > std::numeric_limits<size_t>::max() / k) {
        return false;
    }
    const size_t query_count = n_q * dim;
    const size_t result_count = n_q * k;
    return can_address_array(query_count, sizeof(float)) &&
           can_address_array(result_count, sizeof(float)) &&
           can_address_array(result_count, sizeof(uint64_t));
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
static std::atomic<int64_t> g_turbovec_block_score_calls{ 0 };
#endif

inline float float_score_from_double(double score) {
    if (std::isnan(score)) {
        return -FLT_MAX;
    }
    if (score > static_cast<double>(FLT_MAX)) {
        return FLT_MAX;
    }
    if (score < -static_cast<double>(FLT_MAX)) {
        return -FLT_MAX;
    }
    return static_cast<float>(score);
}

inline double rank_score_from_double(double score) {
    return std::isnan(score) ? -std::numeric_limits<double>::infinity() : score;
}

// Scalar dot product of two `dim`-length f32 vectors.
inline double dot(const float * a, const float * b, int dim) {
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return rank_score_from_double(acc);
}

inline double dot_f32_fast(const float * a, const float * b, int dim) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int   i    = 0;
    for (; i + 4 <= dim; i += 4) {
        acc0 += a[i + 0] * b[i + 0];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; i < dim; ++i) {
        acc += a[i] * b[i];
    }
    return std::isfinite(acc) ? static_cast<double>(acc) : dot(a, b, dim);
}

inline double dot_q8_scalar(const float * query, const int8_t * codes, float scale, int dim) {
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double value = static_cast<double>(codes[i]) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

inline double dot_q4_scalar(const float * query, const uint8_t * codes, float scale, int dim) {
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const double value = static_cast<double>(q4_decode(nibble)) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

inline bool quantized_dot_float_path_is_safe(
        double max_query,
        int dim,
        float scale,
        float max_code) {
    const double max_value = static_cast<double>(max_code) * static_cast<double>(scale);
    return static_cast<double>(dim) * max_query * max_value <= static_cast<double>(FLT_MAX);
}

bool validate_queries_and_maybe_max_abs(
        const float * queries,
        int n_q,
        int dim,
        bool compute_max_abs,
        std::vector<double> & query_max_abs_values) {
    query_max_abs_values.clear();
    if (compute_max_abs) {
        query_max_abs_values.resize(static_cast<size_t>(n_q));
    }
    const size_t dim_sz = static_cast<size_t>(dim);
    for (int q = 0; q < n_q; ++q) {
        const float * query = queries + static_cast<size_t>(q) * dim_sz;
        double max_query = 0.0;
        for (int i = 0; i < dim; ++i) {
            const float value = query[i];
            if (!std::isfinite(value)) {
                return false;
            }
            if (compute_max_abs) {
                max_query = std::max(max_query, std::fabs(static_cast<double>(value)));
            }
        }
        if (compute_max_abs) {
            query_max_abs_values[static_cast<size_t>(q)] = max_query;
        }
    }
    return true;
}

#if GGML_VEC_INDEX_USE_NEON && defined(__aarch64__)

inline double horizontal_sum(float64x2_t v) {
    return vaddvq_f64(v);
}

inline void dot_q8_neon_accum(
        const float * query,
        int16x8_t codes,
        float64x2_t scale,
        float64x2_t & acc0,
        float64x2_t & acc1,
        float64x2_t & acc2,
        float64x2_t & acc3) {
    const int32x4_t q32_lo = vmovl_s16(vget_low_s16(codes));
    const int32x4_t q32_hi = vmovl_s16(vget_high_s16(codes));
    const float32x4_t qf_lo = vcvtq_f32_s32(q32_lo);
    const float32x4_t qf_hi = vcvtq_f32_s32(q32_hi);
    const float64x2_t q0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(qf_lo)), scale);
    const float64x2_t q1 = vmulq_f64(vcvt_f64_f32(vget_high_f32(qf_lo)), scale);
    const float64x2_t q2 = vmulq_f64(vcvt_f64_f32(vget_low_f32(qf_hi)), scale);
    const float64x2_t q3 = vmulq_f64(vcvt_f64_f32(vget_high_f32(qf_hi)), scale);
    acc0 = vmlaq_f64(acc0, vcvt_f64_f32(vld1_f32(query)), q0);
    acc1 = vmlaq_f64(acc1, vcvt_f64_f32(vld1_f32(query + 2)), q1);
    acc2 = vmlaq_f64(acc2, vcvt_f64_f32(vld1_f32(query + 4)), q2);
    acc3 = vmlaq_f64(acc3, vcvt_f64_f32(vld1_f32(query + 6)), q3);
}

inline double dot_q8_neon(const float * query, const int8_t * codes, float scale, int dim) {
    const float64x2_t scale_v = vdupq_n_f64(static_cast<double>(scale));
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        dot_q8_neon_accum(query + i, vmovl_s8(vld1_s8(codes + i)), scale_v, acc0, acc1, acc2, acc3);
    }

    double acc = horizontal_sum(acc0) + horizontal_sum(acc1) + horizontal_sum(acc2) + horizontal_sum(acc3);
    for (; i < dim; ++i) {
        const double value = static_cast<double>(codes[i]) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

inline void dot_q4_neon_accum8(const float * query,
                               uint8x8_t     codes,
                               float64x2_t   scale,
                               float64x2_t & acc0,
                               float64x2_t & acc1,
                               float64x2_t & acc2,
                               float64x2_t & acc3) {
    const int16x8_t q16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(codes)), vdupq_n_s16(8));
    dot_q8_neon_accum(query, q16, scale, acc0, acc1, acc2, acc3);
}

inline double dot_q4_neon(const float * query, const uint8_t * codes, float scale, int dim) {
    const float64x2_t scale_v = vdupq_n_f64(static_cast<double>(scale));
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        const uint8x8_t packed = vld1_u8(codes + static_cast<size_t>(i) / 2);
        const uint8x8_t low = vand_u8(packed, vdup_n_u8(0x0f));
        const uint8x8_t high = vshr_n_u8(packed, 4);
        const uint8x8x2_t zipped = vzip_u8(low, high);

        dot_q4_neon_accum8(query + i, zipped.val[0], scale_v, acc0, acc1, acc2, acc3);
        dot_q4_neon_accum8(query + i + 8, zipped.val[1], scale_v, acc0, acc1, acc2, acc3);
    }

    double acc = horizontal_sum(acc0) + horizontal_sum(acc1) + horizontal_sum(acc2) + horizontal_sum(acc3);
    for (; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const double value = static_cast<double>(q4_decode(nibble)) * static_cast<double>(scale);
        acc += static_cast<double>(query[i]) * value;
    }
    return acc;
}

#endif

#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)

bool cpu_has_avx2() {
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return false;
    }
    __cpuidex(regs, 1, 0);
    constexpr int kOsxsave = 1 << 27;
    constexpr int kAvx = 1 << 28;
    constexpr int kFma = 1 << 12;
    if ((regs[2] & (kOsxsave | kAvx)) != (kOsxsave | kAvx) ||
        (regs[2] & kFma) == 0 ||
        (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#endif

inline double dot_q8(const float * query, const int8_t * codes, float scale, int dim, double max_query) {
    if (dim < 8) {
        return dot_q8_scalar(query, codes, scale, dim);
    }
#if GGML_VEC_INDEX_USE_NEON && defined(__aarch64__)
    if (!quantized_dot_float_path_is_safe(max_query, dim, scale, 127.0f)) {
        return dot_q8_scalar(query, codes, scale, dim);
    }
    const double score = dot_q8_neon(query, codes, scale, dim);
    return std::isfinite(score) ? score : dot_q8_scalar(query, codes, scale, dim);
#elif defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    static const bool has_avx2 = cpu_has_avx2();
    if (has_avx2 && quantized_dot_float_path_is_safe(max_query, dim, scale, 127.0f)) {
        const double score = ggml_vec_index_detail::dot_q8_avx2(query, codes, scale, dim);
        return std::isfinite(score) ? score : dot_q8_scalar(query, codes, scale, dim);
    }
    return dot_q8_scalar(query, codes, scale, dim);
#else
    (void) max_query;
    return dot_q8_scalar(query, codes, scale, dim);
#endif
}

inline double dot_q4(const float * query, const uint8_t * codes, float scale, int dim, double max_query) {
    if (dim < 16) {
        return dot_q4_scalar(query, codes, scale, dim);
    }
#if GGML_VEC_INDEX_USE_NEON && defined(__aarch64__)
    if (!quantized_dot_float_path_is_safe(max_query, dim, scale, 7.0f)) {
        return dot_q4_scalar(query, codes, scale, dim);
    }
    const double score = dot_q4_neon(query, codes, scale, dim);
    return std::isfinite(score) ? score : dot_q4_scalar(query, codes, scale, dim);
#elif defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    static const bool has_avx2 = cpu_has_avx2();
    if (has_avx2 && quantized_dot_float_path_is_safe(max_query, dim, scale, 7.0f)) {
        const double score = ggml_vec_index_detail::dot_q4_avx2(query, codes, scale, dim);
        return std::isfinite(score) ? score : dot_q4_scalar(query, codes, scale, dim);
    }
    return dot_q4_scalar(query, codes, scale, dim);
#else
    (void) max_query;
    return dot_q4_scalar(query, codes, scale, dim);
#endif
}

inline double score_slot(
        const ggml_vec_index_t & idx,
        const float * score_query,
        size_t slot,
        double max_query,
        const uint8_t * turbovec_lut = nullptr,
        float turbovec_lut_scale = 1.0f,
        float turbovec_lut_bias = 0.0f) {
    const int dim = idx.dim;
    return is_turbovec_q2(idx) ?
        dot_turbovec_q2_lut_row(
            turbovec_lut,
            turbovec_lut_scale,
            turbovec_lut_bias,
            turbovec_q2_data_ptr(idx) + slot * turbovec_q2_row_bytes(static_cast<size_t>(dim)),
            idx.turbovec_q2_scale.data() + slot * turbovec_q2_scale_count(static_cast<size_t>(dim)),
            dim) :
        is_turbovec_q4(idx) ?
        dot_turbovec_q4_lut_row(
            turbovec_lut,
            turbovec_lut_scale,
            turbovec_lut_bias,
            turbovec_q4_data_ptr(idx) + slot * turbovec_q4_row_bytes(static_cast<size_t>(dim)),
            idx.turbovec_q4_scale.data() + slot * turbovec_q4_scale_count(static_cast<size_t>(dim)),
            dim) :
        is_q4(idx) ?
        dot_q4(
            score_query,
            q4_data_ptr(idx) + slot * q4_row_bytes(static_cast<size_t>(dim)),
            idx.q4_scale[slot],
            dim,
            max_query) :
        is_q8(idx) ?
        dot_q8(
            score_query,
            q8_data_ptr(idx) + slot * static_cast<size_t>(dim),
            idx.q8_scale[slot],
            dim,
            max_query) :
        dot(
            score_query,
            f32_data_ptr(idx) + slot * static_cast<size_t>(dim),
            dim);
}

void decode_slot_to_f32(const ggml_vec_index_t & idx, size_t slot, float * dst) {
    const int dim = idx.dim;
    if (is_turbovec_q2(idx)) {
        decode_turbovec_q2_row_calibrated(
            turbovec_q2_data_ptr(idx) + slot * turbovec_q2_row_bytes(static_cast<size_t>(dim)),
            idx.turbovec_q2_scale.data() + slot * turbovec_q2_scale_count(static_cast<size_t>(dim)),
            idx.turbovec_tqplus_shift.empty() ? nullptr : idx.turbovec_tqplus_shift.data(),
            idx.turbovec_tqplus_scale.empty() ? nullptr : idx.turbovec_tqplus_scale.data(),
            dst,
            dim);
    } else if (is_turbovec_q4(idx)) {
        decode_turbovec_q4_row_calibrated(
            turbovec_q4_data_ptr(idx) + slot * turbovec_q4_row_bytes(static_cast<size_t>(dim)),
            idx.turbovec_q4_scale.data() + slot * turbovec_q4_scale_count(static_cast<size_t>(dim)),
            idx.turbovec_tqplus_shift.empty() ? nullptr : idx.turbovec_tqplus_shift.data(),
            idx.turbovec_tqplus_scale.empty() ? nullptr : idx.turbovec_tqplus_scale.data(),
            dst,
            dim);
    } else if (is_q4(idx)) {
        const uint8_t * codes =
            q4_data_ptr(idx) + slot * q4_row_bytes(static_cast<size_t>(dim));
        const float scale = idx.q4_scale[slot];
        for (int i = 0; i < dim; ++i) {
            const uint8_t byte = codes[static_cast<size_t>(i) / 2];
            const uint8_t nibble = (i & 1) == 0 ?
                static_cast<uint8_t>(byte & 0x0f) :
                static_cast<uint8_t>(byte >> 4);
            const double value = static_cast<double>(q4_decode(nibble)) * static_cast<double>(scale);
            dst[i] = float_score_from_double(value);
        }
    } else if (is_q8(idx)) {
        const int8_t * codes = q8_data_ptr(idx) + slot * static_cast<size_t>(dim);
        const float scale = idx.q8_scale[slot];
        for (int i = 0; i < dim; ++i) {
            const double value = static_cast<double>(codes[i]) * static_cast<double>(scale);
            dst[i] = float_score_from_double(value);
        }
    } else {
        std::memcpy(
            dst,
            f32_data_ptr(idx) + slot * static_cast<size_t>(dim),
            static_cast<size_t>(dim) * sizeof(float));
    }
}
size_t best_centroid(const float * query, const std::vector<float> & centroids, int n_lists, int dim) {
    size_t best       = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int list = 0; list < n_lists; ++list) {
        const double s = dot_f32_fast(query, centroids.data() + static_cast<size_t>(list) * dim, dim);
        if (s > best_score) {
            best_score = s;
            best = static_cast<size_t>(list);
        }
    }
    return best;
}

// Run a single query against all slots, write top-k into out_scores/out_ids.
// If the index holds fewer than k entries, pad with sentinels.
void search_one(
    const ggml_vec_index_t & idx,
    const float            * query,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids,
    double                   max_query,
    std::vector<ScoreId>   & heap,
    std::vector<ScoreId>   & drained,
    std::vector<float>     & rotated_query_scratch,
    std::vector<float>     & calibrated_query_scratch,
    std::vector<uint8_t>   & turbovec_lut_scratch,
    std::vector<float>     & turbovec_scores_scratch,
    std::vector<uint8_t>   & allowed_blocks_scratch,
    const std::vector<size_t> * allowed_slots = nullptr,
    const float * pre_rotated_turbovec_query = nullptr) {

    const size_t n_slots = idx.slot_to_id.size();

    test_maybe_throw_bad_alloc();
    heap.clear();
    drained.clear();
    rotated_query_scratch.clear();
    calibrated_query_scratch.clear();
    turbovec_lut_scratch.clear();
    turbovec_scores_scratch.clear();
    allowed_blocks_scratch.clear();
    const size_t candidate_hint = allowed_slots != nullptr ?
        std::min(allowed_slots->size(), n_slots) :
        active_count(idx);
    const size_t heap_capacity =
        std::min(static_cast<size_t>(k), candidate_hint);
    heap.reserve(heap_capacity);
    const float * score_query = query;
    std::vector<float> & rotated_query = rotated_query_scratch;
    std::vector<float> & calibrated_query = calibrated_query_scratch;
    std::vector<uint8_t> & turbovec_lut = turbovec_lut_scratch;
    std::vector<float> & turbovec_scores = turbovec_scores_scratch;
    float turbovec_lut_scale = 1.0f;
    float turbovec_lut_bias = 0.0f;
    if (is_turbovec_q2(idx) || is_turbovec_q4(idx)) {
        if (pre_rotated_turbovec_query != nullptr) {
            score_query = pre_rotated_turbovec_query;
        } else {
            rotated_query.resize(static_cast<size_t>(idx.dim));
            rotate_turbovec_query(query, rotated_query.data(), idx.dim);
            score_query = rotated_query.data();
        }
        if (!idx.turbovec_tqplus_shift.empty()) {
            calibrated_query.resize(static_cast<size_t>(idx.dim));
            double bias_correction = 0.0;
            for (int coordinate = 0; coordinate < idx.dim; ++coordinate) {
                const size_t i = static_cast<size_t>(coordinate);
                calibrated_query[i] =
                    score_query[i] / idx.turbovec_tqplus_scale[i];
                bias_correction -=
                    static_cast<double>(score_query[i]) *
                    static_cast<double>(idx.turbovec_tqplus_shift[i]);
            }
            score_query = calibrated_query.data();
            turbovec_lut_bias = static_cast<float>(bias_correction);
        }
        const float tqplus_bias = turbovec_lut_bias;
        if (is_turbovec_q2(idx)) {
            build_turbovec_q2_lut(score_query, idx.dim, turbovec_lut, turbovec_lut_scale, turbovec_lut_bias);
        } else {
            build_turbovec_q4_lut(score_query, idx.dim, turbovec_lut, turbovec_lut_scale, turbovec_lut_bias);
        }
        turbovec_lut_bias += tqplus_bias;

        const int bits = is_turbovec_q2(idx) ? 2 : 4;
        const size_t n_byte_groups =
            static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / bits);
        const size_t expected_blocked_bytes =
            idx.turbovec_blocked_n_blocks * n_byte_groups * 32;
        if (idx.turbovec_blocked_data.size() == expected_blocked_bytes &&
            idx.turbovec_blocked_n_blocks == (n_slots + 31) / 32) {
            std::vector<uint8_t> & allowed_blocks = allowed_blocks_scratch;
            if (allowed_slots != nullptr || active_count(idx) != n_slots) {
                allowed_blocks.assign(idx.turbovec_blocked_n_blocks, 0);
                if (allowed_slots != nullptr) {
                    for (size_t slot : *allowed_slots) {
                        if (slot < n_slots && slot_is_active(idx, slot)) {
                            allowed_blocks[slot / 32] = 1;
                        }
                    }
                } else {
                    for (size_t slot = 0; slot < n_slots; ++slot) {
                        if (slot_is_active(idx, slot)) {
                            allowed_blocks[slot / 32] = 1;
                        }
                    }
                }
            }
            turbovec_scores.assign(n_slots, -std::numeric_limits<float>::infinity());

            const float * vector_scales = is_turbovec_q2(idx) ?
                idx.turbovec_q2_scale.data() :
                idx.turbovec_q4_scale.data();
            std::array<float, 32> block_scores{};
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && !GGML_VEC_INDEX_USE_NEON && GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT
            static const bool has_turbovec_avx2 = cpu_has_avx2();
#endif
            for (size_t block = 0; block < idx.turbovec_blocked_n_blocks; ++block) {
                if (!allowed_blocks.empty() && allowed_blocks[block] == 0) {
                    continue;
                }
#ifdef GGML_VEC_INDEX_TEST_HOOKS
                g_turbovec_block_score_calls.fetch_add(1, std::memory_order_relaxed);
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && !GGML_VEC_INDEX_USE_NEON && GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT
                if (has_turbovec_avx2) {
                    ggml_vec_index_detail::score_turbovec_lut_block_avx2(
                        turbovec_lut.data(),
                        turbovec_lut_scale,
                        turbovec_lut_bias,
                        idx.turbovec_blocked_data.data(),
                        vector_scales,
                        block,
                        n_byte_groups,
                        n_slots,
                        block_scores.data());
                } else
#endif
                {
                    score_turbovec_lut_block(
                        turbovec_lut.data(),
                        turbovec_lut_scale,
                        turbovec_lut_bias,
                        idx.turbovec_blocked_data.data(),
                        vector_scales,
                        block,
                        n_slots,
                        bits,
                        idx.dim,
                        block_scores.data());
                }
                const size_t base_slot = block * 32;
                const size_t count = std::min(static_cast<size_t>(32), n_slots - base_slot);
                for (size_t lane = 0; lane < count; ++lane) {
                    float score = block_scores[lane];
                    if (!std::isfinite(score)) {
                        score = score_slot(
                            idx,
                            score_query,
                            base_slot + lane,
                            max_query,
                            turbovec_lut.data(),
                            turbovec_lut_scale,
                            turbovec_lut_bias);
                    }
                    turbovec_scores[base_slot + lane] = score;
                }
            }
        }
    }

    auto visit_slot = [&](size_t slot) {
        if (!slot_is_active(idx, slot)) {
            return;
        }
        const double score = turbovec_scores.empty() ?
            score_slot(
                idx,
                score_query,
                slot,
                max_query,
                turbovec_lut.data(),
                turbovec_lut_scale,
                turbovec_lut_bias) :
            turbovec_scores[slot];
        const ScoreId candidate{
            score,
            idx.slot_to_id[slot]
        };
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push_back(candidate);
            std::push_heap(heap.begin(), heap.end(), MinHeapCmp());
        } else if (score_id_better(candidate, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), MinHeapCmp());
            heap.back() = candidate;
            std::push_heap(heap.begin(), heap.end(), MinHeapCmp());
        }
    };

    if (allowed_slots != nullptr) {
        for (size_t slot : *allowed_slots) {
            if (slot < n_slots) {
                visit_slot(slot);
            }
        }
    } else if (active_count(idx) < n_slots / 2) {
        for (const auto & entry : idx.id_to_slot) {
            visit_slot(entry.second);
        }
    } else {
        for (size_t slot = 0; slot < n_slots; ++slot) {
            visit_slot(slot);
        }
    }

    // Drain the heap into a temporary descending list.
    drained.reserve(heap.size());
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), MinHeapCmp());
        drained.push_back(heap.back());
        heap.pop_back();
    }
    std::reverse(drained.begin(), drained.end()); // now descending by score

    for (int i = 0; i < k; ++i) {
        if (static_cast<size_t>(i) < drained.size()) {
            out_scores[i] = float_score_from_double(drained[i].score);
            out_ids[i]    = drained[i].id;
        } else {
            out_scores[i] = -FLT_MAX;
            out_ids[i]    = UINT64_MAX;
        }
    }
}

std::vector<size_t> allowed_slots_for_ids(
    const ggml_vec_index_t & idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed) {
    std::vector<size_t> slots;
    slots.reserve(static_cast<size_t>(n_allowed));
    for (int i = 0; i < n_allowed; ++i) {
        const auto it = idx.id_to_slot.find(allowed_ids[i]);
        if (it != idx.id_to_slot.end() && slot_is_active(idx, it->second)) {
            slots.push_back(it->second);
        }
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    return slots;
}

} // namespace

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void turbovec_reset_block_score_call_count_for_test(void) {
    g_turbovec_block_score_calls.store(0, std::memory_order_relaxed);
}

int64_t turbovec_block_score_call_count_for_test(void) {
    return g_turbovec_block_score_calls.load(std::memory_order_relaxed);
}

int turbovec_avx2_available_for_test() {
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && !GGML_VEC_INDEX_USE_NEON && GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT
    return cpu_has_avx2() ? 1 : 0;
#else
    return 0;
#endif
}

int turbovec_avx2_lut_block_matches_scalar_for_test(int bits, int dim) {
    if ((bits != 2 && bits != 4) || dim <= 0 || dim % (8 / bits) != 0) {
        return 0;
    }
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && !GGML_VEC_INDEX_USE_NEON && GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT
    if (!cpu_has_avx2()) {
        return -1;
    }
    constexpr size_t block_size = 32;
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    constexpr size_t n_vectors = block_size + 17;
    constexpr size_t n_blocks = 2;
    std::vector<uint8_t> lut(n_byte_groups * block_size);
    std::vector<uint8_t> blocked_codes(n_blocks * n_byte_groups * block_size);
    std::vector<float> vector_scales(n_vectors);
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = static_cast<uint8_t>((i * 19 + static_cast<size_t>(bits) * 7) & 0x7f);
    }
    for (size_t i = 0; i < blocked_codes.size(); ++i) {
        blocked_codes[i] = static_cast<uint8_t>((i * 23 + static_cast<size_t>(dim) * 3) & 0xff);
    }
    for (size_t i = 0; i < vector_scales.size(); ++i) {
        vector_scales[i] = 0.5f + 0.003f * static_cast<float>((i * 11) % 97);
    }

    constexpr float lut_scale = 0.03125f;
    constexpr float lut_bias = -1.25f;
    std::array<float, block_size> scalar_scores{};
    std::array<float, block_size> avx2_scores{};
    for (size_t block = 0; block < n_blocks; ++block) {
        score_turbovec_lut_block(
            lut.data(),
            lut_scale,
            lut_bias,
            blocked_codes.data(),
            vector_scales.data(),
            block,
            n_vectors,
            bits,
            dim,
            scalar_scores.data());
        ggml_vec_index_detail::score_turbovec_lut_block_avx2(
            lut.data(),
            lut_scale,
            lut_bias,
            blocked_codes.data(),
            vector_scales.data(),
            block,
            n_byte_groups,
            n_vectors,
            avx2_scores.data());
        for (size_t lane = 0; lane < block_size; ++lane) {
            const float scalar_score = scalar_scores[lane];
            const float avx2_score = avx2_scores[lane];
            if (std::isfinite(scalar_score) != std::isfinite(avx2_score)) {
                return 0;
            }
            const float tolerance = std::max(
                1e-4f * std::fabs(scalar_score),
                1e-4f);
            if (std::isfinite(scalar_score) &&
                std::fabs(scalar_score - avx2_score) > tolerance) {
                return 0;
            }
        }
    }
    return 1;
#else
    (void) bits;
    (void) dim;
    return -1;
#endif
}
#endif

static int ggml_vec_index_build_ivf_unlocked(ggml_vec_index_t * idx, int n_lists, int n_iter) {
    try {
        if (idx == nullptr || n_lists <= 0 || n_iter < 0) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if ((is_turbovec_q2(*idx) && !turbovec_q2_supported_dim(idx->dim)) ||
            (is_turbovec_q4(*idx) && !turbovec_q4_supported_dim(idx->dim))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const size_t n_slots = idx->slot_to_id.size();
        const size_t n_live = active_count(*idx);
        const int dim = idx->dim;
        if (n_live == 0) {
            invalidate_ivf(*idx);
            idx->ivf_generation = idx->generation;
            return GGML_VEC_INDEX_OK;
        }

        const int actual_lists = static_cast<int>(
            std::min(static_cast<size_t>(n_lists), n_live));
        const size_t dim_sz = static_cast<size_t>(dim);
        if (dim_sz != 0 &&
            static_cast<size_t>(actual_lists) > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        test_maybe_throw_bad_alloc();

        std::vector<float> centroids(static_cast<size_t>(actual_lists) * dim_sz);
        std::vector<double> next_centroids(centroids.size());
        std::vector<int> counts(static_cast<size_t>(actual_lists));
        std::vector<float> row(dim_sz);
        std::vector<std::vector<size_t>> lists(static_cast<size_t>(actual_lists));
        std::vector<size_t> active_slots;
        active_slots.reserve(n_live);
        for (size_t slot = 0; slot < n_slots; ++slot) {
            if (slot_is_active(*idx, slot)) {
                active_slots.push_back(slot);
            }
        }

        for (int list = 0; list < actual_lists; ++list) {
            const size_t slot = active_slots[static_cast<size_t>(list) * active_slots.size() /
                static_cast<size_t>(actual_lists)];
            float * centroid = centroids.data() + static_cast<size_t>(list) * dim_sz;
            decode_slot_to_f32(*idx, slot, centroid);
        }

        for (int iter = 0; iter < n_iter; ++iter) {
            std::fill(next_centroids.begin(), next_centroids.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t slot : active_slots) {
                decode_slot_to_f32(*idx, slot, row.data());
                const size_t list = best_centroid(row.data(), centroids, actual_lists, dim);
                double * dst = next_centroids.data() + list * dim_sz;
                for (int i = 0; i < dim; ++i) {
                    dst[i] += static_cast<double>(row[static_cast<size_t>(i)]);
                }
                ++counts[list];
            }

            for (int list = 0; list < actual_lists; ++list) {
                float * centroid = centroids.data() + static_cast<size_t>(list) * dim_sz;
                if (counts[static_cast<size_t>(list)] == 0) {
                    continue;
                }
                const double inv_count = 1.0 /
                    static_cast<double>(counts[static_cast<size_t>(list)]);
                const double * src =
                    next_centroids.data() + static_cast<size_t>(list) * dim_sz;
                for (int i = 0; i < dim; ++i) {
                    centroid[i] = static_cast<float>(
                        src[static_cast<size_t>(i)] * inv_count);
                }
            }
        }

        for (size_t slot : active_slots) {
            decode_slot_to_f32(*idx, slot, row.data());
            const size_t list = best_centroid(row.data(), centroids, actual_lists, dim);
            lists[list].push_back(slot);
        }

        idx->ivf_centroids = std::move(centroids);
        idx->ivf_lists = std::move(lists);
        idx->ivf_n_lists = actual_lists;
        idx->ivf_generation = idx->generation;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}
int ggml_vec_index_build_ivf(ggml_vec_index_t * idx, int n_lists, int n_iter) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_build_ivf_unlocked(idx, n_lists, n_iter);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static int ggml_vec_index_search_impl(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    bool                     filtered,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    const ggml_vec_index_filter_t * prepared_filter,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0 ||
        (filtered && prepared_filter == nullptr &&
         (n_allowed < 0 || (n_allowed > 0 && allowed_ids == nullptr)))) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }
    if (queries == nullptr || out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const int dim = idx->dim;
        if ((is_turbovec_q2(*idx) && !turbovec_q2_supported_dim(dim)) ||
            (is_turbovec_q4(*idx) && !turbovec_q4_supported_dim(dim))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz   = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        if (!search_buffers_addressable(n_q_sz, k_sz, dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::vector<double> query_max_abs_values;
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            const size_t value_count = n_q_sz * dim_sz;
            if (!all_finite_abs_less_than(queries, value_count, kTurboVecMaxInputMagnitude)) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
        } else if (!validate_queries_and_maybe_max_abs(
                queries, n_q, dim, is_quantized(*idx), query_max_abs_values)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        std::vector<size_t> allowed_slots;
        const std::vector<size_t> * allowed_ptr = nullptr;
        if (prepared_filter != nullptr) {
            if (prepared_filter->owner != idx ||
                prepared_filter->owner_cookie != idx->filter_cookie ||
                prepared_filter->dim != idx->dim ||
                prepared_filter->bit_width != idx->bit_width ||
                prepared_filter->generation != idx->generation) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            allowed_ptr = &prepared_filter->slots;
        } else if (filtered) {
            allowed_slots = allowed_slots_for_ids(*idx, allowed_ids, n_allowed);
            allowed_ptr = &allowed_slots;
        }

        std::vector<ScoreId> heap;
        std::vector<ScoreId> drained;
        std::vector<float> rotated_query_scratch;
        std::vector<float> calibrated_query_scratch;
        std::vector<uint8_t> turbovec_lut_scratch;
        std::vector<float> turbovec_scores_scratch;
        std::vector<uint8_t> allowed_blocks_scratch;
        std::vector<float> rotated_turbovec_queries;
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            rotated_turbovec_queries.resize(n_q_sz * dim_sz);
            rotate_turbovec_queries(
                queries,
                rotated_turbovec_queries.data(),
                n_q,
                dim);
        }
        for (int q = 0; q < n_q; ++q) {
            const double max_query = query_max_abs_values.empty() ?
                0.0 : query_max_abs_values[static_cast<size_t>(q)];
            search_one(
                *idx,
                queries + static_cast<size_t>(q) * static_cast<size_t>(dim),
                k,
                out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
                out_ids    + static_cast<size_t>(q) * static_cast<size_t>(k),
                max_query,
                heap,
                drained,
                rotated_query_scratch,
                calibrated_query_scratch,
                turbovec_lut_scratch,
                turbovec_scores_scratch,
                allowed_blocks_scratch,
                allowed_ptr,
                rotated_turbovec_queries.empty() ?
                    nullptr :
                    rotated_turbovec_queries.data() + static_cast<size_t>(q) * dim_sz);
        }
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, false, nullptr, 0, nullptr, out_scores, out_ids);
}

int ggml_vec_index_search_filtered(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    float                  * out_scores,
    uint64_t               * out_ids) {
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, true, allowed_ids, n_allowed, nullptr, out_scores, out_ids);
}

ggml_vec_index_filter_t * ggml_vec_index_filter_create(
    const ggml_vec_index_t * idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed) {
    try {
        if (idx == nullptr || n_allowed < 0 ||
            (n_allowed > 0 && allowed_ids == nullptr)) {
            return nullptr;
        }
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        auto * filter = new (std::nothrow) ggml_vec_index_filter();
        if (filter == nullptr) {
            return nullptr;
        }
        std::unique_ptr<ggml_vec_index_filter> owned(filter);
        owned->owner = idx;
        owned->owner_cookie = idx->filter_cookie;
        owned->dim = idx->dim;
        owned->bit_width = idx->bit_width;
        owned->generation = idx->generation;
        owned->slots = allowed_slots_for_ids(*idx, allowed_ids, n_allowed);
        return owned.release();
    } catch (...) {
        return nullptr;
    }
}

void ggml_vec_index_filter_free(ggml_vec_index_filter_t * filter) {
    delete filter;
}

int ggml_vec_index_search_prepared_filtered(
    const ggml_vec_index_t        * idx,
    const ggml_vec_index_filter_t * filter,
    const float                   * queries,
    int                             n_q,
    int                             k,
    float                         * out_scores,
    uint64_t                      * out_ids) {
    if (filter == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, true, nullptr, 0, filter, out_scores, out_ids);
}

int ggml_vec_index_search_ivf(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    int                      nprobe,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0 || nprobe <= 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const int dim = idx->dim;
        if ((is_turbovec_q2(*idx) && !turbovec_q2_supported_dim(dim)) ||
            (is_turbovec_q4(*idx) && !turbovec_q4_supported_dim(dim))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        if (idx->ivf_generation != idx->generation ||
            idx->ivf_n_lists < 0 ||
            static_cast<size_t>(idx->ivf_n_lists) != idx->ivf_lists.size() ||
            idx->ivf_centroids.size() != static_cast<size_t>(idx->ivf_n_lists) * dim_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n_q == 0) {
            return GGML_VEC_INDEX_OK;
        }
        if (queries == nullptr || out_scores == nullptr || out_ids == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!search_buffers_addressable(n_q_sz, k_sz, dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::vector<double> query_max_abs_values;
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            const size_t value_count = n_q_sz * dim_sz;
            if (!all_finite_abs_less_than(queries, value_count, kTurboVecMaxInputMagnitude)) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
        } else if (!validate_queries_and_maybe_max_abs(
                queries, n_q, dim, is_quantized(*idx), query_max_abs_values)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const int probe_count = std::min(nprobe, idx->ivf_n_lists);
        std::vector<ScoreId> centroid_scores;
        std::vector<size_t> selected_lists;
        std::vector<size_t> candidate_slots;
        std::vector<ScoreId> heap;
        std::vector<ScoreId> drained;
        std::vector<float> rotated_query_scratch;
        std::vector<float> calibrated_query_scratch;
        std::vector<uint8_t> turbovec_lut_scratch;
        std::vector<float> turbovec_scores_scratch;
        std::vector<uint8_t> allowed_blocks_scratch;
        std::vector<float> rotated_turbovec_queries;
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            rotated_turbovec_queries.resize(n_q_sz * dim_sz);
            rotate_turbovec_queries(
                queries,
                rotated_turbovec_queries.data(),
                n_q,
                dim);
        }
        centroid_scores.reserve(static_cast<size_t>(std::max(idx->ivf_n_lists, 0)));
        selected_lists.reserve(static_cast<size_t>(probe_count));
        for (int q = 0; q < n_q; ++q) {
            const float * query = queries + static_cast<size_t>(q) * dim_sz;
            float * scores = out_scores + static_cast<size_t>(q) * k_sz;
            uint64_t * ids = out_ids + static_cast<size_t>(q) * k_sz;
            const double max_query = query_max_abs_values.empty() ?
                0.0 : query_max_abs_values[static_cast<size_t>(q)];
            const float * pre_rotated_turbovec_query = rotated_turbovec_queries.empty() ?
                nullptr : rotated_turbovec_queries.data() + static_cast<size_t>(q) * dim_sz;

            if (idx->ivf_n_lists == 0) {
                const std::vector<size_t> empty_slots;
                search_one(
                    *idx,
                    query,
                    k,
                    scores,
                    ids,
                    max_query,
                    heap,
                    drained,
                    rotated_query_scratch,
                    calibrated_query_scratch,
                    turbovec_lut_scratch,
                    turbovec_scores_scratch,
                    allowed_blocks_scratch,
                    &empty_slots,
                    pre_rotated_turbovec_query);
                continue;
            }

            centroid_scores.clear();
            for (int list = 0; list < idx->ivf_n_lists; ++list) {
                if (idx->ivf_lists[static_cast<size_t>(list)].empty()) {
                    continue;
                }
                const double score =
                    dot_f32_fast(query, idx->ivf_centroids.data() + static_cast<size_t>(list) * dim_sz, dim);
                centroid_scores.push_back({ score, static_cast<uint64_t>(list) });
            }
            std::sort(
                centroid_scores.begin(),
                centroid_scores.end(),
                [](const ScoreId & a, const ScoreId & b) {
                    return score_id_better(a, b);
                });

            selected_lists.clear();
            size_t candidate_count = 0;
            for (const ScoreId & centroid : centroid_scores) {
                const size_t list_id = static_cast<size_t>(centroid.id);
                const auto & list = idx->ivf_lists[list_id];
                if (list.empty()) {
                    continue;
                }
                selected_lists.push_back(list_id);
                candidate_count += list.size();
                if (selected_lists.size() == static_cast<size_t>(probe_count)) {
                    break;
                }
            }
            candidate_slots.clear();
            candidate_slots.reserve(candidate_count);
            for (size_t list_id : selected_lists) {
                const auto & list = idx->ivf_lists[list_id];
                candidate_slots.insert(candidate_slots.end(), list.begin(), list.end());
            }
            search_one(
                *idx,
                query,
                k,
                scores,
                ids,
                max_query,
                heap,
                drained,
                rotated_query_scratch,
                calibrated_query_scratch,
                turbovec_lut_scratch,
                turbovec_scores_scratch,
                allowed_blocks_scratch,
                &candidate_slots,
                pre_rotated_turbovec_query);
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}
