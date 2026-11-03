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

// Upstream fuses TurboVec queries into batches on aarch64 and x86.
#if GGML_VEC_INDEX_USE_NEON || defined(__x86_64__) || defined(_M_X64)
#define GGML_VEC_INDEX_TURBOVEC_BATCHES 1
#else
#define GGML_VEC_INDEX_TURBOVEC_BATCHES 0
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

// Upstream turbovec scores 4-bit codes with its permute-dot kernels on
// hosts that run them (turbovec_cpu_vector_major) and with the nibble LUT
// elsewhere; the two round differently, so this follows the same split.
bool turbovec_uses_permute_dot(const ggml_vec_index_t & idx) {
    // Decided with the blocked layout when the index was created; on x86
    // the 2- and 3-bit vector-major layouts take the `vpermb` scan instead.
    return idx.turbovec_blocked_vm && idx.turbovec_bits == 4;
}

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
// Batches TurboVec queries whenever upstream would, past the scratch bound.
static std::atomic<bool> g_turbovec_force_batches{ false };
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
    uint8_t turbovec_row[kTurboVecMaxDim / 2];
    if (turbovec_rows_in_blocked(idx)) {
        turbovec_row_packed(idx, slot, turbovec_row);
    }
    return is_turbovec(idx) ?
        dot_turbovec_bits_lut_row(
            turbovec_lut,
            turbovec_lut_scale,
            turbovec_lut_bias,
            turbovec_rows_in_blocked(idx) ?
                turbovec_row :
                idx.turbovec_data.data() + slot * turbovec_row_bytes(idx.turbovec_bits, static_cast<size_t>(dim)),
            idx.turbovec_scale.data() + slot,
            idx.turbovec_bits,
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
    if (is_turbovec(idx)) {
        uint8_t row[kTurboVecMaxDim / 2];
        decode_turbovec_row_calibrated(
            idx.turbovec_bits,
            idx.turbovec_legacy,
            turbovec_row_packed_ptr(idx, slot, row),
            idx.turbovec_scale.data() + slot,
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

void write_topk_heap(
        const ggml_vec_index_t & idx,
        int k,
        float * out_scores,
        uint64_t * out_ids,
        std::vector<ScoreId> & heap) {
    std::sort(heap.begin(), heap.end(), [](const ScoreId & a, const ScoreId & b) {
        return score_id_better(a, b);
    });

    for (int i = 0; i < k; ++i) {
        if (static_cast<size_t>(i) < heap.size()) {
            out_scores[i] = float_score_from_double(heap[static_cast<size_t>(i)].score);
            out_ids[i] = topk_key_id(idx, heap[static_cast<size_t>(i)].id);
        } else {
            out_scores[i] = -FLT_MAX;
            out_ids[i] = UINT64_MAX;
        }
    }
}

template <typename ScoreFn>
void write_topk_results(
        const ggml_vec_index_t & idx,
        int k,
        float * out_scores,
        uint64_t * out_ids,
        std::vector<ScoreId> & heap,
        const std::vector<size_t> * allowed_slots,
        ScoreFn score_for_slot) {
    const size_t n_slots = idx.slot_to_id.size();
    auto visit_slot = [&](size_t slot) {
        if (slot_is_active(idx, slot)) {
            update_topk_heap(
                heap,
                k,
                { score_for_slot(slot), topk_key(idx, slot) });
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

    write_topk_heap(idx, k, out_scores, out_ids, heap);
}

// One query's nibble-LUT scores for a block, in the index's layout. `split`
// (the query's `vpermb` tables) may be null.
void score_turbovec_lut_block_for(
        const ggml_vec_index_t & idx,
        const uint8_t * lut,
        const uint8_t * split,
        float lut_scale,
        float lut_bias,
        size_t block,
        float * out_scores) {
    const size_t n_slots = idx.slot_to_id.size();
    if (turbovec_scores_vnni_lut(idx)) {
        score_turbovec_lut_block_vnni(
            lut, split, lut_scale, lut_bias, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), block,
            n_slots, idx.turbovec_bits, idx.dim, out_scores);
        return;
    }
    if (idx.turbovec_blocked_vm8) {
        score_turbovec_lut_block_vm8(
            lut, lut_scale, lut_bias, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), block, n_slots,
            idx.turbovec_bits, idx.dim, out_scores);
        return;
    }
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && !GGML_VEC_INDEX_USE_NEON && GGML_VEC_INDEX_TURBOVEC_AVX2_LAYOUT
    static const bool has_turbovec_avx2 = cpu_has_avx2();
    if (has_turbovec_avx2) {
        const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
        ggml_vec_index_detail::score_turbovec_lut_block_avx2(
            lut, lut_scale, lut_bias, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), block,
            n_byte_groups, n_slots, out_scores);
        return;
    }
#endif
    score_turbovec_lut_block(
        lut, lut_scale, lut_bias, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), block, n_slots,
        idx.turbovec_bits, idx.dim, out_scores);
}

// Whether batches score with permute-dot operands (upstream `pd_batched`):
// 4-bit permute-dot indexes, and 2-bit indexes in the vm8 layout.
bool turbovec_pd_batched(const ggml_vec_index_t & idx) {
    return turbovec_uses_permute_dot(idx) || (idx.turbovec_bits == 2 && idx.turbovec_blocked_vm8);
}

struct TurboVecQueryLut {
    const uint8_t * lut;
    float scale;
    float bias;
    const float * score_query;
    // Set when 4-bit codes score with permute-dot arithmetic instead.
    const TurboVecPermuteDot * permute_dot = nullptr;
    // The `vpermb` tables, when the codes take that scan.
    const uint8_t * split = nullptr;
};

// Blocks one pass of a lone query scores together: upstream's x86 kernels
// interleave eight (permute-dot) or two (`vpermb` scan) block streams.
size_t turbovec_query_group(const ggml_vec_index_t & idx, const TurboVecQueryLut & query) {
    if (query.permute_dot != nullptr && turbovec_layout(idx) == kTurboVecLayoutVm && idx.turbovec_x86) {
        return 8;
    }
    return turbovec_scores_vnni_lut(idx) ? 2 : 1;
}

// One query's scores for blocks [first, first + n_group), out[b * 32 + lane].
void score_turbovec_query_blocks(
        const ggml_vec_index_t & idx,
        const TurboVecQueryLut & query,
        size_t first,
        size_t n_group,
        float * out_scores) {
    const size_t n_slots = idx.slot_to_id.size();
    if (query.permute_dot != nullptr) {
        score_turbovec_permute_dot_blocks(
            *query.permute_dot, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), first, n_group, n_slots,
            idx.dim, turbovec_layout(idx), out_scores);
        return;
    }
    if (turbovec_scores_vnni_lut(idx)) {
        score_turbovec_lut_blocks_vnni(
            &query.lut, query.split != nullptr ? &query.split : nullptr, &query.scale, &query.bias, 1,
            idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), first, n_group, n_slots, idx.turbovec_bits,
            idx.dim, &out_scores);
        return;
    }
    for (size_t b = 0; b < n_group; ++b) {
        score_turbovec_lut_block_for(
            idx, query.lut, query.split, query.scale, query.bias, first + b, out_scores + b * 32);
    }
}



// Scores blocks [block_begin, block_end) straight into the top-k heap,
// skipping tombstones, blocks with no allowed slot and disallowed slots.
void score_turbovec_block_range(
        const ggml_vec_index_t & idx,
        const TurboVecQueryLut & query,
        size_t block_begin,
        size_t block_end,
        const uint8_t * allowed_blocks,
        const uint8_t * allowed_mask,
        int k,
        std::vector<ScoreId> & heap) {
    const size_t n_slots = idx.slot_to_id.size();
    const size_t group = turbovec_query_group(idx, query);
    std::array<float, 8 * 32> block_scores{};
    for (size_t first = block_begin; first < block_end; first += group) {
        const size_t n_group = std::min(group, block_end - first);
        // A group is skipped only when none of its blocks has an allowed
        // slot (upstream's fix for its interleaved kernels, which dropped
        // allowed vectors behind a fully masked first block).
        bool any_allowed = allowed_blocks == nullptr;
        for (size_t b = 0; b < n_group && !any_allowed; ++b) {
            any_allowed = allowed_blocks[first + b] != 0;
        }
        if (!any_allowed) {
            continue;
        }
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        g_turbovec_block_score_calls.fetch_add(static_cast<int64_t>(n_group), std::memory_order_relaxed);
#endif
        score_turbovec_query_blocks(idx, query, first, n_group, block_scores.data());
        for (size_t b = 0; b < n_group; ++b) {
            const size_t block = first + b;
            if (allowed_blocks != nullptr && allowed_blocks[block] == 0) {
                continue;
            }
            const size_t base_slot = block * 32;
            const size_t count = std::min(static_cast<size_t>(32), n_slots - base_slot);
            for (size_t lane = 0; lane < count; ++lane) {
                const size_t slot = base_slot + lane;
                if (!slot_is_active(idx, slot) || (allowed_mask != nullptr && allowed_mask[slot] == 0)) {
                    continue;
                }
                float score = block_scores[b * 32 + lane];
                if (!std::isfinite(score) && query.permute_dot == nullptr) {
                    score = score_slot(idx, query.score_query, slot, 0.0, query.lut, query.scale, query.bias);
                }
                update_topk_heap(heap, k, { score, topk_key(idx, slot) });
            }
        }
    }
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
    std::vector<float>     & rotated_query_scratch,
    std::vector<float>     & calibrated_query_scratch,
    std::vector<uint8_t>   & turbovec_lut_scratch,
    std::vector<uint8_t>   & allowed_blocks_scratch,
    const std::vector<size_t> * allowed_slots = nullptr,
    const float * pre_rotated_turbovec_query = nullptr,
    int n_threads = 1,
    bool single_query = false,
    bool * staged_seed_short = nullptr,
    bool staged_sign_single = false) {

    const size_t n_slots = idx.slot_to_id.size();
    if (staged_seed_short != nullptr) {
        *staged_seed_short = false;
    }

    test_maybe_throw_bad_alloc();
    heap.clear();
    rotated_query_scratch.clear();
    calibrated_query_scratch.clear();
    turbovec_lut_scratch.clear();
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
    float turbovec_lut_scale = 1.0f;
    float turbovec_lut_bias = 0.0f;
    if (is_turbovec(idx)) {
        if (pre_rotated_turbovec_query != nullptr) {
            score_query = pre_rotated_turbovec_query;
        } else {
            rotated_query.resize(static_cast<size_t>(idx.dim));
            rotate_turbovec_query(idx.turbovec_legacy, query, rotated_query.data(), idx.dim);
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
        if (idx.turbovec_planes && idx.turbovec_staged_enabled) {
            std::vector<uint8_t> staged_mask;
            if (allowed_slots != nullptr) {
                staged_mask.assign(n_slots, 0);
                for (size_t slot : *allowed_slots) {
                    if (slot < n_slots) {
                        staged_mask[slot] = 1;
                    }
                }
            }
            TurboVecStagedOpts opts;
            opts.single_query = single_query;
            opts.sign_single = staged_sign_single;
            opts.seed_short = staged_seed_short;
            opts.n_threads = n_threads;
            if (turbovec_staged_search(
                    idx, score_query, tqplus_bias, k, staged_mask.empty() ? nullptr : staged_mask.data(), opts,
                    heap)) {
                write_topk_heap(idx, k, out_scores, out_ids, heap);
                return;
            }
        }
        build_turbovec_lut(
            idx.turbovec_bits, idx.turbovec_legacy, score_query, idx.dim, turbovec_lut, turbovec_lut_scale, turbovec_lut_bias);
        turbovec_lut_bias += tqplus_bias;
        // The `vpermb` scan's tables, built once per query.
        std::vector<uint8_t> split_lut;
        if (turbovec_scores_vnni_lut(idx)) {
            split_lut.resize(turbovec_lut.size());
            turbovec_split_lut_for_vnni(turbovec_lut.data(), turbovec_lut.size() / 32, split_lut.data());
        }
        TurboVecPermuteDot permute_dot;
        const bool use_permute_dot = turbovec_uses_permute_dot(idx);
        if (use_permute_dot) {
            build_turbovec_permute_dot(score_query, idx.dim, permute_dot, turbovec_layout(idx) == kTurboVecLayoutVm8);
            permute_dot.bias = tqplus_bias;
        }

        const int bits = idx.turbovec_bits;
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
            const uint8_t * allowed_mask = nullptr;
            std::vector<uint8_t> allowed_mask_storage;
            if (allowed_slots != nullptr) {
                allowed_mask_storage.assign(n_slots, 0);
                for (size_t slot : *allowed_slots) {
                    if (slot < n_slots) {
                        allowed_mask_storage[slot] = 1;
                    }
                }
                allowed_mask = allowed_mask_storage.data();
            }
            const TurboVecQueryLut query_lut = {
                turbovec_lut.data(), turbovec_lut_scale, turbovec_lut_bias, score_query,
                use_permute_dot ? &permute_dot : nullptr, split_lut.empty() ? nullptr : split_lut.data(),
            };
            const uint8_t * block_filter = allowed_blocks.empty() ? nullptr : allowed_blocks.data();
            const size_t n_blocks = idx.turbovec_blocked_n_blocks;
            // Split on the blocks a filter actually leaves to score, so a
            // narrow filter over a large index stays on one thread
            // (upstream #554).
            const size_t n_scored = allowed_blocks.empty() ?
                n_blocks :
                static_cast<size_t>(std::count(allowed_blocks.begin(), allowed_blocks.end(), uint8_t{ 1 }));
            if (n_threads > 1 && n_scored >= kTurboVecParallelMinBlocks) {
                // One query over many blocks: score block ranges on each
                // worker into its own heap, then merge. Top-k uses a strict
                // (score, key) order, so the split cannot change the result.
                const size_t n_ranges = static_cast<size_t>(n_threads) * 4;
                const size_t per_range = (n_blocks + n_ranges - 1) / n_ranges;
                std::vector<std::vector<ScoreId>> range_heaps(n_ranges);
                parallel_for(n_ranges, n_threads, [&](size_t range, size_t) {
                    const size_t begin = range * per_range;
                    const size_t end = std::min(n_blocks, begin + per_range);
                    if (begin < end) {
                        score_turbovec_block_range(
                            idx, query_lut, begin, end, block_filter, allowed_mask, k, range_heaps[range]);
                    }
                });
                for (const std::vector<ScoreId> & range_heap : range_heaps) {
                    for (const ScoreId & candidate : range_heap) {
                        update_topk_heap(heap, k, candidate);
                    }
                }
            } else {
                score_turbovec_block_range(idx, query_lut, 0, n_blocks, block_filter, allowed_mask, k, heap);
            }
            write_topk_heap(idx, k, out_scores, out_ids, heap);
            return;
        }
    }

    write_topk_results(
        idx,
        k,
        out_scores,
        out_ids,
        heap,
        allowed_slots,
        [&](size_t slot) -> double {
            return score_slot(
                idx,
                score_query,
                slot,
                max_query,
                turbovec_lut.data(),
                turbovec_lut_scale,
                turbovec_lut_bias);
        });
}

#if GGML_VEC_INDEX_TURBOVEC_BATCHES
struct TurboVecBatchScratch {
    std::array<std::vector<ScoreId>, kTurboVecMaxBatch> heaps;
    std::array<std::vector<float>, kTurboVecMaxBatch> calibrated_queries;
    std::array<std::vector<uint8_t>, kTurboVecMaxBatch> luts;
    std::array<std::vector<uint8_t>, kTurboVecMaxBatch> splits;
    std::array<TurboVecPermuteDot, kTurboVecMaxBatch> permute_dots;
    TurboVecPermuteDotBatch pd_batch;
};

// One fused batch of `n` queries (upstream's pd_scan, the `vpermb` batch,
// or the 4-query LUT scan). `width` is the kernel's query count: with
// permute-dot operands on x86 a short batch is padded to it with its last
// query, as upstream pads, and the padding is scored and dropped.
// `block_filter` and `allowed_mask` restrict a filtered search, as
// upstream's mask does.
void search_turbovec_query_batch(
        const ggml_vec_index_t & idx,
        const float * rotated_queries,
        int n,
        int width,
        int k,
        float * out_scores,
        uint64_t * out_ids,
        const uint8_t * block_filter,
        const uint8_t * allowed_mask,
        TurboVecBatchScratch & scratch,
        int block_threads) {
    const size_t n_slots = idx.slot_to_id.size();
    const size_t dim_sz = static_cast<size_t>(idx.dim);
    const int bits = idx.turbovec_bits;
    const bool pd_batched = turbovec_pd_batched(idx);
    const bool vnni = turbovec_scores_vnni_lut(idx);
    const float * score_queries[kTurboVecMaxBatch] = {};
    const uint8_t * lut_ptrs[kTurboVecMaxBatch] = {};
    const uint8_t * split_ptrs[kTurboVecMaxBatch] = {};
    float lut_scales[kTurboVecMaxBatch] = {};
    float lut_biases[kTurboVecMaxBatch] = {};

    for (int query = 0; query < n; ++query) {
        test_maybe_throw_bad_alloc();
        scratch.heaps[query].clear();
        scratch.calibrated_queries[query].clear();
        scratch.heaps[query].reserve(
            std::min(static_cast<size_t>(k), n_slots));

        score_queries[query] =
            rotated_queries + static_cast<size_t>(query) * dim_sz;
        float tqplus_bias = 0.0f;
        if (!idx.turbovec_tqplus_shift.empty()) {
            std::vector<float> & calibrated = scratch.calibrated_queries[query];
            calibrated.resize(dim_sz);
            double bias_correction = 0.0;
            for (int coordinate = 0; coordinate < idx.dim; ++coordinate) {
                const size_t i = static_cast<size_t>(coordinate);
                calibrated[i] =
                    score_queries[query][i] / idx.turbovec_tqplus_scale[i];
                bias_correction -=
                    static_cast<double>(score_queries[query][i]) *
                    static_cast<double>(idx.turbovec_tqplus_shift[i]);
            }
            score_queries[query] = calibrated.data();
            tqplus_bias = static_cast<float>(bias_correction);
        }

        if (pd_batched) {
            // The permute-dot kernels carry the TQ+ correction in their bias.
            if (bits == 4) {
                build_turbovec_permute_dot(
                    score_queries[query], idx.dim, scratch.permute_dots[query], turbovec_layout(idx) == kTurboVecLayoutVm8);
            } else {
                build_turbovec_permute_dot_2bit(score_queries[query], idx.dim, scratch.permute_dots[query]);
            }
            scratch.permute_dots[query].bias = tqplus_bias;
            continue;
        }
        scratch.luts[query].clear();
        build_turbovec_lut(
            bits,
            idx.turbovec_legacy,
            score_queries[query],
            idx.dim,
            scratch.luts[query],
            lut_scales[query],
            lut_biases[query]);
        lut_biases[query] += tqplus_bias;
        lut_ptrs[query] = scratch.luts[query].data();
        if (vnni) {
            std::vector<uint8_t> & split = scratch.splits[query];
            split.resize(scratch.luts[query].size());
            turbovec_split_lut_for_vnni(lut_ptrs[query], split.size() / 32, split.data());
            split_ptrs[query] = split.data();
        }
    }
    // The kernel's padding lanes repeat the last query.
    const int lanes = pd_batched ? std::max(n, width) : (vnni ? n : 4);
    for (int query = n; query < lanes; ++query) {
        lut_ptrs[query] = lut_ptrs[n - 1];
        lut_scales[query] = lut_scales[n - 1];
        lut_biases[query] = lut_biases[n - 1];
    }
    TurboVecPermuteDotBatch & pd_batch = scratch.pd_batch;
    if (pd_batched) {
        pd_batch.n = lanes;
        pd_batch.bits = bits;
        pd_batch.layout = turbovec_layout(idx);
        for (int query = 0; query < lanes; ++query) {
            pd_batch.pds[query] = &scratch.permute_dots[std::min(query, n - 1)];
        }
        prepare_turbovec_permute_dot_batch(pd_batch, idx.dim);
    }

    const float * vector_scales = idx.turbovec_scale.data();
    // The 4-query LUT scan takes blocks in pairs on x86 (upstream's
    // AVX-512 BW kernel); everything else one block at a time.
#if defined(__x86_64__) || defined(_M_X64)
    const size_t group = !pd_batched && !vnni ? 2 : 1;
#else
    const size_t group = 1;
#endif
    const size_t n_blocks = idx.turbovec_blocked_n_blocks;
    // Scores blocks [block_begin, block_end) into one top-k heap per query.
    auto scan = [&](size_t block_begin, size_t block_end, std::vector<ScoreId> * heaps) {
    std::array<std::array<float, 64>, kTurboVecMaxBatch> block_scores{};
    float * block_score_ptrs[kTurboVecMaxBatch];
    float * second_block_ptrs[kTurboVecMaxBatch];
    for (int query = 0; query < kTurboVecMaxBatch; ++query) {
        block_score_ptrs[query] = block_scores[query].data();
        second_block_ptrs[query] = block_scores[query].data() + 32;
    }
    for (size_t first = block_begin; first < block_end; first += group) {
        const size_t n_group = std::min(group, block_end - first);
        bool any_allowed = block_filter == nullptr;
        for (size_t b = 0; b < n_group && !any_allowed; ++b) {
            any_allowed = block_filter[first + b] != 0;
        }
        if (!any_allowed) {
            continue;
        }
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        g_turbovec_block_score_calls.fetch_add(static_cast<int64_t>(n_group), std::memory_order_relaxed);
#endif
        if (pd_batched) {
            score_turbovec_permute_dot_batch_block(
                pd_batch,
                idx.turbovec_blocked_data.data(),
                vector_scales,
                first,
                n_slots,
                idx.dim,
                block_score_ptrs);
        } else if (vnni) {
            score_turbovec_lut_blocks_vnni(
                lut_ptrs, split_ptrs, lut_scales, lut_biases, n, idx.turbovec_blocked_data.data(), vector_scales, first,
                1, n_slots, bits, idx.dim, block_score_ptrs);
        } else if (n_group == 2) {
            score_turbovec_lut_block_pair_4(
                lut_ptrs, lut_scales, lut_biases, idx.turbovec_blocked_data.data(), vector_scales, first, n_slots, bits,
                idx.dim, block_score_ptrs, second_block_ptrs);
        } else {
            score_turbovec_lut_block_4(
                lut_ptrs,
                lut_scales,
                lut_biases,
                idx.turbovec_blocked_data.data(),
                vector_scales,
                first,
                n_slots,
                bits,
                idx.dim,
                block_score_ptrs);
        }
        for (size_t b = 0; b < n_group; ++b) {
            const size_t block = first + b;
            if (block_filter != nullptr && block_filter[block] == 0) {
                continue;
            }
            const size_t base_slot = block * 32;
            const size_t count = std::min(static_cast<size_t>(32), n_slots - base_slot);
            for (int query = 0; query < n; ++query) {
                for (size_t lane = 0; lane < count; ++lane) {
                    const size_t slot = base_slot + lane;
                    if (allowed_mask != nullptr && allowed_mask[slot] == 0) {
                        continue;
                    }
                    float score = block_scores[query][b * 32 + lane];
                    if (!std::isfinite(score) && !pd_batched) {
                        score = score_slot(
                            idx,
                            score_queries[query],
                            slot,
                            0.0,
                            lut_ptrs[query],
                            lut_scales[query],
                            lut_biases[query]);
                    }
                    update_topk_heap(
                        heaps[query],
                        k,
                        { score, topk_key(idx, slot) });
                }
            }
        }
    }
    };

    // A batch with threads to spare splits its blocks into ranges, each with
    // its own heaps, merged under the strict (score, key) order, so the split
    // cannot change the result (as a lone query's block split).
    size_t n_scored = n_blocks;
    if (block_filter != nullptr) {
        n_scored = static_cast<size_t>(std::count(block_filter, block_filter + n_blocks, uint8_t{ 1 }));
    }
    if (block_threads <= 1 || n_scored < kTurboVecParallelMinBlocks) {
        scan(0, n_blocks, scratch.heaps.data());
    } else {
        const size_t n_ranges = static_cast<size_t>(block_threads) * 4;
        size_t per_range = (n_blocks + n_ranges - 1) / n_ranges;
        per_range = (per_range + group - 1) / group * group;
        std::vector<std::array<std::vector<ScoreId>, kTurboVecMaxBatch>> range_heaps(n_ranges);
        parallel_for(n_ranges, block_threads, [&](size_t range, size_t) {
            const size_t begin = range * per_range;
            const size_t end = std::min(n_blocks, begin + per_range);
            if (begin < end) {
                scan(begin, end, range_heaps[range].data());
            }
        });
        for (int query = 0; query < n; ++query) {
            for (const auto & heaps : range_heaps) {
                for (const ScoreId & candidate : heaps[query]) {
                    update_topk_heap(scratch.heaps[query], k, candidate);
                }
            }
        }
    }

    for (int query = 0; query < n; ++query) {
        write_topk_heap(
            idx,
            k,
            out_scores + static_cast<size_t>(query) * static_cast<size_t>(k),
            out_ids + static_cast<size_t>(query) * static_cast<size_t>(k),
            scratch.heaps[query]);
    }
}
#endif

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

// One block's scores with the kernel the full scan uses on this host.
void score_turbovec_block(
        const ggml_vec_index_t & idx,
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const TurboVecPermuteDot * permute_dot,
        size_t block,
        float * out_scores,
        const uint8_t * split) {
    if (permute_dot != nullptr) {
        score_turbovec_permute_dot_block(
            *permute_dot, idx.turbovec_blocked_data.data(), idx.turbovec_scale.data(), block, idx.slot_to_id.size(),
            idx.dim, turbovec_layout(idx), out_scores);
        return;
    }
    score_turbovec_lut_block_for(idx, lut, split, lut_scale, lut_bias, block, out_scores);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Upstream's x86 `vpermb`-scan search computed the slow way: each allowed
// row's code bytes read back through the layout-independent accessor, its
// two nibble entries per byte group summed exactly, then (sum * scale +
// bias) times the row's scale, into a strict top-k. Shares nothing with
// the blocked scan but the tables, so the search's kernels, batches and
// block groups can be checked against it. `allowed` (n_allowed ids) may be
// null. 0 when the index does not score that way.
int turbovec_vpermb_reference_search_for_test(
        const ggml_vec_index_t * idx,
        const float * queries,
        int n_q,
        int k,
        const uint64_t * allowed,
        int n_allowed,
        float * out_scores,
        uint64_t * out_ids) {
    if (idx == nullptr || !is_turbovec(*idx) || !turbovec_scores_vnni_lut(*idx)) {
        return 0;
    }
    std::shared_lock<std::shared_mutex> lock(idx->mutex);
    const size_t dim = static_cast<size_t>(idx->dim);
    const size_t n_byte_groups = dim / static_cast<size_t>(8 / idx->turbovec_bits);
    std::vector<uint8_t> allow_mask;
    if (allowed != nullptr) {
        allow_mask.assign(idx->slot_to_id.size(), 0);
        for (const size_t slot : allowed_slots_for_ids(*idx, allowed, n_allowed)) {
            allow_mask[slot] = 1;
        }
    }
    std::vector<float> rotated(static_cast<size_t>(n_q) * dim);
    rotate_turbovec_queries(false, queries, rotated.data(), n_q, idx->dim);
    std::vector<uint8_t> seq(n_byte_groups);
    for (int q = 0; q < n_q; ++q) {
        std::vector<float> query(rotated.begin() + static_cast<std::ptrdiff_t>(q * dim),
            rotated.begin() + static_cast<std::ptrdiff_t>((q + 1) * dim));
        float tqplus_bias = 0.0f;
        if (!idx->turbovec_tqplus_shift.empty()) {
            double bias_correction = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                bias_correction -= static_cast<double>(query[i]) * static_cast<double>(idx->turbovec_tqplus_shift[i]);
                query[i] = query[i] / idx->turbovec_tqplus_scale[i];
            }
            tqplus_bias = static_cast<float>(bias_correction);
        }
        std::vector<uint8_t> lut;
        float scale = 1.0f;
        float bias = 0.0f;
        build_turbovec_lut(idx->turbovec_bits, false, query.data(), idx->dim, lut, scale, bias);
        bias += tqplus_bias;
        std::vector<ScoreId> heap;
        for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
            if (!slot_is_active(*idx, slot) || (!allow_mask.empty() && allow_mask[slot] == 0)) {
                continue;
            }
            turbovec_row_seq(*idx, slot, seq.data());
            uint32_t sum = 0;
            for (size_t g = 0; g < n_byte_groups; ++g) {
                sum += static_cast<uint32_t>(lut[g * 32 + (seq[g] >> 4)]) + lut[g * 32 + 16 + (seq[g] & 0x0f)];
            }
            const float product = static_cast<float>(sum) * scale;
            const float score = (product + bias) * idx->turbovec_scale[slot];
            update_topk_heap(heap, k, { score, topk_key(*idx, slot) });
        }
        write_topk_heap(*idx, k, out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
            out_ids + static_cast<size_t>(q) * static_cast<size_t>(k), heap);
    }
    return 1;
}

void turbovec_force_batches_for_test(int enabled) {
    g_turbovec_force_batches.store(enabled != 0, std::memory_order_relaxed);
}

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
        if (is_turbovec(*idx) && !turbovec_supported_dim(idx->dim, idx->turbovec_legacy)) {
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
        if (is_turbovec(*idx) && !turbovec_supported_dim(dim, idx->turbovec_legacy)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz   = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        if (!search_buffers_addressable(n_q_sz, k_sz, dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::vector<double> query_max_abs_values;
        if (is_turbovec(*idx)) {
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

        std::vector<float> rotated_turbovec_queries;
        if (is_turbovec(*idx)) {
            rotated_turbovec_queries.resize(n_q_sz * dim_sz);
            rotate_turbovec_queries(
                idx->turbovec_legacy,
                queries,
                rotated_turbovec_queries.data(),
                n_q,
                dim);
        }
        // Queries covered by the fused TurboVec batch kernels, as upstream
        // chunks them.
        struct Batch {
            size_t first;
            int n;
            int width;
        };
        int n_batched = 0;
        std::vector<Batch> batches;
#if GGML_VEC_INDEX_TURBOVEC_BATCHES
        const int batch_bits = idx->turbovec_bits;
        const size_t batch_n_byte_groups =
            is_turbovec(*idx) ? dim_sz / static_cast<size_t>(8 / batch_bits) : 0;
        const bool pd_batched = is_turbovec(*idx) && turbovec_pd_batched(*idx);
#if GGML_VEC_INDEX_USE_NEON
        // Upstream's aarch64 batch width: 12, 8 or 4 with permute-dot
        // operands (stepped down so a small nq still batches), 4 for the
        // LUT scan.
        const int qbs = !pd_batched ? 4 : n_q >= 12 ? 12 : n_q >= 8 ? 8 : 4;
#else
        // Upstream's x86 batch width: permute-dot takes 10 queries on one
        // thread when that saves a pass over 8, else 8; the `vpermb` scan
        // 6 (H56); the LUT scan 4 at a time. A short tail still batches,
        // padded to the permute-dot width.
        const bool vnni_batched = is_turbovec(*idx) && turbovec_scores_vnni_lut(*idx);
        const int qbs = pd_batched ?
            (idx->n_threads == 1 && (n_q + 9) / 10 < (n_q + 7) / 8 ? 10 : 8) :
            (vnni_batched ? 6 : 4);
#endif
        const size_t batch_heap_capacity = std::min(k_sz, idx->slot_to_id.size());
        const uint64_t per_query_scratch =
            static_cast<uint64_t>(batch_n_byte_groups) * 32 +
            (idx->turbovec_tqplus_shift.empty() ?
                0 : static_cast<uint64_t>(dim_sz) * sizeof(float));
        const uint64_t single_query_scratch =
            static_cast<uint64_t>(batch_heap_capacity) * sizeof(ScoreId) +
            static_cast<uint64_t>(idx->slot_to_id.size()) * sizeof(float) +
            per_query_scratch;
        const uint64_t batch_query_scratch =
            static_cast<uint64_t>(qbs) * (static_cast<uint64_t>(batch_heap_capacity) * sizeof(ScoreId) +
                 per_query_scratch);
        const bool batch_scratch_is_bounded =
            batch_query_scratch <= 2 * single_query_scratch;
        // A 2-bit vm8 batch scores differently from the single-query LUT
        // scan, so it batches whenever upstream does (filters included).
        bool batch_required = pd_batched && batch_bits == 2;
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        batch_required = batch_required || g_turbovec_force_batches.load(std::memory_order_relaxed);
#endif
        if (is_turbovec(*idx) &&
            !(idx->turbovec_planes && idx->turbovec_staged_enabled) &&
            active_count(*idx) == idx->slot_to_id.size() &&
            (batch_required || (allowed_ptr == nullptr && batch_scratch_is_bounded)) &&
            idx->turbovec_blocked_data.size() == idx->turbovec_blocked_n_blocks * batch_n_byte_groups * 32 &&
            idx->turbovec_blocked_n_blocks == (idx->slot_to_id.size() + 31) / 32) {
            int q = 0;
            for (; q + qbs <= n_q; q += qbs) {
                batches.push_back({ static_cast<size_t>(q), qbs, qbs });
            }
            const int rest = n_q - q;
#if GGML_VEC_INDEX_USE_NEON
            // A tail that matches a narrower width still batches; any other
            // remainder runs query by query.
            if (pd_batched && (rest == 8 || rest == 4)) {
                batches.push_back({ static_cast<size_t>(q), rest, rest });
                q += rest;
            }
#else
            // A tail of two or more queries batches; a lone one runs alone.
            if (rest >= 2) {
                batches.push_back({ static_cast<size_t>(q), rest, qbs });
                q += rest;
            }
#endif
            n_batched = q;
        }
        std::vector<uint8_t> batch_mask;
        std::vector<uint8_t> batch_blocks;
        if (!batches.empty() && allowed_ptr != nullptr) {
            const size_t n_slots = idx->slot_to_id.size();
            batch_mask.assign(n_slots, 0);
            batch_blocks.assign(idx->turbovec_blocked_n_blocks, 0);
            for (size_t slot : *allowed_ptr) {
                if (slot < n_slots) {
                    batch_mask[slot] = 1;
                    batch_blocks[slot / 32] = 1;
                }
            }
        }
#endif
        struct Scratch {
            std::vector<ScoreId> heap;
            std::vector<float> rotated_query;
            std::vector<float> calibrated_query;
            std::vector<uint8_t> lut;
            std::vector<uint8_t> allowed_blocks;
#if GGML_VEC_INDEX_TURBOVEC_BATCHES
            TurboVecBatchScratch batch;
#endif
        };
        // Several queries split across threads; a lone query may instead
        // split its blocks.
        const size_t n_batches = batches.size();
        const size_t n_tasks = n_batches + static_cast<size_t>(n_q - n_batched);
        // Fewer tasks than threads on a large index: run the tasks in turn,
        // each splitting its blocks, so every thread has work (upstream tiles
        // a batch over block ranges the same way).
        const bool split_blocks = n_tasks < static_cast<size_t>(idx->n_threads) && is_turbovec(*idx) &&
            idx->turbovec_blocked_n_blocks >= kTurboVecParallelMinBlocks;
        const int query_threads = n_tasks >= 2 && !split_blocks ? idx->n_threads : 1;
        const int block_threads = query_threads > 1 ? 1 : idx->n_threads;
        std::vector<Scratch> scratch(std::max<size_t>(1, std::min(n_tasks, static_cast<size_t>(query_threads))));
        // Per query: did upstream's sampled seed leave its shortlist short?
        std::vector<uint8_t> seed_short(n_q > 1 ? static_cast<size_t>(n_q) : 0, 0);
        parallel_for(n_tasks, query_threads, [&](size_t task, size_t worker) {
            Scratch & sc = scratch[worker];
#if GGML_VEC_INDEX_TURBOVEC_BATCHES
            if (task < n_batches) {
                const size_t q = batches[task].first;
                search_turbovec_query_batch(
                    *idx,
                    rotated_turbovec_queries.data() + q * dim_sz,
                    batches[task].n,
                    batches[task].width,
                    k,
                    out_scores + q * k_sz,
                    out_ids + q * k_sz,
                    batch_blocks.empty() ? nullptr : batch_blocks.data(),
                    batch_mask.empty() ? nullptr : batch_mask.data(),
                    sc.batch,
                    block_threads);
                return;
            }
#endif
            const size_t q = static_cast<size_t>(n_batched) + (task - n_batches);
            const double max_query = query_max_abs_values.empty() ? 0.0 : query_max_abs_values[q];
            bool query_short = false;
            search_one(
                *idx,
                queries + q * dim_sz,
                k,
                out_scores + q * k_sz,
                out_ids + q * k_sz,
                max_query,
                sc.heap,
                sc.rotated_query,
                sc.calibrated_query,
                sc.lut,
                sc.allowed_blocks,
                allowed_ptr,
                rotated_turbovec_queries.empty() ? nullptr : rotated_turbovec_queries.data() + q * dim_sz,
                block_threads,
                n_q == 1,
                seed_short.empty() ? nullptr : &query_short);
            if (!seed_short.empty()) {
                seed_short[q] = query_short ? 1 : 0;
            }
        });
        // Upstream rescans a batch's short queries unseeded; a lone short
        // query goes through the single-query scan and its narrower sign
        // tables, which can change its shortlist. Several short queries
        // keep the batch tables, as their first pass here already did.
        // The rescan runs alone, so it may split its blocks.
        if (std::count(seed_short.begin(), seed_short.end(), uint8_t{ 1 }) == 1) {
            const size_t q = static_cast<size_t>(std::find(seed_short.begin(), seed_short.end(), uint8_t{ 1 }) - seed_short.begin());
            Scratch & sc = scratch[0];
            search_one(
                *idx,
                queries + q * dim_sz,
                k,
                out_scores + q * k_sz,
                out_ids + q * k_sz,
                query_max_abs_values.empty() ? 0.0 : query_max_abs_values[q],
                sc.heap,
                sc.rotated_query,
                sc.calibrated_query,
                sc.lut,
                sc.allowed_blocks,
                allowed_ptr,
                rotated_turbovec_queries.empty() ? nullptr : rotated_turbovec_queries.data() + q * dim_sz,
                idx->n_threads,
                false,
                nullptr,
                true);
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
        if (is_turbovec(*idx) && !turbovec_supported_dim(dim, idx->turbovec_legacy)) {
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
        if (is_turbovec(*idx)) {
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
        std::vector<float> rotated_query_scratch;
        std::vector<float> calibrated_query_scratch;
        std::vector<uint8_t> turbovec_lut_scratch;
        std::vector<uint8_t> allowed_blocks_scratch;
        std::vector<float> rotated_turbovec_queries;
        if (is_turbovec(*idx)) {
            rotated_turbovec_queries.resize(n_q_sz * dim_sz);
            rotate_turbovec_queries(
                idx->turbovec_legacy,
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
                    rotated_query_scratch,
                    calibrated_query_scratch,
                    turbovec_lut_scratch,
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
                rotated_query_scratch,
                calibrated_query_scratch,
                turbovec_lut_scratch,
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
