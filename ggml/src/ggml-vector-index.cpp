// ggml-vector-index.cpp - CPU implementation of the vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors or per-vector symmetric q8/q4 codes. Search and
// persistence live in companion translation units.

#include "ggml-vector-index-impl.h"

static std::atomic<uint64_t> g_next_filter_cookie{ 1 };

void invalidate_ivf(ggml_vec_index & idx) {
    idx.ivf_generation = std::numeric_limits<uint64_t>::max();
    idx.ivf_n_lists = 0;
    idx.ivf_centroids.clear();
    idx.ivf_lists.clear();
}

bool is_q8(const ggml_vec_index & idx) {
    return idx.bit_width == 8;
}

bool is_q4(const ggml_vec_index & idx) {
    return idx.bit_width == 4 && !idx.turbovec_q4;
}

bool is_turbovec_q2(const ggml_vec_index & idx) {
    return idx.bit_width == 2 && idx.turbovec_q2;
}

bool is_turbovec_q4(const ggml_vec_index & idx) {
    return idx.bit_width == 4 && idx.turbovec_q4;
}

bool is_quantized(const ggml_vec_index & idx) {
    return is_q4(idx) || is_q8(idx) || is_turbovec_q2(idx) || is_turbovec_q4(idx);
}

uint8_t storage_kind(const ggml_vec_index & idx) {
    return is_turbovec_q2(idx) ? kStorageTurboVecQ2 :
        is_turbovec_q4(idx) ? kStorageTurboVecQ4 :
        (is_q4(idx) ? kStorageQ4 : (is_q8(idx) ? kStorageQ8 : kStorageF32));
}

size_t q4_row_bytes(size_t dim) {
    return (dim + 1) / 2;
}

size_t vector_bytes(const ggml_vec_index & idx) {
    const size_t n = idx.slot_to_id.size();
    const size_t dim_sz = static_cast<size_t>(idx.dim);
    if (is_q4(idx)) {
        return n * q4_row_bytes(dim_sz);
    }
    if (is_turbovec_q2(idx)) {
        return n * turbovec_q2_row_bytes(dim_sz);
    }
    if (is_turbovec_q4(idx)) {
        return n * turbovec_q4_row_bytes(dim_sz);
    }
    if (is_q8(idx)) {
        return n * dim_sz * sizeof(int8_t);
    }
    return n * dim_sz * sizeof(float);
}

static size_t grow_capacity(size_t current, size_t required, size_t max_capacity) {
    if (current >= required) {
        return current;
    }

    size_t grown = current == 0 ? required : current + current / 2;
    if (grown < current || grown < required) {
        grown = required;
    }
    return std::min(grown, max_capacity);
}

static bool can_insert_without_rehash(const std::unordered_map<uint64_t, size_t> & map, size_t n) {
    const long double bucket_capacity =
        static_cast<long double>(map.bucket_count()) * static_cast<long double>(map.max_load_factor());
    return static_cast<long double>(n) <= bucket_capacity;
}

bool all_finite(const float * values, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

bool slot_is_active(const ggml_vec_index & idx, size_t slot) {
    return slot < idx.slot_active.size() && idx.slot_active[slot] != 0;
}

size_t active_count(const ggml_vec_index & idx) {
    return idx.n_active;
}

const float * f32_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_data != nullptr ? idx.mapped_data : idx.data.data();
}

const int8_t * q8_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_q8_data != nullptr ? idx.mapped_q8_data : idx.q8_data.data();
}

const uint8_t * q4_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_q4_data != nullptr ? idx.mapped_q4_data : idx.q4_data.data();
}

const uint8_t * turbovec_q4_data_ptr(const ggml_vec_index & idx) {
    return idx.turbovec_q4_data.data();
}

const uint8_t * turbovec_q2_data_ptr(const ggml_vec_index & idx) {
    return idx.turbovec_q2_data.data();
}

bool has_vector_storage(const ggml_vec_index & idx) {
    const size_t bytes = vector_bytes(idx);
    if (idx.read_only_mmap && idx.mapped_file != nullptr) {
        return idx.mapped_vector_bytes == bytes &&
            (bytes == 0 ||
             idx.mapped_data != nullptr ||
             idx.mapped_q8_data != nullptr ||
             idx.mapped_q4_data != nullptr);
    }
    if (is_turbovec_q4(idx)) {
        return idx.turbovec_q4_data.size() == bytes;
    }
    if (is_turbovec_q2(idx)) {
        return idx.turbovec_q2_data.size() == bytes;
    }
    if (is_q4(idx)) {
        return idx.q4_data.size() == bytes;
    }
    if (is_q8(idx)) {
        return idx.q8_data.size() == bytes;
    }
    return idx.data.size() == bytes / sizeof(float);
}

static uint8_t q4_encode(int q) {
    return static_cast<uint8_t>(q + 8);
}

int q4_decode(uint8_t nibble) {
    return static_cast<int>(nibble) - 8;
}

static int round_nearest_even(float value) {
    const float lower_f = std::floor(value);
    const float upper_f = lower_f + 1.0f;
    const float lower_dist = value - lower_f;
    const float upper_dist = upper_f - value;
    if (lower_dist < upper_dist) {
        return static_cast<int>(lower_f);
    }
    if (upper_dist < lower_dist) {
        return static_cast<int>(upper_f);
    }

    const int lower = static_cast<int>(lower_f);
    return (lower % 2) == 0 ? lower : static_cast<int>(upper_f);
}

class ScopedNearestRounding {
public:
    ScopedNearestRounding() : saved_rounding(std::fegetround()) {
        if (saved_rounding != FE_TONEAREST && saved_rounding != -1) {
            std::fesetround(FE_TONEAREST);
        }
    }

    ~ScopedNearestRounding() {
        if (saved_rounding != FE_TONEAREST && saved_rounding != -1) {
            std::fesetround(saved_rounding);
        }
    }

    ScopedNearestRounding(const ScopedNearestRounding &) = delete;
    ScopedNearestRounding & operator=(const ScopedNearestRounding &) = delete;

private:
    int saved_rounding = FE_TONEAREST;
};

float quantization_scale(float max_abs, float divisor) {
    const ScopedNearestRounding rounding_guard;
    if (max_abs == 0.0f) {
        return 1.0f;
    }
    const float scale = max_abs / divisor;
    return scale == 0.0f ? max_abs : scale;
}

void quantize_q8_values(const float * src, int8_t * dst, size_t n, float scale) {
    const ScopedNearestRounding rounding_guard;
    for (size_t i = 0; i < n; ++i) {
        const float scaled = src[i] / scale;
        int q = round_nearest_even(scaled);
        q = std::max(-127, std::min(127, q));
        dst[i] = static_cast<int8_t>(q);
    }
}

void quantize_q4_values(const float * src, uint8_t * dst, size_t offset, size_t n, float scale) {
    const ScopedNearestRounding rounding_guard;
    for (size_t i = 0; i < n; ++i) {
        const float scaled = src[i] / scale;
        int         q      = round_nearest_even(scaled);
        q                  = std::max(-7, std::min(7, q));
        const uint8_t code = q4_encode(q);
        const size_t  component = offset + i;
        uint8_t &     byte      = dst[component / 2];
        if ((component & 1) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | code);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (code << 4));
        }
    }
}

void quantize_q8_row(const float * src, int8_t * dst, int dim, float & scale) {
    if (dim <= 0) {
        scale = 1.0f;
        return;
    }
    const ScopedNearestRounding rounding_guard;
    const size_t dim_sz = static_cast<size_t>(dim);

    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    if (max_abs == 0.0f) {
        scale = 1.0f;
        std::memset(dst, 0, dim_sz);
        return;
    }

    scale = quantization_scale(max_abs, 127.0f);
    quantize_q8_values(src, dst, dim_sz, scale);
}

void quantize_q4_row(const float * src, uint8_t * dst, int dim, float & scale) {
    if (dim <= 0) {
        scale = 1.0f;
        return;
    }
    const ScopedNearestRounding rounding_guard;

    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    std::memset(dst, 0x88, q4_row_bytes(static_cast<size_t>(dim)));
    if (max_abs == 0.0f) {
        scale = 1.0f;
        return;
    }

    scale = quantization_scale(max_abs, 7.0f);
    quantize_q4_values(src, dst, 0, static_cast<size_t>(dim), scale);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

const char * ggml_vec_index_error_to_string(int error) {
    switch (error) {
        case GGML_VEC_INDEX_OK:
            return "ok";
        case GGML_VEC_INDEX_E_INVALID_ARG:
            return "invalid argument";
        case GGML_VEC_INDEX_E_DUPLICATE:
            return "duplicate id";
        case GGML_VEC_INDEX_E_IO:
            return "I/O or corrupt data";
        case GGML_VEC_INDEX_E_BAD_MAGIC:
            return "bad magic";
        case GGML_VEC_INDEX_E_BAD_VERSION:
            return "bad version";
        case GGML_VEC_INDEX_E_OOM:
            return "out of memory";
        case GGML_VEC_INDEX_E_PARTIAL_COMPACT:
            return "partial compaction";
        case GGML_VEC_INDEX_E_NOT_FOUND:
            return "not found";
        case GGML_VEC_INDEX_E_NOT_DURABLE:
            return "not durable";
        case GGML_VEC_INDEX_E_INTERNAL:
            return "internal error";
        default:
            return "unknown error";
    }
}

ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width) {
    try {
        if (dim <= 0) {
            return nullptr;
        }
        if (!is_supported_bit_width(bit_width) || bit_width == 2) {
            return nullptr;
        }
        auto * idx = new (std::nothrow) ggml_vec_index();
        if (idx == nullptr) {
            return nullptr;
        }
        idx->dim       = dim;
        idx->bit_width = bit_width;
        idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        if (idx->filter_cookie == 0) {
            idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        }
        return idx;
    } catch (...) {
        return nullptr;
    }
}

static ggml_vec_index_t * ggml_vec_index_create_turbovec_impl(
        int dim,
        int bit_width,
        bool serialized_compatibility) {
    try {
        const bool supported = serialized_compatibility ?
            turbovec_serialized_dim_supported(dim) :
            (bit_width == 2 ? turbovec_q2_supported_dim(dim) : turbovec_q4_supported_dim(dim));
        if (!supported || (bit_width != 2 && bit_width != 4)) {
            return nullptr;
        }
        auto * idx = new (std::nothrow) ggml_vec_index();
        if (idx == nullptr) {
            return nullptr;
        }
        idx->dim = dim;
        idx->bit_width = bit_width;
        idx->turbovec_q2 = bit_width == 2;
        idx->turbovec_q4 = bit_width == 4;
        idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        if (idx->filter_cookie == 0) {
            idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        }
        try {
            turbovec_retain_rotation(dim);
        } catch (...) {
            delete idx;
            return nullptr;
        }
        return idx;
    } catch (...) {
        return nullptr;
    }
}

ggml_vec_index_t * ggml_vec_index_create_turbovec_q2(int dim) {
    return ggml_vec_index_create_turbovec_impl(dim, 2, false);
}

ggml_vec_index_t * ggml_vec_index_create_turbovec_q4(int dim) {
    return ggml_vec_index_create_turbovec_impl(dim, 4, false);
}

ggml_vec_index_t * ggml_vec_index_create_turbovec_for_load(int dim, int bit_width) {
    return ggml_vec_index_create_turbovec_impl(dim, bit_width, true);
}

void ggml_vec_index_free(ggml_vec_index_t * idx) {
    if (idx != nullptr && (is_turbovec_q2(*idx) || is_turbovec_q4(*idx))) {
        turbovec_release_rotation(idx->dim);
    }
    delete idx;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void rollback_appended_slots_unlocked(
        ggml_vec_index_t * idx,
        size_t base_slot,
        const uint64_t * ids,
        int n) noexcept {
    if (idx == nullptr) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        idx->id_to_slot.erase(ids[i]);
    }
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    if (is_turbovec_q2(*idx)) {
        idx->turbovec_q2_data.resize(base_slot * turbovec_q2_row_bytes(dim_sz));
        idx->turbovec_q2_scale.resize(base_slot * turbovec_q2_scale_count(dim_sz));
        if (base_slot == 0) {
            idx->turbovec_tqplus_shift.clear();
            idx->turbovec_tqplus_scale.clear();
        }
    } else if (is_turbovec_q4(*idx)) {
        idx->turbovec_q4_data.resize(base_slot * turbovec_q4_row_bytes(dim_sz));
        idx->turbovec_q4_scale.resize(base_slot * turbovec_q4_scale_count(dim_sz));
        if (base_slot == 0) {
            idx->turbovec_tqplus_shift.clear();
            idx->turbovec_tqplus_scale.clear();
        }
    } else if (is_q4(*idx)) {
        idx->q4_data.resize(base_slot * q4_row_bytes(dim_sz));
        idx->q4_scale.resize(base_slot);
    } else if (is_q8(*idx)) {
        idx->q8_data.resize(base_slot * dim_sz);
        idx->q8_scale.resize(base_slot);
    } else {
        idx->data.resize(base_slot * dim_sz);
    }
    idx->slot_to_id.resize(base_slot);
    idx->slot_active.resize(base_slot);
    idx->n_active = idx->id_to_slot.size();
}

int ggml_vec_index_add_unlocked(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    bool               finalize) {

    size_t base_slot = 0;
    size_t dim_sz    = 0;
    bool resized     = false;

    auto rollback = [&]() noexcept {
        if (idx == nullptr || !resized) {
            return;
        }
        rollback_appended_slots_unlocked(idx, base_slot, ids, n);
    };

    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n < 0) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (finalize && idx->delta_log_bound) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }
        if ((is_turbovec_q2(*idx) && !turbovec_q2_supported_dim(idx->dim)) ||
            (is_turbovec_q4(*idx) && !turbovec_q4_supported_dim(idx->dim))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (vectors == nullptr || ids == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        // Atomic add: detect duplicates first (against existing AND in-batch),
        // bail before mutating any state.
        test_maybe_throw_bad_alloc();
        std::unordered_set<uint64_t> batch_ids;
        batch_ids.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (!is_valid_id(ids[i])) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
            if (!batch_ids.insert(ids[i]).second) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
        }

        base_slot         = idx->slot_to_id.size();
        dim_sz            = static_cast<size_t>(idx->dim);
        const size_t n_sz = static_cast<size_t>(n);
        if (n_sz > kMaxIndexLen || active_count(*idx) > kMaxIndexLen - n_sz || base_slot > kMaxIndexLen - n_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n_sz > std::numeric_limits<size_t>::max() - base_slot) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t new_slots = base_slot + n_sz;
        size_t value_count = 0;
        if (dim_sz != 0 && n_sz > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        value_count = n_sz * dim_sz;
        if (!can_address_array(value_count, sizeof(float)) ||
            !can_address_array(n_sz, sizeof(uint64_t)) ||
            !can_address_array(new_slots, sizeof(uint64_t))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t row_elements = is_turbovec_q2(*idx) ? turbovec_q2_row_bytes(dim_sz) :
                                    is_turbovec_q4(*idx) ? turbovec_q4_row_bytes(dim_sz) :
                                    is_q4(*idx) ? q4_row_bytes(dim_sz) : dim_sz;
        const size_t element_size = is_quantized(*idx) ? sizeof(uint8_t) : sizeof(float);
        const size_t scale_count = is_turbovec_q2(*idx) ? turbovec_q2_scale_count(dim_sz) :
                                   is_turbovec_q4(*idx) ? turbovec_q4_scale_count(dim_sz) :
                                   (is_q4(*idx) || is_q8(*idx) ? 1 : 0);
        if (!can_address_array(new_slots, row_elements) ||
            !can_address_array(new_slots * row_elements, element_size) ||
            (scale_count != 0 &&
             (!can_address_array(new_slots, scale_count) ||
              !can_address_array(new_slots * scale_count, sizeof(float))))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const bool valid_values = is_turbovec_q2(*idx) || is_turbovec_q4(*idx) ?
            all_finite_abs_less_than(vectors, value_count, kTurboVecMaxInputMagnitude) :
            all_finite(vectors, value_count);
        if (!valid_values) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        size_t max_slots = std::min(kMaxIndexLen, std::numeric_limits<size_t>::max() / sizeof(uint64_t));
        max_slots = std::min(max_slots, std::numeric_limits<size_t>::max() / row_elements / element_size);
        if (scale_count != 0) {
            max_slots = std::min(max_slots, std::numeric_limits<size_t>::max() / scale_count / sizeof(float));
        }
        size_t current_slots = std::min(idx->slot_to_id.capacity(), idx->slot_active.capacity());
        if (is_q4(*idx)) {
            const size_t row_bytes = q4_row_bytes(dim_sz);
            current_slots = std::min(current_slots, idx->q4_data.capacity() / row_bytes);
            current_slots = std::min(current_slots, idx->q4_scale.capacity());
        } else if (is_q8(*idx)) {
            current_slots = std::min(current_slots, idx->q8_data.capacity() / dim_sz);
            current_slots = std::min(current_slots, idx->q8_scale.capacity());
        } else {
            current_slots = std::min(current_slots, idx->data.capacity() / dim_sz);
        }
        const size_t target_slots  = grow_capacity(current_slots, new_slots, max_slots);

        if (is_q4(*idx)) {
            const size_t row_bytes     = q4_row_bytes(dim_sz);
            const size_t target_values = target_slots * row_bytes;
            if (idx->q4_data.capacity() < target_values) {
                idx->q4_data.reserve(target_values);
            }
            if (idx->q4_scale.capacity() < target_slots) {
                idx->q4_scale.reserve(target_slots);
            }
        } else if (is_q8(*idx)) {
            const size_t target_values = target_slots * dim_sz;
            if (idx->q8_data.capacity() < target_values) {
                idx->q8_data.reserve(target_values);
            }
            if (idx->q8_scale.capacity() < target_slots) {
                idx->q8_scale.reserve(target_slots);
            }
        } else {
            const size_t target_values = target_slots * dim_sz;
            if (idx->data.capacity() < target_values) {
                idx->data.reserve(target_values);
            }
        }
        if (idx->slot_to_id.capacity() < target_slots) {
            idx->slot_to_id.reserve(target_slots);
        }
        if (idx->slot_active.capacity() < target_slots) {
            idx->slot_active.reserve(target_slots);
        }
        if (!can_insert_without_rehash(idx->id_to_slot, new_slots)) {
            idx->id_to_slot.reserve(target_slots);
        }

        resized = true;
        if (is_turbovec_q2(*idx)) {
            idx->turbovec_q2_data.resize(new_slots * turbovec_q2_row_bytes(dim_sz));
            idx->turbovec_q2_scale.resize(new_slots * turbovec_q2_scale_count(dim_sz));
        } else if (is_turbovec_q4(*idx)) {
            idx->turbovec_q4_data.resize(new_slots * turbovec_q4_row_bytes(dim_sz));
            idx->turbovec_q4_scale.resize(new_slots * turbovec_q4_scale_count(dim_sz));
        } else if (is_q4(*idx)) {
            idx->q4_data.resize(new_slots * q4_row_bytes(dim_sz));
            idx->q4_scale.resize(new_slots);
        } else if (is_q8(*idx)) {
            idx->q8_data.resize(new_slots * dim_sz);
            idx->q8_scale.resize(new_slots);
        } else {
            idx->data.resize(new_slots * dim_sz);
        }
        idx->slot_to_id.resize(new_slots);
        idx->slot_active.resize(new_slots, 0);
        test_maybe_throw_bad_alloc();

        if (is_turbovec_q2(*idx)) {
            quantize_turbovec_batch(
                vectors,
                n,
                2,
                idx->turbovec_q2_data.data() + base_slot * turbovec_q2_row_bytes(dim_sz),
                idx->turbovec_q2_scale.data() + base_slot,
                idx->dim,
                idx->turbovec_tqplus_shift,
                idx->turbovec_tqplus_scale);
        } else if (is_turbovec_q4(*idx)) {
            quantize_turbovec_batch(
                vectors,
                n,
                4,
                idx->turbovec_q4_data.data() + base_slot * turbovec_q4_row_bytes(dim_sz),
                idx->turbovec_q4_scale.data() + base_slot,
                idx->dim,
                idx->turbovec_tqplus_shift,
                idx->turbovec_tqplus_scale);
        }

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            const float * src = vectors + static_cast<size_t>(i) * dim_sz;
            if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
                (void) src;
            } else if (is_q4(*idx)) {
                quantize_q4_row(
                    src,
                    idx->q4_data.data() + slot * q4_row_bytes(dim_sz),
                    idx->dim,
                    idx->q4_scale[slot]);
            } else if (is_q8(*idx)) {
                quantize_q8_row(
                    src,
                    idx->q8_data.data() + slot * dim_sz,
                    idx->dim,
                    idx->q8_scale[slot]);
            } else {
                std::memcpy(
                    idx->data.data() + slot * dim_sz,
                    src,
                    dim_sz * sizeof(float));
            }
            idx->slot_to_id[slot] = ids[i];
            idx->slot_active[slot] = 1;
            test_maybe_throw_bad_alloc();
            idx->id_to_slot.emplace(ids[i], slot);
        }
        idx->n_active += n_sz;
        if (is_turbovec_q2(*idx)) {
            repack_turbovec_codes_from_slot(
                idx->turbovec_q2_data.data(),
                new_slots,
                2,
                idx->dim,
                base_slot,
                idx->turbovec_blocked_data,
                idx->turbovec_blocked_n_blocks);
        } else if (is_turbovec_q4(*idx)) {
            repack_turbovec_codes_from_slot(
                idx->turbovec_q4_data.data(),
                new_slots,
                4,
                idx->dim,
                base_slot,
                idx->turbovec_blocked_data,
                idx->turbovec_blocked_n_blocks);
        }
        for (size_t slot = base_slot; slot < new_slots; ++slot) {
            add_state_hash(*idx, slot_state_hash(*idx, slot));
        }
        if (finalize) {
            ++idx->generation;
            invalidate_ivf(*idx);
            idx->delta_log_start_allowed = false;
        }
    } catch (const std::bad_alloc &) {
        rollback();
        return GGML_VEC_INDEX_E_OOM;
    } catch (const std::invalid_argument &) {
        rollback();
        return GGML_VEC_INDEX_E_INVALID_ARG;
    } catch (...) {
        rollback();
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_add_unlocked(idx, vectors, n, ids, true);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_remove_unlocked(
        ggml_vec_index_t * idx,
        uint64_t id,
        bool allow_delta_bound) {
    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!is_valid_id(id)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        auto it = idx->id_to_slot.find(id);
        if (it == idx->id_to_slot.end()) {
            return GGML_VEC_INDEX_E_NOT_FOUND;
        }
        const size_t slot = it->second;
        if (!slot_is_active(*idx, slot)) {
            idx->id_to_slot.erase(it);
            return GGML_VEC_INDEX_E_NOT_FOUND;
        }
        if (idx->delta_log_bound && !allow_delta_bound) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        remove_state_hash(*idx, slot_state_hash(*idx, slot));
        idx->slot_active[slot] = 0;
        --idx->n_active;
        idx->id_to_slot.erase(it);
        ++idx->generation;
        invalidate_ivf(*idx);
        if (!allow_delta_bound) {
            idx->delta_log_start_allowed = false;
        }
        return GGML_VEC_INDEX_OK;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_remove_unlocked(idx, id);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static int ggml_vec_index_compact_unlocked(ggml_vec_index_t * idx) {
    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const size_t n_slots = idx->slot_to_id.size();
        const size_t n_live  = active_count(*idx);
        if (idx->delta_log_bound) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        size_t counted_live = 0;
        for (size_t slot = 0; slot < n_slots; ++slot) {
            counted_live += slot_is_active(*idx, slot) ? 1 : 0;
        }
        if (idx->slot_active.size() != n_slots || counted_live != n_live || idx->id_to_slot.size() != n_live ||
            !has_vector_storage(*idx)) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }
        if (n_live == n_slots) {
            return GGML_VEC_INDEX_OK;
        }

        const size_t dim_sz = static_cast<size_t>(idx->dim);
        if (dim_sz != 0 && n_live > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }

        test_maybe_throw_bad_alloc();
        std::vector<uint64_t>                new_slot_to_id;
        std::vector<uint8_t>                 new_slot_active;
        std::unordered_map<uint64_t, size_t> new_id_to_slot;
        new_slot_to_id.reserve(n_live);
        new_slot_active.assign(n_live, 1);
        new_id_to_slot.reserve(n_live);
        std::vector<uint8_t> new_turbovec_blocked_data;
        size_t new_turbovec_blocked_n_blocks = 0;

        if (is_turbovec_q2(*idx)) {
            const size_t row_bytes = turbovec_q2_row_bytes(dim_sz);
            const size_t scale_count = turbovec_q2_scale_count(dim_sz);
            std::vector<uint8_t> new_data;
            std::vector<float> new_scales;
            new_data.resize(n_live * row_bytes);
            new_scales.resize(n_live * scale_count);
            size_t out_slot = 0;
            for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                std::memcpy(
                    new_data.data() + out_slot * row_bytes,
                    idx->turbovec_q2_data.data() + slot * row_bytes,
                    row_bytes);
                std::memcpy(
                    new_scales.data() + out_slot * scale_count,
                    idx->turbovec_q2_scale.data() + slot * scale_count,
                    scale_count * sizeof(float));
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
                ++out_slot;
            }
            if (turbovec_q2_supported_dim(idx->dim)) {
                repack_turbovec_codes(
                    new_data.data(),
                    n_live,
                    2,
                    idx->dim,
                    new_turbovec_blocked_data,
                    new_turbovec_blocked_n_blocks);
            }
            idx->turbovec_q2_data.swap(new_data);
            idx->turbovec_q2_scale.swap(new_scales);
            idx->turbovec_blocked_data.swap(new_turbovec_blocked_data);
            idx->turbovec_blocked_n_blocks = new_turbovec_blocked_n_blocks;
        } else if (is_turbovec_q4(*idx)) {
            const size_t row_bytes = turbovec_q4_row_bytes(dim_sz);
            const size_t scale_count = turbovec_q4_scale_count(dim_sz);
            std::vector<uint8_t> new_data;
            std::vector<float> new_scales;
            new_data.resize(n_live * row_bytes);
            new_scales.resize(n_live * scale_count);
            size_t out_slot = 0;
            for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                std::memcpy(
                    new_data.data() + out_slot * row_bytes,
                    idx->turbovec_q4_data.data() + slot * row_bytes,
                    row_bytes);
                std::memcpy(
                    new_scales.data() + out_slot * scale_count,
                    idx->turbovec_q4_scale.data() + slot * scale_count,
                    scale_count * sizeof(float));
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
                ++out_slot;
            }
            if (turbovec_q4_supported_dim(idx->dim)) {
                repack_turbovec_codes(
                    new_data.data(),
                    n_live,
                    4,
                    idx->dim,
                    new_turbovec_blocked_data,
                    new_turbovec_blocked_n_blocks);
            }
            idx->turbovec_q4_data.swap(new_data);
            idx->turbovec_q4_scale.swap(new_scales);
            idx->turbovec_blocked_data.swap(new_turbovec_blocked_data);
            idx->turbovec_blocked_n_blocks = new_turbovec_blocked_n_blocks;
        } else if (is_q4(*idx)) {
            const size_t row_bytes = q4_row_bytes(dim_sz);
            std::vector<uint8_t> new_q4_data;
            std::vector<float> new_q4_scale;
            new_q4_data.resize(n_live * row_bytes);
            new_q4_scale.reserve(n_live);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_q4_data.data() + out_slot * row_bytes,
                    idx->q4_data.data() + slot * row_bytes,
                    row_bytes * sizeof(uint8_t));
                new_q4_scale.push_back(idx->q4_scale[slot]);
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->q4_data.swap(new_q4_data);
            idx->q4_scale.swap(new_q4_scale);
        } else if (is_q8(*idx)) {
            std::vector<int8_t> new_q8_data;
            std::vector<float> new_q8_scale;
            new_q8_data.resize(n_live * dim_sz);
            new_q8_scale.reserve(n_live);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_q8_data.data() + out_slot * dim_sz,
                    idx->q8_data.data() + slot * dim_sz,
                    dim_sz * sizeof(int8_t));
                new_q8_scale.push_back(idx->q8_scale[slot]);
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->q8_data.swap(new_q8_data);
            idx->q8_scale.swap(new_q8_scale);
        } else {
            std::vector<float> new_data;
            new_data.resize(n_live * dim_sz);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_data.data() + out_slot * dim_sz,
                    idx->data.data() + slot * dim_sz,
                    dim_sz * sizeof(float));
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->data.swap(new_data);
        }

        idx->slot_to_id.swap(new_slot_to_id);
        idx->slot_active.swap(new_slot_active);
        idx->id_to_slot.swap(new_id_to_slot);
        idx->n_active = idx->slot_to_id.size();
        ++idx->generation;
        invalidate_ivf(*idx);
        idx->delta_log_start_allowed = false;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_compact(ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_compact_unlocked(idx);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static bool prepare_delta_log_binding(
        const ggml_vec_index & idx,
        const char * delta_path,
        std::string & path_key) {
    if (!delta_log_path_key(delta_path, path_key)) {
        return false;
    }
    return idx.bound_delta_log_path_key.empty() ||
        idx.bound_delta_log_path_key == path_key ||
        filesystem_paths_equal(idx.bound_delta_log_path_key.c_str(), delta_path);
}

static void commit_delta_log_binding(
        ggml_vec_index & idx,
        std::string & path_key) noexcept {
    if (idx.bound_delta_log_path_key.empty()) {
        idx.bound_delta_log_path_key.swap(path_key);
    }
    idx.delta_log_bound = true;
}

int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path) {
    bool added = false;
    bool prepared_path = false;
    size_t base_slot = 0;
    std::unique_lock<std::shared_mutex> lock;
    auto discard_prepared_path = [&]() noexcept {
        if (prepared_path && idx != nullptr && !idx->delta_log_bound) {
            idx->bound_delta_log_path_key.clear();
            prepared_path = false;
        }
    };
    try {
        if (idx == nullptr || delta_path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        lock = std::unique_lock<std::shared_mutex>(idx->mutex);
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!validate_logged_add_args(idx, vectors, n, ids)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->delta_log_reload_required) {
            return GGML_VEC_INDEX_E_IO;
        }
        const bool first_logged_mutation = !idx->delta_log_bound;
        if (first_logged_mutation && !idx->delta_log_start_allowed) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!first_logged_mutation && !bind_delta_log_path(*idx, delta_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }

        DeltaLogLock delta_lock(delta_path);
        if (!delta_lock.ok()) {
            return GGML_VEC_INDEX_E_IO;
        }
        if (first_logged_mutation && !bind_delta_log_path(*idx, delta_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        prepared_path = first_logged_mutation;
        if (!delta_lock.ensure_data_file_locked(delta_path)) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_IO;
        }
        DeltaLogFormat format = DeltaLogFormat::v4;
        if (!prepare_delta_log_format_for_append_unlocked(idx, delta_path, delta_lock, format)) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_IO;
        }
        if (format == DeltaLogFormat::v1 || format == DeltaLogFormat::v2 || format == DeltaLogFormat::v3) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!delta_log_matches_index_unlocked(idx, delta_path, &delta_lock)) {
            if (!replay_delta_log_unlocked(idx, delta_path, delta_lock)) {
                discard_prepared_path();
                return GGML_VEC_INDEX_E_IO;
            }
            idx->delta_log_start_allowed = false;
            idx->delta_log_bound = true;
            invalidate_delta_tail_cache(*idx);
        }
        const int duplicate_status = check_logged_add_duplicates(idx, n, ids);
        if (duplicate_status != GGML_VEC_INDEX_OK) {
            discard_prepared_path();
            return duplicate_status;
        }

        std::string delta_path_key;
        if (!prepare_delta_log_binding(*idx, delta_path, delta_path_key)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const DeltaStateKind state_kind = delta_state_kind_for_format(format);
        const uint32_t base_crc = current_delta_state(*idx, state_kind);
        const DeltaStateWide base_wide = current_delta_state_wide(*idx);

        std::vector<uint8_t> payload;
        if (format != DeltaLogFormat::v3 &&
            format != DeltaLogFormat::v4 &&
            !build_add_delta_payload_f32(idx, vectors, n, ids, payload)) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        base_slot = idx->slot_to_id.size();
        const int add_status = ggml_vec_index_add_unlocked(idx, vectors, n, ids, false);
        if (add_status != GGML_VEC_INDEX_OK) {
            discard_prepared_path();
            return add_status;
        }
        added = true;
        if ((format == DeltaLogFormat::v3 || format == DeltaLogFormat::v4) &&
            !build_add_delta_payload_from_slots(idx, base_slot, n, payload)) {
            for (size_t slot = base_slot; slot < idx->slot_to_id.size(); ++slot) {
                if (slot_is_active(*idx, slot)) {
                    remove_state_hash(*idx, slot_state_hash(*idx, slot));
                }
            }
            rollback_appended_slots_unlocked(idx, base_slot, ids, n);
            added = false;
            discard_prepared_path();
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const uint32_t added_state_crc = current_delta_state(*idx, state_kind);
        const DeltaStateWide added_state_wide = current_delta_state_wide(*idx);
        const DeltaAppendResult append_result = append_delta_record_locked(
            *idx,
            delta_lock,
            delta_path,
            format,
            kTvidOpAdd,
            static_cast<uint32_t>(n),
            base_crc,
            added_state_crc,
            base_wide,
            added_state_wide,
            payload);
        if (append_result.status != GGML_VEC_INDEX_OK) {
            if (append_result.status == GGML_VEC_INDEX_E_NOT_DURABLE ||
                (append_result.record_complete && !append_result.data_synced)) {
                idx->delta_log_reload_required = true;
            }
            if (append_result.record_complete) {
                ++idx->generation;
                invalidate_ivf(*idx);
                commit_delta_log_binding(*idx, delta_path_key);
                added = false;
                if (!append_result.data_synced) {
                    return GGML_VEC_INDEX_E_IO;
                }
                return append_result.status;
            } else {
                for (size_t slot = base_slot; slot < idx->slot_to_id.size(); ++slot) {
                    if (slot_is_active(*idx, slot)) {
                        remove_state_hash(*idx, slot_state_hash(*idx, slot));
                    }
                }
                rollback_appended_slots_unlocked(idx, base_slot, ids, n);
            }
            added = false;
            discard_prepared_path();
            return append_result.status;
        }
        ++idx->generation;
        invalidate_ivf(*idx);
        commit_delta_log_binding(*idx, delta_path_key);
        added = false;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        if (added) {
            for (size_t slot = base_slot; slot < idx->slot_to_id.size(); ++slot) {
                if (slot_is_active(*idx, slot)) {
                    remove_state_hash(*idx, slot_state_hash(*idx, slot));
                }
            }
            rollback_appended_slots_unlocked(idx, base_slot, ids, n);
        }
        discard_prepared_path();
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        if (added) {
            for (size_t slot = base_slot; slot < idx->slot_to_id.size(); ++slot) {
                if (slot_is_active(*idx, slot)) {
                    remove_state_hash(*idx, slot_state_hash(*idx, slot));
                }
            }
            rollback_appended_slots_unlocked(idx, base_slot, ids, n);
        }
        discard_prepared_path();
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path) {
    bool prepared_path = false;
    std::unique_lock<std::shared_mutex> lock;
    auto discard_prepared_path = [&]() noexcept {
        if (prepared_path && idx != nullptr && !idx->delta_log_bound) {
            idx->bound_delta_log_path_key.clear();
            prepared_path = false;
        }
    };
    try {
        if (idx == nullptr || delta_path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        lock = std::unique_lock<std::shared_mutex>(idx->mutex);
        if (!is_valid_id(id)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->delta_log_reload_required) {
            return GGML_VEC_INDEX_E_IO;
        }
        if (is_turbovec_q2(*idx) || is_turbovec_q4(*idx)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        DeltaLogLock delta_lock(delta_path);
        if (!delta_lock.ok()) {
            return GGML_VEC_INDEX_E_IO;
        }
        const bool first_logged_mutation = !idx->delta_log_bound;
        if (first_logged_mutation && !idx->delta_log_start_allowed) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!bind_delta_log_path(*idx, delta_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        prepared_path = first_logged_mutation;
        if (!delta_lock.ensure_data_file_locked(delta_path)) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_IO;
        }
        DeltaLogFormat format = DeltaLogFormat::v4;
        if (!prepare_delta_log_format_for_append_unlocked(idx, delta_path, delta_lock, format)) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_IO;
        }
        if (format == DeltaLogFormat::v1 || format == DeltaLogFormat::v2 || format == DeltaLogFormat::v3) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!delta_log_matches_index_unlocked(idx, delta_path, &delta_lock)) {
            if (!replay_delta_log_unlocked(idx, delta_path, delta_lock)) {
                discard_prepared_path();
                return GGML_VEC_INDEX_E_IO;
            }
            idx->delta_log_start_allowed = false;
            idx->delta_log_bound = true;
            invalidate_delta_tail_cache(*idx);
        }
        if (idx->id_to_slot.count(id) == 0) {
            discard_prepared_path();
            return GGML_VEC_INDEX_E_NOT_FOUND;
        }
        std::string delta_path_key;
        if (!prepare_delta_log_binding(*idx, delta_path, delta_path_key)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const std::vector<uint8_t> payload = build_remove_delta_payload(id);
        const DeltaStateKind state_kind = delta_state_kind_for_format(format);
        const uint32_t base_crc = current_delta_state(*idx, state_kind);
        const DeltaStateWide base_wide = current_delta_state_wide(*idx);
        const uint32_t post_remove_crc =
            state_kind == DeltaStateKind::legacy_crc ?
                index_state_crc32c_after_remove(*idx, id) :
                index_state_token_after_remove(*idx, id);
        const DeltaStateWide post_remove_wide = index_state_wide_after_remove(*idx, id);
        const DeltaAppendResult append_result = append_delta_record_locked(
            *idx,
            delta_lock,
            delta_path,
            format,
            kTvidOpRemove,
            1,
            base_crc,
            post_remove_crc,
            base_wide,
            post_remove_wide,
            payload);
        if (append_result.status != GGML_VEC_INDEX_OK) {
            if (append_result.status == GGML_VEC_INDEX_E_NOT_DURABLE ||
                (append_result.record_complete && !append_result.data_synced)) {
                idx->delta_log_reload_required = true;
            }
            if (append_result.record_complete) {
                const int remove_status = ggml_vec_index_remove_unlocked(
                    idx, id, /*allow_delta_bound=*/true);
                if (remove_status != GGML_VEC_INDEX_OK) {
                    invalidate_delta_tail_cache(*idx);
                    return GGML_VEC_INDEX_E_INTERNAL;
                }
                commit_delta_log_binding(*idx, delta_path_key);
                if (!append_result.data_synced) {
                    return GGML_VEC_INDEX_E_IO;
                }
                return append_result.status;
            }
            discard_prepared_path();
            return append_result.status;
        }
        const int remove_status = ggml_vec_index_remove_unlocked(
            idx, id, /*allow_delta_bound=*/true);
        if (remove_status != GGML_VEC_INDEX_OK) {
            invalidate_delta_tail_cache(*idx);
            return GGML_VEC_INDEX_E_INTERNAL;
        }
        commit_delta_log_binding(*idx, delta_path_key);
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        discard_prepared_path();
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        discard_prepared_path();
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const auto it = idx->id_to_slot.find(id);
        return it != idx->id_to_slot.end() && slot_is_active(*idx, it->second) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void ggml_vec_index_prepare(ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        if (is_turbovec_q2(*idx) && turbovec_q2_supported_dim(idx->dim)) {
            prepare_turbovec(2, idx->dim);
        } else if (is_turbovec_q4(*idx) && turbovec_q4_supported_dim(idx->dim)) {
            prepare_turbovec(4, idx->dim);
        }
    } catch (...) {
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int ggml_vec_index_len(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const size_t n = active_count(*idx);
        return n > kMaxIndexLen ? std::numeric_limits<int>::max() : static_cast<int>(n);
    } catch (...) {
        return 0;
    }
}

int ggml_vec_index_dim(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        return idx->dim;
    } catch (...) {
        return 0;
    }
}

int ggml_vec_index_bit_width(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        return idx->bit_width;
    } catch (...) {
        return 0;
    }
}
