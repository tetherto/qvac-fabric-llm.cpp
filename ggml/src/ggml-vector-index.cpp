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
    return idx.bit_width == 4;
}

bool is_quantized(const ggml_vec_index & idx) {
    return is_q4(idx) || is_q8(idx);
}

uint8_t storage_kind(const ggml_vec_index & idx) {
    return is_q4(idx) ? kStorageQ4 : (is_q8(idx) ? kStorageQ8 : kStorageF32);
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

bool has_vector_storage(const ggml_vec_index & idx) {
    const size_t bytes = vector_bytes(idx);
    if (idx.read_only_mmap) {
        return idx.mapped_vector_bytes == bytes &&
            (bytes == 0 ||
             idx.mapped_data != nullptr ||
             idx.mapped_q8_data != nullptr ||
             idx.mapped_q4_data != nullptr);
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

void quantize_q8_row(const float * src, int8_t * dst, int dim, float & scale) {
    const ScopedNearestRounding rounding_guard;

    if (dim <= 0) {
        scale = 1.0f;
        return;
    }
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

    scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        scale = max_abs;
    }
    for (int i = 0; i < dim; ++i) {
        const float scaled = src[i] / scale;
        int q = round_nearest_even(scaled);
        q                  = std::max(-127, std::min(127, q));
        dst[i]             = static_cast<int8_t>(q);
    }
}

void quantize_q4_row(const float * src, uint8_t * dst, int dim, float & scale) {
    const ScopedNearestRounding rounding_guard;

    if (dim <= 0) {
        scale = 1.0f;
        return;
    }

    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    std::memset(dst, 0x88, q4_row_bytes(static_cast<size_t>(dim)));
    if (max_abs == 0.0f) {
        scale = 1.0f;
        return;
    }

    scale = max_abs / 7.0f;
    if (scale == 0.0f) {
        scale = max_abs;
    }
    for (int i = 0; i < dim; ++i) {
        const float scaled = src[i] / scale;
        int         q      = round_nearest_even(scaled);
        q                  = std::max(-7, std::min(7, q));
        const uint8_t code = q4_encode(q);
        uint8_t &     byte = dst[static_cast<size_t>(i) / 2];
        if ((i & 1) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | code);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (code << 4));
        }
    }
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
        if (!is_supported_bit_width(bit_width)) {
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

void ggml_vec_index_free(ggml_vec_index_t * idx) {
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
    if (is_q4(*idx)) {
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
            !can_address_array(new_slots, sizeof(uint64_t)) ||
            (is_quantized(*idx) && !can_address_array(new_slots, sizeof(float)))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t row_elements = is_q4(*idx) ? q4_row_bytes(dim_sz) : dim_sz;
        const size_t element_size = is_q4(*idx) ? sizeof(uint8_t) :
                                    (is_q8(*idx) ? sizeof(int8_t) : sizeof(float));
        if (!can_address_array(new_slots, row_elements) ||
            !can_address_array(new_slots * row_elements, element_size)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(vectors, value_count)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        size_t max_slots = std::min(kMaxIndexLen, std::numeric_limits<size_t>::max() / sizeof(uint64_t));
        max_slots = std::min(max_slots, std::numeric_limits<size_t>::max() / row_elements / element_size);
        if (is_quantized(*idx)) {
            max_slots = std::min(max_slots, std::numeric_limits<size_t>::max() / sizeof(float));
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
        if (is_q4(*idx)) {
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

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            const float * src = vectors + static_cast<size_t>(i) * dim_sz;
            if (is_q4(*idx)) {
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

        if (is_q4(*idx)) {
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

int ggml_vec_index_add_logged(
    ggml_vec_index_t *,
    const float *,
    int,
    const uint64_t *,
    const char *) {
    return GGML_VEC_INDEX_E_INVALID_ARG;
}

int ggml_vec_index_remove_logged(
    ggml_vec_index_t *,
    uint64_t,
    const char *) {
    return GGML_VEC_INDEX_E_INVALID_ARG;
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr || !is_valid_id(id)) {
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

void ggml_vec_index_prepare(ggml_vec_index_t *) {}

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
