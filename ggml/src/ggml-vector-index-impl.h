#pragma once

#include "ggml-vector-index.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ggml_vec_index_detail {

constexpr size_t   kTvimHeaderSize   = 16;
constexpr uint64_t kMaxSnapshotBytes = UINT64_C(1) << 32;

inline bool checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

inline bool checked_add_size(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool can_address_array(size_t count, size_t element_size) {
    size_t bytes = 0;
    return checked_mul_size(count, element_size, bytes);
}

inline bool expected_snapshot_size(size_t n, size_t dim, size_t & expected) {
    size_t values        = 0;
    size_t vector_bytes  = 0;
    size_t id_bytes      = 0;
    size_t payload_bytes = 0;
    return checked_mul_size(n, dim, values) && checked_mul_size(values, sizeof(float), vector_bytes) &&
           checked_mul_size(n, sizeof(uint64_t), id_bytes) && checked_add_size(vector_bytes, id_bytes, payload_bytes) &&
           checked_add_size(kTvimHeaderSize, payload_bytes, expected);
}

inline bool supported_snapshot_size(size_t n, size_t dim, size_t & expected) {
    return expected_snapshot_size(n, dim, expected) && expected <= kMaxSnapshotBytes;
}

inline int snapshot_write_preflight(size_t n, size_t dim, size_t data_size) {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<uint32_t>::max()) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    size_t expected_size = 0;
    if (!supported_snapshot_size(n, dim, expected_size)) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    size_t value_count = 0;
    if (!checked_mul_size(n, dim, value_count) || data_size != value_count) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

}  // namespace ggml_vec_index_detail
