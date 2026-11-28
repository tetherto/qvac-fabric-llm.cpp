#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

static inline size_t ggml_vk_buffer_capacity(
        uint64_t max_buffer_size,
        uint64_t max_memory_allocation_size,
        uint64_t max_storage_buffer_range,
        bool     shader_64b_indexing,
        bool     buffer_device_address) {
    (void) max_storage_buffer_range;
    (void) shader_64b_indexing;
    (void) buffer_device_address;

    uint64_t capacity = std::min({
        max_buffer_size,
        max_memory_allocation_size,
        (uint64_t) SIZE_MAX,
    });

    return (size_t) capacity;
}
