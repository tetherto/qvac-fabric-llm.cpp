#pragma once

#include <stddef.h>

static inline size_t ggml_metal_free_memory(size_t working_set_size, size_t allocated_size) {
    return working_set_size > allocated_size ? working_set_size - allocated_size : 0;
}
