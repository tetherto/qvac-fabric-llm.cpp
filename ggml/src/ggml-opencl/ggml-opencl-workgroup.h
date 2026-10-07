#pragma once

#include <cstddef>

static inline bool ggml_opencl_should_use_wide_gemv(
        size_t wave_size,
        int simdgroups,
        size_t kernel_max_workgroup_size,
        int m,
        int max_m,
        int k_blocks) {
    if (simdgroups <= 0) {
        return false;
    }

    const size_t group_count = (size_t) simdgroups;
    const bool workgroup_fits =
        group_count <= kernel_max_workgroup_size &&
        wave_size <= kernel_max_workgroup_size / group_count;

    return workgroup_fits &&
           m <= max_m &&
           k_blocks >= simdgroups;
}
