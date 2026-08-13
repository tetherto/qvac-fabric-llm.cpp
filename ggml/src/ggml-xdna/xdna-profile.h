#pragma once

// Lightweight per-op timing for the XDNA backend (GGML_XDNA_PROFILING=1).
// Uses ggml_time_us() for wall-clock timing.

#include "ggml.h"

#include <cstdint>

struct xdna_timer {
    int64_t t0 = ggml_time_us();

    // Milliseconds since construction.
    double ms() const {
        return (double) (ggml_time_us() - t0) / 1000.0;
    }
};

// Per-call MUL_MAT timing breakdown, filled when profiling is enabled. The
// block phases are summed across all K-blocks.
struct xdna_mul_mat_profile {
    double b_copy    = 0;   // copy the packed-weight slice into the B BO
    double a_pack    = 0;   // pack A rows to bf16 (memset + convert)
    double sync      = 0;   // sync A/B BOs to the device
    double run       = 0;   // kernel submission + wait
    double c_read    = 0;   // read C BO back
    double accum     = 0;   // accumulate the K-block contribution
    double dst_copy  = 0;   // copy the result into dst

    int    n_blocks  = 0;   // K-blocks executed
    double total     = 0;   // whole MUL_MAT
};
