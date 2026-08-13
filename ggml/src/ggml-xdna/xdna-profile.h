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
    double a_pack   = 0;   // pack A rows to bf16 (memset + convert)
    double sync     = 0;   // sync A BO to the device
    double run      = 0;   // kernel submission (start without wait)
    double wait     = 0;   // wait for the previous bank's run (NPU time)
    double read     = 0;   // read back the previous bank's C

    int    n_blocks = 0;   // K-blocks executed
    double total    = 0;   // whole MUL_MAT (submit phase only)
};
