#pragma once

// Lightweight per-op timing for the XDNA backend (GGML_XDNA_PROFILING=1).
// Uses ggml_time_us() for wall-clock timing.

#include "ggml.h"

#include <cstdint>

// GGML_XDNA_RUNNER_PROF=1: where a per-op NPU runner's wall clock actually
// goes. Neither GLUE_PROF (which only sees ops that run on the host) nor
// DESIGN_PROF (which only times dispatches) sees the packing and scattering
// these runners do around their dispatches - and twice that has turned out to
// be the larger cost, so time it explicitly.
bool xdna_runner_prof_on(void);
void xdna_runner_prof_add(const char * runner, const char * phase, int64_t us);

// Times a scope and files it under (runner, phase). Zero cost when off.
struct xdna_rp {
    const char * runner;
    const char * phase;
    int64_t      t0;

    xdna_rp(const char * r, const char * p)
        : runner(r), phase(p), t0(xdna_runner_prof_on() ? ggml_time_us() : 0) {}

    ~xdna_rp() {
        if (t0) {
            xdna_runner_prof_add(runner, phase, ggml_time_us() - t0);
        }
    }
};

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
    double seq      = 0;   // build the TXN stream for this op
    double d_scan   = 0;   // per-row activation scale scan over src1
    double acc_zero = 0;   // zero the per-block int32 accumulators
    double c_acc    = 0;   // accumulate read-back int32 C into acc
    double rescale  = 0;   // rescale acc and write f32 dst
    double wbo      = 0;   // weight BO lookup
    double pool     = 0;   // buffer pool acquire/release

    int    n_blocks = 0;   // K-blocks executed
    double total    = 0;   // whole MUL_MAT (submit phase only)
};
