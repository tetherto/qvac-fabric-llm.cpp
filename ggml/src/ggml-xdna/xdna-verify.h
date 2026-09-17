#pragma once

// Numerical verification of the NPU paths against a host reference
// (GGML_XDNA_VERIFY).
//
// The generic test-backend-ops shapes all fall outside the baked GEMM geometry
// (N must be a multiple of tile_n*n_cols, K of tile_k), so they give this
// backend no coverage at all. Instead every NPU result is recomputed on the
// host from the same inputs while the model runs, and the relative RMS is
// accumulated per op and printed at exit.
//
//   GGML_XDNA_VERIFY=1  accumulate, print the summary at exit
//   GGML_XDNA_VERIFY=2  additionally print every single op
//
// Reference and NPU differ legitimately by weight rounding and summation
// order: a few 1e-3 is the bf16 noise floor, percent-level means a real defect
// (e.g. a lossy weight re-quantization).

#include <cstddef>

// 0 when disabled. Read once from the environment.
int xdna_verify_level(void);

// Layer the fused decode path verifies (GGML_XDNA_VERIFY_LAYER, default 0).
// Checking the fused kernels needs an fp32 copy of that layer's weights, so
// only one layer is instrumented at a time.
int xdna_verify_layer(void);

// Fold one comparison into the statistics of `key` (a stable op label).
void xdna_verify_add(const char * key, double rel_rms);

// Relative RMS of `got` against `ref`, both `n` floats. Returns 0 when the
// reference is all zeros.
double xdna_verify_rel(const float * got, const float * ref, size_t n);
