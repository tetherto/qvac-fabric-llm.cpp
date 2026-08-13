#pragma once

// Operator-specific dispatch for the XDNA backend, kept out of ggml-xdna.cpp
// so backend scaffolding and per-op logic stay separate. Only GEMM is
// implemented so far; other ops get their own helpers here.

#include "xdna-types.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"

#include <map>
#include <string>

struct ggml_tensor;

// Operator-specific state. Data only; the API lives below as C-style
// functions.
struct xdna_ops {
    xdna_kernel_pool * pool = nullptr;

    xdna_gemm_tiles gemm_tiles;     // GEMM geometry baked into the xclbin
    std::string     gemm_xclbin;    // discovered xclbin name (geometry provider)
};

// Discover the GEMM xclbin and set up the fixed geometry.
void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool);

// True when `op` can be run on the NPU. Dispatches per-op (only GEMM so far).
bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op);

// Execute `node` on the NPU. Dispatches per-op; returns false on failure.
bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node);
