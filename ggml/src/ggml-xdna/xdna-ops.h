#pragma once

// Operator-specific dispatch for the XDNA backend, kept out of ggml-xdna.cpp
// so backend scaffolding and per-op logic stay separate. Only GEMM is
// implemented so far; other ops get their own helpers here.

#include "xdna-types.h"

#include <map>
#include <string>

struct ggml_tensor;

// A GEMM kernel variant metadata, parsed from the pool's kernel names.
struct xdna_ops_gemm_variant {
    int         M, K, N;   // baked block dims
    int         n_cols;    // AIE columns the kernel was compiled for
    int         tile_n;    // per-core N tile: 16 for N < 256, else 32
    std::string name;      // pool name / artifact stem
};

// Operator-specific state. Data only; the API lives below as C-style
// functions.
struct xdna_ops {
    xdna_kernel_pool * pool = nullptr;

    // GEMM variants by kernel name, parsed once from pool->names.
    std::map<std::string, xdna_ops_gemm_variant> gemm_variants;
};

// Parse the GEMM-shaped names from the pool.
void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool);

// True when `op` can be run on the NPU. Dispatches per-op (only GEMM so far).
bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op);

// Execute `node` on the NPU. Dispatches per-op; returns false on failure.
bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node);
