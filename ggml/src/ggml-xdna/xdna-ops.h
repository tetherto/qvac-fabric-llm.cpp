#pragma once

// Operator-specific dispatch for the XDNA backend, kept out of ggml-xdna.cpp
// so backend scaffolding and per-op logic stay separate. Only GEMM is
// implemented so far; other ops get their own helpers here.

#include "xdna-types.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "ggml.h"

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

// Operator-specific state. Data only; the API lives below as C-style
// functions.
struct xdna_ops {
    xdna_kernel_pool * pool = nullptr;

    xdna_gemm_tiles gemm_tiles;     // GEMM geometry baked into the xclbin
    std::string     gemm_xclbin;    // discovered xclbin name (geometry provider)

    // Packed-weight cache: tensor data pointer + K + N -> transposed [K x N]
    // bf16. Weights are immutable for the model lifetime, so the pack
    // (transpose + bf16 conversion) is done once per tensor and reused.
    struct weight_key {
        const void * data = nullptr;
        int          K = 0;
        int          N = 0;

        bool operator==(const weight_key & o) const {
            return data == o.data && K == o.K && N == o.N;
        }
    };
    struct weight_hash {
        size_t operator()(const weight_key & k) const {
            size_t h = std::hash<const void *>{}(k.data);
            h = h * 31 + (size_t) k.K;
            h = h * 31 + (size_t) k.N;
            return h;
        }
    };

    std::unordered_map<weight_key, std::vector<ggml_bf16_t>, weight_hash> weight_packs;
    std::mutex weight_mutex;
};

// Discover the GEMM xclbin and set up the fixed geometry.
void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool);

// True when `op` can be run on the NPU. Dispatches per-op (only GEMM so far).
bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op);

// Execute `node` on the NPU. Dispatches per-op; returns false on failure.
bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node);
