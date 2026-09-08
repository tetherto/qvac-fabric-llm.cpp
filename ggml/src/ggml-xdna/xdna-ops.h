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

    // Two GEMM geometries: decode (M=32, native bf16 r=4, exact) and prefill
    // (M=64, bfp16-emulated r=8, 2x mmul). Ops with M >= gemm_big_m_min use
    // the prefill geometry; everything else the decode one.
    xdna_gemm_tiles gemm_tiles;          // decode geometry (default)
    xdna_gemm_tiles gemm_tiles_prefill;  // prefill geometry (M=64, rtp 0xd000)
    std::string gemm_xclbin_decode;      // discovered M32 xclbin stem
    std::string gemm_xclbin_prefill;     // discovered M64 xclbin stem
    int gemm_big_m_min = 64;             // M >= this -> prefill geometry

    // Packed-weight cache: tensor data pointer + K + N -> device BO holding the
    // transposed [K_pad x N] bf16 weights (K_pad = K rounded up to the GEMM
    // block). Weights are immutable, so the pack (transpose + bf16 conversion)
    // is done once per tensor and reused; kernels point into it via B offsets.
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

    std::unordered_map<weight_key, xdna_buffer *, weight_hash> weight_bo;
    std::mutex weight_mutex;

    // Batching: MUL_MAT ops are submitted (started without wait) and collected
    // here; xdna_ops_finalize() waits them all, reads C back, and writes dst.
    // Each op is tiled in M (the baked M block) and K (GEMM_K_MAX blocks): a
    // pending_run is one (M-block, K-block) submission.
    struct pending_run {
        xrt::run       run;
        xdna_buffer *  bo_c = nullptr;   // held until readback
        xdna_buffer *  bo_a = nullptr;   // released after readback
        std::vector<float> c_buf;        // Mk x N readback target
        int N = 0;
        int mb_idx = 0;                  // index into pending_op::m_blocks
    };
    struct pending_m_block {
        int m0 = 0;                      // dst row offset of this M-block
        std::vector<float> c_acc;        // Mk x N accumulator
        std::vector<pending_run> runs;   // one per K-block
    };
    struct pending_op {
        struct ggml_tensor * node = nullptr;
        int Mk = 0;                      // M block of the geometry used (32 or 64)
        int m_off = 0;                   // first row of the range in the op
        int M = 0;                       // total rows of the op
        int N = 0;                       // total columns of the op
        std::vector<pending_m_block> m_blocks;
    };
    std::vector<pending_op> pending;
};

// Discover the GEMM xclbin and set up the fixed geometry.
void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool);

// True when `op` can be run on the NPU. Dispatches per-op (only GEMM so far).
bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op);

// Execute `node` on the NPU. Dispatches per-op; returns false on failure.
// With batching this only submits (starts) the kernels; call xdna_ops_finalize
// after all ops of a layer to wait and write results.
bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node);

// Wait all pending submissions, read C back, accumulate K-blocks, write dst,
// and release buffers. Safe to call when nothing is pending.
bool xdna_ops_finalize(xdna_ops * ops);
