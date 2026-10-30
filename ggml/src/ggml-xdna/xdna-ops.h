#pragma once

// Operator-specific dispatch for the XDNA backend, kept out of ggml-xdna.cpp
// so backend scaffolding and per-op logic stay separate. Only GEMM is
// implemented so far; other ops get their own helpers here.

#include "ggml.h"
#include "xdna-gemv.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-types.h"

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

// Operator-specific state. Data only; the API lives below as C-style
// functions.
struct xdna_ops {
    xdna_kernel_pool * pool = nullptr;

    // Isolation mode (set by ggml-xdna.cpp while the fused decode layer path is
    // active): the per-op NPU kernels are off for the single-token decode
    // (M == ne[1] == 1), whose projections feed the fused layer from
    // host-visible buffers. Prefill batches (M > 1) never touch the fused
    // layer, so their per-op kernels stay enabled.
    bool isolation  = false;
    // The decode GEMV builds its geometry on the merged layer artifact, so a
    // projection outside the fused layers runs on the hardware context that is
    // already resident instead of reconfiguring the array for its own xclbin.
    bool fused_gemv = false;

    // Two GEMM families. Decode: M=32, native bf16 r=4, exact - used for ops
    // with M < GEMM_M_BIG_MIN. Prefill: one or more baked M blocks (the M64
    // base; larger blocks when their artifacts are built), bfp16-emulated r=8
    // (2x mmul), selected per op as the block that minimizes the weighted
    // pad/weight-stream cost. Quantized (Q4_K/Q5_K/Q6_K) prefill weights
    // additionally route to the native int8 GEMM (1 B/val) when an int8
    // artifact for the chosen block is present.
    xdna_gemm_tiles                      gemm_tiles;          // decode geometry (default)
    std::string                          gemm_xclbin_decode;  // discovered M32 xclbin stem
    // Prefill blocks are discovered from the pool and kept sorted descending.
    // The maps give the bf16 / native-int8 xclbin stem per baked M block.
    std::vector<int>                     pref_blocks;  // baked M blocks, descending
    std::unordered_map<int, std::string> pref_xclbin;  // block -> bf16 stem
    std::unordered_map<int, std::string> i8_xclbin;    // block -> int8 stem

    // Packed-weight cache: tensor data pointer + K + N -> device BO holding the
    // transposed [K_pad x N] bf16 weights (K_pad = K rounded up to a whole
    // number of GEMM_K_RUN runs). Weights are immutable, so the pack
    // (transpose + bf16 conversion) is done once per tensor and reused;
    // kernels point into it via B offsets. Projections run as one GEMM are
    // packed side by side along N; `tag` tells their set from the first
    // weight's alone (0 for one weight).
    struct weight_key {
        const void * data = nullptr;
        int          K    = 0;
        int          N    = 0;
        size_t       tag  = 0;

        bool operator==(const weight_key & o) const { return data == o.data && K == o.K && N == o.N && tag == o.tag; }
    };

    struct weight_hash {
        size_t operator()(const weight_key & k) const {
            size_t h = std::hash<const void *>{}(k.data);
            h        = h * 31 + (size_t) k.K;
            h        = h * 31 + (size_t) k.N;
            h        = h * 31 + k.tag;
            return h;
        }
    };

    std::unordered_map<weight_key, xdna_buffer *, weight_hash> weight_bo;
    std::mutex                                                 weight_mutex;

    // Native int8 weight cache for quantized (Q4_K/Q5_K/Q6_K) prefill: packed
    // [K_pad x N] int8 codes (1 B/val vs 2 for the bf16 dequant stream) plus
    // per-column scales, keyed like the gemm weights.
    struct int8_wbo {
        xdna_buffer *      bo = nullptr;  // [K_pad x N] int8 codes
        std::vector<float> dw;            // per-column scale (N)
        int                K = 0;
        int                N = 0;
    };

    std::unordered_map<weight_key, int8_wbo *, weight_hash> int8_wbo_map;
    std::mutex                                              int8_mutex;

    // Batching: MUL_MAT ops are submitted (started without wait) and collected
    // here; xdna_ops_finalize() waits them all, reads C back, and writes dst.
    // Each op is tiled in M (the baked M block) and K (GEMM_K_RUN runs): a
    // pending_run is one (M-block, K-run) submission.
    struct pending_run {
        xrt::run      run;
        xdna_buffer * bo_c       = nullptr;  // held until readback
        xdna_buffer * bo_a       = nullptr;  // released after readback
        int           mb_idx     = 0;        // index into pending_op::m_blocks
        bool          accumulate = false;    // adds to dst: not the op's first K run
    };

    struct pending_m_block {
        int                      m0 = 0;  // dst row offset of this M-block
        std::vector<pending_run> runs;    // one per K run
    };

    // Columns [n0, n0 + n) of an op's C are the rows of node's output
    struct pending_out {
        struct ggml_tensor * node = nullptr;
        int                  n0   = 0;
        int                  n    = 0;
    };

    struct pending_op {
        std::vector<pending_out>     outs;       // one per projection of the op
        int                          Mk    = 0;  // M block of the geometry used (32 or 64)
        int                          m_off = 0;  // first row of the range in the op
        int                          M     = 0;  // total rows of the op
        int                          N     = 0;  // total columns of the op
        std::vector<pending_m_block> m_blocks;
    };

    std::vector<pending_op> pending;

    // The bound kernel of each GEMM stream by shape, so a repeated shape does
    // not build its stream again.
    std::unordered_map<std::string, xdna_kernel *> gemm_kernel;
    std::mutex                                     gemm_kernel_mutex;

    // Decode GEMV runners, keyed by the first weight of the dispatch. The
    // packed weights live in the runner's device buffer for the process
    // lifetime, so the pack cost is paid once.
    std::unordered_map<const void *, xdna_gemv *> gemv;
    std::mutex                                    gemv_mutex;

    // Projections that share an activation run as one dispatch: their weights
    // are concatenated along N, so a layer's gate and up, or its q, k and v,
    // cost one kernel launch instead of two or three. A token is 168 launches
    // otherwise, and a launch is ~70 us against ~13 ms of weight streaming for
    // the whole token.
    struct gemv_group {
        std::vector<struct ggml_tensor *> nodes;
        xdna_gemv_geom                    geom;
        // Projections sharing an activation are grouped however far apart they
        // sit in the graph, so the dispatch cannot write their outputs where it
        // runs - ggml's allocator may still be using that memory for something
        // live. It runs at the first member and keeps the result here; each
        // member copies its own slice when the graph reaches it.
        std::vector<float>                buf;
        std::vector<int64_t>              off;  // each node's first output value in `buf`
        bool                              ran = false;
    };

    std::vector<gemv_group>                             gemv_groups;
    std::unordered_map<const struct ggml_tensor *, int> gemv_group_of;

    // Projections of the bf16 GEMM that the host runs in this graph, because
    // the host is quicker than the array plus the hardware-context switches
    // into and out of the bf16 design (xdna_ops_plan_gemm).
    std::unordered_set<const struct ggml_tensor *> gemm_host;
};

// Find the projections of `cgraph` that share an activation and record them as
// single dispatches. Call once per graph, before computing it.
// Decide which projections share a dispatch. `skip` names nodes another route
// has already claimed - the fused recurrent layers' bodies and the inputs they
// snapshot - which must not be absorbed into a GEMV group.
void xdna_ops_plan_gemv(xdna_ops *                                             ops,
                        const struct ggml_cgraph *                             cgraph,
                        const std::unordered_set<const struct ggml_tensor *> * skip);

// Decide which bf16-GEMM projections of `cgraph` run on the host. A projection
// between commands of another design (the pgemm ones) pays a hardware-context
// switch in and another out, and below a size the host does it in less than
// that. Call once per prefill graph, after xdna_ops_plan_gemv.
void xdna_ops_plan_gemm(xdna_ops * ops, const struct ggml_cgraph * cgraph);

// Discover the GEMM xclbin and set up the fixed geometry.
void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool);

// Free everything packed from a model's weights: the GEMM weight BOs (bf16
// and int8) and the decode GEMV runners. They are keyed by the address of the
// weights, which a later model can reuse, so they must not outlive the model
// they were packed from. The next use packs again.
void xdna_ops_release_weights(xdna_ops * ops);

// True when `op` can be run on the NPU. Dispatches per-op (only GEMM so far).
bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op);

// True when `mm` (n >= 2 projections) can run as one bf16 GEMM: they read one
// activation, every one is on the bf16 route, and their columns together fit
// the GEMM. Their weights are packed side by side, and each output goes to its
// own node.
bool xdna_ops_gemm_group_supported(const xdna_ops * ops, struct ggml_tensor * const * mm, int n);

// Submit the projections of a supported group as one GEMM; xdna_ops_finalize
// writes every output. False on failure, with nothing pending.
bool xdna_ops_compute_group(xdna_ops * ops, struct ggml_tensor * const * mm, int n);

// Execute `node` on the NPU. Dispatches per-op; returns false on failure.
// With batching this only submits (starts) the kernels; call xdna_ops_finalize
// after all ops of a layer to wait and write results.
bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node);

// Wait all pending submissions, read C back, accumulate K-blocks, write dst,
// and release buffers. Safe to call when nothing is pending.
bool xdna_ops_finalize(xdna_ops * ops);
