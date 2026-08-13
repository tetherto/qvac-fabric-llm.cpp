#include "xdna-ops.h"
#include "xdna-runtime.h"
#include "xdna-profile.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --- GEMM internals (static) ------------------------------------------------

// K is split into blocks of at most GEMM_K_MAX: a per-core K-loop count above
// 64 (K > 1024 with tile_k=16) wedges the shared hw_context. GEMM_N_MAX is a
// practical cap on N: wide projections (e.g. the vocabulary output) need a
// huge B transpose and run too long on the NPU.
static const int GEMM_K_MAX = 1024;
static const int GEMM_N_MAX = 16384;

// Per-call breakdown is printed when GGML_XDNA_PROFILING=1.
static bool ggml_xdna_profiling_enabled(void) {
    static const bool en = []() {
        const char * v = getenv("GGML_XDNA_PROFILING");
        return v != nullptr && atoi(v) >= 1;
    }();
    return en;
}

// Return the device BO holding the packed [K_pad x N] bf16 weight for `src0`,
// built and cached on first use. K_pad rounds K up to the GEMM block, so every
// K-block reads a valid slice. Weights are immutable, so the transpose + bf16
// conversion runs once.
static xdna_buffer * gemm_weight_bo(xdna_ops * ops, const struct ggml_tensor * src0, int K, int N) {
    const xdna_ops::weight_key key = { src0->data, K, N };
    {
        std::lock_guard<std::mutex> lock(ops->weight_mutex);
        auto it = ops->weight_bo.find(key);
        if (it != ops->weight_bo.end()) {
            return it->second;
        }
    }

    const int K_pad = (K + GEMM_K_MAX - 1) / GEMM_K_MAX * GEMM_K_MAX;
    xdna_buffer * bo_w = xdna_buffer_alloc(ops->pool->device, (size_t) K_pad * N * sizeof(ggml_bf16_t));
    if (!bo_w) {
        GGML_LOG_ERROR("%s: failed to allocate weight BO\n", "xdna-ops");
        return nullptr;
    }

    ggml_bf16_t * w = (ggml_bf16_t *) bo_w->bo.map();
    std::memset(w, 0, (size_t) K_pad * N * sizeof(ggml_bf16_t));
    if (src0->type == GGML_TYPE_BF16) {
        const auto * d = (const ggml_bf16_t *) src0->data;
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                w[(size_t) k * N + n] = d[(size_t) n * K + k];
            }
        }
    } else { // GGML_TYPE_F16
        std::vector<float> row(K);
        for (int n = 0; n < N; n++) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K, row.data(), K);
            for (int k = 0; k < K; k++) {
                ggml_fp32_to_bf16_row(&row[k], &w[(size_t) k * N + n], 1);
            }
        }
    }
    xdna_buffer_sync_to_device(bo_w);

    std::lock_guard<std::mutex> lock(ops->weight_mutex);
    ops->weight_bo.emplace(key, bo_w);
    return bo_w;
}

// MUL_MAT follows ggml's convention: src0 = weights [K, N] (stored as N rows
// of K), src1 = activations [K, M] (stored as M rows of K), so dst[M, N] =
// src1^T @ src0. src1's memory feeds the kernel directly as A [M, K]; src0's
// memory is transposed to B [K, N]. The stream is built per shape via
// xdna_gemm_seq_build, so only A's rows are padded to the baked M block.
//
// K is split into blocks of at most GEMM_K_MAX: a per-core K-loop count above
// 64 (K > 1024 with tile_k=16) wedges the shared hw_context, so wide-K ops run
// as several sub-GEMMs accumulated on the host.
//
// Submission is batched: the kernel is started without waiting and the run is
// queued in ops->pending; xdna_ops_finalize() waits all runs of the layer and
// completes the op (readback, accumulation, dst).
static bool gemm_compute(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];

    // Guaranteed to be supported: compute is only reached for ops accepted by
    // xdna_ops_supported.
    const xdna_gemm_tiles & tiles = ops->gemm_tiles;
    const int Mk = tiles.M;   // 32-row block

    const bool prof_en = ggml_xdna_profiling_enabled();
    xdna_mul_mat_profile prof;
    const xdna_timer t_op;

    // B is a persistent BO with the packed [K_pad x N] weights; each K-block's
    // stream points into it via its K offset, so no per-call weight copy.
    xdna_buffer * bo_w = gemm_weight_bo(ops, src0, K, N);
    if (!bo_w) {
        return false;
    }

    // All blocks use the full GEMM_K_MAX kernel: a partial K block (K < 1024)
    // after a run of full blocks wedges the shared hw_context, so the last
    // block is zero-padded in K and runs the same K1024 kernel as the rest.
    xdna_ops::pending_op op;
    op.node = node;
    op.M = M;

    for (int m0 = 0; m0 < M; m0 += Mk) {
        const int mc = std::min(Mk, M - m0);
        xdna_ops::pending_m_block mb;
        mb.m0 = m0;
        mb.c_acc.assign((size_t) Mk * N, 0.0f);

        for (int k0 = 0; k0 < K; k0 += GEMM_K_MAX) {
            const int Kb = std::min(GEMM_K_MAX, K - k0);
            const uint32_t b_offset = (uint32_t) k0 * N * 2;

            // Build the instruction stream for (GEMM_K_MAX, N) with this block's
            // B offset and cache the bound kernel under the shape key.
            xdna_seq seq;
            if (!xdna_gemm_seq_build(&seq, &tiles, Mk, GEMM_K_MAX, N, b_offset)) {
                GGML_LOG_ERROR("%s: failed to build GEMM stream M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
                return false;
            }
            std::vector<uint32_t> insts = xdna_seq_build(&seq);

            char name[64];
            snprintf(name, sizeof(name), "gemm_K%d_N%d_b%d", GEMM_K_MAX, N, k0);
            xdna_kernel * kern = xdna_kernel_pool_get_built(ops->pool, name, ops->gemm_xclbin.c_str(),
                                                            insts.data(), insts.size());
            if (!kern) {
                GGML_LOG_ERROR("%s: failed to load GEMM kernel %s\n", "xdna-ops", name);
                return false;
            }

            xdna_ops::pending_run pr;
            pr.bo_a = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
            pr.bo_c = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * N * sizeof(float));
            if (!pr.bo_a || !pr.bo_c) {
                if (pr.bo_a) xdna_kernel_pool_release_buffer(ops->pool, pr.bo_a);
                if (pr.bo_c) xdna_kernel_pool_release_buffer(ops->pool, pr.bo_c);
                GGML_LOG_ERROR("%s: failed to allocate GEMM buffers\n", "xdna-ops");
                return false;
            }
            pr.N = N;
            pr.c_buf.assign((size_t) Mk * N, 0.0f);

            // A: src1[m0:m0+mc, k0:k0+Kb] f32 -> [Mk x GEMM_K_MAX] bf16, M/K
            // padded (a partial block adds nothing).
            {
                const xdna_timer t;
                ggml_bf16_t * a_map = (ggml_bf16_t *) pr.bo_a->bo.map();
                std::memset(a_map, 0, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
                for (int m = 0; m < mc; m++) {
                    ggml_fp32_to_bf16_row((const float *) src1->data + (size_t) (m0 + m) * K + k0,
                                          a_map + (size_t) m * GEMM_K_MAX, Kb);
                }
                if (prof_en) prof.a_pack += t.ms();
            }

            {
                const xdna_timer t;
                xdna_buffer_sync_to_device(pr.bo_a);
                if (prof_en) prof.sync += t.ms();
            }

            xdna_buffer * args[3] = { pr.bo_a, bo_w, pr.bo_c };
            {
                const xdna_timer t;
                pr.run = xdna_kernel_run_start(kern, args, 3);
                if (!pr.run) {
                    GGML_LOG_ERROR("%s: GEMM submit failed M=%d K=%d N=%d kernel=gemm_K%d_N%d_b%d\n",
                                   "xdna-ops", M, K, N, GEMM_K_MAX, N, k0);
                    xdna_kernel_pool_release_buffer(ops->pool, pr.bo_a);
                    xdna_kernel_pool_release_buffer(ops->pool, pr.bo_c);
                    return false;
                }
                if (prof_en) prof.run += t.ms();
            }
            mb.runs.push_back(std::move(pr));
            prof.n_blocks++;
        }
        op.m_blocks.push_back(std::move(mb));
    }

    ops->pending.push_back(std::move(op));

    if (prof_en) {
        prof.total = t_op.ms();
        fprintf(stderr,
                "xdna-profile: MUL_MAT %s M=%d K=%d N=%d blocks=%d "
                "total=%.3fms apack=%.3f sync=%.3f submit=%.3f\n",
                node->name, M, K, N, prof.n_blocks,
                prof.total, prof.a_pack, prof.sync, prof.run);
    }

    GGML_LOG_INFO("%s: MUL_MAT %s M=%d K=%d N=%d blocks=%d\n",
                  "xdna-ops", node->name, M, K, N, prof.n_blocks);
    return true;
}

bool xdna_ops_finalize(xdna_ops * ops) {
    if (ops->pending.empty()) {
        return true;
    }

    const bool prof_en = ggml_xdna_profiling_enabled();
    const xdna_timer t_all;

    // Wait all runs first so all kernels of the layer complete together.
    for (auto & op : ops->pending) {
        for (auto & mb : op.m_blocks) {
            for (auto & pr : mb.runs) {
                if (!xdna_run_wait(pr.run)) {
                    GGML_LOG_ERROR("%s: finalize: kernel wait failed\n", "xdna-ops");
                    return false;
                }
            }
        }
    }

    // Read C, accumulate K-blocks per M-block, write dst, release buffers.
    for (auto & op : ops->pending) {
        const int Mk = ops->gemm_tiles.M;
        for (auto & mb : op.m_blocks) {
            for (auto & pr : mb.runs) {
                xdna_buffer_read(pr.bo_c, pr.c_buf.data(), (size_t) Mk * pr.N * sizeof(float));
                for (size_t i = 0; i < (size_t) Mk * pr.N; i++) {
                    mb.c_acc[i] += pr.c_buf[i];
                }
            }
            const int mc = std::min(Mk, op.M - mb.m0);
            for (int m = 0; m < mc; m++) {
                std::memcpy((char *) op.node->data + (size_t) (mb.m0 + m) * op.node->nb[1],
                            mb.c_acc.data() + (size_t) m * mb.runs.front().N,
                            (size_t) mb.runs.front().N * sizeof(float));
            }
            for (auto & pr : mb.runs) {
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_c);
            }
        }
    }

    if (prof_en) {
        fprintf(stderr, "xdna-profile: FINALIZE ops=%zu total=%.3fms\n",
                ops->pending.size(), t_all.ms());
    }

    ops->pending.clear();
    return true;
}

static bool gemm_supported(const xdna_ops * ops, const struct ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    if (ops->gemm_xclbin.empty()) {
        return false;
    }
    if (M <= 0) {
        return false;
    }
    // The stream is always built for the 32-row block, so the geometry check
    // is on the block M; op M is tiled into blocks.
    if (!xdna_gemm_seq_supported(&ops->gemm_tiles, ops->gemm_tiles.M, K, N)) {
        return false;
    }
    // Practical cap on N: wide projections (e.g. the vocabulary output) need a
    // huge B transpose and run too long on the NPU; those stay on the CPU.
    if (N > GEMM_N_MAX) {
        return false;
    }
    if (src0->ne[0] != K) {
        return false;
    }
    if (src0->ne[2] * src0->ne[3] != 1 || src1->ne[2] * src1->ne[3] != 1) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    return true;
}

// --- public API -------------------------------------------------------------

void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool) {
    ops->pool = pool;
    ops->gemm_tiles = xdna_gemm_tiles{};
    ops->gemm_xclbin.clear();
    for (const std::string & name : pool->names) {
        if (name.rfind("gemm_bf16_f32_", 0) == 0) {
            ops->gemm_xclbin = name;
            break;
        }
    }
}

bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return gemm_supported(ops, op);
        default:
            return false;
    }
}

bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return gemm_compute(ops, node);
        default:
            GGML_LOG_ERROR("%s: unsupported op %d in XDNA graph\n",
                           "xdna-ops", (int) node->op);
            return false;
    }
}
