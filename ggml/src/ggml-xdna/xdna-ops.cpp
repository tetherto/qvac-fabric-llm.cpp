#include "xdna-ops.h"
#include "xdna-runtime.h"
#include "xdna-profile.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

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

static int xdna_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// Pick the geometry for an op: the prefill (M=64) geometry for ops wide
// enough to fill it, otherwise the decode (M=32) geometry. Both the tiles and
// the xclbin stem derive from the same condition, so they never disagree.
static const xdna_gemm_tiles & xdna_pick_tiles(const xdna_ops * ops, int M) {
    if (M >= ops->gemm_big_m_min && !ops->gemm_xclbin_prefill.empty()) {
        return ops->gemm_tiles_prefill;
    }
    return ops->gemm_tiles;
}

static const char * xdna_pick_xclbin(const xdna_ops * ops, int M) {
    if (M >= ops->gemm_big_m_min && !ops->gemm_xclbin_prefill.empty()) {
        return ops->gemm_xclbin_prefill.c_str();
    }
    return ops->gemm_xclbin_decode.c_str();
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
    } else if (src0->type == GGML_TYPE_F16) {
        std::vector<float> row(K);
        for (int n = 0; n < N; n++) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K, row.data(), K);
            for (int k = 0; k < K; k++) {
                ggml_fp32_to_bf16_row(&row[k], &w[(size_t) k * N + n], 1);
            }
        }
    } else if (src0->type == GGML_TYPE_Q4_K) {
        // Quantized weights are dequantized once at pack time and stored as
        // bf16, so the NPU kernel stays unchanged.
        std::vector<float> row(K);
        const int nb = K / QK_K;
        for (int n = 0; n < N; n++) {
            dequantize_row_q4_K((const block_q4_K *) src0->data + (size_t) n * nb, row.data(), K);
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
// Submission is pipelined: two A/C buffer banks are ping-ponged. The run for
// block (m0, k0) is submitted without waiting, then the previous block's run
// is waited and its C read back while the NPU processes the new one, so the
// C readback overlaps the GEMM. The last few runs are finished in
// xdna_ops_finalize().
static bool gemm_compute(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];

    // Guaranteed to be supported: compute is only reached for ops accepted by
    // xdna_ops_supported.
    const xdna_gemm_tiles & tiles = xdna_pick_tiles(ops, M);
    const int Mk = tiles.M;   // baked M block of the selected geometry

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
    op.N = N;
    op.Mk = Mk;
    op.m_off = 0;
    for (int m0 = 0; m0 < M; m0 += Mk) {
        xdna_ops::pending_m_block mb;
        mb.m0 = m0;
        mb.c_acc.assign((size_t) Mk * N, 0.0f);
        op.m_blocks.push_back(std::move(mb));
    }

    // Ping-pong A/C banks. Bank b is reused only after its in-flight run has
    // been waited and its C read back, so the readback overlaps the NPU run
    // that follows.
    xdna_ops::pending_run banks[2];
    int bank = 0;
    int pending_bank = -1;   // bank with a submitted, un-waited run

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
        snprintf(name, sizeof(name), "gemm_K%d_N%d_b%d_m%d", GEMM_K_MAX, N, k0, Mk);
        xdna_kernel * kern = xdna_kernel_pool_get_built(ops->pool, name, xdna_pick_xclbin(ops, M),
                                                        insts.data(), insts.size());
        if (!kern) {
            GGML_LOG_ERROR("%s: failed to load GEMM kernel %s\n", "xdna-ops", name);
            return false;
        }

        for (auto & mb : op.m_blocks) {
            const int mc = std::min(Mk, M - mb.m0);

            // Pack A into the current bank and submit. The NPU starts on this
            // run right away; the previous bank's run (already queued ahead of
            // it) is then waited and read back while this one runs.
            xdna_ops::pending_run & pr = banks[bank];
            if (pr.bo_a == nullptr) {
                pr.bo_a = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
                pr.bo_c = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * N * sizeof(float));
                if (!pr.bo_a || !pr.bo_c) {
                    GGML_LOG_ERROR("%s: failed to allocate GEMM buffers\n", "xdna-ops");
                    return false;
                }
                pr.c_buf.assign((size_t) Mk * N, 0.0f);
                pr.N = N;
            }
            pr.mb_idx = (int) (&mb - op.m_blocks.data());
            {
                const xdna_timer t;
                ggml_bf16_t * a_map = (ggml_bf16_t *) pr.bo_a->bo.map();
                std::memset(a_map, 0, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
                for (int m = 0; m < mc; m++) {
                    ggml_fp32_to_bf16_row((const float *) src1->data + (size_t) (mb.m0 + m) * K + k0,
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
                    return false;
                }
                if (prof_en) prof.run += t.ms();
            }

            // The NPU is now working on `bank`; wait + read back the previous
            // bank's run (queued ahead of this one) in parallel with it.
            if (pending_bank >= 0) {
                xdna_ops::pending_run & prev = banks[pending_bank];
                {
                    const xdna_timer t;
                    if (!xdna_run_wait(prev.run)) {
                        GGML_LOG_ERROR("%s: GEMM wait failed M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
                        return false;
                    }
                    if (prof_en) prof.wait += t.ms();
                }
                {
                    const xdna_timer t;
                    // Read back only the valid rows: rows [mc, Mk) of a padded
                    // block are zero (the padded A rows are zeroed), so the
                    // prefix is exact and the readback drops to mc*N elements.
                    const size_t p_elems = (size_t) std::min(Mk, op.M - op.m_blocks[prev.mb_idx].m0) * prev.N;
                    xdna_buffer_read(prev.bo_c, prev.c_buf.data(), p_elems * sizeof(float));
                    if (prof_en) prof.read += t.ms();
                }
                const size_t p_elems = (size_t) std::min(Mk, op.M - op.m_blocks[prev.mb_idx].m0) * prev.N;
                for (size_t i = 0; i < p_elems; i++) {
                    op.m_blocks[prev.mb_idx].c_acc[i] += prev.c_buf[i];
                }
                xdna_kernel_pool_release_buffer(ops->pool, prev.bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, prev.bo_c);
                banks[pending_bank].bo_a = nullptr;
                banks[pending_bank].bo_c = nullptr;
            }

            pending_bank = bank;
            bank = 1 - bank;
            prof.n_blocks++;
        }
    }

    // Flush the trailing banks into the op so finalize waits+reads them.
    if (pending_bank >= 0) {
        op.m_blocks[banks[pending_bank].mb_idx].runs.push_back(std::move(banks[pending_bank]));
        banks[pending_bank] = xdna_ops::pending_run();
        pending_bank = -1;
    }

    ops->pending.push_back(std::move(op));

    if (prof_en) {
        prof.total = t_op.ms();
        fprintf(stderr,
                "xdna-profile: MUL_MAT %s M=%d K=%d N=%d blocks=%d "
                "total=%.3fms apack=%.3f sync=%.3f submit=%.3f wait=%.3f read=%.3f\n",
                node->name, M, K, N, prof.n_blocks,
                prof.total, prof.a_pack, prof.sync, prof.run, prof.wait, prof.read);
    }

    return true;
}

bool xdna_ops_finalize(xdna_ops * ops) {
    if (ops->pending.empty()) {
        return true;
    }

    const bool prof_en = ggml_xdna_profiling_enabled();
    const xdna_timer t_all;

    // Wait all runs first so all kernels of the layer complete together.
    const xdna_timer t_wait;
    for (auto & op : ops->pending) {
        for (auto & mb : op.m_blocks) {
            for (auto & pr : mb.runs) {
                if (!xdna_run_wait(pr.run)) {
                    GGML_LOG_ERROR("%s: finalize: kernel wait failed\n", "xdna-ops");
                    for (auto & fop : ops->pending) {
                        for (auto & fmb : fop.m_blocks) {
                            for (auto & fpr : fmb.runs) {
                                xdna_kernel_pool_release_buffer(ops->pool, fpr.bo_a);
                                xdna_kernel_pool_release_buffer(ops->pool, fpr.bo_c);
                            }
                        }
                    }
                    ops->pending.clear();
                    return false;
                }
            }
        }
    }
    const double ms_wait = t_wait.ms();

    // Read C, accumulate K-blocks per M-block, write dst, release buffers.
    // Most runs are already waited/read/accumulated inside gemm_compute's
    // pipeline; only the trailing banks land here. m_blocks without runs have
    // their c_acc fully accumulated already, so just scatter them.
    const xdna_timer t_read;
    for (auto & op : ops->pending) {
        const int Mk = op.Mk;   // per-op geometry M block
        for (auto & mb : op.m_blocks) {
            // Trailing runs are read back here; read only the valid rows.
            const size_t mc_elems = (size_t) std::min(Mk, op.M - mb.m0) * op.N;
            for (auto & pr : mb.runs) {
                xdna_buffer_read(pr.bo_c, pr.c_buf.data(), mc_elems * sizeof(float));
                for (size_t i = 0; i < mc_elems; i++) {
                    mb.c_acc[i] += pr.c_buf[i];
                }
            }
            const int mc = std::min(Mk, op.M - mb.m0);
            for (int m = 0; m < mc; m++) {
                std::memcpy((char *) op.node->data + (size_t) (op.m_off + mb.m0 + m) * op.node->nb[1],
                            mb.c_acc.data() + (size_t) m * op.N,
                            (size_t) op.N * sizeof(float));
            }
            for (auto & pr : mb.runs) {
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_c);
            }
        }
    }

    if (prof_en) {
        fprintf(stderr, "xdna-profile: FINALIZE ops=%zu total=%.3fms wait=%.3f read=%.3f\n",
                ops->pending.size(), t_all.ms(), ms_wait, t_read.ms());
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

    if (ops->gemm_xclbin_decode.empty()) {
        return false;
    }
    if (M <= 0) {
        return false;
    }
    // The picked geometry falls back to the next smaller artifact; the op is
    // rejected only when the decode kernel is absent.
    if (M >= ops->gemm_big_m_min && ops->gemm_xclbin_prefill.empty()) {
        return false;
    }
    const xdna_gemm_tiles & tiles = xdna_pick_tiles(ops, M);
    // The stream is always built for the baked M block, so the geometry check
    // is on the block M; op M is tiled into blocks.
    if (!xdna_gemm_seq_supported(&tiles, tiles.M, K, N)) {
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
    if (src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_Q4_K) {
        return false;
    }
    // Q4_K rows are packed in 256-element blocks (QK_K); dequantization needs
    // whole blocks.
    if (src0->type == GGML_TYPE_Q4_K && K % QK_K != 0) {
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
    ops->gemm_tiles_prefill = xdna_gemm_tiles{};
    ops->gemm_tiles_prefill.M = GGML_XDNA_GEMM_M_BIG;
    ops->gemm_tiles_prefill.tile_m = GGML_XDNA_TILE_M_BIG;
    ops->gemm_tiles_prefill.rtp_base = XDNA_RTP_BASE_TILE_M16;
    ops->gemm_xclbin_decode.clear();
    ops->gemm_xclbin_prefill.clear();
    ops->gemm_big_m_min = xdna_env_int("GGML_XDNA_GEMM_BIG_M_MIN", 64);

    // Stems are gemm_bf16_f32_M%d_K%d_N%d_c%d (see CMakeLists).
    for (const std::string & name : pool->names) {
        int M = 0, K = 0, N = 0, C = 0;
        if (sscanf(name.c_str(), "gemm_bf16_f32_M%d_K%d_N%d_c%d", &M, &K, &N, &C) != 4) {
            continue;
        }
        if (M == GGML_XDNA_GEMM_M_BIG) {
            ops->gemm_xclbin_prefill = name;
        } else if (M == GGML_XDNA_GEMM_M) {
            ops->gemm_xclbin_decode = name;
        }
    }

    GGML_LOG_INFO("%s: GEMM geometries: decode=%s prefill=%s (big-M threshold %d)\n", "xdna-ops",
                  ops->gemm_xclbin_decode.c_str(), ops->gemm_xclbin_prefill.c_str(),
                  ops->gemm_big_m_min);
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
