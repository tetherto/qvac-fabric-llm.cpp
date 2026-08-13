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

// Return the packed [K x N] bf16 weight for `src0`, built and cached on first
// use. Weights are immutable, so the transpose + bf16 conversion runs once.
static const std::vector<ggml_bf16_t> & gemm_weight_pack(xdna_ops * ops,
                                                         const struct ggml_tensor * src0, int K, int N) {
    const xdna_ops::weight_key key = { src0->data, K, N };
    {
        std::lock_guard<std::mutex> lock(ops->weight_mutex);
        auto it = ops->weight_packs.find(key);
        if (it != ops->weight_packs.end()) {
            return it->second;
        }
    }

    std::vector<ggml_bf16_t> pack((size_t) K * N);
    if (src0->type == GGML_TYPE_BF16) {
        const auto * d = (const ggml_bf16_t *) src0->data;
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                pack[(size_t) k * N + n] = d[(size_t) n * K + k];
            }
        }
    } else { // GGML_TYPE_F16
        std::vector<float> row(K);
        for (int n = 0; n < N; n++) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K, row.data(), K);
            for (int k = 0; k < K; k++) {
                ggml_fp32_to_bf16_row(&row[k], &pack[(size_t) k * N + n], 1);
            }
        }
    }

    std::lock_guard<std::mutex> lock(ops->weight_mutex);
    return ops->weight_packs.emplace(key, std::move(pack)).first->second;
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
static bool gemm_compute(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    struct ggml_tensor * dst = node;

    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];

    // Guaranteed to be supported: compute is only reached for ops accepted by
    // xdna_ops_supported.
    const xdna_gemm_tiles & tiles = ops->gemm_tiles;
    const int Mk = tiles.M;

    const bool prof_en = ggml_xdna_profiling_enabled();
    xdna_mul_mat_profile prof;
    const xdna_timer t_op;

    std::vector<float> c_acc((size_t) Mk * N, 0.0f);

    // All blocks use the full GEMM_K_MAX kernel: a partial K block (K < 1024)
    // after a run of full blocks wedges the shared hw_context, so the last
    // block is zero-padded in K and runs the same K1024 kernel as the rest.
    xdna_kernel * kern = nullptr;
    for (int k0 = 0; k0 < K; k0 += GEMM_K_MAX) {
        const int Kb = std::min(GEMM_K_MAX, K - k0);

        // Build the instruction stream for (GEMM_K_MAX, N) and cache the bound
        // kernel under the shape key.
        if (!kern) {
            xdna_seq seq;
            if (!xdna_gemm_seq_build(&seq, &tiles, Mk, GEMM_K_MAX, N)) {
                GGML_LOG_ERROR("%s: failed to build GEMM stream M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
                return false;
            }
            std::vector<uint32_t> insts = xdna_seq_build(&seq);

            char name[64];
            snprintf(name, sizeof(name), "gemm_K%d_N%d", GEMM_K_MAX, N);
            kern = xdna_kernel_pool_get_built(ops->pool, name, ops->gemm_xclbin.c_str(),
                                              insts.data(), insts.size());
            if (!kern) {
                GGML_LOG_ERROR("%s: failed to load GEMM kernel %s\n", "xdna-ops", name);
                return false;
            }
        }

        xdna_buffer * bo_a = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
        xdna_buffer * bo_b = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) GEMM_K_MAX * N * sizeof(ggml_bf16_t));
        xdna_buffer * bo_c = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * N * sizeof(float));
        if (!bo_a || !bo_b || !bo_c) {
            if (bo_a) xdna_kernel_pool_release_buffer(ops->pool, bo_a);
            if (bo_b) xdna_kernel_pool_release_buffer(ops->pool, bo_b);
            if (bo_c) xdna_kernel_pool_release_buffer(ops->pool, bo_c);
            GGML_LOG_ERROR("%s: failed to allocate GEMM buffers\n", "xdna-ops");
            return false;
        }

        // The BOs are host-visible, so write straight into their mapped memory.
        // A's padding rows are zeroed (the pool may hand back a reused BO); B's
        // K-padding is zeroed too, so a partial K block adds nothing.

        // B: slice of the cached [K x N] pack -> kernel [GEMM_K_MAX x N] bf16.
        // Full blocks fill the whole region the kernel reads; partial blocks
        // are zero-padded in K.
        {
            const xdna_timer t;
            const std::vector<ggml_bf16_t> & wpack = gemm_weight_pack(ops, src0, K, N);
            ggml_bf16_t * b_map = (ggml_bf16_t *) bo_b->bo.map();
            if (Kb == GEMM_K_MAX) {
                std::memcpy(b_map, wpack.data() + (size_t) k0 * N,
                            (size_t) Kb * N * sizeof(ggml_bf16_t));
            } else {
                std::memset(b_map, 0, (size_t) GEMM_K_MAX * N * sizeof(ggml_bf16_t));
                std::memcpy(b_map, wpack.data() + (size_t) k0 * N,
                            (size_t) Kb * N * sizeof(ggml_bf16_t));
            }
            if (prof_en) prof.b_copy += t.ms();
        }

        // A: src1 [M rows of K] f32 -> [Mk x GEMM_K_MAX] bf16, K-slice + rows
        // padded, zero-padded in K past Kb.
        {
            const xdna_timer t;
            ggml_bf16_t * a_map = (ggml_bf16_t *) bo_a->bo.map();
            std::memset(a_map, 0, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
            for (int m = 0; m < M; m++) {
                ggml_fp32_to_bf16_row((const float *) src1->data + (size_t) m * K + k0,
                                      a_map + (size_t) m * GEMM_K_MAX, Kb);
            }
            if (prof_en) prof.a_pack += t.ms();
        }

        {
            const xdna_timer t;
            xdna_buffer_sync_to_device(bo_a);
            xdna_buffer_sync_to_device(bo_b);
            if (prof_en) prof.sync += t.ms();
        }

        std::vector<float> c_buf((size_t) Mk * N, 0);
        xdna_buffer * args[3] = { bo_a, bo_b, bo_c };
        {
            const xdna_timer t;
            if (!xdna_kernel_run(kern, args, 3)) {
                GGML_LOG_ERROR("%s: GEMM run failed M=%d K=%d N=%d kernel=gemm_K%d_N%d\n",
                               "xdna-ops", M, K, N, GEMM_K_MAX, N);
                xdna_kernel_pool_release_buffer(ops->pool, bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, bo_b);
                xdna_kernel_pool_release_buffer(ops->pool, bo_c);
                return false;
            }
            if (prof_en) prof.run += t.ms();
        }
        {
            const xdna_timer t;
            xdna_buffer_read(bo_c, c_buf.data(), (size_t) Mk * N * sizeof(float));
            if (prof_en) prof.c_read += t.ms();
        }

        xdna_kernel_pool_release_buffer(ops->pool, bo_a);
        xdna_kernel_pool_release_buffer(ops->pool, bo_b);
        xdna_kernel_pool_release_buffer(ops->pool, bo_c);

        // Accumulate this K-block's contribution.
        {
            const xdna_timer t;
            for (size_t i = 0; i < (size_t) Mk * N; i++) {
                c_acc[i] += c_buf[i];
            }
            if (prof_en) prof.accum += t.ms();
        }
        prof.n_blocks++;
    }

    // Write the M x N slice of the accumulated result into dst.
    {
        const xdna_timer t;
        for (int m = 0; m < M; m++) {
            std::memcpy((char *) dst->data + (size_t) m * dst->nb[1],
                        c_acc.data() + (size_t) m * N, (size_t) N * sizeof(float));
        }
        if (prof_en) prof.dst_copy = t.ms();
    }

    if (prof_en) {
        prof.total = t_op.ms();
        fprintf(stderr,
                "xdna-profile: MUL_MAT %s M=%d K=%d N=%d blocks=%d "
                "total=%.3fms bcopy=%.3f apack=%.3f sync=%.3f run=%.3f cread=%.3f accum=%.3f dst=%.3f\n",
                node->name ? node->name : "?", M, K, N, prof.n_blocks,
                prof.total, prof.b_copy, prof.a_pack, prof.sync,
                prof.run, prof.c_read, prof.accum, prof.dst_copy);
    }

    GGML_LOG_INFO("%s: MUL_MAT %s M=%d K=%d N=%d blocks=%d\n",
                  "xdna-ops", node->name ? node->name : "?", M, K, N,
                  (K + GEMM_K_MAX - 1) / GEMM_K_MAX);
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
    if (!xdna_gemm_seq_supported(&ops->gemm_tiles, M, K, N)) {
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
