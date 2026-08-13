#include "xdna-ops.h"
#include "xdna-runtime.h"

#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool) {
    ops->pool = pool;
    ops->gemm_variants.clear();
    for (const std::string & name : pool->names) {
        int M = 0, K = 0, N = 0, n_cols = 0;
        if (sscanf(name.c_str(), "gemm_bf16_f32_M%d_K%d_N%d_c%d", &M, &K, &N, &n_cols) != 4) {
            continue;
        }
        if (M <= 0 || K <= 0 || N <= 0 || n_cols <= 0) {
            continue;
        }
        ops->gemm_variants[name] = {M, K, N, n_cols, N < 256 ? 16 : 32, name};
    }
}

// Best variant for (K, N): exact K, smallest N >= N. nullptr when none.
static const xdna_ops_gemm_variant * gemm_find(const xdna_ops * ops, int K, int N) {
    const xdna_ops_gemm_variant * best = nullptr;
    for (const auto & kv : ops->gemm_variants) {
        const xdna_ops_gemm_variant & v = kv.second;
        if (v.K != K || v.N < N) {
            continue;
        }
        if (!best || v.N < best->N) {
            best = &v;
        }
    }
    return best;
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

    const xdna_ops_gemm_variant * variant = gemm_find(ops, K, N);
    if (!variant || M <= 0 || M > variant->M) {
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

bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return gemm_supported(ops, op);
        default:
            return false;
    }
}

// MUL_MAT follows ggml's convention: src0 = weights [K, N] (stored as N rows
// of K), src1 = activations [K, M] (stored as M rows of K), so dst[M, N] =
// src1^T @ src0. src1's memory feeds the kernel directly as A [M, K]; src0's
// memory is transposed to B [K, N]. Inputs are zero-padded to the chosen
// variant's baked block and the first M x N slice of C is written back to
// dst. Always cross-checks against a CPU reference.
static bool gemm_compute(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    struct ggml_tensor * dst = node;

    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];

    // Guaranteed to have a variant: compute is only reached for ops accepted
    // by xdna_ops_gemm_supported, so just pick the right one.
    const xdna_ops_gemm_variant * variant = gemm_find(ops, K, N);
    GGML_ASSERT(variant != nullptr);

    xdna_kernel * kern = xdna_kernel_pool_get(ops->pool, variant->name);
    if (!kern) {
        GGML_LOG_ERROR("%s: failed to load GEMM kernel %s\n", "xdna-ops", variant->name.c_str());
        return false;
    }

    const int Mk = variant->M;
    const int Kk = variant->K;
    const int Nk = variant->N;

    xdna_buffer * bo_a = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * Kk * sizeof(ggml_bf16_t));
    xdna_buffer * bo_b = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Kk * Nk * sizeof(ggml_bf16_t));
    xdna_buffer * bo_c = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * Nk * sizeof(float));
    if (!bo_a || !bo_b || !bo_c) {
        if (bo_a) xdna_kernel_pool_release_buffer(ops->pool, bo_a);
        if (bo_b) xdna_kernel_pool_release_buffer(ops->pool, bo_b);
        if (bo_c) xdna_kernel_pool_release_buffer(ops->pool, bo_c);
        GGML_LOG_ERROR("%s: failed to allocate GEMM buffers\n", "xdna-ops");
        return false;
    }

    // The BOs are host-visible, so write straight into their mapped memory:
    // one copy instead of host vector + write. Pads are zeroed (stale data
    // from the pool must not leak past the real M x N block).

    // B: ggml src0 [K, N] -> kernel [Kk x Nk] bf16, transposed + padded.
    ggml_bf16_t * b_map = (ggml_bf16_t *) bo_b->bo.map();
    std::memset(b_map, 0, (size_t) Kk * Nk * sizeof(ggml_bf16_t));
    if (src0->type == GGML_TYPE_BF16) {
        const auto * d = (const ggml_bf16_t *) src0->data;
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                b_map[(size_t) k * Nk + n] = d[(size_t) n * K + k];
            }
        }
    } else { // GGML_TYPE_F16
        std::vector<float> row(K);
        for (int n = 0; n < N; n++) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K, row.data(), K);
            for (int k = 0; k < K; k++) {
                ggml_fp32_to_bf16_row(&row[k], &b_map[(size_t) k * Nk + n], 1);
            }
        }
    }

    // A: src1 [M rows of K] f32 -> [Mk x Kk] bf16, rows padded.
    ggml_bf16_t * a_map = (ggml_bf16_t *) bo_a->bo.map();
    std::memset(a_map, 0, (size_t) Mk * Kk * sizeof(ggml_bf16_t));
    for (int m = 0; m < M; m++) {
        ggml_fp32_to_bf16_row((const float *) src1->data + (size_t) m * K,
                              a_map + (size_t) m * Kk, K);
    }

    xdna_buffer_sync_to_device(bo_a);
    xdna_buffer_sync_to_device(bo_b);

    std::vector<float> c_buf((size_t) Mk * Nk, 0);
    xdna_buffer * args[3] = { bo_a, bo_b, bo_c };
    if (!xdna_kernel_run(kern, args, 3)) {
        GGML_LOG_ERROR("%s: GEMM run failed M=%d K=%d N=%d kernel=%s\n",
                       "xdna-ops", M, K, N, variant->name.c_str());
        xdna_kernel_pool_release_buffer(ops->pool, bo_a);
        xdna_kernel_pool_release_buffer(ops->pool, bo_b);
        xdna_kernel_pool_release_buffer(ops->pool, bo_c);
        return false;
    }
    xdna_buffer_read(bo_c, c_buf.data(), (size_t) Mk * Nk * sizeof(float));

    xdna_kernel_pool_release_buffer(ops->pool, bo_a);
    xdna_kernel_pool_release_buffer(ops->pool, bo_b);
    xdna_kernel_pool_release_buffer(ops->pool, bo_c);

    // CPU reference: same bf16 inputs, f32 accumulation.
    std::vector<float> b_f32((size_t) K * N);
    for (int n = 0; n < N; n++) {
        if (src0->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src0->data + (size_t) n * K,
                                  b_f32.data() + (size_t) n * K, K);
        } else {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K,
                                  b_f32.data() + (size_t) n * K, K);
        }
    }
    const float * A = (const float *) src1->data;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) {
                ref += A[(size_t) m * K + k] * b_f32[(size_t) n * K + k];
            }
            const float err = fabsf(c_buf[(size_t) m * Nk + n] - ref);
            max_abs = fmaxf(max_abs, err);
            max_rel = fmaxf(max_rel, err / fmaxf(fabsf(ref), 1e-6f));
        }
    }

    // Write the M x N slice of C into dst.
    for (int m = 0; m < M; m++) {
        std::memcpy((char *) dst->data + (size_t) m * dst->nb[1],
                    c_buf.data() + (size_t) m * Nk, (size_t) N * sizeof(float));
    }

    GGML_LOG_INFO("%s: MUL_MAT %s M=%d K=%d N=%d kernel=%s max_abs=%f max_rel=%f\n",
                  "xdna-ops", node->name ? node->name : "?", M, K, N,
                  variant->name.c_str(), max_abs, max_rel);
    return true;
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
