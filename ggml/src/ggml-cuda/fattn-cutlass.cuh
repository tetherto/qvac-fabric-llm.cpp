// SM90 fused multi-head attention (CUTLASS example 88) for causal prefill; compiled for sm_90a only
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// fixed by the only fp16 head-dim-256 instantiation of the example: tile 128 (queries) x 64 (keys) x 256
constexpr int GGML_FATTN_CUTLASS_D      = 256;
constexpr int GGML_FATTN_CUTLASS_TILE_Q = 128;
constexpr int GGML_FATTN_CUTLASS_TILE_K = 64;

// Layouts (elements): q16/o16 are [D][n_head][n_q] contiguous (n_head = n_head_kv * gqa, group g holds heads
// g*gqa .. g*gqa+gqa-1); k/v are [D][n_kv][n_head_kv] with row stride nb1 and head stride nb2 (elements);
// lse is [n_q][n_head]. The kernel attends cells [0, n_kv_used) with a bottom-right causal mask, softmax
// scale 1/sqrt(D). Requires n_q % 8 == 0, n_q <= n_kv_used <= n_kv.
size_t ggml_cuda_fattn_cutlass_workspace_size(int n_head_kv, int gqa, int n_q, int n_kv_used);

void ggml_cuda_fattn_cutlass(
        const half * q16, const half * k, const half * v, half * o16, float * lse,
        int n_head_kv, int gqa, int n_q, int n_kv_used,
        int64_t k_nb1, int64_t k_nb2, int64_t v_nb1, int64_t v_nb2,
        void * workspace, size_t workspace_size, cudaStream_t stream);
