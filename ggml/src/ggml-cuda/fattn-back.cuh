#pragma once

#include "common.cuh"

// shapes and byte strides of ggml_flash_attn_ext_back, dim 0 of every tensor is contiguous
struct ggml_cuda_flash_attn_ext_back_kargs {
    int32_t target; // ggml_flash_attn_ext_back_target
    float   scale;

    int64_t D;         // head size, hsk == hsv
    int64_t n_q;       // q rows per head
    int64_t n_kv;      // kv rows per head
    int64_t n_head;
    int64_t n_head_kv; // q head h reads kv head h / (n_head/n_head_kv)
    int64_t ne3;

    // q [D, n_q, n_head, ne3], k and v [D, n_kv, n_head_kv, ne3]
    uint64_t nbq1, nbq2, nbq3;
    uint64_t nbk1, nbk2, nbk3;
    uint64_t nbv1, nbv2, nbv3;

    // mask [n_kv, >= n_q, nem2, nem3] F16, broadcast over heads and ne3, zero when absent
    int64_t  nem2, nem3;
    uint64_t nbm1, nbm2, nbm3;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_flash_attn_ext_back_kargs>);

void ggml_cuda_flash_attn_ext_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_back_supported(int device, const ggml_tensor * dst);
