#pragma once

#include "common.cuh"

#define MMVF_ACT_BLOCK_SIZE 1024
#define MMVF_ACT_MAX_ITERS  2
#define MMVF_ACT_MAX_NCOLS  8
#define MMVF_ACT_VEC_WIDTH  ((int) (sizeof(float4) / sizeof(float)))
#define MMVF_ACT_MAX_K      (MMVF_ACT_VEC_WIDTH * MMVF_ACT_BLOCK_SIZE * MMVF_ACT_MAX_ITERS)

enum ggml_cuda_mmvf_act {
    GGML_CUDA_MMVF_ACT_SIGMOID,       // sigmoid(W x)
    GGML_CUDA_MMVF_ACT_SOFTPLUS_GATE, // softplus(W x + bias) * scale, bias and scale per row
};

// f32 weights with few rows times a few f32 columns, the activation applied to each result
bool ggml_cuda_should_use_mmvf_act(const ggml_tensor * mul_mat);

void ggml_cuda_mul_mat_vec_f32_act(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * bias, const ggml_tensor * scale, ggml_cuda_mmvf_act act,
                                   ggml_tensor * dst);
