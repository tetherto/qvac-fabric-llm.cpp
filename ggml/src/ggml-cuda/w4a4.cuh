#pragma once

#include "common.cuh"

// Experimental W4A4 matrix multiplication for Q4_0 weights on RDNA3/RDNA4.
// The f32 activations are quantized to q4_0 blocks (32 values, fp16 scale) and the integer dot products use
// v_dot8_i32_iu4 (GGML_CUDA_W4A4=1) or V_WMMA_I32_16X16X16_IU4 (GGML_CUDA_W4A4=2, RDNA3 only), for batches of at least
// GGML_CUDA_W4A4_MIN_BATCH (default 16) columns. GGML_CUDA_W4A4_COLS_FIRST=0 orders the WMMA blocks by weight tile.
// GGML_CUDA_W4A4_EXCLUDE=ffn_down,... keeps weights whose name contains one of the substrings on the default path.
// GGML_CUDA_W4A4_CHECK=1 compares every result against a naive kernel with the same numerics, =2 reports the error
// relative to the default W4A8 path (both slow, need GGML_CUDA_DISABLE_GRAPHS=1).

bool ggml_cuda_should_use_w4a4(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_w4a4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
