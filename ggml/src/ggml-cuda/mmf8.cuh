#pragma once

#include "common.cuh"

// F8_E4M3 weights [K][N] with 128x128 block scales in dst->src[2] (F32 [N/128][K/128]), F32 activations, F32 output
void ggml_cuda_mul_mat_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
