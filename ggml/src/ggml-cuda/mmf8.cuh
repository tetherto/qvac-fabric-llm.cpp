#pragma once

#include "common.cuh"

// F8_E4M3 weights [K][N] with 128x128 block scales in dst->src[2] (F32 [N/128][K/128]), F32 activations, F32 output
void ggml_cuda_mul_mat_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// whether an F8 mul_mat of these operands takes the Hopper CUTLASS W8A8 route (batch above the GEMV limit, CUTLASS
// built in, not disabled by GGML_CUDA_DISABLE_MMF8_CUTLASS)
bool ggml_cuda_mul_mat_f8_uses_cutlass(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);

// the CUTLASS route for dst = mul_mat(src0, glu) where glu = swiglu_split(gate, up): silu(gate) * up is quantized
// straight into the GEMM's e4m3 input, the GLU output is never written
void ggml_cuda_mul_mat_f8_glu_cutlass(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, ggml_tensor * dst);
