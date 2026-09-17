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

// one GEMV launch for the up and gate mul_mats of a swiglu (same activation, F8 weights of the same shape), batch <= 8;
// writes silu(gate) * up into the GLU output from the same per-row sums the separate launches produce
void ggml_cuda_mul_mat_f8_gemv_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu);

// n (2 or 3) F8 mul_mat nodes sharing one F32 src1: quantize once + one CUTLASS GEMM each (CUTLASS batch), or one
// multi-weight GEMV launch (batch <= 8); dsts are the mul_mat nodes, each with its weight in src[0] and scales in src[2]
void ggml_cuda_mul_mat_f8_shared_cutlass(ggml_backend_cuda_context & ctx, ggml_tensor ** dsts, int n);
void ggml_cuda_mul_mat_f8_shared_gemv(ggml_backend_cuda_context & ctx, ggml_tensor ** dsts, int n);
#define GGML_CUDA_MMF8_SHARED_MAX 3
#define GGML_CUDA_MMF8_GEMV_MAX_NCOLS 8

// the batch up to which the F8 GEMV is used instead of the CUTLASS W8A8 GEMM: the cap above, or GGML_CUDA_MMF8_GEMV_MAX
int ggml_cuda_mmf8_gemv_max_ncols();
