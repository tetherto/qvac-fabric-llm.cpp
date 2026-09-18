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

// the batch up to which the F8 GEMV is used instead of the CUTLASS W8A8 GEMM. The GEMM is 39% faster per decode step
// at 8 tokens, but its activation quantization misses the gate there (mean KLD 0.0087, same top p 97.8%), so the GEMV
// keeps the whole batch range it serves; GGML_CUDA_MMF8_GEMV_MAX overrides it for A/B, up to the cap above
#define GGML_CUDA_MMF8_GEMV_MAX_DEFAULT GGML_CUDA_MMF8_GEMV_MAX_NCOLS
int ggml_cuda_mmf8_gemv_max_ncols();

// up to three F8 matrices sharing one activation in one launch (q/k/v, the delta-net qkv/z): block b computes rows
// of the matrix whose block range holds b, with the per-row arithmetic of the single-weight kernel
#define MMF8_MULTI_MAX 3
struct mmf8_multi_args {
    const uint8_t * x[MMF8_MULTI_MAX];
    const float   * sx[MMF8_MULTI_MAX];
    float         * dst[MMF8_MULTI_MAX];
    int nrows[MMF8_MULTI_MAX];
    int block0[MMF8_MULTI_MAX + 1]; // first block of each matrix; block0[n] = the grid size
    int n;
};

// The f16 tensor-core decode matmul (mmf8-mma.cu): same operands, scales and output as the GEMV, but each weight
// byte is converted once into an f16 MMA tile and accumulated in FP32, which removes the GEMV's per-column FP32
// FMA chain and its repeated conversions. Batch 2..8; batch 1 keeps the GEMV. The activation is converted to f16
// once per launch into a pool buffer, so the B tile costs half the bytes and no conversion per slab; that needs
// the context, and it needs the activation columns to be contiguous (stride_col_y == ncols).
void mul_mat_f8_e4m3_mma_cuda(
    ggml_backend_cuda_context & ctx,
    const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const float * y, float * dst,
    int ncols, int nrows, int nblk_n, int ncols_dst, int stride_col_y, int stride_col_dst, cudaStream_t stream);
void mul_mat_f8_e4m3_mma_multi_cuda(
    ggml_backend_cuda_context & ctx,
    const mmf8_multi_args & a, const float * y, int ncols, int ncols_dst, int stride_col_y, cudaStream_t stream);

// GGML_CUDA_MMF8_MMA_MIN overrides the batch at which the tensor-core matmul takes over from the GEMV;
// GGML_CUDA_MMF8_GEMV_MAX_NCOLS + 1 disables it and is the A/B switch against the GEMV. Measured 2026-09-18 at
// 10k context: batch 8 runs 367 against 206 tok/s, batch 4 215 against 204, batch 2 117 against 144, so the
// default is 4: an 8-wide B tile wastes too many of its columns below that
#define GGML_CUDA_MMF8_MMA_MIN_DEFAULT 4
int  ggml_cuda_mmf8_mma_min_ncols();
bool ggml_cuda_mmf8_use_mma(ggml_backend_cuda_context & ctx, int64_t ntokens);
