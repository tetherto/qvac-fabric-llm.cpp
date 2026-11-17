#pragma once

#include "common.cuh"

// Batched F32 products with a short contraction and few columns, one direct kernel instead of cuBLAS:
// windowed attention over many small windows gives tens of thousands of 16x16 products per call.

// longest contraction (src0->ne[0]): the rows of src0 a thread holds must fit in registers.
// mm_small_launch has one switch case per multiple of 4 up to this value: change both together.
#define CUDA_MM_SMALL_MAX_K 32
// most src1 columns: above this the strided batched cuBLAS GEMM is as fast or faster
#define CUDA_MM_SMALL_MAX_COLS 32
// fewest threads per launch, one per 1, 2 or 4 rows of src0 (see mm_small_rows_per_thread). Each thread loops over
// every src1 column, so a small launch is bound by that loop. Measured on a T4 against MMVF and cuBLAS for K 2 to 32,
// 2 to 32 columns and 16 to 64 rows: from 1024 threads on mm-small was 1.6x to 20x faster in every shape, at 512
// one shape was 1.6x slower. Taller matrices were measured at BridgeDepth's 64 products of 5580 rows: 9.5x faster.
#define CUDA_MM_SMALL_MIN_THREADS 1024

// GPUs where mm-small was measured faster than MMVF / cuBLAS: a T4 (Turing) and a Radeon 8060S (RDNA 3.5, gfx1151).
// On an RTX 5090 (Blackwell) cuBLAS was 1.5x to 6x faster for K 16 and 32 with 8 or more columns. Every other GPU
// keeps MMVF / cuBLAS until measured. GGML_CUDA_FORCE_MM_SMALL=1 skips this check, to test the kernel on any GPU.
static inline bool ggml_cuda_mm_small_measured_cc(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING && cc < GGML_CUDA_CC_AMPERE) ||
           GGML_CUDA_CC_IS_RDNA3_5(cc);
}

bool ggml_cuda_should_use_mm_small(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
