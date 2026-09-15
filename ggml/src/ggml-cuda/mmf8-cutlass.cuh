#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// SM90 block-scaled FP8 GEMM (CUTLASS): D[M][N] = sum_k A[M][K] * B[N][K], e4m3 operands,
// A scales per (token, 128-k group) stored [K/128][M], B scales per 128x128 block stored [K/128][N/128]
size_t ggml_cuda_mmf8_cutlass_workspace_size(int M, int N, int K);

void ggml_cuda_mmf8_cutlass(
        const uint8_t * a, const float * sfa, const uint8_t * b, const float * sfb, float * d,
        int M, int N, int K, void * workspace, size_t workspace_size, cudaStream_t stream);
