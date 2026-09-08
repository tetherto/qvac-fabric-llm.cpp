#pragma once

#include "repack-cutlass-blockscaled.cuh"

bool ggml_cuda_cutlass_compiled();

bool ggml_cuda_repacked_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

enum class ggml_cuda_cutlass_result {
    success,
    fallback,
    failure,
};

constexpr bool ggml_cuda_cutlass_result_can_fallback(ggml_cuda_cutlass_result result) {
    return result == ggml_cuda_cutlass_result::fallback;
}

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
