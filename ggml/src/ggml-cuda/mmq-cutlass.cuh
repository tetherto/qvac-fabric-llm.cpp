#pragma once

#include "repack-cutlass-blockscaled.cuh"

bool ggml_cuda_cutlass_compiled();

bool ggml_cuda_repacked_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

struct ggml_cuda_cutlass_activation_layout {
    size_t size_values     = 0;
    size_t offset_scales   = 0;
    size_t size_scales     = 0;
    size_t offset_rows     = 0;
    size_t size_rows       = 0;
    size_t size_allocation = 0;

    int       m        = 0;
    int       k        = 0;
    int       k_padded = 0;
    ggml_type type     = GGML_TYPE_COUNT;
};

struct ggml_cuda_cutlass_activation {
    const uint8_t * values     = nullptr;
    const uint8_t * scales     = nullptr;
    const float *   row_scales = nullptr;

    int       m        = 0;
    int       k        = 0;
    int       k_padded = 0;
    ggml_type type     = GGML_TYPE_COUNT;
};

bool ggml_cuda_cutlass_get_activation_layout(ggml_backend_cuda_context &           ctx,
                                             const ggml_tensor *                   src0,
                                             const ggml_tensor *                   src1,
                                             const ggml_tensor *                   dst,
                                             ggml_cuda_cutlass_activation_layout & layout);

bool ggml_cuda_cutlass_prepare_activation(ggml_backend_cuda_context &                 ctx,
                                          const ggml_tensor *                         src1,
                                          const ggml_cuda_cutlass_activation_layout & layout,
                                          void *                                      workspace,
                                          size_t                                      workspace_size,
                                          ggml_cuda_cutlass_activation &              activation);

enum class ggml_cuda_cutlass_result {
    success,
    fallback,
    failure,
};

constexpr bool ggml_cuda_cutlass_result_can_fallback(ggml_cuda_cutlass_result result) {
    return result == ggml_cuda_cutlass_result::fallback;
}

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat_prequantized(ggml_backend_cuda_context &          ctx,
                                                                const ggml_tensor *                  src0,
                                                                const ggml_tensor *                  src1,
                                                                ggml_tensor *                        dst,
                                                                const ggml_cuda_cutlass_activation & activation);

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat_prequantized_bf16(ggml_backend_cuda_context &          ctx,
                                                                     const ggml_tensor *                  src0,
                                                                     const ggml_tensor *                  src1,
                                                                     const ggml_tensor *                  dst,
                                                                     const ggml_cuda_cutlass_activation & activation,
                                                                     void *                               output);

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
