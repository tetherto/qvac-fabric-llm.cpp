#pragma once

#include "common.cuh"

bool ggml_cuda_flash_attn_ext_qsa_f16_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_qsa_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
