#pragma once

#include "common.cuh"

// FlashAttention for prompt processing with FP8 (e4m3) tensor core MMAs for both KQ and VKQ.
// Q is quantized per row, K and V use a scale of 1 (same values as an FP8 KV cache), P uses a scale of 448.
bool ggml_cuda_flash_attn_ext_f8_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_f8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
