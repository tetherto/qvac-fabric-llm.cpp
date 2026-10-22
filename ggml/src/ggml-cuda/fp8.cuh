#pragma once

#include "common.cuh"

// llama stores the static activation scale of an FP8 weight (the checkpoint's input_scale) in the weight's
// op_params: op_params[0] = GGML_CUDA_FP8_INPUT_SCALE_TAG, op_params[1] = the scale as float.
#define GGML_CUDA_FP8_INPUT_SCALE_TAG 0x46503853

// static activation scale of an FP8 weight, 0 if there is none (dynamic per-tensor scale)
float ggml_cuda_fp8_input_scale(const ggml_tensor * src0);

// whether src1 can be quantized once and shared by FP8 matmuls (see ggml_cuda_fp8_quantize)
bool ggml_cuda_fp8_activation_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// src1 -> dense e4m3 values (ggml_nelements(src1) bytes) and their dequantization scale (one float on the device)
void ggml_cuda_fp8_quantize(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float input_scale, uint8_t * values, float * scale);

// act_values/act_scale: src1 already quantized by ggml_cuda_fp8_quantize with this weight's input scale
bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale = nullptr, const uint8_t * act_values = nullptr, const float * act_scale = nullptr);
