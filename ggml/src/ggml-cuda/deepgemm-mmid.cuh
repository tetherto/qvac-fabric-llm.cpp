#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Experimental SM90 DeepGEMM implementation for the Qwen4-Exp MUL_MAT_ID
// shapes. Returns false when the tensor layout, device, or runtime opt-in does
// not match so the regular CUDA implementation remains the fallback.
bool ggml_cuda_deepgemm_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fuses a batch-1 down MUL_MAT_ID with the router-weighted expert reduction.
// Returns false unless the complete shape and layout are supported, allowing
// the caller to retain the normal MUL_MAT_ID + reduction path as fallback.
bool ggml_cuda_deepgemm_mul_mat_id_weighted_reduction(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * weights,
        ggml_tensor * reduced_dst);

// Fuses the complete batch-1 merged gate/up -> SwiGLU -> down -> weighted
// reduction chain while retaining the original GGML graph and tensor-split
// boundary. Returns false unless every tensor and shape is supported.
bool ggml_cuda_deepgemm_moe_ffn_b1(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * gate_up_dst,
        ggml_tensor * down_dst,
        const ggml_tensor * weights,
        ggml_tensor * reduced_dst);

// Returns true when the native-FP8 operation is handled entirely by
// stream-ordered DeepGEMM kernels and can therefore be captured in a CUDA
// graph. This intentionally performs the same shape/layout checks as the
// execution path rather than assuming every FP8 MUL_MAT_ID is supported.
bool ggml_cuda_deepgemm_mul_mat_id_graph_compatible(const ggml_tensor * dst, int device);
