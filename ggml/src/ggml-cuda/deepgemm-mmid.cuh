#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Experimental SM90 DeepGEMM implementation for the Qwen4-Exp MUL_MAT_ID
// shapes. Returns false when the tensor layout, device, or runtime opt-in does
// not match so the regular CUDA implementation remains the fallback.
bool ggml_cuda_deepgemm_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Returns true when the native-FP8 operation is handled entirely by
// stream-ordered DeepGEMM kernels and can therefore be captured in a CUDA
// graph. This intentionally performs the same shape/layout checks as the
// execution path rather than assuming every FP8 MUL_MAT_ID is supported.
bool ggml_cuda_deepgemm_mul_mat_id_graph_compatible(const ggml_tensor * dst, int device);
