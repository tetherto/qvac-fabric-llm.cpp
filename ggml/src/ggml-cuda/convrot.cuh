#include "common.cuh"

// Compact I8 + F32 row-scale ConvRot matmul (GGML_OP_MUL_MAT_CONVROT).
void ggml_cuda_op_mul_mat_convrot(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Native H256 transform for a GGML_HINT_SRC0_IS_CONVROT_H256 matmul; returns
// false when the layout needs the plain matmul instead.
bool ggml_cuda_op_convrot_h256(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

bool ggml_cuda_op_mul_mat_use_convrot_h256(const ggml_tensor * op);
