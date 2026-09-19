
#include "common.cuh"

__host__ void
ggml_cuda_op_ssm_conv_back_sx(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

__host__ void
ggml_cuda_op_ssm_conv_back_c(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
