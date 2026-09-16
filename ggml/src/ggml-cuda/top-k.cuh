#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_try_qsa_reduce_scores(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * scores,
        const ggml_tensor * bias,
        ggml_tensor * dst);
