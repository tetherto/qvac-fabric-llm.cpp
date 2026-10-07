#include "common.cuh"

// builds whose TOP_K takes rows longer than 1024 columns, by cub or by radix selection
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
#    define GGML_CUDA_TOP_K_LONG_ROWS
#endif  // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
