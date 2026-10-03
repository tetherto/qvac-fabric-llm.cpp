#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

#define GGML_CUDA_CPY_BATCH_MAX 16

// byte offsets of each batched copy from the first copy's source and destination
struct ggml_cuda_cpy_batch {
    int     n;
    int64_t src_offs[GGML_CUDA_CPY_BATCH_MAX];
    int64_t dst_offs[GGML_CUDA_CPY_BATCH_MAX];
};

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// runs batch.n f32 copies with the layout of src0 -> src1 in one launch
void ggml_cuda_cpy_f32_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1,
                             const ggml_cuda_cpy_batch & batch);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
