#include "sigmoid-back.cuh"

#include "ggml-cuda/common.cuh"

#define CUDA_SIGMOID_BACK_BLOCK_SIZE 256

template < typename T >
__global__ void
sigmoid_back_cuda(const T * GGML_CUDA_RESTRICT data_grad,
                  const T * GGML_CUDA_RESTRICT data_src,
                        T * GGML_CUDA_RESTRICT data_dst,
                  const int64_t k) {

    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= k) {
        return;
    }

    constexpr const float one = 1.0f;
    const float dy = static_cast<float>(data_grad[tid]);
    const float x  = static_cast<float>(data_src[tid]);
    const float s  = one / (one + expf(-x));

    data_dst[tid] = static_cast<T>(dy * s * (one - s));
}

template < typename T >
static __host__ inline void
launch_sigmoid_back(const void * data_grad,
                    const void * data_src,
                          void * data_dst,
                    const int64_t k,
                    cudaStream_t stream) {

    const int64_t num_blocks = (k + CUDA_SIGMOID_BACK_BLOCK_SIZE - 1) / CUDA_SIGMOID_BACK_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = {(dim3)num_blocks, CUDA_SIGMOID_BACK_BLOCK_SIZE, 0, stream};

    ggml_cuda_kernel_launch(sigmoid_back_cuda<T>,
                            launch_params,
                            (const T *)data_grad,
                            (const T *)data_src,
                            (T *)data_dst,
                            k);
}

__host__ void
ggml_cuda_op_sigmoid_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * grad = dst->src[0]; // grads of forward pass output
    const ggml_tensor * src  = dst->src[1]; // input from forward pass

    const void * data_grad = grad->data;
    const void * data_src  = src->data;
          void * data_dst  = dst->data;

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(grad->type == dst->type && src->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(grad) &&
                ggml_is_contiguous(src) &&
                ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(dst, grad) &&
                ggml_are_same_shape(dst, src));

    cudaStream_t stream = ctx.stream();
    const int64_t k = ggml_nelements(src);

    if (dst->type == GGML_TYPE_F16) {
        launch_sigmoid_back<half>(data_grad, data_src, data_dst, k, stream);
    } else {
        launch_sigmoid_back<float>(data_grad, data_src, data_dst, k, stream);
    }
}
