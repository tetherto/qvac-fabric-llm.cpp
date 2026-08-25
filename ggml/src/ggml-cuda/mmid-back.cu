#include "mmid-back.cuh"

#include <type_traits>

#define CUDA_MUL_MAT_ID_BACK_BLOCK_SIZE 256

struct ggml_cuda_mul_mat_id_back_a_kargs {
    uint32_t K;         // dst->ne[0]  (cols of each expert)
    uint32_t n_used;    // ids->ne[0]
    uint32_t n_tok;     // ids->ne[1]
    uint32_t g_nb1;     // grad_out strides (in elements)
    uint32_t g_nb2;
    uint32_t b_nb1;     // b strides
    uint32_t b_nb2;
    uint32_t ids_nb1;   // ids stride
    uint32_t d_nb1;     // dst strides
    uint32_t d_nb2;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_mul_mat_id_back_a_kargs>);

template < const bool broadcast_b >
static __global__ void
mul_mat_id_back_a_cuda(const float   * GGML_CUDA_RESTRICT data_g, // grad_out [N, n_used, n_tok]
                       const float   * GGML_CUDA_RESTRICT data_b, // b        [K, b_ne1, n_tok]
                       const int32_t * GGML_CUDA_RESTRICT data_i, // ids      [n_used, n_tok]
                             float   * GGML_CUDA_RESTRICT data_d, // grad_as  [K, N, n_expert]
                       const ggml_cuda_mul_mat_id_back_a_kargs args) {
    const uint32_t n = blockIdx.x;
    const uint32_t e = blockIdx.y;

    const uint32_t d_offset = n*args.d_nb1 + e*args.d_nb2;

    for (uint32_t k = threadIdx.x; k < args.K; k += blockDim.x) {
        float acc = 0.0;

        for (uint32_t t = 0; t < args.n_tok; ++t) {

            const uint32_t i_offset = t*args.ids_nb1;
            const uint32_t g_offset = n + t*args.g_nb2;
            const uint32_t b_offset = k + t*args.b_nb2;

            for (uint32_t u = 0; u < args.n_used; ++u) {
                if (static_cast<uint32_t>(data_i[i_offset + u]) != e) {
                    continue;
                }

                const float g = data_g[g_offset + u*args.g_nb1];
                float bv;
                if constexpr (broadcast_b) {
                    bv = data_b[b_offset];
                } else {
                    bv = data_b[b_offset + u*args.b_nb1];
                }
                acc = fmaf(g, bv, acc);
            }
        }

        data_d[d_offset + k] = acc;
    }
}

template < const bool broadcast_b >
static __host__ inline void
launch_mul_mat_id_back_a(const void * data_g,
                         const void * data_b,
                         const void * data_i,
                               void * data_d,
                         cudaStream_t stream,
                         uint32_t N,         // dst->ne[1]  (rows of each expert)
                         uint32_t n_expert,  // dst->ne[2]
                         const ggml_cuda_mul_mat_id_back_a_kargs& args) {

    // one block per (n, e), threads stride over k
    const dim3 block_nums = { N, n_expert, 1 };
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, CUDA_MUL_MAT_ID_BACK_BLOCK_SIZE, 0, stream};

    ggml_cuda_kernel_launch(mul_mat_id_back_a_cuda<broadcast_b>,
                            launch_params,
                            (const float *)data_g,
                            (const float *)data_b,
                            (const int32_t *)data_i,
                            (float *)data_d,
                            args);
}

void ggml_cuda_op_mul_mat_id_back_a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    
    const ggml_tensor * grad_out = dst->src[0];
    const ggml_tensor * b        = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];

    const void * data_grad = grad_out->data;
    const void * data_b    = b->data;
    const void * data_i    = ids->data;
          void * data_d    = dst->data;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(grad_out->type == dst->type && b->type == dst->type);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(data_grad && data_b && data_i && data_d);

    // b->ne[1] (n_used or 1 for broadcast)
    const uint32_t b_ne1 = (uint32_t)b->ne[1];

    const uint32_t g_type_size   = ggml_type_size(grad_out->type);
    const uint32_t b_type_size   = ggml_type_size(b->type);
    const uint32_t ids_type_size = ggml_type_size(ids->type);
    const uint32_t d_type_size   = ggml_type_size(dst->type);

    const uint32_t N        = (uint32_t)dst->ne[1];
    const uint32_t n_expert = (uint32_t)dst->ne[2];

    const ggml_cuda_mul_mat_id_back_a_kargs args = {
        .K       = (uint32_t)dst->ne[0],
        .n_used  = (uint32_t)ids->ne[0],
        .n_tok   = (uint32_t)ids->ne[1], 
        .g_nb1   = (uint32_t)(grad_out->nb[1] / g_type_size),
        .g_nb2   = (uint32_t)(grad_out->nb[2] / g_type_size),
        .b_nb1   = (uint32_t)(b->nb[1] / b_type_size),
        .b_nb2   = (uint32_t)(b->nb[2] / b_type_size),
        .ids_nb1 = (uint32_t)(ids->nb[1] / ids_type_size),
        .d_nb1   = (uint32_t)(dst->nb[1] / d_type_size),
        .d_nb2   = (uint32_t)(dst->nb[2] / d_type_size),
    };
    cudaStream_t stream = ctx.stream();
#define LAUNCH_MMID_BACK_A(broadcast_b) \
    launch_mul_mat_id_back_a<broadcast_b>(data_grad, data_b, data_i, data_d, stream, N, n_expert, args)
    
    if (b_ne1 == 1) {
        LAUNCH_MMID_BACK_A(true);
    } else {
        LAUNCH_MMID_BACK_A(false);
    }
#undef LAUNCH_MMID_BACK_A
}
