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

struct ggml_cuda_mul_mat_id_back_b_kargs {
    uint32_t K;         // dst->ne[0]  (cols)
    uint32_t N;         // as->ne[1]   (reduction dim / rows of each expert)
    uint32_t n_used;    // ids->ne[0]
    uint32_t n_tok;     // ids->ne[1]
    uint32_t dst_ne1;   // dst->ne[1] (n_used or 1 for accumulation)
    uint32_t as_nb1;    // as strides (in elements, f32 only)
    uint32_t as_nb2;
    uint32_t g_nb1;     // grad_out strides
    uint32_t g_nb2;
    uint32_t ids_nb1;   // ids stride
    uint32_t d_nb1;     // dst strides
    uint32_t d_nb2;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_mul_mat_id_back_b_kargs>);

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

static __device__ inline float
mmid_read_as(const float * GGML_CUDA_RESTRICT data_a,
             const uint32_t k, const uint32_t n, const uint32_t e,
             const ggml_cuda_mul_mat_id_back_b_kargs& p) {
    return data_a[k + n*p.as_nb1 + e*p.as_nb2];
}

static __device__ inline float
mmid_read_as(const block_q8_0 * GGML_CUDA_RESTRICT data_a,
             const uint32_t k, const uint32_t n, const uint32_t e,
             const ggml_cuda_mul_mat_id_back_b_kargs& p) {
    const uint32_t blocks_per_row = p.K / QK8_0;
    const uint32_t ib  = (e * p.N + n) * blocks_per_row + (k / QK8_0);
    const uint32_t iqs = k % QK8_0;
    return float(data_a[ib].qs[iqs]) * float(data_a[ib].d);
}

// One workgroup per (slot, t), local invocations stride over k.
// grad_b[t, slot, k] = sum over routed slots u (slot, or all of them when the
// output is broadcast to a single slot) of sum_n as[e, n, k] * grad_out[t, u, n],
// with e = ids[t][u].
template < typename AType >
static __global__ void
mul_mat_id_back_b_cuda(const AType   * GGML_CUDA_RESTRICT data_a, // as       [K, N, n_expert]
                       const float   * GGML_CUDA_RESTRICT data_g, // grad_out [N, n_used, n_tok]
                       const int32_t * GGML_CUDA_RESTRICT data_i, // ids      [n_used, n_tok]
                             float   * GGML_CUDA_RESTRICT data_d, // grad_b   [K, dst_ne1, n_tok]
                       const ggml_cuda_mul_mat_id_back_b_kargs p) {
    
    const uint32_t slot = blockIdx.x;
    const uint32_t t    = blockIdx.y;

    const uint32_t u0 = (p.dst_ne1 == 1u) ? 0u       : slot;
    const uint32_t u1 = (p.dst_ne1 == 1u) ? p.n_used : (slot + 1u);

    for (uint32_t k = threadIdx.x; k < p.K; k += blockDim.x) {
        float acc = 0.0;

        for (uint32_t u = u0; u < u1; ++u) {
            const uint32_t e = uint32_t(data_i[u + t*p.ids_nb1]);
            for (uint32_t n = 0; n < p.N; ++n) {
                const float g = data_g[n + u*p.g_nb1 + t*p.g_nb2];
                acc = fmaf(mmid_read_as(data_a, k, n, e, p), g, acc);
            }
        }

        data_d[k + slot*p.d_nb1 + t*p.d_nb2] = acc;
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

template < typename AType >
static __host__ inline void
launch_mul_mat_id_back_b(const void * data_a,
                         const void * data_g,
                         const void * data_i,
                               void * data_d,
                         cudaStream_t stream,
                         const ggml_cuda_mul_mat_id_back_b_kargs& args) {

    // one block per (slot, token), threads stride over k
    const dim3 block_nums = { args.dst_ne1, args.n_tok, 1 };
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, CUDA_MUL_MAT_ID_BACK_BLOCK_SIZE, 0, stream};

    ggml_cuda_kernel_launch(mul_mat_id_back_b_cuda<AType>,
                            launch_params,
                            (const AType *)data_a,
                            (const float *)data_g,
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

void ggml_cuda_op_mul_mat_id_back_b(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * as       = dst->src[0];
    const ggml_tensor * grad_out = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];

    const void * data_grad = grad_out->data;
    const void * data_a    = as->data;
    const void * data_i    = ids->data;
          void * data_d    = dst->data;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(grad_out->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(data_grad && data_a && data_i && data_d);
    // the quantized path derives block indices from K/N, so `as` must be contiguous
    GGML_ASSERT(ggml_blck_size(as->type) == 1 || ggml_is_contiguous(as));

    const uint32_t as_type_size  = ggml_type_size(as->type);
    const uint32_t as_blck_size  = ggml_blck_size(as->type);
    const uint32_t g_type_size   = ggml_type_size(grad_out->type);
    const uint32_t ids_type_size = ggml_type_size(ids->type);
    const uint32_t d_type_size   = ggml_type_size(dst->type);

    // f32 path reads `as` with element strides; the quantized path computes
    // block indices from K/N directly and ignores these.
    const uint32_t as_nb1 = (as_blck_size == 1) ? (uint32_t)(as->nb[1] / as_type_size) : 0;
    const uint32_t as_nb2 = (as_blck_size == 1) ? (uint32_t)(as->nb[2] / as_type_size) : 0;

    const ggml_cuda_mul_mat_id_back_b_kargs args = {
        .K        = (uint32_t)dst->ne[0],
        .N        = (uint32_t)as->ne[1],
        .n_used   = (uint32_t)ids->ne[0],
        .n_tok    = (uint32_t)ids->ne[1],
        .dst_ne1  = (uint32_t)dst->ne[1],
        .as_nb1   = as_nb1,
        .as_nb2   = as_nb2,
        .g_nb1    = (uint32_t)(grad_out->nb[1] / g_type_size),
        .g_nb2    = (uint32_t)(grad_out->nb[2] / g_type_size),
        .ids_nb1  = (uint32_t)(ids->nb[1] / ids_type_size),
        .d_nb1    = (uint32_t)(dst->nb[1] / d_type_size),
        .d_nb2    = (uint32_t)(dst->nb[2] / d_type_size),
    };
    cudaStream_t stream = ctx.stream();
#define LAUNCH_MMID_BACK_B(A_TYPE) \
    launch_mul_mat_id_back_b<A_TYPE>(data_a, data_grad, data_i, data_d, stream, args)

    switch (as->type) {
    case GGML_TYPE_Q8_0:
        LAUNCH_MMID_BACK_B(block_q8_0);
        break;
    case GGML_TYPE_F32:
        LAUNCH_MMID_BACK_B(float);
        break;
    default:
        GGML_ABORT("unsupported type for mul_mat_id_back_b: %s", ggml_type_name(as->type));
    }
#undef LAUNCH_MMID_BACK_B
}
