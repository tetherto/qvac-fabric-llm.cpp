#include "ssm-conv-back.cuh"

#include <type_traits>

#define CUDA_SSM_CONV_BACK_BLOCK_SIZE 256

struct ggml_cuda_ssm_conv_back_sx_kargs {
    uint32_t nc, ncs, nr, n_t, n_s;
    uint32_t grad_nb0, grad_nb1, grad_nb2;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t c_nb1;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_ssm_conv_back_sx_kargs>);

struct ggml_cuda_ssm_conv_back_c_kargs {
    uint32_t nc, ncs, nr, n_t, n_s;
    uint32_t grad_nb0, grad_nb1, grad_nb2;
    uint sx_nb0, sx_nb1, sx_nb2;
    uint dst_nb1;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_ssm_conv_back_c_kargs>);

static __global__ void
ssm_conv_back_sx_cuda(const float * GGML_CUDA_RESTRICT data_grad, // grad_out {d_inner, n_t, n_s}
                      const float * GGML_CUDA_RESTRICT data_c,    // c        {d_conv, d_inner}
                            float * GGML_CUDA_RESTRICT data_dst,  // grad_sx  {ncs, d_inner, n_s}
                      ggml_cuda_ssm_conv_back_sx_kargs args) {

    const uint32_t p  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t i3 = blockIdx.z * blockDim.z + threadIdx.z;

    if (p >= args.ncs || i1 >= args.nr || i3 >= args.n_s) {
        return;
    }

    float sumf = 0.0f;

    if (args.n_t > 0) {
        const uint32_t grad_base = i1*args.grad_nb0 + i3*args.grad_nb2;
        const uint32_t c_base    = i1*args.c_nb1;

        const uint32_t t0 = (p + 1 > args.nc) ? (p + 1 - args.nc) : 0;
        const uint32_t t1 = min(args.n_t - 1, p);

        for (uint32_t t = t0; t <= t1; ++t) {
            sumf += data_grad[grad_base + t*args.grad_nb1] * data_c[c_base + (p - t)];
        }
    }

    data_dst[p*args.dst_nb0 + i1*args.dst_nb1 + i3*args.dst_nb2] = sumf;
}

static __global__ void
ssm_conv_back_c_cuda(const float * GGML_CUDA_RESTRICT data_grad, // {d_inner, n_t, n_s}
                     const float * GGML_CUDA_RESTRICT data_sx,   // {d_conv - 1 + n_t, d_inner, n_s}
                           float * GGML_CUDA_RESTRICT data_dst,  // {d_conv, d_inner}
                     ggml_cuda_ssm_conv_back_c_kargs args) {

    const uint32_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // conv tap [0, nc)
    const uint32_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // channel [0, nr)

    if (i0 >= args.nc || i1 >= args.nr) {
        return;
    }

    float sumf = 0.0f;

    for (uint32_t i3 = 0; i3 < args.n_s; ++i3) {
        const uint32_t grad_base = i1*args.grad_nb0 + i3*args.grad_nb2;
        const uint32_t sx_base   = i0*args.sx_nb0 + i1*args.sx_nb1 + i3*args.sx_nb2;
        for (uint32_t i2 = 0; i2 < args.n_t; ++i2) {
            sumf += data_grad[grad_base + i2*args.grad_nb1] * data_sx[sx_base + i2*args.sx_nb0];
        }
    }

    data_dst[i0 + i1*args.dst_nb1] = sumf;
}

static __host__ inline void
launch_op_ssm_conv_back_sx(const void * data_grad,
                           const void * data_c,
                                 void * data_dst,
                           cudaStream_t stream,
                           const ggml_cuda_ssm_conv_back_sx_kargs& args) {

    const dim3 block_nums = {
        (args.ncs + CUDA_SSM_CONV_BACK_BLOCK_SIZE - 1) / CUDA_SSM_CONV_BACK_BLOCK_SIZE,
        args.nr,
        args.n_s,
    };
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, CUDA_SSM_CONV_BACK_BLOCK_SIZE, 0, stream};

    ggml_cuda_kernel_launch(ssm_conv_back_sx_cuda,
                            launch_params,
                            (const float *)data_grad,
                            (const float *)data_c,
                            (float *)data_dst,
                            args);
}

static __host__ inline void
launch_op_ssm_conv_back_c(const void * data_grad,
                          const void * data_c,
                                void * data_dst,
                          cudaStream_t stream,
                          const ggml_cuda_ssm_conv_back_c_kargs& args) {

    const dim3 block_nums = {
        (args.nc + CUDA_SSM_CONV_BACK_BLOCK_SIZE - 1) / CUDA_SSM_CONV_BACK_BLOCK_SIZE,
        args.nr,
        1,
    };
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, CUDA_SSM_CONV_BACK_BLOCK_SIZE, 0, stream};

    ggml_cuda_kernel_launch(ssm_conv_back_c_cuda,
                            launch_params,
                            (const float *)data_grad,
                            (const float *)data_c,
                            (float *)data_dst,
                            args);
}

__host__ void
ggml_cuda_op_ssm_conv_back_sx(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * grad_out = dst->src[0];
    const ggml_tensor * c        = dst->src[1];

    const void * data_grad = grad_out->data;
    const void * data_c    = c->data;
          void * data_dst  = dst->data;

    constexpr const auto fsz = static_cast<uint32_t>(sizeof(float));

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(grad_out->type == dst->type && c->type == dst->type);
    GGML_ASSERT(data_grad && data_c && data_dst);
    GGML_ASSERT(c->nb[0] == fsz);

    const uint32_t nc  = (uint32_t)c->ne[0];
    const uint32_t ncs = (uint32_t)dst->ne[0];
    const uint32_t nr  = (uint32_t)dst->ne[1];
    const uint32_t n_t = (uint32_t)grad_out->ne[1];
    const uint32_t n_s = (uint32_t)dst->ne[2];

    const ggml_cuda_ssm_conv_back_sx_kargs args = {
        .nc       = nc,
        .ncs      = ncs,
        .nr       = nr,
        .n_t      = n_t,
        .n_s      = n_s,
        .grad_nb0 = (uint32_t)grad_out->nb[0]/fsz,
        .grad_nb1 = (uint32_t)grad_out->nb[1]/fsz,
        .grad_nb2 = (uint32_t)grad_out->nb[2]/fsz,
        .dst_nb0  = (uint32_t)dst->nb[0]/fsz,
        .dst_nb1  = (uint32_t)dst->nb[1]/fsz,
        .dst_nb2  = (uint32_t)dst->nb[2]/fsz,
        .c_nb1    = (uint32_t)c->nb[1]/fsz,
    };
    cudaStream_t stream = ctx.stream();
    launch_op_ssm_conv_back_sx(data_grad, data_c, data_dst, stream, args);
}

__host__ void
ggml_cuda_op_ssm_conv_back_c(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * grad_out = dst->src[0];
    const ggml_tensor * sx       = dst->src[1];

    const void * data_grad = grad_out->data;
    const void * data_sx    = sx->data;
          void * data_dst  = dst->data;

    constexpr const auto fsz = static_cast<uint32_t>(sizeof(float));

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(grad_out->type == dst->type && sx->type == dst->type);
    GGML_ASSERT(data_grad && data_sx && data_dst);
    GGML_ASSERT(sx->nb[0] == fsz);

    const uint32_t nc  = (uint32_t)dst->ne[0];
    const uint32_t ncs = (uint32_t)sx->ne[0];
    const uint32_t nr  = (uint32_t)dst->ne[1];
    const uint32_t n_t = (uint32_t)grad_out->ne[1];
    const uint32_t n_s = (uint32_t)grad_out->ne[2];

    const ggml_cuda_ssm_conv_back_c_kargs args = {
        .nc       = nc,
        .ncs      = ncs,
        .nr       = nr,
        .n_t      = n_t,
        .n_s      = n_s,
        .grad_nb0 = (uint32_t)grad_out->nb[0]/fsz,
        .grad_nb1 = (uint32_t)grad_out->nb[1]/fsz,
        .grad_nb2 = (uint32_t)grad_out->nb[2]/fsz,
        .sx_nb0  = (uint32_t)sx->nb[0]/fsz,
        .sx_nb1  = (uint32_t)sx->nb[1]/fsz,
        .sx_nb2  = (uint32_t)sx->nb[2]/fsz,
        .dst_nb1 = (uint32_t)dst->nb[1]/fsz,
    };
    cudaStream_t stream = ctx.stream();
    launch_op_ssm_conv_back_c(data_grad, data_sx, data_dst, stream, args);
}
