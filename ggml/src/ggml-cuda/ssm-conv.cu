#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_state_f32(
        const float * x_ptr,
        const float * state_ptr,
        const float * weights_ptr,
        const float * bias_ptr,
        int x_nb0, int x_nb1, int x_nb2,
        int state_nb1, int state_nb2,
        int weights_nb1,
        float * dst_ptr, int dst_nb0, int dst_nb1, int dst_nb2,
        int64_t n_t) {
    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x = (const float *) ((const char *) x_ptr +
            bidx*x_nb2 + bidy*split_d_inner*x_nb0);
    const float * state = (const float *) ((const char *) state_ptr +
            bidx*state_nb2 + bidy*split_d_inner*state_nb1);
    const float * weights = (const float *) ((const char *) weights_ptr +
            bidy*split_d_inner*weights_nb1);
    float * dst = (float *) ((char *) dst_ptr +
            bidx*dst_nb2 + bidy*split_d_inner*dst_nb0);

    const int stride_x     = x_nb1 / sizeof(float);
    const int stride_state = state_nb1 / sizeof(float);
    const int stride_w     = weights_nb1 / sizeof(float);
    const int stride_dst   = dst_nb1 / sizeof(float);

    ggml_cuda_pdl_sync();
    float w[d_conv];
#pragma unroll
    for (size_t j = 0; j < d_conv; ++j) {
        w[j] = weights[tid*stride_w + j];
    }
    const float bias = bias_ptr ? bias_ptr[bidy*split_d_inner + tid] : 0.0f;

    for (int64_t it = 0; it < n_t; ++it) {
        float sum = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; ++j) {
            const int64_t pos = it + j;
            const float xv = pos < (int64_t) d_conv - 1
                ? state[tid*stride_state + pos]
                : x[tid + (pos - (d_conv - 1))*stride_x];
            sum += xv*w[j];
        }
        sum += bias;
        dst[it*stride_dst + tid] = apply_silu ? ggml_cuda_op_silu_single(sum) : sum;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_state_long_token_f32(
        const float * __restrict__ x_ptr,
        const float * __restrict__ state_ptr,
        const float * __restrict__ weights_ptr,
        const float * __restrict__ bias_ptr,
        int x_nb0, int x_nb1, int x_nb2,
        int state_nb1, int state_nb2,
        int weights_nb1,
        float * __restrict__ dst_ptr, int dst_nb0, int dst_nb1, int dst_nb2,
        int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x = (const float *) ((const char *) x_ptr +
            bidx*x_nb2 + bidy*split_d_inner*x_nb0);
    const float * state = (const float *) ((const char *) state_ptr +
            bidx*state_nb2 + bidy*split_d_inner*state_nb1);
    const float * weights = (const float *) ((const char *) weights_ptr +
            bidy*split_d_inner*weights_nb1);
    float * dst = (float *) ((char *) dst_ptr +
            bidx*dst_nb2 + bidz*split_n_t*dst_nb1 + bidy*split_d_inner*dst_nb0);

    const int stride_x     = x_nb1 / sizeof(float);
    const int stride_state = state_nb1 / sizeof(float);
    const int stride_w     = weights_nb1 / sizeof(float);
    const int stride_dst   = dst_nb1 / sizeof(float);

    constexpr int load_cols = d_conv - 1 + split_n_t;
    extern __shared__ float smem[];

    // x is channel-major, so one thread owns one channel and the warp loads a
    // contiguous span of channels for each token. For Qwen's d_conv=4, the
    // 35-float destination stride also avoids shared-memory bank conflicts.
#pragma unroll
    for (int col = 0; col < load_cols; ++col) {
        const int64_t pos = (int64_t) bidz*split_n_t + col;
        smem[tid*load_cols + col] = pos < (int64_t) d_conv - 1
            ? state[tid*stride_state + pos]
            : x[tid + (pos - (d_conv - 1))*stride_x];
    }
    __syncthreads();

    float w[d_conv];
#pragma unroll
    for (size_t j = 0; j < d_conv; ++j) {
        w[j] = weights[tid*stride_w + j];
    }
    const float bias = bias_ptr ? bias_ptr[bidy*split_d_inner + tid] : 0.0f;
    const int64_t local_n_t = min(split_n_t, n_t - (int64_t) bidz*split_n_t);

    for (int64_t it = 0; it < local_n_t; ++it) {
        float sum = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; ++j) {
            sum += smem[tid*load_cols + it + j]*w[j];
        }
        sum += bias;
        dst[it*stride_dst + tid] = apply_silu ? ggml_cuda_op_silu_single(sum) : sum;
    }
}

template <bool apply_silu>
static void ssm_conv_state_f32_cuda(
        const float * x, const float * state, const float * weights, const float * bias,
        int x_nb0, int x_nb1, int x_nb2, int state_nb1, int state_nb2, int weights_nb1,
        float * dst, int dst_nb0, int dst_nb1, int dst_nb2,
        int64_t nc, int64_t nr, int64_t n_t, int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, nr/threads, 1);
            const ggml_cuda_kernel_launch_params launch_params =
                ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_state_f32<apply_silu, threads, kNC>, launch_params,
                    x, state, weights, bias, x_nb0, x_nb1, x_nb2, state_nb1, state_nb2,
                    weights_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            constexpr int64_t split_n_t = 32;
            const dim3 blocks(n_s, nr/threads, (n_t + split_n_t - 1)/split_n_t);
            const size_t smem_size = threads*(kNC - 1 + split_n_t)*sizeof(float);
            ssm_conv_state_long_token_f32<apply_silu, threads, kNC, split_n_t>
                <<<blocks, threads, smem_size, stream>>>(
                    x, state, weights, bias, x_nb0, x_nb1, x_nb2, state_nb1, state_nb2,
                    weights_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const struct ggml_tensor * state = dst->src[2]; // optional separate history
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = state ? src0->ne[0] : src0->ne[1]; // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    if (state) {
        GGML_ASSERT(state->type == GGML_TYPE_F32);
        GGML_ASSERT(state->ne[0] == nc - 1 && state->ne[1] == nr && state->ne[2] == n_s);
        GGML_ASSERT(state->nb[0] == sizeof(float));
        GGML_ASSERT(state->nb[1] == state->ne[0] * sizeof(float));
    }

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (state && fuse_silu) {
        ssm_conv_state_f32_cuda<true>(src0_d, (const float *) state->data, src1_d, bias_d,
                          src0->nb[0], src0->nb[1], src0->nb[2], state->nb[1], state->nb[2], src1->nb[1],
                          dst_d, out->nb[0], out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    } else if (state) {
        ssm_conv_state_f32_cuda<false>(src0_d, (const float *) state->data, src1_d, bias_d,
                          src0->nb[0], src0->nb[1], src0->nb[2], state->nb[1], state->nb[2], src1->nb[1],
                          dst_d, out->nb[0], out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    } else if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
