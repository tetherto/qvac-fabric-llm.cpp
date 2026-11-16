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
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

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

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ragged conv: one thread per channel walks a block of tokens with a sliding window, which starts from the
// seq's conv state at each seq start. Threads of a block are consecutive channels, so x and y are coalesced.
template <bool apply_silu, int d_conv>
static __global__ void ssm_conv_ragged_f32(
        const float * __restrict__ x, const float * __restrict__ state, const float * __restrict__ w,
        const int32_t * __restrict__ seq_off, float * __restrict__ y, float * __restrict__ state_out,
        const int64_t C, const int64_t x_stride, const int n_seqs, const int n_tokens, const int tokens_per_block) {
    constexpr int n_win = d_conv - 1;

    const int64_t c       = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int     t_start = blockIdx.y * tokens_per_block;
    const int     t_end   = min(t_start + tokens_per_block, n_tokens);

    ggml_cuda_pdl_sync();

    if (c >= C) {
        return;
    }

    // the seq of the first token of the block
    int lo = 0;
    int hi = n_seqs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (seq_off[mid] <= t_start) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int s  = lo;
    int t0 = seq_off[s];
    int t1 = seq_off[s + 1];

    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = w[c * d_conv + j];
    }

    // input p of the current seq: its conv state before t0, then x
    auto load = [&](const int p) -> float {
        return p >= t0 ? x[(int64_t) p * x_stride + c] : state[((int64_t) s * C + c) * n_win + n_win + p - t0];
    };

    float win[d_conv];
    bool  fresh = true;

    for (int t = t_start; t < t_end; ++t) {
        while (t >= t1) {
            ++s;
            t0    = t1;
            t1    = seq_off[s + 1];
            fresh = true;
        }
        if (fresh) {
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                win[j] = load(t - n_win + j);
            }
            fresh = false;
        } else {
#pragma unroll
            for (int j = 0; j < n_win; ++j) {
                win[j] = win[j + 1];
            }
            win[n_win] = load(t);
        }

        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sum += win[j] * wr[j];
        }
        y[(int64_t) t * C + c] = apply_silu ? ggml_cuda_op_silu_single(sum) : sum;

        if (t == t1 - 1) {
            // the window of the last token holds the last n_win inputs of the seq
#pragma unroll
            for (int j = 0; j < n_win; ++j) {
                state_out[((int64_t) s * C + c) * n_win + j] = win[j + 1];
            }
        }
    }
}

void ggml_cuda_op_ssm_conv_ragged(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_x     = dst->src[0];
    const ggml_tensor * src_state = dst->src[1];
    const ggml_tensor * src_w     = dst->src[2];
    const ggml_tensor * src_off   = dst->src[3];

    const int64_t C        = src_x->ne[0];
    const int64_t n_tokens = src_x->ne[1];
    const int64_t d_conv   = src_w->ne[0];
    const int64_t n_seqs   = src_state->ne[2];
    const bool    silu     = ggml_get_op_params_i32(dst, 0) != 0;

    GGML_ASSERT(src_x->nb[0] == sizeof(float) && src_x->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(n_tokens <= INT_MAX && n_seqs <= INT_MAX);

    const float * x_d     = (const float *) src_x->data;
    const float * state_d = (const float *) src_state->data;
    const float * w_d     = (const float *) src_w->data;
    const int32_t * off_d = (const int32_t *) src_off->data;
    float *       y_d     = (float *) dst->data;
    float *       st_d    = y_d + C * n_tokens;

    const int64_t x_stride = src_x->nb[1] / sizeof(float);

    const int  threads          = 128;
    const int  tokens_per_block = 32;
    const dim3 blocks((C + threads - 1) / threads, (n_tokens + tokens_per_block - 1) / tokens_per_block, 1);

    cudaStream_t stream = ctx.stream();

    auto launch = [&](auto D) {
        constexpr int kD = decltype(D)::value;
        const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
        if (silu) {
            ggml_cuda_kernel_launch(ssm_conv_ragged_f32<true, kD>, params, x_d, state_d, w_d, off_d, y_d, st_d,
                C, x_stride, (int) n_seqs, (int) n_tokens, tokens_per_block);
        } else {
            ggml_cuda_kernel_launch(ssm_conv_ragged_f32<false, kD>, params, x_d, state_d, w_d, off_d, y_d, st_d,
                C, x_stride, (int) n_seqs, (int) n_tokens, tokens_per_block);
        }
    };

    switch (d_conv) {
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        default: GGML_ABORT("unsupported d_conv %d for the ragged conv", (int) d_conv);
    }
}
