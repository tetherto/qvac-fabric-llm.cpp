#include "common.cuh"
#include "conv-state.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

// the block covers one head of a norm slice (or none): its threads reduce the squares of the token's outputs and
// write the normalized row; the two shared buffers alternate per token so a fast warp never overwrites values a slow
// warp still reads from the previous reduction
template <int split_d_inner>
static __device__ __forceinline__ int ssm_conv_l2_slice(const ggml_cuda_ssm_conv_l2 & l2, const int ch_base) {
    int res = -1;
#pragma unroll
    for (int r = 0; r < GGML_CUDA_SSM_CONV_MAX_L2; r++) {
        if (r < l2.n && ch_base >= l2.ch0[r] && ch_base + split_d_inner <= l2.ch1[r]) {
            res = r;
        }
    }
    return res;
}

template <int split_d_inner>
static __device__ __forceinline__ void ssm_conv_l2_row(const ggml_cuda_ssm_conv_l2 & l2, const int r, const int ch_base,
                                                       const int64_t token, const int seq, const int tid, const float y,
                                                       float * s_red) {
    const float ss    = block_reduce<block_reduce_method::SUM, split_d_inner>(y*y, s_red);
    const float scale = rsqrtf(fmaxf(ss, l2.eps[r]*l2.eps[r])) * l2.scale[r];
    const int64_t head = (ch_base - l2.ch0[r]) / split_d_inner;
    l2.dst[r][head*l2.nb1[r] + token*l2.nb2[r] + seq*l2.nb3[r] + tid] = y*scale;
}

template <bool apply_silu, bool with_l2, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t, const ggml_cuda_ssm_conv_l2 l2) {
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

    __shared__ float s_red[2][WARP_SIZE];
    const int ch_base  = bidy * split_d_inner;
    const int l2_slice = with_l2 ? ssm_conv_l2_slice<split_d_inner>(l2, ch_base) : -1;

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
        const float y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y_block[i * stride_y + tid] = y;
        if constexpr (with_l2) {
            if (l2_slice >= 0) { // block-uniform
                ssm_conv_l2_row<split_d_inner>(l2, l2_slice, ch_base, i, bidx, tid, y, s_red[i & 1]);
            }
        }
    }
}

template <bool apply_silu, bool with_l2, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t,
                                               const ggml_cuda_ssm_conv_l2 l2) {
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
    __shared__ float s_red[2][WARP_SIZE];
    const int ch_base  = bidy * split_d_inner;
    const int l2_slice = with_l2 ? ssm_conv_l2_slice<split_d_inner>(l2, ch_base) : -1;

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
        const float y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y_block[i * stride_y + tid] = y;
        if constexpr (with_l2) {
            if (l2_slice >= 0) { // block-uniform
                ssm_conv_l2_row<split_d_inner>(l2, l2_slice, ch_base, bidz * split_n_t + i, bidx, tid, y, s_red[i & 1]);
            }
        }
    }
}

template <bool apply_silu, bool with_l2>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, const ggml_cuda_ssm_conv_l2 & l2, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, with_l2, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t, l2);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, with_l2, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t, l2);
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

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst,
                           const ggml_cuda_ssm_conv_l2 * l2) {
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

    // the norm slices come with the silu (ggml_cuda_try_ssm_conv_l2_fusion)
    GGML_ASSERT(l2 == nullptr || fuse_silu);
    const ggml_cuda_ssm_conv_l2 no_l2 = {};

    if (fuse_silu && l2 != nullptr && l2->n > 0) {
        ssm_conv_f32_cuda<true, true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, *l2, stream);
    } else if (fuse_silu) {
        ssm_conv_f32_cuda<true, false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, no_l2, stream);
    } else {
        ssm_conv_f32_cuda<false, false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, no_l2, stream);
    }
}

// one block = 128 consecutive channels (one head of a norm slice) x a tile of split_t tokens. The tile's inputs are
// loaded into registers up front (split_t independent loads per thread, so the loop is not one memory round trip per
// token), the conv runs from registers with the same accumulation order as ssm_conv_long_token_f32 (bit-identical
// sums), and the per-token L2 norms of a slice are reduced in one batch: a warp reduction per token, one shared
// exchange and one barrier for the tile instead of a block reduction per token
template <int d_conv, bool with_l2, int split_t>
static __global__ void ssm_conv_tokens_f32(const ggml_cuda_ssm_conv_tokens_args a, const ggml_cuda_ssm_conv_l2 l2) {
    constexpr int split_ch = 128;
    constexpr int n_warps  = split_ch/WARP_SIZE;
    ggml_cuda_pdl_lc();
    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid % WARP_SIZE;
    const int ch_base   = blockIdx.x*split_ch;
    const int c         = ch_base + tid;
    const int t0        = blockIdx.y*split_t;
    const int local_n_t = min(split_t, a.n_t - t0);

    __shared__ float s_red[n_warps][split_t];
    const int l2_slice = with_l2 ? ssm_conv_l2_slice<split_ch>(l2, ch_base) : -1;

    // weights and bias never depend on the previous kernel: load them before the grid dependency
    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        w[j] = a.w[c*a.w_stride + j];
    }
    const float b = a.bias != nullptr ? a.bias[c] : 0.0f;

    ggml_cuda_pdl_sync();
    float win[d_conv];
    if (t0 == 0) {
        const float * st = a.states + a.idx[0]*a.states_stride + c*(d_conv - 1);
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            win[j] = st[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            win[j] = a.x[(t0 - (d_conv - 1) + j)*a.x_stride_t + c];
        }
    }
    float xt[split_t];
#pragma unroll
    for (int i = 0; i < split_t; i++) {
        xt[i] = i < local_n_t ? a.x[(t0 + i)*a.x_stride_t + c] : 0.0f;
    }

    float yv[split_t];
#pragma unroll
    for (int i = 0; i < split_t; i++) {
        win[d_conv - 1] = xt[i];
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += win[j]*w[j];
        }
        sumf += b;
        yv[i] = ggml_cuda_op_silu_single(sumf);
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            win[j] = win[j + 1];
        }
    }

    if (a.y != nullptr) { // null when the bf16 packs are the only consumers' copy
#pragma unroll
        for (int i = 0; i < split_t; i++) {
            if (i < local_n_t) {
                a.y[(t0 + i)*(int64_t) a.n_ch + c] = yv[i];
            }
        }
    }
    if (a.v_pack != nullptr && c >= a.v_ch0 && c < a.v_ch0 + a.v_nch) { // block-uniform
#pragma unroll
        for (int i = 0; i < split_t; i++) {
            if (i < local_n_t) {
                a.v_pack[(t0 + i)*(int64_t) a.v_nch + (c - a.v_ch0)] = __float2bfloat16_rn(yv[i]);
            }
        }
    }

    if constexpr (with_l2) {
        if (l2_slice >= 0) { // block-uniform
            // constant indices only: a dynamic index into the parameter struct would copy it to local memory per thread
            int     ch0   = l2.ch0[0];
            int     ch1   = l2.ch1[0];
            float   eps   = l2.eps[0];
            float   scl   = l2.scale[0];
            float * dst   = l2.dst[0];
            int64_t nb1   = l2.nb1[0];
            int64_t nb2   = l2.nb2[0];
            nv_bfloat16 * pack = l2.pack[0];
#pragma unroll
            for (int r = 1; r < GGML_CUDA_SSM_CONV_MAX_L2; r++) {
                if (r == l2_slice) {
                    ch0  = l2.ch0[r];
                    ch1  = l2.ch1[r];
                    eps  = l2.eps[r];
                    scl  = l2.scale[r];
                    dst  = l2.dst[r];
                    nb1  = l2.nb1[r];
                    nb2  = l2.nb2[r];
                    pack = l2.pack[r];
                }
            }
#pragma unroll
            for (int i = 0; i < split_t; i++) {
                const float ss = warp_reduce_sum(yv[i]*yv[i]);
                if (lane == 0) {
                    s_red[warp][i] = ss;
                }
            }
            __syncthreads();
            const int     n_heads = (ch1 - ch0) / split_ch;
            const int64_t head    = (ch_base - ch0) / split_ch;
#pragma unroll
            for (int i = 0; i < split_t; i++) {
                if (i < local_n_t) {
                    // the warp partials are combined in block_reduce's order (a shuffle tree over lanes 0..n_warps-1),
                    // so the sums equal the per-token block_reduce bit for bit. The order is load-bearing: a one-ulp
                    // change of this sum moves the Qwen3.8 GX gate from exact to 0.0136 mean KLD (the delta-net state
                    // amplifies it), so a sequential sum of the partials is not an option
                    const float ss    = warp_reduce_sum(lane < n_warps ? s_red[lane][i] : 0.0f);
                    const float scale = rsqrtf(fmaxf(ss, eps*eps)) * scl;
                    const float o     = yv[i]*scale;
                    if (dst != nullptr) {
                        dst[head*nb1 + (int64_t) (t0 + i)*nb2 + tid] = o;
                    }
                    if (pack != nullptr) {
                        const nv_bfloat16 ob = __float2bfloat16_rn(o);
                        for (int hv = head; hv < l2.pack_heads; hv += n_heads) {
                            pack[((t0 + i)*(int64_t) l2.pack_heads + hv)*split_ch + tid] = ob;
                        }
                    }
                }
            }
        }
    }

    // the next state row = the last d_conv-1 tokens; only the first tile's blocks read the state row, and the same
    // thread read its elements above, so writing the row in place is safe
    if (blockIdx.y == 0) {
        float * cache = a.cache + c*(d_conv - 1);
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            cache[j] = a.x[(a.n_t - (d_conv - 1) + j)*a.x_stride_t + c];
        }
    }
}

bool ggml_cuda_ssm_conv_tokens_supported(const ggml_cuda_ssm_conv_tokens_args & a) {
    return (a.d_conv == 3 || a.d_conv == 4 || a.d_conv == 5 || a.d_conv == 9) &&
        a.n_ch % 128 == 0 && a.n_t > GGML_CUDA_CONV_STATE_MAX_T;
}

void ggml_cuda_op_ssm_conv_tokens(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_tokens_args & a, const ggml_cuda_ssm_conv_l2 & l2) {
    GGML_ASSERT(ggml_cuda_ssm_conv_tokens_supported(a));
    constexpr int split_t = 32;
    const dim3 blocks(a.n_ch/128, (a.n_t + split_t - 1)/split_t, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, 128, 0, ctx.stream());

    auto launch = [&](auto NC) {
        constexpr int d_conv = decltype(NC)::value;
        if (l2.n > 0) {
            ggml_cuda_kernel_launch(ssm_conv_tokens_f32<d_conv, true, split_t>, launch_params, a, l2);
        } else {
            ggml_cuda_kernel_launch(ssm_conv_tokens_f32<d_conv, false, split_t>, launch_params, a, l2);
        }
    };
    switch (a.d_conv) {
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 9: launch(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("ssm_conv_tokens: unsupported d_conv");
    }
}
