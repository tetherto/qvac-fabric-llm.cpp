#include "gated_delta_net_deferred.cuh"

#include <cstdint>

#ifdef GGML_USE_MUSA
// the MUSA compiler crashes on these kernels, so the ops are not supported there (see ggml_backend_cuda_device_supports_op)
void ggml_cuda_op_gated_delta_net_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * base_out) {
    GGML_UNUSED_VARS(ctx, dst, base_out);
    GGML_ABORT("GGML_OP_GATED_DELTA_NET_DEFERRED is not supported on MUSA");
}

void ggml_cuda_op_gdn_conv_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("GGML_OP_GDN_CONV_DEFERRED is not supported on MUSA");
}
#else
// gated delta net with a deferred state commit, see ggml_gated_delta_net_deferred. one warp per state column, as in
// gated_delta_net_cuda and with the same arithmetic: replay the pending recorded rows into the base, then run the
// block, writing the new base when the block reaches it and recording the last R rows.
// the base may be written in place: each warp reads its column before writing it and no other warp touches it.
template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_deferred_cuda(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * state, float * rec, const int32_t * rec_idx, const float * dt, const float * a,
        float * attn_out, float * base_out,
        int64_t H, int64_t H_k, int64_t n_tokens, int K,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sk1, int64_t sk2, int64_t sk3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sg1, int64_t sg2, int64_t sg3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t ss1, int64_t ss2, int64_t ss3,
        int64_t s_rec, int64_t rec_v, int64_t rec_g, int64_t rec_b,
        const uint3 hk_magic, float scale) {
    const uint32_t h   = blockIdx.x;
    const uint32_t seq = blockIdx.y;
    const int      lane = threadIdx.x;
    const int      col  = blockIdx.z * blockDim.y + threadIdx.y;

    constexpr int G         = KDA ? S_v : 1;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;

    const uint32_t hk = fastmodulo(h, hk_magic);

    // before the first read: the previous kernels may write the base state
    ggml_cuda_pdl_sync();

    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous
    float s_shard[rows_per_lane];
    const float * s_in = state + seq * ss3 + h * ss2 + col * ss1;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = s_in[r * warp_size + lane];
    }

    const int32_t * idx     = rec_idx + 3 * seq;
    const int       pending = max(0, min(idx[0], K));

    const float * rec_in  = rec + (int64_t) idx[1] * s_rec;
    float *       rec_out = rec + (int64_t) idx[2] * s_rec;

    // one recurrence step, the same arithmetic as gated_delta_net_cuda
    auto step = [&](const float * k_reg, float v_col, const float * g_ptr, float g_val, float beta_val) {
        if constexpr (!KDA) {
            const float g_exp = expf(g_val);
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_col - g_exp * kv_col) * beta_val;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = g_exp * s_shard[r] + k_reg[r] * delta_col;
            }
        } else {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += expf(g_ptr[r * warp_size + lane]) * s_shard[r] * k_reg[r];
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_col - kv_col) * beta_val;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = expf(g_ptr[r * warp_size + lane]) * s_shard[r] + k_reg[r] * delta_col;
            }
        }
    };

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];

    for (int p = 0; p < pending; p++) {
        const float * k_p = rec_in + (p * H_k + hk) * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_p[r * warp_size + lane];
        }
        const float * g_p = rec_in + rec_g + (p * H + h) * G;
        step(k_reg, rec_in[rec_v + (p * H + h) * S_v + col], g_p, g_p[0], rec_in[rec_b + p * H + h]);
    }

    const int R      = min((int) n_tokens, K);
    const int t_base = (int) n_tokens - R;

    float * base = base_out + (seq * H + h) * S_v * S_v + col * S_v;
    float * attn = attn_out + (seq * n_tokens * H + h) * S_v;

    const float dt_h = dt ? dt[h] : 0.0f;
    const float a_h  = a  ? a[h]  : 0.0f;

    for (int t = 0; t < n_tokens; t++) {
        if (t == t_base) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                base[r * warp_size + lane] = s_shard[r];
            }
        }

        const float * q_t = q + seq * sq3 + t * sq2 + hk * sq1;
        const float * k_t = k + seq * sk3 + t * sk2 + hk * sk1;
        const float * v_t = v + seq * sv3 + t * sv2 + h * sv1;
        const float * g_t = g + seq * sg3 + t * sg2 + h * sg1;

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_t[r * warp_size + lane];
            q_reg[r] = q_t[r * warp_size + lane];
        }

        const float v_col = v_t[col];

        // raw gates (dt, a set): g = softplus(g + dt) * a, beta = sigmoid(beta)
        float g_val    = g_t[0];
        float beta_val = beta[seq * sb3 + t * sb2 + h * sb1];
        if (dt) {
            const float x = g_val + dt_h;
            g_val    = ((x > 20.0f) ? x : logf(1.0f + expf(x))) * a_h;
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        if (t >= t_base) {
            const int rr = t - t_base;
            if (h < H_k && col == 0) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    rec_out[(rr * H_k + h) * S_v + r * warp_size + lane] = k_reg[r];
                }
            }
            if (lane == 0) {
                rec_out[rec_v + (rr * H + h) * S_v + col] = v_col;
            }
            if (col == 0) {
                if constexpr (KDA) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        rec_out[rec_g + (rr * H + h) * G + r * warp_size + lane] = g_t[r * warp_size + lane];
                    }
                } else if (lane == 0) {
                    rec_out[rec_g + rr * H + h] = g_val;
                }
                if (lane == 0) {
                    rec_out[rec_b + rr * H + h] = beta_val;
                }
            }
        }

        step(k_reg, v_col, g_t, g_val, beta_val);

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
        if (lane == 0) {
            attn[col] = attn_col * scale;
        }
        attn += S_v * H;
    }
}

template <bool KDA>
static void launch_gated_delta_net_deferred(int64_t S_v, dim3 grid, dim3 block, cudaStream_t stream,
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * state, float * rec, const int32_t * rec_idx, const float * dt, const float * a,
        float * attn_out, float * base_out,
        int64_t H, int64_t H_k, int64_t n_tokens, int K,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sk1, int64_t sk2, int64_t sk3,
        int64_t sv1, int64_t sv2, int64_t sv3, int64_t sg1, int64_t sg2, int64_t sg3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t ss1, int64_t ss2, int64_t ss3,
        int64_t s_rec, int64_t rec_v, int64_t rec_g, int64_t rec_b, uint3 hk_magic, float scale) {
    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
#define GDN_DEFERRED_LAUNCH(SV)                                                                                          \
    ggml_cuda_kernel_launch(gated_delta_net_deferred_cuda<SV, KDA>, params, q, k, v, g, beta, state, rec, rec_idx, dt, a, \
        attn_out, base_out, H, H_k, n_tokens, K, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sg1, sg2, sg3,             \
        sb1, sb2, sb3, ss1, ss2, ss3, s_rec, rec_v, rec_g, rec_b, hk_magic, scale)
    switch (S_v) {
        case 16:  GDN_DEFERRED_LAUNCH(16);  break;
        case 32:  GDN_DEFERRED_LAUNCH(32);  break;
        case 64:  GDN_DEFERRED_LAUNCH(64);  break;
        case 128: GDN_DEFERRED_LAUNCH(128); break;
        default:  GGML_ABORT("fatal error");
    }
#undef GDN_DEFERRED_LAUNCH
}

void ggml_cuda_op_gated_delta_net_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * base_out) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];
    const ggml_tensor * rec   = dst->src[6];
    const ggml_tensor * idx   = dst->src[7];
    const ggml_tensor * dt    = dst->src[8];
    const ggml_tensor * a     = dst->src[9];

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t H_k      = k->ne[1];
    const int64_t G        = g->ne[0];
    const int     K        = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(q->ne[1] == H_k);

    const bool kda = G == S_v;

    const int64_t rec_v = (int64_t) K * S_v * H_k;
    const int64_t rec_g = rec_v + (int64_t) K * S_v * H;
    const int64_t rec_b = rec_g + (int64_t) K * G * H;

    float * attn_out = (float *) dst->data;
    if (base_out == nullptr) {
        base_out = attn_out + S_v * H * n_tokens * n_seqs;
    }

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    const dim3 grid(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    const dim3 block(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const size_t fs = sizeof(float);

    auto launch = kda ? launch_gated_delta_net_deferred<true> : launch_gated_delta_net_deferred<false>;
    launch(S_v, grid, block, ctx.stream(),
        (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data,
        (const float *) beta->data, (const float *) state->data, (float *) rec->data, (const int32_t *) idx->data,
        dt ? (const float *) dt->data : nullptr, a ? (const float *) a->data : nullptr,
        attn_out, base_out, H, H_k, n_tokens, K,
        q->nb[1] / fs, q->nb[2] / fs, q->nb[3] / fs, k->nb[1] / fs, k->nb[2] / fs, k->nb[3] / fs,
        v->nb[1] / fs, v->nb[2] / fs, v->nb[3] / fs, g->nb[1] / fs, g->nb[2] / fs, g->nb[3] / fs,
        beta->nb[1] / fs, beta->nb[2] / fs, beta->nb[3] / fs, state->nb[1] / fs, state->nb[2] / fs, state->nb[3] / fs,
        rec->nb[1] / fs, rec_v, rec_g, rec_b, init_fastdiv_values(H_k), 1.0f / sqrtf((float) S_v));
}

// causal depthwise conv (d_conv = 4) + SiLU with a deferred conv-state commit, see ggml_gdn_conv_deferred. one block per
// (head of head_dim channels, token, seq), one thread per channel; the leading n_norm heads are L2-normalized
static __global__ void gdn_conv_deferred_cuda(
        const float * x, const float * w, float * hist, const int32_t * rec_idx, float * dst,
        int C, int n_tokens, int K, int n_norm, float eps,
        int64_t sx1, int64_t sx2, int64_t s1, int64_t s2, int64_t s_hist) {
    const int ih = blockIdx.x;
    const int t  = blockIdx.y;
    const int is = blockIdx.z;
    const int c  = ih * blockDim.x + threadIdx.x;

    const int32_t * idx     = rec_idx + 3 * is;
    const int       pending = max(0, min(idx[0], K));

    const float * h_in  = hist + (int64_t) idx[1] * s_hist;
    float *       h_out = hist + (int64_t) idx[2] * s_hist;

    const float * xs = x + is * sx2 + c;

    // conv input column i: the window for i < 3, then the new inputs
    float ci[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = t + j;
        ci[j] = i < 3 ? h_in[(int64_t) (pending + i) * C + c] : xs[(int64_t) (i - 3) * sx1];
    }

    const float * wc = w + 4 * c;

    float sumf = 0.0f;
    sumf += ci[0] * wc[0];
    sumf += ci[1] * wc[1];
    sumf += ci[2] * wc[2];
    sumf += ci[3] * wc[3];

    float y = sumf / (1.0f + expf(-sumf));

    if (ih < n_norm) {
        __shared__ float sh[32];
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        const float sq = warp_reduce_sum(y * y);
        if (lane == 0) {
            sh[warp] = sq;
        }
        __syncthreads();
        float tot = 0.0f;
        for (int i = 0; i < (int) blockDim.x / WARP_SIZE; ++i) {
            tot += sh[i];
        }
        y *= 1.0f / sqrtf(tot + eps);
    }

    dst[t * s1 + is * s2 + c] = y;

    // keep the last R + 3 conv input columns, i.e. column i - (n_tokens - R) for i >= n_tokens - R
    const int i0 = n_tokens - min(n_tokens, K);
    if (t + 3 >= i0) {
        h_out[(int64_t) (t + 3 - i0) * C + c] = ci[3];
    }
    if (t == 0) {
        for (int i = i0; i < 3; ++i) {
            h_out[(int64_t) (i - i0) * C + c] = ci[i];
        }
    }
}

void ggml_cuda_op_gdn_conv_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * w    = dst->src[1];
    const ggml_tensor * hist = dst->src[2];
    const ggml_tensor * idx  = dst->src[3];

    const int C        = x->ne[0];
    const int n_tokens = x->ne[1];
    const int n_seqs   = x->ne[2];
    const int K        = ggml_get_op_params_i32(dst, 0);
    const int n_norm   = ggml_get_op_params_i32(dst, 1);
    const int head_dim = ggml_get_op_params_i32(dst, 2);
    const float eps    = ggml_get_op_params_f32(dst, 3);

    GGML_ASSERT(w->ne[0] == 4 && C % head_dim == 0 && head_dim % WARP_SIZE == 0 && head_dim <= 1024);

    const size_t fs = sizeof(float);

    const dim3 grid(C / head_dim, n_tokens, n_seqs);
    gdn_conv_deferred_cuda<<<grid, head_dim, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) w->data, (float *) hist->data, (const int32_t *) idx->data,
        (float *) dst->data, C, n_tokens, K, n_norm, eps,
        x->nb[1] / fs, x->nb[2] / fs, dst->nb[1] / fs, dst->nb[2] / fs, hist->nb[1] / fs);
}
#endif // GGML_USE_MUSA
