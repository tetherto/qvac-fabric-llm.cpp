#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(__linux__)
#include <dlfcn.h>
#endif
#endif

// serial kernel: one dependent step per token; n_tokens_dst is the token count of the whole op (dst row stride), n_tokens the count processed here
template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_tokens_dst,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs], seq stride is D = H * S_v * S_v
    // output state layout (per-slot D * n_seqs), same per-(seq,head) offset as before
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens_dst * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// chunked prefill kernel (scalar gate only): WY form of the gated delta rule over chunks of GDN_CHUNK tokens
// one block per (head, sequence, GDN_CHUNK_COLS state columns); with D[t][s] = exp(G_t - G_s) for the inclusive log-gate G:
//   A = beta_t (k_t . k_s) D (s < t), U = (I + A)^-1 (beta v - beta exp(G) S0^T k)
//   o_t = scale (exp(G_t) S0^T q_t + sum_{s<=t} (q_t . k_s) D[t][s] U_s), S = exp(G_last) S0 + sum_s D[last][s] k_s U_s^T
#define GDN_CHUNK          64
#define GDN_CHUNK_COLS     32
#define GDN_CHUNK_GROUPS    8
#define GDN_CHUNK_THREADS (GDN_CHUNK_GROUPS * GDN_CHUNK_COLS)

// padded shared memory row strides, chosen so that mma fragment loads hit distinct (or at most pairwise shared) banks;
// the total stays under half of the H100 shared memory so two blocks fit on one SM
#define GDN_CHUNK_KS(S_v) ((S_v) + 4)             // k and q tiles [C][S_v]
#define GDN_CHUNK_MS      (GDN_CHUNK + 4)         // C x C matrix
#define GDN_CHUNK_US      (GDN_CHUNK_COLS + 8)    // U [C][CG]
#define GDN_CHUNK_SS      (GDN_CHUNK_COLS + 4)    // state [S_v][CG]

template <int S_v>
static constexpr size_t gdn_chunk_smem_bytes() {
    return sizeof(float) * (
        2 * GDN_CHUNK * GDN_CHUNK_KS(S_v) +   // k and q tiles
        GDN_CHUNK * GDN_CHUNK_MS +            // A (solve), then B (output)
        GDN_CHUNK * GDN_CHUNK_US +            // U
        S_v * GDN_CHUNK_SS +                  // state columns of this block
        2 * GDN_CHUNK);                       // cumulative gate, beta
}

// TF32 mma.m16n8k8 with fp32 accumulation; lane = 4*r + c: A regs hold rows r, r+8 for two k slots, B regs column r,
// D regs (r, 2c), (r, 2c+1), (r+8, 2c), (r+8, 2c+1); A and B use the same k slot order so the slot assignment does not matter
static __device__ __forceinline__ uint32_t gdn_tf32(float x) {
    uint32_t r;
#ifdef AMPERE_MMA_AVAILABLE
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
#else
    r = __float_as_uint(x);
#endif // AMPERE_MMA_AVAILABLE
    return r;
}

static __device__ __forceinline__ void gdn_mma(float * d, const float * a, const float * b) {
#ifdef AMPERE_MMA_AVAILABLE
    const uint32_t a0 = gdn_tf32(a[0]);
    const uint32_t a1 = gdn_tf32(a[1]);
    const uint32_t a2 = gdn_tf32(a[2]);
    const uint32_t a3 = gdn_tf32(a[3]);
    const uint32_t b0 = gdn_tf32(b[0]);
    const uint32_t b1 = gdn_tf32(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// A operand fragment of a row-major [rows][k] matrix with row stride ld, rows r0.., k slots kk + c, kk + c + 4
static __device__ __forceinline__ void gdn_frag_a(float * a, const float * X, const int ld, const int r0, const int kk, const int r, const int c) {
    a[0] = X[(r0 + r    )*ld + kk + c];
    a[1] = X[(r0 + r + 8)*ld + kk + c];
    a[2] = X[(r0 + r    )*ld + kk + c + 4];
    a[3] = X[(r0 + r + 8)*ld + kk + c + 4];
}

// A operand fragment of the transpose of a row-major [k][rows] matrix (element (row, k) = X[k*ld + row])
static __device__ __forceinline__ void gdn_frag_at(float * a, const float * X, const int ld, const int r0, const int kk, const int r, const int c) {
    a[0] = X[(kk + c    )*ld + r0 + r];
    a[1] = X[(kk + c    )*ld + r0 + r + 8];
    a[2] = X[(kk + c + 4)*ld + r0 + r];
    a[3] = X[(kk + c + 4)*ld + r0 + r + 8];
}

// B operand fragment of a row-major [n][k] matrix (element (k, n) = Y[n*ld + k]), columns n0.., k slots kk + c, kk + c + 4
static __device__ __forceinline__ void gdn_frag_bt(float * b, const float * Y, const int ld, const int n0, const int kk, const int r, const int c) {
    b[0] = Y[(n0 + r)*ld + kk + c];
    b[1] = Y[(n0 + r)*ld + kk + c + 4];
}

// B operand fragment of a row-major [k][n] matrix
static __device__ __forceinline__ void gdn_frag_b(float * b, const float * Y, const int ld, const int n0, const int kk, const int r, const int c) {
    b[0] = Y[(kk + c    )*ld + n0 + r];
    b[1] = Y[(kk + c + 4)*ld + n0 + r];
}

// M[t][s] = f(t, s) * (X_t . Y_s) for the lower-triangular tiles (s <= t); entries above the diagonal inside those
// tiles are written as 0 so a later mma over the row band reads a clean lower-triangular matrix. X and Y are shared tiles.
template <int S_v, bool STRICT>
static __device__ __forceinline__ void gdn_chunk_gram(float * M, const float * X, const float * Y,
        const float * Gs, const float * Bs, const int warp, const int r, const int c) {
    constexpr int C  = GDN_CHUNK;
    constexpr int KS = GDN_CHUNK_KS(S_v);
    constexpr int MS = GDN_CHUNK_MS;
    for (int tile = warp; tile < (C/16)*(C/8); tile += GDN_CHUNK_GROUPS) {
        const int tm = tile / (C/8);
        const int tn = tile % (C/8);
        if (8*tn > 16*tm + 15) {
            continue;
        }
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int kk = 0; kk < S_v; kk += 8) {
            float a[4], b[2];
            gdn_frag_a(a, X, KS, 16*tm, kk, r, c);
            gdn_frag_bt(b, Y, KS, 8*tn, kk, r, c);
            gdn_mma(d, a, b);
        }
        const int t = 16*tm + r;
        const int s = 8*tn + 2*c;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int te = t + 8*(e/2);
            const int se = s + (e%2);
            const bool keep = STRICT ? se < te : se <= te;
            const float f = STRICT ? Bs[te] * expf(Gs[te] - Gs[se]) : expf(Gs[te] - Gs[se]);
            M[te*MS + se] = keep ? f * d[e] : 0.0f;
        }
    }
}

template <int S_v>
__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 2)
gated_delta_net_chunked_cuda(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             float *       state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_tokens_dst,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale) {
    constexpr int C  = GDN_CHUNK;
    constexpr int CG = GDN_CHUNK_COLS;
    constexpr int NW = GDN_CHUNK_GROUPS;
    constexpr int KS = GDN_CHUNK_KS(S_v);
    constexpr int MS = GDN_CHUNK_MS;
    constexpr int US = GDN_CHUNK_US;
    constexpr int SS = GDN_CHUNK_SS;
    constexpr int NT_O = (C/16) * (CG/8);   // output tiles [C][CG]
    constexpr int NT_S = (S_v/16) * (CG/8); // state tiles [S_v][CG]
    static_assert(NT_O % NW == 0 && NT_S % NW == 0, "tile counts must split evenly over the warps");

    extern __shared__ float smem[];
    float * Ks = smem;
    float * Qs = Ks + C*KS;
    float * M  = Qs + C*KS;
    float * Us = M  + C*MS;
    float * Ss = Us + C*US;
    float * Gs = Ss + S_v*SS;
    float * Bs = Gs + C;

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      c0       = blockIdx.z * CG;
    const int      tid      = threadIdx.x;
    const int      warp     = tid / 32;
    const int      lane     = tid % 32;
    const int      r        = lane / 4;
    const int      c        = lane % 4;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base  = q + iq3*sq3 + iq1*sq1;
    const float * k_base  = k + iq3*sq3 + iq1*sq1;
    const float * v_base  = v + sequence*sv3 + h_idx*sv1 + c0;
    const int64_t gb_base = sequence*sb3 + h_idx*sb1;

    const int64_t state_off = (sequence*H + h_idx) * S_v * S_v + (int64_t) c0 * S_v;
    float * attn = dst + (sequence*n_tokens_dst*H + h_idx) * S_v + c0;

    ggml_cuda_pdl_sync();

    // the state is stored transposed in memory (row col is contiguous over i); Ss holds it as [i][col]
    for (int idx = tid; idx < S_v*CG; idx += GDN_CHUNK_THREADS) {
        const int n = idx / S_v;
        const int i = idx % S_v;
        Ss[i*SS + n] = curr_state[state_off + n*S_v + i];
    }

    // rows are contiguous; the float4 path needs 16-byte aligned row starts (true for the model graph, not for every view)
    const bool vec4 = (sq2 % 4 == 0) && ((reinterpret_cast<uintptr_t>(q_base) | reinterpret_cast<uintptr_t>(k_base)) % 16 == 0);

    for (int64_t t0 = 0; t0 < n_tokens; t0 += C) {
        const float * q_chunk = q_base + t0*sq2;
        const float * k_chunk = k_base + t0*sq2;
        if (vec4) {
            for (int idx = tid; idx < C*S_v/4; idx += GDN_CHUNK_THREADS) {
                const int t = idx / (S_v/4);
                const int i = 4*(idx % (S_v/4));
                const float4 kv = *(const float4 *) (k_chunk + t*sq2 + i);
                const float4 qv = *(const float4 *) (q_chunk + t*sq2 + i);
                Ks[t*KS + i] = kv.x; Ks[t*KS + i + 1] = kv.y; Ks[t*KS + i + 2] = kv.z; Ks[t*KS + i + 3] = kv.w;
                Qs[t*KS + i] = qv.x; Qs[t*KS + i + 1] = qv.y; Qs[t*KS + i + 2] = qv.z; Qs[t*KS + i + 3] = qv.w;
            }
        } else {
            for (int idx = tid; idx < C*S_v; idx += GDN_CHUNK_THREADS) {
                const int t = idx / S_v;
                const int i = idx % S_v;
                Ks[t*KS + i] = k_chunk[t*sq2 + i];
                Qs[t*KS + i] = q_chunk[t*sq2 + i];
            }
        }
        if (tid < C) {
            Bs[tid] = beta[gb_base + (t0 + tid)*sb2];
            Gs[tid] = g   [gb_base + (t0 + tid)*sb2];
        }
        __syncthreads();
        if (warp == 0) {
            // inclusive cumulative log-gate: two-level warp scan over 64 values
            float x0 = Gs[lane];
            float x1 = Gs[32 + lane];
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float y0 = __shfl_up_sync(0xffffffff, x0, off, 32);
                const float y1 = __shfl_up_sync(0xffffffff, x1, off, 32);
                if (lane >= off) {
                    x0 += y0;
                    x1 += y1;
                }
            }
            x1 += __shfl_sync(0xffffffff, x0, 31, 32);
            Gs[lane]      = x0;
            Gs[32 + lane] = x1;
        }
        __syncthreads();

        // A = beta_t (k_t . k_s) exp(G_t - G_s), s < t
        gdn_chunk_gram<S_v, true>(M, Ks, Ks, Gs, Bs, warp, r, c);

        // rhs = beta (v - exp(G) S0^T k) into U; S0^T q kept in registers for the output tiles this warp owns
        float qs0[NT_O/NW][4];
#pragma unroll
        for (int j = 0; j < NT_O/NW; j++) {
            const int tile = warp + NW*j;
            const int tm   = tile / (CG/8);
            const int tn   = tile % (CG/8);
            float dk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float dq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
            for (int kk = 0; kk < S_v; kk += 8) {
                float a[4], aq[4], b[2];
                gdn_frag_a(a,  Ks, KS, 16*tm, kk, r, c);
                gdn_frag_a(aq, Qs, KS, 16*tm, kk, r, c);
                gdn_frag_b(b, Ss, SS, 8*tn, kk, r, c);
                gdn_mma(dk, a, b);
                gdn_mma(dq, aq, b);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                qs0[j][e] = dq[e];
                const int t = 16*tm + r + 8*(e/2);
                const int n = 8*tn + 2*c + (e%2);
                Us[t*US + n] = Bs[t] * (v_base[(t0 + t)*sv2 + n] - expf(Gs[t]) * dk[e]);
            }
        }
        __syncthreads();

        if (warp == 0) {
            // forward substitution U[t] = rhs[t] - sum_{s<t} A[t][s] U[s], one lane per column, U column in registers
            float u[C];
#pragma unroll
            for (int t = 0; t < C; t++) {
                float acc = Us[t*US + lane];
#pragma unroll
                for (int s = 0; s < t; s++) {
                    acc -= M[t*MS + s] * u[s];
                }
                u[t] = acc;
                Us[t*US + lane] = acc;
            }
        }
        __syncthreads();

        // B = (q_t . k_s) exp(G_t - G_s), s <= t, reusing M
        gdn_chunk_gram<S_v, false>(M, Qs, Ks, Gs, Bs, warp, r, c);
        __syncthreads();

        // o = scale (exp(G_t) S0^T q_t + sum_{s<=t} B[t][s] U[s])
#pragma unroll
        for (int j = 0; j < NT_O/NW; j++) {
            const int tile = warp + NW*j;
            const int tm   = tile / (CG/8);
            const int tn   = tile % (CG/8);
            const int t    = 16*tm + r;
            float d[4];
            d[0] = expf(Gs[t])     * qs0[j][0];
            d[1] = expf(Gs[t])     * qs0[j][1];
            d[2] = expf(Gs[t + 8]) * qs0[j][2];
            d[3] = expf(Gs[t + 8]) * qs0[j][3];
            for (int kk = 0; kk < 16*tm + 16; kk += 8) {
                float a[4], b[2];
                gdn_frag_a(a, M, MS, 16*tm, kk, r, c);
                gdn_frag_b(b, Us, US, 8*tn, kk, r, c);
                gdn_mma(d, a, b);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int te = t + 8*(e/2);
                const int n  = 8*tn + 2*c + (e%2);
                attn[(t0 + te) * S_v * H + n] = scale * d[e];
            }
        }

        // S = exp(G_last) S0 + sum_s exp(G_last - G_s) k_s U_s^T; every warp updates only the tiles it reads
        const float g_last = Gs[C - 1];
#pragma unroll
        for (int j = 0; j < NT_S/NW; j++) {
            const int tile = warp + NW*j;
            const int im   = tile / (CG/8);
            const int tn   = tile % (CG/8);
            const int i    = 16*im + r;
            const int n    = 8*tn + 2*c;
            float d[4];
            d[0] = expf(g_last) * Ss[ i     *SS + n];
            d[1] = expf(g_last) * Ss[ i     *SS + n + 1];
            d[2] = expf(g_last) * Ss[(i + 8)*SS + n];
            d[3] = expf(g_last) * Ss[(i + 8)*SS + n + 1];
#pragma unroll 2
            for (int kk = 0; kk < C; kk += 8) {
                float a[4], b[2];
                gdn_frag_at(a, Ks, KS, 16*im, kk, r, c);
                gdn_frag_b(b, Us, US, 8*tn, kk, r, c);
                b[0] *= expf(g_last - Gs[kk + c]);
                b[1] *= expf(g_last - Gs[kk + c + 4]);
                gdn_mma(d, a, b);
            }
            Ss[ i     *SS + n]     = d[0];
            Ss[ i     *SS + n + 1] = d[1];
            Ss[(i + 8)*SS + n]     = d[2];
            Ss[(i + 8)*SS + n + 1] = d[3];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < S_v*CG; idx += GDN_CHUNK_THREADS) {
        const int n = idx / S_v;
        const int i = idx % S_v;
        state[state_off + n*S_v + i] = Ss[i*SS + n];
    }
}

template <int S_v>
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_tokens_dst, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    constexpr size_t smem = gdn_chunk_smem_bytes<S_v>();
    CUDA_SET_SHARED_MEMORY_LIMIT((gated_delta_net_chunked_cuda<S_v>), smem);

    const dim3 grid_dims(H, n_seqs, S_v / GDN_CHUNK_COLS);
    const dim3 block_dims(GDN_CHUNK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, smem, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<S_v>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_tokens_dst, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
}

// the chunked kernel needs S_v in {64, 128}, a scalar gate, TF32 mma and the opt-in shared memory size
static bool gdn_chunked_supported(int64_t S_v, bool kda) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(S_v); GGML_UNUSED(kda);
    return false;
#else
    if (kda || (S_v != 64 && S_v != 128)) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (!ampere_mma_available(dev.cc)) {
        return false;
    }
    const size_t smem = S_v == 64 ? gdn_chunk_smem_bytes<64>() : gdn_chunk_smem_bytes<128>();
    return dev.smpbo >= smem;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_tokens_dst, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_tokens_dst, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_tokens_dst, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_tokens_dst, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_tokens_dst, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// opt-in external SM90 fused delta-rule kernel (FlashInfer CuTe DSL AOT export), loaded from GGML_CUDA_GDN_AOT_LIB
using gdn_flashinfer_aot_launch_t = int (*)(
    const void *, const void *, const void *, void *,
    const float *, const float *, float *, const float *,
    void *, const int64_t *, int32_t, int32_t, int32_t, int32_t, int32_t,
    cudaStream_t);

static gdn_flashinfer_aot_launch_t get_gdn_flashinfer_aot_launch() {
#if defined(__linux__)
    static const gdn_flashinfer_aot_launch_t launch = [] {
        const char * path = std::getenv("GGML_CUDA_GDN_AOT_LIB");
        if (path == nullptr || path[0] == '\0') {
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            std::fprintf(stderr, "ggml_cuda: failed to load GDN AOT library %s: %s\n", path, dlerror());
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        void * symbol = dlsym(handle, "flashinfer_gdn_sm90_aot_launch");
        if (symbol == nullptr) {
            std::fprintf(stderr, "ggml_cuda: GDN AOT library is missing flashinfer_gdn_sm90_aot_launch: %s\n", dlerror());
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        std::fprintf(stderr, "ggml_cuda: using FlashInfer SM90 AOT GDN from %s\n", path);
        return reinterpret_cast<gdn_flashinfer_aot_launch_t>(symbol);
    }();
    return launch;
#else
    return nullptr;
#endif
}

__global__ void gated_delta_net_flashinfer_prepare_cuda(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        __nv_bfloat16 * q_bf16,
        __nv_bfloat16 * k_bf16,
        __nv_bfloat16 * v_bf16,
        float * alpha,
        float * beta_packed,
        int64_t S,
        int64_t H_q,
        int64_t H_v,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t rq3) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_seqs * n_tokens * H_v * S;
    if (idx >= n) {
        return;
    }

    const int64_t d = idx % S;
    const int64_t ih = (idx / S) % H_v;
    const int64_t it = (idx / (S * H_v)) % n_tokens;
    const int64_t is = idx / (S * H_v * n_tokens);

    v_bf16[idx] = __float2bfloat16_rn(v[is * sv3 + it * sv2 + ih * sv1 + d]);
    if (d == 0) {
        const int64_t gate_idx = is * n_tokens * H_v + it * H_v + ih;
        const int64_t src_gate_idx = is * sb3 + it * sb2 + ih * sb1;
        alpha[gate_idx] = expf(g[src_gate_idx]);
        beta_packed[gate_idx] = beta[src_gate_idx];
    }

    // keep ggml's grouped-value head mapping (h_v % H_q) by materializing equal-head q/k.
    // FlashInfer's native GVA convention groups adjacent value heads instead.
    const int64_t iq3 = is / rq3;
    const int64_t iq1 = ih % H_q;
    const int64_t q_idx = ((is * n_tokens + it) * H_v + ih) * S + d;
    const int64_t src_idx = iq3 * sq3 + it * sq2 + iq1 * sq1 + d;
    q_bf16[q_idx] = __float2bfloat16_rn(q[src_idx]);
    k_bf16[q_idx] = __float2bfloat16_rn(k[src_idx]);
}

__global__ void gated_delta_net_flashinfer_unpack_cuda(
        const __nv_bfloat16 * src, float * dst, int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// 8 consecutive d per thread: two float4 loads and one 16-byte bf16x8 store per tensor, the same round-to-nearest as
// the scalar kernel; used when S is a multiple of 8 and every stride and base pointer keeps the loads 16-byte aligned
static __device__ __forceinline__ void gdn_pack8_bf16(const float * src, __nv_bfloat16 * dst) {
    const float4 a = *(const float4 *) src;
    const float4 b = *(const float4 *) (src + 4);
    union { __nv_bfloat162 h[4]; uint4 u; } r;
    r.h[0] = __float22bfloat162_rn(make_float2(a.x, a.y));
    r.h[1] = __float22bfloat162_rn(make_float2(a.z, a.w));
    r.h[2] = __float22bfloat162_rn(make_float2(b.x, b.y));
    r.h[3] = __float22bfloat162_rn(make_float2(b.z, b.w));
    *(uint4 *) dst = r.u;
}

__global__ void gated_delta_net_flashinfer_prepare8_cuda(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        __nv_bfloat16 * q_bf16,
        __nv_bfloat16 * k_bf16,
        __nv_bfloat16 * v_bf16,
        float * alpha,
        float * beta_packed,
        int64_t S,
        int64_t H_q,
        int64_t H_v,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t rq3) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t S8  = S / 8;
    const int64_t n8  = n_seqs * n_tokens * H_v * S8;
    if (idx >= n8) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int64_t d8 = idx % S8;
    const int64_t ih = (idx / S8) % H_v;
    const int64_t it = (idx / (S8 * H_v)) % n_tokens;
    const int64_t is = idx / (S8 * H_v * n_tokens);

    gdn_pack8_bf16(v + is * sv3 + it * sv2 + ih * sv1 + d8 * 8, v_bf16 + idx * 8);
    if (d8 == 0) {
        const int64_t gate_idx = is * n_tokens * H_v + it * H_v + ih;
        const int64_t src_gate_idx = is * sb3 + it * sb2 + ih * sb1;
        alpha[gate_idx] = expf(g[src_gate_idx]);
        beta_packed[gate_idx] = beta[src_gate_idx];
    }

    // keep ggml's grouped-value head mapping (h_v % H_q) by materializing equal-head q/k
    const int64_t iq3 = is / rq3;
    const int64_t iq1 = ih % H_q;
    const int64_t src_idx = iq3 * sq3 + it * sq2 + iq1 * sq1 + d8 * 8;
    gdn_pack8_bf16(q + src_idx, q_bf16 + idx * 8);
    gdn_pack8_bf16(k + src_idx, k_bf16 + idx * 8);
}

__global__ void gated_delta_net_flashinfer_unpack8_cuda(
        const __nv_bfloat16 * src, float * dst, int64_t n8) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n8) {
        return;
    }
    ggml_cuda_pdl_sync();
    union { uint4 u; __nv_bfloat162 h[4]; } r;
    r.u = *(const uint4 *) (src + idx * 8);
    const float2 f0 = __bfloat1622float2(r.h[0]);
    const float2 f1 = __bfloat1622float2(r.h[1]);
    const float2 f2 = __bfloat1622float2(r.h[2]);
    const float2 f3 = __bfloat1622float2(r.h[3]);
    *(float4 *) (dst + idx * 8)     = make_float4(f0.x, f0.y, f1.x, f1.y);
    *(float4 *) (dst + idx * 8 + 4) = make_float4(f2.x, f2.y, f3.x, f3.y);
}

__global__ void gated_delta_net_flashinfer_cu_seqlens_cuda(
        int64_t * cu_seqlens, int64_t n_tokens, int64_t n_seqs) {
    const int64_t idx = threadIdx.x;
    if (idx <= n_seqs) {
        cu_seqlens[idx] = idx * n_tokens;
    }
}

// alpha/beta only: the q/k/v packs were written by the fused conv kernel (ctx.gdn_pack)
__global__ void gated_delta_net_flashinfer_gates_cuda(
        const float * g, const float * beta, float * alpha, float * beta_packed,
        int64_t H_v, int64_t n_tokens, int64_t n_seqs, int64_t sb1, int64_t sb2, int64_t sb3) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_seqs * n_tokens * H_v) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int64_t ih = idx % H_v;
    const int64_t it = (idx / H_v) % n_tokens;
    const int64_t is = idx / (H_v * n_tokens);
    const int64_t src_gate_idx = is * sb3 + it * sb2 + ih * sb1;
    alpha[idx]       = expf(g[src_gate_idx]);
    beta_packed[idx] = beta[src_gate_idx];
}

static bool gdn_flashinfer_aot_serves(int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t H_q) {
    const int device = ggml_cuda_get_device();
    const auto & info = ggml_cuda_info().devices[device];
    return get_gdn_flashinfer_aot_launch() != nullptr && info.cc == GGML_CUDA_CC_HOPPER && S_v == 128 &&
        n_tokens >= 64 && H_q > 0 && H % H_q == 0 && n_seqs <= 1023 &&
        n_tokens * n_seqs <= INT32_MAX && H <= INT32_MAX && H_q <= INT32_MAX;
}

// pack_q/k/v: the bf16 inputs already written by the fused conv kernel (nullptr: packed here from the f32 sources)
static bool launch_gated_delta_net_flashinfer_aot(
        ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t H_q, int64_t rq3, cudaStream_t stream,
        __nv_bfloat16 * pack_q, __nv_bfloat16 * pack_k, __nv_bfloat16 * pack_v) {
    const gdn_flashinfer_aot_launch_t launch = get_gdn_flashinfer_aot_launch();
    const int device = ggml_cuda_get_device();
    const auto & info = ggml_cuda_info().devices[device];
    if (!gdn_flashinfer_aot_serves(S_v, H, n_tokens, n_seqs, H_q)) {
        return false;
    }

    const bool packed = pack_q != nullptr;
    const size_t v_elements = (size_t) n_seqs * n_tokens * H * S_v;
    const size_t q_elements = v_elements;
    const size_t gate_elements = (size_t) n_seqs * n_tokens * H;
    const size_t tensormaps_bytes = (size_t) info.nsm * 128;

    ggml_cuda_pool_alloc<__nv_bfloat16> q_bf16(ctx.pool());
    ggml_cuda_pool_alloc<__nv_bfloat16> k_bf16(ctx.pool());
    ggml_cuda_pool_alloc<__nv_bfloat16> v_bf16(ctx.pool());
    if (!packed) {
        pack_q = q_bf16.alloc(q_elements);
        pack_k = k_bf16.alloc(q_elements);
        pack_v = v_bf16.alloc(v_elements);
    }
    ggml_cuda_pool_alloc<__nv_bfloat16> out_bf16(ctx.pool(), v_elements);
    ggml_cuda_pool_alloc<float> alpha(ctx.pool(), gate_elements);
    ggml_cuda_pool_alloc<float> beta(ctx.pool(), gate_elements);
    ggml_cuda_pool_alloc<int64_t> cu_seqlens(ctx.pool(), n_seqs + 1);
    ggml_cuda_pool_alloc<uint8_t> tensormaps(ctx.pool(), tensormaps_bytes);

    // the 8-wide pack needs S a multiple of 8, strides that are multiples of 4 floats and 16-byte aligned bases; the
    // pool buffers are aligned by construction
    const bool vec8 = S_v % 8 == 0 && sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0 &&
        (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 && (uintptr_t) v_d % 16 == 0 && (uintptr_t) dst_d % 16 == 0;
    const size_t prepare_threads = vec8 ? v_elements / 8 : v_elements;

    const int threads = 256;
    if (packed) {
        ggml_cuda_kernel_launch_params gates_params(
            dim3((gate_elements + threads - 1) / threads, 1, 1), dim3(threads, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_flashinfer_gates_cuda, gates_params,
            g_d, b_d, alpha.get(), beta.get(), H, n_tokens, n_seqs, sb1, sb2, sb3);
    } else {
        ggml_cuda_kernel_launch_params prepare_params(
            dim3((prepare_threads + threads - 1) / threads, 1, 1), dim3(threads, 1, 1), 0, stream);
        if (vec8) {
            ggml_cuda_kernel_launch(gated_delta_net_flashinfer_prepare8_cuda, prepare_params,
                q_d, k_d, v_d, g_d, b_d,
                pack_q, pack_k, pack_v, alpha.get(), beta.get(),
                S_v, H_q, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, rq3);
        } else {
            ggml_cuda_kernel_launch(gated_delta_net_flashinfer_prepare_cuda, prepare_params,
                q_d, k_d, v_d, g_d, b_d,
                pack_q, pack_k, pack_v, alpha.get(), beta.get(),
                S_v, H_q, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, rq3);
        }
    }

    ggml_cuda_kernel_launch_params cu_params(
        dim3(1, 1, 1), dim3(n_seqs + 1, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_flashinfer_cu_seqlens_cuda, cu_params,
        cu_seqlens.get(), n_tokens, n_seqs);

    const int rc = launch(
        pack_q, pack_k, pack_v, out_bf16.get(),
        alpha.get(), beta.get(), state_d, s_d, tensormaps.get(), cu_seqlens.get(),
        (int32_t) (n_tokens * n_seqs), (int32_t) H, (int32_t) H, (int32_t) n_seqs,
        (int32_t) tensormaps_bytes, stream);
    if (rc != 0) {
        std::fprintf(stderr, "ggml_cuda: FlashInfer SM90 AOT GDN launch failed with status %d\n", rc);
        return false;
    }

    ggml_cuda_kernel_launch_params unpack_params(
        dim3((prepare_threads + threads - 1) / threads, 1, 1), dim3(threads, 1, 1), 0, stream);
    if (vec8) {
        ggml_cuda_kernel_launch(gated_delta_net_flashinfer_unpack8_cuda, unpack_params,
            out_bf16.get(), dst_d, (int64_t) (v_elements / 8));
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_flashinfer_unpack_cuda, unpack_params,
            out_bf16.get(), dst_d, (int64_t) v_elements);
    }
    return true;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // the fused conv kernel may have packed this node's bf16 inputs (one use per fill)
    ggml_cuda_gdn_pack & pack = ctx.gdn_pack;
    const bool packed = pack.node == dst && pack.elements == (size_t) n_seqs * n_tokens * H * S_v;
    pack.node = nullptr;
    if (!kda && !keep_rs &&
        launch_gated_delta_net_flashinfer_aot(
            ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1, rq3, stream,
            packed ? (__nv_bfloat16 *) pack.q() : nullptr,
            packed ? (__nv_bfloat16 *) pack.k() : nullptr,
            packed ? (__nv_bfloat16 *) pack.v() : nullptr)) {
        return;
    }
    // the fallback kernels read the f32 q/k/v, which the conv kernel skipped when the pack was their only reader
    GGML_ASSERT(!(packed && pack.f32_elided) && "FlashInfer GDN launch failed with conv-packed inputs");
#endif

    // prefill: run whole chunks through the chunked kernel, then the serial kernel finishes the tail from that state.
    // K > 1 keeps at least K tokens in the tail so every snapshot slot is written by the serial kernel.
    const int64_t tail_min   = keep_rs ? K : 0;
    const int64_t n_chunked  = n_tokens > tail_min ? ((n_tokens - tail_min) / GDN_CHUNK) * GDN_CHUNK : 0;
    const int64_t n_tail     = n_tokens - n_chunked;
    const bool    use_chunks = n_chunked >= GDN_CHUNK && gdn_chunked_supported(S_v, kda);

    ggml_cuda_pool_alloc<float> state_mid(ctx.pool());
    const float * s_tail = s_d;
    if (use_chunks) {
        float * chunk_state = state_d;
        if (n_tail > 0) {
            state_mid.alloc(S_v * S_v * H * n_seqs);
            chunk_state = state_mid.get();
            s_tail      = chunk_state;
        }
        if (S_v == 64) {
            launch_gated_delta_net_chunked<64>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, chunk_state,
                H, n_chunked, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        } else {
            launch_gated_delta_net_chunked<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, chunk_state,
                H, n_chunked, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        }
        if (n_tail == 0) {
            return;
        }
        q_d   += n_chunked * sq2;
        k_d   += n_chunked * sq2;
        v_d   += n_chunked * sv2;
        g_d   += n_chunked * sb2;
        b_d   += n_chunked * sb2;
        dst_d += n_chunked * S_v * H;
    }
    const int64_t n_serial = use_chunks ? n_tail : n_tokens;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_tail, dst_d, state_d,
                S_v, H, n_serial, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_tail, dst_d, state_d,
                S_v, H, n_serial, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_tail, dst_d, state_d,
                S_v, H, n_serial, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_tail, dst_d, state_d,
                S_v, H, n_serial, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

bool ggml_cuda_gated_delta_net_flashinfer_aot(const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const ggml_tensor * src_q = dst->src[0];
    const ggml_tensor * src_v = dst->src[2];
    const ggml_tensor * src_g = dst->src[3];
    const bool kda = src_g->ne[0] == src_v->ne[0];
    return !kda && ggml_get_op_params_i32(dst, 0) <= 1 &&
        gdn_flashinfer_aot_serves(src_v->ne[0], src_v->ne[1], src_v->ne[2], src_v->ne[3], src_q->ne[1]);
#else
    GGML_UNUSED(dst);
    return false;
#endif
}
