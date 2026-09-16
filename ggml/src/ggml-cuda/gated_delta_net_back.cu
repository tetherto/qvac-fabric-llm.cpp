#include "gated_delta_net_back.cuh"
#include "ggml-cuda/common.cuh"

#include <type_traits>

#define CUDA_GDN_BACK_MAX_BLOCK_SIZE 1024u

struct ggml_cuda_gated_delta_net_back_kargs {
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    uint32_t off_dk, off_dv, off_dg, off_db, off_ds;
    uint32_t off_scratch;
    uint32_t wg_stride;
    float   scale;
};
static_assert(std::is_trivially_copyable_v<ggml_cuda_gated_delta_net_back_kargs>);

static __host__ __device__ constexpr inline
uint32_t gdn_min(const uint32_t a, const uint32_t b) {
    return a < b ? a : b;
}

static __host__ __device__ constexpr inline
uint32_t gdn_bit_floor(uint32_t x) {
    if (x == 0u)
        return 0u;
    uint32_t r = 1u;
    while (r <= (x >> 1u))
        r <<= 1u;
    return r;
}

static __host__ __device__ constexpr inline
uint32_t compute_lanes_per_col(const uint32_t S_v, const uint32_t warp_size) {
    uint32_t lanes_per_column =
        (S_v >= 128) ? 8u : gdn_min(S_v, warp_size);

    // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
    // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
    while (lanes_per_column > 1u) {
        const bool valid_lanes = (warp_size % lanes_per_column) == 0 &&
                                    (S_v % lanes_per_column) == 0;
        const uint32_t cols_per_wg = valid_lanes ? warp_size / lanes_per_column : 0;
        if (valid_lanes && cols_per_wg > 0 && (S_v % cols_per_wg) == 0) {
            break;
        }
        lanes_per_column >>= 1u;
    }

    return lanes_per_column;
}

static __host__ __device__ constexpr inline uint32_t
gdn_back_threads_per_block(const uint32_t S_v, const uint32_t warp_size) {
    const uint32_t lanes_per_col =
        compute_lanes_per_col(S_v, warp_size);

    const uint32_t max_warps = gdn_min(
        gdn_min(32u, S_v * lanes_per_col / warp_size),
        CUDA_GDN_BACK_MAX_BLOCK_SIZE / warp_size);

    return max_warps != 0u
        ? gdn_bit_floor(max_warps) * warp_size
        : gdn_min(S_v, CUDA_GDN_BACK_MAX_BLOCK_SIZE);
}

template < const uint32_t lanes_per_col, typename T >
static __device__ inline T
warp_reduce_partial(T partial) {
    static_assert(lanes_per_col <= (uint32_t) ggml_cuda_get_physical_warp_size(),
                  "a column's lanes must fit within one warp");

    static_assert((lanes_per_col & (lanes_per_col - 1u)) == 0u,
                  "lanes_per_col must be a power of two");

    static_assert(!std::is_integral_v<T>,
                  "warp_reduce_sum ignores width for integral types on Ampere and newer");
    if constexpr (lanes_per_col == 1) {
        return partial;
    } else {
        return warp_reduce_sum<lanes_per_col>(partial);
    }
}

template < const uint32_t rows_active, const uint32_t warp_size >
static __device__ inline float
reduce_token_block(float v,
                   const uint32_t tid,
                   const uint32_t lane_id,
                   const uint32_t warp_id) {

    if constexpr (rows_active <= warp_size) {
        return warp_reduce_sum<rows_active>(v);
    } else {
        extern __shared__ float shared_warp_sums[];

        const float warp_sum = warp_reduce_sum<warp_size>(v);

        // the run spans rows_active / warp_size warps: combine via shared memory, relying on
        // the linear tid -> warp mapping. The leading barrier orders re-use of
        // shared_warp_sums against the previous call's readers.
        __syncthreads();
        if (lane_id == 0u) {
            shared_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        constexpr uint32_t warps_per_run = rows_active / warp_size;
        const uint32_t first = (tid / rows_active) * warps_per_run;
        float total = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < warps_per_run; ++k) {
            total += shared_warp_sums[first + k];
        }
        return total;
    }
}

template <
    const bool KDA,
    const uint32_t S_v,
    const uint32_t block_size = gdn_back_threads_per_block(S_v, ggml_cuda_get_physical_warp_size())
>
__global__ void __launch_bounds__(block_size)
gated_delta_net_back_cuda(const float * data_q_ptr,
                          const float * data_k_ptr,
                          const float * data_v_ptr,
                          const float * data_g_ptr,
                          const float * data_beta_ptr,
                          const float * data_state_ptr,
                          const float * data_d_ptr,
                                float * data_dst_ptr,
                          const ggml_cuda_gated_delta_net_back_kargs args) {

    constexpr const uint32_t warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr const uint32_t lanes_per_col = compute_lanes_per_col(S_v, warp_size);
    constexpr const uint32_t rows_per_lane = S_v / lanes_per_col;
    constexpr const uint32_t cols_per_step = block_size / lanes_per_col; // columns advanced per wave, across all warps

    // Row-pass decomposition: lanes own contiguous rows; spare threads process extra tokens.
    // At most one of t_tile / row_waves exceeds 1 (both are 1 when block_size == S_v).
    constexpr const uint32_t rows_active = (block_size < S_v) ? block_size : S_v;  // rows in flight per token
    constexpr const uint32_t t_tile      = block_size / rows_active;  // tokens in flight
    constexpr const uint32_t row_waves   = S_v / rows_active;         // row passes per token

    static_assert(block_size % warp_size == 0, "block must be a whole number of warps");
    static_assert(cols_per_step <= S_v && S_v % cols_per_step == 0, "A1/A2 wave loop has no bounds check");
    static_assert(S_v % rows_active == 0, "row pass has no bounds check");

    const uint32_t tid = threadIdx.x;
    const uint32_t iq1 = blockIdx.x; // q/k head
    const uint32_t iq3 = blockIdx.y; // q/k seq

    const uint32_t lane_id = threadIdx.x & (warp_size-1);
    const uint32_t warp_id = threadIdx.x / warp_size;

    // row-pass (phase B) thread mapping
    const uint32_t i_lane = tid % rows_active;
    const uint32_t t_sub  = tid / rows_active;

    const uint32_t H        = args.H;
    const uint32_t n_tokens = args.n_tokens;
    const uint32_t K        = args.K;
    const uint32_t neq1     = args.neq1;
    const uint32_t rq3      = args.rq3;
    const uint32_t group    = H / neq1;
    const float    scale    = args.scale;

    constexpr const uint32_t state_size = S_v * S_v;
    const uint32_t wg_id   = iq1 + neq1 * iq3;
    const uint64_t sc_base = args.off_scratch + (uint64_t) wg_id * args.wg_stride;
    const uint64_t sc_S    = sc_base;
    const uint64_t sc_A    = sc_S + (uint64_t) n_tokens * state_size;
    const uint64_t sc_u    = sc_A + (uint64_t) n_tokens * state_size;
    const uint64_t sc_sd   = sc_u + (uint64_t) n_tokens * S_v;

    const uint32_t state_size_per_snap = state_size * H * args.n_seqs;

    ggml_cuda_pdl_sync();
    const float * GGML_CUDA_RESTRICT data_q     = data_q_ptr;
    const float * GGML_CUDA_RESTRICT data_k     = data_k_ptr;
    const float * GGML_CUDA_RESTRICT data_v     = data_v_ptr;
    const float * GGML_CUDA_RESTRICT data_g     = data_g_ptr;
    const float * GGML_CUDA_RESTRICT data_beta  = data_beta_ptr;
    const float * GGML_CUDA_RESTRICT data_state = data_state_ptr;
    const float * GGML_CUDA_RESTRICT data_d     = data_d_ptr;
          float * GGML_CUDA_RESTRICT data_dst   = data_dst_ptr;

    for (uint32_t gi = 0; gi < group; gi++) {
        const uint32_t iv1 = iq1 + gi * neq1;       // v-head (iv1 % neq1 == iq1)
        for (uint32_t sgi = 0; sgi < rq3; sgi++) {
            const uint32_t iv3 = iq3 * rq3 + sgi;   // v-seq
            // state (the forward op's initial state) has layout [S_v, S_v, H, n_seqs] with no K factor.
            const uint32_t state_in_base  = (iv3 * H + iv1) * state_size;
            const uint32_t state_out_base = (iv3 * H + iv1) * state_size;

            // ---------- phase A1: forward replay, store S_hist / u ----------
            for (uint32_t wave = 0; wave < S_v / cols_per_step; ++wave) {
                const uint32_t which_col = tid / lanes_per_col;
                const uint32_t lane      = tid % lanes_per_col;
                const uint32_t j         = wave * cols_per_step + which_col;

                float s_shard[rows_per_lane];
#pragma unroll
                for(uint32_t r = 0; r < rows_per_lane; ++r) {
                    s_shard[r] = data_state[state_in_base + j * S_v + r * lanes_per_col + lane];
                }

                for (uint32_t t = 0; t < n_tokens; ++t) {
                    const uint32_t k_off  = iq3 * args.sq3 + t * args.sq2 + iq1 * args.sq1;
                    const uint32_t v_off  = iv3 * args.sv3 + t * args.sv2 + iv1 * args.sv1;
                    const uint32_t gb_off = iv3 * args.sb3 + t * args.sb2 + iv1 * args.sb1;
                    const float beta_val = data_beta[gb_off];

                    float k_reg[rows_per_lane];
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        const uint32_t i = r * lanes_per_col + lane;
                        k_reg[r] = data_k[k_off + i];
                    }

                    float eq_reg[rows_per_lane];
                    if constexpr (KDA) {
                        const uint32_t g_base = gb_off * S_v;
#pragma unroll
                        for(uint32_t r = 0; r < rows_per_lane; ++r) {
                            const uint32_t i = r * lanes_per_col + lane;
                            eq_reg[r] = exp(data_g[g_base + i]);
                        }
                    } else {
                        const float g_val = exp(data_g[gb_off]);
#pragma unroll
                        for(uint32_t r = 0; r < rows_per_lane; ++r) {
                            eq_reg[r] = g_val;
                        }
                    }

                    const float v_val = data_v[v_off + j];

                    float kv_shard = 0.0;
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        kv_shard = fma(eq_reg[r], s_shard[r] * k_reg[r], kv_shard);
                    }
                    const float kv_j = warp_reduce_partial<lanes_per_col>(kv_shard);

                    const float u_j     = v_val - kv_j;
                    const float delta_j = u_j * beta_val;

                    if (lane == 0u) {
                        data_dst[sc_u + (uint64_t ) t * S_v + j] = u_j;
                    }

#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        s_shard[r] = fma(eq_reg[r], s_shard[r], k_reg[r] * delta_j);
                        const uint32_t i = r * lanes_per_col + lane;
                        data_dst[sc_S + (uint64_t ) t * state_size + j * S_v + i] = s_shard[r];
                    }
                }
            }

            // No barrier between A1 and A2: A2 reads no scratch (u_hist and S_hist are
            // consumed by the row pass, after scratch_barrier).

            // ---------- phase A2: reverse scan, store A_hist / sd / d_v / d_state ----------
            for (uint32_t wave = 0; wave < S_v / cols_per_step; ++wave) {
                const uint32_t which_col = tid / lanes_per_col;
                const uint32_t lane      = tid % lanes_per_col;
                const uint32_t j         = wave * cols_per_step + which_col;

                float carry_shard[rows_per_lane];
#pragma unroll
                for(uint32_t r = 0; r < rows_per_lane; ++r) {
                    carry_shard[r] = 0.0;
                }

                for (int t = int(n_tokens) - 1; t >= 0; t--) {
                    const uint32_t ut = uint(t);
                    const uint32_t q_off  = iq3 * args.sq3 + ut * args.sq2 + iq1 * args.sq1;
                    const uint32_t k_off  = q_off;
                    const uint32_t gb_off = iv3 * args.sb3 + ut * args.sb2 + iv1 * args.sb1;
                    const float beta_val = data_beta[gb_off];

                    float k_reg[rows_per_lane];
                    float q_reg[rows_per_lane];
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        const uint32_t i = r * lanes_per_col + lane;
                        k_reg[r] = data_k[k_off + i];
                        q_reg[r] = data_q[q_off + i];
                    }

                    const float g_val = KDA ? 0.0f : exp(data_g[gb_off]);

                    const uint32_t do_off = (iv3 * n_tokens * H + iv1) * S_v + ut * S_v * H;
                    const float do_j = data_d[do_off + j];

                    // A += scale * q (x) do ; plus state-output gradient seed (covers K=1 final
                    // state and K>1 snapshots via target_slot; matches the CPU kernel).
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        carry_shard[r] += scale * q_reg[r] * do_j;
                    }

                    {
                        // matches the forward op's slot mapping: slot 0 = most recent state (t = n_tokens-1).
                        const int target_slot = int(n_tokens) - 1 - int(t);
                        if (target_slot >= 0 && target_slot < int(K)) {
                            const uint32_t dss = args.s_off + uint32_t(target_slot) * state_size_per_snap + state_out_base;
#pragma unroll
                            for(uint32_t r = 0; r < rows_per_lane; ++r) {
                                const uint32_t i = r * lanes_per_col + lane;
                                carry_shard[r] += data_d[dss + j * S_v + i];
                            }
                        }
                    }

                    // store A_hist[t] (adjoint used by the row pass)
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        const uint32_t i = r * lanes_per_col + lane;
                        data_dst[sc_A + (uint64_t ) ut * state_size + j * S_v + i] = carry_shard[r];
                    }

                    // sd = (A^T k)_j ; w = beta * sd  (u/delta come from A1's replay)
                    float sd_shard = 0.0;
#pragma unroll
                    for(uint32_t r = 0; r < rows_per_lane; ++r) {
                        sd_shard = fma(carry_shard[r], k_reg[r], sd_shard);
                    }
                    const float sd  = warp_reduce_partial<lanes_per_col>(sd_shard);
                    const float w_j = beta_val * sd;

                    if (lane == 0u) {
                        data_dst[sc_sd + (uint64_t ) ut * S_v + j] = sd;
                        data_dst[args.off_dv + (iv1 + H * (ut + n_tokens * iv3)) * S_v + j] = w_j;
                    }

                    // propagate A_{t-1} = diag(exp(g)) (A - k w^T)
                    if constexpr (KDA) {
                        const uint32_t g_base = gb_off * S_v;
#pragma unroll
                        for (uint32_t r = 0; r < rows_per_lane; ++r) {
                            const uint32_t i = r * lanes_per_col + lane;
                            carry_shard[r] = fma(k_reg[r], -w_j, carry_shard[r]) * exp(data_g[g_base + i]);
                        }
                    } else {
#pragma unroll
                        for (uint32_t r = 0; r < rows_per_lane; ++r) {
                            carry_shard[r] = fma(k_reg[r], -w_j, carry_shard[r]) * g_val;
                        }
                    }
                }

                // initial-state gradient: layout [S_v, S_v, H, n_seqs] (no K factor, unlike the forward output).
#pragma unroll
                for(uint32_t r = 0; r < rows_per_lane; ++r) {
                    const uint32_t i = r * lanes_per_col + lane;
                    data_dst[args.off_ds + (iv3 * H + iv1) * state_size + j * S_v + i] = carry_shard[r];
                }
            }

            // A1/A2 scratch (S_hist / A_hist / u / sd) -> phase B handoff: phase B reads
            // these across all columns, i.e. across threads. This barrier is required.
            __syncthreads();

            // ---------- phase B: row pass for d_q / d_k / d_g / d_beta ----------
            // t_tile tokens per iteration; tail threads (t >= n_tokens) run on clamped
            // addresses with discarded results so control flow stays uniform for the
            // barriers in reduce_token_block.
            for (uint32_t tb = 0; tb < n_tokens; tb += t_tile) {
                const uint32_t t  = tb + t_sub;
                const bool t_ok   = (t < n_tokens);
                const uint32_t tc = t_ok ? t : (n_tokens - 1u);   // clamped for in-bounds addressing

                const uint32_t k_off  = iq3 * args.sq3 + tc * args.sq2 + iq1 * args.sq1;
                const uint32_t gb_off = iv3 * args.sb3 + tc * args.sb2 + iv1 * args.sb1;
                const uint32_t do_off = (iv3 * n_tokens * H + iv1) * S_v + tc * S_v * H;
                const uint32_t tv     = iv1 + H * (t + n_tokens * iv3);  // (v-head, token) output index
                const float beta_val  = data_beta[gb_off];

                // this token's scratch
                const uint64_t  st_base = sc_S  + (uint64_t ) tc * state_size;
                const uint64_t  a_base  = sc_A  + (uint64_t ) tc * state_size;
                const uint64_t  u_base  = sc_u  + (uint64_t ) tc * S_v;
                const uint64_t  sd_base = sc_sd + (uint64_t ) tc * S_v;

                float dbeta  = 0.0;
                float dg_tot = 0.0;

                for (uint32_t rw = 0; rw < row_waves; ++rw) {
                    const uint32_t i = rw * rows_active + i_lane;
                    const float k_i = data_k[k_off + i];

                    float dq = 0.0, dk = 0.0, dg = 0.0;
                    dbeta = 0.0;   // every row wave recomputes the same per-token value
                    for (uint32_t jj = 0; jj < S_v; ++jj) {
                        const float st_ij = data_dst[st_base + jj * S_v + i];
                        const float a_ij  = data_dst[a_base  + jj * S_v + i];
                        const float u_j   = data_dst[u_base  + jj];
                        const float sd_j  = data_dst[sd_base + jj];
                        const float do_j  = data_d[do_off + jj];

                        const float delta_j = beta_val * u_j;
                        const float w_j     = beta_val * sd_j;
                        // exp(g)*S_prev reconstructed from the forward update
                        const float sp_ij   = fma(-k_i, delta_j, st_ij);

                        dq    = fma(st_ij, do_j, dq);
                        dk   += fma(a_ij, delta_j, -sp_ij * w_j);
                        dg    = fma(fma(-k_i, w_j, a_ij), sp_ij, dg);
                        dbeta = fma(sd_j, u_j, dbeta);
                    }

                    // d_q/d_k accumulate over the v-head group (gi/sgi); the first iteration
                    // initialises instead of zeroing up front. Row (t, i)'s writer is the same
                    // thread every iteration, so no cross-thread ordering is needed.
                    if (t_ok) {
                        const uint32_t row = (iq1 + neq1 * (t + n_tokens * iq3)) * S_v + i;
                        if (gi == 0u && sgi == 0u) {
                            data_dst[row]          = scale * dq;
                            data_dst[args.off_dk + row] = dk;
                        } else {
                            data_dst[row]          = fma(scale, dq, data_dst[row]);
                            data_dst[args.off_dk + row] += dk;
                        }
                        if constexpr (KDA) {
                            data_dst[args.off_dg + tv * S_v + i] = dg;
                        }
                    }
                    if constexpr (!KDA) {
                        // scalar d_g: sum over this token's rows
                        dg_tot += reduce_token_block<rows_active, warp_size>(t_ok ? dg : 0.0f,
                                                                            tid,
                                                                            lane_id,
                                                                            warp_id);
                    }
                }

                // dbeta is identical across a token's rows (broadcast operands), so any
                // row wave's value is the full per-token sum.
                if (t_ok && i_lane == 0u) {
                    data_dst[args.off_db + tv] = dbeta;
                    if constexpr (!KDA) {
                        data_dst[args.off_dg + tv] = dg_tot;
                    }
                }
            }

            // scratch is reused by the next group member; the last iteration needs no barrier
            if (gi + 1u < group || sgi + 1u < rq3) {
                __syncthreads();
            }
        }
    }
}

template <const bool KDA, const uint32_t S_v>
static __host__ inline void
launch_gated_delta_net_back_impl(const float * data_q,
                                 const float * data_k,
                                 const float * data_v,
                                 const float * data_g,
                                 const float * data_beta,
                                 const float * data_state,
                                 const float * data_d,
                                       float * data_dst,
                                 const uint32_t neq3,
                                 const struct ggml_cuda_gated_delta_net_back_kargs& kargs,
                                 cudaStream_t stream) {

    const uint32_t warp_size  = (uint32_t)ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const uint32_t block_size = gdn_back_threads_per_block(S_v, warp_size);

    // reduce_token_block's cross-warp stage keeps one float per warp.
    const size_t smem =  KDA ? 0u : (((block_size + warp_size - 1u) / warp_size) * sizeof(float));

    // One block per (q/k head, q/k seq): the block walks its GQA group internally over
    // gi/sgi, and wg_id = iq1 + neq1*iq3 indexes its private scratch region.
    const dim3 grid_dims (kargs.neq1, neq3, 1);
    const dim3 block_dims(block_size, 1, 1);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, smem, stream);

    ggml_cuda_kernel_launch(gated_delta_net_back_cuda<KDA, S_v>, launch_params,
        data_q, data_k, data_v, data_g,
        data_beta, data_state, data_d,
        data_dst, kargs);
}

template <const bool KDA>
static __host__ inline void
launch_gated_delta_net_back(const float * data_q,
                            const float * data_k,
                            const float * data_v,
                            const float * data_g,
                            const float * data_beta,
                            const float * data_state,
                            const float * data_d,
                                  float * data_dst,
                            const uint32_t S_v,
                            const uint32_t neq3,
                            const struct ggml_cuda_gated_delta_net_back_kargs& kargs,
                            cudaStream_t stream) {

#define GDN_BACK_S_V_SIZES(X) \
    X(16)                     \
    X(32)                     \
    X(64)                     \
    X(128)

#define GDN_BACK_LAUNCH_CASE(X_ARG_S_V)                                                  \
    case (X_ARG_S_V):                                                                    \
        launch_gated_delta_net_back_impl<KDA, (X_ARG_S_V)>(                              \
            data_q, data_k, data_v, data_g, data_beta, data_state, data_d, data_dst,     \
            neq3, kargs, stream);                                                        \
        break;

    switch (S_v) {
    GDN_BACK_S_V_SIZES(GDN_BACK_LAUNCH_CASE)
    default:
        GGML_ABORT("unsupported S_v size");
        break;
    }

#undef GDN_BACK_LAUNCH_CASE
#undef GDN_BACK_S_V_SIZES
}

__host__ void
ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_d     = dst->src[6];

    GGML_TENSOR_LOCALS(uint32_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(uint32_t, nbq, src_q, nb);
    GGML_TENSOR_LOCALS(uint32_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(uint32_t, nbk, src_k, nb);
    GGML_TENSOR_LOCALS(uint32_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(uint32_t, nbv, src_v, nb);
    GGML_TENSOR_LOCALS(uint32_t, nbb, src_beta, nb);

    const uint32_t S_v  = nev0;
    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(neq0 == S_v && nek0 == S_v);
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(ggml_is_contiguous(src_d));
    GGML_ASSERT(kda ? (ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta))
                    : (src_g->ne[0] == 1 && ggml_are_same_stride(src_g, src_beta)));
    GGML_ASSERT(neq1 == nek1);

    const float * q_d   = (const float *)src_q->data;
    const float * k_d   = (const float *)src_k->data;
    const float * v_d   = (const float *)src_v->data;
    const float * g_d   = (const float *)src_g->data;
    const float * b_d   = (const float *)src_beta->data;
    const float * s_d   = (const float *)src_state->data;
    const float * d_d   = (const float *)src_d->data;
    float *       dst_d = (float *) dst->data;

    constexpr const auto fsz = static_cast<uint32_t>(sizeof(float));

#define GGML_PAD_ALIGNED_NELEM(src) \
    static_cast<uint32_t>(GGML_PAD(ggml_nelements(src) * fsz, GGML_MEM_ALIGN) / fsz)

    const uint32_t pad_q = GGML_PAD_ALIGNED_NELEM(src_q);
    const uint32_t pad_k = GGML_PAD_ALIGNED_NELEM(src_k);
    const uint32_t pad_v = GGML_PAD_ALIGNED_NELEM(src_v);
    const uint32_t pad_g = GGML_PAD_ALIGNED_NELEM(src_g);
    const uint32_t pad_b = GGML_PAD_ALIGNED_NELEM(src_beta);
    const uint32_t pad_s = GGML_PAD_ALIGNED_NELEM(src_state);
#undef GGML_PAD_ALIGNED_NELEM

    const uint32_t H           = nev1;
    const uint32_t n_tokens    = nev2;
    const uint32_t n_seqs      = nev3;

    const uint32_t off_dk      = pad_q;
    const uint32_t off_dv      = off_dk + pad_k;
    const uint32_t off_dg      = off_dv + pad_v;
    const uint32_t off_db      = off_dg + pad_g;
    const uint32_t off_ds      = off_db + pad_b;
    const uint32_t off_scratch = off_ds + pad_s;

    cudaStream_t stream = ctx.stream();

    const ggml_cuda_gated_delta_net_back_kargs kargs = {
        .H           = H,
        .n_tokens    = n_tokens,
        .n_seqs      = n_seqs,
        .K           = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 0)),
        .s_off       = (S_v * H * n_tokens * n_seqs),
        .sq1         = nbq1 / fsz,
        .sq2         = nbq2 / fsz,
        .sq3         = nbq3 / fsz,
        .sv1         = nbv1 / fsz,
        .sv2         = nbv2 / fsz,
        .sv3         = nbv3 / fsz,
        .sb1         = nbb1 / fsz,
        .sb2         = nbb2 / fsz,
        .sb3         = nbb3 / fsz,
        .neq1        = neq1,
        .rq3         = nev3 / neq3,
        .off_dk      = off_dk,
        .off_dv      = off_dv,
        .off_dg      = off_dg,
        .off_db      = off_db,
        .off_ds      = off_ds,
        .off_scratch = off_scratch,
        .wg_stride   = n_tokens * (2*S_v*S_v + 2*S_v),
        .scale       = 1.0f / sqrtf((float) S_v),
    };
    if (kda) {
        launch_gated_delta_net_back<true>(q_d, k_d, v_d, g_d, b_d, s_d, d_d, dst_d,
                                          S_v, neq3, kargs, stream);
    } else {
        launch_gated_delta_net_back<false>(q_d, k_d, v_d, g_d, b_d, s_d, d_d, dst_d,
                                           S_v, neq3, kargs, stream);
    }
}
