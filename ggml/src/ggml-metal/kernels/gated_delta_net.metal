#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];
constant short FC_gated_delta_net_nt   [[function_constant(FC_GATED_DELTA_NET + 3)]];

static inline float gdn_dot(float  a, float  b) { return a * b; }
static inline float gdn_dot(float2 a, float2 b) { return dot(a, b); }
static inline float gdn_dot(float4 a, float4 b) { return dot(a, b); }

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * cache,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    static_assert(sizeof(T) == NSG * sizeof(float), "");

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x * ntg.y + ty;

    if (i20 >= (uint) S_v) {
        return;
    }

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const T * s_ptr = (device const T *) ((device const float *) (s) + state_in_base);

    T ls = s_ptr[tx];

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    device float * state_base = args.fuse_cache ?
        (device float *) cache :
        (device float *) dst + attn_size;

    const short ntok = FC_gated_delta_net_nt > 0 ? FC_gated_delta_net_nt : args.ne22;

    for (short t = 0; t < ntok; t++) {
        const T kt = ((device const T *) k_ptr)[tx];
        const T qt = ((device const T *) q_ptr)[tx];

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);
            const float s_k = simd_sum(gdn_dot(ls, kt));
            const float d = (v_ptr[i20] - g_exp * s_k) * b_ptr[0];
            ls = ls * g_exp + kt * d;
        } else {
            const T ge = exp(((device const T *) g_ptr)[tx]);
            const float s_k = simd_sum(gdn_dot(ls * ge, kt));
            const float d = (v_ptr[i20] - s_k) * b_ptr[0];
            ls = ls * ge + kt * d;
        }

        const float y = simd_sum(gdn_dot(ls, qt));

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const uint64_t ns_slot = args.fuse_cache ? args.ns_cache : state_size_per_snap;
            const int target_slot = (int) ntok - 1 - (int) t;
            if (target_slot >= 0 && target_slot < (int) K) {
                device T * dst_state = (device T *) (state_base + (uint64_t) target_slot * ns_slot + state_out_base);
                dst_state[tx] = ls;
            }
        }
    }

    if (K == 1) {
        device T * dst_state = (device T *) (state_base + state_out_base);
        dst_state[tx] = ls;
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;

// Backward of gated_delta_net.
constant short FC_gdn_back_S_v [[function_constant(FC_GATED_DELTA_NET + 10)]];
constant short FC_gdn_back_kda [[function_constant(FC_GATED_DELTA_NET + 11)]];
constant short FC_gdn_lanes_per_col [[function_constant(FC_GATED_DELTA_NET + 12)]];

template < const uint cluster_size, typename T  >
inline T simd_clustered_sum(T v) {
    static_assert((cluster_size & (cluster_size - 1)) == 0);
    FOR_UNROLL (uint offset = 1; offset < cluster_size; offset <<= 1) {
        v += simd_shuffle_xor(v, offset);
    }
    return v;
}

template < typename T >
inline T simd_clustered_sum(T v, const uint cluster) {
    switch (cluster) {
        case 2u:
            return simd_clustered_sum<2u>(v);
        case 4u:
            return simd_clustered_sum<4u>(v);
        case 8u:
            return simd_clustered_sum<8u>(v);
        case 16u:
            return simd_clustered_sum<16u>(v);
        case 32u:
            return simd_clustered_sum<32u>(v);
    }
    return v;
}

template < typename T >
inline T gdn_reduce_partial(T partial) {
    if (FC_gdn_lanes_per_col == 1)
        return partial;
    if (FC_gdn_lanes_per_col <= 32)
        return simd_clustered_sum(partial, FC_gdn_lanes_per_col);
    return simd_sum(partial);
}

template < typename T >
inline T gdn_reduce_token_block(T v,
                         const uint tid,
                         const uint rows_active,
                         const uint simdgroup_width,
                         const uint tiisg,
                         const uint sgitg,
                         threadgroup T * sh_sg) {

    if (rows_active < simdgroup_width) {
        return simd_clustered_sum(v, rows_active);
    }

    const T s = simd_sum(v);
    if (rows_active == simdgroup_width) {
        return s;
    }

    // block spans rows_active / simdgroup_width subgroups: combine via shared memory,
    // assuming the linear tid -> subgroup mapping guaranteed by the pipeline's required
    // subgroup size. The leading barrier orders re-use of sh_sg against the previous
    // call's readers.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) {
        sh_sg[sgitg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg_per_block = rows_active / simdgroup_width;
    const uint first = (tid / rows_active) * sg_per_block;
    T total = 0;
    FOR_UNROLL (uint k = 0u; k < sg_per_block; ++k) {
        total += sh_sg[first + k];
    }
    return total;
}

template < const uint rows_per_lane >
inline void gated_delta_net_back_impl(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const float * data_q,
        device const float * data_k,
        device const float * data_v,
        device const float * data_g,
        device const float * data_beta,
        device const float * data_state,
        device const float * data_d,
        device       float * data_dst,
        threadgroup  float * sh_sg,
        uint3 tgpig,
        uint3 tpitg,
        uint3 threads_per_tg,
        uint  tid,
        uint  tiisg,
        uint  simdgroup_width,
        uint  simdgroups_per_tg,
        uint  sgitg) {

    const uint tg_size = threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;

    const uint S_v = (uint) FC_gdn_back_S_v;
    const bool kda = FC_gdn_back_kda != 0;

    const uint lanes_per_col = FC_gdn_lanes_per_col;
    const uint cols_per_wg = simdgroup_width / lanes_per_col; // columns per simdgroup, across all subgroups
    const uint cols_per_step  = tg_size / lanes_per_col;      // columns advanced per wave, across all subgroups

    // Row-pass decomposition: lanes own contiguous rows; spare threads process extra tokens.
    // At most one of T_TILE / row_waves exceeds 1 (both are 1 when WG_SIZE == S_v).
    const uint rows_active = (tg_size < S_v) ? tg_size : S_v;  // rows in flight per token
    const uint t_tile      = tg_size / rows_active;            // tokens in flight
    const uint row_waves   = S_v / rows_active;                // row passes per token

    const uint iq1 = tgpig.x; // q/k head
    const uint iq3 = tgpig.y; // q/k seq

    // row-pass (phase B) thread mapping
    const uint i_lane = tid % rows_active;
    const uint t_sub  = tid / rows_active;

    const uint H        = args.H;
    const uint n_tokens = args.n_tokens;
    const uint K        = args.K;
    const uint neq1     = args.neq1;
    const uint rq3      = args.rq3;
    const uint group    = H / neq1;
    const float scale   = args.scale;

    const uint  state_size = S_v * S_v;
    const uint  wg_id      = iq1 + neq1 * iq3;
    const ulong sc_base    = args.off_scratch + (ulong) wg_id * args.wg_stride;
    const ulong sc_S       = sc_base;
    const ulong sc_A       = sc_S + (ulong) n_tokens * state_size;
    const ulong sc_u       = sc_A + (ulong) n_tokens * state_size;
    const ulong sc_sd      = sc_u + (ulong) n_tokens * S_v;

    const uint state_size_per_snap = state_size * H * args.n_seqs;

    for (uint gi = 0; gi < group; gi++) {
        const uint iv1 = iq1 + gi * neq1;       // v-head (iv1 % neq1 == iq1)
        for (uint sgi = 0; sgi < rq3; sgi++) {
            const uint iv3 = iq3 * rq3 + sgi;   // v-seq
            // state (the forward op's initial state) has layout [S_v, S_v, H, n_seqs] with no K factor.
            const uint state_in_base  = (iv3 * H + iv1) * state_size;
            const uint state_out_base = (iv3 * H + iv1) * state_size;

            // ---------- phase A1: forward replay, store S_hist / u ----------
            for (uint wave = 0; wave < S_v / cols_per_step; ++wave) {
                const uint which_col = tpitg.x / lanes_per_col;
                const uint lane      = tpitg.x % lanes_per_col;
                const uint j         = wave * cols_per_step + which_col;

                float s_shard[rows_per_lane];
                FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                    s_shard[r] = data_state[state_in_base + j * S_v + r * lanes_per_col + lane];
                }

                for (uint t = 0; t < n_tokens; ++t) {
                    const uint k_off  = iq3 * args.sq3 + t * args.sq2 + iq1 * args.sq1;
                    const uint v_off  = iv3 * args.sv3 + t * args.sv2 + iv1 * args.sv1;
                    const uint gb_off = iv3 * args.sb3 + t * args.sb2 + iv1 * args.sb1;
                    const float beta_val = data_beta[gb_off];

                    float k_reg[rows_per_lane];
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        const uint i = r * lanes_per_col + lane;
                        k_reg[r] = data_k[k_off + i];
                    }

                    float eq_reg[rows_per_lane];
                    if (kda) {
                        const uint g_base = gb_off * S_v;
                        FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                            const uint i = r * lanes_per_col + lane;
                            eq_reg[r] = exp(data_g[g_base + i]);
                        }
                    } else {
                        const float g_val = exp(data_g[gb_off]);
                        FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                            eq_reg[r] = g_val;
                        }
                    }

                    const float v_val = data_v[v_off + j];

                    float kv_shard = 0.0;
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        kv_shard = fma(eq_reg[r], s_shard[r] * k_reg[r], kv_shard);
                    }
                    const float kv_j = gdn_reduce_partial(kv_shard);

                    const float u_j     = v_val - kv_j;
                    const float delta_j = u_j * beta_val;

                    if (lane == 0u) {
                        data_dst[sc_u + (ulong) t * S_v + j] = u_j;
                    }

                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        s_shard[r] = fma(eq_reg[r], s_shard[r], k_reg[r] * delta_j);
                        const uint i = r * lanes_per_col + lane;
                        data_dst[sc_S + (ulong) t * state_size + j * S_v + i] = s_shard[r];
                    }
                }
            }

            // No barrier between A1 and A2: A2 reads no scratch (u_hist and S_hist are
            // consumed by the row pass, after scratch_barrier).

            // ---------- phase A2: reverse scan, store A_hist / sd / d_v / d_state ----------
            for (uint wave = 0; wave < S_v / cols_per_step; ++wave) {
                const uint which_col = tpitg.x / lanes_per_col;
                const uint lane      = tpitg.x % lanes_per_col;
                const uint j         = wave * cols_per_step + which_col;

                float carry_shard[rows_per_lane];
                FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                    carry_shard[r] = 0.0;
                }

                for (int t = int(n_tokens) - 1; t >= 0; t--) {
                    const uint ut = uint(t);
                    const uint q_off  = iq3 * args.sq3 + ut * args.sq2 + iq1 * args.sq1;
                    const uint k_off  = q_off;
                    const uint gb_off = iv3 * args.sb3 + ut * args.sb2 + iv1 * args.sb1;
                    const float beta_val = data_beta[gb_off];

                    float k_reg[rows_per_lane];
                    float q_reg[rows_per_lane];
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        const uint i = r * lanes_per_col + lane;
                        k_reg[r] = data_k[k_off + i];
                        q_reg[r] = data_q[q_off + i];
                    }

                    // the scalar gate costs one register, so hoist it here where the token body
                    // covers its latency; the KDA gate is loaded at its use site below instead
                    const float g_val = kda ? 0.0f : exp(data_g[gb_off]);

                    const uint do_off = (iv3 * n_tokens * H + iv1) * S_v + ut * S_v * H;
                    const float do_j = data_d[do_off + j];

                    // A += scale * q (x) do ; plus state-output gradient seed (covers K=1 final
                    // state and K>1 snapshots via target_slot; matches the CPU kernel).
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        const uint i = r * lanes_per_col + lane;
                        carry_shard[r] += scale * q_reg[r] * do_j;
                    }
                    {
                        // matches the forward op's slot mapping: slot 0 = most recent state (t = n_tokens-1).
                        const int target_slot = int(n_tokens) - 1 - int(t);
                        if (target_slot >= 0 && target_slot < int(K)) {
                            const uint dss = args.s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                            FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                                const uint i = r * lanes_per_col + lane;
                                carry_shard[r] += data_d[dss + j * S_v + i];
                            }
                        }
                    }

                    // store A_hist[t] (adjoint used by the row pass)
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        const uint i = r * lanes_per_col + lane;
                        data_dst[sc_A + (ulong) ut * state_size + j * S_v + i] = carry_shard[r];
                    }

                    // sd = (A^T k)_j ; w = beta * sd  (u/delta come from A1's replay)
                    float sd_shard = 0.0;
                    FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                        sd_shard = fma(carry_shard[r], k_reg[r], sd_shard);
                    }
                    const float sd  = gdn_reduce_partial(sd_shard);
                    const float w_j = beta_val * sd;

                    if (lane == 0u) {
                        data_dst[sc_sd + (ulong) ut * S_v + j] = sd;
                        data_dst[args.off_dv + (iv1 + H * (ut + n_tokens * iv3)) * S_v + j] = w_j;
                    }

                    // propagate A_{t-1} = diag(exp(g)) (A - k w^T)
                    if (kda) {
                        const uint g_base = gb_off * S_v;
                        FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                            const uint i = r * lanes_per_col + lane;
                            carry_shard[r] = fma(k_reg[r], -w_j, carry_shard[r]) * exp(data_g[g_base + i]);
                        }
                    } else {
                        FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                            carry_shard[r] = fma(k_reg[r], -w_j, carry_shard[r]) * g_val;
                        }
                    }
                }

                // initial-state gradient: layout [S_v, S_v, H, n_seqs] (no K factor, unlike the forward output).
                FOR_UNROLL (uint r = 0; r < rows_per_lane; ++r) {
                    const uint i = r * lanes_per_col + lane;
                    data_dst[args.off_ds + (iv3 * H + iv1) * state_size + j * S_v + i] = carry_shard[r];
                }
            }

            // A1/A2 scratch (S_hist / A_hist / u / sd) -> phase B handoff: phase B reads
            // these across all columns, i.e. across threads. This barrier is required.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // ---------- phase B: row pass for d_q / d_k / d_g / d_beta ----------
            // T_TILE tokens per iteration; tail threads (t >= n_tokens) run on clamped
            // addresses with discarded results so control flow stays uniform for the
            // barriers in reduce_token_block.
            for (uint tb = 0; tb < n_tokens; tb += t_tile) {
                const uint t    = tb + t_sub;
                const bool t_ok = (t < n_tokens);
                const uint tc   = t_ok ? t : (n_tokens - 1u);   // clamped for in-bounds addressing

                const uint k_off  = iq3 * args.sq3 + tc * args.sq2 + iq1 * args.sq1;
                const uint gb_off = iv3 * args.sb3 + tc * args.sb2 + iv1 * args.sb1;
                const uint do_off = (iv3 * n_tokens * H + iv1) * S_v + tc * S_v * H;
                const uint tv     = iv1 + H * (t + n_tokens * iv3);  // (v-head, token) output index
                const float beta_val = data_beta[gb_off];

                // this token's scratch
                const ulong st_base = sc_S  + (ulong) tc * state_size;
                const ulong a_base  = sc_A  + (ulong) tc * state_size;
                const ulong u_base  = sc_u  + (ulong) tc * S_v;
                const ulong sd_base = sc_sd + (ulong) tc * S_v;

                float dbeta  = 0.0;
                float dg_tot = 0.0;

                for (uint rw = 0; rw < row_waves; ++rw) {
                    const uint i = rw * rows_active + i_lane;
                    const float k_i = data_k[k_off + i];

                    float dq = 0.0, dk = 0.0, dg = 0.0;
                    dbeta = 0.0;   // every row wave recomputes the same per-token value
                    for (uint jj = 0; jj < S_v; ++jj) {
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
                        const uint row = (iq1 + neq1 * (t + n_tokens * iq3)) * S_v + i;
                        if (gi == 0u && sgi == 0u) {
                            data_dst[row]          = scale * dq;
                            data_dst[args.off_dk + row] = dk;
                        } else {
                            data_dst[row]          = fma(scale, dq, data_dst[row]);
                            data_dst[args.off_dk + row] += dk;
                        }
                        if (kda) {
                            data_dst[args.off_dg + tv * S_v + i] = dg;
                        }
                    }
                    if (!kda) {
                        // scalar d_g: sum over this token's rows
                        dg_tot += gdn_reduce_token_block(
                                    t_ok ? dg : 0.0,
                                    tid,
                                    rows_active,
                                    simdgroup_width,
                                    tiisg,
                                    sgitg,
                                    sh_sg);
                    }
                }

                // dbeta is identical across a token's rows (broadcast operands), so any
                // row wave's value is the full per-token sum.
                if (t_ok && i_lane == 0u) {
                    data_dst[args.off_db + tv] = dbeta;
                    if (!kda) {
                        data_dst[args.off_dg + tv] = dg_tot;
                    }
                }
            }

            // scratch is reused by the next group member; the last iteration needs no barrier
            if (gi + 1u < group || sgi + 1u < rq3) {
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            }
        }
    }
}

kernel void kernel_gated_delta_net_back(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const float * data_q,
        device const float * data_k,
        device const float * data_v,
        device const float * data_g,
        device const float * data_beta,
        device const float * data_state,
        device const float * data_d,
        device       float * data_dst,
        threadgroup  float * sh_sg[[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 threads_per_tg[[threads_per_threadgroup]],
        uint  tid[[thread_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  simdgroup_width[[threads_per_simdgroup]],
        uint  simdgroups_per_tg[[simdgroups_per_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

// the impl is templated only on S_v / lanes_per_col; the host restricts S_v to a power
// of two <= 256 and always picks a lanes_per_col that divides it, so the powers of two
// up to 256/8 cover every dispatchable (S_v, lanes_per_col) pair.
#define GDN_FOR_EACH_ROWS_PER_LANE(X) \
    X(1)                              \
    X(2)                              \
    X(4)                              \
    X(8)                              \
    X(16)                             \
    X(32)

#define GDN_BACK_IMPL(ROWS_PER_LANE)                                \
    if (FC_gdn_back_S_v == (ROWS_PER_LANE)*FC_gdn_lanes_per_col) {  \
        gated_delta_net_back_impl<                                  \
            (ROWS_PER_LANE)>(                                       \
            args,                                                   \
            data_q, data_k, data_v, data_g,                         \
            data_beta, data_state,                                  \
            data_d, data_dst, sh_sg,                                \
            tgpig, tpitg, threads_per_tg, tid, tiisg,               \
            simdgroup_width, simdgroups_per_tg, sgitg);             \
        return;                                                     \
    }

GDN_FOR_EACH_ROWS_PER_LANE(GDN_BACK_IMPL)

#undef GDN_BACK_IMPL
#undef GDN_FOR_EACH_ROWS_PER_LANE
}
