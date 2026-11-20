#include "common.h"
//
// differences from ggml-cuda/fattn-back-mma.cuh:
//   - simdgroup_float8x8 runs at full rate on Apple GPUs, so everything stays F32: no F16 copies of q, k, v and grad_o,
//     no 2^e scaling of grad_o, and the kernels read the tensors directly
//
//   - 8x8 matrices instead of 16x8 tiles, a simdgroup owns 8 outer rows instead of 16
//
//   - the matrix element layout is not exposed, so S, dP and dS go through threadgroup memory between products
//
//   - no cp.async, so one tile buffer instead of two, kept at BC rows to fit more threadgroups per core: the stationary
//     rows and the results go through it in rounds, fab_mma_dq loads V and then K into it, and on large grids
//     fab_mma_dkv loads dO, Q, then dO again (13-15 KB instead of 21-23 KB at D = 128)
//
//   - fully masked tiles are found on the fly instead of from fab_mma_mask_flags
//
//   - -FLT_MAX instead of -INFINITY for masked entries, the library is built with fast-math
//
//   - fab_mma_dkv keeps K and V in registers, as F32 they do not fit next to the Q and dO tiles
//
#define FAB_MMA_NTHREADS (OP_FLASH_ATTN_EXT_BACK_MMA_NSG*N_SIMDWIDTH)

// rows [0, nrows) of one head into smem [nrows][D + 4] as F32, zero from row nrows_valid
template <typename T, uint D, uint nrows>
inline void fab_mma_cp_rows(threadgroup float * smem,
                            device const char * src,
                            const uint64_t nb1,
                            const uint nrows_valid,
                            const uint tid) {
    constexpr uint stride = D + 4;
    for (uint c = tid; c < nrows*D; c += FAB_MMA_NTHREADS) {
        const uint r = c/D;
        const uint k = c % D;
        smem[r*stride + k] = r < nrows_valid ?
            float(((device const T *) (src + (uint64_t) r*nb1))[k]) : 0.0f;
    }
}

// loads a mask tile, -FLT_MAX outside the valid rows and columns (0 inside when there is no mask);
// returns whether this thread saw a visible entry
template <uint nrows, uint ncols>
inline bool fab_mma_ld_mask(threadgroup float * smem,
                            device const char * m_h,
                            const uint64_t nbm1,
                            const uint row0, const uint nrows_valid,
                            const uint col0, const uint ncols_valid,
                            const uint tid) {

    constexpr uint stride = ncols + 4;

    bool any = false;
    for (uint c = tid; c < nrows*ncols; c += FAB_MMA_NTHREADS) {
        const uint r = c/ncols;
        const uint k = c % ncols;

        float v = -FLT_MAX;
        if (r < nrows_valid && k < ncols_valid) {
            v = m_h ? float(((device const half *) (m_h + (uint64_t) (row0 + r)*nbm1))[col0 + k]) : 0.0f;
        }

        any = any || v > -MAXHALF;
        smem[r*stride + k] = v;
    }
    return any;
}

// whether the tile has any visible mask entry (CUDA precomputes this in fab_mma_mask_flags);
// it contains a barrier, so every thread must call it, with a barrier between calls
inline bool fab_mma_valid(threadgroup uint * flags, const bool any, const uint warp, const uint lane) {
    const bool warp_any = simd_any(any);
    if (lane == 0) {
        flags[warp] = warp_any ? 1 : 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool res = false;
    FOR_UNROLL (uint w = 0; w < OP_FLASH_ATTN_EXT_BACK_MMA_NSG; ++w) {
        res = res || flags[w] != 0;
    }
    return res;
}

template <uint D>
inline void fab_mma_load_A(thread simdgroup_float8x8 (&A)[D/8],
                           threadgroup const float * rows) {
    constexpr uint stride = D + 4;
    FOR_UNROLL (uint k = 0; k < D/8; ++k) {
        simdgroup_load(A[k], rows + k*8, stride);
    }
}

// rows [0, BR) of the block, zero from nrows_valid, into each simdgroup's A, through nr rows of stage at a time
template <typename T, uint D, uint nr>
inline void fab_mma_stage_A(thread simdgroup_float8x8 (&A)[D/8],
                            threadgroup float * stage,
                            device const char * src,
                            const uint64_t nb1,
                            const uint nrows_valid,
                            const uint warp,
                            const uint tid) {
    static_assert(nr % 8 == 0, "a simdgroup's rows must be in one round");

    constexpr uint stride = D + 4;
    FOR_UNROLL (uint r0 = 0; r0 < OP_FLASH_ATTN_EXT_BACK_MMA_BR; r0 += nr) {

        fab_mma_cp_rows<T, D, nr>(stage, src + (uint64_t) r0*nb1, nb1, nrows_valid > r0 ? nrows_valid - r0 : 0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp*8 >= r0 && warp*8 < r0 + nr) {
            fab_mma_load_A<D>(A, stage + (warp*8 - r0)*stride);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// loads the block's K and V rows into each simdgroup's K_A and V_A
template <typename T_KV, uint D, uint nr>
inline void fab_mma_stage_KV(thread simdgroup_float8x8 (&K_A)[D/8],
                             thread simdgroup_float8x8 (&V_A)[D/8],
                             threadgroup float * stage,
                             device const char * k_h,
                             device const char * v_h,
                             constant ggml_metal_kargs_ext_back_kargs & args,
                             const uint j0,
                             const uint warp,
                             const uint tid) {
    const uint n_kv = (uint) args.n_kv;
    fab_mma_stage_A<T_KV, D, nr>(K_A, stage, k_h + (uint64_t) j0*args.nbk1, args.nbk1, n_kv - j0, warp, tid);
    fab_mma_stage_A<T_KV, D, nr>(V_A, stage, v_h + (uint64_t) j0*args.nbv1, args.nbv1, n_kv - j0, warp, tid);
}

// C = A tile^T into the simdgroup's [8][BC] rows of C
template <uint D>
inline void fab_mma_product(threadgroup float * C,
                            thread simdgroup_float8x8 (&A)[D/8],
                            threadgroup const float * tile) {
    constexpr uint stride = D + 4;
    FOR_UNROLL (uint n = 0; n < OP_FLASH_ATTN_EXT_BACK_MMA_BC/8; ++n) {
        simdgroup_float8x8 Cn = make_filled_simdgroup_matrix<float, 8>(0.0f);

        FOR_UNROLL (uint k = 0; k < D/8; ++k) {
            simdgroup_float8x8 B;
            simdgroup_load(B, tile + (n*8)*stride + k*8, stride, 0, true);
            simdgroup_multiply_accumulate(Cn, A[k], B, Cn);
        }

        simdgroup_store(Cn, C + n*8, OP_FLASH_ATTN_EXT_BACK_MMA_BC);
    }
}

// Acc[m] += A tile, A the simdgroup's [8][BC] rows
template <uint D>
inline void fab_mma_product_t(thread simdgroup_float8x8 (&Acc)[D/8],
                              threadgroup const float * A,
                              threadgroup const float * tile) {
    constexpr uint stride = D + 4;
    FOR_UNROLL (uint kk = 0; kk < OP_FLASH_ATTN_EXT_BACK_MMA_BC/8; ++kk) {
        simdgroup_float8x8 Ak;
        simdgroup_load(Ak, A + kk*8, OP_FLASH_ATTN_EXT_BACK_MMA_BC);

        FOR_UNROLL (uint m = 0; m < D/8; ++m) {
            simdgroup_float8x8 B;
            simdgroup_load(B, tile + (kk*8)*stride + m*8, stride);
            simdgroup_multiply_accumulate(Acc[m], Ak, B, Acc[m]);
        }
    }
}

// the simdgroup's 8 rows of Acc times factor into dst, through its [8][D + 4] staging rows
template <uint D>
inline void fab_mma_store(device float * dst,
                          thread simdgroup_float8x8 (&Acc)[D/8],
                          const uint nrows_valid,
                          const float factor,
                          threadgroup float * rows,
                          const uint warp,
                          const uint lane) {
    constexpr uint stride = D + 4;

    FOR_UNROLL (uint m = 0; m < D/8; ++m) {
        simdgroup_store(Acc[m], rows + m*8, stride);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = lane; c < 8*D; c += N_SIMDWIDTH) {
        const uint row = warp*8 + c/D;
        const uint col = c % D;
        if (row < nrows_valid) {
            dst[(uint64_t) row*D + col] = factor*rows[(c/D)*stride + col];
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// fab_mma_store of every simdgroup, through nr rows of stage at a time; with nr < BR a barrier ends each round
template <uint D, uint nr>
inline void fab_mma_store_staged(device float * dst,
                                 thread simdgroup_float8x8 (&Acc)[D/8],
                                 const uint nrows_valid,
                                 const float factor,
                                 threadgroup float * stage,
                                 const uint warp,
                                 const uint lane) {
    static_assert(nr % 8 == 0, "a simdgroup's rows must be in one round");
    constexpr uint stride = D + 4;

    FOR_UNROLL (uint r0 = 0; r0 < OP_FLASH_ATTN_EXT_BACK_MMA_BR; r0 += nr) {
        if (warp*8 >= r0 && warp*8 < r0 + nr) {
            fab_mma_store<D>(dst, Acc, nrows_valid, factor, stage + (warp*8 - r0)*stride, warp, lane);
        }

        if (nr < OP_FLASH_ATTN_EXT_BACK_MMA_BR) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

inline device const char * fab_mma_mask_head(constant ggml_metal_kargs_ext_back_kargs & args,
                                             device const half * mask,
                                             const uint h,
                                             const uint b) {
    return mask ? (device const char *) mask + (h % (uint) args.nem2)*args.nbm2 + (b % (uint) args.nem3)*args.nbm3
                : nullptr;
}

// grid (ceil(n_q/BR), n_head, ne3)
template <typename T_KV, uint D>
kernel void fab_mma_stats(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const T_KV  * k        [[buffer(2)]],
        device const half  * mask     [[buffer(5)]],
        device const float * o        [[buffer(6)]],
        device const float * grad_o   [[buffer(3)]],
              device float * dst      [[buffer(4)]],
          threadgroup char * smem_raw [[threadgroup(0)]],
                     uint3   tgpig    [[threadgroup_position_in_grid]],
                     ushort  tiitg    [[thread_index_in_threadgroup]],
                     ushort  sgitg    [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg    [[thread_index_in_simdgroup]]) {

    constexpr uint stride = D + 4;
    constexpr uint MS     = OP_FLASH_ATTN_EXT_BACK_MMA_BC + 4;

    threadgroup float * tiles = (threadgroup float *) smem_raw;               // K, also the Q staging area
    threadgroup float * masks = tiles + OP_FLASH_ATTN_EXT_BACK_MMA_BC*stride; // [BR][MS]
    threadgroup float * SC    = masks + OP_FLASH_ATTN_EXT_BACK_MMA_BR*MS;     // [BR][BC]
    threadgroup uint   * flags = (threadgroup uint *) (SC + OP_FLASH_ATTN_EXT_BACK_MMA_BR*OP_FLASH_ATTN_EXT_BACK_MMA_BC);

    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint n_q    = (uint) args.n_q;
    const uint n_kv   = (uint) args.n_kv;
    const uint n_head = (uint) args.n_head;
    const uint i0     = tgpig.x*OP_FLASH_ATTN_EXT_BACK_MMA_BR;
    const uint h      = tgpig.y;
    const uint b      = tgpig.z;
    const uint hk     = h/(n_head/(uint) args.n_head_kv);
    const uint warp   = sgitg;
    const uint lane   = tiisg;
    const uint tid    = tiitg;

    device const char  * q_h  = (device const char *) q + h*args.nbq2  + b*args.nbq3;
    device const char  * k_h  = (device const char *) k + hk*args.nbk2 + b*args.nbk3;
    device const char  * m_h  = fab_mma_mask_head(args, pmask, h, b);
    device const float * o_h  = o      + ((uint64_t) h       + (uint64_t) b*n_head*n_q)*D;
    device const float * g_h  = grad_o + ((uint64_t) h       + (uint64_t) b*n_head*n_q)*D;
    device       float * st_h = dst    + 2*((uint64_t) h*n_q + (uint64_t) b*n_q*n_head);

    // stationary Q rows of the simdgroup, staged through the K tile
    simdgroup_float8x8 Q_A[D/8];
    fab_mma_stage_A<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(Q_A, tiles, q_h + (uint64_t) i0*args.nbq1,
                                                             args.nbq1, n_q - i0, warp, tid);

    // online max and sum of the thread's row, 4 lanes per row
    float M = -FLT_MAX/2.0f;
    float S = 0.0f;

    const uint n_tiles = (n_kv + OP_FLASH_ATTN_EXT_BACK_MMA_BC - 1)/OP_FLASH_ATTN_EXT_BACK_MMA_BC;

    for (uint t = 0; t < n_tiles; ++t) {

        const bool any = fab_mma_ld_mask<OP_FLASH_ATTN_EXT_BACK_MMA_BR, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            masks, m_h, args.nbm1, i0,
            n_q - i0, t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            n_kv - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );

        if (m_h && !fab_mma_valid(flags, any, warp, lane)) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        fab_mma_cp_rows<T_KV, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            tiles,
            k_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*args.nbk1,
            args.nbk1,
            n_kv - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * tile_K = tiles;
        threadgroup const float * tile_M = masks;

        fab_mma_product<D>(SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, Q_A, tile_K);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[OP_FLASH_ATTN_EXT_BACK_MMA_BC/4];
        float mx = -FLT_MAX/2.0f;
        FOR_UNROLL (uint l = 0; l < OP_FLASH_ATTN_EXT_BACK_MMA_BC/4; ++l) {
            const uint i = warp*8 + lane/4;
            const uint j = l*4 + lane%4;

            const float m = tile_M[i*MS + j];
            s[l] = m <= -MAXHALF ? -FLT_MAX
                                 : SC[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j]*args.scale + m;

            mx = max(mx, s[l]);
        }

        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));

        const float M_new = max(M, mx);
        S *= exp(M - M_new);
        M  = M_new;

        FOR_UNROLL (uint l = 0; l < OP_FLASH_ATTN_EXT_BACK_MMA_BC/4; ++l) {
            S += exp(s[l] - M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S += simd_shuffle_xor(S, 1);
    S += simd_shuffle_xor(S, 2);
    {
        const uint row = i0 + warp*8 + lane/4;
        if (lane % 4 == 0 && row < n_q) {
            st_h[2*row] = S == 0.0f ? -INFINITY
                                    : M + log(S);
        }
    }

    // D = dot(grad_o, o) per row, the simdgroup sweeps its 8 rows
    for (uint r = 0; r < 8; ++r) {
        const uint row = i0 + warp*8 + r;
        if (row >= n_q) {
            break;
        }

        device const float * orow = o_h + (uint64_t) row*n_head*D;
        device const float * grow = g_h + (uint64_t) row*n_head*D;

        float d = 0.0f;
        for (uint c = lane; c < D; c += N_SIMDWIDTH) {
            d += orow[c]*grow[c];
        }

        d = simd_sum(d);

        if (lane == 0) {
            st_h[2*row + 1] = d;
        }
    }
}


// grid (ceil(n_q/BR), n_head, ne3)
template <typename T_KV, uint D>
kernel void fab_mma_dq(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const T_KV  * k        [[buffer(2)]],
        device const T_KV  * v        [[buffer(7)]],
        device const half  * mask     [[buffer(5)]],
        device const float * grad_o   [[buffer(3)]],
        device const float * stats    [[buffer(8)]],
              device float * dst      [[buffer(4)]],
          threadgroup char * smem_raw [[threadgroup(0)]],
                     uint3   tgpig    [[threadgroup_position_in_grid]],
                     ushort  tiitg    [[thread_index_in_threadgroup]],
                     ushort  sgitg    [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg    [[thread_index_in_simdgroup]]) {
    constexpr uint stride = D + 4;
    constexpr uint MS     = OP_FLASH_ATTN_EXT_BACK_MMA_BC + 4;

    threadgroup float * tiles = (threadgroup float *) smem_raw;                 // V then K, also stages Q, dO and dQ
    threadgroup float * masks = tiles + OP_FLASH_ATTN_EXT_BACK_MMA_BC*stride; // [BR][MS]
    threadgroup float * SC    = masks + OP_FLASH_ATTN_EXT_BACK_MMA_BR*MS;     // [BR][BC]
    threadgroup float * dP    = SC    + OP_FLASH_ATTN_EXT_BACK_MMA_BR*OP_FLASH_ATTN_EXT_BACK_MMA_BC;                // [BR][BC]
    threadgroup uint   * flags = (threadgroup uint *) (dP + OP_FLASH_ATTN_EXT_BACK_MMA_BR*OP_FLASH_ATTN_EXT_BACK_MMA_BC);

    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint n_q    = (uint) args.n_q;
    const uint n_kv   = (uint) args.n_kv;
    const uint n_head = (uint) args.n_head;
    const uint i0     = tgpig.x*OP_FLASH_ATTN_EXT_BACK_MMA_BR;
    const uint h      = tgpig.y;
    const uint b      = tgpig.z;
    const uint hk     = h/(n_head/(uint) args.n_head_kv);
    const uint warp   = sgitg;
    const uint lane   = tiisg;
    const uint tid    = tiitg;

    device const char  * q_h  = (device const char *) q + h*args.nbq2  + b*args.nbq3;
    device const char  * g_h  = (device const char *) (grad_o + ((uint64_t) h + (uint64_t) b*n_head*n_q)*D);
    device const char  * k_h  = (device const char *) k + hk*args.nbk2 + b*args.nbk3;
    device const char  * v_h  = (device const char *) v + hk*args.nbv2 + b*args.nbv3;

    device const char  * m_h  = fab_mma_mask_head(args, pmask, h, b);
    device const float * st_h = stats + 2*((uint64_t) h*n_q + (uint64_t) b*n_q*n_head);
    device       float * d_h  = dst   +   ((uint64_t) h*n_q + (uint64_t) b*n_q*n_head)*D;

    const uint64_t nbg1 = (uint64_t) n_head*D*sizeof(float);

    // stationary Q and dO rows of the simdgroup
    simdgroup_float8x8 Q_A[D/8];
    simdgroup_float8x8 G_A[D/8];
    fab_mma_stage_A<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
        Q_A,
        tiles,
        q_h + (uint64_t) i0*args.nbq1,
        args.nbq1,
        n_q - i0,
        warp,
        tid
    );

    fab_mma_stage_A<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
        G_A,
        tiles,
        g_h + (uint64_t) i0*nbg1,
        nbg1,
        n_q - i0,
        warp,
        tid
    );

    // stats of the thread's row, rows past n_q are fully masked by fab_mma_ld_mask
    float L, Dd;
    {
        const uint row = i0 + warp*8 + lane/4;
        L  = row < n_q ? st_h[2*row]     : 0.0f;
        Dd = row < n_q ? st_h[2*row + 1] : 0.0f;
    }

    simdgroup_float8x8 dQ[D/8];
    FOR_UNROLL (uint m = 0; m < D/8; ++m) {
        dQ[m] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint n_tiles = (n_kv + OP_FLASH_ATTN_EXT_BACK_MMA_BC - 1)/OP_FLASH_ATTN_EXT_BACK_MMA_BC;

    for (uint t = 0; t < n_tiles; ++t) {

        const bool any = fab_mma_ld_mask<OP_FLASH_ATTN_EXT_BACK_MMA_BR, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            masks,
            m_h,
            args.nbm1,
            i0,
            n_q - i0, t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            n_kv - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );

        if (m_h && !fab_mma_valid(flags, any, warp, lane)) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // one tile: V for dP, then K over it
        threadgroup float * tile_V = tiles;
        threadgroup float * tile_K = tiles;
        fab_mma_cp_rows<T_KV, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            tile_V,
            v_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*args.nbv1,
            args.nbv1,
            n_kv - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        fab_mma_product<D>(dP + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, G_A, tile_V);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        fab_mma_cp_rows<T_KV, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            tile_K,
            k_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*args.nbk1,
            args.nbk1,
            n_kv - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        fab_mma_product<D>(SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, Q_A, tile_K);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * tile_M = masks;

        // dS = P * (dP - D), kept in SC; the softmax scale is applied at the store
        FOR_UNROLL (uint l = 0; l < OP_FLASH_ATTN_EXT_BACK_MMA_BC/4; ++l) {
            const uint i = warp*8 + lane/4;
            const uint j = l*4 + lane%4;

            const float m = tile_M[i*MS + j];
            const float s = SC[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j]*args.scale + m;
            const float p = m <= -MAXHALF ? 0.0f : exp(s - L);

            SC[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j] = p*(dP[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j] - Dd);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        fab_mma_product_t<D>(dQ, SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, tile_K);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    fab_mma_store_staged<D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
        d_h + (uint64_t) i0*D,
        dQ,
        n_q - i0,
        args.scale,
        tiles,
        warp,
        lane
    );
}


// if fab_mma_dkv splits over the q tiles, without a split it compiles to the kernel without one
constant bool FC_flash_attn_ext_back_mma_split [[function_constant(FC_FLASH_ATTN_EXT_BACK_MMA + 0)]];

// if fab_mma_dkv takes dO, Q and dO again through one tile instead of Q and dO side by side, on large grids
constant bool FC_flash_attn_ext_back_mma_dkv_seq [[function_constant(FC_FLASH_ATTN_EXT_BACK_MMA + 1)]];

// grid (ceil(n_kv/BR), n_head_kv, ne3*nsplit),
//     with nsplit > 1 split s writes its partial dK and dV at dst + s*split_stride
template <typename T_KV, uint D>
kernel void fab_mma_dkv(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const T_KV  * k        [[buffer(2)]],
        device const T_KV  * v        [[buffer(7)]],
        device const half  * mask     [[buffer(5)]],
        device const float * grad_o   [[buffer(3)]],
        device const float * stats    [[buffer(8)]],
              device float * dst      [[buffer(4)]],
          threadgroup char * smem_raw [[threadgroup(0)]],
                     uint3   tgpig    [[threadgroup_position_in_grid]],
                     uint3   tgpg     [[threadgroups_per_grid]],
                     ushort  tiitg    [[thread_index_in_threadgroup]],
                     ushort  sgitg    [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg    [[thread_index_in_simdgroup]]) {
    constexpr uint stride = D + 4;
    constexpr uint MS     = OP_FLASH_ATTN_EXT_BACK_MMA_BR + 4; // mask tile [BC q rows][BR kv], read transposed
    const uint nt         = FC_flash_attn_ext_back_mma_dkv_seq ?
                                OP_FLASH_ATTN_EXT_BACK_MMA_BC :
                                MAX(OP_FLASH_ATTN_EXT_BACK_MMA_BR, 2*OP_FLASH_ATTN_EXT_BACK_MMA_BC);

    threadgroup float * tiles = (threadgroup float *) smem_raw;             // [Q][dO] or one tile, also stages K, V, dK and dV
    threadgroup float * masks = tiles + nt*stride;                          // [BC][MS]
    threadgroup float * SC    = masks + OP_FLASH_ATTN_EXT_BACK_MMA_BC*MS; // [BR][BC]
    threadgroup float * dP    = SC    + OP_FLASH_ATTN_EXT_BACK_MMA_BR*OP_FLASH_ATTN_EXT_BACK_MMA_BC; // [BR][BC]
    threadgroup float * sh_LD = dP    + OP_FLASH_ATTN_EXT_BACK_MMA_BR*OP_FLASH_ATTN_EXT_BACK_MMA_BC; // [L[BC], D[BC]]
    threadgroup uint   * flags = (threadgroup uint *) (sh_LD + 2*OP_FLASH_ATTN_EXT_BACK_MMA_BC);

    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint n_q       = (uint) args.n_q;
    const uint n_kv      = (uint) args.n_kv;
    const uint n_head    = (uint) args.n_head;
    const uint n_head_kv = (uint) args.n_head_kv;
    const uint ne3       = (uint) args.ne3;
    const uint j0        = tgpig.x*OP_FLASH_ATTN_EXT_BACK_MMA_BR;
    const uint hk        = tgpig.y;
    const uint b         = FC_flash_attn_ext_back_mma_split ? tgpig.z % ne3 : tgpig.z;
    const uint split     = FC_flash_attn_ext_back_mma_split ? tgpig.z/ne3 : 0;
    const uint nsplit    = FC_flash_attn_ext_back_mma_split ? tgpg.z/ne3  : 1;
    const uint rk2       = n_head/n_head_kv;
    const uint warp      = sgitg;
    const uint lane      = tiisg;
    const uint tid       = tiitg;

    device const char * k_h = (device const char *) k + hk*args.nbk2 + b*args.nbk3;
    device const char * v_h = (device const char *) v + hk*args.nbv2 + b*args.nbv3;

    const uint64_t nbg1 = (uint64_t) n_head*D*sizeof(float);

    // stationary K and V rows of the simdgroup
    simdgroup_float8x8 K_A[D/8];
    simdgroup_float8x8 V_A[D/8];

    if (FC_flash_attn_ext_back_mma_dkv_seq) {
        fab_mma_stage_KV<T_KV, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(K_A, V_A, tiles, k_h, v_h, args, j0, warp, tid);
    } else {
        fab_mma_stage_KV<T_KV, D, OP_FLASH_ATTN_EXT_BACK_MMA_BR>(K_A, V_A, tiles, k_h, v_h, args, j0, warp, tid);
    }

    simdgroup_float8x8 dK[D/8];
    simdgroup_float8x8 dV[D/8];
    FOR_UNROLL (uint m = 0; m < D/8; ++m) {
        dK[m] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        dV[m] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    // the q tiles of every head of the group, one flat sequence
    const uint n_tq    = (n_q + OP_FLASH_ATTN_EXT_BACK_MMA_BC - 1)/OP_FLASH_ATTN_EXT_BACK_MMA_BC;
    const uint u_begin = (uint) ((uint64_t) split      *rk2*n_tq/nsplit);
    const uint n_tiles = (uint) ((uint64_t) (split + 1)*rk2*n_tq/nsplit);

    for (uint u = u_begin; u < n_tiles; ++u) {
        const uint h = hk*rk2 + u/n_tq;
        const uint t = u % n_tq;

        device const char  * q_h  = (device const char *) q + h*args.nbq2 + b*args.nbq3;
        device const char  * g_h  = (device const char *) (grad_o + ((uint64_t) h + (uint64_t) b*n_head*n_q)*D);
        device const char  * m_h  = fab_mma_mask_head(args, pmask, h, b);
        device const float * st_h = stats + 2*((uint64_t) h*n_q + (uint64_t) b*n_q*n_head);

        const bool any = fab_mma_ld_mask<OP_FLASH_ATTN_EXT_BACK_MMA_BC, OP_FLASH_ATTN_EXT_BACK_MMA_BR>(
            masks,
            m_h,
            args.nbm1,
            t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            n_q - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            j0,
            n_kv - j0,
            tid
        );
        if (m_h && !fab_mma_valid(flags, any, warp, lane)) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // with seq one tile: dO for dP, Q over it for S and dK, then dO again for dV
        threadgroup float * tile_Q = tiles;
        threadgroup float * tile_G = FC_flash_attn_ext_back_mma_dkv_seq ?
                                        tiles :
                                        tile_Q + OP_FLASH_ATTN_EXT_BACK_MMA_BC*stride;

        if (!FC_flash_attn_ext_back_mma_dkv_seq) {
            fab_mma_cp_rows<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
                tile_Q,
                q_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*args.nbq1,
                args.nbq1,
                n_q - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
                tid
            );
        }

        fab_mma_cp_rows<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
            tile_G,
            g_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*nbg1,
            nbg1,
            n_q - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
            tid
        );

        if (tid < OP_FLASH_ATTN_EXT_BACK_MMA_BC) {
            const uint col = t*OP_FLASH_ATTN_EXT_BACK_MMA_BC + tid;
            sh_LD[tid]    = col < n_q ? st_h[2*col] : 0.0f;
            sh_LD[OP_FLASH_ATTN_EXT_BACK_MMA_BC + tid] = col < n_q ? st_h[2*col + 1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * tile_M = masks;
        threadgroup const float * sh_L   = sh_LD;
        threadgroup const float * sh_D   = sh_L + OP_FLASH_ATTN_EXT_BACK_MMA_BC;

        // S^T and dP^T, rows kv, columns q
        if (FC_flash_attn_ext_back_mma_dkv_seq) {
            fab_mma_product<D>(dP + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, V_A, tile_G);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            fab_mma_cp_rows<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
                tile_Q, q_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*args.nbq1,
                args.nbq1,
                n_q - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
                tid
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);

            fab_mma_product<D>(SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, K_A, tile_Q);
        } else {
            fab_mma_product<D>(SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, K_A, tile_Q);
            fab_mma_product<D>(dP + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, V_A, tile_G);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // P in SC, dS = P * (dP - D) in dP; the softmax scale is applied at the store
        FOR_UNROLL (uint l = 0; l < OP_FLASH_ATTN_EXT_BACK_MMA_BC/4; ++l) {
            const uint i = warp*8 + lane/4;
            const uint j = l*4 + lane%4;
            const float m = tile_M[j*MS + i];
            const float s = SC[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j]*args.scale + m;
            const float p = m <= -MAXHALF ? 0.0f : exp(s - sh_L[j]);
            SC[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j] = p;
            dP[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j] = p*(dP[i*OP_FLASH_ATTN_EXT_BACK_MMA_BC + j] - sh_D[j]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (FC_flash_attn_ext_back_mma_dkv_seq) {
            fab_mma_product_t<D>(dK, dP + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, tile_Q);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            fab_mma_cp_rows<float, D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(
                tile_G,
                g_h + (uint64_t) t*OP_FLASH_ATTN_EXT_BACK_MMA_BC*nbg1,
                nbg1,
                n_q - t*OP_FLASH_ATTN_EXT_BACK_MMA_BC,
                tid
            );

            threadgroup_barrier(mem_flags::mem_threadgroup);
            fab_mma_product_t<D>(dV, SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, tile_G);
        } else {
            fab_mma_product_t<D>(dV, SC + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, tile_G);
            fab_mma_product_t<D>(dK, dP + warp*8*OP_FLASH_ATTN_EXT_BACK_MMA_BC, tile_Q);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t split_stride = 2*(uint64_t) n_kv*n_head_kv*ne3*D;

    device float * dk_h = dst  + split*split_stride + ((uint64_t) hk*n_kv + (uint64_t) b*n_kv*n_head_kv)*D;
    device float * dv_h = dk_h + (uint64_t) n_kv*n_head_kv*ne3*D;

    if (FC_flash_attn_ext_back_mma_dkv_seq) {
        fab_mma_store_staged<D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(dk_h + (uint64_t) j0*D, dK, n_kv - j0, args.scale, tiles, warp, lane);
        fab_mma_store_staged<D, OP_FLASH_ATTN_EXT_BACK_MMA_BC>(dv_h + (uint64_t) j0*D, dV, n_kv - j0, 1.0f,       tiles, warp, lane);
    } else {
        fab_mma_store_staged<D, OP_FLASH_ATTN_EXT_BACK_MMA_BR>(dk_h + (uint64_t) j0*D, dK, n_kv - j0, args.scale, tiles, warp, lane);
        fab_mma_store_staged<D, OP_FLASH_ATTN_EXT_BACK_MMA_BR>(dv_h + (uint64_t) j0*D, dV, n_kv - j0, 1.0f,       tiles, warp, lane);
    }
}

// fab_mma_sum_splits: dst = the sum of the nsplit partial dK and dV of fab_mma_dkv
kernel void kernel_flash_attn_ext_back_mma_sum_splits(
        constant ggml_metal_kargs_ext_back_sum_splits & args [[buffer(0)]],
        device const float * part [[buffer(1)]],
              device float * dst  [[buffer(2)]],
                     uint    tpig [[thread_position_in_grid]]) {
    const uint64_t n  = (uint64_t) args.n;
    const uint nsplit = (uint) args.nsplit;

    if (tpig >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint s = 0; s < nsplit; ++s) {
        sum += part[s*n + tpig];
    }

    dst[tpig] = sum;
}

// T_KV is in the signature (k, v), so each K/V type needs its own function type
typedef decltype(fab_mma_stats<float, 64>) fab_mma_stats_f32_t;
typedef decltype(fab_mma_stats<half,  64>) fab_mma_stats_f16_t;
typedef decltype(fab_mma_dq   <float, 64>) fab_mma_dq_f32_t;
typedef decltype(fab_mma_dq   <half,  64>) fab_mma_dq_f16_t;
typedef decltype(fab_mma_dkv  <float, 64>) fab_mma_dkv_f32_t;
typedef decltype(fab_mma_dkv  <half,  64>) fab_mma_dkv_f16_t;

template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d32")]]  kernel fab_mma_stats_f32_t fab_mma_stats<float,  32>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d64")]]  kernel fab_mma_stats_f32_t fab_mma_stats<float,  64>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d80")]]  kernel fab_mma_stats_f32_t fab_mma_stats<float,  80>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d96")]]  kernel fab_mma_stats_f32_t fab_mma_stats<float,  96>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d112")]] kernel fab_mma_stats_f32_t fab_mma_stats<float, 112>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f32_d128")]] kernel fab_mma_stats_f32_t fab_mma_stats<float, 128>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d32")]]  kernel fab_mma_stats_f16_t fab_mma_stats<half,   32>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d64")]]  kernel fab_mma_stats_f16_t fab_mma_stats<half,   64>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d80")]]  kernel fab_mma_stats_f16_t fab_mma_stats<half,   80>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d96")]]  kernel fab_mma_stats_f16_t fab_mma_stats<half,   96>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d112")]] kernel fab_mma_stats_f16_t fab_mma_stats<half,  112>;
template [[host_name("kernel_flash_attn_ext_back_mma_stats_f16_d128")]] kernel fab_mma_stats_f16_t fab_mma_stats<half,  128>;

template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d32")]]     kernel fab_mma_dq_f32_t    fab_mma_dq   <float,  32>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d64")]]     kernel fab_mma_dq_f32_t    fab_mma_dq   <float,  64>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d80")]]     kernel fab_mma_dq_f32_t    fab_mma_dq   <float,  80>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d96")]]     kernel fab_mma_dq_f32_t    fab_mma_dq   <float,  96>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d112")]]    kernel fab_mma_dq_f32_t    fab_mma_dq   <float, 112>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f32_d128")]]    kernel fab_mma_dq_f32_t    fab_mma_dq   <float, 128>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d32")]]     kernel fab_mma_dq_f16_t    fab_mma_dq   <half,   32>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d64")]]     kernel fab_mma_dq_f16_t    fab_mma_dq   <half,   64>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d80")]]     kernel fab_mma_dq_f16_t    fab_mma_dq   <half,   80>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d96")]]     kernel fab_mma_dq_f16_t    fab_mma_dq   <half,   96>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d112")]]    kernel fab_mma_dq_f16_t    fab_mma_dq   <half,  112>;
template [[host_name("kernel_flash_attn_ext_back_mma_dq_f16_d128")]]    kernel fab_mma_dq_f16_t    fab_mma_dq   <half,  128>;

template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d32")]]    kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float,  32>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d64")]]    kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float,  64>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d80")]]    kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float,  80>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d96")]]    kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float,  96>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d112")]]   kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float, 112>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f32_d128")]]   kernel fab_mma_dkv_f32_t   fab_mma_dkv  <float, 128>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d32")]]    kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,   32>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d64")]]    kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,   64>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d80")]]    kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,   80>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d96")]]    kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,   96>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d112")]]   kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,  112>;
template [[host_name("kernel_flash_attn_ext_back_mma_dkv_f16_d128")]]   kernel fab_mma_dkv_f16_t   fab_mma_dkv  <half,  128>;
