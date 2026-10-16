#pragma once

#include "common.cuh"
#include "fattn-back.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// written for the NVIDIA fragment layouts, ldmatrix and cp.async; other targets use the warp kernel
static bool ggml_cuda_flash_attn_ext_back_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_flash_attn_ext_back_kargs & args) {
    GGML_UNUSED_VARS(ctx, dst, args);
    return false;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

#include "cp-async.cuh"
#include "mma.cuh"

#define FAB_MMA_NWARPS   4
#define FAB_MMA_NTHREADS (FAB_MMA_NWARPS*WARP_SIZE)
#define FAB_MMA_BR       64 // outer rows per block
#define FAB_MMA_BC       32 // inner columns per tile
#define FAB_MMA_FB       32 // mask flag block: one flag per FB x FB mask entries, FB == BC

using namespace ggml_cuda_mma;

typedef tile<16, 8, half2> fab_tile_A;
typedef tile< 8, 8, half2> fab_tile_B;
typedef tile<16, 8, float> fab_tile_C;

// grad_o is rounded to F16 for the tensor cores;
static __global__ void fab_mma_absmax(const float * __restrict__ x, const int64_t n, int * __restrict__ out) {
    float m = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        m = fmaxf(m, fabsf(x[i]));
    }
    m = warp_reduce_max(m);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMax(out, __float_as_int(m));
    }
}

static __device__ __forceinline__ int fab_mma_gexp(const int * __restrict__ gmax) {
    const float m = __int_as_float(*gmax);
    return m > 0.0f && m < INFINITY ? max(-64, min(64, -ilogbf(m))) : 0;
}


template <typename T>
static __global__ void fab_mma_cvt_rows(
        const T * __restrict__ src, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3,
        half * __restrict__ dst, const int n_rows, const int rows_pad, const int D, const int * __restrict__ gmax) {
    const int cpr   = D/8;
    const int chunk = blockIdx.x*blockDim.x + threadIdx.x;
    if (chunk >= rows_pad*cpr) {
        return;
    }
    const int r = chunk / cpr;
    const int c = (chunk % cpr)*8;
    const int head = blockIdx.y;
    const int b    = blockIdx.z;

    half2 out[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    const float scale = gmax ? ldexpf(1.0f, fab_mma_gexp(gmax)) : 1.0f;
    if (r < n_rows) {
        const T * s = (const T *) ((const char *) src + b*nb3 + head*nb2 + (uint64_t) r*nb1) + c;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (std::is_same_v<T, half>) {
                out[i] = __halves2half2(s[2*i], s[2*i + 1]);
            } else {
                out[i] = __floats2half2_rn(scale*s[2*i], scale*s[2*i + 1]);
            }
        }
    }
    *(uint4 *) (dst + (((size_t) b*gridDim.y + head)*rows_pad + r)*D + c) = *(const uint4 *) out;
}


// one flag per FB x FB block of the mask: 1 if any entry inside [0, n_q) x [0, n_kv) is finite
static __global__ void fab_mma_mask_flags(
        const half * __restrict__ mask, const uint64_t nbm1, const uint64_t nbm2, const uint64_t nbm3, const int nem2,
        uint8_t * __restrict__ flags, const int n_q, const int n_kv, const int q_pad, const int kv_pad) {
    const int kb = blockIdx.x;
    const int qb = blockIdx.y;
    const int hb = blockIdx.z;
    const int h  = hb % nem2;
    const int b  = hb / nem2;

    bool any = false;
    for (int e = threadIdx.x; e < FAB_MMA_FB*FAB_MMA_FB; e += blockDim.x) {
        const int q  = qb*FAB_MMA_FB + e / FAB_MMA_FB;
        const int kv = kb*FAB_MMA_FB + e % FAB_MMA_FB;
        if (q < n_q && kv < n_kv) {
            const half v = mask ? ((const half *) ((const char *) mask + b*nbm3 + h*nbm2 + (uint64_t) q*nbm1))[kv] : __float2half(0.0f);
            any = any || __half2float(v) != -INFINITY;
        }
    }
    any = __syncthreads_or(any);
    if (threadIdx.x == 0) {
        flags[((size_t) hb*(q_pad/FAB_MMA_FB) + qb)*(kv_pad/FAB_MMA_FB) + kb] = any ? 1 : 0;
    }
}


template <int D, int nrows>
static __device__ __forceinline__ void fab_mma_cp_rows(half2 * __restrict__ smem, const half * __restrict__ src) {
    constexpr int cpr    = D/8; // 16 byte chunks per row
    constexpr int stride = D/2 + 4;
    const unsigned dst = ggml_cuda_cvta_generic_to_shared(smem);
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int c = tid; c < nrows*cpr; c += FAB_MMA_NTHREADS) {
        const int r = c / cpr;
        const int k = c % cpr;
        cp_async_cg_16<0>(dst + (r*stride + k*4)*sizeof(half2), src + r*D + k*8);
    }
}

// mask rows [row0, row0 + nrows) x columns [col0, col0 + ncols) of one head into smem [nrows][ncols + 8], -inf outside
// [0, nrows_valid) x [0, ncols_valid), 0 inside when there is no mask; in-bounds aligned chunks go through cp.async
template <int nrows, int ncols>
static __device__ __forceinline__ void fab_mma_ld_mask(
        half * __restrict__ smem, const half * __restrict__ m_h, const uint64_t nbm1, const bool aligned,
        const int row0, const int nrows_valid, const int col0, const int ncols_valid) {
    constexpr int cpr    = ncols/8;
    constexpr int stride = ncols + 8;
    const unsigned dst = ggml_cuda_cvta_generic_to_shared(smem);
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int c = tid; c < nrows*cpr; c += FAB_MMA_NTHREADS) {
        const int r = c / cpr;
        const int k = (c % cpr)*8;
        const bool row_ok = r < nrows_valid;
        const half * src = m_h && row_ok ? (const half *) ((const char *) m_h + (uint64_t) (row0 + r)*nbm1) + col0 + k : nullptr;
        if (aligned && row_ok && k + 8 <= ncols_valid) {
            cp_async_cg_16<0>(dst + (r*stride + k)*sizeof(half), src);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                half v = __float2half(-INFINITY);
                if (row_ok && k + e < ncols_valid) {
                    v = src ? src[e] : __float2half(0.0f);
                }
                smem[r*stride + k + e] = v;
            }
        }
    }
}

template <int D>
static __device__ __forceinline__ void fab_mma_load_A(fab_tile_A * __restrict__ A, const half2 * __restrict__ rows) {
    constexpr int stride = D/2 + 4;
#pragma unroll
    for (int k = 0; k < D/16; ++k) {
        load_ldmatrix(A[k], rows + k*8, stride);
    }
}

template <int D, bool A_in_reg>
static __device__ __forceinline__ void fab_mma_product(
        fab_tile_C * __restrict__ C, const fab_tile_A * __restrict__ A_reg, const half2 * __restrict__ A_rows,
        const half2 * __restrict__ tile) {
    constexpr int stride = D/2 + 4;
#pragma unroll
    for (int n = 0; n < FAB_MMA_BC/8; ++n) {
        C[n] = fab_tile_C();
    }
#pragma unroll
    for (int k = 0; k < D/16; ++k) {
        fab_tile_A A;
        if constexpr (A_in_reg) {
            A = A_reg[k];
        } else {
            load_ldmatrix(A, A_rows + k*8, stride);
        }
#pragma unroll
        for (int n = 0; n < FAB_MMA_BC/8; ++n) {
            fab_tile_B B;
            load_ldmatrix(B, tile + (n*8)*stride + k*8, stride);
            mma(C[n], A, B);
        }
    }
}

// Acc[m] += A
template <int D>
static __device__ __forceinline__ void fab_mma_product_t(
        fab_tile_C * __restrict__ Acc, const fab_tile_A * __restrict__ A, const half2 * __restrict__ tile) {
    constexpr int stride = D/2 + 4;
#pragma unroll
    for (int kk = 0; kk < FAB_MMA_BC/16; ++kk) {
#pragma unroll
        for (int m = 0; m < D/8; ++m) {
            fab_tile_B B;
            load_ldmatrix_trans(B, tile + (kk*16)*stride + m*4, stride);
            mma(Acc[m], A[kk], B);
        }
    }
}

static __device__ __forceinline__ fab_tile_A fab_mma_c_to_a(const fab_tile_C & c0, const fab_tile_C & c1) {
    const tile<16, 4, half2> h0 = get_half2(c0);
    const tile<16, 4, half2> h1 = get_half2(c1);
    fab_tile_A a;
    a.x[0] = h0.x[0];
    a.x[1] = h0.x[1];
    a.x[2] = h1.x[0];
    a.x[3] = h1.x[1];
    return a;
}

template <int D>
static __device__ __forceinline__ void fab_mma_store(
        float * __restrict__ dst, const fab_tile_C * __restrict__ Acc, const int nrows_valid, const float factor) {
#pragma unroll
    for (int m = 0; m < D/8; ++m) {
#pragma unroll
        for (int l = 0; l < fab_tile_C::ne; l += 2) {
            const int row = threadIdx.y*16 + fab_tile_C::get_i(l);
            const int col = m*8 + fab_tile_C::get_j(l);
            if (row < nrows_valid) {
                *(float2 *) (dst + (int64_t) row*D + col) = make_float2(factor*Acc[m].x[l], factor*Acc[m].x[l + 1]);
            }
        }
    }
}

// pointers of the converted arrays, the mask and the flags, per launch
struct fab_mma_bufs {
    const half    * q;     // [ne3][n_head][q_pad][D]
    const half    * g;     // grad_o * 2^e, same layout as q
    const half    * k;     // [ne3][n_head_kv][kv_pad][D]
    const half    * v;
    const half    * m;     // the original mask or null, strides in args
    const uint8_t * flags; // [nem3*nem2][q_pad/FB][kv_pad/FB]
    const int     * gmax;  // bits of max|grad_o|, gives e
    bool m_aligned;        // mask rows can be copied in 16 byte chunks
    int q_pad;
    int kv_pad;
};

static __device__ __forceinline__ const half * fab_mma_mask_head(
        const fab_mma_bufs & bufs, const ggml_cuda_flash_attn_ext_back_kargs & args, const int h, const int b) {
    return bufs.m ? (const half *) ((const char *) bufs.m + (h % args.nem2)*args.nbm2 + (b % args.nem3)*args.nbm3) : nullptr;
}


template <int D>
static __global__ void __launch_bounds__(FAB_MMA_NTHREADS, 1) fab_mma_stats(
        const fab_mma_bufs bufs,
        const float * __restrict__ o,
        const float * __restrict__ grad_o,
              float * __restrict__ dst,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int stride = D/2 + 4;
    constexpr int MS     = FAB_MMA_BC + 8;
    extern __shared__ char smem_raw[];
    half2 * tiles = (half2 *) smem_raw;                        // [buf][K], together also the Q staging area
    half  * masks = (half *) (tiles + 2*FAB_MMA_BC*stride);    // [buf][BR][MS]

    const int n_q    = (int) args.n_q;
    const int n_head = (int) args.n_head;
    const int i0     = blockIdx.x*FAB_MMA_BR;
    const int h      = blockIdx.y;
    const int b      = blockIdx.z;
    const int hk     = h/(n_head/(int) args.n_head_kv);
    const int hb     = (b % (int) args.nem3)*(int) args.nem2 + h % (int) args.nem2;
    const int warp   = threadIdx.y;
    const int lane   = threadIdx.x;
    const int nfb    = bufs.kv_pad/FAB_MMA_FB;

    const half    * q_h  = bufs.q + ((size_t) b*n_head + h)*bufs.q_pad*D;
    const half    * k_h  = bufs.k + ((size_t) b*args.n_head_kv + hk)*bufs.kv_pad*D;
    const half    * m_h  = fab_mma_mask_head(bufs, args, h, b);
    const uint8_t * f_h  = bufs.flags + ((size_t) hb*(bufs.q_pad/FAB_MMA_FB) + i0/FAB_MMA_FB)*nfb;
    const float   * o_h  = o      + ((int64_t) h + (int64_t) b*n_head*n_q)*D;
    const float   * g_h  = grad_o + ((int64_t) h + (int64_t) b*n_head*n_q)*D;
    float         * st_h = dst    + 2*((int64_t) h*n_q + (int64_t) b*n_q*n_head);

    fab_tile_A Q_A[D/16];
    fab_mma_cp_rows<D, FAB_MMA_BR>(tiles, q_h + (size_t) i0*D);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();
    fab_mma_load_A<D>(Q_A, tiles + warp*16*stride);
    __syncthreads();

    // online max and sum of the thread's two rows
    float M[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float S[2] = {0.0f, 0.0f};

    const int n_tiles = nfb;
    auto valid = [&](const int t) -> bool {
        return f_h[t] | f_h[nfb + t];
    };
    auto issue = [&](const int t, const int buf) {
        fab_mma_cp_rows<D, FAB_MMA_BC>(tiles + buf*FAB_MMA_BC*stride, k_h + (size_t) t*FAB_MMA_BC*D);
        fab_mma_ld_mask<FAB_MMA_BR, FAB_MMA_BC>(masks + buf*FAB_MMA_BR*MS, m_h, args.nbm1, bufs.m_aligned,
            i0, n_q - i0, t*FAB_MMA_BC, (int) args.n_kv - t*FAB_MMA_BC);
        cp_async_commit_group();
    };

    int t = 0;
    while (t < n_tiles && !valid(t)) {
        ++t;
    }
    int buf = 0;
    if (t < n_tiles) {
        issue(t, buf);
    }
    while (t < n_tiles) {
        int tn = t + 1;
        while (tn < n_tiles && !valid(tn)) {
            ++tn;
        }
        if (tn < n_tiles) {
            issue(tn, buf ^ 1);
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const half2 * tile_K = tiles + buf*FAB_MMA_BC*stride;
        const half  * tile_M = masks + buf*FAB_MMA_BR*MS;

        fab_tile_C SC[FAB_MMA_BC/8];
        fab_mma_product<D, true>(SC, Q_A, nullptr, tile_K);

        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < FAB_MMA_BC/8; ++n) {
#pragma unroll
            for (int l = 0; l < fab_tile_C::ne; ++l) {
                const int i = warp*16 + fab_tile_C::get_i(l);
                const int j = n*8 + fab_tile_C::get_j(l);
                const float s = SC[n].x[l]*args.scale + __half2float(tile_M[i*MS + j]);
                SC[n].x[l] = s;
                mx[l/2] = fmaxf(mx[l/2], s);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xFFFFFFFF, mx[r], 1, WARP_SIZE));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xFFFFFFFF, mx[r], 2, WARP_SIZE));
            const float M_new = fmaxf(M[r], mx[r]);
            S[r] *= expf(M[r] - M_new);
            M[r]  = M_new;
        }
#pragma unroll
        for (int n = 0; n < FAB_MMA_BC/8; ++n) {
#pragma unroll
            for (int l = 0; l < fab_tile_C::ne; ++l) {
                S[l/2] += expf(SC[n].x[l] - M[l/2]);
            }
        }

        __syncthreads();
        t    = tn;
        buf ^= 1;
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        S[r] += __shfl_xor_sync(0xFFFFFFFF, S[r], 1, WARP_SIZE);
        S[r] += __shfl_xor_sync(0xFFFFFFFF, S[r], 2, WARP_SIZE);
        const int row = i0 + warp*16 + lane/4 + r*8;
        if (lane % 4 == 0 && row < n_q) {
            st_h[2*row] = S[r] == 0.0f ? -INFINITY : M[r] + logf(S[r]);
        }
    }

    // D = dot(grad_o, o) per row, the warp sweeps its 16 rows
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = i0 + warp*16 + r;
        if (row >= n_q) {
            break;
        }
        const float * orow = o_h + (int64_t) row*n_head*D;
        const float * grow = g_h + (int64_t) row*n_head*D;
        float d = 0.0f;
#pragma unroll
        for (int c = lane; c < D; c += WARP_SIZE) {
            d += orow[c]*grow[c];
        }
        d = warp_reduce_sum(d);
        if (lane == 0) {
            st_h[2*row + 1] = d;
        }
    }
#else
    GGML_UNUSED_VARS(bufs, o, grad_o, dst, args);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}


template <int D>
static __global__ void __launch_bounds__(FAB_MMA_NTHREADS, 1) fab_mma_dq(
        const fab_mma_bufs bufs,
        const float * __restrict__ stats,
              float * __restrict__ dst,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int stride = D/2 + 4;
    constexpr int MS     = FAB_MMA_BC + 8;
    extern __shared__ char smem_raw[];
    half2 * tiles = (half2 *) smem_raw;                        // [buf][K][V], buffer 0 also stages Q and dO
    half  * masks = (half *) (tiles + 2*2*FAB_MMA_BC*stride);  // [buf][BR][MS]

    const int n_q    = (int) args.n_q;
    const int n_head = (int) args.n_head;
    const int i0     = blockIdx.x*FAB_MMA_BR;
    const int h      = blockIdx.y;
    const int b      = blockIdx.z;
    const int hk     = h/(n_head/(int) args.n_head_kv);
    const int hb     = (b % (int) args.nem3)*(int) args.nem2 + h % (int) args.nem2;
    const int warp   = threadIdx.y;
    const int lane   = threadIdx.x;
    const int nfb    = bufs.kv_pad/FAB_MMA_FB;

    const half    * q_h  = bufs.q + ((size_t) b*n_head + h)*bufs.q_pad*D;
    const half    * g_h  = bufs.g + ((size_t) b*n_head + h)*bufs.q_pad*D;
    const half    * k_h  = bufs.k + ((size_t) b*args.n_head_kv + hk)*bufs.kv_pad*D;
    const half    * v_h  = bufs.v + ((size_t) b*args.n_head_kv + hk)*bufs.kv_pad*D;
    const half    * m_h  = fab_mma_mask_head(bufs, args, h, b);
    const uint8_t * f_h  = bufs.flags + ((size_t) hb*(bufs.q_pad/FAB_MMA_FB) + i0/FAB_MMA_FB)*nfb;
    const float   * st_h = stats + 2*((int64_t) h*n_q + (int64_t) b*n_q*n_head);
    float         * d_h  = dst   + ((int64_t) h*n_q + (int64_t) b*n_q*n_head)*D;

    // stationary Q and dO rows of the warp as A fragments
    fab_tile_A Q_A[D/16];
    fab_tile_A G_A[D/16];
    fab_mma_cp_rows<D, FAB_MMA_BR>(tiles, q_h + (size_t) i0*D);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();
    fab_mma_load_A<D>(Q_A, tiles + warp*16*stride);
    __syncthreads();
    fab_mma_cp_rows<D, FAB_MMA_BR>(tiles, g_h + (size_t) i0*D);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();
    fab_mma_load_A<D>(G_A, tiles + warp*16*stride);
    __syncthreads();

    // stats of the thread's two rows, D scaled like grad_o
    const int ge = fab_mma_gexp(bufs.gmax);
    float L[2], Dd[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = i0 + warp*16 + lane/4 + r*8;
        L[r]  = row < n_q ? st_h[2*row]                 : INFINITY;
        Dd[r] = row < n_q ? ldexpf(st_h[2*row + 1], ge) : 0.0f;
    }

    fab_tile_C dQ[D/8];

    const int n_tiles = nfb;
    auto valid = [&](const int t) -> bool {
        return f_h[t] | f_h[nfb + t];
    };
    auto issue = [&](const int t, const int buf) {
        half2 * tile_K = tiles + buf*2*FAB_MMA_BC*stride;
        fab_mma_cp_rows<D, FAB_MMA_BC>(tile_K,                        k_h + (size_t) t*FAB_MMA_BC*D);
        fab_mma_cp_rows<D, FAB_MMA_BC>(tile_K + FAB_MMA_BC*stride,    v_h + (size_t) t*FAB_MMA_BC*D);
        fab_mma_ld_mask<FAB_MMA_BR, FAB_MMA_BC>(masks + buf*FAB_MMA_BR*MS, m_h, args.nbm1, bufs.m_aligned,
            i0, n_q - i0, t*FAB_MMA_BC, (int) args.n_kv - t*FAB_MMA_BC);
        cp_async_commit_group();
    };

    int t = 0;
    while (t < n_tiles && !valid(t)) {
        ++t;
    }
    int buf = 0;
    if (t < n_tiles) {
        issue(t, buf);
    }
    while (t < n_tiles) {
        int tn = t + 1;
        while (tn < n_tiles && !valid(tn)) {
            ++tn;
        }
        if (tn < n_tiles) {
            issue(tn, buf ^ 1);
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const half2 * tile_K = tiles + buf*2*FAB_MMA_BC*stride;
        const half2 * tile_V = tile_K + FAB_MMA_BC*stride;
        const half  * tile_M = masks + buf*FAB_MMA_BR*MS;

        fab_tile_C SC[FAB_MMA_BC/8];
        fab_tile_C dP[FAB_MMA_BC/8];
        fab_mma_product<D, true>(SC, Q_A, nullptr, tile_K);
        fab_mma_product<D, true>(dP, G_A, nullptr, tile_V);

        // dS * 2^e = P * (dP - D) * 2^e, kept in SC; the softmax scale and 2^-e are applied at the store
#pragma unroll
        for (int n = 0; n < FAB_MMA_BC/8; ++n) {
#pragma unroll
            for (int l = 0; l < fab_tile_C::ne; ++l) {
                const int i = warp*16 + fab_tile_C::get_i(l);
                const int j = n*8 + fab_tile_C::get_j(l);
                const float s = SC[n].x[l]*args.scale + __half2float(tile_M[i*MS + j]);
                const float p = s == -INFINITY ? 0.0f : expf(s - L[l/2]);
                SC[n].x[l] = p*(dP[n].x[l] - Dd[l/2]);
            }
        }

        fab_tile_A dS_A[FAB_MMA_BC/16];
#pragma unroll
        for (int kk = 0; kk < FAB_MMA_BC/16; ++kk) {
            dS_A[kk] = fab_mma_c_to_a(SC[2*kk], SC[2*kk + 1]);
        }
        fab_mma_product_t<D>(dQ, dS_A, tile_K);

        __syncthreads();
        t    = tn;
        buf ^= 1;
    }

    fab_mma_store<D>(d_h + (int64_t) i0*D, dQ, n_q - i0, args.scale*ldexpf(1.0f, -ge));
#else
    GGML_UNUSED_VARS(bufs, stats, dst, args);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}


template <int D>
static __global__ void __launch_bounds__(FAB_MMA_NTHREADS, 1) fab_mma_dkv(
        const fab_mma_bufs bufs,
        const float * __restrict__ stats,
              float * __restrict__ out,
        const int64_t split_stride,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int stride = D/2 + 4;
    constexpr int MS     = FAB_MMA_BR + 8;                    // mask tile [BC q rows][BR kv], read transposed
    extern __shared__ char smem_raw[];
    half2 * tile_K = (half2 *) smem_raw;                       // stationary K and V rows of the block
    half2 * tile_V = tile_K + FAB_MMA_BR*stride;
    half2 * tiles  = tile_V + FAB_MMA_BR*stride;               // [buf][Q][dO]
    half  * masks  = (half *) (tiles + 2*2*FAB_MMA_BC*stride); // [buf][BC][MS]
    float * sh_LD  = (float *) (masks + 2*FAB_MMA_BC*MS);      // [buf][L[BC], D[BC]]

    const int n_q    = (int) args.n_q;
    const int n_kv   = (int) args.n_kv;
    const int n_head = (int) args.n_head;
    const int j0     = blockIdx.x*FAB_MMA_BR;
    const int hk     = blockIdx.y;
    const int b      = blockIdx.z % (int) args.ne3;
    const int split  = blockIdx.z / (int) args.ne3;
    const int nsplit = gridDim.z / (int) args.ne3;
    const int rk2    = n_head/(int) args.n_head_kv;
    const int warp   = threadIdx.y;
    const int tid    = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int nfb    = bufs.kv_pad/FAB_MMA_FB;
    const int kb0    = j0/FAB_MMA_FB;

    const half * k_h = bufs.k + ((size_t) b*args.n_head_kv + hk)*bufs.kv_pad*D;
    const half * v_h = bufs.v + ((size_t) b*args.n_head_kv + hk)*bufs.kv_pad*D;

    fab_mma_cp_rows<D, FAB_MMA_BR>(tile_K, k_h + (size_t) j0*D);
    fab_mma_cp_rows<D, FAB_MMA_BR>(tile_V, v_h + (size_t) j0*D);
    cp_async_commit_group();

    fab_tile_C dK[D/8];
    fab_tile_C dV[D/8];

    // the q tiles of every head of the group, one flat sequence
    const int n_tq    = bufs.q_pad/FAB_MMA_BC;
    const int u_begin = (int) ((int64_t) split      *rk2*n_tq/nsplit);
    const int n_tiles = (int) ((int64_t) (split + 1)*rk2*n_tq/nsplit);
    const int ge      = fab_mma_gexp(bufs.gmax);
    auto head_of = [&](const int u) -> int { return hk*rk2 + u/n_tq; };
    auto hb_of   = [&](const int h) -> int { return (b % (int) args.nem3)*(int) args.nem2 + h % (int) args.nem2; };
    auto valid = [&](const int u) -> bool {
        const int t = u % n_tq;
        const uint8_t * f = bufs.flags + ((size_t) hb_of(head_of(u))*(bufs.q_pad/FAB_MMA_FB) + t)*nfb + kb0;
        return f[0] | f[1];
    };
    auto issue = [&](const int u, const int buf) {
        const int h = head_of(u);
        const int t = u % n_tq;
        const half  * q_h  = bufs.q + ((size_t) b*n_head + h)*bufs.q_pad*D;
        const half  * g_h  = bufs.g + ((size_t) b*n_head + h)*bufs.q_pad*D;
        const half  * m_h  = fab_mma_mask_head(bufs, args, h, b);
        const float * st_h = stats + 2*((int64_t) h*n_q + (int64_t) b*n_q*n_head);
        half2 * tile_Q = tiles + buf*2*FAB_MMA_BC*stride;
        fab_mma_cp_rows<D, FAB_MMA_BC>(tile_Q,                     q_h + (size_t) t*FAB_MMA_BC*D);
        fab_mma_cp_rows<D, FAB_MMA_BC>(tile_Q + FAB_MMA_BC*stride, g_h + (size_t) t*FAB_MMA_BC*D);
        fab_mma_ld_mask<FAB_MMA_BC, FAB_MMA_BR>(masks + buf*FAB_MMA_BC*MS, m_h, args.nbm1, bufs.m_aligned,
            t*FAB_MMA_BC, n_q - t*FAB_MMA_BC, j0, n_kv - j0);
        cp_async_commit_group();
        if (tid < FAB_MMA_BC) {
            const int col = t*FAB_MMA_BC + tid;
            sh_LD[buf*2*FAB_MMA_BC + tid]              = col < n_q ? st_h[2*col]     : INFINITY;
            sh_LD[buf*2*FAB_MMA_BC + FAB_MMA_BC + tid] = col < n_q ? ldexpf(st_h[2*col + 1], ge) : 0.0f;
        }
    };

    int u = u_begin;
    while (u < n_tiles && !valid(u)) {
        ++u;
    }
    int buf = 0;
    if (u < n_tiles) {
        issue(u, buf);
    }
    while (u < n_tiles) {
        int un = u + 1;
        while (un < n_tiles && !valid(un)) {
            ++un;
        }
        if (un < n_tiles) {
            issue(un, buf ^ 1);
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const half2 * tile_Q = tiles + buf*2*FAB_MMA_BC*stride;
        const half2 * tile_G = tile_Q + FAB_MMA_BC*stride;
        const half  * tile_M = masks + buf*FAB_MMA_BC*MS;
        const float * sh_L   = sh_LD + buf*2*FAB_MMA_BC;
        const float * sh_D   = sh_L + FAB_MMA_BC;

        fab_tile_C SC[FAB_MMA_BC/8]; // S^T, rows kv, columns q
        fab_tile_C dP[FAB_MMA_BC/8];
        fab_mma_product<D, false>(SC, nullptr, tile_K + warp*16*stride, tile_Q);
        fab_mma_product<D, false>(dP, nullptr, tile_V + warp*16*stride, tile_G);

        // P in SC, dS * 2^e = P * (dP - D) * 2^e in dP; the softmax scale and 2^-e are applied at the store
#pragma unroll
        for (int n = 0; n < FAB_MMA_BC/8; ++n) {
#pragma unroll
            for (int l = 0; l < fab_tile_C::ne; ++l) {
                const int i = warp*16 + fab_tile_C::get_i(l);
                const int j = n*8 + fab_tile_C::get_j(l);
                const float s = SC[n].x[l]*args.scale + __half2float(tile_M[j*MS + i]);
                const float p = s == -INFINITY ? 0.0f : expf(s - sh_L[j]);
                SC[n].x[l] = p;
                dP[n].x[l] = p*(dP[n].x[l] - sh_D[j]);
            }
        }

        fab_tile_A P_A[FAB_MMA_BC/16];
        fab_tile_A dS_A[FAB_MMA_BC/16];
#pragma unroll
        for (int kk = 0; kk < FAB_MMA_BC/16; ++kk) {
            P_A[kk]  = fab_mma_c_to_a(SC[2*kk], SC[2*kk + 1]);
            dS_A[kk] = fab_mma_c_to_a(dP[2*kk], dP[2*kk + 1]);
        }
        fab_mma_product_t<D>(dV, P_A,  tile_G);
        fab_mma_product_t<D>(dK, dS_A, tile_Q);

        __syncthreads();
        u    = un;
        buf ^= 1;
    }

    cp_async_wait_group<0>();

    float * dk_h = out  + split*split_stride + ((int64_t) hk*n_kv + (int64_t) b*n_kv*args.n_head_kv)*D;
    float * dv_h = dk_h + (int64_t) n_kv*args.n_head_kv*args.ne3*D;
    const float f = ldexpf(1.0f, -ge);
    fab_mma_store<D>(dk_h + (int64_t) j0*D, dK, n_kv - j0, args.scale*f);
    fab_mma_store<D>(dv_h + (int64_t) j0*D, dV, n_kv - j0, f);
#else
    GGML_UNUSED_VARS(bufs, stats, out, split_stride, args);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __global__ void fab_mma_sum_splits(const float * __restrict__ part, float * __restrict__ dst, const int64_t n, const int nsplit) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float sum = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        sum += part[s*n + i];
    }
    dst[i] = sum;
}

template <int D>
static constexpr size_t fab_mma_smem(const int target) {
    constexpr size_t stride = D/2 + 4;
    switch (target) {
        case GGML_FLASH_ATTN_EXT_BACK_STATS:
            return 2*FAB_MMA_BC*stride*sizeof(half2) + 2*FAB_MMA_BR*(FAB_MMA_BC + 8)*sizeof(half);
        case GGML_FLASH_ATTN_EXT_BACK_DQ:
            return 2*2*FAB_MMA_BC*stride*sizeof(half2) + 2*FAB_MMA_BR*(FAB_MMA_BC + 8)*sizeof(half);
        default:
            return (2*FAB_MMA_BR + 2*2*FAB_MMA_BC)*stride*sizeof(half2) + 2*FAB_MMA_BC*(FAB_MMA_BR + 8)*sizeof(half) +
                   2*2*FAB_MMA_BC*sizeof(float);
    }
}

template <typename T>
static void fab_mma_cvt_rows_launch(
        const T * src, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3, half * dst,
        const int n_rows, const int rows_pad, const int n_heads, const int ne3, const int D, cudaStream_t stream,
        const int * gmax = nullptr) {
    const int chunks = rows_pad*(D/8);
    const dim3 block_nums((chunks + 255)/256, n_heads, ne3);
    fab_mma_cvt_rows<T><<<block_nums, 256, 0, stream>>>(src, nb1, nb2, nb3, dst, n_rows, rows_pad, D, gmax);
}

template <int D>
static void fab_mma_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_cuda_flash_attn_ext_back_kargs & args) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * k      = dst->src[1];
    const ggml_tensor * v      = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];
    const ggml_tensor * o      = dst->src[4];
    const ggml_tensor * grad_o = dst->src[5];
    const ggml_tensor * stats  = dst->src[6];

    const int n_q       = (int) args.n_q;
    const int n_kv      = (int) args.n_kv;
    const int n_head    = (int) args.n_head;
    const int n_head_kv = (int) args.n_head_kv;
    const int ne3       = (int) args.ne3;
    const int nem2      = mask ? (int) args.nem2 : 1;
    const int nem3      = mask ? (int) args.nem3 : 1;
    const int q_pad     = ((n_q  + FAB_MMA_BR - 1)/FAB_MMA_BR)*FAB_MMA_BR;
    const int kv_pad    = ((n_kv + FAB_MMA_BR - 1)/FAB_MMA_BR)*FAB_MMA_BR;

    const bool need_g = args.target != GGML_FLASH_ATTN_EXT_BACK_STATS;
    const bool need_v = args.target != GGML_FLASH_ATTN_EXT_BACK_STATS;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half>    q16(pool, (size_t) ne3*n_head*q_pad*D);
    ggml_cuda_pool_alloc<half>    g16(pool, need_g ? (size_t) ne3*n_head*q_pad*D : 1);
    ggml_cuda_pool_alloc<half>    k16(pool, (size_t) ne3*n_head_kv*kv_pad*D);
    ggml_cuda_pool_alloc<half>    v16(pool, need_v ? (size_t) ne3*n_head_kv*kv_pad*D : 1);
    ggml_cuda_pool_alloc<uint8_t> flags(pool, (size_t) nem3*nem2*(q_pad/FAB_MMA_FB)*(kv_pad/FAB_MMA_FB));
    ggml_cuda_pool_alloc<int>     gmax(pool, 1);

    fab_mma_cvt_rows_launch<float>((const float *) q->data, args.nbq1, args.nbq2, args.nbq3, q16.get(), n_q, q_pad, n_head, ne3, D, stream);
    if (need_g) {
        const int64_t n_g = (int64_t) ne3*n_head*n_q*D;
        CUDA_CHECK(cudaMemsetAsync(gmax.get(), 0, sizeof(int), stream));
        fab_mma_absmax<<<(unsigned) std::min<int64_t>((n_g + 255)/256, 1024), 256, 0, stream>>>((const float *) grad_o->data, n_g, gmax.get());
        // grad_o [D, n_head, n_q, ne3]: row stride n_head*D, head stride D, batch stride n_head*n_q*D elements
        fab_mma_cvt_rows_launch<float>((const float *) grad_o->data,
            (uint64_t) n_head*D*sizeof(float), (uint64_t) D*sizeof(float), (uint64_t) n_head*n_q*D*sizeof(float),
            g16.get(), n_q, q_pad, n_head, ne3, D, stream, gmax.get());
    }
    if (k->type == GGML_TYPE_F16) {
        fab_mma_cvt_rows_launch<half>((const half *) k->data, args.nbk1, args.nbk2, args.nbk3, k16.get(), n_kv, kv_pad, n_head_kv, ne3, D, stream);
        if (need_v) {
            fab_mma_cvt_rows_launch<half>((const half *) v->data, args.nbv1, args.nbv2, args.nbv3, v16.get(), n_kv, kv_pad, n_head_kv, ne3, D, stream);
        }
    } else {
        fab_mma_cvt_rows_launch<float>((const float *) k->data, args.nbk1, args.nbk2, args.nbk3, k16.get(), n_kv, kv_pad, n_head_kv, ne3, D, stream);
        if (need_v) {
            fab_mma_cvt_rows_launch<float>((const float *) v->data, args.nbv1, args.nbv2, args.nbv3, v16.get(), n_kv, kv_pad, n_head_kv, ne3, D, stream);
        }
    }
    {
        const dim3 block_nums(kv_pad/FAB_MMA_FB, q_pad/FAB_MMA_FB, nem2*nem3);
        fab_mma_mask_flags<<<block_nums, FAB_MMA_NTHREADS, 0, stream>>>(
            mask ? (const half *) mask->data : nullptr, args.nbm1, args.nbm2, args.nbm3, nem2,
            flags.get(), n_q, n_kv, q_pad, kv_pad);
    }
    CUDA_CHECK(cudaGetLastError());

    fab_mma_bufs bufs;
    bufs.q      = q16.get();
    bufs.g      = g16.get();
    bufs.k      = k16.get();
    bufs.v      = v16.get();
    bufs.m      = mask ? (const half *) mask->data : nullptr;
    bufs.flags  = flags.get();
    bufs.gmax   = gmax.get();
    bufs.m_aligned = mask && (uintptr_t) mask->data % 16 == 0 && args.nbm1 % 16 == 0 && args.nbm2 % 16 == 0 && args.nbm3 % 16 == 0;
    bufs.q_pad  = q_pad;
    bufs.kv_pad = kv_pad;

    const float * o_d     = (const float *) o->data;
    const float * grad_o_d = (const float *) grad_o->data;
    const float * stats_d = stats ? (const float *) stats->data : nullptr;
    float       * dst_d   = (float *) dst->data;

    const dim3 block_dims(WARP_SIZE, FAB_MMA_NWARPS, 1);
    const size_t smem = fab_mma_smem<D>(args.target);

    switch (args.target) {
        case GGML_FLASH_ATTN_EXT_BACK_STATS: {
            CUDA_SET_SHARED_MEMORY_LIMIT(fab_mma_stats<D>, (int) smem);
            const dim3 block_nums((unsigned) (q_pad/FAB_MMA_BR), (unsigned) n_head, (unsigned) ne3);
            const ggml_cuda_kernel_launch_params lp = {block_nums, block_dims, smem, stream};
            ggml_cuda_kernel_launch(fab_mma_stats<D>, lp, bufs, o_d, grad_o_d, dst_d, args);
        } break;
        case GGML_FLASH_ATTN_EXT_BACK_DQ: {
            CUDA_SET_SHARED_MEMORY_LIMIT(fab_mma_dq<D>, (int) smem);
            const dim3 block_nums((unsigned) (q_pad/FAB_MMA_BR), (unsigned) n_head, (unsigned) ne3);
            const ggml_cuda_kernel_launch_params lp = {block_nums, block_dims, smem, stream};
            ggml_cuda_kernel_launch(fab_mma_dq<D>, lp, bufs, stats_d, dst_d, args);
        } break;
        case GGML_FLASH_ATTN_EXT_BACK_DKV: {
            CUDA_SET_SHARED_MEMORY_LIMIT(fab_mma_dkv<D>, (int) smem);
            static int blocks_per_sm[GGML_CUDA_MAX_DEVICES] = { 0 };
            if (blocks_per_sm[ctx.device] == 0) {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm[ctx.device], fab_mma_dkv<D>, FAB_MMA_NTHREADS, smem));
            }
            const int wave    = ggml_cuda_info().devices[ctx.device].nsm*std::max(1, blocks_per_sm[ctx.device]);
            const int n_kvb   = kv_pad/FAB_MMA_BR;
            const int n_units = (n_head/n_head_kv)*(q_pad/FAB_MMA_BC);
            const int base    = n_kvb*n_head_kv*ne3;
            const int nsplit  = std::max(1, std::min({n_units, 16, wave/base}));

            const int64_t n_out = 2*(int64_t) n_kv*n_head_kv*ne3*D;
            ggml_cuda_pool_alloc<float> part(pool, nsplit > 1 ? (size_t) nsplit*n_out : 1);

            const dim3 block_nums((unsigned) n_kvb, (unsigned) n_head_kv, (unsigned) (ne3*nsplit));
            const ggml_cuda_kernel_launch_params lp = {block_nums, block_dims, smem, stream};
            ggml_cuda_kernel_launch(fab_mma_dkv<D>, lp, bufs, stats_d, nsplit > 1 ? part.get() : dst_d, n_out, args);
            if (nsplit > 1) {
                fab_mma_sum_splits<<<(unsigned) ((n_out + 255)/256), 256, 0, stream>>>(part.get(), dst_d, n_out, nsplit);
                CUDA_CHECK(cudaGetLastError());
            }
        } break;
        default:
            GGML_ABORT("flash_attn_ext_back: invalid target");
    }
}

static bool ggml_cuda_flash_attn_ext_back_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_flash_attn_ext_back_kargs & args) {
    static const bool disabled = getenv("GGML_CUDA_FA_BACK_NO_MMA") != nullptr;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (disabled || !ampere_mma_available(cc) || !cp_async_available(cc)) {
        return false;
    }
    switch (args.D) {
        case  32: fab_mma_launch< 32>(ctx, dst, args); return true;
        case  64: fab_mma_launch< 64>(ctx, dst, args); return true;
        case  80: fab_mma_launch< 80>(ctx, dst, args); return true;
        case  96: fab_mma_launch< 96>(ctx, dst, args); return true;
        case 112: fab_mma_launch<112>(ctx, dst, args); return true;
        case 128: fab_mma_launch<128>(ctx, dst, args); return true;
        default:  return false;
    }
}

#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
