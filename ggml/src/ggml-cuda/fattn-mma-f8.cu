#include "common.cuh"
#include "cp-async.cuh"
#include "fattn-mma-f8.cuh"

#include <cstdlib>

#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
#define FATTN_F8_AVAILABLE
#endif

// Tile sizes: BQ = 16*NW query rows per block (one Q head), BK keys per iteration.
static constexpr int FATTN_F8_NW = 8;
static constexpr int FATTN_F8_BQ = 16*FATTN_F8_NW;
static constexpr int FATTN_F8_BK = 64;

// P is in [0, 1]; scale it up so that small probabilities stay above the e4m3 subnormal range.
static constexpr float FATTN_F8_P_SCALE = 448.0f;
static constexpr float FATTN_F8_LOG2E   = 1.4426950408889634f;

#ifdef FATTN_F8_AVAILABLE
static __device__ __forceinline__ uint32_t fattn_f8_pack4(const float a, const float b, const float c, const float d) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

static __device__ __forceinline__ void fattn_f8_mma(float * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void fattn_f8_ldsm_x4(uint32_t * r, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ggml_cuda_cvta_generic_to_shared((void *) p)));
}
#endif // FATTN_F8_AVAILABLE

// Key order of the transposed V tiles: in each block of 32 keys, position p holds the key that the KQ accumulator
// layout puts in byte p of the P fragment, so P can go from the KQ accumulators into the VKQ MMA without shuffles.
static __host__ __device__ __forceinline__ int fattn_f8_v_key(const int p) {
    return (p/16)*16 + ((p%4)/2)*8 + ((p%16)/4)*2 + p%2;
}

// K (F16) -> e4m3, contiguous [stream][head][key][D].
static __global__ void fattn_f8_convert_k(
        const char * __restrict__ K, uint8_t * __restrict__ K8, const int D, const int n_kv, const int n_kvh,
        const int64_t nb11, const int64_t nb12, const int64_t nb13, const int64_t n_items) {
#ifdef FATTN_F8_AVAILABLE
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_items) {
        return;
    }
    const int     d8   = i % (D/8);
    const int64_t row  = i / (D/8);
    const int     key  = row % n_kv;
    const int     hk   = (row / n_kv) % n_kvh;
    const int     s    = row / ((int64_t) n_kv*n_kvh);

    const uint4 raw = *(const uint4 *) (K + s*nb13 + hk*nb12 + key*nb11 + d8*16);
    const half2 * h2 = (const half2 *) &raw;
    float2 f[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        f[j] = __half22float2(h2[j]);
    }
    uint2 out;
    out.x = fattn_f8_pack4(f[0].x, f[0].y, f[1].x, f[1].y);
    out.y = fattn_f8_pack4(f[2].x, f[2].y, f[3].x, f[3].y);
    *(uint2 *) (K8 + row*D + d8*8) = out;
#else
    GGML_UNUSED_VARS(K, K8, D, n_kv, n_kvh, nb11, nb12, nb13, n_items);
    NO_DEVICE_CODE;
#endif // FATTN_F8_AVAILABLE
}

// V (F16 or e4m3) -> e4m3, transposed [stream][head][D][key] with the key order of fattn_f8_v_key per 32 keys.
template <int D>
static __global__ void fattn_f8_transpose_v(
        const char * __restrict__ V, uint8_t * __restrict__ Vt, const bool v_f8, const int n_kv, const int n_kvh,
        const int64_t nb21, const int64_t nb22, const int64_t nb23) {
#ifdef FATTN_F8_AVAILABLE
    __shared__ __align__(16) uint8_t tile[32][D + 16];

    const int blk = blockIdx.x;
    const int hk  = blockIdx.y;
    const int s   = blockIdx.z;
    const char * Vh = V + s*nb23 + hk*nb22;

    for (int i = threadIdx.x; i < 32*(D/8); i += blockDim.x) {
        const int key = i / (D/8);
        const int d8  = i % (D/8);
        const char * src = Vh + (int64_t) (blk*32 + key)*nb21;
        uint2 out;
        if (v_f8) {
            out = *(const uint2 *) (src + d8*8);
        } else {
            const uint4 raw = *(const uint4 *) (src + d8*16);
            const half2 * h2 = (const half2 *) &raw;
            const float2 f0 = __half22float2(h2[0]);
            const float2 f1 = __half22float2(h2[1]);
            const float2 f2 = __half22float2(h2[2]);
            const float2 f3 = __half22float2(h2[3]);
            out.x = fattn_f8_pack4(f0.x, f0.y, f1.x, f1.y);
            out.y = fattn_f8_pack4(f2.x, f2.y, f3.x, f3.y);
        }
        *(uint2 *) &tile[key][d8*8] = out;
    }
    __syncthreads();

    uint8_t * dst = Vt + ((int64_t) (s*n_kvh + hk)*D)*n_kv + blk*32;
    for (int i = threadIdx.x; i < D*8; i += blockDim.x) {
        const int d  = i / 8;
        const int p0 = (i % 8)*4;
        uint32_t v = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v |= (uint32_t) tile[fattn_f8_v_key(p0 + k)][d] << (8*k);
        }
        *(uint32_t *) (dst + (int64_t) d*n_kv + p0) = v;
    }
#else
    GGML_UNUSED_VARS(V, Vt, v_f8, n_kv, n_kvh, nb21, nb22, nb23);
    NO_DEVICE_CODE;
#endif // FATTN_F8_AVAILABLE
}

// Per query tile, the end of the last key tile that is not fully masked.
static __global__ void fattn_f8_kv_max(
        const half * __restrict__ mask, int * __restrict__ KV_max, const int n_kv, const int ne31, const int64_t sm1, const int64_t sm3) {
    const int jt = blockIdx.x;
    const int s  = blockIdx.y;
    const half * M = mask + s*sm3;

    int kv = n_kv;
    for (; kv > 0; kv -= FATTN_F8_BK) {
        int any = 0;
        for (int i = threadIdx.x; i < FATTN_F8_BQ*(FATTN_F8_BK/2); i += blockDim.x) {
            const int r = i / (FATTN_F8_BK/2);
            const int c = i % (FATTN_F8_BK/2);
            const int q = min(jt*FATTN_F8_BQ + r, ne31 - 1);
            const float2 v = __half22float2(((const half2 *) (M + q*sm1 + kv - FATTN_F8_BK))[c]);
            any |= v.x != -INFINITY || v.y != -INFINITY;
        }
        if (__syncthreads_or(any)) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        KV_max[s*gridDim.x + jt] = kv;
    }
}

template <int D>
__launch_bounds__(FATTN_F8_NW*WARP_SIZE, 1)
static __global__ void flash_attn_ext_f8(
        const char * __restrict__ Q, const uint8_t * __restrict__ K, const uint8_t * __restrict__ Vt,
        const half * __restrict__ mask, const int * __restrict__ KV_max, float * __restrict__ dst,
        const float scale_log2, const int n_q, const int n_head, const int gqa_ratio,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int ne31, const int64_t sm1, const int64_t sm3) {
#ifdef FATTN_F8_AVAILABLE
    constexpr int NW = FATTN_F8_NW;
    constexpr int BQ = FATTN_F8_BQ;
    constexpr int BK = FATTN_F8_BK;
    constexpr int KS = D/32; // KQ k steps
    constexpr int NT = D/8;  // VKQ n tiles
    static_assert(BQ <= 2*BK, "Q staging must fit into the K buffers");

    extern __shared__ __align__(16) uint8_t smem_f8[];
    uint8_t * sK  = smem_f8;                        // [2][BK][D], 16 byte chunks swizzled by row
    uint8_t * sV  = sK + 2*BK*D;                    // [2][D][BK], transposed V
    half    * sM  = (half *) (sV + 2*D*BK);         // [2][BQ][BK]
    float   * sQs = (float *) (sM + 2*BQ*BK);       // [BQ]
    uint8_t * sQ  = sK;                             // [BQ][D], only before the first K tile

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int g    = lane / 4;
    const int tig  = lane % 4;
    const int q0   = blockIdx.x*BQ;
    const int h    = blockIdx.y;
    const int s    = blockIdx.z;

    const char    * Qh = Q + s*nb03 + h*nb02;
    const uint8_t * Kh = K + s*sk3 + (h/gqa_ratio)*sk2;
    const uint8_t * Vh = Vt + s*sv3 + (h/gqa_ratio)*sv2;
    const half    * Mh = mask + s*sm3;

    // Q: one row per warp iteration, per-row scale amax/448.
    constexpr int QPL = D/WARP_SIZE; // values per lane
    for (int r = warp*16; r < warp*16 + 16; ++r) {
        const int q = q0 + r;
        float x[QPL];
        if (q < n_q) {
            const float4 * qr = (const float4 *) (Qh + q*nb01) + lane*(QPL/4);
#pragma unroll
            for (int j = 0; j < QPL/4; ++j) {
                const float4 t = qr[j];
                x[4*j + 0] = t.x; x[4*j + 1] = t.y; x[4*j + 2] = t.z; x[4*j + 3] = t.w;
            }
        } else {
#pragma unroll
            for (int j = 0; j < QPL; ++j) {
                x[j] = 0.0f;
            }
        }
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < QPL; ++j) {
            amax = fmaxf(amax, fabsf(x[j]));
        }
        amax = warp_reduce_max(amax);
        const float qs  = amax > 0.0f ? amax/448.0f : 1.0f;
        const float iqs = 1.0f/qs;

        uint32_t packed[QPL/4];
#pragma unroll
        for (int j = 0; j < QPL/4; ++j) {
            packed[j] = fattn_f8_pack4(x[4*j + 0]*iqs, x[4*j + 1]*iqs, x[4*j + 2]*iqs, x[4*j + 3]*iqs);
        }
        const int byte  = lane*QPL;
        const int chunk = byte / 16;
        uint8_t * dq = sQ + r*D + ((chunk ^ (r & 7))*16) + byte % 16;
        if constexpr (QPL == 8) {
            *(uint2 *) dq = make_uint2(packed[0], packed[1]);
        } else {
            static_assert(QPL == 4, "unsupported head size");
            *(uint32_t *) dq = packed[0];
        }
        if (lane == 0) {
            sQs[r] = qs;
        }
    }
    __syncthreads();

    uint32_t qf[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        const int m     = lane / 8;
        const int row   = warp*16 + (m % 2)*8 + lane % 8;
        const int chunk = 2*ks + m/2;
        fattn_f8_ldsm_x4(qf[ks], sQ + row*D + ((chunk ^ (row & 7))*16));
    }
    const int   rA  = warp*16 + g;
    const int   rB  = rA + 8;
    const float qsA = sQs[rA]*scale_log2;
    const float qsB = sQs[rB]*scale_log2;
    __syncthreads(); // sQ aliases sK

    const int kv_end = KV_max[s*gridDim.x + blockIdx.x];

    auto load_tile = [&](const int kt, const int buf) {
        uint8_t * dK = sK + buf*BK*D;
        for (int i = tid; i < BK*(D/16); i += NW*WARP_SIZE) {
            const int row = i / (D/16);
            const int c   = i % (D/16);
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dK + row*D + ((c ^ (row & 7))*16)), Kh + (kt + row)*sk1 + c*16);
        }
        uint8_t * dV = sV + buf*D*BK;
        for (int i = tid; i < D*(BK/16); i += NW*WARP_SIZE) {
            const int d = i / (BK/16);
            const int c = i % (BK/16);
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dV + d*BK + ((c ^ ((d/2) % 4))*16)), Vh + d*sv1 + kt + c*16);
        }
        half * dM = sM + buf*BQ*BK;
        for (int i = tid; i < BQ*(BK/8); i += NW*WARP_SIZE) {
            const int r = i / (BK/8);
            const int c = i % (BK/8);
            const int q = min(q0 + r, ne31 - 1);
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dM + r*BK + ((c ^ (r & 7))*8)), Mh + q*sm1 + kt + c*8);
        }
        cp_async_commit_group();
    };

    float oacc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            oacc[nt][l] = 0.0f;
        }
    }
    float mA = -INFINITY;
    float mB = -INFINITY;
    float lA = 0.0f;
    float lB = 0.0f;

    if (kv_end > 0) {
        load_tile(0, 0);
    }
    int buf = 0;
    for (int kt = 0; kt < kv_end; kt += BK, buf ^= 1) {
        if (kt + BK < kv_end) {
            load_tile(kt + BK, buf ^ 1);
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // S = Q K^T
        const uint8_t * cK = sK + buf*BK*D;
        float sacc[BK/8][4];
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                sacc[j][l] = 0.0f;
            }
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
            for (int np = 0; np < BK/16; ++np) {
                const int m     = lane / 8;
                const int key   = np*16 + (m/2)*8 + lane % 8;
                const int chunk = 2*ks + m % 2;
                uint32_t b[4];
                fattn_f8_ldsm_x4(b, cK + key*D + ((chunk ^ (key & 7))*16));
                fattn_f8_mma(sacc[2*np + 0], qf[ks], b[0], b[1]);
                fattn_f8_mma(sacc[2*np + 1], qf[ks], b[2], b[3]);
            }
        }

        // online softmax, base 2
        const half * cM = sM + buf*BQ*BK;
        float mxA = -INFINITY;
        float mxB = -INFINITY;
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
            const float2 fA = __half22float2(*(const half2 *) (cM + rA*BK + ((j ^ (rA & 7))*8) + 2*tig));
            const float2 fB = __half22float2(*(const half2 *) (cM + rB*BK + ((j ^ (rB & 7))*8) + 2*tig));
            sacc[j][0] = sacc[j][0]*qsA + fA.x*FATTN_F8_LOG2E;
            sacc[j][1] = sacc[j][1]*qsA + fA.y*FATTN_F8_LOG2E;
            sacc[j][2] = sacc[j][2]*qsB + fB.x*FATTN_F8_LOG2E;
            sacc[j][3] = sacc[j][3]*qsB + fB.y*FATTN_F8_LOG2E;
            mxA = fmaxf(mxA, fmaxf(sacc[j][0], sacc[j][1]));
            mxB = fmaxf(mxB, fmaxf(sacc[j][2], sacc[j][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off *= 2) {
            mxA = fmaxf(mxA, __shfl_xor_sync(0xFFFFFFFF, mxA, off));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xFFFFFFFF, mxB, off));
        }
        const float mnA  = fmaxf(mA, mxA);
        const float mnB  = fmaxf(mB, mxB);
        const float refA = mnA == -INFINITY ? 0.0f : mnA;
        const float refB = mnB == -INFINITY ? 0.0f : mnB;
        const float cA   = exp2f(mA - refA);
        const float cB   = exp2f(mB - refB);
        mA = mnA;
        mB = mnB;
        lA *= cA;
        lB *= cB;
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
            sacc[j][0] = exp2f(sacc[j][0] - refA);
            sacc[j][1] = exp2f(sacc[j][1] - refA);
            sacc[j][2] = exp2f(sacc[j][2] - refB);
            sacc[j][3] = exp2f(sacc[j][3] - refB);
            lA += sacc[j][0] + sacc[j][1];
            lB += sacc[j][2] + sacc[j][3];
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            oacc[nt][0] *= cA;
            oacc[nt][1] *= cA;
            oacc[nt][2] *= cB;
            oacc[nt][3] *= cB;
        }

        // P -> e4m3 A fragments, 32 keys per k step
        uint32_t pa[BK/32][4];
#pragma unroll
        for (int k2 = 0; k2 < BK/32; ++k2) {
            const float ps = FATTN_F8_P_SCALE;
            pa[k2][0] = fattn_f8_pack4(sacc[4*k2 + 0][0]*ps, sacc[4*k2 + 0][1]*ps, sacc[4*k2 + 1][0]*ps, sacc[4*k2 + 1][1]*ps);
            pa[k2][1] = fattn_f8_pack4(sacc[4*k2 + 0][2]*ps, sacc[4*k2 + 0][3]*ps, sacc[4*k2 + 1][2]*ps, sacc[4*k2 + 1][3]*ps);
            pa[k2][2] = fattn_f8_pack4(sacc[4*k2 + 2][0]*ps, sacc[4*k2 + 2][1]*ps, sacc[4*k2 + 3][0]*ps, sacc[4*k2 + 3][1]*ps);
            pa[k2][3] = fattn_f8_pack4(sacc[4*k2 + 2][2]*ps, sacc[4*k2 + 2][3]*ps, sacc[4*k2 + 3][2]*ps, sacc[4*k2 + 3][3]*ps);
        }

        // O += P V
        const uint8_t * cV = sV + buf*D*BK;
#pragma unroll
        for (int k2 = 0; k2 < BK/32; ++k2) {
#pragma unroll
            for (int dp = 0; dp < D/16; ++dp) {
                const int m     = lane / 8;
                const int d     = dp*16 + (m/2)*8 + lane % 8;
                const int chunk = 2*k2 + m % 2;
                uint32_t b[4];
                fattn_f8_ldsm_x4(b, cV + d*BK + ((chunk ^ ((d/2) % 4))*16));
                fattn_f8_mma(oacc[2*dp + 0], pa[k2], b[0], b[1]);
                fattn_f8_mma(oacc[2*dp + 1], pa[k2], b[2], b[3]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 1; off < 4; off *= 2) {
        lA += __shfl_xor_sync(0xFFFFFFFF, lA, off);
        lB += __shfl_xor_sync(0xFFFFFFFF, lB, off);
    }
    const float iA = lA > 0.0f ? 1.0f/(lA*FATTN_F8_P_SCALE) : 0.0f;
    const float iB = lB > 0.0f ? 1.0f/(lB*FATTN_F8_P_SCALE) : 0.0f;
    const int qA = q0 + rA;
    const int qB = q0 + rB;
    if (qA < n_q) {
        float * o = dst + ((int64_t) (s*n_q + qA)*n_head + h)*D;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            *(float2 *) (o + nt*8 + 2*tig) = make_float2(oacc[nt][0]*iA, oacc[nt][1]*iA);
        }
    }
    if (qB < n_q) {
        float * o = dst + ((int64_t) (s*n_q + qB)*n_head + h)*D;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            *(float2 *) (o + nt*8 + 2*tig) = make_float2(oacc[nt][2]*iB, oacc[nt][3]*iB);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, Vt, mask, KV_max, dst, scale_log2, n_q, n_head, gqa_ratio, nb01, nb02, nb03,
        sk1, sk2, sk3, sv1, sv2, sv3, ne31, sm1, sm3);
    NO_DEVICE_CODE;
#endif // FATTN_F8_AVAILABLE
}

static int fattn_f8_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_FA_F8");
        return env ? atoi(env) : 0;
    }();
    return mode;
}

bool ggml_cuda_flash_attn_ext_f8_supported(int device, const ggml_tensor * dst) {
    if (fattn_f8_mode() == 0) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!fp8_mma_hardware_available(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t D = Q->ne[0];
    const bool kv_type_ok = (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_F8_E4M3) &&
                            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_F8_E4M3);
    return (D == 128 || D == 256) && K->ne[0] == D && V->ne[0] == D && kv_type_ok &&
        Q->type == GGML_TYPE_F32 && Q->nb[0] == sizeof(float) && ggml_cuda_is_aligned(Q, 16) &&
        ggml_cuda_is_aligned(K, 16) && ggml_cuda_is_aligned(V, 16) &&
        Q->ne[1] >= 64 && Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        (K->ne[3] == Q->ne[3] || K->ne[3] == 1) && V->ne[3] == K->ne[3] &&
        K->ne[1] % FATTN_F8_BK == 0 && V->ne[1] == K->ne[1] &&
        K->nb[0] == ggml_type_size(K->type) && V->nb[0] == ggml_type_size(V->type) &&
        K->nb[1] % 16 == 0 && V->nb[1] % 16 == 0 &&
        mask != nullptr && mask->type == GGML_TYPE_F16 && ggml_cuda_is_aligned(mask, 16) && mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] &&
        mask->ne[2] == 1 && (mask->ne[3] == Q->ne[3] || mask->ne[3] == 1) && mask->nb[1] % 16 == 0 &&
        sinks == nullptr && max_bias == 0.0f && logit_softcap == 0.0f && ggml_get_op_params_i32(dst, 4) == 0;
}

template <int D>
static void ggml_cuda_flash_attn_ext_f8_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();

    const int n_kv  = K->ne[1];
    const int n_kvh = K->ne[2];
    const int n_skv = K->ne[3];
    const int n_q   = Q->ne[1];
    const int n_s   = Q->ne[3];

    ggml_cuda_pool_alloc<uint8_t> K8_alloc(ctx.pool());
    const uint8_t * K8 = (const uint8_t *) K->data;
    int64_t sk1 = K->nb[1];
    int64_t sk2 = K->nb[2];
    int64_t sk3 = K->nb[3];
    if (K->type != GGML_TYPE_F8_E4M3) {
        const int64_t n_items = (int64_t) n_kv*n_kvh*n_skv*(D/8);
        K8_alloc.alloc((size_t) n_kv*n_kvh*n_skv*D);
        fattn_f8_convert_k<<<(n_items + 255)/256, 256, 0, stream>>>(
            (const char *) K->data, K8_alloc.ptr, D, n_kv, n_kvh, K->nb[1], K->nb[2], K->nb[3], n_items);
        K8  = K8_alloc.ptr;
        sk1 = D;
        sk2 = (int64_t) n_kv*D;
        sk3 = sk2*n_kvh;
    }
    if (n_skv == 1) {
        sk3 = 0;
    }

    ggml_cuda_pool_alloc<uint8_t> Vt(ctx.pool(), (size_t) n_kv*n_kvh*n_skv*D);
    fattn_f8_transpose_v<D><<<dim3(n_kv/32, n_kvh, n_skv), 256, 0, stream>>>(
        (const char *) V->data, Vt.ptr, V->type == GGML_TYPE_F8_E4M3, n_kv, n_kvh, V->nb[1], V->nb[2], V->nb[3]);
    const int64_t sv1 = n_kv;
    const int64_t sv2 = (int64_t) D*n_kv;
    const int64_t sv3 = n_skv == 1 ? 0 : sv2*n_kvh;

    const int ntiles = (n_q + FATTN_F8_BQ - 1)/FATTN_F8_BQ;
    const int64_t sm1 = mask->nb[1]/sizeof(half);
    const int64_t sm3 = mask->ne[3] == 1 ? 0 : mask->nb[3]/sizeof(half);
    ggml_cuda_pool_alloc<int> KV_max(ctx.pool(), (size_t) ntiles*n_s);
    fattn_f8_kv_max<<<dim3(ntiles, n_s), 256, 0, stream>>>((const half *) mask->data, KV_max.ptr, n_kv, mask->ne[1], sm1, sm3);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const size_t nbytes_shared = 2*FATTN_F8_BK*D + 2*D*FATTN_F8_BK + 2*FATTN_F8_BQ*FATTN_F8_BK*sizeof(half) + FATTN_F8_BQ*sizeof(float);
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attn_ext_f8<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        smem_set[ctx.device] = true;
    }

    flash_attn_ext_f8<D><<<dim3(ntiles, Q->ne[2], n_s), FATTN_F8_NW*WARP_SIZE, nbytes_shared, stream>>>(
        (const char *) Q->data, K8, Vt.ptr, (const half *) mask->data, KV_max.ptr, (float *) dst->data,
        scale*FATTN_F8_LOG2E, n_q, Q->ne[2], Q->ne[2]/n_kvh,
        Q->nb[1], Q->nb[2], Q->nb[3], sk1, sk2, sk3, sv1, sv2, sv3,
        mask->ne[1], sm1, sm3);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_f8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[0]) {
        case 128:
            ggml_cuda_flash_attn_ext_f8_case<128>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_f8_case<256>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
