#include "gated-delta-net-mma.cuh"
#include "common.cuh"
#include "mma.cuh"

#include <climits>
#include <limits>
#include <mutex>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "gated-delta-net-split.cuh"
#endif

namespace {

constexpr int GDN_D = 128;
constexpr int GDN_WARPS = 8;
constexpr int GDN_THREADS = 32 * GDN_WARPS;
#ifdef GGML_USE_HIP
constexpr int GDN_CHUNK = 32;
constexpr int GDN_N = 16;
#else
constexpr int GDN_CHUNK = 64;
constexpr int GDN_N = 8;
#endif

using bf16 = nv_bfloat16;

// Buffers alias only after their consumers have finished. The ROCm chunk
// leaves enough LDS for the full BF16 state within the 64 KiB block limit.
struct alignas(128) gdn_shared {
    union {
        bf16 q[GDN_CHUNK * GDN_D];
        bf16 value_hi[GDN_CHUNK * GDN_D];
    };
    bf16 k[GDN_CHUNK * GDN_D];
    bf16 qk[GDN_CHUNK * GDN_CHUNK];
    float inverse[GDN_CHUNK * GDN_CHUNK];
    union {
        float lower[GDN_CHUNK * GDN_CHUNK];
        bf16 state[GDN_D * GDN_D];
        bf16 value_lo[GDN_CHUNK * GDN_D];
    };
    float prefix[GDN_CHUNK];
    float beta[GDN_CHUNK];
};

__device__ __forceinline__ int gdn_index(int row, int col, int stride) {
#ifdef GGML_USE_HIP
    return row * stride + col;
#else
    return row * stride + (col ^ ((row & (stride / 8 - 1) & 7) << 3));
#endif
}

#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
using namespace ggml_cuda_mma;
#ifdef GGML_USE_HIP
using gdn_acc = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
using gdn_a = tile<16, 8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>;
using gdn_b = gdn_a;
#else
using gdn_acc = tile<16, 8, float>;
using gdn_a = tile<16, 8, nv_bfloat162>;
using gdn_b = tile<8, 8, nv_bfloat162>;
#endif

template<int stride_t, bool transpose_t = false>
struct gdn_bf16_view {
    const bf16 * data;
    __device__ __forceinline__ float get(int row, int col) const {
        return __bfloat162float(data[transpose_t ? gdn_index(col, row, stride_t) : gdn_index(row, col, stride_t)]);
    }
};

struct gdn_inverse_view {
    const float * data;
    __device__ __forceinline__ float get(int row, int col) const {
        return data[row * GDN_CHUNK + col];
    }
};

struct gdn_key_low_view {
    const bf16 * data;
    int64_t stride;
    int valid;
    __device__ __forceinline__ float get(int row, int col) const {
        return row < valid ? __bfloat162float(data[row * stride + col]) : 0.0f;
    }
};

template<class tile_t, bool low_t = false, class view_t>
__device__ __forceinline__ tile_t gdn_load(const view_t & src, int row, int col) {
    tile_t result;
#pragma unroll
    for (int i = 0; i < tile_t::ne; ++i) {
        const int r = row + tile_t::get_i(i);
        const int c = col + 2 * tile_t::get_j(i);
        float x = src.get(r, c);
        float y = src.get(r, c + 1);
        if constexpr (low_t) {
            x -= __bfloat162float(__float2bfloat16(x));
            y -= __bfloat162float(__float2bfloat16(y));
        }
        result.x[i] = __float22bfloat162_rn(make_float2(x, y));
    }
    return result;
}

template<int k_t, class a_view_t, class b_view_t>
__device__ __forceinline__ void gdn_product(gdn_acc & acc, const a_view_t & a, const b_view_t & b, int row, int col) {
#pragma unroll
    for (int k = 0; k < k_t; k += 16) {
        const auto ra = gdn_load<gdn_a>(a, row, k);
        const auto rb = gdn_load<gdn_b>(b, col, k);
        mma(acc, ra, rb);
    }
}

__device__ __forceinline__ void gdn_store_value(gdn_shared & smem, const gdn_acc & acc, int row, int col) {
#pragma unroll
    for (int i = 0; i < gdn_acc::ne; ++i) {
        const int index = gdn_index(col + gdn_acc::get_j(i), row + gdn_acc::get_i(i), GDN_D);
        const bf16 hi = __float2bfloat16(acc.x[i]);
        smem.value_hi[index] = hi;
        smem.value_lo[index] = __float2bfloat16(acc.x[i] - __bfloat162float(hi));
    }
}
#endif

static __global__ void gdn_prepack(
        const float * q, const float * k, bf16 * packed_q, bf16 * packed_k, bf16 * packed_k_low,
        int64_t H_k, int64_t tokens, int64_t sequences, int64_t rq3, int64_t sq1, int64_t sq2, int64_t sq3) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sequences * tokens * H_k * GDN_D) {
        return;
    }
    const int d = index % GDN_D;
    const int64_t row = index / GDN_D;
    const int h = row % H_k;
    const int t = (row / H_k) % tokens;
    const int seq = row / (H_k * tokens);
    const int64_t src = (seq / rq3) * sq3 + t * sq2 + h * sq1 + d;
    packed_q[index] = __float2bfloat16(q[src]);
    const bf16 high = __float2bfloat16(k[src]);
    packed_k[index] = high;
    packed_k_low[index] = __float2bfloat16(k[src] - __bfloat162float(high));
}

#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
__device__ __forceinline__ void gdn_prepare_chunk(gdn_shared &  smem,
                                                  const bf16 *  packed_q,
                                                  const bf16 *  packed_k,
                                                  const float * g,
                                                  const float * beta,
                                                  int64_t       q_offset,
                                                  int64_t       chunk,
                                                  int           valid,
                                                  int           seq,
                                                  int           head,
                                                  int64_t       H_k,
                                                  int64_t       sb1,
                                                  int64_t       sb2,
                                                  int64_t       sb3,
                                                  float         scale) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    for (int i = tid; i < GDN_CHUNK * GDN_D; i += GDN_THREADS) {
        const int     t     = i / GDN_D;
        const int     d     = i % GDN_D;
        const int64_t src   = q_offset + (chunk + t) * H_k * GDN_D + d;
        const int     index = gdn_index(t, d, GDN_D);
        smem.q[index]       = t < valid ? packed_q[src] : __float2bfloat16(0.0f);
        smem.k[index]       = t < valid ? packed_k[src] : __float2bfloat16(0.0f);
    }
    if (tid == 0) {
        float prefix = 0.0f;
        for (int t = 0; t < GDN_CHUNK; ++t) {
            const int64_t src = seq * sb3 + head * sb1 + (chunk + t) * sb2;
            prefix += t < valid ? g[src] : 0.0f;
            smem.prefix[t] = prefix;
            smem.beta[t]   = t < valid ? beta[src] : 0.0f;
        }
    }
    __syncthreads();
    const gdn_bf16_view<GDN_D> queries{ smem.q }, keys{ smem.k };
    for (int tile_index = warp % 4; tile_index < GDN_CHUNK * GDN_CHUNK / (16 * GDN_N); tile_index += 4) {
        const int row = (tile_index / (GDN_CHUNK / GDN_N)) * 16;
        const int col = (tile_index % (GDN_CHUNK / GDN_N)) * GDN_N;
        gdn_acc   acc;
        if (warp < 4) {
            gdn_product<GDN_D>(acc, keys, keys, row, col);
        } else {
            gdn_product<GDN_D>(acc, queries, keys, row, col);
        }
#    pragma unroll
        for (int i = 0; i < gdn_acc::ne; ++i) {
            const int r = row + gdn_acc::get_i(i);
            const int c = col + gdn_acc::get_j(i);
            if (warp < 4) {
                smem.lower[r * GDN_CHUNK + c] =
                    r < valid && c < r ? acc.x[i] * smem.beta[r] * expf(smem.prefix[r] - smem.prefix[c]) : 0.0f;
            } else {
                const float x = r < valid && c <= r ? acc.x[i] * scale * expf(smem.prefix[r] - smem.prefix[c]) : 0.0f;
                smem.qk[gdn_index(r, c, GDN_CHUNK)] = __float2bfloat16(x);
            }
        }
    }
    __syncthreads();
    // Invert the unit lower-triangular system in FP32, one row per warp.
    for (int row = warp; row < GDN_CHUNK; row += GDN_WARPS) {
        float x[GDN_CHUNK / 32];
#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / 32; ++j) {
            const int col = lane + 32 * j;
            x[j]          = col == row ? 1.0f : -smem.lower[row * GDN_CHUNK + col];
        }
        for (int pivot = row - 1; pivot > 0; --pivot) {
            const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#    pragma unroll
            for (int j = 0; j < GDN_CHUNK / 32; ++j) {
                const int col = lane + 32 * j;
                if (col < pivot) {
                    x[j] -= xp * smem.lower[pivot * GDN_CHUNK + col];
                }
            }
        }
#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / 32; ++j) {
            smem.inverse[row * GDN_CHUNK + lane + 32 * j] = x[j];
        }
    }
    __syncthreads();
}
#endif

static __global__ __launch_bounds__(GDN_THREADS, 1) void gdn_persistent(const bf16 *  packed_q,
                                                                        const bf16 *  packed_k,
                                                                        const bf16 *  packed_k_low,
                                                                        const float * v,
                                                                        const float * g,
                                                                        const float * beta,
                                                                        const float * state_in,
                                                                        float *       dst,
                                                                        float *       state_out,
                                                                        int64_t       H,
                                                                        int64_t       H_k,
                                                                        int64_t       tokens,
                                                                        int64_t       sb1,
                                                                        int64_t       sb2,
                                                                        int64_t       sb3,
                                                                        int64_t       sv1,
                                                                        int64_t       sv2,
                                                                        int64_t       sv3,
                                                                        float         scale) {
#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto &                                         smem         = *reinterpret_cast<gdn_shared *>(shared_bytes);
    const int                                      lane         = threadIdx.x;
    const int                                      warp         = threadIdx.y;
    const int                                      tid          = warp * 32 + lane;
    const int                                      seq          = blockIdx.x / H;
    const int                                      head         = blockIdx.x % H;
    const int                                      value_base   = warp * 16;
    const int64_t                                  state_offset = ((int64_t) seq * H + head) * GDN_D * GDN_D;
    const int64_t                                  q_offset     = ((int64_t) seq * tokens * H_k + head % H_k) * GDN_D;
    const int64_t                                  v_offset     = seq * sv3 + head * sv1;
    float *                                        out          = dst + ((int64_t) seq * tokens * H + head) * GDN_D;
    // Each warp owns 16 value rows and keeps their FP32 state across chunks.
    gdn_acc                                        state[GDN_D / GDN_N];

#    pragma unroll
    for (int j = 0; j < GDN_D / GDN_N; ++j) {
#    pragma unroll
        for (int i = 0; i < gdn_acc::ne; ++i) {
            const int row = value_base + gdn_acc::get_i(i);
            const int col = j * GDN_N + gdn_acc::get_j(i);
            state[j].x[i] = state_in[state_offset + row * GDN_D + col];
        }
    }

    for (int64_t chunk = 0; chunk < tokens; chunk += GDN_CHUNK) {
        const int valid = min((int64_t) GDN_CHUNK, tokens - chunk);

        gdn_prepare_chunk(smem, packed_q, packed_k, g, beta, q_offset, chunk, valid, seq, head, H_k, sb1, sb2, sb3,
                          scale);

        const gdn_bf16_view<GDN_D> queries{ smem.q }, keys{ smem.k };

#    pragma unroll
        for (int j = 0; j < GDN_D / GDN_N; ++j) {
#    pragma unroll
            for (int i = 0; i < gdn_acc::ne; ++i) {
                smem.state[gdn_index(value_base + gdn_acc::get_i(i), j * GDN_N + gdn_acc::get_j(i), GDN_D)] =
                    __float2bfloat16(state[j].x[i]);
            }
        }
        __syncthreads();
        gdn_acc                    residual[GDN_CHUNK / GDN_N];
        const gdn_bf16_view<GDN_D> state_view{ smem.state };
        const gdn_key_low_view     key_low{ packed_k_low + q_offset + chunk * H_k * GDN_D, H_k * GDN_D, valid };
#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / GDN_N; ++j) {
            gdn_acc output;

            gdn_product<GDN_D>(output, state_view, queries, value_base, j * GDN_N);

            gdn_product<GDN_D>(residual[j], state_view, keys, value_base, j * GDN_N);
            gdn_product<GDN_D>(residual[j], state_view, key_low, value_base, j * GDN_N);
#    pragma unroll
            for (int i = 0; i < gdn_acc::ne; ++i) {
                const int   t     = j * GDN_N + gdn_acc::get_j(i);
                const int   value = value_base + gdn_acc::get_i(i);
                const float decay = expf(smem.prefix[t]);
                const float vv    = t < valid ? v[v_offset + (chunk + t) * sv2 + value] : 0.0f;
                residual[j].x[i]  = (vv - decay * residual[j].x[i]) * smem.beta[t];
                if (t < valid) {
                    out[(chunk + t) * H * GDN_D + value] = scale * decay * output.x[i];
                }
            }
        }
        __syncthreads();
#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / GDN_N; ++j) {
            gdn_store_value(smem, residual[j], value_base, j * GDN_N);
        }
        __syncthreads();
        const gdn_bf16_view<GDN_D, true> values_hi{ smem.value_hi }, values_lo{ smem.value_lo };
        gdn_a                            a_hi[GDN_CHUNK / 16], a_lo[GDN_CHUNK / 16];
#    pragma unroll
        for (int k = 0; k < GDN_CHUNK / 16; ++k) {
            a_hi[k] = gdn_load<gdn_a>(values_hi, value_base, k * 16);
            a_lo[k] = gdn_load<gdn_a>(values_lo, value_base, k * 16);
        }
        __syncwarp();
        // Preserve residuals through the solve: a_hi*b_hi + a_lo*b_hi + a_hi*b_lo.
        // A single BF16 product loses accuracy for weak/zero-decay sequences.
        const gdn_inverse_view inverse{ smem.inverse };
#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / GDN_N; ++j) {
            gdn_acc delta;
#    pragma unroll
            for (int k = 0; k < GDN_CHUNK / 16; ++k) {
                const auto b_hi = gdn_load<gdn_b>(inverse, j * GDN_N, k * 16);
                const auto b_lo = gdn_load<gdn_b, true>(inverse, j * GDN_N, k * 16);
                mma(delta, a_hi[k], b_hi);
                mma(delta, a_lo[k], b_hi);
                mma(delta, a_hi[k], b_lo);
            }
            gdn_store_value(smem, delta, value_base, j * GDN_N);
        }
        __syncthreads();
#    pragma unroll
        for (int k = 0; k < GDN_CHUNK / 16; ++k) {
            a_hi[k] = gdn_load<gdn_a>(values_hi, value_base, k * 16);
            a_lo[k] = gdn_load<gdn_a>(values_lo, value_base, k * 16);
        }
        __syncthreads();
        for (int i = tid; i < GDN_CHUNK * GDN_D; i += GDN_THREADS) {
            const int t = i / GDN_D;
            const int d = i % GDN_D;
            smem.q[gdn_index(t, d, GDN_D)] =
                t < valid ? packed_k_low[q_offset + (chunk + t) * H_k * GDN_D + d] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        const gdn_bf16_view<GDN_CHUNK> qk{ smem.qk };

#    pragma unroll
        for (int j = 0; j < GDN_CHUNK / GDN_N; ++j) {
            gdn_acc output;
#    pragma unroll
            for (int k = 0; k < GDN_CHUNK / 16; ++k) {
                const auto b = gdn_load<gdn_b>(qk, j * GDN_N, k * 16);
                mma(output, a_hi[k], b);
                mma(output, a_lo[k], b);
            }
#    pragma unroll
            for (int i = 0; i < gdn_acc::ne; ++i) {
                const int t = j * GDN_N + gdn_acc::get_j(i);
                if (t < valid) {
                    out[(chunk + t) * H * GDN_D + value_base + gdn_acc::get_i(i)] += output.x[i];
                }
            }
        }

#    pragma unroll
        for (int k = 0; k < GDN_CHUNK / 16; ++k) {
#    pragma unroll
            for (int i = 0; i < gdn_a::ne; ++i) {
                const int    t       = k * 16 + 2 * gdn_a::get_j(i);
                const float2 hi      = __bfloat1622float2(a_hi[k].x[i]);
                const float2 lo      = __bfloat1622float2(a_lo[k].x[i]);
                const float  x       = (hi.x + lo.x) * expf(smem.prefix[valid - 1] - smem.prefix[t]);
                const float  y       = (hi.y + lo.y) * expf(smem.prefix[valid - 1] - smem.prefix[t + 1]);
                a_hi[k].x[i]         = __float22bfloat162_rn(make_float2(x, y));
                const float2 rounded = __bfloat1622float2(a_hi[k].x[i]);
                a_lo[k].x[i]         = __float22bfloat162_rn(make_float2(x - rounded.x, y - rounded.y));
            }
        }
        const gdn_bf16_view<GDN_D, true> keys_t{ smem.k }, keys_low_t{ smem.q };
        const float                      decay = expf(smem.prefix[valid - 1]);
#    pragma unroll
        for (int j = 0; j < GDN_D / GDN_N; ++j) {
#    pragma unroll
            for (int i = 0; i < gdn_acc::ne; ++i) {
                state[j].x[i] *= decay;
            }
#    pragma unroll
            for (int k = 0; k < GDN_CHUNK / 16; ++k) {
                const auto b_hi = gdn_load<gdn_b>(keys_t, j * GDN_N, k * 16);
                const auto b_lo = gdn_load<gdn_b>(keys_low_t, j * GDN_N, k * 16);
                mma(state[j], a_hi[k], b_hi);
                mma(state[j], a_lo[k], b_hi);
                mma(state[j], a_hi[k], b_lo);
            }
        }
        __syncthreads();
    }

#    pragma unroll
    for (int j = 0; j < GDN_D / GDN_N; ++j) {
#    pragma unroll
        for (int i = 0; i < gdn_acc::ne; ++i) {
            state_out[state_offset + (value_base + gdn_acc::get_i(i)) * GDN_D + j * GDN_N + gdn_acc::get_j(i)] =
                state[j].x[i];
        }
    }

#else
    GGML_UNUSED_VARS(packed_q, packed_k, packed_k_low, v, g, beta, state_in, dst, state_out, H, H_k, tokens, sb1, sb2,
                     sb3, sv1, sv2, sv3, scale);
    NO_DEVICE_CODE;
#endif
}

struct gdn_workspace {
    bool   value_split = false;
    size_t elements    = 0;
    size_t bytes       = 0;
};

static bool gdn_size_mul(size_t & value, size_t factor) {
    if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor) {
        return false;
    }
    value *= factor;
    return true;
}

static bool gdn_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        const auto & info = ggml_cuda_info().devices[device];
#ifdef GGML_USE_HIP
        const bool supported = info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151;
#elif defined(GGML_USE_MUSA)
        const bool supported = false;
#else
        // Older PTX can JIT on these GPUs, but lacks the Ampere MMA kernel body.
        const bool supported = ampere_mma_available(info.cc) &&
                               (info.cc == GGML_CUDA_CC_BLACKWELL || info.cc == GGML_CUDA_CC_DGX_SPARK);
#endif
        if (!supported || sizeof(gdn_shared) > info.smpbo) {
            return;
        }
        const cudaError_t status = cudaFuncSetAttribute(
            (const void *) gdn_persistent, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(gdn_shared));
        available[device] = status == cudaSuccess;

        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static bool gdn_split_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        const auto status =
            cudaFuncSetAttribute((const void *) gdn_value_split::gdn_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 gdn_value_split::SHARED_BYTES);
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

static bool gdn_use_value_split(int device, const ggml_cuda_gdn_mma_args & args) {
    const auto & info = ggml_cuda_info().devices[device];
    if (!ampere_mma_available(info.cc) || (info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_DGX_SPARK) ||
        args.n_tokens < 512 || args.n_tokens > INT_MAX - gdn_value_split::CHUNK || args.n_seqs > 65535 ||
        args.n_seqs > (info.nsm - 1) / args.H) {
        return false;
    }
    // Four independent value slices increase the block count when heads/sequences underfill the GPU.
    // The fused loads require contiguous Q/K and head-contiguous V; other layouts use the existing path.
    const uintptr_t input_alignment = uintptr_t(args.q) | uintptr_t(args.k) | uintptr_t(args.v);
    return (input_alignment & 15) == 0 && args.rq3 == 1 && args.sq1 == GDN_D && args.sq2 == GDN_D * args.H_k &&
           args.sq3 == args.n_tokens * args.sq2 && args.sv1 == GDN_D && args.sv2 % 4 == 0 &&
           args.sv3 == args.n_tokens * args.sv2 && args.sb1 == 1 && args.sb2 == args.H &&
           args.sb3 == args.n_tokens * args.H && gdn_split_init(device);
}
#endif

static bool gdn_plan(int device, const ggml_cuda_gdn_mma_args & args, gdn_workspace & workspace) {
    if (!args.eligible || (args.H != 16 && args.H != 32 && args.H != 48 && args.H != 64) || args.H_k != 16 ||
        args.n_tokens < 64 || args.n_seqs <= 0) {
        return false;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (gdn_use_value_split(device, args)) {
        workspace.value_split = true;
        return true;
    }
#endif
    if (!gdn_init(device)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    // Use WMMA from the shortest prompt size with a measured win on gfx1151.
    if (cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 && args.n_tokens < 2048) {
        return false;
    }
    // Persistent MMA regresses with 16/32 value heads on SM120.
    if (cc == GGML_CUDA_CC_BLACKWELL && (args.H == 16 || args.H == 32 || (args.H == 48 && args.n_tokens < 256))) {
        return false;
    }
    size_t elements = (size_t) args.n_seqs;
    if (!gdn_size_mul(elements, args.n_tokens) || !gdn_size_mul(elements, args.H_k) || !gdn_size_mul(elements, GDN_D) ||
        elements > (size_t) INT_MAX * 256) {
        return false;
    }
    size_t bytes = elements;
    if (!gdn_size_mul(bytes, 3 * sizeof(bf16))) {
        return false;
    }

    workspace = { false, elements, bytes };
    return true;
}

}  // namespace

bool ggml_cuda_gdn_mma_available(int device, const ggml_cuda_gdn_mma_args & args) {
    gdn_workspace workspace;
    return gdn_plan(device, args, workspace);
}

size_t ggml_cuda_gdn_mma_get_alloc_size(int device, const ggml_cuda_gdn_mma_args & args, size_t logical_size) {
    gdn_workspace workspace;
    if (!gdn_plan(device, args, workspace)) {
        return logical_size;
    }
    constexpr size_t alignment = 128;
    GGML_ASSERT(logical_size <= SIZE_MAX - (alignment - 1));
    const size_t offset = (logical_size + alignment - 1) & ~(alignment - 1);
    GGML_ASSERT(offset <= SIZE_MAX - workspace.bytes);
    return offset + workspace.bytes;
}

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream) {
    gdn_workspace workspace;
    if (!args.state_out || !gdn_plan(device, args, workspace)) {
        return false;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (workspace.value_split) {
        gdn_value_split::gdn_fused<<<dim3(args.H * 4, args.n_seqs), gdn_value_split::NTHREADS,
                                     gdn_value_split::SHARED_BYTES, stream>>>(
            args.n_tokens, args.H_k, args.H, GDN_D, args.q, args.k, args.v, args.g, args.beta, args.dst, args.state,
            args.state_out, args.sv2);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
#endif
    if (!args.workspace || workspace.bytes > args.workspace_size) {
        return false;
    }
    auto *    q      = static_cast<bf16 *>(args.workspace);
    auto *    k      = q + workspace.elements;
    auto *    low    = k + workspace.elements;
    const int blocks = (workspace.elements + 255) / 256;
    gdn_prepack<<<blocks, 256, 0, stream>>>(args.q, args.k, q, k, low, args.H_k, args.n_tokens, args.n_seqs, args.rq3,
                                            args.sq1, args.sq2, args.sq3);
    CUDA_CHECK(cudaGetLastError());
    gdn_persistent<<<args.n_seqs * args.H, dim3(32, GDN_WARPS), sizeof(gdn_shared), stream>>>(
        q, k, low, args.v, args.g, args.beta, args.state, args.dst, args.state_out, args.H, args.H_k, args.n_tokens,
        args.sb1, args.sb2, args.sb3, args.sv1, args.sv2, args.sv3, args.scale);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
