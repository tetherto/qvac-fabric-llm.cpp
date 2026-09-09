#ifdef GGML_CUDA_CUTLASS

#include "gated-delta-net-cutlass.cuh"

#include "common.cuh"
#include "gdn-cute-inverse.cuh"

#include <cute/tensor.hpp>
#include <cutlass/tfloat32.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/cutlass.h>

#include <cmath>
#include <limits>
#include <mutex>

namespace ggml_gdn_cute {

using namespace cute;

using tf32 = cutlass::tfloat32_t;

CUTE_DEVICE tf32 to_tf32(float value) {
    return cutlass::NumericConverter<tf32, float, cutlass::FloatRoundStyle::round_to_nearest>{}(value);
}

constexpr int GDN_D       = 128;
constexpr int GDN_CHUNK   = 64;
constexpr int GDN_THREADS = 256;

using MmaAtom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
using WarpMma = decltype(make_tiled_mma(
    MmaAtom{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{}));

template <class Accumulator>
CUTE_DEVICE auto state_to_tf32(const Accumulator & acc) {
    static_assert(size(Accumulator{}) == 32);
    Tensor operand = make_fragment_like<tf32>(acc);
    const int lane = threadIdx.x & 31;
    // TF32 operands and FP32 accumulators assign columns to different lanes.
#pragma unroll
    for (int block = 0; block < 8; ++block) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
#pragma unroll
            for (int key = 0; key < 2; ++key) {
                const int src_lane = (lane & ~3) + ((lane & 3) >> 1) + key * 2;
                const float a = __shfl_sync(0xffffffff, acc(block * 4 + row * 2), src_lane);
                const float b = __shfl_sync(0xffffffff, acc(block * 4 + row * 2 + 1), src_lane);
                operand(block * 4 + row + key * 2) = to_tf32((lane & 1) ? b : a);
            }
        }
    }
    return operand;
}

struct alignas(128) GdnSharedStorage {
    union {
        tf32 q[GDN_CHUNK * GDN_D];
        tf32 new_v[GDN_D * GDN_CHUNK];
    };
    tf32 k[GDN_CHUNK * GDN_D];
    tf32 qk[GDN_CHUNK * GDN_CHUNK];
    cutlass::half_t inverse[GDN_CHUNK * GDN_CHUNK];
    float prefix[GDN_CHUNK];
    float beta[GDN_CHUNK];
};

static_assert(sizeof(GdnSharedStorage) <= 101376);

__global__ void gdn_prepack_qk_f32_tf32(
        const float * __restrict__ q,
        const float * __restrict__ k,
        tf32 * __restrict__ packed_q,
        tf32 * __restrict__ packed_k,
        tf32 * __restrict__ packed_k_low,
        int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t q_elems = n_seqs * n_tokens * H_k * GDN_D;
    if (index < q_elems) {
        const int d = index % GDN_D;
        const int64_t row = index / GDN_D;
        const int h = row % H_k;
        const int64_t token_seq = row / H_k;
        const int t = token_seq % n_tokens;
        const int seq = token_seq / n_tokens;
        const int iq3 = seq / rq3;
        const int64_t src = iq3 * sq3 + t * sq2 + h * sq1 + d;
        packed_q[index] = to_tf32(q[src]);
        const float key = k[src];
        const tf32 high = to_tf32(key);
        packed_k[index] = high;
        // Preserve key rounding errors for the state update.
        packed_k_low[index] = to_tf32(key - float(high));
    }
}

template <class TensorC>
CUTE_DEVICE void load_state_fragment(
        TensorC & state_frag, const float * state, int value_base, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c = coord(i);
        state_frag(i) = state[(value_base + get<0>(c)) * GDN_D + get<1>(c)];
    }
}

template <class TensorC>
CUTE_DEVICE void store_state_fragment(
        const TensorC & state_frag, float * state, int value_base, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c = coord(i);
        state[(value_base + get<0>(c)) * GDN_D + get<1>(c)] = state_frag(i);
    }
}

template <class TensorC>
CUTE_DEVICE void store_matrix_fragment_tf32(
        const TensorC & frag, tf32 * dst, int ld, int row_base, int col_base,
        int valid_rows, int valid_cols, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const auto c = coord(i);
        const int r = row_base + get<0>(c);
        const int col = col_base + get<1>(c);
        if (r < valid_rows && col < valid_cols) {
            dst[col * ld + (r ^ ((col & 7) << 2))] = to_tf32(frag(i));
        }
    }
}

template <class TensorC>
CUTE_DEVICE void store_output_fragment(
        const TensorC & frag, float * dst, int64_t token_stride, int value_base,
        int token_base, int valid_tokens, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const auto c = coord(i);
        const int value = value_base + get<0>(c);
        const int token = token_base + get<1>(c);
        if (token < valid_tokens) {
            dst[(int64_t) token * token_stride + value] = frag(i);
        }
    }
}

template <class TensorC>
CUTE_DEVICE void apply_token_prefix_scale(
        TensorC & frag, const float * prefix, int token_base, float scale, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        frag(i) *= scale * expf(prefix[token_base + get<1>(coord(i))]);
    }
}

template <class TensorC>
CUTE_DEVICE void subtract_v(
        TensorC & frag, const float * v, int64_t token_stride, const float * beta,
        int value_base, int token_base, int valid_tokens, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const int value = value_base + get<0>(coord(i));
        const int token = token_base + get<1>(coord(i));
        const float vv = token < valid_tokens ? v[token * token_stride + value] : 0.0f;
        frag(i) = (vv - frag(i)) * beta[token];
    }
}

template <class TensorC>
CUTE_DEVICE void scale_accumulator(TensorC & frag, float scale) {
    for (int i = 0; i < size(frag); ++i) {
        frag(i) *= scale;
    }
}

__global__ __maxnreg__(192) void gdn_cute_sm12x(
        const tf32 * __restrict__ packed_q,
        const tf32 * __restrict__ packed_k,
        const tf32 * __restrict__ packed_k_low,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ state_in,
        float * __restrict__ dst,
        float * __restrict__ state_out,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t sv1, int64_t sv2, int64_t sv3, float output_scale) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto & smem = *reinterpret_cast<GdnSharedStorage *>(smem_raw);

    const int seq = blockIdx.x / H;
    const int vh = blockIdx.x % H;
    const int kh = vh % H_k;
    const int math_tid = threadIdx.x;
    const int math_warp = math_tid >> 5;
    const int lane = threadIdx.x & 31;
    const int value_base = math_warp * 16;
    const int64_t state_offset = ((int64_t) seq * H + vh) * GDN_D * GDN_D;
    float * out = dst + ((int64_t) seq * n_tokens * H + vh) * GDN_D;

    WarpMma mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto state_frag = partition_fragment_C(mma, Shape<_16, _128>{});
    load_state_fragment(state_frag, state_in + state_offset, value_base, mma);

    const int64_t q_head_offset = (((int64_t) seq * n_tokens) * H_k + kh) * GDN_D;
    const int64_t v_head_offset = seq * sv3 + vh * sv1;

    for (int64_t chunk = 0; chunk < n_tokens; chunk += GDN_CHUNK) {
        const int valid = min((int64_t) GDN_CHUNK, n_tokens - chunk);
        float * chunk_out = out + chunk * H * GDN_D;

        if (threadIdx.x < 128) {
            constexpr int vectors_per_row = GDN_D * sizeof(tf32) / sizeof(int4);
            for (int i = threadIdx.x; i < GDN_CHUNK * vectors_per_row; i += 128) {
                const int t = i / vectors_per_row;
                const int vector = i % vectors_per_row;
                int4 * sq = reinterpret_cast<int4 *>(smem.q + t * GDN_D) + (vector ^ (t & 7));
                int4 * sk = reinterpret_cast<int4 *>(smem.k + t * GDN_D) + (vector ^ (t & 7));
                if (t < valid) {
                    const int64_t qoff = q_head_offset + (chunk + t) * H_k * GDN_D;
                    *sq = reinterpret_cast<const int4 *>(packed_q + qoff)[vector];
                    *sk = reinterpret_cast<const int4 *>(packed_k + qoff)[vector];
                } else {
                    *sq = make_int4(0, 0, 0, 0);
                    *sk = make_int4(0, 0, 0, 0);
                }
            }
        }
        if (threadIdx.x < GDN_CHUNK) {
            const int t = threadIdx.x;
            if (t < valid) {
                const int64_t off = seq * sb3 + vh * sb1 + (chunk + t) * sb2;
                smem.prefix[t] = g[off];
                smem.beta[t] = beta[off];
            } else {
                smem.prefix[t] = 0.0f;
                smem.beta[t] = 0.0f;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int t = 0; t < GDN_CHUNK; ++t) {
                sum += smem.prefix[t];
                smem.prefix[t] = sum;
            }
        }
        __syncthreads();

        auto sQ = make_tensor(make_smem_ptr(smem.q), composition(Swizzle<3, 2, 5>{}, make_layout(Shape<_64, _128>{}, LayoutRight{})));
        auto sK = make_tensor(make_smem_ptr(smem.k), composition(Swizzle<3, 2, 5>{}, make_layout(Shape<_64, _128>{}, LayoutRight{})));

        if (threadIdx.x < 128) {
            const int warp = threadIdx.x >> 5;
            const int row_base = warp * 16;
            for (int col_base = 0; col_base < GDN_CHUNK; col_base += 16) {
                auto a = local_tile(sK, Shape<_16, _128>{}, make_coord(warp, 0));
                auto b = local_tile(sK, Shape<_16, _128>{}, make_coord(col_base / 16, 0));
                auto ra = thr_mma.partition_fragment_A(a);
                auto rb = thr_mma.partition_fragment_B(b);
                copy(thr_mma.partition_A(a), ra);
                copy(thr_mma.partition_B(b), rb);
                auto rc = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(rc);
                gemm(mma, ra, rb, rc);
                auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
                for (int i = 0; i < size(rc); ++i) {
                    const int r = row_base + get<0>(coord(i));
                    const int c = col_base + get<1>(coord(i));
                    float x = 0.0f;
                    if (r < valid && c < valid) {
                        if (r == c) {
                            x = 1.0f;
                        } else if (r > c) {
                            x = rc(i) * smem.beta[r] * expf(smem.prefix[r] - smem.prefix[c]);
                        }
                    }
                    smem.inverse[r * GDN_CHUNK + c] = cutlass::half_t(x);
                }
            }
        } else {
            const int warp = (threadIdx.x - 128) >> 5;
            const int row_base = warp * 16;
            for (int col_base = 0; col_base < GDN_CHUNK; col_base += 16) {
                auto a = local_tile(sQ, Shape<_16, _128>{}, make_coord(warp, 0));
                auto b = local_tile(sK, Shape<_16, _128>{}, make_coord(col_base / 16, 0));
                auto ra = thr_mma.partition_fragment_A(a);
                auto rb = thr_mma.partition_fragment_B(b);
                copy(thr_mma.partition_A(a), ra);
                copy(thr_mma.partition_B(b), rb);
                auto rc = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(rc);
                gemm(mma, ra, rb, rc);
                auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
                for (int i = 0; i < size(rc); ++i) {
                    const int r = row_base + get<0>(coord(i));
                    const int c = col_base + get<1>(coord(i));
                    float x = 0.0f;
                    if (r < valid && c < valid && r >= c) {
                        x = rc(i) * output_scale * expf(smem.prefix[r] - smem.prefix[c]);
                    }
                    smem.qk[r * GDN_CHUNK + (c ^ ((r & 7) << 2))] = to_tf32(x);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x < 128) {
            auto inverse = make_tensor(make_smem_ptr(smem.inverse),
                make_layout(Shape<_64, _64>{}, LayoutRight{}));
            flat::collective::CollectiveInverse<cutlass::half_t, true, true>(3).compute(inverse);
        }
        __syncthreads();

        {
            auto state_layout = partition_fragment_C(mma, Shape<_16, _64>{}).layout();
            auto state_first = make_tensor(state_frag.data(), state_layout);
            auto state_second = make_tensor(state_frag.data() + 32, state_layout);
            auto state_op_first = state_to_tf32(state_first);
            auto state_op_second = state_to_tf32(state_second);
            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto o1 = partition_fragment_C(mma, Shape<_16, _16>{});
                auto sk = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(o1);
                clear(sk);
#pragma unroll
                for (int key_base = 0; key_base < GDN_D; key_base += 64) {
                    auto & state_op = key_base == 0 ? state_op_first : state_op_second;
                    auto q_tile = local_tile(sQ, Shape<_16, _64>{}, make_coord(token_base / 16, key_base / 64));
                    auto rq = thr_mma.partition_fragment_B(q_tile);
                    copy(thr_mma.partition_B(q_tile), rq);
                    gemm(mma, state_op, rq, o1);
                    auto k_tile = local_tile(sK, Shape<_16, _64>{}, make_coord(token_base / 16, key_base / 64));
                    auto rk = thr_mma.partition_fragment_B(k_tile);
                    copy(thr_mma.partition_B(k_tile), rk);
                    gemm(mma, state_op, rk, sk);
                }
                apply_token_prefix_scale(o1, smem.prefix, token_base, output_scale, mma);
                apply_token_prefix_scale(sk, smem.prefix, token_base, 1.0f, mma);
                subtract_v(sk, v + v_head_offset + chunk * sv2, sv2, smem.beta,
                    value_base, token_base, valid, mma);
                store_output_fragment(o1, chunk_out, H * GDN_D, value_base,
                    token_base, valid, mma);
                // Reuse each query tile after all math warps have read it.
                __syncthreads();
                store_matrix_fragment_tf32(sk, smem.new_v, GDN_D,
                    value_base, token_base, GDN_D, valid, mma);
            }
        }
        __syncthreads();

        {
            auto sNewV = make_tensor(make_smem_ptr(smem.new_v), composition(Swizzle<3, 2, 5>{}, make_layout(Shape<_128, _64>{}, Stride<_1, _128>{})));
            auto sCorrectionT = make_tensor(make_smem_ptr(smem.inverse),
                make_layout(Shape<_64, _64>{}, LayoutRight{}));
            auto sQKT = make_tensor(make_smem_ptr(smem.qk),
                composition(Swizzle<3, 2, 4>{}, make_layout(Shape<_64, _64>{}, LayoutRight{})));
            auto sKT = make_tensor(make_smem_ptr(smem.k),
                composition(Swizzle<3, 2, 5>{}, make_layout(Shape<_128, _64>{}, Stride<_1, _128>{})));

            auto delta_base = local_tile(sNewV, Shape<_16, _64>{}, make_coord(math_warp, 0));
            auto ra = thr_mma.partition_fragment_A(delta_base);
            copy(thr_mma.partition_A(delta_base), ra);
            // Finish reading this warp's rows before overwriting them.
            __syncwarp();
            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto corr = local_tile(sCorrectionT, Shape<_16, _64>{}, make_coord(token_base / 16, 0));
                auto rb = thr_mma.partition_fragment_B(corr);
                auto src = thr_mma.partition_B(corr);
                for (int i = 0; i < size(rb); ++i) { rb(i) = to_tf32(float(src(i))); }
                auto newv_tile = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(newv_tile);
                gemm(mma, ra, rb, newv_tile);
                store_matrix_fragment_tf32(newv_tile, smem.new_v, GDN_D,
                    value_base, token_base, GDN_D, valid, mma);
            }
            __syncthreads();

            auto newv_all = local_tile(sNewV, Shape<_16, _64>{}, make_coord(math_warp, 0));
            auto rnewv = thr_mma.partition_fragment_A(newv_all);
            copy(thr_mma.partition_A(newv_all), rnewv);
            __syncthreads();
            constexpr int vectors_per_row = GDN_D * sizeof(tf32) / sizeof(int4);
            for (int i = math_tid; i < GDN_CHUNK * vectors_per_row; i += 256) {
                const int token = i / vectors_per_row;
                const int vector = i % vectors_per_row;
                int4 * target = reinterpret_cast<int4 *>(smem.q + token * GDN_D) + (vector ^ (token & 7));
                const int64_t off = q_head_offset + (chunk + token) * H_k * GDN_D;
                *target = token < valid ? reinterpret_cast<const int4 *>(packed_k_low + off)[vector] : make_int4(0, 0, 0, 0);
            }
            __syncthreads();
            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto qk_tile = local_tile(sQKT, Shape<_16, _64>{}, make_coord(token_base / 16, 0));
                auto rqk = thr_mma.partition_fragment_B(qk_tile);
                copy(thr_mma.partition_B(qk_tile), rqk);
                auto o2 = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(o2);
                gemm(mma, rnewv, rqk, o2);
                auto thr = mma.get_thread_slice(lane);
                auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
                for (int i = 0; i < size(o2); ++i) {
                    const int value = value_base + get<0>(coord(i));
                    const int token = token_base + get<1>(coord(i));
                    if (token < valid) {
                        chunk_out[(int64_t) token * H * GDN_D + value] += o2(i);
                    }
                }
            }

            const float block_decay = expf(smem.prefix[valid - 1]);
            scale_accumulator(state_frag, block_decay);
            auto rk = thr_mma.partition_fragment_B(sKT);
            copy(thr_mma.partition_B(sKT), rk);
            auto weighted_newv = make_fragment_like<tf32>(rnewv);
            auto coord_a = thr_mma.partition_A(make_identity_tensor(Shape<_16, _64>{}));
            for (int i = 0; i < size(rnewv); ++i) {
                const int tok = get<1>(coord_a(i));
                const float w = tok < valid ? expf(smem.prefix[valid - 1] - smem.prefix[tok]) : 0.0f;
                weighted_newv(i) = to_tf32(float(rnewv(i)) * w);
            }
            gemm(mma, weighted_newv, rk, state_frag);
            auto sKLowT = make_tensor(make_smem_ptr(smem.q),
                composition(Swizzle<3, 2, 5>{}, make_layout(Shape<_128, _64>{}, Stride<_1, _128>{})));
#pragma unroll
            for (int key_base = 0; key_base < GDN_D; key_base += 16) {
                auto k_tile = local_tile(sKLowT, Shape<_16, _64>{}, make_coord(key_base / 16, 0));
                auto low_k = thr_mma.partition_fragment_B(k_tile);
                copy(thr_mma.partition_B(k_tile), low_k);
                auto state_layout = partition_fragment_C(mma, Shape<_16, _16>{}).layout();
                auto state_slice = make_tensor(state_frag.data() + key_base / 2, state_layout);
                gemm(mma, weighted_newv, low_k, state_slice);
            }
        }
        __syncthreads();
    }

    store_state_fragment(state_frag, state_out + state_offset, value_base, mma);
}

} // namespace ggml_gdn_cute

using namespace ggml_gdn_cute;

static bool ggml_cuda_gdn_cute_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool available[GGML_CUDA_MAX_DEVICES] = {};

    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        const auto & info = ggml_cuda_info().devices[device];
        if ((info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_DGX_SPARK) ||
            sizeof(GdnSharedStorage) > info.smpbo) {
            return;
        }
        const cudaError_t status = cudaFuncSetAttribute(
            gdn_cute_sm12x, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(GdnSharedStorage));
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

bool ggml_cuda_gdn_cute_available(int device, const ggml_cuda_gdn_cute_args & args) {
    if (!args.eligible || (args.H != 16 && args.H != 32 && args.H != 48 && args.H != 64) || args.H_k != 16 ||
        args.n_tokens < GDN_CHUNK || args.n_seqs <= 0) {
        return false;
    }
    // The native kernel is faster for these SM120 workloads.
    if (ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_BLACKWELL &&
        (args.H == 16 || (args.H == 32 && args.n_tokens < 512))) {
        return false;
    }
    return ggml_cuda_gdn_cute_init(device);
}

struct gdn_cute_workspace_layout {
    size_t q_elems;
    size_t bytes;
};

static bool gdn_cute_size_mul(size_t & value, size_t factor) {
    if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor) {
        return false;
    }
    value *= factor;
    return true;
}

static bool ggml_cuda_gdn_cute_workspace_layout(
        const ggml_cuda_gdn_cute_args & args, gdn_cute_workspace_layout & layout) {
    size_t q_elems = (size_t) args.n_seqs;
    if (!gdn_cute_size_mul(q_elems, (size_t) args.n_tokens) ||
        !gdn_cute_size_mul(q_elems, (size_t) args.H_k) ||
        !gdn_cute_size_mul(q_elems, GDN_D)) {
        return false;
    }

    size_t total_bytes = q_elems;
    if (!gdn_cute_size_mul(total_bytes, 3 * sizeof(tf32))) {
        return false;
    }

    layout = { q_elems, total_bytes };
    return true;
}

size_t ggml_cuda_gdn_cute_get_alloc_size(
        int device, const ggml_cuda_gdn_cute_args & args, size_t logical_size) {
    ggml_cuda_set_device(device);
    if (!ggml_cuda_gdn_cute_available(device, args)) {
        return logical_size;
    }

    gdn_cute_workspace_layout workspace;
    if (!ggml_cuda_gdn_cute_workspace_layout(args, workspace)) {
        GGML_ABORT("GDN CuTe workspace size overflow");
    }

    constexpr size_t alignment = 128;
    if (logical_size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        GGML_ABORT("GDN CuTe allocation size overflow");
    }
    const size_t workspace_offset = (logical_size + alignment - 1) & ~(alignment - 1);
    if (workspace_offset > std::numeric_limits<size_t>::max() - workspace.bytes) {
        GGML_ABORT("GDN CuTe allocation size overflow");
    }
    return workspace_offset + workspace.bytes;
}

bool ggml_cuda_gdn_cute_launch(const ggml_cuda_gdn_cute_args & args, cudaStream_t stream) {
    if (args.workspace == nullptr || args.state_out == nullptr) {
        return false;
    }

    gdn_cute_workspace_layout workspace;
    if (!ggml_cuda_gdn_cute_workspace_layout(args, workspace) ||
        workspace.bytes > args.workspace_size) {
        return false;
    }

    tf32 * packed_q = static_cast<tf32 *>(args.workspace);
    tf32 * packed_k = packed_q + workspace.q_elems;
    tf32 * packed_k_low = packed_k + workspace.q_elems;
    const int blocks = (int) ((workspace.q_elems + 255) / 256);
    gdn_prepack_qk_f32_tf32<<<blocks, 256, 0, stream>>>(
        args.q, args.k, packed_q, packed_k, packed_k_low,
        args.H_k, args.n_tokens, args.n_seqs, args.rq3,
        args.sq1, args.sq2, args.sq3);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    const size_t smem = sizeof(GdnSharedStorage);
    gdn_cute_sm12x<<<args.n_seqs * args.H, GDN_THREADS, smem, stream>>>(
        packed_q, packed_k, packed_k_low, args.v, args.g, args.beta, args.state, args.dst, args.state_out,
        args.H, args.H_k, args.n_tokens, args.n_seqs,
        args.sb1, args.sb2, args.sb3, args.sv1, args.sv2, args.sv3, args.scale);
    return cudaGetLastError() == cudaSuccess;
}

#endif // GGML_CUDA_CUTLASS
