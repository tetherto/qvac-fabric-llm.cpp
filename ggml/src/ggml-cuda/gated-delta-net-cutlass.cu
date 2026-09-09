#ifdef GGML_CUDA_CUTLASS

#include "gated-delta-net-cutlass.cuh"

#include "common.cuh"
#include "gdn-cute-inverse.cuh"

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>

#include <cmath>
#include <limits>
#include <mutex>

namespace ggml_gdn_cute {

using namespace cute;

using bf16 = cutlass::bfloat16_t;

constexpr int GDN_D       = 128;
constexpr int GDN_CHUNK   = 64;
constexpr int GDN_THREADS = 384;

template <int Id, int Threads>
CUTE_DEVICE void named_barrier() {
    asm volatile("bar.sync %0, %1;" :: "n"(Id), "n"(Threads) : "memory");
}

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using WarpMma = decltype(make_tiled_mma(
    MmaAtom{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{}));

template <class Accumulator>
CUTE_DEVICE auto accumulator_to_bf16(const Accumulator & acc, const WarpMma & mma) {
    constexpr auto c_frag_atom_size = size<0>(typename Accumulator::layout_type{});
    constexpr auto a_frag_atom_size = size<1>(typename WarpMma::AtomLayoutA_TV{});
    static_assert(a_frag_atom_size % c_frag_atom_size == 0);
    constexpr auto ratio = a_frag_atom_size / c_frag_atom_size;
    constexpr auto c_layout = typename Accumulator::layout_type{};
    constexpr auto operand_layout = [] {
        if constexpr (ratio == 1) {
            return c_layout;
        } else {
            constexpr auto divided = logical_divide(c_layout, make_shape(_, _, Int<ratio>{}));
            return make_layout(
                flatten(make_layout(get<0>(divided), get<2, 0>(divided))),
                get<1>(divided), get<2, 1>(divided));
        }
    }();
    Tensor operand = make_fragment_like<bf16>(operand_layout);
    Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());
    copy(acc, operand_as_acc);
    return operand;
}

struct alignas(128) GdnSharedStorage {
    bf16 q[GDN_CHUNK * GDN_D];
    bf16 k[GDN_CHUNK * GDN_D];
    bf16 v[GDN_CHUNK * GDN_D];
    bf16 qk[GDN_CHUNK * GDN_CHUNK];
    bf16 correction[GDN_CHUNK * GDN_CHUNK];
    bf16 new_v[GDN_D * GDN_CHUNK];
    cutlass::half_t inverse[GDN_CHUNK * GDN_CHUNK];
    float prefix[GDN_CHUNK];
    float beta[GDN_CHUNK];
};

static_assert(sizeof(GdnSharedStorage) <= 101376);

__global__ void gdn_prepack_f32_bf16(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        bf16 * __restrict__ packed_q,
        bf16 * __restrict__ packed_k,
        bf16 * __restrict__ packed_v,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t q_elems = n_seqs * n_tokens * H_k * GDN_D;
    const int64_t v_elems = n_seqs * n_tokens * H * GDN_D;

    if (index < q_elems) {
        const int d = index % GDN_D;
        const int64_t row = index / GDN_D;
        const int h = row % H_k;
        const int64_t token_seq = row / H_k;
        const int t = token_seq % n_tokens;
        const int seq = token_seq / n_tokens;
        const int iq3 = seq / rq3;
        const int64_t src = iq3 * sq3 + t * sq2 + h * sq1 + d;
        packed_q[index] = bf16(q[src]);
        packed_k[index] = bf16(k[src]);
    }
    if (index < v_elems) {
        const int d = index % GDN_D;
        const int64_t row = index / GDN_D;
        const int h = row % H;
        const int64_t token_seq = row / H;
        const int t = token_seq % n_tokens;
        const int seq = token_seq / n_tokens;
        const int64_t src = seq * sv3 + t * sv2 + h * sv1 + d;
        packed_v[index] = bf16(v[src]);
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
CUTE_DEVICE void store_matrix_fragment_bf16(
        const TensorC & frag, bf16 * dst, int ld, int row_base, int col_base,
        int valid_rows, int valid_cols, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const auto c = coord(i);
        const int r = row_base + get<0>(c);
        const int col = col_base + get<1>(c);
        if (r < valid_rows && col < valid_cols) {
            dst[r * ld + col] = bf16(frag(i));
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
        TensorC & frag, const bf16 * v, int value_base, int token_base,
        int valid_tokens, const WarpMma & mma) {
    auto thr = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const int value = value_base + get<0>(coord(i));
        const int token = token_base + get<1>(coord(i));
        const float vv = token < valid_tokens ? float(v[token * GDN_D + value]) : 0.0f;
        frag(i) = vv - frag(i);
    }
}

template <class TensorC>
CUTE_DEVICE void scale_accumulator(TensorC & frag, float scale) {
    for (int i = 0; i < size(frag); ++i) {
        frag(i) *= scale;
    }
}

__global__ __launch_bounds__(GDN_THREADS, 1) void gdn_cute_sm12x(
        const bf16 * __restrict__ packed_q,
        const bf16 * __restrict__ packed_k,
        const bf16 * __restrict__ packed_v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ state_in,
        float * __restrict__ dst,
        float * __restrict__ state_out,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sb1, int64_t sb2, int64_t sb3, float output_scale) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto & smem = *reinterpret_cast<GdnSharedStorage *>(smem_raw);

    const int seq = blockIdx.x / H;
    const int vh = blockIdx.x % H;
    const int kh = vh % H_k;
    const int math_tid = threadIdx.x - 128;
    const int math_warp = math_tid >> 5;
    const int lane = threadIdx.x & 31;
    const int value_base = math_warp * 16;
    const int64_t state_offset = ((int64_t) seq * H + vh) * GDN_D * GDN_D;
    float * out = dst + ((int64_t) seq * n_tokens * H + vh) * GDN_D;

    WarpMma mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto state_frag = partition_fragment_C(mma, Shape<_16, _128>{});
    if (threadIdx.x >= 128) {
        load_state_fragment(state_frag, state_in + state_offset, value_base, mma);
    }

    const int64_t q_head_offset = (((int64_t) seq * n_tokens) * H_k + kh) * GDN_D;
    const int64_t v_head_offset = (((int64_t) seq * n_tokens) * H + vh) * GDN_D;

    for (int64_t chunk = 0; chunk < n_tokens; chunk += GDN_CHUNK) {
        const int valid = min((int64_t) GDN_CHUNK, n_tokens - chunk);
        float * chunk_out = out + chunk * H * GDN_D;

        if (threadIdx.x < 128) {
            constexpr int vectors_per_row = GDN_D * sizeof(bf16) / sizeof(int4);
            for (int i = threadIdx.x; i < GDN_CHUNK * vectors_per_row; i += 128) {
                const int t = i / vectors_per_row;
                const int vector = i % vectors_per_row;
                int4 * sq = reinterpret_cast<int4 *>(smem.q + t * GDN_D) + vector;
                int4 * sk = reinterpret_cast<int4 *>(smem.k + t * GDN_D) + vector;
                int4 * sv = reinterpret_cast<int4 *>(smem.v + t * GDN_D) + vector;
                if (t < valid) {
                    const int64_t qoff = q_head_offset + (chunk + t) * H_k * GDN_D;
                    const int64_t voff = v_head_offset + (chunk + t) * H * GDN_D;
                    *sq = reinterpret_cast<const int4 *>(packed_q + qoff)[vector];
                    *sk = reinterpret_cast<const int4 *>(packed_k + qoff)[vector];
                    *sv = reinterpret_cast<const int4 *>(packed_v + voff)[vector];
                } else {
                    *sq = make_int4(0, 0, 0, 0);
                    *sk = make_int4(0, 0, 0, 0);
                    *sv = make_int4(0, 0, 0, 0);
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

        auto sQ = make_tensor(make_smem_ptr(smem.q), make_layout(Shape<_64, _128>{}, LayoutRight{}));
        auto sK = make_tensor(make_smem_ptr(smem.k), make_layout(Shape<_64, _128>{}, LayoutRight{}));

        if (threadIdx.x >= 128 && threadIdx.x < 256) {
            const int warp = (threadIdx.x - 128) >> 5;
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
        } else if (threadIdx.x >= 256) {
            const int warp = (threadIdx.x - 256) >> 5;
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
                    smem.qk[r * GDN_CHUNK + c] = bf16(x);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x >= 128 && threadIdx.x < 256) {
            auto inverse = make_tensor(make_smem_ptr(smem.inverse),
                make_layout(Shape<_64, _64>{}, LayoutRight{}));
            flat::collective::CollectiveInverse<cutlass::half_t, true, true>(3).compute(inverse);
        }
        __syncthreads();

        if (threadIdx.x < GDN_CHUNK) {
            const int col = threadIdx.x;
            for (int row = 0; row < valid; ++row) {
                smem.correction[row * GDN_CHUNK + col] =
                    col <= row ? bf16(float(smem.inverse[row * GDN_CHUNK + col]) * smem.beta[col]) : bf16(0.0f);
            }
        }
        __syncthreads();

        if (threadIdx.x >= 128) {
            auto state_op = accumulator_to_bf16(state_frag, mma);

            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto q_tile = local_tile(sQ, Shape<_16, _128>{}, make_coord(token_base / 16, 0));
                auto k_tile = local_tile(sK, Shape<_16, _128>{}, make_coord(token_base / 16, 0));
                auto rq = thr_mma.partition_fragment_B(q_tile);
                auto rk = thr_mma.partition_fragment_B(k_tile);
                copy(thr_mma.partition_B(q_tile), rq);
                copy(thr_mma.partition_B(k_tile), rk);

                auto o1 = partition_fragment_C(mma, Shape<_16, _16>{});
                auto sk = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(o1);
                clear(sk);
                gemm(mma, state_op, rq, o1);
                gemm(mma, state_op, rk, sk);

                apply_token_prefix_scale(o1, smem.prefix, token_base, output_scale, mma);
                apply_token_prefix_scale(sk, smem.prefix, token_base, 1.0f, mma);
                subtract_v(sk, smem.v, value_base, token_base, valid, mma);
                store_matrix_fragment_bf16(sk, smem.new_v, GDN_CHUNK,
                    value_base, token_base, GDN_D, valid, mma);
                store_output_fragment(o1, chunk_out, H * GDN_D, value_base,
                    token_base, valid, mma);
            }
        }
        __syncthreads();

        if (threadIdx.x >= 128) {
            auto sNewV = make_tensor(make_smem_ptr(smem.new_v), make_layout(Shape<_128, _64>{}, LayoutRight{}));
            auto sCorrectionT = make_tensor(make_smem_ptr(smem.correction),
                make_layout(Shape<_64, _64>{}, LayoutRight{}));
            auto sQKT = make_tensor(make_smem_ptr(smem.qk),
                make_layout(Shape<_64, _64>{}, LayoutRight{}));
            auto sKT = make_tensor(make_smem_ptr(smem.k),
                make_layout(Shape<_128, _64>{}, Stride<_1, _128>{}));

            auto delta_base = local_tile(sNewV, Shape<_16, _64>{}, make_coord(math_warp, 0));
            auto ra = thr_mma.partition_fragment_A(delta_base);
            copy(thr_mma.partition_A(delta_base), ra);
            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto corr = local_tile(sCorrectionT, Shape<_16, _64>{}, make_coord(token_base / 16, 0));
                auto rb = thr_mma.partition_fragment_B(corr);
                copy(thr_mma.partition_B(corr), rb);
                auto newv_tile = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(newv_tile);
                gemm(mma, ra, rb, newv_tile);
                store_matrix_fragment_bf16(newv_tile, smem.new_v, GDN_CHUNK,
                    value_base, token_base, GDN_D, valid, mma);
            }
            named_barrier<1, 256>();

            auto newv_all = local_tile(sNewV, Shape<_16, _64>{}, make_coord(math_warp, 0));
            auto rnewv = thr_mma.partition_fragment_A(newv_all);
            copy(thr_mma.partition_A(newv_all), rnewv);
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
            auto weighted_newv = make_fragment_like<bf16>(rnewv);
            auto coord_a = thr_mma.partition_A(make_identity_tensor(Shape<_16, _64>{}));
            for (int i = 0; i < size(rnewv); ++i) {
                const int tok = get<1>(coord_a(i));
                const float w = tok < valid ? expf(smem.prefix[valid - 1] - smem.prefix[tok]) : 0.0f;
                weighted_newv(i) = bf16(float(rnewv(i)) * w);
            }
            gemm(mma, weighted_newv, rk, state_frag);
        }
        __syncthreads();
    }

    if (threadIdx.x >= 128) {
        store_state_fragment(state_frag, state_out + state_offset, value_base, mma);
    }
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
    if (!args.eligible || args.H != 48 || args.H_k != 16 ||
        args.n_tokens < GDN_CHUNK || args.n_seqs <= 0) {
        return false;
    }
    return ggml_cuda_gdn_cute_init(device);
}

struct gdn_cute_workspace_layout {
    size_t q_elems;
    size_t v_elems;
    size_t bytes;
};

static bool gdn_cute_size_mul(size_t & value, size_t factor) {
    if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor) {
        return false;
    }
    value *= factor;
    return true;
}

static bool gdn_cute_size_add(size_t & value, size_t increment) {
    if (value > std::numeric_limits<size_t>::max() - increment) {
        return false;
    }
    value += increment;
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

    size_t v_elems = (size_t) args.n_seqs;
    if (!gdn_cute_size_mul(v_elems, (size_t) args.n_tokens) ||
        !gdn_cute_size_mul(v_elems, (size_t) args.H) ||
        !gdn_cute_size_mul(v_elems, GDN_D)) {
        return false;
    }

    size_t total_elems = q_elems;
    if (!gdn_cute_size_add(total_elems, q_elems) ||
        !gdn_cute_size_add(total_elems, v_elems) ||
        !gdn_cute_size_mul(total_elems, sizeof(bf16))) {
        return false;
    }

    layout = { q_elems, v_elems, total_elems };
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

    bf16 * packed_q = static_cast<bf16 *>(args.workspace);
    bf16 * packed_k = packed_q + workspace.q_elems;
    bf16 * packed_v = packed_k + workspace.q_elems;
    const int blocks = (int) ((max(workspace.q_elems, workspace.v_elems) + 255) / 256);
    gdn_prepack_f32_bf16<<<blocks, 256, 0, stream>>>(
        args.q, args.k, args.v, packed_q, packed_k, packed_v,
        args.H, args.H_k, args.n_tokens, args.n_seqs, args.rq3,
        args.sq1, args.sq2, args.sq3, args.sv1, args.sv2, args.sv3);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    const size_t smem = sizeof(GdnSharedStorage);
    gdn_cute_sm12x<<<args.n_seqs * args.H, GDN_THREADS, smem, stream>>>(
        packed_q, packed_k, packed_v, args.g, args.beta, args.state, args.dst, args.state_out,
        args.H, args.H_k, args.n_tokens, args.n_seqs,
        args.sb1, args.sb2, args.sb3, args.scale);
    return cudaGetLastError() == cudaSuccess;
}

#endif // GGML_CUDA_CUTLASS
