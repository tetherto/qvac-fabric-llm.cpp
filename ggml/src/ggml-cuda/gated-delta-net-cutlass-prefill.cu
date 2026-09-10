#ifdef GGML_CUDA_CUTLASS

#    include "common.cuh"
#    include "gated-delta-net-cutlass-prefill.cuh"
#    include "gdn-cute-inverse.cuh"

#    include <cutlass/bfloat16.h>
#    include <cutlass/cutlass.h>
#    include <cutlass/numeric_conversion.h>

#    include <algorithm>
#    include <climits>
#    include <cmath>
#    include <cute/tensor.hpp>
#    include <limits>
#    include <mutex>

namespace ggml_gdn_chunked {

using namespace cute;

using bf16 = cutlass::bfloat16_t;

constexpr int GDN_D             = 128;
constexpr int GDN_CHUNK         = 64;
constexpr int GDN_THREADS       = 256;
constexpr int GDN_STATE_THREADS = 128;
constexpr int GDN_STATE_ROWS    = 64;
constexpr int GDN_OUTPUT_ROWS   = 128;

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using WarpMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{}));

CUTE_DEVICE bf16 to_bf16(float value) {
    return bf16(value);
}

template <class Accumulator> CUTE_DEVICE auto accumulator_to_bf16(const Accumulator & acc, const WarpMma & mma) {
    constexpr auto c_frag_atom_size = size<0>(typename Accumulator::layout_type{});
    constexpr auto a_frag_atom_size = size<1>(typename WarpMma::AtomLayoutA_TV{});
    static_assert(a_frag_atom_size % c_frag_atom_size == 0);
    constexpr auto ratio          = a_frag_atom_size / c_frag_atom_size;
    constexpr auto c_layout       = typename Accumulator::layout_type{};
    constexpr auto operand_layout = [] {
        if constexpr (ratio == 1) {
            return c_layout;
        } else {
            constexpr auto divided = logical_divide(c_layout, make_shape(_, _, Int<ratio>{}));
            return make_layout(flatten(make_layout(get<0>(divided), get<2, 0>(divided))), get<1>(divided),
                               get<2, 1>(divided));
        }
    }();
    Tensor operand        = make_fragment_like<bf16>(operand_layout);
    Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());
    copy(acc, operand_as_acc);
    return operand;
}

template <class TensorC>
CUTE_DEVICE void load_state_fragment(TensorC & state_frag, const float * state, int value_base, const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c  = coord(i);
        state_frag(i) = state[(value_base + get<0>(c)) * GDN_D + get<1>(c)];
    }
}

template <class TensorC>
CUTE_DEVICE void store_state_fragment(const TensorC & state_frag, float * state, int value_base, const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c                                        = coord(i);
        state[(value_base + get<0>(c)) * GDN_D + get<1>(c)] = state_frag(i);
    }
}

template <class TensorC>
CUTE_DEVICE void store_state_fragment_bf16(const TensorC & state_frag,
                                           bf16 *          state,
                                           int             value_base,
                                           const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c                                        = coord(i);
        state[(value_base + get<0>(c)) * GDN_D + get<1>(c)] = to_bf16(state_frag(i));
    }
}

template <class TensorC>
CUTE_DEVICE void load_state_fragment_bf16(TensorC &       state_frag,
                                          const bf16 *    state,
                                          int             value_base,
                                          const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _128>{}));
    for (int i = 0; i < size(state_frag); ++i) {
        const auto c  = coord(i);
        state_frag(i) = float(state[(value_base + get<0>(c)) * GDN_D + get<1>(c)]);
    }
}

template <class TensorC>
CUTE_DEVICE void store_matrix_bf16(const TensorC & frag,
                                   bf16 *          dst,
                                   int64_t         ld,
                                   int             row_base,
                                   int             col_base,
                                   int             valid_rows,
                                   const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const int row                 = row_base + get<0>(coord(i));
        const int col                 = col_base + get<1>(coord(i));
        dst[(int64_t) row * ld + col] = row < valid_rows ? to_bf16(frag(i)) : to_bf16(0.0f);
    }
}

template <class TensorC>
CUTE_DEVICE void store_output(const TensorC & frag,
                              float *         dst,
                              int64_t         token_stride,
                              int             value_base,
                              int             token_base,
                              int             valid_tokens,
                              bool            add,
                              const WarpMma & mma) {
    auto thr   = mma.get_thread_slice(threadIdx.x & 31);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    for (int i = 0; i < size(frag); ++i) {
        const int value = value_base + get<0>(coord(i));
        const int token = token_base + get<1>(coord(i));
        if (token < valid_tokens) {
            float * out = dst + (int64_t) token * token_stride + value;
            if (add) {
                *out += frag(i);
            } else {
                *out = frag(i);
            }
        }
    }
}

template <class TensorC> CUTE_DEVICE void scale_accumulator(TensorC & frag, float scale) {
    for (int i = 0; i < size(frag); ++i) {
        frag(i) *= scale;
    }
}

struct alignas(128) KktSharedStorage {
    bf16            k[GDN_CHUNK * GDN_D];
    cutlass::half_t inverse[GDN_CHUNK * GDN_CHUNK];
    float           prefix[GDN_CHUNK];
    float           beta[GDN_CHUNK];
};

struct alignas(128) WuSharedStorage {
    bf16  factor[GDN_CHUNK * GDN_CHUNK];
    bf16  operand[GDN_D * GDN_CHUNK];
    float prefix[GDN_CHUNK];
    float beta[GDN_CHUNK];
};

struct alignas(128) StateSharedStorage {
    union {
        bf16 w[GDN_CHUNK * GDN_D];
        bf16 k[GDN_D * GDN_CHUNK];
    };

    bf16  new_v[GDN_STATE_ROWS * GDN_CHUNK];
    float prefix[GDN_CHUNK];
};

struct alignas(128) OutputSharedStorage {
    union {
        bf16 q[GDN_CHUNK * GDN_D];
        bf16 new_v[GDN_OUTPUT_ROWS * GDN_CHUNK];
    };

    bf16  k[GDN_CHUNK * GDN_D];
    bf16  qk[GDN_CHUNK * GDN_CHUNK];
    float prefix[GDN_CHUNK];
};

static_assert(sizeof(KktSharedStorage) <= 101376);
static_assert(sizeof(WuSharedStorage) <= 101376);
static_assert(sizeof(StateSharedStorage) <= 101376);
static_assert(sizeof(OutputSharedStorage) <= 101376);

CUTE_DEVICE void decode_head_chunk(int block, int n_chunks, int64_t H, int & seq, int & vh, int & chunk) {
    chunk = block % n_chunks;
    block /= n_chunks;
    vh  = block % H;
    seq = block / H;
}

__global__ void gdn_chunk_kkt_solve_sm120(const float * __restrict__ k,
                                          const float * __restrict__ g,
                                          const float * __restrict__ beta,
                                          bf16 * __restrict__ factors,
                                          int64_t H,
                                          int64_t H_k,
                                          int64_t n_tokens,
                                          int     n_chunks,
                                          int64_t rq3,
                                          int64_t sq1,
                                          int64_t sq2,
                                          int64_t sq3,
                                          int64_t sb1,
                                          int64_t sb2,
                                          int64_t sb3) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto &                                         smem = *reinterpret_cast<KktSharedStorage *>(smem_raw);

    int seq;
    int vh;
    int chunk_id;
    decode_head_chunk(blockIdx.x, n_chunks, H, seq, vh, chunk_id);
    const int kh          = vh % H_k;
    const int token_start = chunk_id * GDN_CHUNK;
    const int valid       = min(GDN_CHUNK, (int) (n_tokens - token_start));
    const int iq3         = seq / rq3;

    for (int i = threadIdx.x; i < GDN_CHUNK * GDN_D; i += blockDim.x) {
        const int token = i / GDN_D;
        const int d     = i % GDN_D;
        if (token < valid) {
            const int64_t src = (int64_t) iq3 * sq3 + (token_start + token) * sq2 + kh * sq1 + d;
            smem.k[i]         = to_bf16(k[src]);
        } else {
            smem.k[i] = to_bf16(0.0f);
        }
    }
    if (threadIdx.x < GDN_CHUNK) {
        const int token = threadIdx.x;
        if (token < valid) {
            const int64_t src  = (int64_t) seq * sb3 + vh * sb1 + (token_start + token) * sb2;
            smem.prefix[token] = g[src];
            smem.beta[token]   = beta[src];
        } else {
            smem.prefix[token] = 0.0f;
            smem.beta[token]   = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int token = 0; token < GDN_CHUNK; ++token) {
            sum += smem.prefix[token];
            smem.prefix[token] = sum;
        }
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        WarpMma   mma;
        auto      thr_mma  = mma.get_thread_slice(threadIdx.x & 31);
        auto      sK       = make_tensor(make_smem_ptr(smem.k), make_layout(Shape<_64, _128>{}, LayoutRight{}));
        const int row_base = (threadIdx.x >> 5) * 16;
        for (int col_base = 0; col_base < GDN_CHUNK; col_base += 16) {
            auto a  = local_tile(sK, Shape<_16, _128>{}, make_coord(row_base / 16, 0));
            auto b  = local_tile(sK, Shape<_16, _128>{}, make_coord(col_base / 16, 0));
            auto ra = thr_mma.partition_fragment_A(a);
            auto rb = thr_mma.partition_fragment_B(b);
            copy(thr_mma.partition_A(a), ra);
            copy(thr_mma.partition_B(b), rb);
            auto rc = partition_fragment_C(mma, Shape<_16, _16>{});
            clear(rc);
            gemm(mma, ra, rb, rc);
            auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
            for (int i = 0; i < size(rc); ++i) {
                const int row   = row_base + get<0>(coord(i));
                const int col   = col_base + get<1>(coord(i));
                float     value = 0.0f;
                if (row < valid && col < valid) {
                    if (row == col) {
                        value = 1.0f;
                    } else if (row > col) {
                        value = rc(i) * smem.beta[row] * expf(smem.prefix[row] - smem.prefix[col]);
                    }
                }
                smem.inverse[row * GDN_CHUNK + col] = cutlass::half_t(value);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        auto inverse = make_tensor(make_smem_ptr(smem.inverse), make_layout(Shape<_64, _64>{}, LayoutRight{}));
        flat::collective::CollectiveInverse<cutlass::half_t, true, true>(3).compute(inverse);
    }
    __syncthreads();

    const int64_t record = ((int64_t) seq * H + vh) * n_chunks + chunk_id;
    bf16 *        dst    = factors + record * GDN_CHUNK * GDN_CHUNK;
    for (int i = threadIdx.x; i < GDN_CHUNK * GDN_CHUNK; i += blockDim.x) {
        dst[i] = to_bf16(float(smem.inverse[i]));
    }
}

__global__ void gdn_chunk_recompute_wu_sm120(const float * __restrict__ k,
                                             const float * __restrict__ v,
                                             const float * __restrict__ g,
                                             const float * __restrict__ beta,
                                             const bf16 * __restrict__ factors,
                                             bf16 * __restrict__ w,
                                             bf16 * __restrict__ u,
                                             int64_t H,
                                             int64_t H_k,
                                             int64_t n_tokens,
                                             int     n_chunks,
                                             int64_t rq3,
                                             int64_t sq1,
                                             int64_t sq2,
                                             int64_t sq3,
                                             int64_t sv1,
                                             int64_t sv2,
                                             int64_t sv3,
                                             int64_t sb1,
                                             int64_t sb2,
                                             int64_t sb3) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto &                                         smem = *reinterpret_cast<WuSharedStorage *>(smem_raw);

    int seq;
    int vh;
    int chunk_id;
    decode_head_chunk(blockIdx.x, n_chunks, H, seq, vh, chunk_id);
    const int     kh          = vh % H_k;
    const int     token_start = chunk_id * GDN_CHUNK;
    const int     valid       = min(GDN_CHUNK, (int) (n_tokens - token_start));
    const int     iq3         = seq / rq3;
    const int64_t record      = ((int64_t) seq * H + vh) * n_chunks + chunk_id;

    const bf16 * factor = factors + record * GDN_CHUNK * GDN_CHUNK;
    for (int i = threadIdx.x; i < GDN_CHUNK * GDN_CHUNK; i += blockDim.x) {
        smem.factor[i] = factor[i];
    }
    if (threadIdx.x < GDN_CHUNK) {
        const int token = threadIdx.x;
        if (token < valid) {
            const int64_t src  = (int64_t) seq * sb3 + vh * sb1 + (token_start + token) * sb2;
            smem.prefix[token] = g[src];
            smem.beta[token]   = beta[src];
        } else {
            smem.prefix[token] = 0.0f;
            smem.beta[token]   = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int token = 0; token < GDN_CHUNK; ++token) {
            sum += smem.prefix[token];
            smem.prefix[token] = sum;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < GDN_D * GDN_CHUNK; i += blockDim.x) {
        const int d     = i / GDN_CHUNK;
        const int token = i % GDN_CHUNK;
        if (token < valid) {
            const int64_t src = (int64_t) iq3 * sq3 + (token_start + token) * sq2 + kh * sq1 + d;
            const bf16    key = to_bf16(k[src]);
            smem.operand[i]   = to_bf16(float(key) * smem.beta[token] * expf(smem.prefix[token]));
        } else {
            smem.operand[i] = to_bf16(0.0f);
        }
    }
    __syncthreads();

    WarpMma   mma;
    auto      thr_mma  = mma.get_thread_slice(threadIdx.x & 31);
    auto      sA       = make_tensor(make_smem_ptr(smem.factor), make_layout(Shape<_64, _64>{}, LayoutRight{}));
    auto      sB       = make_tensor(make_smem_ptr(smem.operand), make_layout(Shape<_128, _64>{}, LayoutRight{}));
    const int warp     = threadIdx.x >> 5;
    const int row_base = (warp >> 1) * 16;
    const int col_half = (warp & 1) * 64;
    bf16 *    w_record = w + record * GDN_CHUNK * GDN_D;

    for (int col_base = col_half; col_base < col_half + 64; col_base += 16) {
        auto acc = partition_fragment_C(mma, Shape<_16, _16>{});
        clear(acc);
        for (int inner = 0; inner < GDN_CHUNK; inner += 16) {
            auto a  = local_tile(sA, Shape<_16, _16>{}, make_coord(row_base / 16, inner / 16));
            auto b  = local_tile(sB, Shape<_16, _16>{}, make_coord(col_base / 16, inner / 16));
            auto ra = thr_mma.partition_fragment_A(a);
            auto rb = thr_mma.partition_fragment_B(b);
            copy(thr_mma.partition_A(a), ra);
            copy(thr_mma.partition_B(b), rb);
            gemm(mma, ra, rb, acc);
        }
        store_matrix_bf16(acc, w_record, GDN_D, row_base, col_base, valid, mma);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < GDN_D * GDN_CHUNK; i += blockDim.x) {
        const int d     = i / GDN_CHUNK;
        const int token = i % GDN_CHUNK;
        if (token < valid) {
            const int64_t src = (int64_t) seq * sv3 + vh * sv1 + (token_start + token) * sv2 + d;
            smem.operand[i]   = to_bf16(v[src] * smem.beta[token]);
        } else {
            smem.operand[i] = to_bf16(0.0f);
        }
    }
    __syncthreads();

    bf16 * u_record = u + record * GDN_CHUNK * GDN_D;
    for (int col_base = col_half; col_base < col_half + 64; col_base += 16) {
        auto acc = partition_fragment_C(mma, Shape<_16, _16>{});
        clear(acc);
        for (int inner = 0; inner < GDN_CHUNK; inner += 16) {
            auto a  = local_tile(sA, Shape<_16, _16>{}, make_coord(row_base / 16, inner / 16));
            auto b  = local_tile(sB, Shape<_16, _16>{}, make_coord(col_base / 16, inner / 16));
            auto ra = thr_mma.partition_fragment_A(a);
            auto rb = thr_mma.partition_fragment_B(b);
            copy(thr_mma.partition_A(a), ra);
            copy(thr_mma.partition_B(b), rb);
            gemm(mma, ra, rb, acc);
        }
        store_matrix_bf16(acc, u_record, GDN_D, row_base, col_base, valid, mma);
    }
}

__global__ __launch_bounds__(GDN_STATE_THREADS, 2) void gdn_chunk_state_sm120(const float * __restrict__ k,
                                                                              const float * __restrict__ g,
                                                                              const float * __restrict__ state_in,
                                                                              bf16 * __restrict__ w,
                                                                              bf16 * __restrict__ u,
                                                                              bf16 * __restrict__ state_chunks,
                                                                              float * __restrict__ state_out,
                                                                              int64_t H,
                                                                              int64_t H_k,
                                                                              int64_t n_tokens,
                                                                              int     n_chunks,
                                                                              int64_t rq3,
                                                                              int64_t sq1,
                                                                              int64_t sq2,
                                                                              int64_t sq3,
                                                                              int64_t sb1,
                                                                              int64_t sb2,
                                                                              int64_t sb3) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto &                                         smem = *reinterpret_cast<StateSharedStorage *>(smem_raw);

    const int     value_tile      = blockIdx.x % (GDN_D / GDN_STATE_ROWS);
    int           head_block      = blockIdx.x / (GDN_D / GDN_STATE_ROWS);
    const int     vh              = head_block % H;
    const int     seq             = head_block / H;
    const int     kh              = vh % H_k;
    const int     iq3             = seq / rq3;
    const int     value_tile_base = value_tile * GDN_STATE_ROWS;
    const int     warp            = threadIdx.x >> 5;
    const bool    math_warp       = warp < GDN_STATE_ROWS / 16;
    const int     value_base      = value_tile_base + warp * 16;
    const int64_t state_offset    = ((int64_t) seq * H + vh) * GDN_D * GDN_D;

    WarpMma mma;
    auto    state_frag = partition_fragment_C(mma, Shape<_16, _128>{});
    if (math_warp) {
        load_state_fragment(state_frag, state_in + state_offset, value_base, mma);
    }

    for (int chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        const int     token_start  = chunk_id * GDN_CHUNK;
        const int     valid        = min(GDN_CHUNK, (int) (n_tokens - token_start));
        const int64_t record       = ((int64_t) seq * H + vh) * n_chunks + chunk_id;
        const bf16 *  w_record     = w + record * GDN_CHUNK * GDN_D;
        bf16 *        u_record     = u + record * GDN_CHUNK * GDN_D;
        bf16 *        state_record = state_chunks + record * GDN_D * GDN_D;

        if (math_warp) {
            store_state_fragment_bf16(state_frag, state_record, value_base, mma);
        }
        for (int i = threadIdx.x; i < GDN_CHUNK * GDN_D; i += blockDim.x) {
            smem.w[i] = w_record[i];
        }
        if (threadIdx.x < GDN_CHUNK) {
            const int token = threadIdx.x;
            if (token < valid) {
                const int64_t src  = (int64_t) seq * sb3 + vh * sb1 + (token_start + token) * sb2;
                smem.prefix[token] = g[src];
            } else {
                smem.prefix[token] = 0.0f;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int token = 0; token < GDN_CHUNK; ++token) {
                sum += smem.prefix[token];
                smem.prefix[token] = sum;
            }
        }
        __syncthreads();

        if (math_warp) {
            auto thr_mma  = mma.get_thread_slice(threadIdx.x & 31);
            auto state_op = accumulator_to_bf16(state_frag, mma);
            auto sW       = make_tensor(make_smem_ptr(smem.w), make_layout(Shape<_64, _128>{}, LayoutRight{}));

            for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
                auto wh = partition_fragment_C(mma, Shape<_16, _16>{});
                clear(wh);
                auto w_tile = local_tile(sW, Shape<_16, _128>{}, make_coord(token_base / 16, 0));
                auto rw     = thr_mma.partition_fragment_B(w_tile);
                copy(thr_mma.partition_B(w_tile), rw);
                gemm(mma, state_op, rw, wh);
                auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
                for (int i = 0; i < size(wh); ++i) {
                    const int value = get<0>(coord(i));
                    const int token = token_base + get<1>(coord(i));
                    float     new_v = 0.0f;
                    if (token < valid) {
                        new_v = float(u_record[(int64_t) token * GDN_D + value_base + value]) - wh(i);
                    }
                    const bf16 stored                                      = to_bf16(new_v);
                    u_record[(int64_t) token * GDN_D + value_base + value] = stored;
                    smem.new_v[(warp * 16 + value) * GDN_CHUNK + token]    = stored;
                }
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < GDN_D * GDN_CHUNK; i += blockDim.x) {
            const int key   = i / GDN_CHUNK;
            const int token = i % GDN_CHUNK;
            if (token < valid) {
                const int64_t src = (int64_t) iq3 * sq3 + (token_start + token) * sq2 + kh * sq1 + key;
                smem.k[i]         = to_bf16(k[src]);
            } else {
                smem.k[i] = to_bf16(0.0f);
            }
        }
        __syncthreads();

        if (math_warp) {
            auto thr_mma        = mma.get_thread_slice(threadIdx.x & 31);
            auto sNewV          = make_tensor(make_smem_ptr(smem.new_v), make_layout(Shape<_32, _64>{}, LayoutRight{}));
            auto sK             = make_tensor(make_smem_ptr(smem.k), make_layout(Shape<_128, _64>{}, LayoutRight{}));
            auto new_v_tile     = local_tile(sNewV, Shape<_16, _64>{}, make_coord(warp, 0));
            auto weighted_new_v = thr_mma.partition_fragment_A(new_v_tile);
            copy(thr_mma.partition_A(new_v_tile), weighted_new_v);
            auto        coord = thr_mma.partition_A(make_identity_tensor(Shape<_16, _64>{}));
            const float last  = smem.prefix[valid - 1];
            for (int i = 0; i < size(weighted_new_v); ++i) {
                const int token   = get<1>(coord(i));
                weighted_new_v(i) = to_bf16(float(weighted_new_v(i)) * expf(last - smem.prefix[token]));
            }
            scale_accumulator(state_frag, expf(last));
            auto rk = thr_mma.partition_fragment_B(sK);
            copy(thr_mma.partition_B(sK), rk);
            gemm(mma, weighted_new_v, rk, state_frag);
        }
        __syncthreads();
    }

    if (math_warp) {
        store_state_fragment(state_frag, state_out + state_offset, value_base, mma);
    }
}

__global__ void gdn_chunk_output_sm120(const float * __restrict__ q,
                                       const float * __restrict__ k,
                                       const float * __restrict__ g,
                                       const bf16 * __restrict__ new_v,
                                       const bf16 * __restrict__ state_chunks,
                                       float * __restrict__ dst,
                                       int64_t H,
                                       int64_t H_k,
                                       int64_t n_tokens,
                                       int     n_chunks,
                                       int64_t rq3,
                                       int64_t sq1,
                                       int64_t sq2,
                                       int64_t sq3,
                                       int64_t sb1,
                                       int64_t sb2,
                                       int64_t sb3,
                                       float   output_scale) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto &                                         smem = *reinterpret_cast<OutputSharedStorage *>(smem_raw);

    const int value_tile = blockIdx.x % (GDN_D / GDN_OUTPUT_ROWS);
    int       head_chunk = blockIdx.x / (GDN_D / GDN_OUTPUT_ROWS);
    int       seq;
    int       vh;
    int       chunk_id;
    decode_head_chunk(head_chunk, n_chunks, H, seq, vh, chunk_id);
    const int     kh              = vh % H_k;
    const int     iq3             = seq / rq3;
    const int     token_start     = chunk_id * GDN_CHUNK;
    const int     valid           = min(GDN_CHUNK, (int) (n_tokens - token_start));
    const int     value_tile_base = value_tile * GDN_OUTPUT_ROWS;
    const int64_t record          = ((int64_t) seq * H + vh) * n_chunks + chunk_id;

    for (int i = threadIdx.x; i < GDN_CHUNK * GDN_D; i += blockDim.x) {
        const int token = i / GDN_D;
        const int d     = i % GDN_D;
        if (token < valid) {
            const int64_t src = (int64_t) iq3 * sq3 + (token_start + token) * sq2 + kh * sq1 + d;
            smem.q[i]         = to_bf16(q[src]);
            smem.k[i]         = to_bf16(k[src]);
        } else {
            smem.q[i] = to_bf16(0.0f);
            smem.k[i] = to_bf16(0.0f);
        }
    }
    if (threadIdx.x < GDN_CHUNK) {
        const int token = threadIdx.x;
        if (token < valid) {
            const int64_t src  = (int64_t) seq * sb3 + vh * sb1 + (token_start + token) * sb2;
            smem.prefix[token] = g[src];
        } else {
            smem.prefix[token] = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int token = 0; token < GDN_CHUNK; ++token) {
            sum += smem.prefix[token];
            smem.prefix[token] = sum;
        }
    }
    __syncthreads();

    WarpMma   mma;
    auto      thr_mma = mma.get_thread_slice(threadIdx.x & 31);
    const int warp    = threadIdx.x >> 5;
    auto      sQ      = make_tensor(make_smem_ptr(smem.q), make_layout(Shape<_64, _128>{}, LayoutRight{}));
    auto      sK      = make_tensor(make_smem_ptr(smem.k), make_layout(Shape<_64, _128>{}, LayoutRight{}));

    {
        const int    value_base   = value_tile_base + warp * 16;
        const bf16 * state_record = state_chunks + record * GDN_D * GDN_D;
        auto         state_frag   = partition_fragment_C(mma, Shape<_16, _128>{});
        load_state_fragment_bf16(state_frag, state_record, value_base, mma);
        auto    state_op = accumulator_to_bf16(state_frag, mma);
        float * out      = dst + ((int64_t) seq * n_tokens * H + vh) * GDN_D + token_start * H * GDN_D;

        for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
            auto o1 = partition_fragment_C(mma, Shape<_16, _16>{});
            clear(o1);
            auto q_tile = local_tile(sQ, Shape<_16, _128>{}, make_coord(token_base / 16, 0));
            auto rq     = thr_mma.partition_fragment_B(q_tile);
            copy(thr_mma.partition_B(q_tile), rq);
            gemm(mma, state_op, rq, o1);
            auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
            for (int i = 0; i < size(o1); ++i) {
                o1(i) *= output_scale * expf(smem.prefix[token_base + get<1>(coord(i))]);
            }
            store_output(o1, out, H * GDN_D, value_base, token_base, valid, false, mma);
        }
    }
    __syncthreads();

    if (warp < 4) {
        const int row_base = warp * 16;
        for (int col_base = 0; col_base < GDN_CHUNK; col_base += 16) {
            auto a  = local_tile(sQ, Shape<_16, _128>{}, make_coord(row_base / 16, 0));
            auto b  = local_tile(sK, Shape<_16, _128>{}, make_coord(col_base / 16, 0));
            auto ra = thr_mma.partition_fragment_A(a);
            auto rb = thr_mma.partition_fragment_B(b);
            copy(thr_mma.partition_A(a), ra);
            copy(thr_mma.partition_B(b), rb);
            auto qk = partition_fragment_C(mma, Shape<_16, _16>{});
            clear(qk);
            gemm(mma, ra, rb, qk);
            auto coord = thr_mma.partition_C(make_identity_tensor(Shape<_16, _16>{}));
            for (int i = 0; i < size(qk); ++i) {
                const int row   = row_base + get<0>(coord(i));
                const int col   = col_base + get<1>(coord(i));
                float     value = 0.0f;
                if (row < valid && col < valid && row >= col) {
                    value = qk(i) * output_scale * expf(smem.prefix[row] - smem.prefix[col]);
                }
                smem.qk[row * GDN_CHUNK + col] = to_bf16(value);
            }
        }
    }
    __syncthreads();

    const bf16 * new_v_record = new_v + record * GDN_CHUNK * GDN_D;
    for (int i = threadIdx.x; i < GDN_OUTPUT_ROWS * GDN_CHUNK; i += blockDim.x) {
        const int value = i / GDN_CHUNK;
        const int token = i % GDN_CHUNK;
        smem.new_v[i] = token < valid ? new_v_record[(int64_t) token * GDN_D + value_tile_base + value] : to_bf16(0.0f);
    }
    __syncthreads();

    {
        const int value_base = value_tile_base + warp * 16;
        auto      sNewV      = make_tensor(make_smem_ptr(smem.new_v), make_layout(Shape<_128, _64>{}, LayoutRight{}));
        auto      sQK        = make_tensor(make_smem_ptr(smem.qk), make_layout(Shape<_64, _64>{}, LayoutRight{}));
        auto      new_v_tile = local_tile(sNewV, Shape<_16, _64>{}, make_coord(warp, 0));
        auto      rnew_v     = thr_mma.partition_fragment_A(new_v_tile);
        copy(thr_mma.partition_A(new_v_tile), rnew_v);
        float * out = dst + ((int64_t) seq * n_tokens * H + vh) * GDN_D + token_start * H * GDN_D;

        for (int token_base = 0; token_base < GDN_CHUNK; token_base += 16) {
            auto p  = local_tile(sQK, Shape<_16, _64>{}, make_coord(token_base / 16, 0));
            auto rp = thr_mma.partition_fragment_B(p);
            copy(thr_mma.partition_B(p), rp);
            auto o2 = partition_fragment_C(mma, Shape<_16, _16>{});
            clear(o2);
            gemm(mma, rnew_v, rp, o2);
            store_output(o2, out, H * GDN_D, value_base, token_base, valid, true, mma);
        }
    }
}

bool size_mul(size_t & value, size_t factor) {
    if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor) {
        return false;
    }
    value *= factor;
    return true;
}

bool size_add(size_t & value, size_t increment) {
    if (value > std::numeric_limits<size_t>::max() - increment) {
        return false;
    }
    value += increment;
    return true;
}

}  // namespace ggml_gdn_chunked

using namespace ggml_gdn_chunked;

bool ggml_cuda_gdn_chunked_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};

    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        const auto & info = ggml_cuda_info().devices[device];
        if (info.cc != GGML_CUDA_CC_BLACKWELL || sizeof(KktSharedStorage) > info.smpbo ||
            sizeof(WuSharedStorage) > info.smpbo || sizeof(StateSharedStorage) > info.smpbo ||
            sizeof(OutputSharedStorage) > info.smpbo) {
            return;
        }
        cudaError_t status = cudaFuncSetAttribute(
            gdn_chunk_kkt_solve_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(KktSharedStorage));
        if (status == cudaSuccess) {
            status = cudaFuncSetAttribute(gdn_chunk_recompute_wu_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int) sizeof(WuSharedStorage));
        }
        if (status == cudaSuccess) {
            status = cudaFuncSetAttribute(gdn_chunk_state_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int) sizeof(StateSharedStorage));
        }
        if (status == cudaSuccess) {
            status = cudaFuncSetAttribute(gdn_chunk_output_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int) sizeof(OutputSharedStorage));
        }
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

bool ggml_cuda_gdn_chunked_workspace_layout(const ggml_cuda_gdn_cute_args &   args,
                                            ggml_cuda_gdn_chunked_workspace & workspace) {
    if (args.n_tokens <= 0 || args.n_seqs <= 0 || args.H <= 0) {
        return false;
    }
    const size_t n_chunks = ((size_t) args.n_tokens + GDN_CHUNK - 1) / GDN_CHUNK;
    size_t       records  = (size_t) args.n_seqs;
    if (!size_mul(records, (size_t) args.H) || !size_mul(records, n_chunks) ||
        records > INT_MAX / (GDN_D / GDN_OUTPUT_ROWS)) {
        return false;
    }

    size_t factor_bytes = records;
    size_t state_bytes  = records;
    size_t matrix_bytes = records;
    if (!size_mul(factor_bytes, GDN_CHUNK * GDN_CHUNK * sizeof(bf16)) ||
        !size_mul(state_bytes, GDN_D * GDN_D * sizeof(bf16)) ||
        !size_mul(matrix_bytes, GDN_CHUNK * GDN_D * sizeof(bf16))) {
        return false;
    }

    // The state kernel starts after all factor consumers complete.
    const size_t state_region = std::max(factor_bytes, state_bytes);
    size_t       total        = state_region;
    const size_t w_offset     = total;
    if (!size_add(total, matrix_bytes)) {
        return false;
    }
    const size_t u_offset = total;
    if (!size_add(total, matrix_bytes)) {
        return false;
    }
    workspace = { n_chunks, w_offset, u_offset, total };
    return true;
}

bool ggml_cuda_gdn_chunked_launch(const ggml_cuda_gdn_cute_args &         args,
                                  const ggml_cuda_gdn_chunked_workspace & workspace,
                                  cudaStream_t                            stream) {
    if (args.workspace == nullptr || args.state_out == nullptr || workspace.bytes > args.workspace_size) {
        return false;
    }

    char *    base         = static_cast<char *>(args.workspace);
    auto *    factors      = reinterpret_cast<bf16 *>(base);
    auto *    state_chunks = reinterpret_cast<bf16 *>(base);
    auto *    w            = reinterpret_cast<bf16 *>(base + workspace.w_offset);
    auto *    u            = reinterpret_cast<bf16 *>(base + workspace.u_offset);
    const int n_chunks     = (int) workspace.n_chunks;
    const int head_chunks  = (int) (args.n_seqs * args.H * n_chunks);

    gdn_chunk_kkt_solve_sm120<<<head_chunks, GDN_THREADS, sizeof(KktSharedStorage), stream>>>(
        args.k, args.g, args.beta, factors, args.H, args.H_k, args.n_tokens, n_chunks, args.rq3, args.sq1, args.sq2,
        args.sq3, args.sb1, args.sb2, args.sb3);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    gdn_chunk_recompute_wu_sm120<<<head_chunks, GDN_THREADS, sizeof(WuSharedStorage), stream>>>(
        args.k, args.v, args.g, args.beta, factors, w, u, args.H, args.H_k, args.n_tokens, n_chunks, args.rq3, args.sq1,
        args.sq2, args.sq3, args.sv1, args.sv2, args.sv3, args.sb1, args.sb2, args.sb3);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    const int state_blocks = (int) (args.n_seqs * args.H * (GDN_D / GDN_STATE_ROWS));
    gdn_chunk_state_sm120<<<state_blocks, GDN_STATE_THREADS, sizeof(StateSharedStorage), stream>>>(
        args.k, args.g, args.state, w, u, state_chunks, args.state_out, args.H, args.H_k, args.n_tokens, n_chunks,
        args.rq3, args.sq1, args.sq2, args.sq3, args.sb1, args.sb2, args.sb3);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    const int output_blocks = head_chunks * (GDN_D / GDN_OUTPUT_ROWS);
    gdn_chunk_output_sm120<<<output_blocks, GDN_THREADS, sizeof(OutputSharedStorage), stream>>>(
        args.q, args.k, args.g, u, state_chunks, args.dst, args.H, args.H_k, args.n_tokens, n_chunks, args.rq3,
        args.sq1, args.sq2, args.sq3, args.sb1, args.sb2, args.sb3, args.scale);
    return cudaGetLastError() == cudaSuccess;
}

#endif  // GGML_CUDA_CUTLASS
