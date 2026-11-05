// The row command of the merged prefill design's row-op columns (kernels/pfx.py): R1 over a block of 8 positions of
// one head, the RMS norm times its weight and then neox RoPE, every product a bf16 multiply-accumulate on 3 terms
// (hi*hi + hi*lo + lo*hi) into the f32 accumulator. The command's setup object carries the parameters and the weight's
// bf16 split; they stay in the core's GEMM accumulator, which the command leaves unused.
// The row sizes must be multiples of 64 (EmbeddingGemma 2's are). Layouts 1-3 (256-dim heads) stage the block's rows
// in the W tile and place them in the C object where the C join sends them into the pfx attention's Q, K or V^T.
#include "xdna-math.h"

#include <stdint.h>

#include <aie_api/aie.hpp>

#define ROWOP_PRAGMA(x)    _Pragma(#x)
#define ROWOP_UNROLL_FULL  ROWOP_PRAGMA(clang loop unroll(full))
#define ROWOP_UNROLL_NEVER ROWOP_PRAGMA(clang loop unroll(disable))

using namespace xdna;

namespace {

constexpr int VL          = 16;
constexpr int WL          = 32;
constexpr int UN          = 2;
constexpr int TERMS       = 3;
constexpr int STEP        = WL * UN;
constexpr int BLOCK_ROWS  = 8;
constexpr int RSQRT_MAGIC = 0x5F3759DF;

// The setup object's words, as the accumulator keeps them: the parameters, then the weight's bf16 hi [d] and lo [d]
constexpr int P_MODE      = 0;
constexpr int P_D         = 1;
constexpr int P_N_ROT     = 2;
constexpr int P_SCALE     = 3;
constexpr int P_EPS       = 4;
constexpr int P_LAYOUT    = 5;
constexpr int GS_WORD     = 16;
constexpr int SETUP_WORDS = 272;
// The C object's half the next block goes to
constexpr int TOGGLE_WORD = SETUP_WORDS;
constexpr int C_HALF      = 2048;  // bf16

enum rowop_mode : int32_t {
    RO_NONE     = 0,  // the data path only
    RO_R1_BLOCK = 5,  // x [8][d] f32; tab cos [8][n_rot / 2], sin [8][n_rot / 2] f32; out [8][d] bf16
};

// The output layouts (xdna-pfx-row.h): the C object's half, or the attention's Q, K or V^T stream positions
enum rowop_layout : int32_t {
    RL_C  = 0,
    RL_Q  = 1,
    RL_K  = 2,
    RL_VT = 3,
};

constexpr int HEAD_D    = 256;  // layouts 1-3
constexpr int TILE      = 8;    // an 8 x 8 bf16 tile: the block's rows at 8 dims
constexpr int LINE_BF16 = 16;   // a 32 B line, the C join's unit
constexpr int Q_LINES   = 4;    // a transposed tile's lines: pairs of dims
constexpr int K_CHUNKS  = HEAD_D / LINE_BF16;

using acc32 = aie::accum<accfloat, 32>;
using v8bf  = aie::vector<bfloat16, TILE>;
using v64bf = aie::vector<bfloat16, TILE * TILE>;

// Where the block's rows go: row 0 at first, row j >= 1 at rest + (j - 1) stride
struct block_rows {
    bfloat16 * first;
    bfloat16 * rest;
    int        stride;
};

inline bfloat16 * row_at(const block_rows & b, int j) {
    return j == 0 ? b.first : b.rest + (j - 1) * b.stride;
}

inline float as_float(int32_t w) {
    float f;
    __builtin_memcpy(&f, &w, 4);
    return f;
}

struct pair_bf {
    v32bf h;
    v32bf l;
};

template <int K> __attribute__((always_inline)) inline pair_bf split(v32f x) {
    pair_bf p;
    p.h = to_bf16(x);
    if constexpr (K == 3) {
        p.l = to_bf16(aie::sub(x, to_f32(p.h)));
    }
    return p;
}

template <int K> __attribute__((always_inline)) inline acc32 mul_k(const pair_bf & a, const pair_bf & b) {
    acc32 acc = aie::mul(a.h, b.h);
    if constexpr (K == 3) {
        acc = aie::mac(acc, a.h, b.l);
        acc = aie::mac(acc, a.l, b.h);
    }
    return acc;
}

template <int K> __attribute__((always_inline)) inline acc32 mac_k(acc32 acc, const pair_bf & a, const pair_bf & b) {
    acc = aie::mac(acc, a.h, b.h);
    if constexpr (K == 3) {
        acc = aie::mac(acc, a.h, b.l);
        acc = aie::mac(acc, a.l, b.h);
    }
    return acc;
}

template <int K> __attribute__((always_inline)) inline acc32 msc_k(acc32 acc, const pair_bf & a, const pair_bf & b) {
    acc = aie::msc(acc, a.h, b.h);
    if constexpr (K == 3) {
        acc = aie::msc(acc, a.h, b.l);
        acc = aie::msc(acc, a.l, b.h);
    }
    return acc;
}

// The rms scale of each of the block's rows, lane j row j.
__attribute__((noinline, minsize)) v16f rms_scales_of(v16f ss, int n, float eps, float scale) {
    const v16f v    = aie::add(vec_mul(ss, bc<VL>(sc_div(1.0f, sc_i2f(n)))), bc<VL>(eps));
    const v16f half = vec_mul(v, bc<VL>(0.5f));
    v16f       r    = aie::sub(bci<VL>(RSQRT_MAGIC), aie::downshift(v.cast_to<int32_t>(), 1)).cast_to<float>();
    for (int it = 0; it < 2; ++it) {
        r = vec_mul(r, aie::sub(bc<VL>(1.5f), vec_mul(half, vec_mul(r, r))));
    }
    return vec_mul(r, bc<VL>(scale));
}

template <int K> __attribute__((always_inline)) inline acc32 squares_acc(const float * x, int n) {
    acc32 ss = aie::zeros<accfloat, WL>();
    ROWOP_UNROLL_NEVER
    for (int i = 0; i < n; i += STEP) {
        ROWOP_UNROLL_FULL
        for (int u = 0; u < UN; ++u) {
            const pair_bf v = split<K>(aie::load_v<WL>(x + i + u * WL));
            ss              = mac_k<K>(ss, v, v);
        }
    }
    return ss;
}

// The block's sums of squares, lane j row j.
__attribute__((noinline)) v16f block_sums(const float * x, int n) {
    v16f s = aie::zeros<float, VL>();
    ROWOP_UNROLL_NEVER
    for (int j = 0; j < BLOCK_ROWS; ++j) {
        s.set(aie::reduce_add(squares_acc<TERMS>(x + j * n, n).to_vector<float>()), j);
    }
    return s;
}

// The weight's split at dims [i, i + 32): hi at gs, lo at gs + n.
__attribute__((always_inline)) inline pair_bf gamma_at(const bfloat16 * gs, int n, int i) {
    return pair_bf{ aie::load_v<WL>(gs + i), aie::load_v<WL>(gs + n + i) };
}

// x * (r * gamma) at dims [i, i + 32), as a split ready for the next product.
template <int K>
__attribute__((always_inline)) inline pair_bf scaled_g(const float * x, const pair_bf & g, const pair_bf & r, int i) {
    const pair_bf gr = split<K>(mul_k<K>(r, g).template to_vector<float>());
    return split<K>(mul_k<K>(split<K>(aie::load_v<WL>(x + i)), gr).template to_vector<float>());
}

// Dims [lo, hi) of one row, normed.
__attribute__((noinline)) void norm_row_blk(const float * x, const bfloat16 * gs, const pair_bf & r, bfloat16 * out,
                                            int n, int lo, int hi) {
    ROWOP_UNROLL_NEVER
    for (int i = lo; i < hi; i += STEP) {
        ROWOP_UNROLL_FULL
        for (int u = 0; u < UN; ++u) {
            const int     j  = i + u * WL;
            const pair_bf gr = split<TERMS>(mul_k<TERMS>(r, gamma_at(gs, n, j)).to_vector<float>());
            aie::store_v(out + j, mul_k<TERMS>(split<TERMS>(aie::load_v<WL>(x + j)), gr).to_vector<bfloat16>());
        }
    }
}

// Dims [0, 2 half) of one row, normed and rotated in neox pairs (i, i + half).
__attribute__((noinline)) void rope_row_blk(const float * x, const bfloat16 * gs, const float * cs, const float * sn,
                                            const pair_bf & r, bfloat16 * out, int d, int half) {
    ROWOP_UNROLL_NEVER
    for (int i = 0; i < half; i += STEP) {
        ROWOP_UNROLL_FULL
        for (int u = 0; u < UN; ++u) {
            const int     j = i + u * WL;
            const pair_bf a = scaled_g<TERMS>(x, gamma_at(gs, d, j), r, j);
            const pair_bf b = scaled_g<TERMS>(x, gamma_at(gs, d, half + j), r, half + j);
            const pair_bf c = split<TERMS>(aie::load_v<WL>(cs + j));
            const pair_bf s = split<TERMS>(aie::load_v<WL>(sn + j));
            aie::store_v(out + j, msc_k<TERMS>(mul_k<TERMS>(a, c), b, s).to_vector<bfloat16>());
            aie::store_v(out + half + j, mac_k<TERMS>(mul_k<TERMS>(a, s), b, c).to_vector<bfloat16>());
        }
    }
}

// Each row of the block, scaled by its lane of r.
void r1_rows(const float * x, const bfloat16 * gs, const float * cs, const float * sn, v16f r, const block_rows & out,
             int d, int n_rot) {
    const int half = n_rot / 2;
    ROWOP_UNROLL_NEVER
    for (int j = 0; j < BLOCK_ROWS; ++j) {
        const pair_bf rj = split<TERMS>(bc<WL>(r.get(j)));
        if (n_rot > 0) {
            rope_row_blk(x + j * d, gs, cs + j * half, sn + j * half, rj, row_at(out, j), d, half);
        }
        norm_row_blk(x + j * d, gs, rj, row_at(out, j), d, n_rot, d);
    }
}

// R1 over a block: 8 positions of one head.
void r1_block(const float * x, const float * tab, const block_rows & out, const float * acc) {
    const int32_t *  prm   = (const int32_t *) acc;
    const int        d     = prm[P_D];
    const int        n_rot = prm[P_N_ROT];
    const bfloat16 * gs    = (const bfloat16 *) (acc + GS_WORD);
    const v16f       r = rms_scales_of(block_sums(x, d), d, as_float(prm[P_EPS]), as_float(prm[P_SCALE]));
    r1_rows(x, gs, tab, tab + BLOCK_ROWS * (n_rot / 2), r, out, d, n_rot);
}

// Rows j of the C object's half, d apart
block_rows rows_in_c(bfloat16 * half, int d) {
    return block_rows{ half, half + d, d };
}

// Rows staged in the W tile: row 0 past the block's x, row j >= 1 over x row j - 1 (read before row j is written)
block_rows rows_staged(float * x, int d) {
    return block_rows{ (bfloat16 *) (x + BLOCK_ROWS * d), (bfloat16 *) x, 2 * d };
}

// The C object's line the C join sends to stream line l (xdna-pgemm-path.h's join, inverted on 32 B lines)
inline int join_line(int l) {
    return 32 * (l / 32) + 8 * (l % 4) + (l / 4) % 8;
}

// The block's rows at dims [d0, d0 + 8), dim-major: 8 rows a dim
__attribute__((always_inline)) inline v64bf dims_tile(const block_rows & b, int d0) {
    v64bf t;
    ROWOP_UNROLL_FULL
    for (int j = 0; j < BLOCK_ROWS; ++j) {
        t.insert(j, aie::load_v<TILE>(row_at(b, j) + d0));
    }
    return aie::transpose(t, TILE, TILE);
}

// Q: tile t = 16 h + c is stream lines 8 t + 4 bp + k, a pair of dims each
__attribute__((noinline)) void place_q(const block_rows & b, bfloat16 * out, int bp) {
    ROWOP_UNROLL_NEVER
    for (int t = 0; t < HEAD_D / TILE; ++t) {
        const v64bf v = dims_tile(b, TILE * t);
        ROWOP_UNROLL_FULL
        for (int k = 0; k < Q_LINES; ++k) {
            aie::store_v(out + LINE_BF16 * join_line(TILE * t + 4 * bp + k), v.extract<LINE_BF16>(k));
        }
    }
}

// K, row j: dims [16 c, 16 c + 16) (c = 8 h + 4 q + k) are stream line 128 h + 64 bp + 32 q + 4 j + k
__attribute__((noinline)) void place_k_row(const bfloat16 * row, bfloat16 * out, int bp, int j) {
    ROWOP_UNROLL_NEVER
    for (int c = 0; c < K_CHUNKS; ++c) {
        const int l = 128 * (c / 8) + 64 * bp + 32 * ((c / 4) % 2) + 4 * j + c % 4;
        aie::store_v(out + LINE_BF16 * join_line(l), aie::load_v<LINE_BF16>(row + LINE_BF16 * c));
    }
}

__attribute__((noinline)) void place_k(const block_rows & b, bfloat16 * out, int bp) {
    ROWOP_UNROLL_NEVER
    for (int j = 0; j < BLOCK_ROWS; ++j) {
        place_k_row(row_at(b, j), out, bp, j);
    }
}

// V^T: dim 8 c + e of tile t = 16 h + c is half bp of stream line 128 h + 16 (c / 2) + 2 e + c % 2
__attribute__((noinline)) void place_vt(const block_rows & b, bfloat16 * out, int bp) {
    ROWOP_UNROLL_NEVER
    for (int t = 0; t < HEAD_D / TILE; ++t) {
        const v64bf v = dims_tile(b, TILE * t);
        const int   h = t / 16, c = t % 16;
        ROWOP_UNROLL_FULL
        for (int e = 0; e < TILE; ++e) {
            const int l = 128 * h + 16 * (c / 2) + 2 * e + c % 2;
            aie::store_v(out + LINE_BF16 * join_line(l) + TILE * bp, v.extract<TILE>(e));
        }
    }
}

void place(int layout, const block_rows & b, bfloat16 * out, int bp) {
    if (layout == RL_Q) {
        place_q(b, out, bp);
    } else if (layout == RL_K) {
        place_k(b, out, bp);
    } else {
        place_vt(b, out, bp);
    }
}

// R1 into today's C half, or staged and placed for the attention's objects
void r1_layout(float * x, const float * tab, bfloat16 * out, const float * acc) {
    const int32_t * prm = (const int32_t *) acc;
    const int       bp  = prm[TOGGLE_WORD];
    if (prm[P_LAYOUT] == RL_C) {
        r1_block(x, tab, rows_in_c(out + bp * C_HALF, prm[P_D]), acc);
        return;
    }
    const block_rows rows = rows_staged(x, prm[P_D]);
    r1_block(x, tab, rows, acc);
    place(prm[P_LAYOUT], rows, out, bp);
}

__attribute__((noinline)) void copy_words(const int32_t * src, int32_t * dst, int n) {
    ROWOP_UNROLL_NEVER
    for (int i = 0; i < n; i += VL) {
        aie::store_v(dst + i, aie::load_v<VL>(src + i));
    }
}

}  // namespace

extern "C" {

// The command's setup object w into acc; the first block goes to the C object's first half.
void ggml_xdna_row_setup(const uint8_t * w, float * acc) {
    int32_t * a = (int32_t *) acc;
    copy_words((const int32_t *) w, a, SETUP_WORDS);
    a[TOGGLE_WORD] = 0;
}

// One block: x on the W tile, its positions' cos and sin in tab, the bf16 result into half of the C object out, or
// (layouts 1-3) staged over x and placed across the C object.
void ggml_xdna_rowop(float * x, const float * tab, bfloat16 * out, float * acc) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t * a = (int32_t *) acc;
    if (a[P_MODE] == RO_R1_BLOCK) {
        r1_layout(x, tab, out, acc);
    }
    a[TOGGLE_WORD] ^= 1;
}

}  // extern "C"
