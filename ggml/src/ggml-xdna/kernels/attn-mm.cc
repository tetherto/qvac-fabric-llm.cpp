//
// Prefill attention on the mmul: one core of a
// pair that owns FA_R output rows, holding half of the head dim.
//
// The rows are (query head, position) pairs of one KV head: row block qb (8
// rows) is one query head at 8 consecutive positions. A key tile is FA_NK
// keys. Everything is computed transposed, keys or head dim down and rows
// across, so a row's softmax statistics are elementwise over the tile's rows
// and a row's correction is the same eight lanes repeated - no lane
// reductions and no transposes (the 8 x 8 f32 transpose does not exist).
// Per tile the core
//   - scores its half: S^T_half = K_half . Q_half^T,
//   - adds its neighbour's half from shared memory, scales, masks and runs
//     the online softmax (f32 max and sum), writing P^T in bf16,
//   - rescales its output half and accumulates O^T_half += V_half^T . P^T.
// Both cores of the pair run the same softmax on the same full scores, so
// they agree on P bit for bit.
//
// Which keys a row sees is data, not position: each pass carries, per row, the
// first and last visible key index (a causal, a bidirectional or a banded
// mask, packed sequences), and the keys of the tiles it streams.
//
// Layouts are the mmul's (8, 8) sub-tiles:
//   K half     (key/8, d/8, 8, 8) bf16 - A
//   Q half^T   (d/8, row/8, 8, 8) bf16 - B
//   S^T, P^T   (key/8, row/8, 8, 8)    - C of the one, B of the other
//   V half^T   (d/8, key/8, 8, 8) bf16 - A
//   O half^T   (d/8, row/8, 8, 8) f32
// A tile is two objects of the K/V stream, K half then V half^T, so the
// stream carries the next one while this one is used.

#define NOCPP

// Before aie_api: it selects the mmul's bfp16 form (8 x 8 x 8) from this. Set
// after the include, the bf16 8 x 8 x 8 mmul compiles to a slow composite
// (7.8 us a tile against ~0.5).
#define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 1

#include "xdna-math.h"

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <type_traits>

#define bf16_f32_ONLY
#include "aie_kernels/aie2p/mm.cc"

#ifndef FA_R
#    define FA_R 32  // rows a pair owns
#endif
#ifndef FA_DH
#    define FA_DH 128  // head-dim half a core holds
#endif
#ifndef FA_NK
#    define FA_NK 32  // keys a tile
#endif
#ifndef FA_TR
#    define FA_TR 256  // bf16 elements after a pass's Q rows: the pass trailer
#endif
#ifndef FA_TAU
#    define FA_TAU 8.0f  // log2 headroom before the output is rescaled
#endif

namespace {

constexpr int R  = FA_R;
constexpr int DH = FA_DH;
constexpr int NK = FA_NK;
constexpr int RB = R / 8;   // row blocks
constexpr int KB = NK / 8;  // key blocks a tile
constexpr int DB = DH / 8;  // head-dim blocks a half

// m starts far below any score but above the mask, so a row that sees no key
// of a tile keeps its max and the masked keys weigh 2^(-huge) = 0.
constexpr float M0   = -1.0e30f;
constexpr float MASK = -1.0e34f;

using v16f  = aie::vector<float, 16>;
using v64f  = aie::vector<float, 64>;
using v16bf = aie::vector<bfloat16, 16>;

// 2^x in bf16 for n scores (a multiple of 32), to bf16 precision. x = n + f
// with n the nearest integer: 2^n goes into the result's exponent field and
// 2^f, |f| <= 1/2, from exp2_bf16_poly. The masked scores' n is garbage below
// the low clamp and their f is 0, so only the low end is clamped here. Not
// used: f32 products (emulated, ~60 cycles a vector), the hardware exp2
// (piecewise linear, 6% off), and to_fixed / to_float / float max (emulated,
// ~150 bundles an iteration).
__attribute__((noinline)) void exp2_block(const float * xp, bfloat16 * out, int n) {
    using v16i      = aie::vector<int32, 16>;
    const v16i nmin = aie::broadcast<int32, 16>(-126);
    for (int i = 0; i < n; i += 32) {
        chess_prepare_for_pipelining {
            const v16f                   x0 = aie::load_v<16>(xp + i);
            const v16f                   x1 = aie::load_v<16>(xp + i + 16);
            v16i                         n0, n1;
            const v16f                   f0 = xdna::exp2_frac(x0, n0);
            const v16f                   f1 = xdna::exp2_frac(x1, n1);
            const v16i                   s0 = aie::upshift(aie::max(n0, nmin), 23);
            const v16i                   s1 = aie::upshift(aie::max(n1, nmin), 23);
            const aie::vector<float, 32> e  = xdna::exp2_bf16_poly(xdna::to_bf16(aie::concat(f0, f1)));
            const v16f                   r0 = aie::add(e.extract<16>(0).cast_to<int32>(), s0).cast_to<float>();
            const v16f                   r1 = aie::add(e.extract<16>(1).cast_to<int32>(), s1).cast_to<float>();
            aie::store_v(out + i, xdna::to_bf16(aie::concat(r0, r1)));
        }
    }
}

// The exponents of a tile, then the weights (P^T).
alignas(64) float g_x[FA_R * FA_NK];

// The key row k of element (k, c) of an (8 x 8) tile, for the visibility mask.
alignas(64) const int32_t g_krow[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
};

// The per-column (row of the output) reduction of an (8 x 8) tile, element
// (key, row) at key * 8 + row: eight chunks combined elementwise.
inline aie::vector<float, 8> col_max(const v64f & t) {
    aie::vector<float, 8> m = t.extract<8>(0);
    m                       = aie::max(m, t.extract<8>(1));
    m                       = aie::max(m, t.extract<8>(2));
    m                       = aie::max(m, t.extract<8>(3));
    m                       = aie::max(m, t.extract<8>(4));
    m                       = aie::max(m, t.extract<8>(5));
    m                       = aie::max(m, t.extract<8>(6));
    m                       = aie::max(m, t.extract<8>(7));
    return m;
}

inline aie::vector<float, 8> col_sum(const v64f & t) {
    aie::vector<float, 8> s = t.extract<8>(0);
    s                       = aie::add(s, t.extract<8>(1));
    s                       = aie::add(s, t.extract<8>(2));
    s                       = aie::add(s, t.extract<8>(3));
    s                       = aie::add(s, t.extract<8>(4));
    s                       = aie::add(s, t.extract<8>(5));
    s                       = aie::add(s, t.extract<8>(6));
    s                       = aie::add(s, t.extract<8>(7));
    return s;
}

// An (8 x 8) tile whose column c is v[c] throughout.
inline v64f col_bcast(const aie::vector<float, 8> & v) {
    return aie::concat(aie::concat(v, v, v, v), aie::concat(v, v, v, v));
}

inline aie::vector<int32, 64> col_bcast_i(const aie::vector<int32, 8> & v) {
    return aie::concat(aie::concat(v, v, v, v), aie::concat(v, v, v, v));
}

// The next tile's first key: a pass streams consecutive tiles from its first.
int32_t g_k0   = 0;
// The whole-array design (kernels/attn_mm.py): the current pass's rows, from
// the trailer of its Q object, and what the pass says about its tiles.
int32_t g_pass = 0;
alignas(64) int32_t g_lo[FA_R];  // first visible key of each row
alignas(64) int32_t g_hi[FA_R];  // last visible key of each row
int32_t g_lo_min[FA_R / 8], g_lo_max[FA_R / 8], g_hi_min[FA_R / 8], g_hi_max[FA_R / 8];  // per row block
int32_t g_vis_lo = 0, g_vis_hi = 0;  // the keys any row of the pass sees
bool    g_first  = true;             // no tile of the row blocks has been seen yet
bool    g_skip   = false;            // the current tile has no visible key for this pair

}  // namespace

extern "C" {

// Start a row block: O half zero, m = M0, l = 0. `ml` is m[R] then l[R].
void fa_begin(float * o, float * ml) {
    g_first      = true;
    const v16f z = aie::zeros<float, 16>();
    for (int i = 0; i < R * DH; i += 16) {
        aie::store_v(o + i, z);
    }
    for (int i = 0; i < R; i += 16) {
        aie::store_v(ml + i, aie::broadcast<float, 16>(M0));
        aie::store_v(ml + R + i, z);
    }
}

// This core's half of the tile's scores: S^T = K_half . Q_half^T.
void fa_scores(const bfloat16 * q, const bfloat16 * kv, float * s) {
    // a tile outside every row's keys is skipped by all three steps of a tile,
    // and by the neighbour: it holds the same rows
    g_skip = g_k0 + NK <= g_vis_lo || g_k0 > g_vis_hi;
    if (g_skip) {
        return;
    }
    const v16f z = aie::zeros<float, 16>();
    for (int i = 0; i < R * NK; i += 16) {
        aie::store_v(s + i, z);
    }
    matmul_vectorized_2x2_mmul<bfloat16, float, NK / 8, DH / 8, R / 8, 8, 8, 8, true, true>(kv, q, s);
}

// The online softmax over the tile, from both halves of its scores. The
// host scales Q by log2(e) * the attention scale, so a score is already the
// exponent of 2 its weight is. A row sees the keys g_lo .. g_hi of its pass;
// the tile's keys are the next NK. Writes P^T (bf16).
//
// A row's max is only moved - and the output and sum rescaled by 2^(m_old -
// m_new) - when the tile's max passes the kept one by more than FA_TAU: the
// weights then stay under 2^FA_TAU, which bf16 and the f32 output carry, and
// the rescale (4096 f32 products a tile, which this core emulates) is rare.
// The same correction goes into the output and the sum, so it cancels in the
// final division whatever its rounding.
void fa_update(const float * s_own, const float * s_nb, float * o, bfloat16 * p, float * ml) {
    const int32_t k0 = g_k0;
    g_k0 += NK;
    if (g_skip) {
        return;
    }
    const bool first = g_first;
    g_first          = false;
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int qb = 0; qb < RB; qb++) {
        // every row of the block sees the whole tile
        const bool clear = k0 >= g_lo_max[qb] && k0 + NK - 1 <= g_hi_min[qb];
        v64f       f[KB];
        for (int kb = 0; kb < KB; kb++) {
            const int off = (kb * RB + qb) * 64;
            f[kb]         = aie::add(aie::load_v<64>(s_own + off), aie::load_v<64>(s_nb + off));
            if (!clear) {
                const int32_t key0 = k0 + kb * 8;
                if (key0 > g_hi_max[qb] || key0 + 7 < g_lo_min[qb]) {
                    f[kb] = aie::broadcast<float, 64>(MASK);
                } else if (key0 < g_lo_max[qb] || key0 + 7 > g_hi_min[qb]) {
                    const auto key  = aie::add(aie::broadcast<int32, 64>(key0), aie::load_v<64>(g_krow));
                    const auto lo   = col_bcast_i(aie::load_v<8>(g_lo + qb * 8));
                    const auto hi   = col_bcast_i(aie::load_v<8>(g_hi + qb * 8));
                    f[kb]           = aie::select(f[kb], aie::broadcast<float, 64>(MASK), aie::lt(key, lo));
                    f[kb]           = aie::select(f[kb], aie::broadcast<float, 64>(MASK), aie::gt(key, hi));
                }
            }
        }
        v64f mx = f[0];
        for (int kb = 1; kb < KB; kb++) {
            mx = aie::max(mx, f[kb]);
        }
        const aie::vector<float, 8> cm   = col_max(mx);
        aie::vector<float, 8>       m    = aie::load_v<8>(ml + qb * 8);
        aie::vector<float, 8>       l    = aie::load_v<8>(ml + R + qb * 8);
        const auto                  grow = aie::gt(cm, aie::add(m, aie::broadcast<float, 8>(FA_TAU)));
        if (!grow.empty()) {
            const aie::vector<float, 8> m_new = aie::max(m, cm);
            if (!first) {
                const aie::vector<float, 16> d = aie::concat(aie::sub(m, m_new), aie::zeros<float, 8>());
                alignas(64) float            xd[32];
                alignas(64) bfloat16         ed[32];
                aie::store_v(xd, aie::concat(d, d));
                exp2_block(xd, ed, 32);
                const aie::vector<float, 8> alpha =
                    aie::accum<accfloat, 16>(aie::load_v<16>(ed)).to_vector<float>(0).extract<8>(0);
                l             = aie::mul(l, alpha).to_vector<float>(0);
                const v64f ab = col_bcast(alpha);
                for (int db = 0; db < DB; db++) {
                    float * t = o + (db * RB + qb) * 64;
                    aie::store_v(t, aie::mul(aie::load_v<64>(t), ab).to_vector<float>(0));
                }
            }
            m = m_new;
            aie::store_v(ml + qb * 8, m);
        }
        const v64f mb = col_bcast(m);
        for (int kb = 0; kb < KB; kb++) {
            aie::store_v(g_x + (kb * RB + qb) * 64, aie::sub(f[kb], mb));
        }
        aie::store_v(ml + R + qb * 8, l);
    }
    exp2_block(g_x, p, R * NK);
    for (int qb = 0; qb < RB; qb++) {
        v64f psum = aie::zeros<float, 64>();
        for (int kb = 0; kb < KB; kb++) {
            psum =
                aie::add(psum, aie::accum<accfloat, 64>(aie::load_v<64>(p + (kb * RB + qb) * 64)).to_vector<float>(0));
        }
        aie::store_v(ml + R + qb * 8, aie::add(aie::load_v<8>(ml + R + qb * 8), col_sum(psum)));
    }
}

// The call's header, the first object of the core's Q stream: word 0 the
// passes.
void fa_hdr(const bfloat16 * h, int32_t * cnt) {
    const int32_t * w = (const int32_t *) h;
    cnt[0]            = w[0];
    g_pass            = 0;
}

// Start a pass from the trailer after its Q rows: word 0 the first key of
// the first tile it streams, word 1 the tiles, then each row's first and last
// visible key. cnt[1] is the pass's tile count.
void fa_pass(float * o, float * ml, int32_t * cnt, const bfloat16 * q) {
    const int32_t * w = (const int32_t *) (q + R * DH);
    g_k0              = w[0];
    cnt[1]            = w[1];
    g_vis_lo          = w[2];
    g_vis_hi          = w[2 + R];
    for (int r = 0; r < R; r++) {
        g_lo[r]   = w[2 + r];
        g_hi[r]   = w[2 + R + r];
        g_vis_lo  = g_lo[r] < g_vis_lo ? g_lo[r] : g_vis_lo;
        g_vis_hi  = g_hi[r] > g_vis_hi ? g_hi[r] : g_vis_hi;
    }
    for (int qb = 0; qb < RB; qb++) {
        int32_t lo_min = g_lo[qb * 8], lo_max = lo_min, hi_min = g_hi[qb * 8], hi_max = hi_min;
        for (int c = 1; c < 8; c++) {
            const int32_t lo = g_lo[qb * 8 + c], hi = g_hi[qb * 8 + c];
            lo_min = lo < lo_min ? lo : lo_min;
            lo_max = lo > lo_max ? lo : lo_max;
            hi_min = hi < hi_min ? hi : hi_min;
            hi_max = hi > hi_max ? hi : hi_max;
        }
        g_lo_min[qb] = lo_min;
        g_lo_max[qb] = lo_max;
        g_hi_min[qb] = hi_min;
        g_hi_max[qb] = hi_max;
    }
    g_pass++;
    fa_begin(o, ml);
}

void fa_update_p(const float * s_own, const float * s_nb, float * o, bfloat16 * p, float * ml) {
    fa_update(s_own, s_nb, o, p, ml);
}

// O^T_half += V_half^T . P^T; V^T is the tile's second object.
void fa_pv(const bfloat16 * p, const bfloat16 * v, float * o) {
    if (g_skip) {
        return;
    }
    matmul_vectorized_2x2_mmul<bfloat16, float, DH / 8, NK / 8, R / 8, 8, 8, 8, true, true>(v, p, o);
}

// End a row block: O half / l.
void fa_end(float * o, const float * ml) {
    for (int qb = 0; qb < RB; qb++) {
        const aie::vector<float, 8> l  = aie::load_v<8>(ml + R + qb * 8);
        // l >= 1: the row's largest key weighs exactly 1
        const v16f                  x  = aie::inv(aie::concat(l, aie::broadcast<float, 8>(1.0f)));
        const v64f                  ib = col_bcast(x.extract<8>(0));
        for (int db = 0; db < DB; db++) {
            float * t = o + (db * RB + qb) * 64;
            aie::store_v(t, aie::mul(aie::load_v<64>(t), ib).to_vector<float>(0));
        }
    }
}

}  // extern "C"
