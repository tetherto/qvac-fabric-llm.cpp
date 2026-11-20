// Decode attention on the GEMV pool: one query token against llama's KV cache,
// split across the sixteen pool cores. A dispatch runs it instead of a
// projection when the activation header says so (gemv_q4.py, _core_lead_att
// and _core_part_att); the three functions share this file's state, so each
// core links one build of it.
//
// Per dispatch, per core:
//   att_q     x2  the eight query heads (bf16, the 1/sqrt(D) scale already in),
//                 four a tile, and the count of valid cache positions
//   att_chunk xN  the first object carries this core's index among the
//                 sixteen; every later one is a chunk of AD_P positions of the
//                 cache - K of both kv heads, then V of both, as f16 - the core
//                 k = 16 * j + index of the whole cache
//   att_emit  x33 the core's partial state, 64 floats a piece:
//                 m[2 groups][16] l[2][16] (lane i: head i % 4 of the
//                 group), o[2 groups][32 blocks][4 heads][8 dims]
//   q arrives tiled: per group [32 blocks][8 dims][4 heads]
// The host combines the sixteen partials (log-sum-exp) into the output.
#include "xdna-vec.h"

#include <stdint.h>

#include <aie_api/aie.hpp>

#define AD_D     256                    // head dim
#define AD_H     8                      // query heads
#define AD_G     4                      // query heads a kv head serves
#define AD_KVH   2
#define AD_P     5                      // cache positions a chunk
#define AD_ML    64                     // m then l, per group 16 lanes, lane i = head i % 4
#define AD_ST    (AD_ML + AD_H * AD_D)  // 2112 floats = 33 pieces of 64
#define AD_PIECE 64
// Below any real score, so a masked row keeps its max and contributes
// exp(-huge) = 0; the reference max may move AD_TAU before o is rescaled.
#define AD_NINF  (-1.0e30f)
#define AD_TAU   8.0f

namespace {

alignas(64) bfloat16 q_b[AD_H * AD_D];
alignas(64) float st[AD_ST];  // m[2][16] l[2][16] o
alignas(64) const int32_t lane_pos[16] = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
int n_valid, core_id, chunk_i, piece_i, skip;

__attribute__((noinline)) aie::vector<float, 16> exp_v(const aie::vector<float, 16> & x) {
    // bf16-precision exp: its error scales p, and the rescale it computes
    // multiplies o and l alike, so it cancels in o / l.
    const auto               t = aie::mul(x, aie::broadcast<float, 16>(1.4426950408889634f)).to_vector<float>(0);
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::exp2<bfloat16>(t), 0);
    return a.to_vector<float>(0);
}

}  // namespace

extern "C" {

__attribute__((minsize)) void ggml_xdna_att_q(const int32_t * a) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int idx = a[AD_G * AD_D / 2 + 1];
    if (idx == 0) {
        n_valid = a[AD_G * AD_D / 2];
        skip    = a[AD_G * AD_D / 2 + 2];  // diagnostic: move the data, compute nothing
        chunk_i = 0;
        piece_i = 0;
        aie::store_v(st, aie::broadcast<float, 16>(AD_NINF));
        aie::store_v(st + 16, aie::broadcast<float, 16>(AD_NINF));
#pragma clang loop unroll(disable)
        for (int i = 32; i < AD_ST; i += 16) {
            aie::store_v(st + i, aie::zeros<float, 16>());
        }
    }
    const bfloat16 * src = (const bfloat16 *) a;
    bfloat16 *       dst = q_b + idx * AD_G * AD_D;
#pragma clang loop unroll(disable)
    for (int i = 0; i < AD_G * AD_D; i += 32) {
        aie::store_v(dst + i, aie::load_v<32>(src + i));
    }
}

// The chunk's K and V of one kv group, bf16 and tiled for the matrix
// multiplies: [32 blocks of 8 dims][8 positions][8 dims]. Rows 5..7 are
// padding; they are zeroed once a dispatch and only ever hold finite values.
#define AD_TILE 64
#define AD_NB   (AD_D / 8)

// o[4 heads][8 dims] blocks of one group times alpha per head, for when a
// head's reference max moves
__attribute__((noinline)) void rescale_group(float * Og, const aie::vector<float, 16> & a16) {
    // eight rows of the four heads, transposed: four rows of one head each
    const aie::vector<float, 32> arep = aie::transpose(aie::concat(a16, a16), 8, 4);
#pragma clang loop unroll(disable)
    for (int b = 0; b < AD_NB; b++) {
        aie::store_v(Og + b * 32, aie::mul(aie::load_v<32>(Og + b * 32), arep).template to_vector<float>(0));
    }
}

void ggml_xdna_att_chunk(uint8_t * w, bfloat16 * scr) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    if (chunk_i == 0) {
        core_id = ((const int32_t *) w)[0];
        chunk_i = 1;
#pragma clang loop vectorize(disable) unroll(disable)
        for (int i = 0; i < 2 * AD_NB * AD_TILE; i += 32) {
            aie::store_v(scr + i, aie::zeros<bfloat16, 32>());
        }
        return;
    }
    const int k = 16 * (chunk_i - 1) + core_id;
    chunk_i++;
    if (skip) {
        return;
    }
    int valid = n_valid - AD_P * k;
    if (valid <= 0) {
        return;
    }
    if (valid > AD_P) {
        valid = AD_P;
    }
    const uint16_t * K  = (const uint16_t *) w;
    bfloat16 *       Kt = scr;
    bfloat16 *       Vt = scr + AD_NB * AD_TILE;

#pragma clang loop unroll(disable)
    for (int g = 0; g < AD_KVH; g++) {
        // to bf16, into the tiles, two positions at once: their rows are
        // adjacent in a tile, one 256-bit store a block (a lone last row is
        // paired with zeros - the pad rows must stay finite, P V reads them)
        // K and V rows alike: the chunk's V follows its K, the tiles too
#pragma clang loop unroll(disable)
        for (int kv = 0; kv < 2; kv++) {
            const uint16_t * base = K + kv * AD_P * AD_KVH * AD_D + g * AD_D;
            bfloat16 *       tile = scr + kv * AD_NB * AD_TILE;
#pragma clang loop unroll(disable)
            for (int p = 0; p < valid; p += 2) {
                const uint16_t *              s0   = base + p * AD_KVH * AD_D;
                const bool                    two  = p + 1 < valid;
                const uint16_t *              s1   = two ? s0 + AD_KVH * AD_D : s0;
                // the lone row's partner zeroed by a mask, not a branch: a
                // branch in the body keeps its loads from overlapping
                const aie::vector<uint16, 32> keep = aie::broadcast<uint16, 32>(two ? 0xFFFF : 0);
                bfloat16 *                    dst  = tile + p * 8;
                // the next step's raw values load while this one converts
                // (the load past the row's end stays inside the W object)
                aie::vector<int16, 32>        h0   = aie::load_v<32>((const int16 *) s0);
                aie::vector<int16, 32>        h1   = aie::load_v<32>((const int16 *) s1);
#pragma clang loop unroll(disable)
                for (int i = 0; i < AD_D / 32; i++) {
                    const aie::vector<int16, 32> n0 = aie::load_v<32>((const int16 *) (s0 + 32 * (i + 1)));
                    const aie::vector<int16, 32> n1 = aie::load_v<32>((const int16 *) (s1 + 32 * (i + 1)));
                    const auto                   x0 = xdna::f16_bits_to_bf16(h0);
                    const auto                   x1 =
                        aie::bit_and(xdna::f16_bits_to_bf16(h1).cast_to<uint16>(), keep).cast_to<bfloat16>();
                    h0           = n0;
                    h1           = n1;
                    const auto z = aie::interleave_zip(x0, x1, 8);
                    bfloat16 * d = dst + 4 * i * AD_TILE;
                    aie::store_v(d, z.first.template extract<16>(0));
                    aie::store_v(d + AD_TILE, z.first.template extract<16>(1));
                    aie::store_v(d + 2 * AD_TILE, z.second.template extract<16>(0));
                    aie::store_v(d + 3 * AD_TILE, z.second.template extract<16>(1));
                }
            }
        }

        // S[8 positions][4 heads] = K Q^T, reduced over the dims by the MMUL
        const bfloat16 *                       qt = q_b + g * AD_G * AD_D;  // [block][8 dims][4 heads]
        aie::mmul<8, 8, 4, bfloat16, bfloat16> mq;
        mq.mul(aie::load_v<64>(Kt), aie::load_v<32>(qt));
#pragma clang loop unroll(disable)
        for (int b = 1; b < AD_NB; b++) {
            mq.mac(aie::load_v<64>(Kt + b * AD_TILE), aie::load_v<32>(qt + b * 32));
        }
        // rows past the valid positions out: lane i of the lower half is
        // position i / 4, of the upper 4 + i / 4
        const aie::vector<float, 32> s    = mq.template to_vector<float>();
        const aie::vector<int32, 16> pos  = aie::load_v<16>(lane_pos);
        const aie::vector<float, 16> ninf = aie::broadcast<float, 16>(AD_NINF);
        const aie::vector<float, 16> s_lo =
            aie::select(ninf, s.template extract<16>(0), aie::lt(pos, aie::broadcast<int32, 16>(valid)));
        const aie::vector<float, 16> s_hi =
            aie::select(ninf, s.template extract<16>(1), aie::lt(pos, aie::broadcast<int32, 16>(valid - 4)));

        // per head over the rows: lane i holds head i % 4 throughout, the
        // rotations keep every lane valid. The weights are taken against a
        // reference max that only moves when the chunk passes it by more
        // than 8 (e^8 is nothing to fp32 or bf16), so o is rescaled rarely
        // rather than every chunk.
        aie::vector<float, 16> t          = aie::max(s_lo, s_hi);
        t                                 = aie::max(t, aie::shuffle_down_rotate(t, 8));
        t                                 = aie::max(t, aie::shuffle_down_rotate(t, 4));
        const aie::vector<float, 16> mo   = aie::load_v<16>(st + g * 16);
        const auto                   over = aie::gt(t, aie::add(mo, aie::broadcast<float, 16>(AD_TAU)));
        const aie::vector<float, 16> mn   = aie::select(mo, aie::max(mo, t), over);
        const auto                   e_lo = exp_v(aie::sub(s_lo, mn));
        const auto                   e_hi = exp_v(aie::sub(s_hi, mn));
        aie::vector<float, 16>       su   = aie::add(e_lo, e_hi);
        su                                = aie::add(su, aie::shuffle_down_rotate(su, 8));
        su                                = aie::add(su, aie::shuffle_down_rotate(su, 4));
        const bool             moved      = !over.empty();
        aie::vector<float, 16> lv         = aie::load_v<16>(st + 32 + g * 16);
        float *                Og         = st + AD_ML + g * AD_G * AD_D;  // [block][4 heads][8 dims]
        if (moved) {
            const aie::vector<float, 16> a16 = exp_v(aie::sub(mo, mn));
            lv                               = aie::mul(lv, a16).template to_vector<float>(0);
            aie::store_v(st + g * 16, mn);
            rescale_group(Og, a16);
        }
        aie::store_v(st + 32 + g * 16, aie::add(lv, su));

        // P[4 heads][8 positions], bf16
        aie::accum<accfloat, 32> ea;
        ea.from_vector(aie::concat(e_lo, e_hi), 0);
        const aie::vector<bfloat16, 32> pv = aie::transpose(ea.template to_vector<bfloat16>(), 8, 4);

        // O[4 heads][8 dims] per block += P V through the MMUL, two
        // independent blocks an iteration (four cost the pool core's program
        // memory more than they gave)
#pragma clang loop unroll(disable)
        for (int b = 0; b < AD_NB; b += 2) {
            aie::accum<accfloat, 32> c0, c1;
            c0.from_vector(aie::load_v<32>(Og + b * 32), 0);
            c1.from_vector(aie::load_v<32>(Og + (b + 1) * 32), 0);
            aie::mmul<4, 8, 8, bfloat16, bfloat16> m0(c0), m1(c1);
            m0.mac(pv, aie::load_v<64>(Vt + b * AD_TILE));
            m1.mac(pv, aie::load_v<64>(Vt + (b + 1) * AD_TILE));
            aie::store_v(Og + b * 32, m0.template to_vector<float>());
            aie::store_v(Og + (b + 1) * 32, m1.template to_vector<float>());
        }
    }
}

__attribute__((minsize)) void ggml_xdna_att_emit(float * o) {
    const float * src = st + piece_i * AD_PIECE;
#pragma clang loop unroll(disable)
    for (int i = 0; i < AD_PIECE; i += 16) {
        aie::store_v(o + i, aie::load_v<16>(src + i));
    }
    piece_i++;
}

}  // extern "C"
