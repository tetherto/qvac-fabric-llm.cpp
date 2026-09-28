//
// The prefill GEMM's expander: one K step of a
// decode GEMV weight tile into the bf16 B operand of the mmul micro-kernel.
//
// The tile is the decode's, as gemv-q4.cc reads it - (K_TILE x 64) for one
// core, one lane group of 64 columns:
//
//   for g in K_TILE/32 groups:  codes (32 rows of 64 values: q4g32 two a byte,
//                               low nibble first; 8-bit one int8 a value),
//                               then d8[64], m8[64] bf16 (integers)
//   then dS_hi[64], dS_lo[64], mS_hi[64], mS_lo[64] bf16 - the pair a
//   super-block's integers scale: w = q * (dS * d8) + mS * m8.
//
// A step is 128 rows of K: a q4g32 tile (K_TILE 256) is two, an 8-bit one
// (K_TILE 128) is one. The output is 128 x 64 bf16 in the order the mmul reads
// B, sub-tiles of s x t = 8 x MAC_T: (kb, nb, si, ti) for row kb*8+si and
// column nb*MAC_T+ti (kernels/gemm.py streams row-major B into the same
// order). The bf16 mmul on AIE2P is (r, s, t) = (4, 8, 4).
//
// Per group the parameters are formed in f32 exactly (bf16 x integer lands in
// the f32 accumulator exactly) and d is split back into a bf16 pair, so a row
// costs two native bf16 multiply-accumulates: w = v * d_hi + v * d_lo + m',
// with v = 128 + q built from the code's bits (0x4300 | q is bf16 128 + q for
// q < 128) and m' = m - 128 d folding the 128 back out. A signed 8-bit code
// takes one more for its top bit. The result is rounded to bf16 once, to
// nearest even, from the f32 accumulator.

#include "xdna-math.h"

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef MAC_T
#    define MAC_T 4
#endif
#ifndef EXP_UNROLL
#    define EXP_UNROLL 1
#endif

namespace {

constexpr int LANE = 64;
constexpr int T    = MAC_T;
static_assert(T == 4 || T == 8, "a sub-tile row is 4 or 8 values");
constexpr int GROUP = 32;
#ifndef EXP_STEP
#    define EXP_STEP 128
#endif
constexpr int STEP = EXP_STEP;              // K rows a call expands (64 or 128)
static_assert(STEP == 64 || STEP == 128, "a step is 64 or 128 rows of K");
constexpr int GPS      = STEP / GROUP;      // groups a step
constexpr int Q4_CODE  = GROUP * LANE / 2;  // 1024 B of nibbles a group
constexpr int Q8_CODE  = GROUP * LANE;      // 2048 B of int8 a group
constexpr int PARAM    = 4 * LANE;          // d8[64], m8[64] bf16
constexpr int Q4_BLOCK = Q4_CODE + PARAM;
constexpr int Q8_BLOCK = Q8_CODE + PARAM;

using v32bf  = aie::vector<bfloat16, 32>;
using v64bf  = aie::vector<bfloat16, 64>;
using v32f   = aie::vector<float, 32>;
using facc32 = aie::accum<accfloat, 32>;

// bf16 128 + q for 32 codes q in 0..127, from their bits
inline v32bf biased(const aie::vector<uint16, 32> & q) {
    return aie::add(q, aie::broadcast<uint16, 32>(0x4300)).cast_to<bfloat16>();
}

// Store one row's 32 values (columns n0 .. n0 + 31) in (kb, nb, si, ti) order:
// nb advances every T columns, 8 * T values apart in the output.
inline void put_row(bfloat16 * out, int k, int n0, const v32bf & w) {
    const int  kb = k >> 3, si = k & 7;
    bfloat16 * base = out + ((kb * (LANE / T) + n0 / T) * 8 + si) * T;
    aie::store_v(base + 0 * 64, w.extract<8>(0));
    aie::store_v(base + 1 * 64, w.extract<8>(1));
    aie::store_v(base + 2 * 64, w.extract<8>(2));
    aie::store_v(base + 3 * 64, w.extract<8>(3));
}

// Rows k, k + 1 (k even) of all 64 columns, T = 8: in each sub-tile the two
// rows' eight values are adjacent, one 256-bit store a sub-tile.
inline void put_rows2(bfloat16 * out, int k, const v64bf & a, const v64bf & b) {
    const int  kb = k >> 3, si = k & 7;
    const auto z    = aie::interleave_zip(a, b, 8);  // a0..7 b0..7 a8..15 b8..15 ...
    bfloat16 * base = out + (kb * (LANE / T) * 8 + si) * T;
    aie::store_v(base + 0 * 64, z.first.extract<16>(0));
    aie::store_v(base + 1 * 64, z.first.extract<16>(1));
    aie::store_v(base + 2 * 64, z.first.extract<16>(2));
    aie::store_v(base + 3 * 64, z.first.extract<16>(3));
    aie::store_v(base + 4 * 64, z.second.extract<16>(0));
    aie::store_v(base + 5 * 64, z.second.extract<16>(1));
    aie::store_v(base + 6 * 64, z.second.extract<16>(2));
    aie::store_v(base + 7 * 64, z.second.extract<16>(3));
}

// With T = 4 a sub-tile row is 64 bits, under the narrowest vector store, so
// eight rows (one kb) go out together: their 32 values each are eight 4-value
// chunks, and three zips of doubling width turn chunk (si, nb) of the rows
// into block nb = (si, ti), 32 values, one store each.
inline void put_block(bfloat16 * out, int kb, int n0, v32bf (&v)[8]) {
    v32bf a[8], b[8];
    for (int i = 0; i < 4; i++) {
        const auto z = aie::interleave_zip(v[2 * i], v[2 * i + 1], 4);
        a[2 * i]     = z.first;
        a[2 * i + 1] = z.second;
    }
    // a[2i]: chunks 0..3 of rows 2i, 2i+1 alternating; a[2i+1]: chunks 4..7
    for (int i = 0; i < 2; i++) {
        for (int h = 0; h < 2; h++) {
            const auto z         = aie::interleave_zip(a[4 * i + h], a[4 * i + 2 + h], 8);
            b[4 * i + 2 * h]     = z.first;
            b[4 * i + 2 * h + 1] = z.second;
        }
    }
    // b[4i + j]: chunks 2j, 2j+1 of rows 4i..4i+3, each four rows long
    bfloat16 * base = out + (kb * (LANE / T) + n0 / T) * 8 * T;
    for (int j = 0; j < 4; j++) {
        const auto z = aie::interleave_zip(b[j], b[4 + j], 16);
        aie::store_v(base + (2 * j) * 32, z.first);
        aie::store_v(base + (2 * j + 1) * 32, z.second);
    }
}

// The group's rows for MAC_T = 8 (the prefill GEMM), both halves of a row
// at once. The format is a template argument, so the row loop has no branch
// and runs pipelined: with the format tested inside it, and the nibbles
// unpacked 4 -> 8 -> 16 bits (the unpack size register rewritten twice a
// row), every row was a serial chain of ~40 bundles.
template <bool Q8>
__attribute__((noinline)) void expand_group8(const uint8_t *  blk,
                                             const bfloat16 * d8,
                                             const bfloat16 * m8,
                                             const bfloat16 * sup,
                                             bfloat16 *       out,
                                             int              k0) {
    v32bf  dh[2], dl[2], th[2], tl[2];
    facc32 a0[2];
    for (int h = 0; h < 2; h++) {
        const int  n0 = h * 32;
        const v32f d  = xdna::pair_times<32>(sup + n0, sup + LANE + n0, d8 + n0);
        const v32f m  = xdna::pair_times<32>(sup + 2 * LANE + n0, sup + 3 * LANE + n0, m8 + n0);
        dh[h]         = xdna::to_bf16(d);
        dl[h]         = xdna::to_bf16(aie::sub(d, xdna::to_f32(dh[h])));
        // m' = m - 128 d (and - 256 d for the signed form, whose top bit adds
        // 128 d back), as bf16 products of d's pair: a power of two times a
        // bf16 is exact, and the f32 product is emulated (~20 bundles)
        const v32bf k = aie::broadcast<bfloat16, 32>((bfloat16) (Q8 ? -256.0f : -128.0f));
        a0[h].from_vector(m, 0);
        a0[h] = aie::mac(a0[h], dh[h], k);
        a0[h] = aie::mac(a0[h], dl[h], k);
        if constexpr (Q8) {
            const v32bf c128 = aie::broadcast<bfloat16, 32>((bfloat16) 128.0f);
            th[h]            = aie::mul(dh[h], c128).to_vector<bfloat16>(0);
            tl[h]            = aie::mul(dl[h], c128).to_vector<bfloat16>(0);
        }
    }
    if constexpr (Q8) {
        const aie::vector<int16, 32> c128 = aie::broadcast<int16, 32>(128);
        const aie::vector<int16, 32> m7f  = aie::broadcast<int16, 32>(0x7F);
        const aie::vector<int16, 32> one  = aie::broadcast<int16, 32>(0x3F80);
        const aie::vector<int16, 32> zero = aie::zeros<int16, 32>();
        // a row of 64 signed codes: code + 128 is 0..255, its low seven bits
        // as the biased value, its top bit as a 0/1 bf16 against 128 d. Two
        // rows an iteration, pipelined by hand one row ahead (the compiler
        // does not): the next row's codes unpack while this one multiplies;
        // held in plain variables - as arrays in a struct they went to the
        // stack, 3x slower
        auto                         cv   = [&](const uint8_t * src, v32bf & v, v32bf & t) {
            const aie::vector<int16, 32> u = aie::add(aie::unpack(aie::load_v<32>((const int8 *) src)), c128);
            v = biased(aie::bit_and(u, m7f).cast_to<uint16>());
            t = aie::select(zero, one, aie::ge(u, c128)).cast_to<bfloat16>();
        };
        auto mh = [&](const v32bf & v, const v32bf & t, int h) -> v32bf {
            facc32 a = aie::mac(a0[h], v, dh[h]);
            a        = aie::mac(a, v, dl[h]);
            a        = aie::mac(a, t, th[h]);
            a        = aie::mac(a, t, tl[h]);
            return a.to_vector<bfloat16>(0);
        };
        v32bf v0, t0, v1, t1;
        cv(blk, v0, t0);
        cv(blk + 32, v1, t1);
#pragma clang loop min_iteration_count(16)
        for (int r = 0; r < GROUP; r += 2) {
            v32bf w0, u0, w1, u1;
            cv(blk + (r + 1) * LANE, w0, u0);
            cv(blk + (r + 1) * LANE + 32, w1, u1);
            const v64bf o0 = aie::concat(mh(v0, t0, 0), mh(v1, t1, 1));
            // (the row past the group is read and not used: in the tile)
            cv(blk + (r + 2) * LANE, v0, t0);
            cv(blk + (r + 2) * LANE + 32, v1, t1);
            put_rows2(out, k0 + r, o0, aie::concat(mh(w0, u0, 0), mh(w1, u1, 1)));
        }
    } else {
        const aie::vector<uint16, 32> lo4 = aie::broadcast<uint16, 32>(0x0F);
        const aie::vector<uint16, 32> hi4 = aie::broadcast<uint16, 32>(0xF0);

        // a row's 64 codes as the biased bf16 of both column halves; the high
        // nibble is masked before its shift, so the shift is exact in the
        // kernel's rounding mode and no row rewrites crrnd
        struct codes {
            v32bf v0, v1;
        };

        auto prep = [&](const uint8_t * src) -> codes {
            const aie::vector<uint16, 32> w = aie::unpack(aie::load_v<32>(src));
            aie::accum<acc32, 32>         hacc;
            hacc.from_vector(aie::bit_and(w, hi4), 0);
            const auto z = aie::interleave_zip(aie::bit_and(w, lo4), hacc.to_vector<uint16>(4), 1);
            return { biased(z.first), biased(z.second) };
        };
        auto row64 = [&](const codes & c) -> v64bf {
            facc32 a = aie::mac(a0[0], c.v0, dh[0]);
            a        = aie::mac(a, c.v0, dl[0]);
            facc32 b = aie::mac(a0[1], c.v1, dh[1]);
            b        = aie::mac(b, c.v1, dl[1]);
            return aie::concat(a.to_vector<bfloat16>(0), b.to_vector<bfloat16>(0));
        };
        // pipelined by hand (the compiler does not): an iteration unpacks the
        // next pair of rows while this pair multiplies and stores
        codes c0 = prep(blk), c1 = prep(blk + LANE / 2);
#pragma clang loop min_iteration_count(15)
        for (int r = 0; r < GROUP - 2; r += 2) {
            const codes n0 = prep(blk + (r + 2) * (LANE / 2));
            const codes n1 = prep(blk + (r + 3) * (LANE / 2));
            put_rows2(out, k0 + r, row64(c0), row64(c1));
            c0 = n0;
            c1 = n1;
        }
        put_rows2(out, k0 + GROUP - 2, row64(c0), row64(c1));
    }
}

// out = silu(g) * u for one block's 64 x 32 gate and up halves of the
// accumulator (see ggml_xdna_pg_out). g u is formed from bf16 hi/lo pairs in
// the f32 accumulator; sigmoid(g) = 1 / (1 + 2^t), t = -g log2(e), with 2^t
// as attention's exp2 (2^n in the exponent field, a cubic for 2^f) and the
// reciprocal by the core's inverse: bf16-level, which is what the next GEMM
// reads of it.
// silu(gate) * up of the accumulator's halves. BF16 = false: f32 in the
// (8, 8) sub-tile order the MemTile's join turns into rows. BF16 = true
// (the next projection's A, mode 2): bf16 [64 rows][32] as 1024 words at the
// front of the object - each word placed where the join's reordering (a
// stream word w of [rb 8][cb 4][r 8][c 8] to row-major [64][32] word
// address) sends it to word row * 16 + col / 2 - and the back half left as
// it is, which the host lets fall on the next block's rows.
template <bool BF16> __attribute__((noinline)) void silu_mul(const float * acc, float * out) {
    constexpr int SUB = 64, CB = 8, HB = CB / 2;
    const v32bf   nl2h = aie::broadcast<bfloat16, 32>((bfloat16) -1.4375f);
    const v32bf   nl2l = aie::broadcast<bfloat16, 32>((bfloat16) (-1.44269504f + 1.4375f));
    for (int rb = 0; rb < 8; rb++) {
        const float * gp = acc + rb * CB * SUB;
        const float * up = gp + HB * SUB;
        float *       op = out + rb * HB * SUB;
        for (int i = 0; i < HB * SUB; i += 32) {
            const v32f  g  = aie::load_v<32>(gp + i);
            const v32f  u  = aie::load_v<32>(up + i);
            const v32bf gh = xdna::to_bf16(g);
            const v32bf gl = xdna::to_bf16(aie::sub(g, xdna::to_f32(gh)));
            const v32bf uh = xdna::to_bf16(u);
            const v32bf ul = xdna::to_bf16(aie::sub(u, xdna::to_f32(uh)));
            // t = -g log2(e)
            facc32      ta = aie::mul(gh, nl2h);
            ta             = aie::mac(ta, gh, nl2l);
            const v32f t   = ta.to_vector<float>(0);
            xdna::v16f e0, e1;
            xdna::exp2_bf16_32(t, e0, e1);
            // 1 + 2^t, then its inverse - the sigmoid - by two Newton steps
            // from the bit-pattern guess (12% off, then 1.5%, then bf16's
            // own), each r (2 - d r) in bf16 products (the float inverse and
            // products are emulated)
            const v32f                   den = aie::add(aie::concat(e0, e1), aie::broadcast<float, 32>(1.0f));
            const v32bf                  db  = xdna::to_bf16(den);
            const aie::vector<int32, 32> gi  = aie::sub(aie::broadcast<int32, 32>(0x7EF311C7), den.cast_to<int32>());
            v32bf                        rcp = xdna::to_bf16(gi.cast_to<float>());
            for (int it = 0; it < 2; it++) {
                const v32f  dr = aie::mul(db, rcp).to_vector<float>(0);
                const v32bf e2 = xdna::to_bf16(aie::sub(aie::broadcast<float, 32>(2.0f), dr));
                rcp            = aie::mul(rcp, e2).to_vector<bfloat16>(0);
            }
            const v32bf sb = rcp;
            // g u from the pairs, then times the sigmoid
            facc32      pa = aie::mul(gh, uh);
            pa             = aie::mac(pa, gh, ul);
            pa             = aie::mac(pa, gl, uh);
            const v32f  pu = pa.to_vector<float>(0);
            const v32bf ph = xdna::to_bf16(pu);
            const v32bf pl = xdna::to_bf16(aie::sub(pu, xdna::to_f32(ph)));
            facc32      oa = aie::mul(ph, sb);
            oa             = aie::mac(oa, pl, sb);
            if constexpr (!BF16) {
                aie::store_v(op + i, oa.to_vector<float>(0));
            } else {
                // rows rb * 8 + r0 .. + 3, columns cb * 8 .. + 7
                const aie::vector<bfloat16, 32> ob = oa.to_vector<bfloat16>(0);
                const int                       cb = i / SUB, r0 = (i % SUB) / 8, wc = cb * 4;
                bfloat16 *                      ow = (bfloat16 *) out;
                auto                            at = [&](int row) {
                    return 2 * ((((row / 16) * 4 + (row % 2) * 2 + wc / 8) * 8 + (row % 16) / 2) * 8 + wc % 8);
                };
                // constant eighths: extract<8>(k) with a runtime k picks the wrong one
                const int row = rb * 8 + r0;
                aie::store_v(ow + at(row), ob.extract<8>(0));
                aie::store_v(ow + at(row + 1), ob.extract<8>(1));
                aie::store_v(ow + at(row + 2), ob.extract<8>(2));
                aie::store_v(ow + at(row + 3), ob.extract<8>(3));
            }
        }
    }
}

}  // namespace

extern "C" {

// Expand K step `step` (0 .. K_TILE / EXP_STEP - 1: a q4g32 tile is 256 rows,
// an 8-bit one 128) of `tile` into `out` (EXP_STEP x 64 bf16). `q8`: the tile
// is the 8-bit form.
void ggml_xdna_expand(const uint8_t * tile, bfloat16 * out, int32_t step, int32_t q8) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    // groups a tile (not a step): a q4g32 tile is 256 rows, an 8-bit one 128
    const int        ng    = q8 ? 128 / GROUP : 256 / GROUP;
    const int        block = q8 ? Q8_BLOCK : Q4_BLOCK;
    const int        code  = q8 ? Q8_CODE : Q4_CODE;
    const bfloat16 * sup   = (const bfloat16 *) (tile + ng * block);
    if constexpr (T == 8) {
        for (int gs = 0; gs < GPS; gs++) {
            const int        g   = step * GPS + gs;
            const uint8_t *  blk = tile + g * block;
            const bfloat16 * d8  = (const bfloat16 *) (blk + code);
            if (q8) {
                expand_group8<true>(blk, d8, d8 + LANE, sup, out, gs * GROUP);
            } else {
                expand_group8<false>(blk, d8, d8 + LANE, sup, out, gs * GROUP);
            }
        }
        return;
    }
    for (int gs = 0; gs < GPS; gs++) {
        const int        g   = step * GPS + gs;
        const uint8_t *  blk = tile + g * block;
        const bfloat16 * d8  = (const bfloat16 *) (blk + code);
        const bfloat16 * m8  = d8 + LANE;
        for (int h = 0; h < 2; h++) {  // two halves of 32 columns
            const int   n0   = h * 32;
            const v32f  d    = xdna::pair_times<32>(sup + n0, sup + LANE + n0, d8 + n0);
            const v32f  m    = xdna::pair_times<32>(sup + 2 * LANE + n0, sup + 3 * LANE + n0, m8 + n0);
            const v32bf d_hi = xdna::to_bf16(d);
            const v32bf d_lo = xdna::to_bf16(aie::sub(d, xdna::to_f32(d_hi)));
            // m' = m - 128 d (and - 256 d for the signed form, whose top bit
            // adds 128 d back)
            const v32f  d128 = aie::mul(d, aie::broadcast<float, 32>(128.0f)).to_vector<float>(0);
            v32f        mb   = aie::sub(m, d128);
            v32bf       t_hi, t_lo;
            if (q8) {
                mb   = aie::sub(mb, d128);
                t_hi = xdna::to_bf16(d128);
                t_lo = xdna::to_bf16(aie::sub(d128, xdna::to_f32(t_hi)));
            }
            facc32 a0;
            a0.from_vector(mb, 0);
            v32bf rows[8];
#if EXP_UNROLL > 1
#    pragma clang loop unroll_count(EXP_UNROLL)
#endif
            for (int r = 0; r < GROUP; r++) {
                facc32 a = a0;
                if (q8) {
                    // signed code + 128 is 0..255: its low seven bits as the
                    // biased value, its top bit as a 0/1 bf16 against 128 d
                    const aie::vector<int16, 32> u =
                        aie::add(aie::unpack(aie::load_v<32>((const int8 *) (blk + r * LANE + n0))),
                                 aie::broadcast<int16, 32>(128));
                    const v32bf v  = biased(aie::bit_and(u, aie::broadcast<int16, 32>(0x7F)).cast_to<uint16>());
                    const v32bf tb = aie::select(aie::zeros<int16, 32>(), aie::broadcast<int16, 32>(0x3F80),
                                                 aie::ge(u, aie::broadcast<int16, 32>(128)))
                                         .cast_to<bfloat16>();
                    a = aie::mac(a, v, d_hi);
                    a = aie::mac(a, v, d_lo);
                    a = aie::mac(a, tb, t_hi);
                    a = aie::mac(a, tb, t_lo);
                } else {
                    // 32 nibbles: 16 bytes of the row, low nibble first
                    const aie::vector<uint8, 32> c =
                        aie::unpack(aie::load_v<32>((const uint4 *) (blk + r * (LANE / 2) + n0 / 2)));
                    const v32bf v = biased(aie::unpack(c));
                    a             = aie::mac(a, v, d_hi);
#if !defined(EXP_ONE_MAC)
                    // the pair's second half: EXP_ONE_MAC drops it (5.6 -> 5.0
                    // us a 128-row step) but triples the GEMM's error under
                    // bfp16 (9.2e-3 -> 2.7e-2 rel rms), so it stays
                    a = aie::mac(a, v, d_lo);
#endif
                }
                if constexpr (T == 8) {
                    put_row(out, gs * GROUP + r, n0, a.to_vector<bfloat16>(0));
                } else {
                    rows[r & 7] = a.to_vector<bfloat16>(0);
                    if ((r & 7) == 7) {
                        put_block(out, (gs * GROUP + r) >> 3, n0, rows);
                    }
                }
            }
        }
    }
}

// The prefill GEMM's cores (kernels/pgemm.py) run what the first object of
// their A stream says, not counts the host pokes into their memory: words
// 0..4 are the output blocks the call has (M blocks x chunks), the K tiles a
// block, whether the tiles are the 8-bit form, the K steps a tile and the
// pairs of them; word 5 the C objects a block leaves as, word 6 the mode
// (0: both column halves of the accumulator, 1: SwiGLU of them, 2: that in
// bf16 as the next projection's A), word 7 the
// pairs a tile before its last. Two neighbouring cores hold the same tile and each expands
// every other K step of it for both, so a core expands half the steps it
// multiplies; `parity` is the core's first step.
static int32_t pg_step = 0;
static int32_t pg_par  = 0;
static int32_t pg_half = 0;

void ggml_xdna_pg_hdr_p(const bfloat16 * a, int32_t * cnt, int32_t parity) {
    const int32_t * h = (const int32_t *) a;
    for (int i = 0; i < 8; i++) {
        cnt[i] = h[i];
    }
    pg_par  = parity;
    pg_step = parity;
    pg_half = 0;
}

// The core's next step of the tile it holds, wrapping at the tile's end.
void ggml_xdna_expand_pair(const uint8_t * tile, bfloat16 * out, const int32_t * cnt) {
    ggml_xdna_expand(tile, out, pg_step, cnt[2]);
    pg_step = pg_step + 2 >= cnt[3] ? pg_par : pg_step + 2;
}

// One C object of the block in the accumulator (64 x 64 f32 in (8, 8)
// sub-tiles, [row block][column block]): 64 rows x 32 columns, the same
// sub-tile order. Mode 0 hands out column half 0 and then half 1; mode 1
// (a gate/up weight whose core columns are 32 gate then 32 up) the SwiGLU
// silu(gate) * up of the halves.
void ggml_xdna_pg_out(const float * acc, float * out, const int32_t * cnt) {
    constexpr int SUB = 64, RB = 8, CB = 8, HB = CB / 2;
    if (cnt[6] == 0) {
        const int h = pg_half;
        pg_half ^= 1;
        for (int rb = 0; rb < RB; rb++) {
            chess_prepare_for_pipelining {
                const float * src = acc + (rb * CB + h * HB) * SUB;
                float *       dst = out + rb * HB * SUB;
                for (int i = 0; i < HB * SUB; i += 32) {
                    aie::store_v(dst + i, aie::load_v<32>(src + i));
                }
            }
        }
        return;
    }
    if (cnt[6] == 2) {
        aie::set_rounding(aie::rounding_mode::conv_even);
        silu_mul<true>(acc, out);
    } else {
        silu_mul<false>(acc, out);
    }
}

}  // extern "C"
