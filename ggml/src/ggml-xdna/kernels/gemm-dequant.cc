// SPDX-License-Identifier: MIT
//
// On-chip expansion of the packed NPU weight formats (see xdna-quant.h) into
// the bf16 B tile the mmul micro-kernel consumes.
//
// Decode is memory bound: what crosses DDR decides the speed. Handing the GEMM
// bf16 weights costs 2 B per value; the packed formats cost 0.75 B (q4g32) and
// 1.25 B (q8g16) and are bit-exact with respect to the ggml block they come
// from, so this expansion buys 2.7x less weight traffic at no accuracy cost.
//
// Layout contract with the host packer (xdna-ops.cpp) and the design
// (dequant_b.py). One B tile is (K_TILE x N_TILE) and is stored sub-tile major
// in the order the L2->L1 stream used to produce, so the expansion is a linear
// read and a linear write:
//
//   codes   the sub-tiles of the output, each S*T values (si, ti) row-major,
//           but ordered column block first: (nb, kb, si, ti). That is the
//           order the expansion reads them in, so the load stays linear while
//           the group parameters are held constant across a whole run.
//           q4g32 packs two values per byte, low nibble first; q8g16 stores
//           one int8 per value.
//   D, M    the group scale and min planes, K_TILE/GROUP groups of two N_TILE
//           bf16 rows: the rounded value then the residual, which sum to the
//           parameter. Element (kb, nb, si, ti) uses group (kb*S + si)/GROUP,
//           which is kb*S/GROUP because S divides GROUP, and column nb*T + ti.
//           So a sub-tile needs one T-wide slice of each row, repeated to fill
//           the 32-lane vector. q8g16 has no min plane.
//
// Each group parameter is a pair of bf16 values rather than one bf16 or one
// f32. One bf16 is not accurate enough - the parameter is shared by the whole
// group, so rounding it does not average out over K, and w = q*d + m is a
// difference of two similar terms for small weights. An f32 parameter is
// accurate but halves the throughput, because AIE2P multiplies f32 by
// emulation while bf16 is native. The pair reconstructs ~16 mantissa bits from
// native multiply-accumulates into the f32 accumulator, and only the result is
// rounded to bf16 - the same rounding the bf16 GEMM applies anyway.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

// The core's default rounding mode is toward -inf, which biases about half the
// weights by one bf16 ulp against ggml's round-half-to-even. Setting the mode
// once per call makes the accumulator's own bf16 conversion match exactly, so
// no fixup is needed on the store path.
inline void set_round_half_even()
{
    aie::set_rounding(aie::rounding_mode::conv_even);
}

// Widen a T-lane f32 pattern to a full 32-lane vector by repeating it. Every
// value of a sub-tile shares one group along K, so its scale depends only on
// the column, and the column index repeats every MAC_T lanes.
template <int T> inline aie::vector<bfloat16, 32> repeat_to_32(const aie::vector<bfloat16, T> &v)
{
    static_assert(T <= 32 && (32 % T) == 0, "MAC_T must divide the 32-lane vector");
    if constexpr (T == 32) {
        return v;
    } else {
        return repeat_to_32<2 * T>(aie::concat(v, v));
    }
}

} // namespace

extern "C" {

#ifndef K_TILE
#define K_TILE 64
#endif
#ifndef N_TILE
#define N_TILE 64
#endif
#ifndef MAC_S
#define MAC_S 8
#endif
#ifndef MAC_T
#define MAC_T 8
#endif
#ifndef Q4_GROUP
#define Q4_GROUP 32
#endif

// q4g32: unsigned 4-bit codes, w = q * d + m.
void dequant_q4g32_bf16(const uint8_t *in, bfloat16 *out)
{
    constexpr int KB = K_TILE / MAC_S;      // sub-tile rows
    constexpr int NB = N_TILE / MAC_T;      // sub-tile cols
    constexpr int VEC = 32;                 // lanes per vector op
    constexpr int ST = MAC_S * MAC_T;       // values per sub-tile
    constexpr int CH = ST / VEC;            // vector ops per sub-tile
    constexpr int NG = K_TILE / Q4_GROUP;   // groups along K in this tile
    constexpr int KG = Q4_GROUP / MAC_S;    // sub-tile rows per group
    constexpr int CODE_BYTES = K_TILE * N_TILE / 2;
    static_assert(ST % VEC == 0, "sub-tile must be a whole number of vectors");

    const uint8_t *__restrict pC = in;
    const bfloat16 *__restrict pD = (const bfloat16 *)(in + CODE_BYTES);
    const bfloat16 *__restrict pM = pD + 2 * NG * N_TILE;

    event0();
    set_round_half_even();
    const aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>((bfloat16)1.0f);

    // Column block outermost, then group: the four parameter vectors are then
    // loaded once per (nb, g) and the inner run is KG*CH straight vector ops
    // with constant operands. The codes arrive in exactly this order, so the
    // read stays linear and only the store strides, by one sub-tile per kb.
    for (int nb = 0; nb < NB; nb++) {
        const int n0 = nb * MAC_T;
        for (int g = 0; g < NG; g++) {
            const bfloat16 *dg = pD + 2 * g * N_TILE;
            const bfloat16 *mg = pM + 2 * g * N_TILE;
            const aie::vector<bfloat16, VEC> dhi = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(dg + n0));
            const aie::vector<bfloat16, VEC> dlo = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(dg + N_TILE + n0));
            const aie::vector<bfloat16, VEC> mhi = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(mg + n0));
            const aie::vector<bfloat16, VEC> mlo = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(mg + N_TILE + n0));

            bfloat16 *__restrict pO = out + (size_t)(g * KG * NB + nb) * ST;
            for (int i = 0; i < KG * CH; i++)
                chess_prepare_for_pipelining chess_loop_range(KG * CH, ) {
                    // VEC nibbles -> uint8 -> uint16 -> bf16 (values 0..15).
                    aie::vector<uint4, VEC> q4 = aie::load_v<VEC>((const uint4 *)pC);
                    pC += VEC / 2;
                    aie::vector<uint8, VEC> q8 = aie::unpack(q4);
                    aie::vector<uint16, VEC> q16 = aie::unpack(q8);
                    // q is exact in bf16, so every product is exact in the f32
                    // accumulator: w = q*d_hi + q*d_lo + m_hi + m_lo.
                    aie::vector<bfloat16, VEC> qb = aie::to_float<bfloat16>(q16, 0);
                    aie::accum<accfloat, VEC> acc = aie::mul(qb, dhi);
                    acc = aie::mac(acc, qb, dlo);
                    acc = aie::mac(acc, mhi, one);
                    acc = aie::mac(acc, mlo, one);
                    aie::store_v(pO + (i % CH) * VEC, acc.template to_vector<bfloat16>(0));
                    if ((i % CH) == CH - 1) {
                        pO += (size_t) NB * ST;
                    }
                }
        }
    }
    event1();
}

// q8g16: signed 8-bit codes, w = q * d (the min plane is absent).
void dequant_q8g16_bf16(const uint8_t *in, bfloat16 *out)
{
    constexpr int NB = N_TILE / MAC_T;
    constexpr int VEC = 32;
    constexpr int ST = MAC_S * MAC_T;
    constexpr int CH = ST / VEC;
    constexpr int Q8_GROUP = 16;
    constexpr int NG = K_TILE / Q8_GROUP;
    constexpr int KG = Q8_GROUP / MAC_S;
    constexpr int CODE_BYTES = K_TILE * N_TILE;
    static_assert(ST % VEC == 0, "sub-tile must be a whole number of vectors");
    static_assert(Q8_GROUP % MAC_S == 0, "a sub-tile row must sit inside one group");

    const int8_t *__restrict pC = (const int8_t *)in;
    const bfloat16 *__restrict pD = (const bfloat16 *)(in + CODE_BYTES);

    event0();
    set_round_half_even();

    for (int nb = 0; nb < NB; nb++) {
        const int n0 = nb * MAC_T;
        for (int g = 0; g < NG; g++) {
            const bfloat16 *dg = pD + 2 * g * N_TILE;
            const aie::vector<bfloat16, VEC> dhi = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(dg + n0));
            const aie::vector<bfloat16, VEC> dlo = repeat_to_32<MAC_T>(aie::load_v<MAC_T>(dg + N_TILE + n0));

            bfloat16 *__restrict pO = out + (size_t)(g * KG * NB + nb) * ST;
            for (int i = 0; i < KG * CH; i++)
                chess_prepare_for_pipelining chess_loop_range(KG * CH, ) {
                    aie::vector<int8, VEC> q8 = aie::load_v<VEC>(pC);
                    pC += VEC;
                    aie::vector<int16, VEC> q16 = aie::unpack(q8);
                    aie::vector<bfloat16, VEC> qb = aie::to_float<bfloat16>(q16, 0);
                    aie::accum<accfloat, VEC> acc = aie::mul(qb, dhi);
                    acc = aie::mac(acc, qb, dlo);
                    aie::store_v(pO + (i % CH) * VEC, acc.template to_vector<bfloat16>(0));
                    if ((i % CH) == CH - 1) {
                        pO += (size_t) NB * ST;
                    }
                }
        }
    }
    event1();
}

} // extern "C"
