//
// The recurrent layers' conv input on the array (kernels/gdn_conv.py): the
// in-projection's rows, f32, convolved along the tokens with the layer's KW
// = 4 taps, SiLU'd, K and Q L2-normalized per token, and written as the
// prefill GDN's chunk inputs (kernels/gdn-mm.cc): K (C x 128), Q (C x 128)
// and a value half (C x 64), bf16 in 8 x 8 tiles, one object per value half.
//
// A core takes one head's every other chunk. A chunk's object is its C
// tokens' rows and the KW - 1 before them, [row][q 128 | k 128 | v 128] f32,
// so a chunk needs nothing from the one before. The first object is the
// header: the chunks this core takes, the index of its first, the tokens,
// the norms' eps squared, then from word 64 the taps [tap][channel] f32.
//
// Products are the native bf16 ones, the f32 values as hi/lo pairs into the
// f32 accumulator (x_h w_h + x_h w_l + x_l w_h), near f32 exact: the core
// emulates f32 products. The norm's 1 / max(|x|, eps) is a per-token factor,
// which rounded to bf16 would scale the token's whole output coherently, so
// it is worked out in f32 by Newton steps; the SiLU is bf16-level, an error
// independent from value to value.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int C   = 16;              // tokens a chunk
constexpr int KW  = 4;               // taps
constexpr int D   = 128;             // a head's channels of each of q, k, v
constexpr int NCH = 3 * D;           // a row
constexpr int DV  = 64;              // a value half
constexpr int OBJ = 2 * C * D + C * DV;   // bf16 of one GDN input object's K, Q, V

using v16f  = aie::vector<float, 16>;
using v32f  = aie::vector<float, 32>;
using v32bf = aie::vector<bfloat16, 32>;
using facc = aie::accum<accfloat, 32>;

alignas(64) bfloat16 g_wh[KW * NCH];  // the taps, hi and lo
alignas(64) bfloat16 g_wl[KW * NCH];
int32_t g_k    = 0;                   // the next chunk's index
int32_t g_ntok = 0;
int32_t g_eps2 = 0;                   // the bits of eps^2 (no scalar f32 here)

inline void hl(const v32f &x, v32bf &h, v32bf &l)
{
    h = facc(x).to_vector<bfloat16>(0);
    l = facc(aie::sub(x, facc(h).to_vector<float>(0))).to_vector<bfloat16>(0);
}

// sigmoid(x) = 1 / (1 + 2^t), t = -x log2(e), well past bf16 precision: 2^n
// in the exponent field and 2^f a cubic in the bf16 products (f's rounding
// costs 0.07%, the cubic's truncation 0.06%), the inverse of the f32 sum by
// two Newton steps from the bit-pattern guess (6%): one in plain bf16 (to
// ~0.5%), one whose products take the hi/lo pairs (to ~3e-5) - rounded to
// bf16 instead, it was a 0.4% error on every value (KLD 0.0047 -> 0.0050)
__attribute__((always_inline)) inline v32f sigmoid32(const v32f &x)
{
    using v16i = aie::vector<int32, 16>;
    v32bf xh, xl;
    hl(x, xh, xl);
    const v32bf l2h = aie::broadcast<bfloat16, 32>((bfloat16) -1.4375f);
    const v32bf l2l = aie::broadcast<bfloat16, 32>((bfloat16) (-1.44269504f + 1.4375f));
    facc ta = aie::mul(xh, l2h);
    ta = aie::mac(ta, xh, l2l);
    ta = aie::mac(ta, xl, l2h);
    const v32f t = ta.to_vector<float>(0);
    const v16f magic = aie::broadcast<float, 16>(12582912.0f);
    const v16i mbits = aie::broadcast<int32, 16>(0x4B400000);
    const v16i lo = aie::broadcast<int32, 16>(-126), hi = aie::broadcast<int32, 16>(126);
    const v16f t0 = t.extract<16>(0), t1 = t.extract<16>(1);
    const v16f y0 = aie::add(t0, magic), y1 = aie::add(t1, magic);
    const v16i n0 = aie::min(aie::max(aie::sub(y0.cast_to<int32>(), mbits), lo), hi);
    const v16i n1 = aie::min(aie::max(aie::sub(y1.cast_to<int32>(), mbits), lo), hi);
    const v32bf fb = facc(aie::concat(aie::sub(t0, aie::sub(y0, magic)),
                                      aie::sub(t1, aie::sub(y1, magic)))).to_vector<bfloat16>(0);
    const v32bf f2 = aie::mul(fb, fb).to_vector<bfloat16>(0);
    const v32bf f3 = aie::mul(f2, fb).to_vector<bfloat16>(0);
    facc ea;
    ea.from_vector(aie::broadcast<float, 32>(1.0f), 0);
    ea = aie::mac(ea, fb, aie::broadcast<bfloat16, 32>((bfloat16) 0.69140625f));
    ea = aie::mac(ea, fb, aie::broadcast<bfloat16, 32>((bfloat16) (0.69314718f - 0.69140625f)));
    ea = aie::mac(ea, f2, aie::broadcast<bfloat16, 32>((bfloat16) 0.24022651f));
    ea = aie::mac(ea, f3, aie::broadcast<bfloat16, 32>((bfloat16) 0.05550411f));
    const v32f e = ea.to_vector<float>(0);
    const v16f e0 = aie::add(e.extract<16>(0).cast_to<int32>(), aie::upshift(n0, 23)).cast_to<float>();
    const v16f e1 = aie::add(e.extract<16>(1).cast_to<int32>(), aie::upshift(n1, 23)).cast_to<float>();
    const v32f den = aie::add(aie::concat(e0, e1), aie::broadcast<float, 32>(1.0f));
    v32bf dh, dl;
    hl(den, dh, dl);
    // r = 1 / den: bit-pattern guess, then r (2 - den r) with den's pair and
    // r's pair, all in the f32 accumulator
    v32f r = aie::sub(aie::broadcast<int32, 32>(0x7EF311C7), den.cast_to<int32>()).cast_to<float>();
    {
        // the first step in plain bf16: 6% -> ~0.5%
        const v32bf r0 = facc(r).to_vector<bfloat16>(0);
        const v32f dr = aie::mul(dh, r0).to_vector<float>(0);
        const v32bf g = facc(aie::sub(aie::broadcast<float, 32>(2.0f), dr)).to_vector<bfloat16>(0);
        r = aie::mul(r0, g).to_vector<float>(0);
    }
    {
        v32bf rh, rl;
        hl(r, rh, rl);
        facc dr = aie::mul(dh, rh);
        dr = aie::mac(dr, dh, rl);
        dr = aie::mac(dr, dl, rh);
        v32bf gh, gl;
        hl(aie::sub(aie::broadcast<float, 32>(2.0f), dr.to_vector<float>(0)), gh, gl);
        facc rr = aie::mul(rh, gh);
        rr = aie::mac(rr, rh, gl);
        rr = aie::mac(rr, rl, gh);
        r = rr.to_vector<float>(0);
    }
    return r;
}

// silu of two independent vectors: one block the scheduler can interleave -
// a single chain is latency-bound, half its slots empty
__attribute__((noinline)) void silu64(const v32f &c0, const v32f &c1, v32f &o0, v32f &o1)
{
    const v32f s0 = sigmoid32(c0), s1 = sigmoid32(c1);
    v32bf ch0, cl0, sh0, sl0, ch1, cl1, sh1, sl1;
    hl(c0, ch0, cl0);
    hl(s0, sh0, sl0);
    hl(c1, ch1, cl1);
    hl(s1, sh1, sl1);
    facc a0 = aie::mul(ch0, sh0), a1 = aie::mul(ch1, sh1);
    a0 = aie::mac(a0, ch0, sl0);
    a1 = aie::mac(a1, ch1, sl1);
    a0 = aie::mac(a0, cl0, sh0);
    a1 = aie::mac(a1, cl1, sh1);
    o0 = a0.to_vector<float>(0);
    o1 = a1.to_vector<float>(0);
}

// 1 / sqrt(max(s, eps^2)) in f32 for the sums of squares s of a token's q
// (lane 0) and k (lane 1) rows, from the bit-pattern guess by three Newton
// steps (emulated products, a few), both at once
__attribute__((noinline)) v16f rsqrt_qk(const v32f &sq, const v32f &sk)
{
    const float fq = aie::reduce_add(sq), fk = aie::reduce_add(sk);
    int32_t iq, ik;
    __builtin_memcpy(&iq, &fq, 4);
    __builtin_memcpy(&ik, &fk, 4);
    iq = iq > g_eps2 ? iq : g_eps2;          // both non-negative: max as integers
    ik = ik > g_eps2 ? ik : g_eps2;
    aie::vector<int32, 16> ib = aie::broadcast<int32, 16>(iq);
    ib[1] = ik;
    const v16f sv = ib.cast_to<float>();
    aie::vector<int32, 16> rb = aie::broadcast<int32, 16>(0x5F3759DF - (iq >> 1));
    rb[1] = 0x5F3759DF - (ik >> 1);
    v16f r = rb.cast_to<float>();
    for (int it = 0; it < 3; it++) {
        const v16f r2 = aie::mul(r, r).to_vector<float>(0);
        const v16f h  = aie::mul(aie::mul(sv, r2).to_vector<float>(0), aie::broadcast<float, 16>(0.5f))
                            .to_vector<float>(0);
        r = aie::mul(r, aie::sub(aie::broadcast<float, 16>(1.5f), h)).to_vector<float>(0);
    }
    return r;
}

// element (r, c) of a (rows x cols) matrix in 8 x 8 tiles; c a multiple of 8
inline int t8(int r, int c, int cols)
{
    return ((r >> 3) * (cols >> 3) + (c >> 3)) * 64 + (r & 7) * 8;
}

// o's four eighths to offsets o0 .. o3 of a (and of b when given)
inline void put8(bfloat16 *a, bfloat16 *b, int o0, int o1, int o2, int o3, const v32bf &o)
{
    aie::store_v(a + o0, o.extract<8>(0));
    aie::store_v(a + o1, o.extract<8>(1));
    aie::store_v(a + o2, o.extract<8>(2));
    aie::store_v(a + o3, o.extract<8>(3));
    if (b) {
        aie::store_v(b + o0, o.extract<8>(0));
        aie::store_v(b + o1, o.extract<8>(1));
        aie::store_v(b + o2, o.extract<8>(2));
        aie::store_v(b + o3, o.extract<8>(3));
    }
}

} // namespace

extern "C" {

void gdn_conv_hdr(const float *h, int32_t *cnt)
{
    const int32_t *w = (const int32_t *) h;
    cnt[0] = w[0];
    g_k    = w[1];
    g_ntok = w[2];
    g_eps2 = w[3];
    for (int i = 0; i < KW * NCH; i += 32) {
        v32bf a, b;
        hl(aie::load_v<32>(h + 64 + i), a, b);
        aie::store_v(g_wh + i, a);
        aie::store_v(g_wl + i, b);
    }
}

// One chunk: `x` its rows, out the two GDN objects' K, Q, V (OBJ bf16 each).
void gdn_conv_chunk(const float *x, bfloat16 *out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t t0 = g_k * C;
    g_k += 2;
    bfloat16 *oa = out, *ob = out + OBJ;
    for (int tt = 0; tt < C; tt++) {
        const bool valid = t0 + tt < g_ntok;
        alignas(64) float row[NCH];
        // 64 channels a step, as two independent chains
        for (int d = 0; d < NCH; d += 64) {
            facc a0, a1;
            a0.from_vector(aie::zeros<float, 32>(), 0);
            a1.from_vector(aie::zeros<float, 32>(), 0);
#pragma clang loop unroll(full)
            for (int i = 0; i < KW; i++) {
                const float *xr = x + (tt + i) * NCH + d;
                v32bf xh0, xl0, xh1, xl1;
                hl(aie::load_v<32>(xr), xh0, xl0);
                hl(aie::load_v<32>(xr + 32), xh1, xl1);
                const v32bf wh0 = aie::load_v<32>(g_wh + i * NCH + d);
                const v32bf wh1 = aie::load_v<32>(g_wh + i * NCH + d + 32);
                a0 = aie::mac(a0, xh0, wh0);
                a1 = aie::mac(a1, xh1, wh1);
                a0 = aie::mac(a0, xh0, aie::load_v<32>(g_wl + i * NCH + d));
                a1 = aie::mac(a1, xh1, aie::load_v<32>(g_wl + i * NCH + d + 32));
                a0 = aie::mac(a0, xl0, wh0);
                a1 = aie::mac(a1, xl1, wh1);
            }
            v32f s0, s1;
            silu64(a0.to_vector<float>(0), a1.to_vector<float>(0), s0, s1);
            aie::store_v(row + d, valid ? s0 : aie::zeros<float, 32>());
            aie::store_v(row + d + 32, valid ? s1 : aie::zeros<float, 32>());
        }
        // q (channels 0 ..) and k (D ..) normalized; the object is K, then Q
        facc sq0, sq1;
        sq0.from_vector(aie::zeros<float, 32>(), 0);
        sq1.from_vector(aie::zeros<float, 32>(), 0);
#pragma clang loop unroll(full)
        for (int d = 0; d < D; d += 32) {
            v32bf h0, l0, h1, l1;
            hl(aie::load_v<32>(row + d), h0, l0);
            hl(aie::load_v<32>(row + D + d), h1, l1);
            sq0 = aie::mac(sq0, h0, h0);
            sq1 = aie::mac(sq1, h1, h1);
            sq0 = aie::mac(sq0, h0, l0);
            sq1 = aie::mac(sq1, h1, l1);
            sq0 = aie::mac(sq0, l0, h0);
            sq1 = aie::mac(sq1, l1, h1);
        }
        const v16f rqk = rsqrt_qk(sq0.to_vector<float>(0), sq1.to_vector<float>(0));
        for (int part = 0; part < 2; part++) {
            const float *p = row + part * D;
            // lane `part` everywhere (constant lanes: part is unrolled)
            const v16f r = aie::broadcast<float, 16>(part == 0 ? rqk[0] : rqk[1]);
            v32bf rh, rl;
            hl(aie::concat(r, r), rh, rl);
            const int base = part == 0 ? C * D : 0;
            for (int d = 0; d < D; d += 32) {
                v32bf h, l;
                hl(aie::load_v<32>(p + d), h, l);
                facc a = aie::mul(h, rh);
                a = aie::mac(a, h, rl);
                a = aie::mac(a, l, rh);
                const v32bf o = a.to_vector<bfloat16>(0);
                // constant eighths: extract<8>(k) with a runtime k picks the
                // wrong one
                put8(oa, ob, base + t8(tt, d, D), base + t8(tt, d + 8, D), base + t8(tt, d + 16, D),
                     base + t8(tt, d + 24, D), o);
            }
        }
        // v: half 0 to the first object, half 1 to the second
        for (int d = 0; d < D; d += 32) {
            const v32bf o = facc(aie::load_v<32>(row + 2 * D + d)).to_vector<bfloat16>(0);
            bfloat16 *dst = (d < DV ? oa : ob) + 2 * C * D;
            const int dd = d % DV;
            put8(dst, nullptr, t8(tt, dd, DV), t8(tt, dd + 8, DV), t8(tt, dd + 16, DV), t8(tt, dd + 24, DV), o);
        }
    }
}

} // extern "C"
