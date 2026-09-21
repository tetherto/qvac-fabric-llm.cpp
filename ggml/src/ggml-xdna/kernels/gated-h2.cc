// The per-head gated epilogue of the fused recurrent core: normalization,
// silu and gamma for HG heads carried in one object. Read by
// attn_gdn_gated.py and compiled by IRON as "ggml_xdna_gated_h2". The same math as
// rec-gated.cc's ggml_xdna_gated_head, with gamma read from the object.
//
// HG, AZG_N and ATTN_ONCHIP are compile flags; see attn_gdn_gated.py.
#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;

#if ATTN_ONCHIP
extern "C" void ggml_xdna_gated_h2(uint8_t * out, const float * azg4,
                         const float * att_lo, const float * att_hi) {
#else
extern "C" void ggml_xdna_gated_h2(uint8_t * out, const float * azg4) {
#endif
    // One object carries HG heads. The stage is one tile walking them in
    // order, and with a head per object the fifo handshake, not the
    // arithmetic, was what it spent its time on: stubbing both of its kernels
    // left the stage exactly as expensive.

    for (int hj = 0; hj < HG; hj++) {
    const float * azg = azg4 + hj * AZG_N;
    const int hh = (int)azg[384];
    float * gbuf = (float *)out + hh * 128;
#if ATTN_ONCHIP
    // The gdn block hands its attn over on chip, a round at a time, so a head
    // arrives as two objects of NC_GDN chunks. Copying them into one local
    // array costs eight vector moves and leaves the rest of the kernel - and
    // the DDR layout it falls back to - exactly as it was.
    alignas(64) float aloc[128];
    for (int i = 0; i < 64; i += 16) {
        aie::store_v(aloc + i,      aie::load_v<16>(att_lo + i));
        aie::store_v(aloc + 64 + i, aie::load_v<16>(att_hi + i));
    }
    const float * a = aloc;
#else
    const float * a = azg;
#endif
    const float * z = azg + 128;
    const float * gamma = azg + 256;
    const auto bc_h = aie::broadcast<float, 16>(0.5f);
    const auto bc1 = aie::broadcast<bfloat16, 16>(1.0f);
    // Vector sum of squares; the scalar loop this replaces was 128 dependent
    // float adds per head.
    aie::vector<float, 16> sq = aie::zeros<float, 16>();
    for (int i = 0; i < 128; i += 16) {
        auto v = aie::load_v<16>(a + i);
        sq = aie::add(sq, aie::mul(v, v).to_vector<float>());
    }
    alignas(64) float sql[16];
    aie::store_v(sql, sq);
    float ms = 0.0f;
    for (int i = 0; i < 16; i++) ms += sql[i];
    const float rsc = 1.0f / aie::sqrt(ms / 128.0f + 1e-6f);
    const auto bc_r = aie::broadcast<float, 16>(rsc);
    for (int i = 0; i < 128; i += 16) {
        auto a16 = aie::load_v<16>(a + i);
        auto z16 = aie::load_v<16>(z + i);
        auto g16 = aie::load_v<16>(gamma + i);
        auto t16 = aie::mul(z16, bc_h).to_vector<float>();
        auto thb = aie::tanh(t16);
        auto th  = aie::mul(thb, bc1).to_vector<float>();
        auto thh = aie::mul(th, bc_h).to_vector<float>();
        auto sig = aie::add(thh, bc_h);
        auto silu = aie::mul(z16, sig).to_vector<float>();
        auto g1 = aie::mul(a16, bc_r).to_vector<float>();
        auto g2 = aie::mul(g1, g16).to_vector<float>();
        aie::vector<float, 16> g = aie::mul(g2, silu).to_vector<float>();
        aie::store_v(gbuf + i, g);
    }
    }
}
