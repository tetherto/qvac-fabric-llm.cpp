// The per-head gated epilogue of the fused recurrent core: normalization,
// silu and gamma for HG heads carried in one object. Read by
// attn_gdn_gated.py and compiled by IRON as "ggml_xdna_gated_h2". The same math as
// rec-gated.cc's ggml_xdna_gated_head, with gamma read from the object.
//
// HG, AZG_N and ATTN_ONCHIP are compile flags; see attn_gdn_gated.py.
#include "xdna-math.h"

#include <math.h>
#include <stdint.h>

#include <aie_api/aie.hpp>
using namespace aie;

namespace {

constexpr int GH_D      = 128;       // head dim: attn, z and gamma
constexpr int GH_Z_OFF  = GH_D;      // z after attn
constexpr int GH_G_OFF  = 2 * GH_D;  // gamma after z
constexpr int GH_HH_OFF = 3 * GH_D;  // the head's slot, at the tail

}  // namespace

#if ATTN_ONCHIP
extern "C" void ggml_xdna_gated_h2(uint8_t * out, const float * azg4, const float * att_lo, const float * att_hi) {
    // The rounding mode is the core's, left by whatever ran on it before:
    // set it, or the first dispatch after another design rounds differently.
    aie::set_rounding(aie::rounding_mode::conv_even);
#else
extern "C" void ggml_xdna_gated_h2(uint8_t * out, const float * azg4) {
    // The rounding mode is the core's, left by whatever ran on it before:
    // set it, or the first dispatch after another design rounds differently.
    aie::set_rounding(aie::rounding_mode::conv_even);
#endif
    // One object carries HG heads. The stage is one tile walking them in
    // order, and with a head per object the fifo handshake, not the
    // arithmetic, was what it spent its time on: stubbing both of its kernels
    // left the stage exactly as expensive.

    for (int hj = 0; hj < HG; hj++) {
        const float * azg  = azg4 + hj * AZG_N;
        const int     hh   = (int) azg[GH_HH_OFF];
        float *       gbuf = (float *) out + hh * GH_D;
#if ATTN_ONCHIP
        // The gdn block hands its attn over on chip, a round at a time, so a head
        // arrives as two objects of NC_GDN chunks. Copying them into one local
        // array costs eight vector moves and leaves the rest of the kernel - and
        // the DDR layout it falls back to - exactly as it was.
        alignas(64) float aloc[GH_D];
        for (int i = 0; i < GH_D / 2; i += 16) {
            aie::store_v(aloc + i, aie::load_v<16>(att_lo + i));
            aie::store_v(aloc + GH_D / 2 + i, aie::load_v<16>(att_hi + i));
        }
        const float * a = aloc;
#else
        const float * a = azg;
#endif
        const float *          z     = azg + GH_Z_OFF;
        const float *          gamma = azg + GH_G_OFF;
        // Vector sum of squares; the scalar loop this replaces was 128 dependent
        // float adds per head.
        aie::vector<float, 16> sq    = aie::zeros<float, 16>();
        for (int i = 0; i < GH_D; i += 16) {
            auto v = aie::load_v<16>(a + i);
            sq     = aie::add(sq, aie::mul(v, v).to_vector<float>());
        }
        alignas(64) float sql[16];
        aie::store_v(sql, sq);
        float ms = 0.0f;
        for (int i = 0; i < 16; i++) {
            ms += sql[i];
        }
        xdna::gated_head(gbuf, a, z, gamma, xdna::rms_scale(ms, GH_D, xdna::RMS_EPS));
    }
}
