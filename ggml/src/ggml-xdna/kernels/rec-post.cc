// The transition between a layer's ssm_out projection and its FFN, on the
// array instead of the host: add the residual, normalise, scale by the layer's
// gamma and quantize into the activation tiles the next projection reads.
// Read by rec_post.py and attn_gdn_gated.py and compiled by IRON as
// "ggml_xdna_post_norm".
//
// It exists to remove the host from the middle of a layer. As long as the host
// computes this, the projection before it and the projection after it cannot
// be in one dispatch, and the array's fixed per-phase cost is paid twice.
//
// PD, PNT, PKT, PGPT, PACT and GATED_FMT are compile flags; see
// attn_gdn_gated.py. They are inlined below as literals by rec_post.py.
//
// One object in, one object out, so the tile needs a single channel each way:
//   in   [so_out D f32][residual D f32][gamma D f32][flags i32][pad]
//   out  [hattn D f32][header tile][NT tiles]
// where a tile is ACT_TILE bytes of ACT_KT int8 codes, a f32 code sum and a
// f32 scale per group of PGRP, and the two trailing flag words.
#include "xdna-math.h"

#include <math.h>
#include <stdint.h>

#include <aie_api/aie.hpp>
using namespace aie;

namespace {

// The activation tiles this kernel writes are the ones the gated stage
// (rec-gated.cc) writes, chosen by the model's ssm_out weight type: the 8-bit
// form is tiles of 128 codes with the signed code floor, the default the 4-bit
// form (tiles of PKT codes). A mismatch reads as garbage.
#if defined(GATED_FMT) && GATED_FMT == 1
constexpr int   ACT_KT   = 128;
constexpr int   ACT_FMTW = 1;
constexpr float ACT_VLO  = xdna::CODE_MIN;
#else
constexpr int   ACT_KT   = @PKT@;
constexpr int   ACT_FMTW = 0;
constexpr float ACT_VLO  = -xdna::CODE_MAX;
#endif
constexpr int ACT_NG      = ACT_KT / @PGRP@;  // groups a tile
constexpr int ACT_FLAGS_W = @PACT@ / 4 - 2;
constexpr int ACT_WIDTH_W = @PACT@ / 4 - 1;

}  // namespace

extern "C" void ggml_xdna_post_norm(const float * in, uint8_t * out) {
    // The rounding mode is the core's, left by whatever ran on it before:
    // set it, or the first dispatch after another design rounds differently.
    aie::set_rounding(aie::rounding_mode::conv_even);
    event0();
    const float * so         = in;
    const float * res        = in + @PD@;
    const float * gam        = in + 2 * @PD@;
    // What the last tile tells the projection that reads it: close the chunk,
    // and for a pair, quantize the epilogue's output in the next format's
    // grouping. The host knows the weight types; the design does not.
    const int32_t last_flags = ((const int32_t *) in)[3 * @PD@];
    float *       hattn      = (float *) out;
    uint8_t *     act        = out + @PD@ * 4;

    // Pass one: the residual add, kept in hattn - which the next layer reads
    // as its own residual - and the sum of squares along with it.
    aie::vector<float, 16> sq = aie::zeros<float, 16>();
    for (int i = 0; i < @PD@; i += 16) {
        auto v = aie::add(aie::load_v<16>(so + i), aie::load_v<16>(res + i));
        aie::store_v(hattn + i, v);
        sq = aie::add(sq, aie::mul(v, v).to_vector<float>());
    }
    alignas(64) float lanes[16];
    aie::store_v(lanes, sq);
    float ss = 0.0f;
    for (int i = 0; i < 16; i++) {
        ss += lanes[i];
    }
    const auto vscale = aie::broadcast<float, 16>(xdna::rms_scale(ss, @PD@, xdna::RMS_EPS));

    // Pass two: scale by gamma and quantize.
    int32_t * hdr    = (int32_t *) act;
    hdr[0]           = @PNT@;
    hdr[1]           = 1;
    hdr[ACT_FLAGS_W] = 0;
    hdr[ACT_WIDTH_W] = ACT_FMTW;
    const auto vlo   = aie::broadcast<float, 16>(ACT_VLO);
    for (int t = 0; t < @PNT@; t++) {
        uint8_t * tile = act + (1 + t) * @PACT@;
        int8 *    code = (int8 *) tile;
        float *   gsum = (float *) (tile + ACT_KT);
        float *   gd   = gsum + ACT_NG;
        for (int g = 0; g < ACT_NG; g++) {
            const int base = t * ACT_KT + g * @PGRP@;
            auto      y0   = aie::mul(aie::mul(aie::load_v<16>(hattn + base), vscale).to_vector<float>(),
                                      aie::load_v<16>(gam + base))
                          .to_vector<float>();
            auto y1 = aie::mul(aie::mul(aie::load_v<16>(hattn + base + 16), vscale).to_vector<float>(),
                               aie::load_v<16>(gam + base + 16))
                          .to_vector<float>();
            float     gd_g;
            // The projection's kernel folds the code sum against the weight
            // offsets, so it is summed here rather than there.
            const int sum = xdna::quant_group(y0, y1, vlo, (int8_t *) (code + g * @PGRP@), gd_g);
            gsum[g]       = (float) sum;
            gd[g]         = gd_g;
        }
        ((int32_t *) tile)[ACT_WIDTH_W] = ACT_FMTW;
        ((int32_t *) tile)[ACT_FLAGS_W] = (t == @PNT@ - 1) ? last_flags : 0;
    }
    event1();
}
