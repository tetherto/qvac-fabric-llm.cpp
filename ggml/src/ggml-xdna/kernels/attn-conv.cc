// The conv stage of the fused recurrent core: silu(conv1d) over the q/k/v
// channel slices a gated-delta-net head needs. Read by attn_cn.py (and
// attn_gdn_gated.py through it) and compiled by IRON as the "ggml_xdna_attn_conv" kernel.
//
// @GPO@ and @SLOT@ are filled in per build (CONV_GPO / CONV_SLOT); see
// attn_cn.py.
#include <aie_api/aie.hpp>
#include "silu-f32.h"
using namespace aie;
// gather fp32 src[base + i*stride] into an aligned bf16 buffer; the bf16
// conversion is the silu/gdn accum trick verbatim
static inline void f32gb(const float * src, int base, int stride, bfloat16 * dst) {
    alignas(64) float st[32];
    for (int i = 0; i < 32; ++i) {
        st[i] = src[base + i * stride];
    }
    for (int o = 0; o < 32; o += 16) {
        aie::accum<accfloat, 16> ga;
        ga.from_vector(aie::load_v<16>(st + o), 0);
        aie::store_v(dst + o, ga.to_vector<bfloat16>());
    }
}
extern "C" void ggml_xdna_attn_conv(const float * feed0, float * x0, float * hist0) {
    // Round fp32 -> bf16 to nearest-even. The core's default rounding mode
    // truncates toward zero, and the taps and weights rounded here came
    // out biased toward zero (see gdn-v.cc).
    aie::set_rounding(aie::rounding_mode::conv_even);
    // An object carries @GPO@ feed groups, not one: a 4 KB transfer never
    // reaches the shim's rate, and the stage's time is all transfer.
    for (int gpo_ = 0; gpo_ < @GPO@; ++gpo_) {
    const float * feed = feed0 + gpo_ * @SLOT@;
    float * x = x0 + gpo_ * @S_V@;
    float * hist = hist0 + gpo_ * 3 * @S_V@;
    const float * hi = feed + @F_H@;
    const float * qv = feed + @F_Q@;
    // With a short slot the weights are not in the object at all - this is the
    // diagnostic that prices the stage's bytes, and its output is wrong.
    const float * w = @SLOT@ < @FEED_N@ ? feed : feed + @F_W@;
    alignas(64) bfloat16 hb[3][32];
    alignas(64) bfloat16 wb[4][32];
    alignas(64) bfloat16 qv_b[32];
    // conv dot per 32 channels: gathered bf16 taps x weights -> fp32 accum
    for (int c = 0; c < @S_V@; c += 32) {
        for (int t = 0; t < 3; ++t) {
            f32gb(hi, c*3 + t, 3, hb[t]);
        }
        for (int t = 0; t < 4; ++t) {
            f32gb(w, c*4 + t, 4, wb[t]);
        }
        f32gb(qv, c, 1, qv_b);
        aie::accum<accfloat, 32> acc;
        acc = aie::mul(aie::load_v<32>(hb[0]), aie::load_v<32>(wb[0]));
        acc = aie::mac(acc, aie::load_v<32>(hb[1]), aie::load_v<32>(wb[1]));
        acc = aie::mac(acc, aie::load_v<32>(hb[2]), aie::load_v<32>(wb[2]));
        acc = aie::mac(acc, aie::load_v<32>(qv_b), aie::load_v<32>(wb[3]));
        // hold raw accs in the x output buffer, then overwrite with silu below
        aie::store_v(x + c, acc.to_vector<float>());
    }
    // history shift (3 taps per channel) stays scalar; hist is a separate BO
    for (int c = 0; c < @S_V@; ++c) {
        hist[c*3+0] = hi[c*3+1];
        hist[c*3+1] = hi[c*3+2];
        hist[c*3+2] = qv[c];
    }
    // silu on the hardware tanh, around fp32 accumulators (silu-f32.h: the
    // fp32 silu here costs 1.4 ms a token for KLD 0.0050 -> 0.0042)
    for (int o = 0; o < @S_V@; o += 16) {
        aie::store_v(x + o, silu_hw(aie::load_v<16>(x + o)));
    }
    }
}
