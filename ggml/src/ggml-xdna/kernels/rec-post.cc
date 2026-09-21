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
// where a tile is ACT_TILE bytes of K_TILE int8 codes, a f32 code sum and a
// f32 scale per group of 32, and the two trailing flag words.
#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;

extern "C" void ggml_xdna_post_norm(const float * in, uint8_t * out) {
    event0();
    const float * so = in;
    const float * res = in + @PD@;
    const float * gam = in + 2 * @PD@;
    // What the last tile tells the projection that reads it: close the chunk,
    // and for a pair, quantize the epilogue's output in the next format's
    // grouping. The host knows the weight types; the design does not.
    const int32_t last_flags = ((const int32_t *)in)[3 * @PD@];
    float * hattn = (float *)out;
    uint8_t * act = out + @PD@ * 4;

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
    const float scale = 1.0f / aie::sqrt(ss / (float)@PD@ + 1e-6f);
    const auto vscale = aie::broadcast<float, 16>(scale);

    // Pass two: scale by gamma and quantize. The activation layout must
    // match what the projection's kernel reads - the same two forms the
    // gated stage writes (rec-gated.cc), chosen by the model's ssm_out
    // weight type: GATED_FMT=1 is the 8-bit form (tiles of 128 codes,
    // groups of 16, gsum at tile+128, fmt word 1), the default the 4-bit
    // form (tiles of 256, groups of 32). A mismatch reads as garbage.
    int32_t * hdr = (int32_t *)act;
    hdr[0] = @PNT@;
    hdr[1] = 1;
    hdr[@PACT@ / 4 - 2] = 0;
#if defined(GATED_FMT) && GATED_FMT == 1
    hdr[@PACT@ / 4 - 1] = 1;
#else
    hdr[@PACT@ / 4 - 1] = 0;
#endif
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
#if defined(GATED_FMT) && GATED_FMT == 1
    const auto vlo = aie::broadcast<float, 16>(-128.0f);
#else
    const auto vlo = aie::broadcast<float, 16>(-127.0f);
#endif
    const auto vhi = aie::broadcast<float, 16>(127.0f);
    const auto magic = aie::broadcast<float, 16>(12582912.0f);
    const auto magici = magic.cast_to<int32>();
    for (int t = 0; t < @PNT@; t++) {
        uint8_t * tile = act + (1 + t) * @PACT@;
        int8 * code = (int8 *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
        float * gsum = (float *)(tile + 128);
        float * gd = gsum + 8;
        for (int g = 0; g < 8; g++) {
            const int base = t * 128 + g * 16;
            auto y0 = aie::mul(aie::mul(aie::load_v<16>(hattn + base), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base)).to_vector<float>();
            const auto a0 = aie::bit_and(y0.cast_to<int32>(), absmask)
                                .cast_to<float>();
            const float ga = aie::reduce_max(a0);
            const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
            const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
            auto q0 = aie::min(aie::max(aie::mul(y0, ginv).to_vector<float>(),
                                        vlo), vhi);
            const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(),
                                      magici);
            aie::store_v(code + g * 16, aie::pack(aie::pack(i0)));
            gsum[g] = (float) aie::reduce_add(i0);
            gd[g] = gdv;
        }
#else
        float * gsum = (float *)(tile + @PKT@);
        float * gd = gsum + @PGPT@;
        for (int g = 0; g < @PGPT@; g++) {
            const int base = t * @PKT@ + g * @PGRP@;
            auto y0 = aie::mul(aie::mul(aie::load_v<16>(hattn + base), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base)).to_vector<float>();
            auto y1 = aie::mul(aie::mul(aie::load_v<16>(hattn + base + 16), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base + 16)).to_vector<float>();
            auto a0 = aie::bit_and(y0.cast_to<int32>(), absmask).cast_to<float>();
            auto a1 = aie::bit_and(y1.cast_to<int32>(), absmask).cast_to<float>();
            auto vm = aie::max(a0, a1);
            alignas(64) float ml[16];
            aie::store_v(ml, vm);
            float amax = 0.0f;
            for (int i = 0; i < 16; i++) {
                if (ml[i] > amax) amax = ml[i];
            }
            const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
            const auto vinv = aie::broadcast<float, 16>(1.0f / d);
            auto q0 = aie::min(aie::max(aie::mul(y0, vinv).to_vector<float>(),
                                        vlo), vhi);
            auto q1 = aie::min(aie::max(aie::mul(y1, vinv).to_vector<float>(),
                                        vlo), vhi);
            auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
            auto i1 = aie::sub(aie::add(q1, magic).cast_to<int32>(), magici);
            aie::store_v(code + g * @PGRP@, aie::pack(aie::pack(i0)));
            aie::store_v(code + g * @PGRP@ + 16, aie::pack(aie::pack(i1)));
            // The projection's kernel folds the code sum against the weight
            // offsets, so it is summed here rather than there.
            auto s0 = aie::add(i0, i1);
            alignas(64) int32_t sl[16];
            aie::store_v(sl, s0);
            int sum = 0;
            for (int i = 0; i < 16; i++) {
                sum += sl[i];
            }
            gsum[g] = (float)sum;
            gd[g] = d;
        }
#endif
        int32_t * tw = (int32_t *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
        tw[@PACT@ / 4 - 1] = 1;
#else
        tw[@PACT@ / 4 - 1] = 0;
#endif
        tw[@PACT@ / 4 - 2] = (t == @PNT@ - 1) ? last_flags : 0;
    }
    event1();
}
