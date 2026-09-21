// The gated-attention epilogue of the fused recurrent core: per-head
// normalisation, silu and gamma (ggml_xdna_gated_head), then the row quantisation and
// the activation tiles the projection reads (ggml_xdna_gated_fin). Read by rec_gated.py
// and attn_gdn_gated.py and compiled by IRON as "ggml_xdna_gated_head" / "ggml_xdna_gated_fin".
//
// The per-build knobs (GATED_FMT, ACT_SPLIT, ACT_TILE, ACT_OFF, K_GATE) are
// compile flags; see attn_gdn_gated.py.
//
// State is carried in the OUT object so ggml_xdna_gated_head and ggml_xdna_gated_fin can share it
// without cross-TU statics (IRON compiles each ExternalFunction from the same
// source into its own TU). OUT layout:
//   [0 .. 8191]      gated f32 scratch (2048 floats), written per head
//   [8192 .. 10239]  aq int8 codes, one scale for the row
//   [10240 .. 10243] d_a f32
//   [ACT_OFF ..]     the same values as ssm_out's GEMV activation: a header
//                    tile then one per k_tile, int8 codes with a scale and a
//                    code sum per group of 32. Writing it here is what lets
//                    the projection read it without a repack on the host.
#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;

// az per head = [attn 128][z 128][hh] (hh at the tail keeps attn/z 64B-aligned;
// each call finds its slot by hh).
extern "C" void ggml_xdna_gated_head(uint8_t * out, const float * az, const float * gamma) {
    const int hh = (int)az[256];
    float * gbuf = (float *)out + hh * 128;
    const float * a = az;
    const float * z = az + 128;
    const auto bc_h = aie::broadcast<float, 16>(0.5f);
    const auto bc1 = aie::broadcast<bfloat16, 16>(1.0f);
    float ms = 0.0f;
    for (int i = 0; i < 128; i++) ms += a[i] * a[i];
    const float rsc = 1.0f / aie::sqrt(ms / 128.0f + 1e-6f);
    const auto bc_r = aie::broadcast<float, 16>(rsc);
    for (int i = 0; i < 128; i += 16) {
        auto a16 = aie::load_v<16>(a + i);
        auto z16 = aie::load_v<16>(z + i);
        auto g16 = aie::load_v<16>(gamma + i);
        // silu(z) = z * 0.5*(1+tanh(z/2)); tanh is hw bf16, lifted to float
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

#if ACT_SPLIT
extern "C" void ggml_xdna_gated_fin(uint8_t * out, uint8_t * actbuf) {
#else
extern "C" void ggml_xdna_gated_fin(uint8_t * out) {
#endif
    const float * gbuf = (const float *)out;
    // Both passes are vector work. They used to be scalar loops over 2048
    // floats with a divide per element, which measured as the single most
    // expensive thing in the fused core - more than the per-head epilogue and
    // four times the data movement of the whole stage.
    aie::vector<float, 16> vmax = aie::zeros<float, 16>();
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
    for (int k = 0; k < 2048; k += 16) {
        auto v = aie::load_v<16>(gbuf + k);
        // aie::abs does not hold for f32 on this target; mask the sign bit.
        auto av = aie::bit_and(v.cast_to<int32>(), absmask).cast_to<float>();
        vmax = aie::max(vmax, av);
    }
    alignas(64) float lanes[16];
    aie::store_v(lanes, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        if (lanes[i] > amax) amax = lanes[i];
    }

    const float da = amax > 0.0f ? amax / 127.0f : 1.0f;
    // One divide for the whole buffer instead of 2048 of them.
    const float inv = 1.0f / da;
    // The float-to-int step goes through the "magic constant": adding
    // 1.5*2^23 puts the rounded integer in the low mantissa bits, and
    // subtracting the constant's own bit pattern leaves it as an int32. The
    // library's to_fixed does not hold on this target, and the scalar loop
    // this replaces was the other half of the stage.
    const auto vinv  = aie::broadcast<float, 16>(inv);
    const auto vlo   = aie::broadcast<float, 16>(-128.0f);
    const auto vhi   = aie::broadcast<float, 16>(127.0f);
    const auto magic = aie::broadcast<float, 16>(12582912.0f);
    const auto magici = magic.cast_to<int32>();
    int8 * aq = (int8 *)(out + 8192);
    for (int k = 0; k < 2048; k += 16) {
        auto v = aie::mul(aie::load_v<16>(gbuf + k), vinv).to_vector<float>();
        v = aie::min(aie::max(v, vlo), vhi);
        // aie::add of two vectors is a vector, not an accumulator.
        auto qi = aie::sub(aie::add(v, magic).cast_to<int32>(), magici);
        aie::store_v(aq + k, aie::pack(aie::pack(qi)));
    }
    float * dap = (float *)(out + 10240);
    dap[0] = da;

    // The same codes again in the GEMV's activation tile layout, but with a
    // scale and a code sum per group of 32 rather than one for the row: a
    // group scale is local to the values a core holds, which is what the
    // projection's kernel reads. Only the 4-bit weight form is covered - the
    // 8-bit one groups by 16 - so the host keeps the other.
    {
#if defined(GATED_FMT) && GATED_FMT == 1
        // The 8-bit weight form: tiles of 128 codes, groups of 16, the layout
        // the projection's q8 kernel reads. Which form the model needs is a
        // property of its ssm_out weights - Q5_K and Q6_K are the 8-bit form -
        // so the design is built per model (GATED_FMT=1) and the tag covers it.
        const int NT = K_GATE / 128;
#else
        const int NT = K_GATE / 256;
#endif
        // A drain of its own when the activation is split off: the projection
        // that reads it back in the same stream needs a plainly patched
        // descriptor, and the gated output's own drain cannot have one.
#if ACT_SPLIT
        uint8_t * act = actbuf;
#else
        uint8_t * act = out + ACT_OFF;
#endif
        int32_t * hdr = (int32_t *)act;
        hdr[0] = NT;
        hdr[1] = 1;
        hdr[ACT_TILE / 4 - 2] = 0;
#if defined(GATED_FMT) && GATED_FMT == 1
        hdr[ACT_TILE / 4 - 1] = 1;
#else
        hdr[ACT_TILE / 4 - 1] = 0;
#endif
        for (int t = 0; t < NT; t++) {
            uint8_t * tile = act + (1 + t) * ACT_TILE;
            int8 * code = (int8 *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
            float * gsum = (float *)(tile + 128);
            float * gd = gsum + 8;
            for (int g = 0; g < 8; g++) {
                const float * v = gbuf + t * 128 + g * 16;
                const auto v0 = aie::load_v<16>(v);
                const auto a0 = aie::bit_and(v0.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                const float ga = aie::reduce_max(a0);
                const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
                const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
                auto q0 = aie::mul(v0, ginv).to_vector<float>();
                q0 = aie::min(aie::max(q0, vlo), vhi);
                const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
                aie::store_v(code + g * 16, aie::pack(aie::pack(i0)));
                gsum[g] = (float) aie::reduce_add(i0);
                gd[g] = gdv;
            }
            ((int32_t *)tile)[ACT_TILE / 4 - 2] = 0;
            ((int32_t *)tile)[ACT_TILE / 4 - 1] = 1;
#else
            float * gsum = (float *)(tile + 256);
            float * gd = gsum + 8;
            for (int g = 0; g < 8; g++) {
                const float * v = gbuf + t * 256 + g * 32;
                const auto v0 = aie::load_v<16>(v);
                const auto v1 = aie::load_v<16>(v + 16);
                const auto a0 = aie::bit_and(v0.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                const auto a1 = aie::bit_and(v1.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                // reduce_max, not a scalar scan of the sixteen lanes: at
                // sixty-four groups a branchy scalar loop per group cost the
                // stage 50 us.
                const float ga = aie::reduce_max(aie::max(a0, a1));
                const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
                const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
                auto q0 = aie::mul(v0, ginv).to_vector<float>();
                auto q1 = aie::mul(v1, ginv).to_vector<float>();
                q0 = aie::min(aie::max(q0, vlo), vhi);
                q1 = aie::min(aie::max(q1, vlo), vhi);
                const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
                const auto i1 = aie::sub(aie::add(q1, magic).cast_to<int32>(), magici);
                aie::store_v(code + g * 32, aie::pack(aie::pack(i0)));
                aie::store_v(code + g * 32 + 16, aie::pack(aie::pack(i1)));
                gsum[g] = (float) aie::reduce_add(aie::add(i0, i1));
                gd[g] = gdv;
            }
            ((int32_t *)tile)[ACT_TILE / 4 - 2] = 0;
            ((int32_t *)tile)[ACT_TILE / 4 - 1] = 0;
#endif
        }
    }

}
