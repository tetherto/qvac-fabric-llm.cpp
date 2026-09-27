// silu(x) = x * sigmoid(x) in fp32, for the fused layer's stages.
//
// AIE2P's tanh and exp2 are hardware instructions with a bf16 result, and
// the silu built on them (x * 0.5 * (1 + tanh(x/2))) was the largest error of
// the fused recurrent core: 1.5% on the conv stage's output against 0.42% for
// the same stage with an exact silu, and the gated stage's silu(z) on top.
// Here exp is built from fp32 vector arithmetic - 2^n from the exponent bits,
// 2^f from a degree-6 polynomial - and 1 / (1 + e) from a bit-trick guess and
// three Newton steps: fp32 accuracy, about twenty vector operations for
// sixteen values.
#pragma once
#include <aie_api/aie.hpp>

static inline aie::vector<float, 16> silu_f32(aie::vector<float, 16> x) {
    const auto bc = [](float v) { return aie::broadcast<float, 16>(v); };
    // t = -x * log2(e), clamped where e^-x under- or overflows the exponent
    auto t = aie::mul(x, bc(-1.4426950408889634f)).to_vector<float>();
    t = aie::min(aie::max(t, bc(-126.0f)), bc(126.0f));
    // n = round(t) with the 1.5 * 2^23 trick, f = t - n in [-0.5, 0.5]
    const auto magic = bc(12582912.0f);
    const auto tm = aie::add(t, magic);
    const auto n = aie::sub(tm, magic);
    const auto f = aie::sub(t, n);
    const auto ni = aie::sub(tm.cast_to<int32>(), magic.cast_to<int32>());
    // 2^f, minimax on [-0.5, 0.5], relative error ~2e-7
    auto p = aie::add(aie::mul(f, bc(1.5403530393381606e-4f)).to_vector<float>(), bc(1.3333558146428443e-3f));
    p = aie::add(aie::mul(p, f).to_vector<float>(), bc(9.6181291076284772e-3f));
    p = aie::add(aie::mul(p, f).to_vector<float>(), bc(5.5504108664821580e-2f));
    p = aie::add(aie::mul(p, f).to_vector<float>(), bc(2.4022650695910071e-1f));
    p = aie::add(aie::mul(p, f).to_vector<float>(), bc(6.9314718055994531e-1f));
    p = aie::add(aie::mul(p, f).to_vector<float>(), bc(1.0f));
    // 2^n: the biased exponent in the float's bits
    const auto two_n = aie::add(ni, aie::broadcast<int32, 16>(127))
                           .cast_to<int32>();
    const auto sc = aie::upshift(two_n, 23).cast_to<float>();
    const auto e = aie::mul(p, sc).to_vector<float>();
    // r = 1 / d, d = 1 + e >= 1: a bit-trick first guess, then Newton
    const auto d = aie::add(e, bc(1.0f));
    auto r = aie::sub(aie::broadcast<int32, 16>(0x7EF311C7), d.cast_to<int32>()).cast_to<float>();
    for (int it = 0; it < 3; ++it) {
        const auto dr = aie::mul(d, r).to_vector<float>();
        r = aie::mul(r, aie::sub(bc(2.0f), dr)).to_vector<float>();
    }
    return aie::mul(x, r).to_vector<float>();
}

// silu on the hardware tanh (bf16 result), for a stage whose output does not
// need more: x * 0.5 * (1 + tanh(x/2)).
static inline aie::vector<float, 16> silu_hw(aie::vector<float, 16> x) {
    const auto h = aie::broadcast<float, 16>(0.5f);
    const auto th = aie::tanh(aie::mul(x, h).to_vector<float>());
    aie::accum<accfloat, 16> ta;
    ta.from_vector(th, 0);
    const auto sig = aie::add(aie::mul(ta.to_vector<float>(), h).to_vector<float>(), h);
    return aie::mul(x, sig).to_vector<float>();
}
