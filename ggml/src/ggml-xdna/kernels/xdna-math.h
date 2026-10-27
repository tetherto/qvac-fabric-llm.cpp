#pragma once
// Shared nonlinear math for the AIE kernels. The cores emulate f32 products
// (or have bf16-only lookup tables), so every function here is built from
// native bf16 multiply-accumulates into the f32 accumulator and integer
// exponent-field arithmetic. Each site's accuracy is a deliberate choice and
// is kept as a named variant rather than folded into one compromise.
#include "xdna-vec.h"

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace xdna {

template <unsigned N> inline aie::vector<float, N> bc(float v) {
    return aie::broadcast<float, N>(v);
}

template <unsigned N> inline aie::vector<int32_t, N> bci(int32_t v) {
    return aie::broadcast<int32_t, N>(v);
}

using v16f  = aie::vector<float, 16>;
using v16i  = aie::vector<int32_t, 16>;
using v32f  = aie::vector<float, 32>;
using v16bf = aie::vector<bfloat16, 16>;
using v32bf = aie::vector<bfloat16, 32>;

// The conversions between an exponential and a power of two.
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.69314718055994531f;

// The GDN attention scale, 1 / sqrt(DK) for the DK = 128 the model carries.
constexpr float GDN_SCALE = 0.08838834764831845f;

// The epsilon a kernel bakes into its rms normalisation.
constexpr float RMS_EPS = 1e-6f;

// x = n + f with n the nearest integer (nearest even) and f in [-0.5, 0.5]:
// 1.5 * 2^23 puts n in the low mantissa bits of the f32 sum.
__attribute__((always_inline)) inline v16f exp2_frac(const v16f & x, v16i & n) {
    const v16f magic = bc<16>(12582912.0f);
    const v16f y     = aie::add(x, magic);
    n                = aie::sub(y.cast_to<int32_t>(), bci<16>(0x4B400000));
    return aie::sub(x, aie::sub(y, magic));
}

// 2^f on [-0.5, 0.5], degree-6 minimax, relative error ~2e-7.
__attribute__((always_inline)) inline v16f exp2_poly(const v16f & f) {
    const float c[6] = { 1.3333558146428443e-3f, 9.6181291076284772e-3f, 5.5504108664821580e-2f,
                         2.4022650695910071e-1f, 6.9314718055994531e-1f, 1.0f };
    v16f        p    = bc<16>(1.5403530393381606e-4f);
    for (int i = 0; i < 6; i++) {
        p = aie::add(aie::mul(p, f).to_vector<float>(), bc<16>(c[i]));
    }
    return p;
}

// 2^t for a t already in log2 units: 2^n in the exponent field, 2^f from the
// polynomial.
__attribute__((always_inline)) inline v16f exp2_t(const v16f & t) {
    v16i       n;
    const v16f f  = exp2_frac(t, n);
    const v16f p  = exp2_poly(f);
    const v16f sc = aie::upshift(aie::add(n, bci<16>(127)), 23).cast_to<float>();
    return aie::mul(p, sc).to_vector<float>();
}

// 2^x on 16 f32 lanes. n is clamped after the split, where the exponent under-
// or overflows.
__attribute__((always_inline)) inline v16f exp2_f32(const v16f & x) {
    v16i       n;
    const v16f p = exp2_poly(exp2_frac(x, n));
    n            = aie::min(aie::max(n, bci<16>(-126)), bci<16>(127));
    return aie::add(p.cast_to<int32_t>(), aie::upshift(n, 23)).cast_to<float>();
}

// e^x on 16 f32 lanes, as 2^(x log2 e).
__attribute__((always_inline)) inline v16f exp_f32(const v16f & x) {
    v16f t = aie::mul(x, bc<16>(LOG2E)).to_vector<float>();
    t      = aie::min(aie::max(t, bc<16>(-126.0f)), bc<16>(126.0f));
    return exp2_t(t);
}

// 1 / d for d > 0: a bit-trick seed and three Newton steps, ~1e-7 relative.
// Out of line: three call sites share one copy, which the tight cores' program
// memory needs.
__attribute__((noinline)) inline v16f recip_f32(v16f d) {
    const v16f two = bc<16>(2.0f);
    v16f       r   = aie::sub(bci<16>(0x7EF311C7), d.cast_to<int32_t>()).cast_to<float>();
    for (int it = 0; it < 3; ++it) {
        r = aie::mul(r, aie::sub(two, aie::mul(d, r).to_vector<float>())).to_vector<float>();
    }
    return r;
}

// Scalar float arithmetic has no hardware on these cores, so every divide,
// multiply, add, compare and conversion is a software routine of a few hundred
// bytes. These do it in one vector lane, out of line so the callers share one
// copy.
__attribute__((noinline)) inline v16f vec_mul(v16f a, v16f b) {
    return aie::mul(a, b).to_vector<float>();
}

__attribute__((noinline, minsize)) inline float sc_div(float a, float b) {
    return vec_mul(bc<16>(a), recip_f32(bc<16>(b))).get(0);
}

inline float sc_mul(float a, float b) {
    return vec_mul(bc<16>(a), bc<16>(b)).get(0);
}

inline float sc_add(float a, float b) {
    return aie::add(bc<16>(a), bc<16>(b)).get(0);
}

inline float sc_max(float a, float b) {
    return aie::max(bc<16>(a), bc<16>(b)).get(0);
}

inline float sc_i2f(int i) {
    return aie::to_float(bci<16>(i), 0).get(0);
}

// a > 0 for a finite value: the sign and the bits, as an integer.
inline bool sc_pos(float a) {
    int32_t w;
    __builtin_memcpy(&w, &a, 4);
    return w > 0;
}

// 1 / sqrt(mean(x^2) + eps), the rms normalisation's scale.
inline float rms_scale(float ss, int n, float eps) {
    return 1.0f / aie::sqrt(ss / (float) n + eps);
}

// The int8 code range an activation tile's scale divides by. The group
// quantiser clamps to it; the gated epilogue's row quantiser uses the signed
// floor (-128) while the GEMV's group quantiser keeps the symmetric (-127) one.
constexpr float CODE_MAX = 127.0f;
constexpr float CODE_MIN = -128.0f;

// The integer a value rounds to, nearest even, clamped into the code range
// first: 1.5 * 2^23 puts it in the low mantissa bits of the f32 sum and the
// constant's own bit pattern subtracts back out. The library's to_fixed does
// not hold on this target.
inline v16i quant_i32(v16f v, v16f lo, v16f hi) {
    v                = aie::min(aie::max(v, lo), hi);
    const v16f magic = bc<16>(12582912.0f);
    return aie::sub(aie::add(v, magic).cast_to<int32_t>(), magic.cast_to<int32_t>());
}

// One group of 32 values: its scale (max |v| / CODE_MAX, or 1 for an all-zero
// group, whose codes are zero whatever the scale is), the codes as int8 into
// `code`, and their sum. The activation tile carries the scale, so it is
// written back to the caller.
inline int quant_group(v16f v0, v16f v1, v16f lo, int8_t * code, float & scale) {
    const auto  absmask = bci<16>(0x7FFFFFFF);
    const auto  a       = aie::max(aie::bit_and(v0.cast_to<int32_t>(), absmask).cast_to<float>(),
                                   aie::bit_and(v1.cast_to<int32_t>(), absmask).cast_to<float>());
    const float ga      = aie::reduce_max(a);
    scale               = ga > 0.0f ? ga / CODE_MAX : 1.0f;
    const v16f ginv     = bc<16>(1.0f / scale);
    const v16i i0       = quant_i32(aie::mul(v0, ginv).to_vector<float>(), lo, bc<16>(CODE_MAX));
    const v16i i1       = quant_i32(aie::mul(v1, ginv).to_vector<float>(), lo, bc<16>(CODE_MAX));
    aie::store_v(code, aie::pack(aie::pack(i0)));
    aie::store_v(code + 16, aie::pack(aie::pack(i1)));
    return aie::reduce_add(aie::add(i0, i1));
}

// The bits of an activation tile's flags word, the one before its last. The
// prologue (act-att.cc) and the GEMV cores (gemv-q4.cc) share the word.
enum : int {
    ACT_FLAG_EPI  = 1 << 0,  // the accumulation is complete: close the gate/up epilogue
    ACT_FLAG_DEVL = 1 << 1,  // the activation came from a core, not the host
    ACT_FLAG_QOUT = 1 << 2,  // write the epilogue as an activation tile, not f32
    ACT_FLAG_G16  = 1 << 3,  // group the written tile by Q8_GROUP
    ACT_FLAG_RMS  = 1 << 4,  // the prologue left the norm's rsqrt in the tile
    ACT_FLAG_ATTN = 1 << 5,  // the attention layer's / the layer boundary's work
};

// e^x for the attention prologue: the degree-4 polynomial (4e-5 at the ends of
// [-0.5, 0.5], two fp32 products less), its products out of line. A separate
// entry point from exp_f32 because that core's program memory cannot afford
// the degree-6 form.
__attribute__((noinline, minsize)) inline v16f exp_f32_fast(v16f x) {
    v16f t = vec_mul(x, bc<16>(LOG2E));
    t      = aie::min(aie::max(t, bc<16>(-126.0f)), bc<16>(126.0f));
    v16i       n;
    const v16f f  = exp2_frac(t, n);
    v16f       p  = aie::add(vec_mul(f, bc<16>(9.6181291076284772e-3f)), bc<16>(5.5504108664821580e-2f));
    p             = aie::add(vec_mul(p, f), bc<16>(2.4022650695910071e-1f));
    p             = aie::add(vec_mul(p, f), bc<16>(6.9314718055994531e-1f));
    p             = aie::add(vec_mul(p, f), bc<16>(1.0f));
    const v16f sc = aie::upshift(aie::add(n, bci<16>(127)), 23).cast_to<float>();
    return vec_mul(p, sc);
}

// log2(y) for y >= 1: the exponent from the bits, the mantissa folded into
// [sqrt(1/2), sqrt(2)) and ln m = 2 atanh((m - 1) / (m + 1)) to t^7.
__attribute__((noinline, minsize)) inline v16f log2_f32(v16f y) {
    const v16i b   = y.cast_to<int32_t>();
    v16i       e   = aie::sub(aie::bit_and(aie::downshift(b, 23), bci<16>(0xFF)), bci<16>(127));
    v16f       m   = aie::bit_or(aie::bit_and(b, bci<16>(0x7FFFFF)), bci<16>(0x3F800000)).cast_to<float>();
    const auto big = aie::gt(m, bc<16>(1.41421356f));
    m              = aie::select(m, aie::mul(m, bc<16>(0.5f)).to_vector<float>(), big);
    e              = aie::select(e, aie::add(e, bci<16>(1)), big);
    const v16f t   = vec_mul(aie::sub(m, bc<16>(1.0f)), recip_f32(aie::add(m, bc<16>(1.0f))));
    const v16f t2  = vec_mul(t, t);
    v16f       p   = aie::add(vec_mul(t2, bc<16>(1.0f / 7)), bc<16>(1.0f / 5));
    p              = aie::add(vec_mul(p, t2), bc<16>(1.0f / 3));
    p              = aie::add(vec_mul(p, t2), bc<16>(1.0f));
    const v16f ln  = vec_mul(vec_mul(p, t), bc<16>(2.0f));
    return aie::add(vec_mul(ln, bc<16>(LOG2E)), aie::to_float(e, 0));
}

// silu(x) = x / (1 + e^-x) in f32: e^-x as 2^(-x log2 e).
__attribute__((always_inline)) inline v16f silu_f32(const v16f & x) {
#if defined(SILU_IDENTITY)
    // Diagnostic: the epilogue runs but the nonlinearity is the identity.
    return x;
#else
    v16f t       = aie::mul(x, bc<16>(-LOG2E)).to_vector<float>();
    t            = aie::min(aie::max(t, bc<16>(-126.0f)), bc<16>(126.0f));
    const v16f d = aie::add(exp2_t(t), bc<16>(1.0f));
    return aie::mul(x, recip_f32(d)).to_vector<float>();
#endif
}

// The per-head epilogue of the fused recurrent core: 128 values of
// a * rsc * gamma * silu(z), with rsc = 1 / rms(a) the caller's.
inline void gated_head(float * g, const float * a, const float * z, const float * gamma, float rsc) {
    const v16f r = bc<16>(rsc);
    for (int i = 0; i < 128; i += 16) {
        const v16f g1 = aie::mul(aie::load_v<16>(a + i), r).to_vector<float>();
        const v16f g2 = aie::mul(g1, aie::load_v<16>(gamma + i)).to_vector<float>();
        aie::store_v(g + i, aie::mul(g2, silu_f32(aie::load_v<16>(z + i))).to_vector<float>());
    }
}

// silu on the hardware tanh (bf16 result), for a stage whose output does not
// need more: x * 0.5 * (1 + tanh(x / 2)).
inline v16f silu_hw(const v16f & x) {
    const v16f               h  = bc<16>(0.5f);
    const auto               th = aie::tanh(aie::mul(x, h).to_vector<float>());
    aie::accum<accfloat, 16> ta;
    ta.from_vector(th, 0);
    const v16f sig = aie::add(aie::mul(ta.to_vector<float>(), h).to_vector<float>(), h);
    return aie::mul(x, sig).to_vector<float>();
}

// 2^f for |f| <= 0.5 to bf16 precision: a cubic in native bf16 products
// accumulated in f32. The caller folds 2^n into the result's exponent field.
__attribute__((always_inline)) inline v32f exp2_bf16_poly(const v32bf & fb) {
    const v32bf              c1h = aie::broadcast<bfloat16, 32>((bfloat16) 0.69140625f);
    const v32bf              c1l = aie::broadcast<bfloat16, 32>((bfloat16) (0.69314718f - 0.69140625f));
    const v32bf              c2  = aie::broadcast<bfloat16, 32>((bfloat16) 0.24022651f);
    const v32bf              c3  = aie::broadcast<bfloat16, 32>((bfloat16) 0.05550411f);
    const v32bf              f2  = aie::mul(fb, fb).to_vector<bfloat16>(0);
    const v32bf              f3  = aie::mul(f2, fb).to_vector<bfloat16>(0);
    aie::accum<accfloat, 32> ea;
    ea.from_vector(bc<32>(1.0f), 0);
    ea = aie::mac(ea, fb, c1h);
    ea = aie::mac(ea, fb, c1l);
    ea = aie::mac(ea, f2, c2);
    ea = aie::mac(ea, f3, c3);
    return ea.to_vector<float>(0);
}

// 2^t for 32 f32 lanes to bf16 precision: n in the exponent field, 2^f from
// exp2_bf16_poly. Returns the two 16-lane halves.
__attribute__((always_inline)) inline void exp2_bf16_32(const v32f & t, v16f & e0, v16f & e1) {
    const v16f magic = bc<16>(12582912.0f);
    const v16i mbits = bci<16>(0x4B400000);
    const v16i nmin  = bci<16>(-126);
    const v16i nmax  = bci<16>(126);
    const v16f t0 = t.template extract<16>(0), t1 = t.template extract<16>(1);
    const v16f y0 = aie::add(t0, magic), y1 = aie::add(t1, magic);
    const v16f f0 = aie::sub(t0, aie::sub(y0, magic));
    const v16f f1 = aie::sub(t1, aie::sub(y1, magic));
    const v16i n0 = aie::min(aie::max(aie::sub(y0.cast_to<int32_t>(), mbits), nmin), nmax);
    const v16i n1 = aie::min(aie::max(aie::sub(y1.cast_to<int32_t>(), mbits), nmin), nmax);
    const v32f e  = exp2_bf16_poly(to_bf16(aie::concat(f0, f1)));
    e0            = aie::add(e.template extract<16>(0).cast_to<int32_t>(), aie::upshift(n0, 23)).cast_to<float>();
    e1            = aie::add(e.template extract<16>(1).cast_to<int32_t>(), aie::upshift(n1, 23)).cast_to<float>();
}

}  // namespace xdna
