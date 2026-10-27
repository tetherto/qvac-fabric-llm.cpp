#pragma once
// Vector conversions shared by the AIE kernels: bf16 <-> f32, the bf16 hi/lo
// pair the cores use where one bf16 is not accurate enough, and the f16
// formats the KV cache and the attention tiles are in.
#include <stdint.h>

#include <aie_api/aie.hpp>

namespace xdna {

// The cores' default rounding truncates toward -inf, which biases about half
// the weights by one bf16 ulp against ggml's round-half-to-even.
inline void round_half_even() {
    aie::set_rounding(aie::rounding_mode::conv_even);
}

// f32 -> bf16 through the f32 accumulator, so the kernel's rounding mode
// applies (the fused designs run conv_even, the cores' default truncates).
template <unsigned N>
__attribute__((always_inline)) inline aie::vector<bfloat16, N> to_bf16(const aie::vector<float, N> & x) {
    return aie::accum<accfloat, N>(x).template to_vector<bfloat16>(0);
}

// bf16 -> f32, exact: the widening lands in the f32 accumulator.
template <unsigned N>
__attribute__((always_inline)) inline aie::vector<float, N> to_f32(const aie::vector<bfloat16, N> & x) {
    return aie::accum<accfloat, N>(x).template to_vector<float>(0);
}

// x as a bf16 pair whose sum reconstructs it to ~16 mantissa bits.
template <unsigned N>
__attribute__((always_inline)) inline void split_hl(const aie::vector<float, N> & x,
                                                    aie::vector<bfloat16, N> &    hi,
                                                    aie::vector<bfloat16, N> &    lo) {
    hi = to_bf16(x);
    lo = to_bf16(aie::sub(x, to_f32(hi)));
}

// Array forms, for the kernels that walk a buffer in place.
inline void to_bf16(const float * x, bfloat16 * y, int n) {
    for (int i = 0; i < n; i += 32) {
        aie::store_v(y + i, to_bf16(aie::load_v<32>(x + i)));
    }
}

inline void split_hl(const float * x, bfloat16 * hi, bfloat16 * lo, int n) {
    for (int i = 0; i < n; i += 32) {
        const aie::vector<float, 32> v = aie::load_v<32>(x + i);
        aie::vector<bfloat16, 32>    h, l;
        split_hl(v, h, l);
        aie::store_v(hi + i, h);
        aie::store_v(lo + i, l);
    }
}

// (hi + lo) * k with k an integer under 256: exact in the f32 accumulator.
template <unsigned N>
inline aie::vector<float, N> pair_times(const bfloat16 * hi, const bfloat16 * lo, const bfloat16 * k) {
    aie::accum<accfloat, N> a = aie::mul(aie::load_v<N>(hi), aie::load_v<N>(k));
    a                         = aie::mac(a, aie::load_v<N>(lo), aie::load_v<N>(k));
    return a.template to_vector<float>(0);
}

// f16 bit patterns -> bf16: rebias the exponent (15 -> 127), drop three
// mantissa bits, flush subnormals and zero to zero. AIE2P has no f16
// arithmetic; the cache is f16 because that is llama's default.
inline aie::vector<bfloat16, 32> f16_bits_to_bf16(const aie::vector<int16, 32> & h) {
    const auto            sign = aie::bit_and(h, aie::broadcast<int16, 32>((int16) 0x8000));
    const auto            mag  = aie::bit_and(h, aie::broadcast<int16, 32>((int16) 0x7FFF));
    // the three low mantissa bits rounded off to nearest even through the
    // accumulator: aie::downshift wraps every call in crrnd saves and restores
    aie::accum<acc32, 32> ma;
    ma.from_vector(mag, 0);
    auto b = aie::add(ma.to_vector<int16>(3), aie::broadcast<int16, 32>((int16) 0x3800));
    b      = aie::select(b, aie::zeros<int16, 32>(), aie::lt(mag, aie::broadcast<int16, 32>((int16) 0x0400)));
    return aie::bit_or(b, sign).cast_to<bfloat16>();
}

// 512 f32 to 256 f16 words (nearest even), written as raw bits. Values too
// small for a normal f16 flush to zero, too large saturate to infinity.
__attribute__((noinline, minsize)) inline void f32_to_f16(const float * x, int32_t * dst) {
    alignas(64) int32_t h[16];
    uint16_t *          d16 = (uint16_t *) dst;
    const auto          bi  = [](int32_t v) {
        return aie::broadcast<int32, 16>(v);
    };
    for (int i = 0; i < 512; i += 16) {
        const aie::vector<int32, 16> b    = aie::load_v<16>(x + i).cast_to<int32>();
        const auto                   sign = aie::bit_and(aie::downshift(b, 16), bi(0x8000));
        const auto                   e    = aie::sub(aie::bit_and(aie::downshift(b, 23), bi(0xFF)), bi(112));
        const auto                   m    = aie::bit_and(b, bi(0x7FFFFF));
        // round: add half an ulp less one, plus the kept lsb (ties to even)
        const auto                   lsb  = aie::bit_and(aie::downshift(m, 13), bi(1));
        const auto                   mr   = aie::add(aie::add(m, bi(0xFFF)), lsb);
        // exponent and mantissa together, so a carry out of the mantissa
        // moves the exponent
        auto                         v    = aie::add(aie::upshift(e, 10), aie::downshift(mr, 13));
        v                                 = aie::select(v, bi(0), aie::lt(e, bi(1)));
        v                                 = aie::select(v, bi(0x7C00), aie::gt(v, bi(0x7BFF)));
        aie::store_v(h, aie::bit_or(v, sign));
#pragma clang loop vectorize(disable) unroll(disable)
        for (int j = 0; j < 16; j++) {
            d16[i + j] = (uint16_t) h[j];
        }
    }
}

// Widen a T-lane bf16 pattern to 32 lanes by repeating it: a sub-tile's values
// share one group along K, so a scale depends only on the column and the
// column index repeats every MAC_T lanes.
template <unsigned T> inline aie::vector<bfloat16, 32> repeat_to_32(const aie::vector<bfloat16, T> & v) {
    static_assert(T <= 32 && (32 % T) == 0, "T must divide the 32-lane vector");
    if constexpr (T == 32) {
        return v;
    } else {
        return repeat_to_32<2 * T>(aie::concat(v, v));
    }
}

}  // namespace xdna
