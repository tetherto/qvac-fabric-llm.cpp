#ifdef TYPE_F16
enable f16;
#define TYPE f16
#else
#define TYPE f32
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<TYPE>;

#ifndef INPLACE
@group(0) @binding(1)
var<storage, read_write> dst: array<TYPE>;
#define PARAMS_BINDING 2
#else
#define PARAMS_BINDING 1
#endif

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    ne0: u32,
    ne1: u32,
    ne2: u32,
#ifdef CLAMP
    clamp_min: f32,
    clamp_max: f32,
#endif
#ifdef FILL
    fill_val: f32,
#endif
#ifdef XIELU
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,
#endif

};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

#if defined(SIN) || defined(COS)
const TWO_OVER_PI_WORDS = array<u32, 6>(0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u, 0xdb629599u, 0x3c439041u);
const PI_OVER_FOUR: f32 = 0.7853981633974483;
const PI_OVER_TWO: f32 = 1.5707963267948966;
const F32_SIGN_MASK: u32 = 0x80000000u;
const F32_EXPONENT_MASK: u32 = 0x7f800000u;
const F32_MANTISSA_MASK: u32 = 0x007fffffu;
const F32_IMPLICIT_ONE: u32 = 0x00800000u;
const F32_MANTISSA_BITS: u32 = 23u;
const F32_EXPONENT_BIAS: i32 = 127;
const F32_QUIET_NAN: u32 = 0x7fc00000u;
const HALF_WORD_BITS: u32 = 16u;
const HALF_WORD_MASK: u32 = 0xffffu;
const WORD_BITS: u32 = 32u;
const FIRST_WORD_MAX_SCALE: i32 = 33;
const WORD_SKIP_BIAS: i32 = 2;
const QUADRANT_WINDOW_LOW_BIT: i32 = 34;
const QUADRANT_SHIFT: u32 = 30u;
const QUADRANT_MASK: u32 = 3u;
const FRACTION_MASK: u32 = 0x3fffffffu;
const FRACTION_HALF: u32 = 0x20000000u;
const FRACTION_ONE: i32 = 0x40000000;
const FRACTION_SCALE: f32 = 9.313225746154785e-10;
const COS_QUADRANT_OFFSET: u32 = 1u;

struct ReducedAngle {
    r: f32,
    quadrant: u32,
};

fn mul_u32_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & HALF_WORD_MASK;
    let a_hi = a >> HALF_WORD_BITS;
    let b_lo = b & HALF_WORD_MASK;
    let b_hi = b >> HALF_WORD_BITS;
    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let middle = (lo_lo >> HALF_WORD_BITS) + (hi_lo & HALF_WORD_MASK) + (lo_hi & HALF_WORD_MASK);
    let lo = (middle << HALF_WORD_BITS) | (lo_lo & HALF_WORD_MASK);
    let hi = a_hi * b_hi + (hi_lo >> HALF_WORD_BITS) + (lo_hi >> HALF_WORD_BITS) + (middle >> HALF_WORD_BITS);
    return vec2<u32>(lo, hi);
}

fn carry_of(sum: u32, addend: u32) -> u32 {
    return select(0u, 1u, sum < addend);
}

fn mul_mantissa_by_two_over_pi(mantissa: u32, first_word: u32) -> vec4<u32> {
    let low = mul_u32_wide(mantissa, TWO_OVER_PI_WORDS[first_word + 2u]);
    let mid = mul_u32_wide(mantissa, TWO_OVER_PI_WORDS[first_word + 1u]);
    let high = mul_u32_wide(mantissa, TWO_OVER_PI_WORDS[first_word]);
    let limb1 = low.y + mid.x;
    let carry1 = carry_of(limb1, low.y);
    let partial2 = mid.y + high.x;
    let carry2 = carry_of(partial2, mid.y);
    let limb2 = partial2 + carry1;
    let carry3 = carry_of(limb2, partial2);
    return vec4<u32>(low.x, limb1, limb2, high.y + carry2 + carry3);
}

fn funnel_shift_right(lo: u32, hi: u32, shift: u32) -> u32 {
    return select((lo >> shift) | (hi << (WORD_BITS - shift)), lo, shift == 0u);
}

fn window_high_word(product: vec4<u32>, window_low_bit: u32) -> u32 {
    let upper_limbs = window_low_bit >= WORD_BITS;
    let below = select(product.y, product.z, upper_limbs);
    let above = select(product.z, product.w, upper_limbs);
    return funnel_shift_right(below, above, window_low_bit % WORD_BITS);
}

fn reduce_by_pio2(x: f32) -> ReducedAngle {
    let bits = bitcast<u32>(x);
    if ((bits & F32_EXPONENT_MASK) == F32_EXPONENT_MASK) {
        return ReducedAngle(bitcast<f32>(bits | F32_QUIET_NAN), 0u);
    }
    if (abs(x) < PI_OVER_FOUR) {
        return ReducedAngle(x, 0u);
    }
    let exponent = i32((bits & F32_EXPONENT_MASK) >> F32_MANTISSA_BITS) - F32_EXPONENT_BIAS;
    let mantissa = (bits & F32_MANTISSA_MASK) | F32_IMPLICIT_ONE;
    let scale = exponent - i32(F32_MANTISSA_BITS);
    let first_word = select(0, (scale - WORD_SKIP_BIAS) / i32(WORD_BITS), scale > FIRST_WORD_MAX_SCALE);
    let window_scale = scale - first_word * i32(WORD_BITS);
    let product = mul_mantissa_by_two_over_pi(mantissa, u32(first_word));
    let window = window_high_word(product, u32(QUADRANT_WINDOW_LOW_BIT - window_scale));
    let fraction_bits = window & FRACTION_MASK;
    let round_up = fraction_bits >= FRACTION_HALF;
    let fraction = f32(i32(fraction_bits) - select(0, FRACTION_ONE, round_up)) * FRACTION_SCALE;
    let quadrant = ((window >> QUADRANT_SHIFT) + select(0u, 1u, round_up)) & QUADRANT_MASK;
    let r = fraction * PI_OVER_TWO;
    let negative = (bits & F32_SIGN_MASK) != 0u;
    return ReducedAngle(select(r, -r, negative), select(quadrant, (0u - quadrant) & QUADRANT_MASK, negative));
}

fn sin_in_quadrant(angle: ReducedAngle, quadrant_offset: u32) -> f32 {
    let quadrant = angle.quadrant + quadrant_offset;
    let magnitude = select(sin(angle.r), cos(angle.r), (quadrant & 1u) == 1u);
    return select(magnitude, -magnitude, (quadrant & 2u) == 2u);
}
#endif

fn erf_approx(x: TYPE) -> TYPE {
    let x_f32 = f32(x);
    let s = select(-1.0, 1.0, x_f32 >= 0.0);
    let ax = abs(x_f32);

    let t = 1.0 / (1.0 + 0.3275911 * ax);

    let y = 1.0 -
        (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t) *
        exp(-ax * ax);

    return TYPE(s * y);
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups)       num_wg:  vec3<u32>) {
    let threads_per_group = u32(WG_SIZE);
    let flat_i = gid.x + (num_wg.x * threads_per_group) * gid.y;
    if (flat_i >= params.ne) {
        return;
    }
    var i = flat_i;
    let ne2 = params.ne2;
#ifdef DIAG
    let ne1 = params.ne0;
#else
    let ne1 = params.ne1;
#endif
    let ne0 = params.ne0;

    let i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    let i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    let i1 = i / ne0;
    let i0 = i % ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 + i2 * params.stride_src2 + i3 * params.stride_src3;

#ifdef ABS
    let res = abs(src[params.offset_src + src_idx]);
#endif
#ifdef SGN
    let res = select(TYPE(select(0.0, -1.0, src[params.offset_src + src_idx] < 0.0)), TYPE(1.0), src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef NEG
    let res = -src[params.offset_src + src_idx];
#endif
#ifdef STEP
    let res = TYPE(select(0.0, 1.0, src[params.offset_src + src_idx] > 0.0));
#endif
#ifdef TANH
    let res = tanh(clamp(src[params.offset_src + src_idx], -9.010913, 9.010913));
#endif
#ifdef RELU
    let res = select(0.0, src[params.offset_src + src_idx], src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef ELU
    let res = select(exp(src[params.offset_src + src_idx]) - 1.0, src[params.offset_src + src_idx], src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef HARDSIGMOID
    let res = min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef SIGMOID
    let res = 1.0 / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef SILU
    let res = src[params.offset_src + src_idx] / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef EXP
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32));
#endif
#ifdef LOG
    let res = TYPE(log(f32(src[params.offset_src + src_idx])));
#endif
#ifdef CLAMP
    let res = clamp(src[params.offset_src + src_idx], TYPE(params.clamp_min), TYPE(params.clamp_max));
#endif
#ifdef FILL
    let res = TYPE(params.fill_val);
#endif
#ifdef HARDSWISH
    let res = src[params.offset_src + src_idx] * min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef GELU
    let res = 0.5 * src[params.offset_src + src_idx] * (1.0 + tanh(clamp(0.7978845608028654 * (src[params.offset_src + src_idx] + 0.044715 * src[params.offset_src + src_idx] * src[params.offset_src + src_idx] * src[params.offset_src + src_idx]), -9.010913, 9.010913)));
#endif
#ifdef GELU_QUICK
    let res = src[params.offset_src + src_idx] * (1.0 / (1.0 + exp(clamp(-1.702 * src[params.offset_src + src_idx], -80.0, 80.0))));
#endif
#ifdef GELU_ERF
    let res = 0.5 * src[params.offset_src + src_idx] * (1.0 + erf_approx(src[params.offset_src + src_idx] * 0.7071067811865476));
#endif
#ifdef XIELU
    let val = f32(src[params.offset_src + src_idx]);
    let res =
        TYPE(select(
            ((exp(min(val, params.eps)) - 1.0) - val) * params.alpha_n + params.beta * val,
            params.alpha_p * val * val + params.beta * val,
            val > 0.0));
#endif
#ifdef SOFTPLUS
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(select(log(1.0 + exp(src_f32)), src_f32, src_f32 > 20.0));
#endif
#ifdef EXPM1
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32) - 1.0);
#endif
#ifdef FLOOR
    let res = floor(src[params.offset_src + src_idx]);
#endif
#ifdef CEIL
    let res = ceil(src[params.offset_src + src_idx]);
#endif
#ifdef ROUND
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let result = select(ceil(src_f32 - 0.5), floor(src_f32 + 0.5), src_f32 >= 0.0);
    let res = TYPE(result);
#endif
#ifdef TRUNC
    let res = trunc(src[params.offset_src + src_idx]);
#endif
#ifdef SQR
    let res = src[params.offset_src + src_idx] * src[params.offset_src + src_idx];
#endif
#ifdef SQRT
    let res = TYPE(sqrt(f32(src[params.offset_src + src_idx])));
#endif
#ifdef SIN
    let res_f32 = sin_in_quadrant(reduce_by_pio2(f32(src[params.offset_src + src_idx])), 0u);
    let res = TYPE(res_f32);
#endif
#ifdef COS
    let res_f32 = sin_in_quadrant(reduce_by_pio2(f32(src[params.offset_src + src_idx])), COS_QUADRANT_OFFSET);
    let res = TYPE(res_f32);
#endif
#ifdef DIAG
    let res = select(0.0, src[params.offset_src + i0 + i2 * params.stride_src2 + i3 * params.stride_src3], i0 == i1);
#endif
#ifdef TRI
#ifdef TRI_TYPE_LOWER
    let res = select(0.0, src[params.offset_src + src_idx], i0 < i1);
#elif TRI_TYPE_LOWER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 <= i1);
#elif TRI_TYPE_UPPER
    let res = select(0.0, src[params.offset_src + src_idx], i0 > i1);
#elif TRI_TYPE_UPPER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 >= i1);
#endif
#endif

#ifdef INPLACE
    src[params.offset_src + src_idx] = res;
#else
    dst[params.offset_dst + flat_i] = res;
#endif
}
