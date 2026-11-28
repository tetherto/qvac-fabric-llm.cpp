// Program-memory spike for the merged prefill design (probes/pfx_s1.py): the
// row ops a core would run between GEMMs, written as the real ones would be.
// It is built for its size; nothing calls it with data.
#include "xdna-math.h"

#include <stdint.h>

#include <aie_api/aie.hpp>

using namespace xdna;

namespace {

constexpr int   VL          = 16;
constexpr int   RSQRT_MAGIC = 0x5F3759DF;
constexpr float GELU_K0     = -2.0f * 0.7978845608f * LOG2E;  // -2 sqrt(2 / pi) log2(e)
constexpr float GELU_K1     = 0.044715f;

enum rowop_mode : int32_t {
    RO_HEAD_NORM_ROPE = 1,  // q/k head norm, rope, bf16 attention layout
    RO_POST_ADD_PRE   = 2,  // post-norm, residual add, scale, next pre-norm into bf16 A
    RO_GELU_MUL       = 4,  // gelu(gate) times the per-layer input, bf16
};

inline float as_float(int32_t w) {
    float f;
    __builtin_memcpy(&f, &w, 4);
    return f;
}

// 1 / sqrt(v) in every lane: the integer seed on the scalar unit, Newton in the vector lanes.
__attribute__((noinline, minsize)) v16f rsqrt_v(float v) {
    int32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    const v16f x    = bc<VL>(v);
    const v16f half = vec_mul(x, bc<VL>(0.5f));
    v16f       r    = bci<VL>(RSQRT_MAGIC - (bits >> 1)).cast_to<float>();
    for (int it = 0; it < 2; ++it) {
        r = vec_mul(r, aie::sub(bc<VL>(1.5f), vec_mul(half, vec_mul(r, r))));
    }
    return r;
}

__attribute__((noinline, minsize)) float sum_squares(const float * x, int n) {
    v16f ss = aie::zeros<float, VL>();
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += VL) {
        const v16f v = aie::load_v<VL>(x + i);
        ss           = aie::add(ss, vec_mul(v, v));
    }
    return aie::reduce_add(ss);
}

// The rms scale of a row times `scale`, in every lane.
__attribute__((noinline, minsize)) v16f rms_lanes(const float * x, int n, float eps, float scale) {
    const float mean = sc_mul(sum_squares(x, n), sc_div(1.0f, sc_i2f(n)));
    return vec_mul(rsqrt_v(sc_add(mean, eps)), bc<VL>(scale));
}

// y = x * r * gamma (gamma null: no weight), f32.
__attribute__((noinline, minsize)) void scale_row(const float * x, const float * gamma, v16f r, float * y, int n) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += VL) {
        v16f v = vec_mul(aie::load_v<VL>(x + i), r);
        if (gamma) {
            v = vec_mul(v, aie::load_v<VL>(gamma + i));
        }
        aie::store_v(y + i, v);
    }
}

// The first n_rot dims of y rotated in neox pairs (i, i + n_rot / 2).
__attribute__((noinline, minsize)) void rope_row(float * y, const float * cs, const float * sn, int n_rot) {
    const int half = n_rot / 2;
#pragma clang loop unroll(disable)
    for (int i = 0; i < half; i += VL) {
        const v16f a = aie::load_v<VL>(y + i);
        const v16f b = aie::load_v<VL>(y + half + i);
        const v16f c = aie::load_v<VL>(cs + i);
        const v16f s = aie::load_v<VL>(sn + i);
        aie::store_v(y + i, aie::sub(vec_mul(a, c), vec_mul(b, s)));
        aie::store_v(y + half + i, aie::add(vec_mul(a, s), vec_mul(b, c)));
    }
}

// y in (8, 8) sub-tiles of bf16 for the attention's Q/K layout: row `row` of a 32-row tile.
__attribute__((noinline, minsize)) void to_tile_bf16(const float * y, bfloat16 * out, int row, int n) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += 8) {
        const aie::vector<float, 8> v = aie::load_v<8>(y + i);
        aie::store_v(out + ((i / 8) * 32 + row) * 8, to_bf16(aie::concat(v, v)).extract<8>(0));
    }
}

__attribute__((noinline, minsize)) void to_row_bf16(const float * y, bfloat16 * out, int n) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += VL) {
        aie::store_v(out + i, to_bf16(aie::load_v<VL>(y + i)));
    }
}

// R1: one head of one row.
void head_norm_rope(const float * x, const float * gamma, const float * cs, const float * sn, float * y,
                    bfloat16 * out, const int32_t * cnt) {
    const int   d     = cnt[11];
    const int   n_rot = cnt[12];
    const float scale = as_float(cnt[13]);
    scale_row(x, gamma, rms_lanes(x, d, RMS_EPS, scale), y, d);
    if (n_rot > 0) {
        rope_row(y, cs, sn, n_rot);
    }
    to_tile_bf16(y, out, cnt[14], d);
}

// R2: resid = (resid + post_norm(proj)) * out_scale; out = bf16(pre_norm(resid)).
void post_add_pre(const float * proj, float * resid, const float * g_post, const float * g_pre, float * y,
                  bfloat16 * out, const int32_t * cnt) {
    const int   n         = cnt[11];
    const float out_scale = as_float(cnt[13]);
    scale_row(proj, g_post, rms_lanes(proj, n, RMS_EPS, 1.0f), y, n);
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += VL) {
        const v16f s = aie::add(aie::load_v<VL>(resid + i), aie::load_v<VL>(y + i));
        aie::store_v(resid + i, vec_mul(s, bc<VL>(out_scale)));
    }
    scale_row(resid, g_pre, rms_lanes(resid, n, RMS_EPS, 1.0f), y, n);
    to_row_bf16(y, out, n);
}

// R4: out = bf16(gelu(gate) * x), gelu in its tanh form as a sigmoid of 2^t.
void gelu_mul(const float * gate, const float * x, bfloat16 * out, const int32_t * cnt) {
    const int n = cnt[11];
#pragma clang loop unroll(disable)
    for (int i = 0; i < n; i += VL) {
        const v16f g  = aie::load_v<VL>(gate + i);
        const v16f g2 = vec_mul(g, g);
        const v16f t  = vec_mul(vec_mul(g, bc<VL>(GELU_K0)), aie::add(bc<VL>(1.0f), vec_mul(g2, bc<VL>(GELU_K1))));
        const v16f sg = recip_f32(aie::add(bc<VL>(1.0f), exp2_f32(t)));
        aie::store_v(out + i, to_bf16(vec_mul(vec_mul(g, sg), aie::load_v<VL>(x + i))));
    }
}

}  // namespace

extern "C" {

// One block of rows: x the op's first input, y its second (and the tables), out the bf16 result. Header word 10 is the
// mode; words 11-14 its sizes, scale and tile row.
void ggml_xdna_rowop(const float * x, float * y, bfloat16 * out, const int32_t * cnt) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    float * scratch = y + cnt[15];
    switch (cnt[10]) {
        case RO_HEAD_NORM_ROPE:
            head_norm_rope(x, y, y + cnt[11], y + 2 * cnt[11], scratch, out, cnt);
            break;
        case RO_POST_ADD_PRE:
            post_add_pre(x, y, y + cnt[11], y + 2 * cnt[11], scratch, out, cnt);
            break;
        case RO_GELU_MUL:
            gelu_mul(x, y, out, cnt);
            break;
        default:
            break;
    }
}

}  // extern "C"
