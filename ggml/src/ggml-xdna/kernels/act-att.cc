// The attention layer's work on the prologue tile (appended to gemv-q4.cc's
// ACT_PRO build, so it shares that object's state). Besides the activation
// broadcast the tile has two streams of its own: a side input from DDR and an
// output to DDR. A main tile that sets flags bit 5 names how many side objects
// it takes and how many objects it emits before its own output; every other
// tile takes and emits none, so the projections' streams are unchanged.
//
// Words of a main tile with bit 5 set:
//   ACT_TILE/4 - 6   side objects to take before the output
//   ACT_TILE/4 - 7   objects to emit before the output
//   AA_MODE_W        1: prepare a q tile, 2: the combine's activation tile
//
// Prepare (the attention phase's two q objects, group t = word 513):
//   main  f32 gamma_q[256] at 0; words 512-514 (n_valid, t, skip) go to the
//         pool as they are; 515 n_rot, 516 the q scale (f32), 517 eps (f32)
//   side  t = 0: consts (gamma_k f32[256], cos[128], sin[128]), q of heads
//         0-1, q of 2-3, k of both kv heads, v of both; t = 1: q of 4-5, 6-7.
//         Every head's 256 values f32, straight from the projection.
//   emit  t = 0: this position's K row, then its V row, f16 - into the cache
//   out   the q tile attn-dec.cc takes: bf16 [32 blocks][8 dims][4 heads],
//         normed, rotated and scaled, then words 512-514
//
// Combine (the attention output projection's activation tiles, k = word 517):
//   head  the projection's count header (mode 3) takes the gate, two heads
//         an object, while the pool is on the attention: sigmoid(gate)
//   side  tile 0 only: the sixteen cores' m and l (attn-dec.cc: 64 floats
//         a core, 2 objects), then their o (2048 floats a core, 64 objects),
//         word 519 objects in all
//   out   activation tile k of softmax(QK)V * sigmoid(gate), q4g32 codes

// Row (mode 4: the activation tiles of a projection whose input is a norm of
// the residual stream, so nothing of the layer boundary is the host's):
//   main  word 517 the tile's index in its chunk, 516 the code format (0
//         q4g32, 1 q8g16), 515 the row length D, 514 eps (f32), 513 whether a
//         residual is added
//   side  the first tile only: acc (D floats, the previous projection's
//         drain), the residual (D floats) when word 513 says so, gamma (D
//         floats); 512 floats an object
//   emit  the first tile only: h = acc + residual, D floats, 256 an object -
//         the residual stream's next value
//   out   tile k of rms_norm(h) * gamma, quantized as the host would pack it
//         (every later tile and every replayed chunk quantizes from the row)
//   gates word 512 rows (a GDN layer's in-projection: 32). The first tile
//         keeps x as bf16 for them; the last tile of the last chunk (word 510
//         set) takes one object of constants (dt[16], a[16], the scale) and
//         the rows of ssm_alpha then ssm_beta, bf16, one an object, and emits
//         x's per-head tails - exp(softplus(alpha + dt) * a), sigmoid(beta),
//         the scale - in the first 48 words of its one object. By then the
//         pool is on the last chunk: the gates cost the projection nothing.

#define AA_SIDE_WORDS 512   // a side object: 2048 B
#define AA_EMIT_WORDS 256   // an emitted object: 1024 B
#define AA_NS_W   (ACT_TILE / 4 - 6)
#define AA_NE_W   (ACT_TILE / 4 - 7)
#define AA_MODE_W 518
#define AA_NPART_W 519

#define AA_D     256          // head dim
#define AA_H     8            // query heads
#define AA_G     4            // query heads a kv head serves
#define AA_CORES 16
#define AA_ST    2112         // a core's partial state, floats
#define AA_ML    64           // m[2][16] l[2][16] at its head

namespace {

alignas(64) float    aa_gk[AA_D];            // k norm's gamma
alignas(64) float    aa_cos[128], aa_sin[128];
alignas(64) bfloat16 aa_qt[AA_G * AA_D];     // the q tile being built
alignas(64) int32_t  aa_kv[2][AA_EMIT_WORDS];// this position's K and V rows, f16
alignas(64) float    aa_o[AA_H * AA_D];      // combine: o, the partial layout
alignas(64) float    aa_out[AA_H * AA_D];    // combine: the gated output, h*256+d
alignas(64) float    aa_ml[AA_CORES * AA_ML];   // combine: every core's m and l
aie::vector<bfloat16, 16> aa_cb[AA_CORES][2][2];  // combine: a core's coefficients, heads 0-1 and 2-3 of a group
alignas(64) float    aa_il[16];                 // combine: 1 / l per head
alignas(64) const int32_t aa_lane[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
alignas(64) float    aa_y[2 * AA_D];         // a head or two, normed and rotated
alignas(64) bfloat16 aa_yb[AA_D];
int aa_side, aa_emit;
float aa_r;                 // row: the norm's 1 / rms
alignas(64) float aa_ab[32], aa_gc[48];   // gates: alpha|beta, constants
alignas(64) int32_t aa_tails[AA_EMIT_WORDS];  // gates: x's tails, 3 a head

inline aie::vector<float, 16> bcf(float v) { return aie::broadcast<float, 16>(v); }

// e^x in fp32: 2^n from the exponent bits, 2^f from a degree-6 polynomial
// (silu-f32.h's). Clamped where the exponent under- or overflows.
__attribute__((noinline, minsize)) aie::vector<float, 16> exp_f32(const aie::vector<float, 16> x)
{
    auto t = vmul(x, bcf(1.4426950408889634f));
    t = aie::min(aie::max(t, bcf(-126.0f)), bcf(126.0f));
    const auto magic = bcf(12582912.0f);
    const auto tm = aie::add(t, magic);
    const auto n = aie::sub(tm, magic);
    const auto f = aie::sub(t, n);
    const auto ni = aie::sub(tm.cast_to<int32>(), magic.cast_to<int32>());
    // degree 4: 4e-5 at the ends of [-0.5, 0.5], and two fp32 products less
    auto p = aie::add(vmul(f, bcf(9.6181291076284772e-3f)), bcf(5.5504108664821580e-2f));
    p = aie::add(vmul(p, f), bcf(2.4022650695910071e-1f));
    p = aie::add(vmul(p, f), bcf(6.9314718055994531e-1f));
    p = aie::add(vmul(p, f), bcf(1.0f));
    const auto sc = aie::upshift(aie::add(ni, aie::broadcast<int32, 16>(127)), 23).cast_to<float>();
    return vmul(p, sc);
}

// One head: y = rms_norm(x) * gamma, the first n_rot dims rotated in
// neox pairs (i, i + n_rot/2) - the rotation ggml's multi-section rope applies
// to text, with the host's cos/sin for this position - then times `scale`.
__attribute__((noinline, minsize)) void norm_rope(const float *x, const float *gamma, float *y,
                                         int n_rot, float eps, float scale)
{
    aie::vector<float, 16> ss = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int i = 0; i < AA_D; i += 16) {
        const auto v = aie::load_v<16>(x + i);
        ss = aie::add(ss, vmul(v, v));
    }
    const float mean = v_mul(aie::reduce_add(ss), 1.0f / AA_D);
    const float r = v_div(scale, aie::sqrt(v_add(mean, eps)));   // rope is linear: scale first
#pragma clang loop unroll(disable)
    for (int i = 0; i < AA_D; i += 16) {
        const auto v = vmul(aie::load_v<16>(x + i), bcf(r));
        aie::store_v(y + i, vmul(v, aie::load_v<16>(gamma + i)));
    }
    const int half = n_rot / 2;
#pragma clang loop unroll(disable)
    for (int i = 0; i < half; i += 16) {
        const auto a = aie::load_v<16>(y + i);
        const auto b = aie::load_v<16>(y + half + i);
        const auto c = aie::load_v<16>(aa_cos + i);
        const auto s = aie::load_v<16>(aa_sin + i);
        aie::store_v(y + i, aie::sub(vmul(a, c),
                                     vmul(b, s)));
        aie::store_v(y + half + i, aie::add(vmul(a, s),
                                            vmul(b, c)));
    }
}

// 512 f32 to f16 (round to nearest even), as 256 words. Values too small for
// a normal f16 flush to zero, too large saturate to infinity.
__attribute__((noinline, minsize)) void to_f16(const float *x, int32_t *dst)
{
    alignas(64) int32_t h[16];
    uint16_t *d16 = (uint16_t *) dst;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 2 * AA_D; i += 16) {
        const aie::vector<int32, 16> b = aie::load_v<16>(x + i).cast_to<int32>();
        const auto bi = [](int32_t v) { return aie::broadcast<int32, 16>(v); };
        const auto sign = aie::bit_and(aie::downshift(b, 16), bi(0x8000));
        const auto e = aie::sub(aie::bit_and(aie::downshift(b, 23), bi(0xFF)), bi(112));
        const auto m = aie::bit_and(b, bi(0x7FFFFF));
        // round: add half an ulp less one, plus the kept lsb (ties to even)
        const auto lsb = aie::bit_and(aie::downshift(m, 13), bi(1));
        const auto mr = aie::add(aie::add(m, bi(0xFFF)), lsb);
        // exponent and mantissa together, so a carry out of the mantissa
        // moves the exponent
        auto v = aie::add(aie::upshift(e, 10), aie::downshift(mr, 13));
        v = aie::select(v, bi(0), aie::lt(e, bi(1)));
        v = aie::select(v, bi(0x7C00), aie::gt(v, bi(0x7BFF)));
        aie::store_v(h, aie::bit_or(v, sign));
#pragma clang loop vectorize(disable) unroll(disable)
        for (int j = 0; j < 16; j++) {
            d16[i + j] = (uint16_t) h[j];
        }
    }
}

// Lanes 0-7 one value, 8-15 another.
inline aie::vector<float, 16> two8(float lo, float hi)
{
    return aie::select(bcf(lo), bcf(hi),
                       aie::ge(aie::load_v<16>(aa_lane), aie::broadcast<int32, 16>(8)));
}

// Every core's m and l are in: the combine's coefficient of each core and
// head against the heads' overall max, and 1 / l of each head. Lane i of a
// group's vector is head 4g + i % 4, as attn-dec.cc keeps them.
__attribute__((noinline, minsize)) void combine_coef(void)
{
#pragma clang loop unroll(disable)
    for (int k = 0; k < AA_H * AA_D; k += 16) {
        aie::store_v(aa_o + k, aie::zeros<float, 16>());
    }
#pragma clang loop unroll(disable)
    for (int g = 0; g < 2; g++) {
        aie::vector<float, 16> mx = aie::load_v<16>(aa_ml + g * 16);
#pragma clang loop unroll(disable)
        for (int c = 1; c < AA_CORES; c++) {
            mx = aie::max(mx, aie::load_v<16>(aa_ml + c * AA_ML + g * 16));
        }
        aie::vector<float, 16> l = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
        for (int c = 0; c < AA_CORES; c++) {
            // bf16 coefficients, and l from the same rounded values, so the
            // weights the combine applies are the ones it divides by
            aie::accum<accfloat, 16> ab;
            ab.from_vector(exp_f32(aie::sub(aie::load_v<16>(aa_ml + c * AA_ML + g * 16), mx)), 0);
            const aie::vector<bfloat16, 16> a16 = ab.template to_vector<bfloat16>();
            ab.from_vector(a16, 0);
            const aie::vector<float, 16> a = ab.template to_vector<float>(0);
            alignas(64) float t[16];
            aie::store_v(t, a);
            aie::accum<accfloat, 16> pa;
            pa.from_vector(two8(t[0], t[1]), 0);
            aa_cb[c][g][0] = pa.template to_vector<bfloat16>();
            pa.from_vector(two8(t[2], t[3]), 0);
            aa_cb[c][g][1] = pa.template to_vector<bfloat16>();
            l = aie::add(l, vmul(aie::load_v<16>(aa_ml + c * AA_ML + 32 + g * 16), a));
        }
        alignas(64) float lf[16];
        aie::store_v(lf, l);
#pragma clang loop vectorize(disable) unroll(disable)
        for (int hh = 0; hh < AA_G; hh++) {
            aa_il[g * AA_G + hh] = v_pos(lf[hh]) ? v_div(1.0f, lf[hh]) : 0.0f;
        }
    }
}

// A quarter of one core's o - a group's 16 blocks of [4 heads][8 dims] -
// times its coefficients, into the combine's o: a bf16 multiply into fp32
// accumulators, the MAC's own, where an fp32 product is three of them.
__attribute__((noinline)) void combine_o(const float *s, int j)
{
    const int c = j >> 2, g = (j >> 1) & 1;
    float *o = aa_o + g * 1024 + (j & 1) * 512;
    const aie::vector<bfloat16, 16> A = aa_cb[c][g][0], B = aa_cb[c][g][1];
    for (int b = 0; b < 512; b += 16) {
        aie::accum<accfloat, 16> x, acc;
        x.from_vector(aie::load_v<16>(s + b), 0);
        acc.from_vector(aie::load_v<16>(o + b), 0);
        acc = aie::mac(acc, x.template to_vector<bfloat16>(), (b & 16) ? B : A);
        aie::store_v(o + b, acc.template to_vector<float>(0));
    }
}

// Two heads of the gate, while the pool runs the attention: sigmoid(gate)
// into the output, in its order h*256+d.
__attribute__((noinline, minsize)) void combine_gate(const float *s, int j)
{
#pragma clang loop unroll(disable)
    for (int i = 0; i < 2 * AA_D; i += 16) {
        const auto e = exp_f32(aie::neg(aie::load_v<16>(s + i)));
        aie::store_v(aa_out + j * 2 * AA_D + i, recip_f32(aie::add(e, bcf(1.0f))));
    }
}

// The output: o / l * sigmoid(gate). o is in the partial layout [2 groups]
// [32 blocks][4 heads][8 dims], so sixteen dims of a head are two blocks'
// eights: the two aligned halves that hold them, one rotated by eight lanes,
// and a select.
__attribute__((noinline, minsize)) void combine_out(void)
{
    const auto hi = aie::ge(aie::load_v<16>(aa_lane), aie::broadcast<int32, 16>(8));
#pragma clang loop unroll(disable)
    for (int h = 0; h < AA_H; h++) {
        const int g = h / AA_G, hh = h % AA_G;
        const float *o = aa_o + g * 1024 + (hh >> 1) * 16;
        const auto il = bcf(aa_il[h]);
#pragma clang loop unroll(disable)
        for (int d = 0; d < AA_D; d += 16) {
            const auto a = aie::load_v<16>(o + (d >> 3) * 32);
            const auto b = aie::load_v<16>(o + (d >> 3) * 32 + 32);
            const auto v = (hh & 1) ? aie::select(aie::shuffle_down_rotate(a, 8), b, hi)
                                    : aie::select(a, aie::shuffle_down_rotate(b, 8), hi);
            float *y = aa_out + h * AA_D + d;
            aie::store_v(y, vmul(vmul(v, aie::load_v<16>(y)), il));
        }
    }
}


inline float as_f(int32_t w) { float f; __builtin_memcpy(&f, &w, 4); return f; }

// log2(y) for y >= 1 in fp32: the exponent from the bits, the mantissa
// folded into [sqrt(1/2), sqrt(2)) and ln m = 2 atanh((m-1)/(m+1)) to t^7.
__attribute__((noinline, minsize)) aie::vector<float, 16> log2_f32(const aie::vector<float, 16> y)
{
    const auto bi = [](int32_t v) { return aie::broadcast<int32, 16>(v); };
    const aie::vector<int32, 16> b = y.cast_to<int32>();
    aie::vector<int32, 16> e = aie::sub(aie::bit_and(aie::downshift(b, 23), bi(0xFF)), bi(127));
    aie::vector<float, 16> m = aie::bit_or(aie::bit_and(b, bi(0x7FFFFF)), bi(0x3F800000)).cast_to<float>();
    const auto big = aie::gt(m, bcf(1.41421356f));
    m = aie::select(m, vmul(m, bcf(0.5f)), big);
    e = aie::select(e, aie::add(e, bi(1)), big);
    const auto t = vmul(aie::sub(m, bcf(1.0f)), recip_f32(aie::add(m, bcf(1.0f))));
    const auto t2 = vmul(t, t);
    auto p = aie::add(vmul(t2, bcf(1.0f / 7)), bcf(1.0f / 5));
    p = aie::add(vmul(p, t2), bcf(1.0f / 3));
    p = aie::add(vmul(p, t2), bcf(1.0f));
    const auto ln = vmul(vmul(p, t), bcf(2.0f));
    return aie::add(vmul(ln, bcf(1.4426950408889634f)), aie::to_float(e, 0));
}

// The GDN gates from alpha, beta and the constants: x's tails.
__attribute__((noinline, minsize)) void gdn_tails(void)
{
    const auto z = aie::add(aie::load_v<16>(aa_ab), aie::load_v<16>(aa_gc));      // alpha + dt
    // softplus(z) = ln(1 + e^z), z itself past 20 (ggml's)
    const auto y = aie::add(exp_f32(z), bcf(1.0f));
    auto sp = vmul(log2_f32(y), bcf(0.69314718055994531f));
    sp = aie::select(sp, z, aie::gt(z, bcf(20.0f)));
    const auto eg = exp_f32(vmul(sp, aie::load_v<16>(aa_gc + 16)));
    const auto b = recip_f32(aie::add(exp_f32(aie::neg(aie::load_v<16>(aa_ab + 16))), bcf(1.0f)));
    alignas(64) int32_t te[16], tb[16];
    aie::store_v(te, eg.cast_to<int32>());
    aie::store_v(tb, b.cast_to<int32>());
    const int32_t sc = ((const int32_t *) aa_gc)[32];
#pragma clang loop vectorize(disable) unroll(disable)
    for (int h = 0; h < 16; h++) {
        aa_tails[3 * h]     = te[h];
        aa_tails[3 * h + 1] = tb[h];
        aa_tails[3 * h + 2] = sc;
    }
}

// The row's side objects: acc into h, the residual onto it, then gamma - at
// its first object the norm's reduction, then x = h / rms * gamma. h is the
// first D floats of aa_o, x the next D.
__attribute__((noinline, minsize)) void row_side(const float *s, int i, const int32_t *in)
{
    const int D = in[515];
    const int n = D / AA_SIDE_WORDS;
    const int res = in[513];
    float *h = aa_o, *x = aa_o + D;
    const int nrow = in[510] ? 0 : (res ? 3 : 2) * n;   // the gates' tile takes only them
    if (i >= nrow) {
        // the gates: constants, then one weight row an object
        const int r = i - nrow - 1;
        if (r < 0) {
#pragma clang loop unroll(disable)
            for (int k = 0; k < 48; k += 16) {
                aie::store_v(aa_gc + k, aie::load_v<16>(s + k));
            }
            return;
        }
        const bfloat16 *xb = (const bfloat16 *) (aa_out + 1024);
        const bfloat16 *w = (const bfloat16 *) s;
        aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
        for (int k = 0; k < D; k += 16) {
            acc = aie::mac(acc, aie::load_v<16>(xb + k), aie::load_v<16>(w + k));
        }
        aa_ab[r] = aie::reduce_add(acc.to_vector<float>(0));
        if (r == in[512] - 1) {
            gdn_tails();
        }
        return;
    }
    const int role = i / n, j = (i % n) * AA_SIDE_WORDS;
    if (role == 0) {
#pragma clang loop unroll(disable)
        for (int k = 0; k < AA_SIDE_WORDS; k += 16) {
            aie::store_v(h + j + k, aie::load_v<16>(s + k));
        }
        return;
    }
    if (role == 1 && res) {
#pragma clang loop unroll(disable)
        for (int k = 0; k < AA_SIDE_WORDS; k += 16) {
            aie::store_v(h + j + k, aie::add(aie::load_v<16>(h + j + k), aie::load_v<16>(s + k)));
        }
        return;
    }
    if (j == 0) {
        aie::vector<float, 16> ss = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
        for (int k = 0; k < D; k += 16) {
            const auto v = aie::load_v<16>(h + k);
            ss = aie::add(ss, vmul(v, v));
        }
        aa_r = v_div(1.0f, aie::sqrt(v_add(v_div(aie::reduce_add(ss), v_i2f(D)), as_f(in[514]))));
    }
#pragma clang loop unroll(disable)
    for (int k = 0; k < AA_SIDE_WORDS; k += 16) {
        const auto v = vmul(aie::load_v<16>(h + j + k), bcf(aa_r));
        aie::store_v(x + j + k, vmul(v, aie::load_v<16>(s + k)));
    }
    if (in[512] > 0 && i == nrow - 1) {
        // the gates' dot products read x as bf16
        bfloat16 *xb = (bfloat16 *) (aa_out + 1024);
        aie::accum<accfloat, 16> t;
#pragma clang loop unroll(disable)
        for (int k = 0; k < D; k += 16) {
            t.from_vector(aie::load_v<16>(x + k), 0);
            aie::store_v(xb + k, t.template to_vector<bfloat16>());
        }
    }
}

} // namespace

// n values into an activation tile of groups of `grp` (32: q4g32, 16:
// q8g16): int8 codes, then a code sum and a scale per group - gemv_pack_act.
__attribute__((noinline, minsize)) void quant_tile(const float *x, int n, int grp, int8_t *code)
{
    float *gsum = (float *) (code + n);
    float *gd   = gsum + n / grp;
#pragma clang loop unroll(disable)
    for (int g = 0; g < n / grp; g++) {
        float amax = 0.0f;
#pragma clang loop unroll(disable)
        for (int p = 0; p < grp; p += 16) {
            const float m = aie::reduce_max(aie::abs(aie::load_v<16>(x + g * grp + p)));
            amax = v_max(m, amax);
        }
        const float inv = v_pos(amax) ? v_div(127.0f, amax) : 1.0f;
        int sum = 0;
#pragma clang loop unroll(disable)
        for (int p = 0; p < grp; p += 16) {
            const auto q = vmul(aie::load_v<16>(x + g * grp + p), bcf(inv));
            const aie::vector<int32_t, 16> qi = aie::to_fixed<int32_t>(q, 0);
            aie::store_v(code + g * grp + p, aie::pack(aie::pack(qi)));
            sum += aie::reduce_add(qi);
        }
        gsum[g] = v_i2f(sum);
        gd[g]   = v_pos(amax) ? v_mul(amax, 1.0f / 127.0f) : 1.0f;
    }
}


extern "C" {

// The two counts the worker loops over, zero unless the tile asks.
__attribute__((minsize)) void ggml_xdna_act_cnt(const int32_t *in, int32_t *cnt)
{
    const int flags = in[ACT_TILE / 4 - 2];
    const int on = (flags >> 5) & 1;
    cnt[0] = on ? in[AA_NS_W] : 0;
    cnt[1] = on ? in[AA_NE_W] : 0;
    aa_side = 0;
    aa_emit = 0;
}

__attribute__((minsize)) void ggml_xdna_act_side(const int32_t *side, const int32_t *in)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *s = (const float *) side;
    const int i = aa_side++;
    if (in[AA_MODE_W] == 4) {
        row_side(s, i, in);
        return;
    }
    if (in[AA_MODE_W] == 3) {
        combine_gate(s, i);
        return;
    }
    if (in[AA_MODE_W] == 2) {
        // [m and l of every core: 2 objects | o, a quarter core an object: 64]
        const int npart = in[AA_NPART_W];
        if (i < 2) {
#pragma clang loop unroll(disable)
            for (int k = 0; k < AA_SIDE_WORDS; k += 16) {
                aie::store_v(aa_ml + i * AA_SIDE_WORDS + k, aie::load_v<16>(s + k));
            }
            if (i == 1) {
                combine_coef();
            }
        } else {
            combine_o(s, i - 2);
            if (i == npart - 1) {
                combine_out();
            }
        }
        return;
    }
    const int t = in[513];
    const int n_rot = in[515];
    const float scale = as_f(in[516]);
    const float eps = as_f(in[517]);
    const float *gq = (const float *) in;
    // the role of the i-th side object of this tile
    int role, j = 0;
    if (t == 0) {
        role = i == 0 ? 0 : i <= 2 ? 1 : i == 3 ? 2 : 3;
        j = i - 1;
    } else {
        role = 1;
        j = i;
    }
    if (role == 0) {
#pragma clang loop unroll(disable)
        for (int k = 0; k < AA_D; k += 16) {
            aie::store_v(aa_gk + k, aie::load_v<16>(s + k));
        }
#pragma clang loop unroll(disable)
        for (int k = 0; k < 128; k += 16) {
            aie::store_v(aa_cos + k, aie::load_v<16>(s + AA_D + k));
            aie::store_v(aa_sin + k, aie::load_v<16>(s + AA_D + 128 + k));
        }
    } else if (role == 1) {
        // two heads of this group, into the tile [block][8 dims][4 heads]
        float *y = aa_y;
        bfloat16 *yb = aa_yb;
#pragma clang loop unroll(disable)
        for (int k = 0; k < 2; k++) {
            norm_rope(s + k * AA_D, gq, y, n_rot, eps, scale);
            aie::accum<accfloat, 16> acc;
#pragma clang loop unroll(disable)
            for (int d = 0; d < AA_D; d += 16) {
                acc.from_vector(aie::load_v<16>(y + d), 0);
                aie::store_v(yb + d, acc.template to_vector<bfloat16>());
            }
            const int hh = 2 * j + k;
#pragma clang loop vectorize(disable) unroll(disable)
            for (int d = 0; d < AA_D; d++) {
                aa_qt[(d >> 3) * 32 + (d & 7) * 4 + hh] = yb[d];
            }
        }
    } else if (role == 2) {
        float *y = aa_y;
        norm_rope(s, aa_gk, y, n_rot, eps, 1.0f);
        norm_rope(s + AA_D, aa_gk, y + AA_D, n_rot, eps, 1.0f);
        to_f16(y, aa_kv[0]);
    } else {
        to_f16(s, aa_kv[1]);
    }
}

__attribute__((minsize)) void ggml_xdna_act_emit(int32_t *out, const int32_t *in)
{
    const int e = aa_emit++;
    // row: h, then (gates) x's tails in the first 48 words of one more
    const int32_t *src = in[AA_MODE_W] != 4 ? aa_kv[e & 1]
                       : !in[510] && e * AA_EMIT_WORDS < in[515] ? (const int32_t *) aa_o + e * AA_EMIT_WORDS
                                                                 : aa_tails;
#pragma clang loop unroll(disable)
    for (int i = 0; i < AA_EMIT_WORDS; i += 16) {
        aie::store_v(out + i, aie::load_v<16>(src + i));
    }
}

// The main tile's own output (the prologue's entry point sends it here).
__attribute__((minsize)) void act_att_tile(const int32_t *in, int32_t *out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
    for (int i = 0; i < ACT_TILE / 4; i += 16) {
        aie::store_v(out + i, aie::zeros<int32, 16>());
    }
    if (in[AA_MODE_W] == 4) {
        // The row is quantized once, at the tile that took it, into a
        // compact copy of every tile's codes and group parameters (aa_out,
        // which only the combine uses); every tile of every chunk is a copy.
        const int q8 = in[516];
        const int kt = q8 ? K_TILE_Q8 : K_TILE_Q4;
        const int grp = q8 ? Q8_GROUP : Q4_GROUP;
        // a tile's payload in words, rounded to 64 bytes: the copy below is
        // 512-bit loads, and one at 32 mod 64 reads the wrong words
        const int tw = ((kt + 8 * (kt / grp)) / 4 + 15) & ~15;
        int32_t *cache = (int32_t *) aa_out;
        if (in[AA_NS_W] > 0 && !in[510]) {
#pragma clang loop unroll(disable)
            for (int t = 0; t < in[515] / kt; t++) {
                quant_tile(aa_o + in[515] + t * kt, kt, grp, (int8_t *) (cache + t * tw));
            }
        }
        const int32_t *src = cache + in[517] * tw;
#pragma clang loop unroll(disable)
        for (int i = 0; i < tw; i += 16) {
            aie::store_v(out + i, aie::load_v<16>(src + i));
        }
        out[ACT_TILE / 4 - 1] = in[ACT_TILE / 4 - 1];
        out[ACT_TILE / 4 - 2] = in[ACT_TILE / 4 - 2] & ~(1 << 5);
        return;
    }
    if (in[AA_MODE_W] == 3) {
        // the projection's count header, which took the gate
#pragma clang loop unroll(disable)
        for (int i = 0; i < ACT_TILE / 4; i += 16) {
            aie::store_v(out + i, aie::load_v<16>(in + i));
        }
        out[ACT_TILE / 4 - 2] = in[ACT_TILE / 4 - 2] & ~(1 << 5);
        return;
    }
    if (in[AA_MODE_W] == 2) {
        const int k = in[517];
        quant_tile(aa_out + k * K_TILE_Q4, K_TILE_Q4, Q4_GROUP, (int8_t *) out);
        out[ACT_TILE / 4 - 1] = in[ACT_TILE / 4 - 1];
        out[ACT_TILE / 4 - 2] = in[ACT_TILE / 4 - 2] & ~(1 << 5);
        return;
    }
    const int32_t *q = (const int32_t *) aa_qt;
#pragma clang loop unroll(disable)
    for (int i = 0; i < AA_G * AA_D / 2; i += 16) {
        aie::store_v(out + i, aie::load_v<16>(q + i));
    }
    out[512] = in[512];
    out[513] = in[513];
    out[514] = in[514];
}

} // extern "C"
