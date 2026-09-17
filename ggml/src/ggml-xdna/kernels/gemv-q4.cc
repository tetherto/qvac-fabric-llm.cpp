// SPDX-License-Identifier: MIT
//
// Decode GEMV over the packed NPU weight formats (xdna-quant.h): int8
// activations against 4-bit affine or 8-bit symmetric weights, with the group
// parameters applied to the accumulator instead of materialising bf16 weights.
//
//   acc[n] = sum_g ( d[g][n] * sum_{k in g} a[k]*q[k][n] + m[g][n] * sum_{k in g} a[k] )
//
// The inner sum over a group is an int8 x int4 integer accumulation; the
// f32 rescale happens once per 32 values of K, so its cost is amortised 32x
// and the weights never leave their packed form. That is the difference from
// the expansion kernel (dequant_b.cc): expanding a tile costs about nine times
// the mmul that consumes it, while this applies the same scales for a
// thirty-second of the work.
//
// One activation row is the whole point: at decode M is 1, so the arithmetic
// is far below the DDR rate either way and what matters is that the weight
// stream stays at 0.75 B/value.
//
// Layout contract with the host packer and gemv_q4.py. A weight tile is
// (K_TILE x N_CORE), stored lane-group major so every load is linear:
//
//   for j in N_CORE/32 lane groups:
//     for g in K_TILE/GROUP groups:
//       codes   GROUP rows of 32 columns; q4g32 packs two per byte, low
//               nibble first, q8g16 one int8 per value
//       params  d8[32], m8[32] bf16: this group's integer scale and min for
//               each of the 32 columns
//   then, for j in N_CORE/32 lane groups, for s in NSUP super-blocks:
//       dS_hi[32], dS_lo[32], mS_hi[32], mS_lo[32] bf16 - the pair every d8
//       and m8 under it scales (see xdna-quant.h). A group's parameters are
//       dS*d8 and mS*m8, which reproduces the ggml super-block exactly.
//
// The activation tile is K_TILE int8 codes, then one f32 sum and one f32
// scale per group, then the code width. It is passed as int32 elements so the
// design can read the dispatch counts out of the first object of the same
// stream.
//
// The scale is per group, not one factor for the whole row. That is what lets
// an activation be produced on the device: a row scale is a reduction across
// every core, while a group scale is local to the values a core already holds,
// so a chained operator can quantize its own slice without talking to the
// others. It also removes the row scale's own weakness, that it has to cover
// the row's outliers.
//
// The codes are int8, and that is an arithmetic decision rather than a size
// one. The multiply is what the q4g32 dispatch is limited by - it moves half
// the bytes per value of q8g16, so it has twice the work per byte - and an
// int8 x int8 multiply is the machine's widest integer mode and needs no
// widening of either operand. With int16 codes a 4-bit weight had to be
// unpacked twice on its way to the multiply; now it is unpacked once and the
// activation is already the right width. A group of 32 with its own scale
// leaves about 0.4% on the activation, against a weight that is 4 bits.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

// Columns one multiply covers. AIE2P's int8 datapath is 64 lanes wide, so at
// 32 the kernel uses half of it; a design whose cores are at least 64 columns
// wide builds at 64 and does twice the work per multiply.
#ifndef GEMV_VEC
#define GEMV_VEC 32
#endif
constexpr int VEC = GEMV_VEC;
// Independent MAC chains. Four at 32 lanes fills the multiply's latency; at 64
// an accumulator is four registers, so two chains cost the same eight.
constexpr int CHAINS = VEC >= 64 ? 2 : 4;

// Two rows of 32 packed nibbles at once. The codes are unsigned 0..15, so the
// widening is exact either way; taking two rows per load and per unpack is
// what makes the loop's op count fall - the extracts are subregisters and
// cost nothing.
inline aie::vector<uint8, 2 * VEC> q4_rows(const uint8_t *p)
{
    return aie::unpack(aie::load_v<2 * VEC>((const uint4 *)p));
}

// Two rows of 32 int8 codes, which are already the multiply's operand width.
inline aie::vector<int8, 2 * VEC> q8_rows(const uint8_t *p)
{
    return aie::load_v<2 * VEC>((const int8 *)p);
}

// bf16 pair -> f32 vector. The parameters are stored as two bf16 values that
// sum to the parameter; see xdna-quant.h for why one bf16 is not enough.
inline aie::vector<float, VEC> pair_to_f32(const bfloat16 *hi, const bfloat16 *lo)
{
    // Widening a bf16 vector through a multiply by one: the product of two
    // bf16 values lands in the f32 accumulator exactly, so this is a free and
    // well-defined bf16 -> f32 conversion.
    const aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>((bfloat16)1.0f);
    aie::accum<accfloat, VEC> a = aie::mul(aie::load_v<VEC>(hi), one);
    a = aie::mac(a, aie::load_v<VEC>(lo), one);
    return a.to_vector<float>(0);
}

// silu(x) = x / (1 + exp(-x)) on 16 f32 lanes.
//
// The AIE lookup tables are bf16 only, and a bf16 sigmoid carries ~4e-3
// relative error - two orders worse than what the projections around it
// achieve, so it would undo them. This is range reduction plus a degree-5
// polynomial in f32 instead: exp(-x) = 2^-n * exp(-f) with |f| <= ln2/2, and a
// Newton reciprocal. The volume is 16 lanes per core per chunk, so the cost of
// working in f32 here does not matter.
inline aie::vector<float, 16> silu_f32(const aie::vector<float, 16> &x)
{
#ifndef SILU_STEP
#define SILU_STEP 0
#endif
#if defined(SILU_IDENTITY)
    // Diagnostic: the epilogue runs but the nonlinearity is the identity, so
    // a failure can only be the plumbing around it.
    return x;
#else
    const aie::vector<float, 16> one = aie::broadcast<float, 16>(1.0f);

    // sigma(x) = 1/(1+exp(-x)), computed on x itself rather than on |x|. The
    // series below is accurate across the whole clamped range, so there is no
    // reduction whose sign has to be undone afterwards - no magnitude taken
    // off the bit pattern, no sign bit held live across the exponential, no
    // final 1-2r correction. That correction was the one step the device got
    // wrong once a core owned two lane groups instead of one: probing the
    // intermediates showed the reciprocal exact and only the sign fold after
    // it destroyed.
    //
    // The clamp keeps exp(-x) inside f32; beyond it sigma is already 0 or 1 to
    // f32 precision. The product uses the unclamped x, which is what silu is
    // outside the clamp anyway.
    const aie::vector<float, 16> xc =
        aie::max(aie::min(x, aie::broadcast<float, 16>(40.0f)),
                 aie::broadcast<float, 16>(-40.0f));

    // exp(-xc) by repeated squaring: at t = -xc/256 the argument is small
    // enough that a degree-5 series is good to ~2e-8, and eight squarings
    // bring the range back. All multiplies and adds, so there is no
    // exponent-field arithmetic and no int-float round trip to get wrong.
    constexpr float INV_256 = 1.0f / 256.0f;
    const aie::vector<float, 16> t =
        aie::mul(xc, aie::broadcast<float, 16>(-INV_256)).to_vector<float>(0);

    aie::vector<float, 16> e = aie::broadcast<float, 16>(1.0f / 5.0f);
    e = aie::add(aie::mul(e, t).to_vector<float>(0), aie::broadcast<float, 16>(1.0f / 4.0f));
    e = aie::add(aie::mul(e, t).to_vector<float>(0), aie::broadcast<float, 16>(1.0f / 3.0f));
    e = aie::add(aie::mul(e, t).to_vector<float>(0), aie::broadcast<float, 16>(1.0f / 2.0f));
    e = aie::add(aie::mul(e, t).to_vector<float>(0), one);
    e = aie::add(aie::mul(e, t).to_vector<float>(0), one);
#if SILU_STEP == 1
    return xc;
#endif
    // Written out rather than looped: this sits inside the lane-group loop,
    // and a loop nested there is scheduled together with the outer one.
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
    e = aie::mul(e, e).to_vector<float>(0);
#if SILU_STEP == 2
    return e;
#endif
    const aie::vector<float, 16> a = aie::add(e, one);
#if SILU_STEP == 3
    return a;
#endif

    // 1/a by Newton, r <- r*(2 - a*r). The AIE reciprocal table is bf16 and
    // ships for AIE2 only; a > 0 here, so the bit-pattern seed is well behaved
    // at ~3% and three steps reach f32.
    const aie::vector<int32, 16> abits = a.cast_to<int32>();
    aie::vector<float, 16> r =
        aie::sub(aie::broadcast<int32, 16>(0x7EF311C3), abits).cast_to<float>();
    const aie::vector<float, 16> two = aie::broadcast<float, 16>(2.0f);
    r = aie::mul(r, aie::sub(two, aie::mul(a, r).to_vector<float>(0))).to_vector<float>(0);
    r = aie::mul(r, aie::sub(two, aie::mul(a, r).to_vector<float>(0))).to_vector<float>(0);
    r = aie::mul(r, aie::sub(two, aie::mul(a, r).to_vector<float>(0))).to_vector<float>(0);
#if SILU_STEP == 4 || SILU_STEP == 5
    return r;
#endif
    return aie::mul(x, r).to_vector<float>(0);
#endif
}

// A core's 32 output lanes are the gate half and the up half of the same 16
// values, so the FFN's activation closes locally: silu(g)*u, written as 16
// floats. Without the epilogue the 32 accumulators are stored as they are.

// Quantize a core's epilogue output in place of writing it as f32: int8 codes
// with a scale and a code sum per group of Q4_GROUP, in the layout an
// activation tile carries them. The point is that the next dispatch can then
// read this straight out of DDR - the activation never goes back to the host
// to be quantized, which is the one thing that stops two dispatches being
// chained into a single stream.
//
// A quantization group is 32 values and a core owns N_CORE/2 of them, so this
// needs a core at least 64 columns wide; the fused layer's geometry is 128.
// `grp` is the consumer's activation group, not this kernel's: the tile is
// read back by whichever format the next dispatch's weights are in, and q4g32
// groups 32 values where q8g16 groups 16.
inline void store_quant(float *out, const float *mid, int n, int GRP)
{
    int8_t * codes = (int8_t *) out;
    float *  par   = out + (N_CORE / 2) / 4;    // after the codes, as f32 slots
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
    for (int g = 0; g < n / GRP; g++) {
        const float * v = mid + g * GRP;
        aie::vector<float, 16> vmax = aie::zeros<float, 16>();
        for (int i = 0; i < GRP; i += 16) {
            auto x = aie::load_v<16>(v + i);
            // aie::abs does not hold for f32 here; mask the sign bit.
            vmax = aie::max(vmax, aie::bit_and(x.cast_to<int32>(), absmask)
                                      .cast_to<float>());
        }
        alignas(64) float lanes[16];
        aie::store_v(lanes, vmax);
        float amax = 0.0f;
        for (int i = 0; i < 16; i++) {
            if (lanes[i] > amax) amax = lanes[i];
        }
        const float d   = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / d;
        int sum = 0;
        for (int i = 0; i < GRP; i++) {
            float t = v[i] * inv;
            if (t > 127.0f) t = 127.0f; else if (t < -127.0f) t = -127.0f;
            const int q = (int) (t >= 0.0f ? t + 0.5f : t - 0.5f);
            codes[g * GRP + i] = (int8_t) q;
            sum += q;
        }
        par[g]             = (float) sum;
        par[n / GRP + g]   = d;
    }
}

// silu over a half-vector, built from the 16-lane form.
inline aie::vector<float, VEC / 2> silu_half(const aie::vector<float, VEC / 2> &x)
{
#if GEMV_VEC >= 64
    return aie::concat(silu_f32(x.template extract<16>(0)),
                       silu_f32(x.template extract<16>(1)));
#else
    return silu_f32(x);
#endif
}

inline void store_out(float *out, const aie::accum<accfloat, VEC> &o, int epi)
{
    const aie::vector<float, VEC> ov = o.to_vector<float>(0);
    if (epi) {
        const auto g = ov.template extract<VEC / 2>(0);
        const auto u = ov.template extract<VEC / 2>(1);
        const auto mid = aie::mul(silu_half(g), u).template to_vector<float>(0);
        // Written as a full store with the tail zeroed: a half-lane store to
        // this buffer does not land correctly, and the descriptor drains the
        // whole slot either way.
        aie::store_v(out, aie::concat(mid, aie::zeros<float, VEC / 2>()));
    } else {
        aie::store_v(out, ov);
    }
}

} // namespace

extern "C" {

// Two code widths share one artifact: a 4-bit tile spans twice the K of an
// 8-bit one, which makes both tiles the same number of bytes and lets one
// design serve both. A second artifact would be a second hardware context,
// and the NPU reconfigures the array between them - measured at 2.5 ms
// against 0.1 ms for the work itself.
#ifndef K_TILE_Q4
#define K_TILE_Q4 512
#endif
#ifndef K_TILE_Q8
#define K_TILE_Q8 256
#endif
// Activation tile: K_TILE int8 codes, a f32 sum and a f32 scale per group,
// then the epilogue flag and the code width. Sized for the wider code path
// (512 values, 16 groups), which is 1152 B of payload - the two trailing words
// must sit past it, not inside the scales.
#ifndef ACT_TILE
#define ACT_TILE 2112
#endif
#ifndef N_CORE
#define N_CORE 64
#endif
#ifndef Q4_GROUP
#define Q4_GROUP 32
#endif
#ifndef Q8_GROUP
#define Q8_GROUP 16
#endif

// ACT_RAW: the GEMV cores read a tile a prologue core built, which means one
// extra scalar - the norm's rsqrt, applied to the accumulator before silu.
// ACT_PRO: this object IS that prologue core. The two are separate builds of
// the same source because the quantizer is about a kilobyte of program memory
// and the GEMV cores have none to spare; the prologue tile has a program of
// its own.
#ifndef ACT_RAW
#define ACT_RAW 0
#endif
#ifndef ACT_PRO
#define ACT_PRO 0
#endif
// The rsqrt the prologue leaves in the last tile of a chunk, just before the
// two words every tile ends with.
#define ACT_RMS_W (ACT_TILE / 4 - 3)
// The row length the norm divides by, which the host writes into every raw
// tile. Carrying it per tile is what lets the prologue work one object in,
// one object out, with no header to read and no count of its own to keep in
// step with what the stream pushes.
#define ACT_D_W   (ACT_TILE / 4 - 4)
// Non-zero on the first tile of a chunk. The reduction starts here rather
// than being cleared when the previous one ended, so nothing a dispatch leaves
// behind can reach the next - and the prologue holds no state across a
// dispatch boundary at all.
#define ACT_FIRST_W (ACT_TILE / 4 - 5)
#if ACT_PRO
// Where a tile carries numbers instead of codes the host packed. The codes a
// raw tile would have held are the prologue's output, not its input, so the
// whole tile up to the trailing words is free:
//
//   +0                 acc, K_TILE f32   - written by the dispatch before this
//   +4*K_TILE          hres, K_TILE bf16 - the residual, from the host
//   +6*K_TILE          gamma, K_TILE bf16 - the norm's weight, from the host
//
// 2048 B at K_TILE 256, inside the 2104 a tile leaves. acc is f32 and the
// other two are not, because acc is the one a shim descriptor delivers
// straight out of the projection that produced it - and a DMA cannot convert.
#ifndef ACT_RAW_OFF
#define ACT_RAW_OFF 0
#endif

// The FFN's activation, built on the core out of what the dispatch before it
// left in DDR instead of by the host. A tile carries three bf16 runs at
// ACT_RAW_OFF - the projection's output, the residual and the norm's gamma -
// and this turns them into the codes and group parameters the matmul reads:
//
//   h = acc + hres            the residual add
//   ss += h*h                 the norm's reduction, running over the chunk
//   x = h * gamma             per channel, so it has to precede quantization
//   code, gsum, gd = quant(x) the same per-group scale gemv_pack_act applies
//
// The rsqrt is deliberately absent. It is one scalar over the whole row, the
// matmul is linear in the activation, and the epilogue is where the
// nonlinearity needs it - so it is applied to the accumulator there, which
// also means a tile can be quantized before the reduction it belongs to has
// finished.
//
// Vector code because the core's program memory is full: the scalar form of
// the same function overflowed .text by 1120 bytes.
static float g_ss;   // summed over a chunk's tiles, spent at the last one

// Only the q4g32 path carries it: the stage that needs an activation built on
// the array is the FFN's gate/up, and that is the Q4_K one. Every bound is a
// constant so the loops collapse - with k_tile and the group width passed in,
// the same function was twice the size and did not fit.
__attribute__((noinline)) void act_prologue(const uint8_t *acc,
                                            const uint8_t *hst, int8_t *code,
                                            float *gsum, float *gd)
{
    constexpr int KT = K_TILE_Q4;
    constexpr int GRP = Q4_GROUP;
    const float    *pa = (const float *) acc;
    const bfloat16 *pr = (const bfloat16 *) hst;
    const bfloat16 *pg = pr + KT;
    float ss = 0.0f;
    for (int g = 0; g < KT / GRP; g++) {
        alignas(32) float xb[GRP];
        float amax = 0.0f;
        for (int p = 0; p < GRP; p += 16) {
            const int o = g * GRP + p;
            aie::accum<accfloat, 16> hr, hg;
            hr.from_vector(aie::load_v<16>(pr + o));
            hg.from_vector(aie::load_v<16>(pg + o));
            const aie::vector<float, 16> h =
                aie::add(aie::load_v<16>(pa + o), hr.to_vector<float>(0));
            ss += aie::reduce_add(aie::mul(h, h).to_vector<float>(0));
            const aie::vector<float, 16> x =
                aie::mul(h, hg.to_vector<float>(0)).to_vector<float>(0);
            const float m = aie::reduce_max(aie::abs(x));
            amax = m > amax ? m : amax;
            aie::store_v(xb + p, x);
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
        int sum = 0;
        for (int p = 0; p < GRP; p += 16) {
            const aie::vector<float, 16> q =
                aie::mul(aie::load_v<16>(xb + p),
                         aie::broadcast<float, 16>(inv)).to_vector<float>(0);
            const aie::vector<int32_t, 16> qi = aie::to_fixed<int32_t>(q, 0);
            aie::store_v(code + g * GRP + p, aie::pack(aie::pack(qi)));
            sum += aie::reduce_add(qi);
        }
        gsum[g] = (float) sum;
        gd[g]   = amax * (1.0f / 127.0f);
    }
    g_ss += ss;
}
// The prologue core's entry point. One object in, one out, with nothing
// remembered between dispatches but the running reduction - which the epi
// flag ends. Anything the tile does not carry itself would have to be
// counted, and a count that drifts from what the stream pushes deadlocks.
extern "C" {

// `in` is the tile the dispatch before this one drained into - acc and nothing
// else, so no host ever writes it. `hst` is the host's half: the residual,
// gamma and the words that describe the tile, in a buffer no dispatch writes.
// Keeping the two apart is the whole point: writes from the host and from the
// array hold in one order only when they are not to the same buffer.
void ggml_xdna_act_pro(const int32_t *in, const int32_t *hst, int32_t *out)
{
    const int flags = hst[ACT_TILE / 4 - 2];
    if (!((flags >> 4) & 1)) {
        // Not a raw tile: pass it through. From the host's half when there are
        // two buffers - it is the authoritative description, and the tiles are
        // written only by a dispatch, which does not write the header at all.
        const int32_t *src = (hst == in) ? in : hst;
        for (int i = 0; i < ACT_TILE / 4; i++) {
            out[i] = src[i];
        }
        return;
    }
    if (hst[ACT_FIRST_W]) {
        g_ss = 0.0f;
    }
    uint8_t *o8 = (uint8_t *) out;
    // One buffer for both inputs means the tile carries its own host half, at
    // the offset the host packs it to; two buffers mean the host half starts
    // at zero in its own.
    const uint8_t *h8 = (const uint8_t *) hst +
                        (hst == in ? 4 * K_TILE_Q4 : 0);
    act_prologue((const uint8_t *) in, h8, (int8_t *) o8,
                 (float *) (o8 + K_TILE_Q4),
                 (float *) (o8 + K_TILE_Q4) + K_TILE_Q4 / Q4_GROUP);
    out[ACT_TILE / 4 - 1] = hst[ACT_TILE / 4 - 1];
    out[ACT_TILE / 4 - 2] = flags;
    if (flags & 1) {
        // Last tile of the chunk: the reduction is complete, so the scalar
        // the epilogue needs travels with it. Every chunk replays the same
        // tiles, so the sum starts again here.
        const int d = hst[ACT_D_W];
        const float mean = g_ss / (float) (d > 0 ? d : 1);
        ((float *) out)[ACT_RMS_W] = 1.0f / aie::sqrt(mean + 1e-6f);
    }
}

}  // extern "C"

#endif  // ACT_PRO

// q4g32: unsigned 4-bit codes, w = q*d + m.
static void gemv_q4g32(const uint8_t *w, const int32_t *a32, float *out)
{
    constexpr int K_TILE = K_TILE_Q4;
    constexpr int LG = N_CORE / VEC;               // lane groups of 32 columns
    constexpr int NG = K_TILE / Q4_GROUP;          // groups along K in this tile
    constexpr int CODE_BYTES = Q4_GROUP * VEC / 2; // 32 rows of 32 nibbles
    constexpr int PARAM_BYTES = 4 * VEC;           // d8[32], m8[32] bf16
    constexpr int BLOCK = CODE_BYTES + PARAM_BYTES;
    constexpr int SBG   = 256 / Q4_GROUP;          // groups to a super-block
    constexpr int NSUP  = K_TILE_Q4 / 256 > 0 ? K_TILE_Q4 / 256 : 1;
    constexpr int SUP_BYTES = 4 * VEC * 2;

    const uint8_t *a = (const uint8_t *)a32;
    const int8_t *__restrict ac = (const int8_t *)a;
    const float *__restrict ag = (const float *)(a + K_TILE);
    const float *__restrict ad = ag + NG;   // per-group activation scale
    // Set on the last tile of a chunk: the accumulation is complete, so the
    // gate/up epilogue can close here instead of going back through the host.
    const int flags = a32[ACT_TILE / 4 - 2];
    const int epi = flags & 1;
    // Bit 1 says the activation was written by the cores rather than packed by
    // the host, in which case it is laid out the way a core emits it: blocks of
    // one core's output, each N_CORE/2 codes followed by that block's sums and
    // scales. Keeping the core's own layout means its output object drains
    // into the next dispatch's activation with a single descriptor, which is
    // what lets the two share one instruction stream.
    const int devl = (flags >> 1) & 1;
    // Bit 2 asks the epilogue to write its result as an activation tile rather
    // than as f32, which is what lets the dispatch that consumes it share this
    // one's instruction stream. It is a run-time flag and not a build one
    // because the same artifact also serves the per-op path, where the host
    // reads the f32 back.
    const int qout = (flags >> 2) & 1;
    // Bit 3: group the tile it writes by 16 rather than 32, because the
    // dispatch that reads it is q8g16.
    const int g16 = (flags >> 3) & 1;
    constexpr int BLK_B = N_CORE * 4;              // bytes a core's object holds
    constexpr int GPB   = (N_CORE / 2) / Q4_GROUP; // groups in one such block

    event0();
    alignas(64) float midbuf[N_CORE / 2];
    // The super-block pairs sit after every block of the tile.
    const bfloat16 *sup = (const bfloat16 *)(w + (size_t) LG * NG * BLOCK);
    for (int j = 0; j < LG; j++) {
        aie::accum<accfloat, VEC> o;
        o.from_vector(aie::load_v<VEC>(out + j * VEC));

        // The super-block pair is the same for every group under it, so it is
        // read once here rather than four loads a group. Held as bf16, which
        // is one register each - as f32 it was four and the kernel spilled.
        for (int sb = 0; sb < NSUP; sb++) {
        const bfloat16 *sp = sup + (size_t)(j * NSUP + sb) * (SUP_BYTES / 2);
        const aie::vector<bfloat16, VEC> sdh = aie::load_v<VEC>(sp);
        const aie::vector<bfloat16, VEC> sdl = aie::load_v<VEC>(sp + VEC);
        const aie::vector<bfloat16, VEC> smh = aie::load_v<VEC>(sp + 2 * VEC);
        const aie::vector<bfloat16, VEC> sml = aie::load_v<VEC>(sp + 3 * VEC);
        for (int g = sb * (NG / NSUP); g < (sb + 1) * (NG / NSUP); g++) {
            // Explicit block addressing: a pointer carried across the inner
            // loop reaches the parameter reads with the wrong value once the
            // loop is pipelined.
            const uint8_t *blk = w + (size_t)(j * NG + g) * BLOCK;
            const int8_t *arow = devl
                ? (const int8_t *)(a + (size_t)(g / GPB) * BLK_B) + (g % GPB) * Q4_GROUP
                : ac + g * Q4_GROUP;

            // Four accumulators, not one. A single `p = mac(p, ...)` chain runs
            // at the MAC's latency rather than its throughput - about fifteen
            // cycles for thirty-two lanes of work - and that, not the weight
            // stream, was what held the dispatch to 15.8 GB/s of the array's
            // 28.6. Four independent chains fill the pipeline; the partials
            // are exact, so summing them at the end changes nothing.
            //
            // Four, not eight: at eight the loop needs sixteen accumulator
            // registers and spills, which measures worse than the single chain
            // it replaced - gate/up 302 us at four, 367 at eight, 348 at one.
            aie::accum<acc32, VEC> p0 = aie::zeros<acc32, VEC>();
            aie::accum<acc32, VEC> p1 = aie::zeros<acc32, VEC>();
            aie::vector<int32, VEC> psum;
            if constexpr (CHAINS == 2) {
                for (int k = 0; k < Q4_GROUP; k += 2)
                    chess_prepare_for_pipelining chess_loop_range(Q4_GROUP / 2, ) {
                        const uint8_t *b = blk + k * (VEC / 2);
                        const aie::vector<uint8, 2 * VEC> u01 = q4_rows(b);
                        p0 = aie::mac(p0, u01.template extract<VEC>(0).template cast_to<int8>(),
                                      arow[k + 0]);
                        p1 = aie::mac(p1, u01.template extract<VEC>(1).template cast_to<int8>(),
                                      arow[k + 1]);
                    }
                psum = aie::add(p0.template to_vector<int32>(0),
                                p1.template to_vector<int32>(0));
            } else {
                aie::accum<acc32, VEC> p2 = aie::zeros<acc32, VEC>();
                aie::accum<acc32, VEC> p3 = aie::zeros<acc32, VEC>();
                for (int k = 0; k < Q4_GROUP; k += 4)
                    chess_prepare_for_pipelining chess_loop_range(Q4_GROUP / 4, ) {
                        const uint8_t *b = blk + k * (VEC / 2);
                        const aie::vector<uint8, 2 * VEC> u01 = q4_rows(b);
                        const aie::vector<uint8, 2 * VEC> u23 = q4_rows(b + VEC);
                        // The scalar overload of mac, not a broadcast: the
                        // vector unit takes the coefficient from the scalar
                        // register, so the broadcast that materialised it was
                        // one op per multiply of pure overhead.
                        p0 = aie::mac(p0, u01.template extract<VEC>(0).template cast_to<int8>(), arow[k + 0]);
                        p1 = aie::mac(p1, u01.template extract<VEC>(1).template cast_to<int8>(), arow[k + 1]);
                        p2 = aie::mac(p2, u23.template extract<VEC>(0).template cast_to<int8>(), arow[k + 2]);
                        p3 = aie::mac(p3, u23.template extract<VEC>(1).template cast_to<int8>(), arow[k + 3]);
                    }
                psum = aie::add(aie::add(p0.template to_vector<int32>(0), p1.template to_vector<int32>(0)),
                                aie::add(p2.template to_vector<int32>(0), p3.template to_vector<int32>(0)));
            }

            // d = dS * d8 and m = mS * m8, both exact: dS and mS carry the
            // ggml super-block's f16 parameters in a bf16 pair and d8, m8 are
            // its integer per-group scales.
            // The same two native multiply-accumulates pair_to_f32 does, with
            // the group's integer in place of the one: (hi + lo) * d8 in the
            // f32 accumulator, which is exact because d8 is an integer under
            // 256. Read here and not before the multiply loop - holding the
            // pair across it costs two vector registers the kernel has none to
            // spare of, and hoisting measured 25% slower.
            const bfloat16 *dp = (const bfloat16 *)(blk + CODE_BYTES);
            const auto d8 = aie::load_v<VEC>(dp);
            const auto m8 = aie::load_v<VEC>(dp + VEC);
            aie::accum<accfloat, VEC> da_ = aie::mul(sdh, d8);
            da_ = aie::mac(da_, sdl, d8);
            aie::accum<accfloat, VEC> ma_ = aie::mul(smh, m8);
            ma_ = aie::mac(ma_, sml, m8);
            const aie::vector<float, VEC> d = da_.template to_vector<float>(0);
            const aie::vector<float, VEC> m = ma_.template to_vector<float>(0);

            // The integer partial is exact (|sum| < 2^24), so the only
            // rounding is the f32 rescale, once per group. The group's own
            // activation scale closes the term here, so no factor is left for
            // the host to apply.
            const float * apar = (const float *)(a + (size_t)(g / GPB) * BLK_B)
                                 + (N_CORE / 2) / 4;
            const float asum  = devl ? apar[g % GPB] : ag[g];
            const float ascal = devl ? apar[GPB + (g % GPB)] : ad[g];
            const aie::vector<float, VEC> da = aie::broadcast<float, VEC>(ascal);
            const aie::vector<float, VEC> pf = aie::to_float<float>(psum, 0);
            aie::vector<float, VEC> t = aie::mul(pf, d).template to_vector<float>(0);
            t = aie::add(t, aie::mul(m, aie::broadcast<float, VEC>(asum))
                                 .template to_vector<float>(0));
            o = aie::add(o, aie::mul(t, da).template to_vector<float>(0));
        }
        }

        if (epi && qout) {
            auto ov = o.to_vector<float>(0);
#if ACT_RAW
            // The prologue quantized this activation without the norm's
            // rsqrt - one scalar over the whole row, which the matmul's
            // linearity lets us apply here instead, where silu needs it.
            if ((flags >> 4) & 1) {
                ov = aie::mul(ov, aie::broadcast<float, VEC>(
                                      ((const float *) a)[ACT_RMS_W]))
                         .template to_vector<float>(0);
            }
#endif
            aie::store_v(midbuf + j * (VEC / 2),
                         aie::mul(silu_half(ov.template extract<VEC / 2>(0)),
                                  ov.template extract<VEC / 2>(1))
                             .template to_vector<float>(0));
        } else {
            store_out(out + j * VEC, o, epi);
        }
    }
    if (epi && qout) {
        store_quant(out, midbuf, N_CORE / 2, g16 ? Q8_GROUP : Q4_GROUP);
    }
    event1();
}

// q8g16: signed 8-bit codes, w = q*d.
static void gemv_q8g16(const uint8_t *w, const int32_t *a32, float *out)
{
    constexpr int K_TILE = K_TILE_Q8;
    constexpr int LG = N_CORE / VEC;
    constexpr int NG = K_TILE / Q8_GROUP;
    constexpr int CODE_BYTES = Q8_GROUP * VEC;
    constexpr int PARAM_BYTES = 4 * VEC;       // d8[32], m8[32] bf16
    constexpr int BLOCK = CODE_BYTES + PARAM_BYTES;
    constexpr int SBG   = 256 / Q8_GROUP;
    constexpr int NSUP  = K_TILE_Q4 / 256 > 0 ? K_TILE_Q4 / 256 : 1;
    constexpr int SUP_BYTES = 4 * VEC * 2;

    const uint8_t *a = (const uint8_t *)a32;
    const int8_t *__restrict ac = (const int8_t *)a;
    const float *__restrict ag = (const float *)(a + K_TILE);
    const float *__restrict ad = ag + NG;   // per-group activation scale
    // Set on the last tile of a chunk: the accumulation is complete, so the
    // gate/up epilogue can close here instead of going back through the host.
    const int flags = a32[ACT_TILE / 4 - 2];
    const int epi = flags & 1;
    // The same flags as the q4 path, and for the same reasons: this is the
    // format the FFN's down projection is usually in, so it is the one reading
    // what the epilogue wrote - with Q8_GROUP-wide groups, which is what bit 3
    // asks the producing core for.
    const int devl  = (flags >> 1) & 1;
    const int qout  = (flags >> 2) & 1;
    const int g16   = (flags >> 3) & 1;
    constexpr int BLK_B = N_CORE * 4;
    constexpr int GPB   = (N_CORE / 2) / Q8_GROUP;

    event0();
    alignas(64) float midbuf[N_CORE / 2];
    const bfloat16 *sup = (const bfloat16 *)(w + (size_t) LG * NG * BLOCK);
    for (int j = 0; j < LG; j++) {
        aie::accum<accfloat, VEC> o;
        o.from_vector(aie::load_v<VEC>(out + j * VEC));

        // The super-block pair is the same for every group under it, so it is
        // read once here rather than four loads a group. Held as bf16, which
        // is one register each - as f32 it was four and the kernel spilled.
        for (int sb = 0; sb < NSUP; sb++) {
        const bfloat16 *sp = sup + (size_t)(j * NSUP + sb) * (SUP_BYTES / 2);
        const aie::vector<bfloat16, VEC> sdh = aie::load_v<VEC>(sp);
        const aie::vector<bfloat16, VEC> sdl = aie::load_v<VEC>(sp + VEC);
        const aie::vector<bfloat16, VEC> smh = aie::load_v<VEC>(sp + 2 * VEC);
        const aie::vector<bfloat16, VEC> sml = aie::load_v<VEC>(sp + 3 * VEC);
        for (int g = sb * (NG / NSUP); g < (sb + 1) * (NG / NSUP); g++) {
            const uint8_t *blk = w + (size_t)(j * NG + g) * BLOCK;
            const int8_t *arow = devl
                ? (const int8_t *)(a + (size_t)(g / GPB) * BLK_B) + (g % GPB) * Q8_GROUP
                : ac + g * Q8_GROUP;

            // Four chains, for the reason the q4 path gives above.
            aie::accum<acc32, VEC> p0 = aie::zeros<acc32, VEC>();
            aie::accum<acc32, VEC> p1 = aie::zeros<acc32, VEC>();
            aie::vector<int32, VEC> psum;
            if constexpr (CHAINS == 2) {
                for (int k = 0; k < Q8_GROUP; k += 2)
                    chess_prepare_for_pipelining chess_loop_range(Q8_GROUP / 2, ) {
                        const uint8_t *b = blk + k * VEC;
                        const aie::vector<int8, 2 * VEC> u01 = q8_rows(b);
                        p0 = aie::mac(p0, u01.template extract<VEC>(0), arow[k + 0]);
                        p1 = aie::mac(p1, u01.template extract<VEC>(1), arow[k + 1]);
                    }
                psum = aie::add(p0.template to_vector<int32>(0),
                                p1.template to_vector<int32>(0));
            } else {
                aie::accum<acc32, VEC> p2 = aie::zeros<acc32, VEC>();
                aie::accum<acc32, VEC> p3 = aie::zeros<acc32, VEC>();
                for (int k = 0; k < Q8_GROUP; k += 4)
                    chess_prepare_for_pipelining chess_loop_range(Q8_GROUP / 4, ) {
                        const uint8_t *b = blk + k * VEC;
                        const aie::vector<int8, 2 * VEC> u01 = q8_rows(b);
                        const aie::vector<int8, 2 * VEC> u23 = q8_rows(b + 2 * VEC);
                        p0 = aie::mac(p0, u01.template extract<VEC>(0), arow[k + 0]);
                        p1 = aie::mac(p1, u01.template extract<VEC>(1), arow[k + 1]);
                        p2 = aie::mac(p2, u23.template extract<VEC>(0), arow[k + 2]);
                        p3 = aie::mac(p3, u23.template extract<VEC>(1), arow[k + 3]);
                    }
                psum = aie::add(aie::add(p0.template to_vector<int32>(0), p1.template to_vector<int32>(0)),
                                aie::add(p2.template to_vector<int32>(0), p3.template to_vector<int32>(0)));
            }

            // d = dS * d8 and m = mS * m8, both exact: dS and mS carry the
            // ggml super-block's f16 parameters in a bf16 pair and d8, m8 are
            // its integer per-group scales.
            // The same two native multiply-accumulates pair_to_f32 does, with
            // the group's integer in place of the one: (hi + lo) * d8 in the
            // f32 accumulator, which is exact because d8 is an integer under
            // 256. Read here and not before the multiply loop - holding the
            // pair across it costs two vector registers the kernel has none to
            // spare of, and hoisting measured 25% slower.
            const bfloat16 *dp = (const bfloat16 *)(blk + CODE_BYTES);
            const auto d8 = aie::load_v<VEC>(dp);
            const auto m8 = aie::load_v<VEC>(dp + VEC);
            aie::accum<accfloat, VEC> da_ = aie::mul(sdh, d8);
            da_ = aie::mac(da_, sdl, d8);
            aie::accum<accfloat, VEC> ma_ = aie::mul(smh, m8);
            ma_ = aie::mac(ma_, sml, m8);
            const aie::vector<float, VEC> d = da_.template to_vector<float>(0);
            const aie::vector<float, VEC> m = ma_.template to_vector<float>(0);
            const float * apar = (const float *)(a + (size_t)(g / GPB) * BLK_B)
                                 + (N_CORE / 2) / 4;
            const float asum  = devl ? apar[g % GPB] : ag[g];
            const float ascal = devl ? apar[GPB + (g % GPB)] : ad[g];
            const aie::vector<float, VEC> da = aie::broadcast<float, VEC>(ascal);
            const aie::vector<float, VEC> pf = aie::to_float<float>(psum, 0);
            aie::vector<float, VEC> t = aie::mul(pf, d).template to_vector<float>(0);
            t = aie::add(t, aie::mul(m, aie::broadcast<float, VEC>(asum))
                                 .template to_vector<float>(0));
            o = aie::add(o, aie::mul(t, da).template to_vector<float>(0));
        }
        }

        if (epi && qout) {
            const auto ov = o.to_vector<float>(0);
            aie::store_v(midbuf + j * (VEC / 2),
                         aie::mul(silu_half(ov.template extract<VEC / 2>(0)),
                                  ov.template extract<VEC / 2>(1))
                             .template to_vector<float>(0));
        } else {
            store_out(out + j * VEC, o, epi);
        }
    }
    if (epi && qout) {
        store_quant(out, midbuf, N_CORE / 2, g16 ? Q8_GROUP : Q4_GROUP);
    }
    event1();
}

// The activation tile carries the code width of this dispatch in its last
// word, so one entry point serves both.
void ggml_xdna_gemv(const uint8_t *w, const int32_t *a32, float *out)
{
    if (a32[ACT_TILE / 4 - 1] == 0) {
        gemv_q4g32(w, a32, out);
    } else {
        gemv_q8g16(w, a32, out);
    }
}

} // extern "C"
