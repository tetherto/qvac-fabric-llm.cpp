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
//               nibble first, the 8-bit form one int8 per value (both
//               groups of 32; the 8-bit tile is padded to the 4-bit size)
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
// 1 / d in fp32 lanes: a bit-trick guess and three Newton steps. Shared by
// silu and the quantizer rather than inlined into both.
__attribute__((noinline)) aie::vector<float, 16> recip_f32(const aie::vector<float, 16> d)
{
    const auto two = aie::broadcast<float, 16>(2.0f);
    auto r = aie::sub(aie::broadcast<int32, 16>(0x7EF311C7), d.cast_to<int32>()).cast_to<float>();
    for (int it = 0; it < 3; ++it) {
        r = aie::mul(r, aie::sub(two, aie::mul(d, r).to_vector<float>(0))).to_vector<float>(0);
    }
    return r;
}

inline aie::vector<float, 16> silu_f32(const aie::vector<float, 16> &x)
{
#if defined(SILU_IDENTITY)
    // Diagnostic: the epilogue runs but the nonlinearity is the identity, so
    // a failure can only be the plumbing around it.
    return x;
#else
    // sigma(x) = 1 / (1 + exp(-x)): exp(-x) = 2^n * 2^f from the exponent
    // bits and a degree-6 polynomial on f in [-0.5, 0.5] (~2e-7), then the
    // reciprocal by a bit-trick guess and three Newton steps. The same as
    // silu-f32.h; the repeated-squaring form it replaces was 1.4 KB larger in
    // a program memory the attention mode needs.
    const auto bc = [](float v) { return aie::broadcast<float, 16>(v); };
    auto t = aie::mul(x, bc(-1.4426950408889634f)).to_vector<float>(0);
    t = aie::min(aie::max(t, bc(-126.0f)), bc(126.0f));
    const auto magic = bc(12582912.0f);
    const auto tm = aie::add(t, magic);
    const auto f  = aie::sub(t, aie::sub(tm, magic));
    const auto ni = aie::sub(tm.cast_to<int32>(), magic.cast_to<int32>());
    auto p = aie::add(aie::mul(f, bc(1.5403530393381606e-4f)).to_vector<float>(0), bc(1.3333558146428443e-3f));
    p = aie::add(aie::mul(p, f).to_vector<float>(0), bc(9.6181291076284772e-3f));
    p = aie::add(aie::mul(p, f).to_vector<float>(0), bc(5.5504108664821580e-2f));
    p = aie::add(aie::mul(p, f).to_vector<float>(0), bc(2.4022650695910071e-1f));
    p = aie::add(aie::mul(p, f).to_vector<float>(0), bc(6.9314718055994531e-1f));
    p = aie::add(aie::mul(p, f).to_vector<float>(0), bc(1.0f));
    const auto sc = aie::upshift(aie::add(ni, aie::broadcast<int32, 16>(127)), 23).cast_to<float>();
    const auto d = aie::add(aie::mul(p, sc).to_vector<float>(0), bc(1.0f));
    return aie::mul(x, recip_f32(d)).to_vector<float>(0);
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
// read back by whichever format the next dispatch's weights are in (both
// group 32 values now; the flag stays for a build with other groups).
inline void store_quant(float *out, const float *mid, int n, int GRP)
{
    // All in vector lanes: the core's scalar unit has no FPU, and the scalar
    // form of this - a divide, a reciprocal, a compare and a clamp per value -
    // pulled in ~3 KB of software float that the GEMV cores' program memory
    // could not spare. Codes round to nearest-even (the 1.5 * 2^23 trick) and
    // the reciprocal is a bit-trick guess refined by Newton.
    int8_t * codes = (int8_t *) out;
    float *  par   = out + (N_CORE / 2) / 4;    // after the codes, as f32 slots
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
    const auto magic   = aie::broadcast<float, 16>(12582912.0f);
    const auto magici  = magic.cast_to<int32>();
    const auto lo      = aie::broadcast<float, 16>(-127.0f);
    const auto hi      = aie::broadcast<float, 16>(127.0f);
    for (int g = 0; g < n / GRP; g++) {
        const float * v = mid + g * GRP;
        aie::vector<float, 16> vmax = aie::zeros<float, 16>();
        for (int i = 0; i < GRP; i += 16) {
            auto x = aie::load_v<16>(v + i);
            vmax = aie::max(vmax, aie::bit_and(x.cast_to<int32>(), absmask)
                                      .cast_to<float>());
        }
        // amax in every lane; a zero group gets a harmless scale (its codes
        // are zero whatever the scale is)
        const auto am = aie::max(aie::broadcast<float, 16>(aie::reduce_max(vmax)),
                                 aie::broadcast<float, 16>(1e-30f));
        const auto inv = aie::mul(recip_f32(am), hi).to_vector<float>(0);   // 127 / amax
        const auto dv  = aie::mul(am, aie::broadcast<float, 16>(1.0f / 127.0f)).to_vector<float>(0);
        aie::vector<float, 16> qsum = aie::zeros<float, 16>();
        for (int i = 0; i < GRP; i += 16) {
            auto t = aie::mul(aie::load_v<16>(v + i), inv).to_vector<float>(0);
            t = aie::min(aie::max(t, lo), hi);
            const auto tm = aie::add(t, magic);
            qsum = aie::add(qsum, aie::sub(tm, magic));
            const auto qi = aie::sub(tm.cast_to<int32>(), magici);
            aie::store_v(codes + g * GRP + i, aie::pack(aie::pack(qi)));
        }
        par[g]           = aie::reduce_add(qsum);
        par[n / GRP + g] = dv[0];
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
#define Q8_GROUP 32
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
// The prologue tile: every activation object passes it on the way to the
// cores. Tiles with flags bit 5 are the attention layer's and the layer
// boundary's work (act-att.cc, appended to this build); every other one
// passes through.

// Scalar float arithmetic has no hardware on this core: every divide,
// multiply, add, compare and int conversion was a software routine, 2.4 KB of
// a program the attention layer's work needs. These do it in a vector lane.
static inline aie::vector<float, 16> v_bc(float a) { return aie::broadcast<float, 16>(a); }
// An fp32 vector product is several of the MAC's bf16 ones, a few hundred
// bytes inline each: out of line, once.
__attribute__((noinline)) aie::vector<float, 16> vmul(const aie::vector<float, 16> a,
                                                      const aie::vector<float, 16> b)
{
    return aie::mul(a, b).to_vector<float>(0);
}
__attribute__((noinline)) float v_div(float a, float b)
{
    return vmul(v_bc(a), recip_f32(v_bc(b))).get(0);
}
static inline float v_mul(float a, float b) { return vmul(v_bc(a), v_bc(b)).get(0); }
static inline float v_add(float a, float b) { return aie::add(v_bc(a), v_bc(b)).get(0); }
static inline float v_max(float a, float b) { return aie::max(v_bc(a), v_bc(b)).get(0); }
static inline float v_i2f(int i) { return aie::to_float(aie::broadcast<int32, 16>(i), 0).get(0); }
// a > 0 for a finite value: the sign and the bits, as an integer
static inline bool v_pos(float a) { int32_t w; __builtin_memcpy(&w, &a, 4); return w > 0; }

// act-att.cc
void quant_tile(const float *x, int n, int grp, int8_t *code);

extern "C" {

// An attention layer's tile (act-att.cc, appended to this build).
void act_att_tile(const int32_t *in, int32_t *out);

void ggml_xdna_act_pro(const int32_t *in, const int32_t *hst, int32_t *out)
{
    const int flags = hst[ACT_TILE / 4 - 2];
    if ((flags >> 5) & 1) {
        act_att_tile(in, out);
        return;
    }
    // Every other tile passes through. (The raw tiles the host half-packed
    // for this tile to finish are gone: the layer boundaries are rows now,
    // act-att.cc's mode 4.)
    (void) hst;
    for (int i = 0; i < ACT_TILE / 4; i += 16) {
        aie::store_v(out + i, aie::load_v<16>(in + i));
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
    constexpr int NSUP  = K_TILE / 256 > 0 ? K_TILE / 256 : 1;
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
    // Bit 3: group the tile it writes by Q8_GROUP rather than Q4_GROUP,
    // because the dispatch that reads it is the 8-bit form (the two are both
    // 32 now, so it changes nothing).
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

// The 8-bit form: signed 8-bit codes, w = q*d + m, groups of Q8_GROUP.
static void gemv_q8g16(const uint8_t *w, const int32_t *a32, float *out)
{
    constexpr int K_TILE = K_TILE_Q8;
    constexpr int LG = N_CORE / VEC;
    constexpr int NG = K_TILE / Q8_GROUP;
    constexpr int CODE_BYTES = Q8_GROUP * VEC;
    constexpr int PARAM_BYTES = 4 * VEC;       // d8[32], m8[32] bf16
    constexpr int BLOCK = CODE_BYTES + PARAM_BYTES;
    constexpr int SBG   = 256 / Q8_GROUP;
    constexpr int NSUP  = K_TILE / 256 > 0 ? K_TILE / 256 : 1;
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
