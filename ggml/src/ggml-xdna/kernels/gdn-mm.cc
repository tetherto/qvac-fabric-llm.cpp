//
// The prefill gated delta rule on the mmul (FLM_PREFILL_PLAN.md, step 3): one
// core holds one head's state for half of the value columns, 128 x 64, for
// the whole ubatch, and advances it a chunk of GDN_C tokens at a time in the
// chunked form probes/gdn_chunk_ref.py checks against ggml's recurrence:
//
//   gam = cumsum(g) (in log2 units: the host scales g by log2(e))
//   D[t, s] = 2^(gam_t - gam_s), s <= t
//   T = (I + tril(beta_t D (K K^T), -1))^-1        (forward substitution)
//   Delta = T diag(beta) (V - diag(G sigma) K S)
//   O = scale (diag(G sigma) Q S + tril(D (Q K^T)) Delta)
//   S += (diag(2^(gam_C - gam) / sigma') K)^T Delta,  sigma' = sigma 2^gam_C
//
// The true state is sigma * S; sigma is kept as its log2 and S rescaled by a
// power of two in the exponent field when it runs low, so no f32 product ever
// touches the state (this core emulates f32 products, ~60 cycles a vector).
// S is a bf16 hi/lo pair: hi is the matmul operand, hi + lo what the update
// adds to. The matmuls are the native bf16 (4, 8, 8): bfp16 products would be
// 4.5x the error through the recurrence.
//
// Every matrix is stored as 8 x 8 tiles in row-major order, (rows/8, cols/8,
// 8, 8); a transposed operand is read through the 16-bit 8 x 8 transpose.
//
// The input object of a chunk: K (C x 128), Q (C x 128), V half (C x 64), all
// bf16 in 8 x 8 tiles, then f32: G sigma, G sigma scale, (G_C / G) / sigma'
// and beta (C each), beta_t D (strictly lower) and D scale (lower), C x C
// in 8 x 8 tiles, and the power of two the state is to be scaled down by
// before the chunk's update (0: none). The host works the scalars out in f32
// - the gates' cumulative sums, sigma and when it is renormalized depend on
// nothing else - so the core does no scalar f32 arithmetic (its software
// library did not fit the program memory), and every per-token factor is
// f32-exact: rounded to bf16 a factor scales a token's whole output, a
// coherent error the next projection does not average out (KLD 0.11 against
// 0.005 at the same rms error as independent noise).

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GDN_C
#define GDN_C 16
#endif

namespace {

constexpr int C  = GDN_C;     // tokens a chunk
constexpr int DK = 128;       // key dim, the state's rows
constexpr int DV = 64;        // this core's value columns

using v16f  = aie::vector<float, 16>;
using v32f  = aie::vector<float, 32>;
using v64f  = aie::vector<float, 64>;
using v32bf = aie::vector<bfloat16, 32>;
using v64bf = aie::vector<bfloat16, 64>;
using MMUL  = aie::mmul<4, 8, 8, bfloat16, bfloat16, accauto>;

// C += op(A) op(B), op(X) = X or X^T; M x K by K x N. `lda` etc. are the
// tiles a stored row of each operand spans, so a block of a larger matrix
// can be named by its first tile. Two by two output tiles a pass over K, so
// four accumulators keep the mmul fed (one a pass ran at ~13 GMAC/s); a
// transposed A tile serves both of its halves from one transpose.
// A's rows 4z .. 4z + 7 (z even) against k-block i: one 8 x 8 tile, both
// halves of it.
template <bool AT>
inline v64bf a_pair(const bfloat16 *a, int lda, int z, int i)
{
    if constexpr (AT) {
        // A = X^T, X (K x M): the tile (k-block i, m-block z/2), transposed
        return aie::transpose(aie::load_v<64>(a + (i * lda + (z >> 1)) * 64), 8, 8);
    } else {
        return aie::load_v<64>(a + ((z >> 1) * lda + i) * 64);
    }
}

template <bool BT>
inline v64bf b_tile(const bfloat16 *b, int ldb, int i, int j)
{
    if constexpr (BT) {
        // B = Y^T, Y (N x K): the tile (n-block j, k-block i)
        return aie::transpose(aie::load_v<64>(b + (j * ldb + i) * 64), 8, 8);
    } else {
        return aie::load_v<64>(b + (i * ldb + j) * 64);
    }
}

template <bool AT, bool BT>
__attribute__((noinline)) void mm(int M, int K, int N, const bfloat16 *a, int lda,
                                  const bfloat16 *b, int ldb, float *c, int ldc)
{
    for (int z = 0; z < M / 4; z += 2) {
        for (int j = 0; j < N / 8; j += 2) {
            float *c00 = c + ((z >> 1) * ldc + j) * 64;       // rows z, z + 1 are one tile row
            float *c01 = c00 + 64;
            MMUL a00(aie::load_v<32>(c00)), a01(aie::load_v<32>(c01));
            MMUL a10(aie::load_v<32>(c00 + 32)), a11(aie::load_v<32>(c01 + 32));
            for (int i = 0; i < K / 8; i++)
                chess_prepare_for_pipelining chess_loop_range(2, )
            {
                v32bf x0, x1;
                if constexpr (AT) {
                    const v64bf x = a_pair<AT>(a, lda, z, i);
                    x0 = x.template extract<32>(0);
                    x1 = x.template extract<32>(1);
                } else {
                    const bfloat16 *pa = a + ((z >> 1) * lda + i) * 64;
                    x0 = aie::load_v<32>(pa);
                    x1 = aie::load_v<32>(pa + 32);
                }
                const v64bf y0 = b_tile<BT>(b, ldb, i, j);
                const v64bf y1 = b_tile<BT>(b, ldb, i, j + 1);
                a00.mac(x0, y0);
                a01.mac(x0, y1);
                a10.mac(x1, y0);
                a11.mac(x1, y1);
            }
            aie::store_v(c00, a00.template to_vector<float>());
            aie::store_v(c01, a01.template to_vector<float>());
            aie::store_v(c00 + 32, a10.template to_vector<float>());
            aie::store_v(c01 + 32, a11.template to_vector<float>());
        }
    }
}

// n f32 to bf16, in order
void to_bf16(const float *x, bfloat16 *y, int n)
{
    for (int i = 0; i < n; i += 32) {
        aie::store_v(y + i, aie::accum<accfloat, 32>(aie::load_v<32>(x + i)).to_vector<bfloat16>(0));
    }
}

void zero_f(float *x, int n)
{
    for (int i = 0; i < n; i += 16) {
        aie::store_v(x + i, aie::zeros<float, 16>());
    }
}

// y = a * b, n f32 (this core emulates the products; only C x C of them)
void mul_f(const float *a, const float *b, float *y, int n)
{
    for (int i = 0; i < n; i += 16) {
        aie::store_v(y + i, aie::mul(aie::load_v<16>(a + i), aie::load_v<16>(b + i)).to_vector<float>(0));
    }
}

// x (C x C, f32, 8 x 8 tiles) times a per-column factor, in place
void scale_cols_f(float *x, const float *f, float *y)
{
    for (int cb = 0; cb < C / 8; cb++) {
        alignas(64) float p[64];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                p[r * 8 + c] = f[cb * 8 + c];
            }
        }
        for (int rb = 0; rb < C / 8; rb++) {
            const int o = (rb * (C / 8) + cb) * 64;
            mul_f(x + o, p, y + o, 64);
        }
    }
}

// n f32 into a bf16 hi/lo pair
void split_hl(const float *x, bfloat16 *hi, bfloat16 *lo, int n)
{
    for (int i = 0; i < n; i += 32) {
        const v32f v = aie::load_v<32>(x + i);
        const v32bf h = aie::accum<accfloat, 32>(v).to_vector<bfloat16>(0);
        aie::store_v(hi + i, h);
        aie::store_v(lo + i, aie::accum<accfloat, 32>(
                                 aie::sub(v, aie::accum<accfloat, 32>(h).to_vector<float>(0)))
                                 .to_vector<bfloat16>(0));
    }
}

// y = the rows of x (rows x cols, f32) scaled by f[row] (f32), both as bf16
// hi/lo pairs: x_h p_h + x_h p_l + x_l p_h accumulated in f32, near f32
// exact at native speed (an f32 product is emulated)
void scale_rows_ff(const float *x, int rows, int cols, const float *f, float *y)
{
    for (int rb = 0; rb < rows / 8; rb++) {
        alignas(64) float p[64];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                p[r * 8 + c] = f[rb * 8 + r];
            }
        }
        for (int h = 0; h < 2; h++) {
            const v32f pf = aie::load_v<32>(p + 32 * h);
            const v32bf ph = aie::accum<accfloat, 32>(pf).to_vector<bfloat16>(0);
            const v32bf pl = aie::accum<accfloat, 32>(
                                 aie::sub(pf, aie::accum<accfloat, 32>(ph).to_vector<float>(0)))
                                 .to_vector<bfloat16>(0);
            for (int cb = 0; cb < cols / 8; cb++) {
                const int o = (rb * (cols / 8) + cb) * 64 + 32 * h;
                const v32f xf = aie::load_v<32>(x + o);
                const v32bf xh = aie::accum<accfloat, 32>(xf).to_vector<bfloat16>(0);
                const v32bf xl = aie::accum<accfloat, 32>(
                                     aie::sub(xf, aie::accum<accfloat, 32>(xh).to_vector<float>(0)))
                                     .to_vector<bfloat16>(0);
                aie::accum<accfloat, 32> acc = aie::mul(xh, ph);
                acc = aie::mac(acc, xh, pl);
                acc = aie::mac(acc, xl, ph);
                aie::store_v(y + o, acc.to_vector<float>(0));
            }
        }
    }
}

// T = (I + L)^-1 for L strictly lower (C x C, f32, 8 x 8 tiles), row by
// row: T_t = e_t - sum_{s<t} L_ts T_s. The coefficients and the rows are
// bf16 hi/lo pairs, their products accumulated in f32 - the core's native
// arithmetic, near f32 exact.
void inverse_unit_lower(float *lt, bfloat16 *lh, bfloat16 *ll, bfloat16 *th, bfloat16 *tl)
{
    static_assert(C == 16, "a row of T is one 16-lane half of a 32-lane vector");
    // L row-major as a hi/lo pair (lh, ll: C x C); th, tl: the rows of T,
    // each padded to 32
    alignas(64) float lr[C];
    for (int t = 0; t < C; t++) {
        for (int s = 0; s < C; s++) {
            lr[s] = lt[((t / 8) * (C / 8) + s / 8) * 64 + (t % 8) * 8 + s % 8];
        }
        const v16f r = aie::load_v<16>(lr);
        const aie::vector<bfloat16, 16> h = aie::accum<accfloat, 16>(r).to_vector<bfloat16>(0);
        aie::store_v(lh + t * C, h);
        aie::store_v(ll + t * C, aie::accum<accfloat, 16>(
                                     aie::sub(r, aie::accum<accfloat, 16>(h).to_vector<float>(0)))
                                     .to_vector<bfloat16>(0));
    }
    for (int t = 0; t < C; t++) {
        alignas(64) float e[32] = {};
        e[t] = 1.0f;
        aie::accum<accfloat, 32> acc;
        acc.from_vector(aie::load_v<32>(e), 0);
        for (int s = 0; s < t; s++) {
            const v32bf ch = aie::broadcast<bfloat16, 32>(lh[t * C + s]);
            const v32bf cl = aie::broadcast<bfloat16, 32>(ll[t * C + s]);
            const v32bf rh = aie::load_v<32>(th + s * 32);
            acc = aie::msc(acc, ch, rh);
            acc = aie::msc(acc, ch, aie::load_v<32>(tl + s * 32));
            acc = aie::msc(acc, cl, rh);
        }
        const v32f row = acc.to_vector<float>(0);
        const v32bf h = aie::accum<accfloat, 32>(row).to_vector<bfloat16>(0);
        aie::store_v(th + t * 32, h);
        aie::store_v(tl + t * 32, aie::accum<accfloat, 32>(
                                      aie::sub(row, aie::accum<accfloat, 32>(h).to_vector<float>(0)))
                                      .to_vector<bfloat16>(0));
        alignas(64) float rr[32];
        aie::store_v(rr, row);
        for (int s = 0; s < C; s++) {
            lt[((t / 8) * (C / 8) + s / 8) * 64 + (t % 8) * 8 + s % 8] = rr[s];
        }
    }
}

// The chunk's scratch, all in 8 x 8 tiles.
alignas(64) float    g_x[C * DV];          // K S, R, Delta, the update's block
float *const         g_cc = g_x;           // K K^T, T, Q K^T: done before K S
alignas(64) bfloat16 g_l[C * C];           // L, then A hi
alignas(64) bfloat16 g_t[C * C];           // T beta hi
alignas(64) bfloat16 g_p[C * C];           // T beta lo
alignas(64) bfloat16 g_ip[C * C];          // A lo
alignas(64) bfloat16 g_rb[C * DV];         // beta R, then Delta

// The chunk's per-token factors, worked out here in the input object's own
// factor region from what the host puts at its start (the gates g (natural
// log), beta, log2 sigma before and after the chunk's renormalization, the
// renormalization's shift), in f32: they scale a token's whole output, so
// each must be f32-exact (a rounded one is a coherent error the next
// projection does not average out). The products are the core's emulated
// f32 ones - a few dozen vectors a chunk.
int32_t g_shift = 0;
float   g_sigma = 1.0f;                 // 2^ls after the last chunk: the state's factor

using v16i = aie::vector<int32, 16>;

// 2^x for 16 f32: 2^n into the exponent field (n clamped: past it the
// factor is 0 or overflows as it would in f32), 2^f, |f| <= 1/2, a degree-6
// polynomial (relative error ~2e-7)
__attribute__((noinline)) v16f exp2_f(const v16f &x)
{
    const v16f magic = aie::broadcast<float, 16>(12582912.0f);
    const v16f y = aie::add(x, magic);
    const v16f f = aie::sub(x, aie::sub(y, magic));
    v16i n = aie::sub(y.cast_to<int32>(), aie::broadcast<int32, 16>(0x4B400000));
    n = aie::min(aie::max(n, aie::broadcast<int32, 16>(-126)), aie::broadcast<int32, 16>(127));
    const float c[6] = { 1.339887440e-3f, 9.618437357e-3f, 5.550332471e-2f, 2.402264791e-1f,
                         6.931472028e-1f, 1.0f };
    v16f p = aie::broadcast<float, 16>(1.535336188e-4f);
    for (int i = 0; i < 6; i++) {
        p = aie::add(aie::mul(p, f).to_vector<float>(0), aie::broadcast<float, 16>(c[i]));
    }
    return aie::add(p.cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>();
}

// lane i of the inclusive prefix sum, for 16 f32 (a lane-wise loop through
// memory: the core has no scalar f32 adds)
__attribute__((noinline)) v16f prefix16(const v16f &x)
{
    alignas(64) float b[32];
    aie::store_v(b, aie::zeros<float, 16>());
    aie::store_v(b + 16, x);
    v16f s = x;
    for (int k = 1; k < 16; k <<= 1) {
        aie::store_v(b + 16, s);
        s = aie::add(s, aie::load_unaligned_v<16>(b + 16 - k));
    }
    return s;
}

void gdn_factors(float *in)
{
    const float scale = 0.08838834764831845f;          // 1 / sqrt(128)
    const v16f g   = aie::load_v<16>(in);
    const v16f bt  = aie::load_v<16>(in + 16);
    const v16f ls  = aie::broadcast<float, 16>(in[32]);
    const v16f lsn = aie::broadcast<float, 16>(in[33]);
    g_shift = ((const int32_t *) in)[34];
    // gam = cumsum(g) log2(e)
    const v16f gam = prefix16(aie::mul(g, aie::broadcast<float, 16>(1.4426950408889634f)).to_vector<float>(0));
    alignas(64) float gb[16];
    aie::store_v(gb, gam);
    const v16f gl  = aie::broadcast<float, 16>(gb[C - 1]);
    const v16f mid = aie::broadcast<float, 16>(gb[C / 2]);
    const v16f gs  = exp2_f(aie::add(gam, ls));
    // (the raw values are all in registers by now: the region is rewritten)
    aie::store_v(in, gs);
    aie::store_v(in + C, aie::mul(gs, aie::broadcast<float, 16>(scale)).to_vector<float>(0));
    aie::store_v(in + 2 * C, exp2_f(aie::sub(aie::sub(gl, gam), lsn)));
    aie::store_v(in + 3 * C, bt);
    // D[t, s] = e_t f_s = 2^(gam_t - gam_s), the centre keeping both in range
    alignas(64) float e[16], f[16], be[16], se[16];
    aie::store_v(e, exp2_f(aie::sub(gam, mid)));
    aie::store_v(f, exp2_f(aie::sub(mid, gam)));
    aie::store_v(be, aie::mul(bt, aie::load_v<16>(e)).to_vector<float>(0));
    aie::store_v(se, aie::mul(aie::load_v<16>(e), aie::broadcast<float, 16>(scale)).to_vector<float>(0));
    const v16f fv = aie::load_v<16>(f);
    const v16i idx = aie::broadcast<int32, 16>(0);
    alignas(64) int32_t ii[16];
    for (int i = 0; i < 16; i++) {
        ii[i] = i;
    }
    const v16i lane = aie::load_v<16>(ii);
    float *ld = in + 4 * C, *ad = ld + C * C;
    for (int t = 0; t < C; t++) {
        // row t: beta_t e_t f_s (s < t) and scale e_t f_s (s <= t), into 8 x 8 tiles
        const v16f rl = aie::mul(aie::broadcast<float, 16>(be[t]), fv).to_vector<float>(0);
        const v16f ra = aie::mul(aie::broadcast<float, 16>(se[t]), fv).to_vector<float>(0);
        const v16i tv = aie::add(idx, aie::broadcast<int32, 16>(t));
        const v16f ml = aie::select(aie::zeros<float, 16>(), rl, aie::lt(lane, tv));
        const v16f ma = aie::select(aie::zeros<float, 16>(), ra, aie::le(lane, tv));
        alignas(64) float bl[16], ba[16];
        aie::store_v(bl, ml);
        aie::store_v(ba, ma);
        for (int sb = 0; sb < 2; sb++) {
            aie::store_v(ld + ((t / 8) * 2 + sb) * 64 + (t % 8) * 8, aie::load_v<8>(bl + 8 * sb));
            aie::store_v(ad + ((t / 8) * 2 + sb) * 64 + (t % 8) * 8, aie::load_v<8>(ba + 8 * sb));
        }
    }
}

} // namespace

extern "C" {

// The call's header, the first object of the core's stream: word 0 the
// chunks, word 1 sigma = 2^ls after the last (f32), the new state's factor.
void gdn_hdr(const bfloat16 *h, int32_t *cnt)
{
    cnt[0] = ((const int32_t *) h)[0];
    g_sigma = ((const float *) h)[1];
}

// The state arrives as ggml keeps it, f32: this core's value columns cc,
// each a row of the 128 key rows r (S transposed), 16 of them an object. It
// is split into the bf16 hi/lo pair and turned into 8 x 8 tiles of S itself
// (r down, cc across) through the 16-bit transpose.
void gdn_state_in(const bfloat16 *in, bfloat16 *st, int32_t part)
{
    const float *src = (const float *) in;
    for (int cb = 0; cb < 2; cb++) {
        for (int rb = 0; rb < DK / 8; rb++) {
            alignas(64) float blk[64];               // [cc][r] of the block
            for (int j = 0; j < 8; j++) {
                aie::store_v(blk + 8 * j, aie::load_v<8>(src + (cb * 8 + j) * DK + rb * 8));
            }
            const v32f x0 = aie::load_v<32>(blk), x1 = aie::load_v<32>(blk + 32);
            const v32bf h0 = aie::accum<accfloat, 32>(x0).to_vector<bfloat16>(0);
            const v32bf h1 = aie::accum<accfloat, 32>(x1).to_vector<bfloat16>(0);
            const v32bf l0 = aie::accum<accfloat, 32>(aie::sub(x0, aie::accum<accfloat, 32>(h0).to_vector<float>(0))).to_vector<bfloat16>(0);
            const v32bf l1 = aie::accum<accfloat, 32>(aie::sub(x1, aie::accum<accfloat, 32>(h1).to_vector<float>(0))).to_vector<bfloat16>(0);
            const int t = (rb * (DV / 8) + part * 2 + cb) * 64;
            aie::store_v(st + t, aie::transpose(aie::concat(h0, h1), 8, 8));
            aie::store_v(st + DK * DV + t, aie::transpose(aie::concat(l0, l1), 8, 8));
        }
    }
}

// ... and leaves as ggml keeps it: sigma (hi + lo), f32, 8 of the value
// columns an object (sigma = 2^ls from the header, g_sigma).

void gdn_state_out(const bfloat16 *st, float *out, int32_t part)
{
    const aie::vector<float, 32> sg = aie::broadcast<float, 32>(g_sigma);
    for (int rb = 0; rb < DK / 8; rb++) {
        const int t = (rb * (DV / 8) + part) * 64;
        const v64bf hi = aie::transpose(aie::load_v<64>(st + t), 8, 8);            // [cc][r]
        const v64bf lo = aie::transpose(aie::load_v<64>(st + DK * DV + t), 8, 8);
        alignas(64) float blk[64];
        for (int h = 0; h < 2; h++) {
            const v32f v = aie::add(aie::accum<accfloat, 32>(hi.extract<32>(h)).to_vector<float>(0),
                                    aie::accum<accfloat, 32>(lo.extract<32>(h)).to_vector<float>(0));
            aie::store_v(blk + 32 * h, aie::mul(v, sg).to_vector<float>(0));
        }
        // column j's 8 rows: element (j, r) of the object's [8][128] rows,
        // which the MemTile reads as the 8 x 8 tiles of a (C x DV) chunk
        // output (kernels/gdn_mm.py) - so it is stored at that tile place
        for (int j = 0; j < 8; j++) {
            const int row = j * 2 + rb / 8, col = (rb % 8) * 8;
            aie::store_v(out + ((row / 8) * (DV / 8) + col / 8) * 64 + (row % 8) * 8, aie::load_v<8>(blk + 8 * j));
        }
    }
}

void gdn_chunk(bfloat16 *in, bfloat16 *st, float *out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *k = in;
    bfloat16 *q = in + C * DK;
    bfloat16 *v = in + 2 * C * DK;             // then Delta's lo half
    bfloat16 *hi = st;
    bfloat16 *lo = st + DK * DV;
    float *fx = (float *) (in + 2 * C * DK + C * DV);
    gdn_factors(fx);
    const float *fg = fx;                                           // G sigma
    const float *fq = fg + C;                                       // G sigma scale
    const float *fu = fg + 2 * C;                                   // (G_C / G) / sigma'
    const float *fb = fg + 3 * C;                                   // beta
    const float *ld = fg + 4 * C;                                   // beta_t D, s < t
    const float *ad = ld + C * C;                                   // D scale, s <= t
    const int32_t shift = g_shift;

    // L = (beta_t D) (K K^T), T = (I + L)^-1 by forward substitution. Not by
    // doubling, (I - L)(I + L^2)(I + L^4)...: when a chunk's keys are close
    // and its decay weak, L^8 is thousands and the products cancel beyond
    // what bf16 keeps (some heads went to 1e25)
    zero_f(g_cc, C * C);
    mm<false, true>(C, DK, C, k, DK / 8, k, DK / 8, g_cc, C / 8);
    mul_f(g_cc, ld, g_cc, C * C);
    // L's hi/lo in T's own buffers, T's rows in Delta's: both free until later
    inverse_unit_lower(g_cc, g_t, g_p, g_rb, g_rb + C * 32);
    // T diag(beta) as a hi/lo pair: a token's factors must be f32-exact -
    // rounded to bf16 they scale the token's whole output coherently, which
    // the next projection does not average out
    scale_cols_f(g_cc, fb, g_cc);
    split_hl(g_cc, g_t, g_p, C * C);           // T beta: hi, lo

    // A = (D scale) (Q K^T), before Q is scaled, as a hi/lo pair
    zero_f(g_cc, C * C);
    mm<false, true>(C, DK, C, q, DK / 8, k, DK / 8, g_cc, C / 8);
    mul_f(g_cc, ad, g_cc, C * C);
    split_hl(g_cc, g_l, g_ip, C * C);          // A: hi, lo

    // Delta = T diag(beta) (V - diag(G sigma) K S): K S from the inputs as
    // they came, the factor applied to its f32 result - rounding K diag(G
    // sigma) instead doubled the error where V and K S cancel (close keys)
    zero_f(g_x, C * DV);
    mm<false, false>(C, DK, DV, k, DK / 8, hi, DV / 8, g_x, DV / 8);
    mm<false, false>(C, DK, DV, k, DK / 8, lo, DV / 8, g_x, DV / 8);
    scale_rows_ff(g_x, C, DV, fg, g_x);
    for (int i = 0; i < C * DV; i += 32) {
        const v32f r = aie::sub(aie::accum<accfloat, 32>(aie::load_v<32>(v + i)).to_vector<float>(0),
                                aie::load_v<32>(g_x + i));
        aie::store_v(g_x + i, r);
    }
    to_bf16(g_x, g_rb, C * DV);                // R
    zero_f(g_x, C * DV);
    mm<false, false>(C, C, DV, g_t, C / 8, g_rb, DV / 8, g_x, DV / 8);
    mm<false, false>(C, C, DV, g_p, C / 8, g_rb, DV / 8, g_x, DV / 8);
    // Delta as a hi/lo pair (its lo where V was): rounded to bf16 it both
    // leaves each output and enters the state, where close keys amplify it
    bfloat16 *dh = g_rb, *dl = v;
    split_hl(g_x, dh, dl, C * DV);

    // O = diag(G sigma scale) (Q S) + A Delta, the factor on Q S's f32
    zero_f(out, C * DV);
    mm<false, false>(C, DK, DV, q, DK / 8, hi, DV / 8, out, DV / 8);
    mm<false, false>(C, DK, DV, q, DK / 8, lo, DV / 8, out, DV / 8);
    scale_rows_ff(out, C, DV, fq, out);
    mm<false, false>(C, C, DV, g_l, C / 8, dh, DV / 8, out, DV / 8);
    mm<false, false>(C, C, DV, g_l, C / 8, dl, DV / 8, out, DV / 8);
    mm<false, false>(C, C, DV, g_ip, C / 8, dh, DV / 8, out, DV / 8);

    // the update's Delta'' = diag((G_C / G) / sigma') Delta, from Delta's
    // f32 (still in g_x), as a hi/lo pair in the same place: K stays the
    // input as it came
    scale_rows_ff(g_x, C, DV, fu, g_x);
    split_hl(g_x, dh, dl, C * DV);

    // the state scaled down by the power of two the host says sigma gained -
    // before the update, whose factors (G_C / G) / sigma' are then bounded
    // by 2^32 however strong the chunk's decay (a gate of -5 a token takes
    // them past bf16's range otherwise)
    if (shift > 0) {
        const aie::vector<int16, 32> sh = aie::broadcast<int16, 32>((int16) (shift << 7));
        const aie::vector<int16, 32> em = aie::broadcast<int16, 32>((int16) 0x7F80);
        for (int i = 0; i < 2 * DK * DV; i += 32) {
            const aie::vector<int16, 32> b = aie::load_v<32>(st + i).cast_to<int16>();
            // exponents that would go under 1 flush to zero
            const auto keep = aie::gt(aie::bit_and(b, em), sh);
            aie::store_v(st + i, aie::select(aie::zeros<int16, 32>(), aie::sub(b, sh), keep)
                                     .cast_to<bfloat16>());
        }
    }

    // S += K^T Delta'', 16 state rows at a time into the hi/lo pair
    for (int r0 = 0; r0 < DK; r0 += 16) {
        zero_f(g_x, 16 * DV);
        mm<true, false>(16, C, DV, k + (r0 / 8) * 64, DK / 8, dh, DV / 8, g_x, DV / 8);
        mm<true, false>(16, C, DV, k + (r0 / 8) * 64, DK / 8, dl, DV / 8, g_x, DV / 8);
        bfloat16 *h = hi + (r0 / 8) * (DV / 8) * 64;
        bfloat16 *l = lo + (r0 / 8) * (DV / 8) * 64;
        for (int i = 0; i < 16 * DV; i += 32) {
            const v32f s = aie::add(aie::add(
                aie::accum<accfloat, 32>(aie::load_v<32>(h + i)).to_vector<float>(0),
                aie::accum<accfloat, 32>(aie::load_v<32>(l + i)).to_vector<float>(0)),
                aie::load_v<32>(g_x + i));
            const v32bf nh = aie::accum<accfloat, 32>(s).to_vector<bfloat16>(0);
            const v32f rem = aie::sub(s, aie::accum<accfloat, 32>(nh).to_vector<float>(0));
            aie::store_v(h + i, nh);
            aie::store_v(l + i, aie::accum<accfloat, 32>(rem).to_vector<bfloat16>(0));
        }
    }

}

} // extern "C"
