#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

// Flash attention for the six full-attention prefill layers of Qwen3.5.
// Semantics match GGML_OP_FLASH_ATTN_EXT with a plain causal mask, no sinks
// and no logit softcap (see llm_build_qwen35::build_attention_layer, which
// passes a null mask/sinks and kq_scale = 1/sqrt(head_dim)).
//
//   S[i][j] = scale * dot(Q[i], K[j])   masked to j <= pos(i)
//   P       = softmax_j(S)
//   O[i]    = sum_j P[i][j] * V[j]
//
// THE QUERIES SIT ON THE VECTOR LANES. That one choice decides the rest: both
// dot products then accumulate along the head dim with i on the lanes, so a
// tile costs no lane reductions at all - the thing that stalls an AIE pipeline
// ([[aie-scalar-roundtrip-stall]]) - and, better, K and V are read exactly as
// the KV cache already stores them, one contiguous head-dim row per key. Only
// Q has to arrive transposed, [DH][MT], and that is one small transpose per
// query block rather than one per key tile.
//
// The online softmax state (O, m, l) lives in the output object, so a core
// holds one Q block (8 KB), one K/V tile (8 KB) and one accumulator (16.5 KB)
// and never spills.
//
// Compile flags: FA_D (head dim), FA_MT (queries per block = vector lanes),
// FA_JT (keys per tile), FA_ROWS (cores per column), FA_SCALE.

#ifndef FA_D
#define FA_D 256
#endif
#ifndef FA_MT
#define FA_MT 16
#endif
#ifndef FA_JT
#define FA_JT 8
#endif
#ifndef FA_ROWS
#define FA_ROWS 4
#endif
#ifndef FA_SCALE
#define FA_SCALE 0.0625f
#endif
#ifndef FA_QHDR
#define FA_QHDR 16
#endif

// m starts here rather than at -inf so that a tile a row cannot see at all
// leaves it alone: the masked bias below is far more negative still, so the
// running max does not move and the tile contributes exp(-huge) = 0. With
// m = -inf the two would be equal and every masked key would weigh 1.
#define FA_M0   (-1.0e30f)
#define FA_MASK (-1.0e34f)

using vf   = ::aie::vector<float, FA_MT>;
using vbf  = ::aie::vector<bfloat16, FA_MT>;
using accf = ::aie::accum<accfloat, FA_MT>;

// Accumulator object: [O transposed, FA_D x FA_MT][m FA_MT][l FA_MT].
#define FA_O(p)  (p)
#define FA_MV(p) ((p) + FA_D * FA_MT)
#define FA_LV(p) ((p) + FA_D * FA_MT + FA_MT)

// exp(x) for x <= 0, the same range-reduction-and-square form the decode GEMV
// uses for its sigmoid (gemv_q4.cc): the AIE lookup tables are bf16 only and
// carry ~4e-3, which softmax would turn into visible drift over 2048 keys.
// At t = x/256 a degree-5 series is good to ~2e-8 and eight squarings bring
// the range back. Everything is multiplies and adds - no exponent-field
// arithmetic, no int/float round trip.
static inline vf fa_exp(const vf & x) {
    const vf one = ::aie::broadcast<float, FA_MT>(1.0f);
    const vf xc  = ::aie::max(x, ::aie::broadcast<float, FA_MT>(-40.0f));
    const vf t   = ::aie::mul(xc, ::aie::broadcast<float, FA_MT>(1.0f / 256.0f))
                       .template to_vector<float>(0);
    vf e = ::aie::broadcast<float, FA_MT>(1.0f / 5.0f);
    e = ::aie::add(::aie::mul(e, t).template to_vector<float>(0),
                   ::aie::broadcast<float, FA_MT>(1.0f / 4.0f));
    e = ::aie::add(::aie::mul(e, t).template to_vector<float>(0),
                   ::aie::broadcast<float, FA_MT>(1.0f / 3.0f));
    e = ::aie::add(::aie::mul(e, t).template to_vector<float>(0),
                   ::aie::broadcast<float, FA_MT>(1.0f / 2.0f));
    e = ::aie::add(::aie::mul(e, t).template to_vector<float>(0), one);
    e = ::aie::add(::aie::mul(e, t).template to_vector<float>(0), one);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    e = ::aie::mul(e, e).template to_vector<float>(0);
    return e;
}

// 1/a for a > 0: Newton from the bit-pattern seed, as in gemv_q4.cc. The AIE
// reciprocal table is bf16 and AIE2-only.
static inline vf fa_rcp(const vf & a) {
    const vf two = ::aie::broadcast<float, FA_MT>(2.0f);
    vf r = ::aie::sub(::aie::broadcast<int32, FA_MT>(0x7EF311C3),
                      a.template cast_to<int32>()).template cast_to<float>();
    r = ::aie::mul(r, ::aie::sub(two, ::aie::mul(a, r).template to_vector<float>(0)))
            .template to_vector<float>(0);
    r = ::aie::mul(r, ::aie::sub(two, ::aie::mul(a, r).template to_vector<float>(0)))
            .template to_vector<float>(0);
    r = ::aie::mul(r, ::aie::sub(two, ::aie::mul(a, r).template to_vector<float>(0)))
            .template to_vector<float>(0);
    return r;
}

// Lane index as a float, for the causal compare. Built once per call from a
// constant table: a vector iota is not in the API for every element type.
static const float fa_lane_tbl[64] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
};

extern "C" void ggml_xdna_fa_zero(float * acc) {
    const vf z = ::aie::zeros<float, FA_MT>();
    for (int i = 0; i < FA_D; i++) {
        ::aie::store_v(FA_O(acc) + i * FA_MT, z);
    }
    ::aie::store_v(FA_MV(acc), ::aie::broadcast<float, FA_MT>(FA_M0));
    ::aie::store_v(FA_LV(acc), z);
}

// One K/V tile against one query block. The key range of a batch is walked in
// chunks of FA_NJ tiles, one dispatch each, with the accumulator left in place
// between them.
//
// `qpar` carries the chunk's parameters ahead of the query block: a core has
// only two input DMA channels, so they ride with Q rather than on a stream of
// their own, and the header is int32 because the loop trip counts in it are
// read by the design, not by this kernel. FA_QHDR int32 of header keeps the
// query block 64-byte aligned.
//
//   [0] zero the accumulator (first chunk)   [2] first tile of this chunk
//   [1] normalise it (last chunk)            [3] keys cached before the batch
extern "C" void ggml_xdna_fa_step(float * acc, int32_t * qpar, bfloat16 * kv,
                                  int32_t jt, int32_t mb, int32_t row) {
    const bfloat16 * qt = (const bfloat16 *) (qpar + FA_QHDR);
    const int32_t jtb   = qpar[2];
    const int32_t npast = qpar[3];
    const bfloat16 * K = kv;
    const bfloat16 * V = kv + FA_JT * FA_D;
    float *          O = FA_O(acc);

    const vf lane  = ::aie::load_v<FA_MT>(fa_lane_tbl);
    const int j0 = ((int) jtb + (int) jt) * FA_JT;
    // Query i of this block sits at cache position qpos0 + i, so it may see
    // keys 0 .. qpos0 + i.
    const int qpos0 = (int) npast + ((int) mb * FA_ROWS + (int) row) * FA_MT;

    const vf m_old = ::aie::load_v<FA_MT>(FA_MV(acc));
    const vf l_old = ::aie::load_v<FA_MT>(FA_LV(acc));

    // One int-to-float conversion for the whole tile: __floatsisf is a call,
    // and inside the j loop it would be eight of them.
    float thr = (float) (j0 - qpos0);

    vf S[FA_JT];
    vf m_new = m_old;
    for (int j = 0; j < FA_JT; j++) {
        const bfloat16 * kr = K + j * FA_D;
        accf a = ::aie::zeros<accfloat, FA_MT>();
#pragma clang loop unroll_count(8)
        for (int d = 0; d < FA_D; d++) {
            a = ::aie::mac(a, ::aie::load_v<FA_MT>(qt + d * FA_MT),
                           ::aie::broadcast<bfloat16, FA_MT>(kr[d]));
        }
        vf s = ::aie::mul(a.template to_vector<float>(),
                          ::aie::broadcast<float, FA_MT>(FA_SCALE))
                   .template to_vector<float>(0);
        // Causal mask without a select: lane i keeps key j0+j when
        // i >= j0 + j - qpos0, so min(lane - threshold, 0) is zero where the
        // key is visible and negative where it is not.
        const vf d0 = ::aie::sub(lane, ::aie::broadcast<float, FA_MT>(thr));
        const vf bias = ::aie::mul(::aie::min(d0, ::aie::zeros<float, FA_MT>()),
                                   ::aie::broadcast<float, FA_MT>(-FA_MASK))
                            .template to_vector<float>(0);
        s     = ::aie::add(s, bias);
        S[j]  = s;
        m_new = ::aie::max(m_new, s);
        thr  += 1.0f;
    }

    const vf corr = fa_exp(::aie::sub(m_old, m_new));
    vf l_new = ::aie::mul(l_old, corr).template to_vector<float>(0);
    vbf P[FA_JT];
    for (int j = 0; j < FA_JT; j++) {
        const vf p = fa_exp(::aie::sub(S[j], m_new));
        l_new = ::aie::add(l_new, p);
        accf pa;
        pa.from_vector(p);
        P[j] = pa.template to_vector<bfloat16>();
    }

    // O <- O * corr + sum_j P[j] * V[j], one head-dim row at a time with the
    // queries still on the lanes. P is bf16 so this is the same one-cycle
    // bf16 MAC the QK loop above uses.
    for (int d = 0; d < FA_D; d++) {
        accf a;
        a.from_vector(::aie::mul(::aie::load_v<FA_MT>(O + d * FA_MT), corr)
                          .template to_vector<float>(0));
#pragma clang loop unroll_count(8)
        for (int j = 0; j < FA_JT; j++) {
            a = ::aie::mac(a, P[j],
                           ::aie::broadcast<bfloat16, FA_MT>(V[j * FA_D + d]));
        }
        ::aie::store_v(O + d * FA_MT, a.template to_vector<float>());
    }

    ::aie::store_v(FA_MV(acc), m_new);
    ::aie::store_v(FA_LV(acc), l_new);
}

// Divide the accumulated O by l, in place. A row with no visible key cannot
// happen in a causal batch - every query sees at least itself - so l > 0.
extern "C" void ggml_xdna_fa_norm(float * acc) {
    const vf inv = fa_rcp(::aie::load_v<FA_MT>(FA_LV(acc)));
    float * O = FA_O(acc);
    for (int d = 0; d < FA_D; d++) {
        ::aie::store_v(O + d * FA_MT,
                       ::aie::mul(::aie::load_v<FA_MT>(O + d * FA_MT), inv)
                           .template to_vector<float>(0));
    }
}
