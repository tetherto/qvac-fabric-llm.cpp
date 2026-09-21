// The norm stage of the fused recurrent core: per head read
// [x_head 384 | eg b scale], L2-normalize q/k and emit the head's pkv chunks
// [kn|qn|v16|eg|b|scale]. Read by attn_cn.py (and attn_gdn_gated.py through it)
// and compiled by IRON as the "ggml_xdna_attn_norm" kernel, or as one of the single-chunk
// variants "normj<g>_<half>".
//
// @NAME@, @HALF@, @ONE@, @N_OBJ@ and @PKV_N@ are filled in per build; see
// attn_cn.py.
#include <aie_api/aie.hpp>
using namespace aie;
// Internal linkage: a core links several variants of this kernel when the
// norm cores emit single chunks, and an external definition collides.
static inline float rsqrtf_scalar(float x) {
    // no libm on AIE: Quake rsqrt + 2 Newton iterations (~1e-7 rel)
    union { float f; unsigned u; } y;
    y.f = x;
    y.u = 0x5f3759dfu - (y.u >> 1);
    float r = y.f;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}
extern "C" void @NAME@(const float * in, float * out) {
    // in = [ q(128) | k(128) | v(128) | eg | b | scale ]; out = the head's pkv
    // chunks, all eight or one half of them
    const float * q = in;
    const float * k = in + @S_V@;
    const float * v = in + 2*@S_V@;
    const float eg = in[3*@S_V@];
    const float b = in[3*@S_V@+1];
    const float scale = in[3*@S_V@+2];
    // fp32 q/k -> bf16 once (the 387-float pkv chunk stride is not 64B
    // aligned, so the per-chunk qn/kn copies below stay scalar)
    alignas(64) bfloat16 qb[@S_V@];
    alignas(64) bfloat16 kb[@S_V@];
    alignas(64) float qn[@S_V@];
    alignas(64) float kn[@S_V@];
    for (int o = 0; o < @S_V@; o += 16) {
        aie::accum<accfloat, 16> aq;
        aq.from_vector(aie::load_v<16>(q + o), 0);
        aie::store_v(qb + o, aq.to_vector<bfloat16>());
        aie::accum<accfloat, 16> ak;
        ak.from_vector(aie::load_v<16>(k + o), 0);
        aie::store_v(kb + o, ak.to_vector<bfloat16>());
    }
    float sq = 0.0f, sk = 0.0f;
    for (int blk = 0; blk < @S_V@/32; ++blk) {
        auto qv = aie::load_v<32>(qb + blk*32);
        auto kv = aie::load_v<32>(kb + blk*32);
        sq += aie::reduce_add<float>(aie::mul(qv, qv));
        sk += aie::reduce_add<float>(aie::mul(kv, kv));
    }
    const float iq = rsqrtf_scalar(sq);
    const float ik = rsqrtf_scalar(sk);
    const auto reg_iq = aie::broadcast<bfloat16, 32>(iq);
    const auto reg_ik = aie::broadcast<bfloat16, 32>(ik);
    for (int blk = 0; blk < @S_V@/32; ++blk) {
        aie::accum<accfloat, 32> aq;
        aq = aie::mul(aie::load_v<32>(qb + blk*32), reg_iq);
        aie::store_v(qn + blk*32, aq.to_vector<float>());
        aie::accum<accfloat, 32> ak;
        ak = aie::mul(aie::load_v<32>(kb + blk*32), reg_ik);
        aie::store_v(kn + blk*32, ak.to_vector<float>());
    }
    // With a half given the kernel writes four of the head's eight chunks -
    // one gdn round - so the stage can hand them straight to the gdn cores
    // through a MemTile instead of a round trip through DDR. The per-head
    // normalisation above is repeated for the second half, which is 128 values
    // against the round trip it replaces.
    const int j0 = @ONE@ >= 0 ? @ONE@ : (@HALF@ < 0 ? 0 : @HALF@ * (@N_OBJ@ / 2));
    const int jn = @ONE@ >= 0 ? 1 : (@HALF@ < 0 ? @N_OBJ@ : @N_OBJ@ / 2);
    for (int jj = 0; jj < jn; ++jj) {
        const int j = j0 + jj;
        float * o = out + jj * @PKV_N@;
        // [ kn(128) | qn(128) | v16(16) | eg | b | scale ]
        for (int i = 0; i < @S_V@; ++i) { o[i] = kn[i]; o[@S_V@+i] = qn[i]; }
        for (int i = 0; i < @CHUNK@; ++i) { o[2*@S_V@+i] = v[j*@CHUNK@+i]; }
        o[3*@S_V@] = eg;
        o[3*@S_V@+1] = b;
        o[3*@S_V@+2] = scale;
    }
}
