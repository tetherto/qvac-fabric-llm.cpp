// The gated-delta-net v-stage of the fused recurrent core: consumes the norm
// stage's per-chunk pkv object and updates the persistent state. Read by
// ggml_xdna_gdn_v.py (and attn_gdn_gated.py through it) and compiled by IRON as the
// "ggml_xdna_gdn_v" kernel.
//
// @S_V@, @CHUNK@, @O_V@ and @O_EG@ are filled in per build; see ggml_xdna_gdn_v.py.
#include <aie_api/aie.hpp>
using namespace aie;
extern "C" void ggml_xdna_gdn_v(const float * pkv, const bfloat16 * rows,
                      bfloat16 * sout, float * attn) {
    const float * v16 = pkv + @O_V@;
    const float eg = pkv[@O_EG@];
    const float b = pkv[@O_EG@+1];
    const float scale = pkv[@O_EG@+2];
    // fp32 kn/qn -> bf16 once per chunk (pkv object is the norm-stage output)
    alignas(64) bfloat16 kb[@S_V@];
    alignas(64) bfloat16 qb[@S_V@];
    for (int o = 0; o < @S_V@; o += 16) {
        aie::accum<accfloat, 16> ak;
        ak.from_vector(aie::load_v<16>(pkv + o), 0);
        aie::store_v(kb + o, ak.to_vector<bfloat16>());
        aie::accum<accfloat, 16> aq;
        aq.from_vector(aie::load_v<16>(pkv + @S_V@ + o), 0);
        aie::store_v(qb + o, aq.to_vector<bfloat16>());
    }
    const auto reg_eg = aie::broadcast<bfloat16, 32>(eg);
    // One horizontal reduction per dot product rather than one per block: the
    // blocks accumulate into the same 32 lanes first, which sums to the same
    // 128 products. The reductions were what this kernel spent its time on -
    // 132 of them per chunk object where 33 do.
    aie::accum<accfloat, 32> akq = aie::mul(aie::load_v<32>(kb),
                                            aie::load_v<32>(qb));
    for (int blk = 1; blk < @S_V@/32; ++blk) {
        akq = aie::mac(akq, aie::load_v<32>(kb + blk*32),
                       aie::load_v<32>(qb + blk*32));
    }
    const float kq = aie::reduce_add<float>(akq);
    // The k and q vectors are the same for every row, so they are loaded once
    // for the whole chunk rather than once per row. The blocks are written out
    // because there are only @S_V@/32 of them - a loop that short does not
    // pipeline, and it was reloading the same two vectors each time.
    const auto k0 = aie::load_v<32>(kb);
    const auto k1 = aie::load_v<32>(kb + 32);
    const auto k2 = aie::load_v<32>(kb + 64);
    const auto k3 = aie::load_v<32>(kb + 96);
    const auto q0 = aie::load_v<32>(qb);
    const auto q1 = aie::load_v<32>(qb + 32);
    const auto q2 = aie::load_v<32>(qb + 64);
    const auto q3 = aie::load_v<32>(qb + 96);
    // Two passes over the rows, not one. In one pass every row is a single
    // dependency chain - four macs, a horizontal reduction, scalar float math,
    // a broadcast, then the update that needs it - and the two accumulator to
    // scalar transfers in the middle of it are what the stage spends its time
    // on: the loop cannot be pipelined across rows because the update of row j
    // waits on a scalar that row j has only just produced.
    //
    // Split, the first pass only stores its reductions and the second only
    // reads them back, so both pipeline; and with every dot product of the
    // chunk in hand, dj and attn are sixteen lanes of one vector instead of
    // sixteen scalar computations.
    alignas(64) float dkv[@CHUNK@];
    alignas(64) float dqv[@CHUNK@];
    alignas(64) float djv[@CHUNK@];
    for (int j = 0; j < @CHUNK@; ++j)
        chess_prepare_for_pipelining chess_loop_range(@CHUNK@, ) {
        const bfloat16 * row = rows + j*@S_V@;
        const auto r0 = aie::load_v<32>(row);
        const auto r1 = aie::load_v<32>(row + 32);
        const auto r2 = aie::load_v<32>(row + 64);
        const auto r3 = aie::load_v<32>(row + 96);
        // Two chains per dot product rather than one: four macs deep is four
        // multiply latencies, and the halves cost one vector add to rejoin.
        aie::accum<accfloat, 32> ka = aie::mul(r0, k0);
        aie::accum<accfloat, 32> kb = aie::mul(r1, k1);
        ka = aie::mac(ka, r2, k2);
        kb = aie::mac(kb, r3, k3);
        aie::accum<accfloat, 32> qa = aie::mul(r0, q0);
        aie::accum<accfloat, 32> qb = aie::mul(r1, q1);
        qa = aie::mac(qa, r2, q2);
        qb = aie::mac(qb, r3, q3);
        dkv[j] = aie::reduce_add(aie::add(ka.to_vector<float>(0),
                                          kb.to_vector<float>(0)));
        dqv[j] = aie::reduce_add(aie::add(qa.to_vector<float>(0),
                                          qb.to_vector<float>(0)));
    }
    {
        // dj = (v - eg*dotk) * b and attn = scale * (eg*dotq + dj*kq), for
        // every row of the chunk at once.
        const auto egv = aie::broadcast<float, @CHUNK@>(eg);
        const auto dk  = aie::load_v<@CHUNK@>(dkv);
        const auto dq  = aie::load_v<@CHUNK@>(dqv);
        const auto vv  = aie::load_unaligned_v<@CHUNK@>(v16);
        const auto dj  = aie::mul(aie::sub(vv, aie::mul(egv, dk).to_vector<float>(0)),
                                  aie::broadcast<float, @CHUNK@>(b)).to_vector<float>(0);
        aie::store_v(djv, dj);
        auto at = aie::add(aie::mul(egv, dq).to_vector<float>(0),
                           aie::mul(dj, aie::broadcast<float, @CHUNK@>(kq)).to_vector<float>(0));
        aie::store_unaligned_v(attn,
                               aie::mul(at, aie::broadcast<float, @CHUNK@>(scale)).to_vector<float>(0));
    }
    for (int j = 0; j < @CHUNK@; ++j)
        chess_prepare_for_pipelining chess_loop_range(@CHUNK@, ) {
        const bfloat16 * row = rows + j*@S_V@;
        bfloat16 * orow = sout + j*@S_V@;
        const auto reg_dj = aie::broadcast<bfloat16, 32>((bfloat16) djv[j]);
        auto a0 = aie::mul(aie::load_v<32>(row), reg_eg);
        a0 = aie::mac(a0, reg_dj, k0);
        aie::store_v(orow, a0.to_vector<bfloat16>());
        auto a1 = aie::mul(aie::load_v<32>(row + 32), reg_eg);
        a1 = aie::mac(a1, reg_dj, k1);
        aie::store_v(orow + 32, a1.to_vector<bfloat16>());
        auto a2 = aie::mul(aie::load_v<32>(row + 64), reg_eg);
        a2 = aie::mac(a2, reg_dj, k2);
        aie::store_v(orow + 64, a2.to_vector<bfloat16>());
        auto a3 = aie::mul(aie::load_v<32>(row + 96), reg_eg);
        a3 = aie::mac(a3, reg_dj, k3);
        aie::store_v(orow + 96, a3.to_vector<bfloat16>());
    }
}
