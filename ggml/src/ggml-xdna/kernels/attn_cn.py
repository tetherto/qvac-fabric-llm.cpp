#!/usr/bin/env python3
# attn_cn.py -*- Python -*-
#
# attn_cg.py with the gdn stage removed (conv + norm only, 4 shim columns):
# the norm stage still emits the per-(head,chunk) pkv objects [kn|qn|v16|eg|
# b|scale] (387 floats each) into PKVB, which a separate bf16-vector gdn kernel
# (gdn_v.py / gdn.xclbin) consumes as its input, reading/writing the persistent
# ssm state on the device. The conv feed / x tails / pkvb geometry is identical
# to attn_cg.py so the fused llama hook packs the same BOs.
#
# Usage: python attn_cn.py -d npu2 --workdir build/bin
#
# S2b: gated-delta-net input-mixing conv+norm stage in ONE xclbin / ONE run
# (conv + norm on their own shim columns so no column exceeds 2 S2MM/2 MM2S).
#
#   conv (cols 0-1, row2) : silu(conv1d) of the q/k/v channel slices a head
#                           needs -> x_bo grouped [head][q|k|v] (384/head),
#                           conv history written back to the feed buffer.
#   norm (cols 2-3, row3) : per head read [x_head 384 | eg b scale],
#                           L2-normalize q/k, emit 8 chunk objects
#                           [kn|qn|v16|eg|b|scale] (387 each) -> pkvb_bo.
#
# Host writes, per token: qkv into each conv feed slot, [eg,b,scale] into each
# head's x_bo tail.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

CH = 6144
S_V = 128
N_VH = 16
CHUNK = 16
N_OBJ = S_V // CHUNK          # 8

# conv stage
NC_CONV = 2                   # columns for conv (CH/NC_CONV channels each)
CN = CH // (NC_CONV * S_V)    # 24 group objects (128 channels each) / col
F_H = 0
F_Q = 3 * S_V
F_W = 4 * S_V
FEED_N = 8 * S_V              # 1024

# norm stage: 2 columns, 8 heads per column
NC_NORM = 2
HP_NORM = N_VH // NC_NORM     # 8 heads per norm col
HEAD_NORM = 3 * S_V + 3       # x-head(384) + eg,b,scale(3) = 387
PKV_N = 3 * S_V + 3           # kn(128)|qn(128)|v16(16)|eg|b|scale = 387
PKVB_N = N_OBJ * PKV_N        # 3096 per head
K_GATE = N_VH * S_V           # 2048 (gdn attn row dim)


def _conv_src(gpo: int = 1, feed_slot: int = 0):
    slot = feed_slot or FEED_N
    return f"""
#include <aie_api/aie.hpp>
using namespace aie;
// gather fp32 src[base + i*stride] into an aligned bf16 buffer; the bf16
// conversion is the silu/gdn accum trick verbatim
static inline void f32gb(const float * src, int base, int stride, bfloat16 * dst) {{
    alignas(64) float st[32];
    for (int i = 0; i < 32; ++i) {{
        st[i] = src[base + i * stride];
    }}
    for (int o = 0; o < 32; o += 16) {{
        aie::accum<accfloat, 16> ga;
        ga.from_vector(aie::load_v<16>(st + o), 0);
        aie::store_v(dst + o, ga.to_vector<bfloat16>());
    }}
}}
// What this stage costs is not its arithmetic - emptying this kernel leaves
// its time unchanged (240 us against 249 with STAGE_STUB=1, and the object
// names carry their flags now, so the stub really is compiled in) - and not
// its bytes either: 144 KB a conv column is 33 us at the 4.3 GB/s a shim tile
// measures, against the ~200 us the stage takes. Seven explanations have been
// tried and are not it:
//
//   - fewer fifo objects: several channel groups per object is right but only
//     just - 229 / 216 / 204 us at 1 / 2 / 4 groups (CONV_GPO), where four
//     times the object should have been four times the rate if the transfer
//     count were the cost;
//   - fewer descriptors: one feed descriptor per column instead of 24 changes
//     nothing (schedule 6), and so does one strided descriptor per direction
//     per column, 72 -> 4, which is every transfer of the stage (schedule 7:
//     247 us against 245);
//   - fewer token waits: a descriptor bank keeping five slots in flight
//     changes nothing (schedule 5);
//   - more objects in flight: a fifo depth of 4 through shim, MemTile and
//     core changes nothing (CONV_DEPTH=4: 235 us against 234), with or
//     without the descriptor bank;
//   - the MemTile hop: dropping it is worth 6% at 4 KB objects (245 -> 230,
//     CONV_DIRECT) and nothing at all at 16 KB (207 against 202);
//   - the shim tile's port: the feed streams in and x and the history write
//     back on the same column, so all three share one port - which is exactly
//     what made the gdn block's second state stream pointless. Sending the
//     drains to columns the GEMV's joined outputs freed, idle while conv runs,
//     changes nothing (CONV_SPREAD=1: 202 us against 204);
//   - the bytes, directly: half of every feed slot is the conv weights, which
//     do not change between tokens. Streaming only the half that does
//     (CONV_SLOT=512, which leaves the kernel multiplying by garbage and is a
//     diagnostic, not a mode) changes nothing at all - 201 us against 198, and
//     the fused core 408 against 407. So there is no point converting the feed
//     to bf16 either, which the kernel's own arithmetic would have allowed;
//   - the host: the per-token feed upload is 48 windowed syncs a layer and
//     measures 6 us (GGML_XDNA_REC_TIME), against 383 us in the kernel.
//
// What the sweep does say is where the stage's cost lives. Against the object
// count it is linear and shallow - 229 / 216 / 204 us at 24 / 12 / 6 objects a
// column, about 1.4 us an object - and the fused core follows it, 456 -> 407
// us over the same range. The rest is a constant of about 200 us in the phase
// profiler, of which ~84 is what running any phase as its own dispatch costs
// (the four phases sum to 659 against the fused core's 407).
//
// So the stage is worth about 15 us a layer of object cost at four groups to
// an object, and the ~200 the phase profiler shows for it is mostly not the
// stage. Read the phase split as shares, not as absolute times.
extern "C" void convh(const float * feed0, float * x0, float * hist0) {{
#if defined(STAGE_STUB) && (STAGE_STUB & 1)
    (void)feed0; (void)x0; (void)hist0;
    return;
#endif
    // An object carries {gpo} feed groups, not one: a 4 KB transfer never
    // reaches the shim's rate, and the stage's time is all transfer.
    for (int gpo_ = 0; gpo_ < {gpo}; ++gpo_) {{
    const float * feed = feed0 + gpo_ * {slot};
    float * x = x0 + gpo_ * {S_V};
    float * hist = hist0 + gpo_ * 3 * {S_V};
    const float * hi = feed + {F_H};
    const float * qv = feed + {F_Q};
    // With a short slot the weights are not in the object at all - this is the
    // diagnostic that prices the stage's bytes, and its output is wrong.
    const float * w = {slot} < {FEED_N} ? feed : feed + {F_W};
    alignas(64) bfloat16 gb[32];
    alignas(64) bfloat16 hb[3][32];
    alignas(64) bfloat16 wb[4][32];
    alignas(64) bfloat16 qv_b[32];
    const auto reg_half16 = aie::broadcast<bfloat16, 16>(0.5f);
    const auto reg_half32 = aie::broadcast<bfloat16, 32>(0.5f);
    const auto reg_one32 = aie::broadcast<bfloat16, 32>(1.0f);
    // conv dot per 32 channels: gathered bf16 taps x weights -> fp32 accum
    for (int c = 0; c < {S_V}; c += 32) {{
        for (int t = 0; t < 3; ++t) {{
            f32gb(hi, c*3 + t, 3, hb[t]);
        }}
        for (int t = 0; t < 4; ++t) {{
            f32gb(w, c*4 + t, 4, wb[t]);
        }}
        f32gb(qv, c, 1, qv_b);
        aie::accum<accfloat, 32> acc;
        acc = aie::mul(aie::load_v<32>(hb[0]), aie::load_v<32>(wb[0]));
        acc = aie::mac(acc, aie::load_v<32>(hb[1]), aie::load_v<32>(wb[1]));
        acc = aie::mac(acc, aie::load_v<32>(hb[2]), aie::load_v<32>(wb[2]));
        acc = aie::mac(acc, aie::load_v<32>(qv_b), aie::load_v<32>(wb[3]));
        // hold raw accs in the x output buffer, then overwrite with silu below
        aie::store_v(x + c, acc.to_vector<float>());
    }}
    // history shift (3 taps per channel) stays scalar; hist is a separate BO
    for (int c = 0; c < {S_V}; ++c) {{
        hist[c*3+0] = hi[c*3+1];
        hist[c*3+1] = hi[c*3+2];
        hist[c*3+2] = qv[c];
    }}
    // silu(a) = a*0.5*(1+tanh(a/2)) in bf16 (swiglu_mm.cc silu epilogue verbatim)
    for (int o = 0; o < {S_V}; o += 32) {{
        for (int j = 0; j < 2; j++) {{
            aie::accum<accfloat, 16> ga;
            ga.from_vector(aie::load_v<16>(x + o + j * 16), 0);
            aie::store_v(gb + j * 16, ga.to_vector<bfloat16>());
        }}
        aie::vector<bfloat16, 32> input = aie::load_v<32>(gb);
        auto half_lo = aie::mul(input.extract<16>(0), reg_half16);
        auto half_hi = aie::mul(input.extract<16>(1), reg_half16);
        auto tanh_lo = aie::tanh<bfloat16>(half_lo.to_vector<float>());
        auto tanh_hi = aie::tanh<bfloat16>(half_hi.to_vector<float>());
        aie::vector<bfloat16, 32> tanh_half_x = aie::concat(tanh_lo, tanh_hi);
        aie::vector<bfloat16, 32> sig =
            aie::mul(aie::add(tanh_half_x, reg_one32), reg_half32).to_vector<bfloat16>();
        auto silu = aie::mul(input, sig).to_vector<bfloat16>();
        aie::accum<accfloat, 32> wa;
        wa.from_vector(silu, 0);
        aie::store_v(x + o, wa.to_vector<float>());
    }}
    }}
}}
"""


def _norm_src(name: str = "normh", half: int = -1, one: int = -1):
    """`half` emits four of the head's eight chunks, `one` a single chunk by
    index - which is what lets a norm core sit above a gdn core and hand it
    exactly the chunks it takes, with no MemTile and no DDR in between."""
    return f"""
#include <aie_api/aie.hpp>
using namespace aie;
// Internal linkage: a core links several variants of this kernel when the
// norm cores emit single chunks, and an external definition collides.
static inline float rsqrtf_scalar(float x) {{
    // no libm on AIE: Quake rsqrt + 2 Newton iterations (~1e-7 rel)
    union {{ float f; unsigned u; }} y;
    y.f = x;
    y.u = 0x5f3759dfu - (y.u >> 1);
    float r = y.f;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}}
extern "C" void {name}(const float * in, float * out) {{
#if defined(STAGE_STUB) && (STAGE_STUB & 2)
    (void)in; (void)out;
    return;
#endif
    // in = [ q(128) | k(128) | v(128) | eg | b | scale ]; out = the head's pkv
    // chunks, all eight or one half of them
    const float * q = in;
    const float * k = in + {S_V};
    const float * v = in + 2*{S_V};
    const float eg = in[3*{S_V}];
    const float b = in[3*{S_V}+1];
    const float scale = in[3*{S_V}+2];
    // fp32 q/k -> bf16 once (the 387-float pkv chunk stride is not 64B
    // aligned, so the per-chunk qn/kn copies below stay scalar)
    alignas(64) bfloat16 qb[{S_V}];
    alignas(64) bfloat16 kb[{S_V}];
    alignas(64) float qn[{S_V}];
    alignas(64) float kn[{S_V}];
    for (int o = 0; o < {S_V}; o += 16) {{
        aie::accum<accfloat, 16> aq;
        aq.from_vector(aie::load_v<16>(q + o), 0);
        aie::store_v(qb + o, aq.to_vector<bfloat16>());
        aie::accum<accfloat, 16> ak;
        ak.from_vector(aie::load_v<16>(k + o), 0);
        aie::store_v(kb + o, ak.to_vector<bfloat16>());
    }}
    float sq = 0.0f, sk = 0.0f;
    for (int blk = 0; blk < {S_V}/32; ++blk) {{
        auto qv = aie::load_v<32>(qb + blk*32);
        auto kv = aie::load_v<32>(kb + blk*32);
        sq += aie::reduce_add<float>(aie::mul(qv, qv));
        sk += aie::reduce_add<float>(aie::mul(kv, kv));
    }}
    const float iq = rsqrtf_scalar(sq);
    const float ik = rsqrtf_scalar(sk);
    const auto reg_iq = aie::broadcast<bfloat16, 32>(iq);
    const auto reg_ik = aie::broadcast<bfloat16, 32>(ik);
    for (int blk = 0; blk < {S_V}/32; ++blk) {{
        aie::accum<accfloat, 32> aq;
        aq = aie::mul(aie::load_v<32>(qb + blk*32), reg_iq);
        aie::store_v(qn + blk*32, aq.to_vector<float>());
        aie::accum<accfloat, 32> ak;
        ak = aie::mul(aie::load_v<32>(kb + blk*32), reg_ik);
        aie::store_v(kn + blk*32, ak.to_vector<float>());
    }}
    // With a half given the kernel writes four of the head's eight chunks -
    // one gdn round - so the stage can hand them straight to the gdn cores
    // through a MemTile instead of a round trip through DDR. The per-head
    // normalisation above is repeated for the second half, which is 128 values
    // against the round trip it replaces.
    const int j0 = {one} >= 0 ? {one} : ({half} < 0 ? 0 : {half} * ({N_OBJ} / 2));
    const int jn = {one} >= 0 ? 1 : ({half} < 0 ? {N_OBJ} : {N_OBJ} / 2);
    for (int jj = 0; jj < jn; ++jj) {{
        const int j = j0 + jj;
        float * o = out + jj * {PKV_N};
        // [ kn(128) | qn(128) | v16(16) | eg | b | scale ]
        for (int i = 0; i < {S_V}; ++i) {{ o[i] = kn[i]; o[{S_V}+i] = qn[i]; }}
        for (int i = 0; i < {CHUNK}; ++i) {{ o[2*{S_V}+i] = v[j*{CHUNK}+i]; }}
        o[3*{S_V}] = eg;
        o[3*{S_V}+1] = b;
        o[3*{S_V}+2] = scale;
    }}
}}
"""


@iron.jit
def attn_cn(*, dev_name: CompileTime[str] = "npu2"):
    FEED_T = np.ndarray[(FEED_N,), np.dtype[np.float32]]
    X_T = np.ndarray[(S_V,), np.dtype[np.float32]]
    HIST_T = np.ndarray[(3 * S_V,), np.dtype[np.float32]]
    HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
    PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]

    conv_k = iron.ExternalFunction(name="convh", source_string=_conv_src(),
                                   arg_types=[FEED_T, X_T, HIST_T],
                                   compile_flags=["-O2", "-DNDEBUG"], inline=True)
    norm_k = iron.ExternalFunction(name="normh", source_string=_norm_src(),
                                   arg_types=[HN_T, PKVB_T],
                                   compile_flags=["-O2", "-DNDEBUG"], inline=True)
    FEED_g = np.ndarray[(NC_CONV * CN * FEED_N,), np.dtype[np.float32]]
    X_g = np.ndarray[(N_VH * HEAD_NORM,), np.dtype[np.float32]]
    PKVB_g = np.ndarray[(N_VH * PKVB_N,), np.dtype[np.float32]]

    workers = []
    rt_args = [FEED_g, X_g, PKVB_g]

    # ---- conv: cols 0..NC_CONV-1, row 2 ----
    for col in range(NC_CONV):
        f3 = ObjectFifo(FEED_T, name=f"f3_{col}", depth=2)
        f2 = f3.cons().forward(obj_type=FEED_T, name=f"f2_{col}", tile=Tile(col, 1))
        x23 = ObjectFifo(X_T, name=f"x23_{col}", depth=2)
        x12 = x23.prod().join([0], obj_types=[X_T], names=[f"x12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        h23 = ObjectFifo(HIST_T, name=f"h23_{col}", depth=2)
        h12 = h23.prod().join([0], obj_types=[HIST_T], names=[f"h12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def conv_fn(fc, xp, hc, k, n=CN):
            for _ in range_(n):
                f = fc.acquire(1)
                x = xp.acquire(1)
                ho = hc.acquire(1)
                k(f, x, ho)
                xp.release(1)
                hc.release(1)
                fc.release(1)

        workers.append(Worker(
            conv_fn, [f2.cons(), x12.prod(), h12.prod(), conv_k],
            tile=Tile(col, 2), stack_size=0xD00))
        rt_args += [f3.prod(tile=Tile(col, 0)),
                    x23.cons(tile=Tile(col, 0)),
                    h23.cons(tile=Tile(col, 0))]

    # ---- norm: cols NC_CONV..NC_CONV+NC_NORM-1, row 3 ----
    nbar = [WorkerRuntimeBarrier() for _ in range(NC_NORM)]
    for ci in range(NC_NORM):
        col = NC_CONV + ci
        n3 = ObjectFifo(HN_T, name=f"n3_{col}", depth=2)
        n2 = n3.cons().forward(obj_type=HN_T, name=f"n2_{col}", tile=Tile(col, 1))
        o23 = ObjectFifo(PKVB_T, name=f"no23_{col}", depth=2)
        o12 = o23.prod().join([0], obj_types=[PKVB_T], names=[f"no12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def norm_fn(nc, oc, k, bar, nh=HP_NORM):
            bar.wait_for_value(1)
            for _ in range_(nh):
                ni = nc.acquire(1)
                oo = oc.acquire(1)
                k(ni, oo)
                oc.release(1)
                nc.release(1)

        workers.append(Worker(
            norm_fn, [n2.cons(), o12.prod(), norm_k, nbar[ci]],
            tile=Tile(col, 3), stack_size=0xD00))
        rt_args += [n3.prod(tile=Tile(col, 0)),
                    o23.cons(tile=Tile(col, 0)),
                    nbar[ci]]

    def group_base(col, g):
        return col * CN * S_V + g * S_V

    # absolute channel -> (head h 0..15, region r 0=q/1=k/2=v)
    def head_region(ach):
        return (ach % 2048) // S_V, ach // 2048

    def seq_fn(FEED, X, PKVB, *fifos):
        it = iter(fifos)

        def nxt():
            return next(it)

        cfeed = []
        cxdrain = []
        chist = []
        for _col in range(NC_CONV):
            cfeed.append(nxt())
            cxdrain.append(nxt())
            chist.append(nxt())
        nfill = []
        ndrain = []
        nbarv = []
        for _ci in range(NC_NORM):
            nfill.append(nxt())
            ndrain.append(nxt())
            nbarv.append(nxt())
        # 1) conv feeds + drains (x placed per head/region, hist back)
        for col in range(NC_CONV):
            fp = cfeed[col]; xc = cxdrain[col]; hc = chist[col]
            for g in range(CN):
                ach = group_base(col, g)
                h, r = head_region(ach)
                gi = TaskGroup()
                fp.fill(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN * FEED_N,),
                    offset=(col * CN + g) * FEED_N, sizes=[FEED_N],
                    strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                xc.drain(X, tap=TensorAccessPattern(
                    (N_VH * HEAD_NORM,),
                    offset=h * HEAD_NORM + r * S_V, sizes=[S_V],
                    strides=[1]), wait=True, group=go)
                hc.drain(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN * FEED_N,),
                    offset=(col * CN + g) * FEED_N + F_H,
                    sizes=[3 * S_V], strides=[1]), wait=True, group=go)
                go.finish()

        # 2) norm: col ci handles heads [ci*HP_NORM, ...)
        for ci in range(NC_NORM):
            nbarv[ci].set(1)
        for ci in range(NC_NORM):
            np_ = nfill[ci]; no_ = ndrain[ci]
            for hs in range(HP_NORM):
                head = ci * HP_NORM + hs
                gi = TaskGroup()
                np_.fill(X, tap=TensorAccessPattern(
                    (N_VH * HEAD_NORM,), offset=head * HEAD_NORM,
                    sizes=[HEAD_NORM], strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                no_.drain(PKVB, tap=TensorAccessPattern(
                    (N_VH * PKVB_N,), offset=head * PKVB_N,
                    sizes=[PKVB_N], strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=NC_CONV + NC_NORM), rt, workers)
    return prog.resolve_program()


def main():
    ap = argparse.ArgumentParser(prog="attn_cn")
    add_compile_args(ap)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--model",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-bf16.gguf")
    ap.add_argument("--qmodel",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-q4km.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"

    os.makedirs(opts.workdir, exist_ok=True)
    base = os.path.join(opts.workdir, "attn_cn")
    spec = attn_cn.specialize(dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(
        xclbin_path=base + ".xclbin", inst_path=base + ".insts.bin")
    if not opts.run:
        print("compiled", xclbin_path)
        return
    if opts.run:
        raise SystemExit("attn_cn --run not supported; validate via llama fused path")




if __name__ == "__main__":
    main()
