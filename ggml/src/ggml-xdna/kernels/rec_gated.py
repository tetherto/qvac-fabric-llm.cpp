#!/usr/bin/env python3
"""Standalone gated-activation epilogue on-chip (increment 0 of the A-full
fused recurrent layer).

The gated activation currently lives on the host in xdna_rec_so_run: for the
16 value heads it computes

    gated[hh*128+i] = attn[hh*128+i] * rsc[hh] * gamma[i] * silu(z[hh*128+i])
    rsc[hh] = 1/sqrt(mean(attn[hh*128:hh*128+128]^2) + 1e-6)

then a GLOBAL int8 scale over all 2048 rows d_a = amax/127 and rounds
aq = round(gated/d_a). This design runs exactly that on one AIE tile so it can
later be fused between the gdn and ssm_out stages of attn_gdn_txn without a
host round trip. Inputs/outputs (a single worker on col 0):

    arg0 az   16 head objects, each [attn 128 | z 128 | hh] fp32 (hh at the
             tail keeps attn/z 64B-aligned for the vector loads)
    arg1 gamma[128] fp32                    (shared across heads)
    arg2 out  [gated f32 scratch 2048][aq int8 2048][d_a f32] = 10244 B

The worker holds the OUT object (state is carried in it, since IRON compiles
each ExternalFunction into its own TU so statics do not share), streams the 16
az heads into it via gated_head (per-head rsc + tanh-based fp32 silu) and then
gated_fin does the global amax -> d_a -> int8 quant. --run validates aq/d_a
against the fp64 python reference (the xdna_rec_so_run epilogue): d_a matches
to ~1e-6 and ~95% of aq codes are identical, the rest differ by one code on
rounding boundaries (fp32/bf16-tanh silu vs fp64 exp silu).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args

S_V = 128
N_VH = 16
K = N_VH * S_V
EPS = 1e-6

_AHEAD = """#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;
"""


def _gated_src():
    # State is carried in the OUT object so gated_head and gated_fin can share
    # it without cross-TU statics (IRON compiles each ExternalFunction from the
    # same source string into its own TU).  OUT layout:
    #   [0 .. 8191]      gated f32 scratch (2048 floats), written per head
    #   [8192 .. 10239]  aq int8 codes, one scale for the row
    #   [10240 .. 10243] d_a f32
    #   [ACT_OFF ..]     the same values as ssm_out's GEMV activation: a header
    #                    tile then one per k_tile, int8 codes with a scale and a
    #                    code sum per group of 32. Writing it here is what lets
    #                    the projection read it without a repack on the host.
    # az per head = [attn 128][z 128][hh] (hh at the tail keeps attn/z 64B-aligned; each call finds its slot by hh).
    return _AHEAD + """
extern "C" void gated_head(uint8_t * out, const float * az, const float * gamma) {
    const int hh = (int)az[256];
    float * gbuf = (float *)out + hh * 128;
    const float * a = az;
    const float * z = az + 128;
    const auto bc_h = aie::broadcast<float, 16>(0.5f);
    const auto bc1 = aie::broadcast<bfloat16, 16>(1.0f);
    float ms = 0.0f;
    for (int i = 0; i < 128; i++) ms += a[i] * a[i];
    const float rsc = 1.0f / aie::sqrt(ms / 128.0f + 1e-6f);
    const auto bc_r = aie::broadcast<float, 16>(rsc);
    for (int i = 0; i < 128; i += 16) {
        auto a16 = aie::load_v<16>(a + i);
        auto z16 = aie::load_v<16>(z + i);
        auto g16 = aie::load_v<16>(gamma + i);
        // silu(z) = z * 0.5*(1+tanh(z/2)); tanh is hw bf16, lifted to float
        auto t16 = aie::mul(z16, bc_h).to_vector<float>();
        auto thb = aie::tanh(t16);
        auto th  = aie::mul(thb, bc1).to_vector<float>();
        auto thh = aie::mul(th, bc_h).to_vector<float>();
        auto sig = aie::add(thh, bc_h);
        auto silu = aie::mul(z16, sig).to_vector<float>();
        auto g1 = aie::mul(a16, bc_r).to_vector<float>();
        auto g2 = aie::mul(g1, g16).to_vector<float>();
        aie::vector<float, 16> g = aie::mul(g2, silu).to_vector<float>();
        aie::store_v(gbuf + i, g);
    }
}

#if ACT_SPLIT
extern "C" void gated_fin(uint8_t * out, uint8_t * actbuf) {
#else
extern "C" void gated_fin(uint8_t * out) {
#endif
#if GATED_STUB & 2
    (void)out;
    return;
#else
    const float * gbuf = (const float *)out;
    // Both passes are vector work. They used to be scalar loops over 2048
    // floats with a divide per element, which measured as the single most
    // expensive thing in the fused core - more than the per-head epilogue and
    // four times the data movement of the whole stage.
    aie::vector<float, 16> vmax = aie::zeros<float, 16>();
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
    for (int k = 0; k < 2048; k += 16) {
        auto v = aie::load_v<16>(gbuf + k);
        // aie::abs does not hold for f32 on this target; mask the sign bit.
        auto av = aie::bit_and(v.cast_to<int32>(), absmask).cast_to<float>();
        vmax = aie::max(vmax, av);
    }
    alignas(64) float lanes[16];
    aie::store_v(lanes, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        if (lanes[i] > amax) amax = lanes[i];
    }

    const float da = amax > 0.0f ? amax / 127.0f : 1.0f;
    // One divide for the whole buffer instead of 2048 of them.
    const float inv = 1.0f / da;
    // The float-to-int step goes through the "magic constant": adding
    // 1.5*2^23 puts the rounded integer in the low mantissa bits, and
    // subtracting the constant's own bit pattern leaves it as an int32. The
    // library's to_fixed does not hold on this target, and the scalar loop
    // this replaces was the other half of the stage.
    const auto vinv  = aie::broadcast<float, 16>(inv);
    const auto vlo   = aie::broadcast<float, 16>(-128.0f);
    const auto vhi   = aie::broadcast<float, 16>(127.0f);
    const auto magic = aie::broadcast<float, 16>(12582912.0f);
    const auto magici = magic.cast_to<int32>();
    int8 * aq = (int8 *)(out + 8192);
    for (int k = 0; k < 2048; k += 16) {
        auto v = aie::mul(aie::load_v<16>(gbuf + k), vinv).to_vector<float>();
        v = aie::min(aie::max(v, vlo), vhi);
        // aie::add of two vectors is a vector, not an accumulator.
        auto qi = aie::sub(aie::add(v, magic).cast_to<int32>(), magici);
        aie::store_v(aq + k, aie::pack(aie::pack(qi)));
    }
    float * dap = (float *)(out + 10240);
    dap[0] = da;

    // The same codes again in the GEMV's activation tile layout, but with a
    // scale and a code sum per group of 32 rather than one for the row: a
    // group scale is local to the values a core holds, which is what the
    // projection's kernel reads. Only the 4-bit weight form is covered - the
    // 8-bit one groups by 16 - so the host keeps the other.
#if !defined(GATED_ACT) || GATED_ACT
    {
#if defined(GATED_FMT) && GATED_FMT == 1
        // The 8-bit weight form: tiles of 128 codes, groups of 16, the layout
        // the projection's q8 kernel reads. Which form the model needs is a
        // property of its ssm_out weights - Q5_K and Q6_K are the 8-bit form -
        // so the design is built per model (GATED_FMT=1) and the tag covers it.
        const int NT = K_GATE / 128;
#else
        const int NT = K_GATE / 256;
#endif
        // A drain of its own when the activation is split off: the projection
        // that reads it back in the same stream needs a plainly patched
        // descriptor, and the gated output's own drain cannot have one.
#if ACT_SPLIT
        uint8_t * act = actbuf;
#else
        uint8_t * act = out + ACT_OFF;
#endif
        int32_t * hdr = (int32_t *)act;
        hdr[0] = NT;
        hdr[1] = 1;
        hdr[ACT_TILE / 4 - 2] = 0;
#if defined(GATED_FMT) && GATED_FMT == 1
        hdr[ACT_TILE / 4 - 1] = 1;
#else
        hdr[ACT_TILE / 4 - 1] = 0;
#endif
        for (int t = 0; t < NT; t++) {
            uint8_t * tile = act + (1 + t) * ACT_TILE;
            int8 * code = (int8 *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
            float * gsum = (float *)(tile + 128);
            float * gd = gsum + 8;
            for (int g = 0; g < 8; g++) {
                const float * v = gbuf + t * 128 + g * 16;
                const auto v0 = aie::load_v<16>(v);
                const auto a0 = aie::bit_and(v0.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                const float ga = aie::reduce_max(a0);
                const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
                const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
                auto q0 = aie::mul(v0, ginv).to_vector<float>();
                q0 = aie::min(aie::max(q0, vlo), vhi);
                const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
                aie::store_v(code + g * 16, aie::pack(aie::pack(i0)));
                gsum[g] = (float) aie::reduce_add(i0);
                gd[g] = gdv;
            }
            ((int32_t *)tile)[ACT_TILE / 4 - 2] = 0;
            ((int32_t *)tile)[ACT_TILE / 4 - 1] = 1;
#else
            float * gsum = (float *)(tile + 256);
            float * gd = gsum + 8;
            for (int g = 0; g < 8; g++) {
                const float * v = gbuf + t * 256 + g * 32;
                const auto v0 = aie::load_v<16>(v);
                const auto v1 = aie::load_v<16>(v + 16);
                const auto a0 = aie::bit_and(v0.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                const auto a1 = aie::bit_and(v1.cast_to<int32>(), absmask)
                                    .cast_to<float>();
                // reduce_max, not a scalar scan of the sixteen lanes: at
                // sixty-four groups a branchy scalar loop per group cost the
                // stage 50 us.
                const float ga = aie::reduce_max(aie::max(a0, a1));
                const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
                const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
                auto q0 = aie::mul(v0, ginv).to_vector<float>();
                auto q1 = aie::mul(v1, ginv).to_vector<float>();
                q0 = aie::min(aie::max(q0, vlo), vhi);
                q1 = aie::min(aie::max(q1, vlo), vhi);
                const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
                const auto i1 = aie::sub(aie::add(q1, magic).cast_to<int32>(), magici);
                aie::store_v(code + g * 32, aie::pack(aie::pack(i0)));
                aie::store_v(code + g * 32 + 16, aie::pack(aie::pack(i1)));
                gsum[g] = (float) aie::reduce_add(aie::add(i0, i1));
                gd[g] = gdv;
            }
            ((int32_t *)tile)[ACT_TILE / 4 - 2] = 0;
            ((int32_t *)tile)[ACT_TILE / 4 - 1] = 0;
#endif
        }
    }
#endif
#endif
}
"""

@iron.jit
def rec_gated(*, dev_name: CompileTime[str] = "npu2"):
    OUTN = K * 4 + K + 4                     # gated f32 scratch + aq + d_a
    AZ_T = np.ndarray[(1 + 2 * S_V,), np.dtype[np.float32]]  # [hh|attn|z]
    GM_T = np.ndarray[(S_V,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(OUTN,), np.dtype[np.uint8]]

    kh = iron.ExternalFunction(name="gated_head", source_string=_gated_src(),
                               arg_types=[OUT_T, AZ_T, GM_T],
                               compile_flags=["-O2", "-DNDEBUG"], inline=True)
    kf = iron.ExternalFunction(name="gated_fin", source_string=_gated_src(),
                               arg_types=[OUT_T],
                               compile_flags=["-O2", "-DNDEBUG"], inline=True)

    az3 = ObjectFifo(AZ_T, name="gaz3", depth=2)
    az2 = az3.cons().forward(obj_type=AZ_T, name="gaz2", tile=Tile(0, 1))
    gm = ObjectFifo(GM_T, name="ggm3", depth=2)
    g2 = gm.cons().forward(obj_type=GM_T, name="ggm2", tile=Tile(0, 1))
    o23 = ObjectFifo(OUT_T, name="go23", depth=1)
    o12 = o23.prod().join([0], obj_types=[OUT_T], names=["go12"],
                          depths=[1], tile=Tile(0, 1))[0]

    def gated_fn(ac, gc, oc, kh_, kf_):
        o = oc.acquire(1)
        g = gc.acquire(1)
        for _ in range(N_VH):
            a = ac.acquire(1)
            kh_(o, a, g)
            ac.release(1)
        gc.release(1)
        kf_(o)
        oc.release(1)

    workers = [Worker(gated_fn, [az2.cons(), g2.cons(), o12.prod(), kh, kf],
                      tile=Tile(0, 2), stack_size=0x3000)]

    AZ_g = np.ndarray[(N_VH * (1 + 2 * S_V),), np.dtype[np.float32]]
    GM_g = np.ndarray[(S_V,), np.dtype[np.float32]]
    OUT_g = np.ndarray[(OUTN,), np.dtype[np.uint8]]

    def seq_fn(AZ, GM, OUT, azf, gmf, outf):
        tg = TaskGroup()
        # one contiguous BD streams all 16 head objects (each [hh|attn|z])
        azf.fill(AZ, tap=TensorAccessPattern(
            (N_VH * (1 + 2 * S_V),), offset=0,
            sizes=[N_VH * (1 + 2 * S_V)], strides=[1]), group=tg)
        gmf.fill(GM, tap=TensorAccessPattern(
            (S_V,), offset=0, sizes=[S_V], strides=[1]), group=tg)
        tg.finish()
        go = TaskGroup()
        outf.drain(OUT, tap=TensorAccessPattern(
            (OUTN,), offset=0, sizes=[OUTN], strides=[1]), wait=True, group=go)
        go.finish()

    rt = Runtime(seq_fn,
                 [AZ_g, GM_g, OUT_g, az3.prod(tile=Tile(0, 0)),
                  gm.prod(tile=Tile(0, 0)), o23.cons(tile=Tile(0, 0))])
    prog = Program(from_name(dev_name, n_cols=1), rt, workers)
    return prog.resolve_program()


# ---- host fp64 reference (mirror of xdna_rec_so_run) -------------------------

def host_gated(attn, z, gamma):
    """attn/z [2048], gamma [128] fp64 -> (gated fp64, aq int8, d_a)."""
    gated = np.empty(K, dtype=np.float64)
    for hh in range(N_VH):
        a = attn[hh * S_V:(hh + 1) * S_V].astype(np.float64)
        ms = float(np.dot(a, a))
        rsc = 1.0 / np.sqrt(ms / S_V + EPS)
        for i in range(S_V):
            kk = hh * S_V + i
            zv = float(z[kk])
            s = zv / (1.0 + np.exp(-zv))
            gated[kk] = float(attn[kk]) * rsc * float(gamma[i]) * s
    amax = np.abs(gated).max()
    da = float(amax / 127.0) if amax > 0 else 1.0
    aq = np.clip(np.rint(gated / da), -128, 127).astype(np.int8)
    return gated, aq, np.float32(da)


def main():
    ap = argparse.ArgumentParser(prog="rec_gated")
    add_compile_args(ap)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--tag", default=None)
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.tag is None:
        opts.tag = time.strftime("gated_%H%M%S")
    wd = os.path.join(opts.workdir, opts.tag)
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "rec_gated")
    xclbin_path = base + ".xclbin"
    insts_path = base + ".insts.bin"
    t0 = time.time()
    spec = rec_gated.specialize(dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(xclbin_path=xclbin_path,
                                           inst_path=insts_path)
    print(f"compiled {xclbin_path} ({time.time()-t0:.0f}s)")
    if not opts.run:
        return

    import pyxrt as xrt

    rng = np.random.default_rng(7)
    attn = rng.standard_normal(K).astype(np.float32)
    z = rng.standard_normal(K).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, S_V).astype(np.float32)

    gated_ref, aq_ref, da_ref = host_gated(attn.astype(np.float64),
                                            z.astype(np.float64),
                                            gamma.astype(np.float64))

    az = np.ascontiguousarray(
        np.stack([np.concatenate([attn[hh*S_V:(hh+1)*S_V],
                                  z[hh*S_V:(hh+1)*S_V], [float(hh)]])
                  for hh in range(N_VH)]).reshape(-1),
        dtype=np.float32)

    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(ctx, xb.get_kernels()[0].get_name())
    insts = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
    insts_bo = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    np.frombuffer(insts_bo.map(), dtype=np.uint32)[:] = insts
    insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def mk_bo(arr):
        bo = xrt.bo(dev, arr.nbytes, xrt.bo.host_only, 0)
        np.frombuffer(bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    az_bo = mk_bo(az)
    gm_bo = mk_bo(gamma)
    OUTN = K * 4 + K + 4
    out_bo = xrt.bo(dev, OUTN, xrt.bo.host_only, 0)

    run = xrt.run(kernel)
    run.set_arg(0, 3)
    run.set_arg(1, insts_bo)
    run.set_arg(2, insts.nbytes)
    run.set_arg(3, az_bo)
    run.set_arg(4, gm_bo)
    run.set_arg(5, out_bo)
    run.start()
    run.wait()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(out_bo.map(), dtype=np.uint8)[:]
    aq = out[K * 4:K * 4 + K].copy().view(np.int8)
    da = np.frombuffer(out[K * 4 + K:K * 4 + K + 4].copy(), dtype=np.float32)[0]
    nmis = int(np.count_nonzero(aq != aq_ref))
    print(f"[host] da_ref={da_ref:.6e} aq mean={aq_ref.mean():.2f}")
    print(f"[dev ] da    ={da:.6e} aq mean={aq.mean():.2f}  mismatches={nmis}/{K}")
    if nmis == 0:
        print("rec_gated: aq IDENTICAL to host fp64 reference")
    else:
        print(f"rec_gated: {nmis} aq codes differ (fp32 silu vs fp64 reference)")


if __name__ == "__main__":
    main()
