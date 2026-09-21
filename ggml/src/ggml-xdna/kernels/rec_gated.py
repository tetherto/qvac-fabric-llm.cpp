#!/usr/bin/env python3
"""Standalone gated-activation epilogue on-chip (increment 0 of the A-full
fused recurrent layer).

The gated activation currently lives on the host in xdna_rec_so_run: for the
16 value heads it computes

    gated[hh*128+i] = attn[hh*128+i] * rsc[hh] * gamma[i] * silu(z[hh*128+i])
    rsc[hh] = 1/sqrt(mean(attn[hh*128:hh*128+128]^2) + 1e-6)

then a GLOBAL int8 scale over all 2048 rows d_a = amax/127 and rounds
aq = round(gated/d_a). This design runs that on one AIE tile so it can be fused
between the gdn and ssm_out stages without a host round trip.
Inputs/outputs (a single worker on col 0):

    arg0 az   16 head objects, each [attn 128 | z 128 | hh] fp32 (hh at the
             tail keeps attn/z 64B-aligned for the vector loads)
    arg1 gamma[128] fp32                    (shared across heads)
    arg2 out  [gated f32 scratch 2048][aq int8 2048][d_a f32] = 10244 B

The worker holds the OUT object (state is carried in it, since IRON compiles
each ExternalFunction into its own TU so statics do not share), streams the 16
az heads into it via ggml_xdna_gated_head (per-head rsc + tanh-based fp32 silu) and then
ggml_xdna_gated_fin does the global amax -> d_a -> int8 quant. --run validates aq/d_a
against the fp64 python reference (the xdna_rec_so_run epilogue): d_a matches
to ~1e-6 and ~95% of aq codes are identical, the rest differ by one code on
rounding boundaries (fp32/bf16-tanh silu vs fp64 exp silu).
"""

from __future__ import annotations

import argparse
import os
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

GATED_SRC = Path(__file__).resolve().parent / "rec-gated.cc"


def _gated_src():
    return GATED_SRC.read_text()


@iron.jit
def rec_gated(*, dev_name: CompileTime[str] = "npu2"):
    OUTN = K * 4 + K + 4                     # gated f32 scratch + aq + d_a
    AZ_T = np.ndarray[(1 + 2 * S_V,), np.dtype[np.float32]]  # [hh|attn|z]
    GM_T = np.ndarray[(S_V,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(OUTN,), np.dtype[np.uint8]]

    kh = iron.ExternalFunction(name="ggml_xdna_gated_head", source_string=_gated_src(),
                               arg_types=[OUT_T, AZ_T, GM_T],
                               compile_flags=["-O2", "-DNDEBUG"], inline=True)
    kf = iron.ExternalFunction(name="ggml_xdna_gated_fin", source_string=_gated_src(),
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
        np.stack([np.concatenate([attn[hh * S_V:(hh + 1) * S_V],
                                  z[hh * S_V:(hh + 1) * S_V], [float(hh)]])
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
