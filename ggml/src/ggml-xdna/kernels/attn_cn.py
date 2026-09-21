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


CONV_SRC = Path(__file__).resolve().parent / "attn-conv.cc"
NORM_SRC = Path(__file__).resolve().parent / "attn-norm.cc"


def _fill(template: str, **kw) -> str:
    """Substitute the @NAME@ markers in a kernel source template."""
    for k, v in kw.items():
        template = template.replace(f"@{k}@", str(v))
    return template


def _conv_src(gpo: int = 1, feed_slot: int = 0):
    slot = feed_slot or FEED_N
    return _fill(CONV_SRC.read_text(),
                 GPO=gpo, SLOT=slot, S_V=S_V, F_H=F_H, F_Q=F_Q,
                 F_W=F_W, FEED_N=FEED_N)


def _norm_src(name: str = "ggml_xdna_attn_norm", half: int = -1, one: int = -1):
    """`half` emits four of the head's eight chunks, `one` a single chunk by
    index - which is what lets a norm core sit above a gdn core and hand it
    exactly the chunks it takes, with no MemTile and no DDR in between."""
    return _fill(NORM_SRC.read_text(),
                 NAME=name, HALF=half, ONE=one, S_V=S_V, CHUNK=CHUNK,
                 N_OBJ=N_OBJ, PKV_N=PKV_N)


@iron.jit
def attn_cn(*, dev_name: CompileTime[str] = "npu2"):
    FEED_T = np.ndarray[(FEED_N,), np.dtype[np.float32]]
    X_T = np.ndarray[(S_V,), np.dtype[np.float32]]
    HIST_T = np.ndarray[(3 * S_V,), np.dtype[np.float32]]
    HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
    PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]

    conv_k = iron.ExternalFunction(name="ggml_xdna_attn_conv", source_string=_conv_src(),
                                   arg_types=[FEED_T, X_T, HIST_T],
                                   compile_flags=["-O2", "-DNDEBUG"], inline=True)
    norm_k = iron.ExternalFunction(name="ggml_xdna_attn_norm", source_string=_norm_src(),
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
            fp = cfeed[col]
            xc = cxdrain[col]
            hc = chist[col]
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
            np_ = nfill[ci]
            no_ = ndrain[ci]
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
