#!/usr/bin/env python3
# gdn_conv.py -*- Python -*-
#
# The recurrent layers' conv input on the array for the prefill GDN
# (kernels/gdn_mm.py): the in-projection's rows convolved along the tokens,
# SiLU'd, K and Q L2-normalized, written as the GDN's chunk inputs. Every core
# runs kernels/gdn-conv.cc:
#
#   column c: heads 2c (A) and 2c + 1 (B); rows 2, 3, 4, 5 are (A, even
#   chunks), (B, even), (A, odd), (B, odd)
#
# A head's input is one stream on its own shim channel - the header pair,
# then the chunks in order, each its C + KW - 1 rows - and the MemTile hands
# a pair of chunks to the head's two cores. A column's output is one stream
# joined from its four cores in the order above, so per chunk it is the GDN
# objects i = 0 .. 3 of the chunk (i = 2 (head % 2) + value half), K, Q, V
# of each: the GDN's input layout, which a shim descriptor walks. The host
# builds the instruction stream; the one compiled here is a sample.
#
#   python3 gdn_conv.py -d npu2 --xclbin-path build/bin/gdn_conv_c8.xclbin \
#       --insts-path build/bin/gdn_conv_c8.insts.bin

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import ml_dtypes
import numpy as np

import kernelsrc

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

COLS, C, KW, D, DV = 8, 16, 4, 128, 64
ROWS = C + KW - 1
NCH = 3 * D
IN_N = ROWS * NCH                           # f32 a chunk (and a header) object
OBJ = 2 * C * D + C * DV                    # bf16 of one GDN input object's K, Q, V
OUT_N = 2 * OBJ                             # bf16 a core's output object
bf16 = ml_dtypes.bfloat16

_here = Path(__file__).resolve().parent
_src = kernelsrc.load(_here / "gdn-conv.cc")
_obj = "gdncv_" + hashlib.md5(_src.encode()).hexdigest()[:8] + ".o"

in_ty = np.ndarray[(IN_N,), np.dtype[np.float32]]
in2_ty = np.ndarray[(2 * IN_N,), np.dtype[np.float32]]
out_ty = np.ndarray[(OUT_N,), np.dtype[bf16]]
out4_ty = np.ndarray[(4 * OUT_N,), np.dtype[bf16]]
n_ty = np.ndarray[(8,), np.dtype[np.int32]]


def core_of(i: int):
    """(head of the column 0/1, chunk parity) of the core on row 2 + i."""
    return i % 2, i // 2


def build(dev_name: str = "npu2"):
    """The workers and the fifos' endpoints: in ([column][head]), out (a
    column each)."""
    mk = lambda name, types: ExternalFunction(name, object_file_name=_obj, source_string=_src,
                                              arg_types=types, include_dirs=_include_dirs())
    k_hdr = mk("gdn_conv_hdr", [in_ty, n_ty])
    k_chunk = mk("gdn_conv_chunk", [in_ty, out_ty])

    ins, cores_in, outs, cores_out = [], [], [], []
    for col in range(COLS):
        per = []
        for h in range(2):
            f = ObjectFifo(in2_ty, name=f"i{col}{h}", depth=2)
            ins.append((col, h, f))
            per.append(f.cons().split(offsets=[0, IN_N], obj_types=[in_ty, in_ty], depths=[1, 1],
                                      names=[f"i{col}{h}_{p}" for p in range(2)], tile=Tile(col, 1)))
        cores_in.append(per)
        f = ObjectFifo(out4_ty, name=f"o{col}", depth=1)
        outs.append(f)
        cores_out.append(f.prod().join(offsets=[i * OUT_N for i in range(4)], obj_types=[out_ty] * 4,
                                       depths=[1] * 4, names=[f"o{col}_{i}" for i in range(4)],
                                       tile=Tile(col, 1)))

    def core(x_in, o_out, hdr, chunk, cnt):
        h = x_in.acquire(1)
        hdr(h, cnt)
        x_in.release(1)
        for _ in range_(cnt[0]):
            xx = x_in.acquire(1)
            oo = o_out.acquire(1)
            chunk(xx, oo)
            o_out.release(1)
            x_in.release(1)

    workers = []
    for col in range(COLS):
        for i in range(4):
            h, p = core_of(i)
            workers.append(Worker(core, fn_args=[cores_in[col][h][p].cons(), cores_out[col][i].prod(),
                                                 k_hdr, k_chunk, Buffer(n_ty, name=f"n{col}_{i}")],
                                  tile=Tile(col, 2 + i), stack_size=0x1200))
    eps = ([f.prod(tile=Tile(col, 0)) for col, h, f in ins]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(outs)])
    return workers, eps


@iron.jit
def gdn_conv(x: In, hdr: In, o: Out, *, NCH_: CompileTime[int] = 2, dev_name: CompileTime[str] = "npu2"):
    """A sample: NCH_ chunks (even) of every head from x = [16 heads][NCH_][IN_N],
    headers [16 heads][2][IN_N]; o per column its joined stream: per chunk the
    four GDN objects' K, Q, V."""
    X_ELEMS = 16 * NCH_ * IN_N
    H_ELEMS = 16 * 2 * IN_N
    O_ELEMS = COLS * NCH_ * 2 * OUT_N
    workers, eps = build(dev_name)

    def seq(x_h, h_h, o_h, *e):
        ii, oo = e[:2 * COLS], e[2 * COLS:]
        for k, (col, h, _) in enumerate([(c, hh, None) for c in range(COLS) for hh in range(2)]):
            head = 2 * col + h
            ii[k].fill(h_h, tap=TensorAccessPattern([1, H_ELEMS], head * 2 * IN_N,
                                                    [2 * ROWS, NCH], [NCH, 1]))
            ii[k].fill(x_h, tap=TensorAccessPattern([1, X_ELEMS], head * NCH_ * IN_N,
                                                    [NCH_ * ROWS, NCH], [NCH, 1]))
        per_o = NCH_ * 2 * OUT_N
        for col in range(COLS):
            oo[col].drain(o_h, tap=TensorAccessPattern([1, O_ELEMS], col * per_o,
                                                       [per_o // 1024, 1024], [1024, 1]), wait=True)

    rt = Runtime(seq, [np.ndarray[(X_ELEMS,), np.dtype[np.float32]],
                       np.ndarray[(H_ELEMS,), np.dtype[np.float32]],
                       np.ndarray[(O_ELEMS,), np.dtype[bf16]]] + eps)
    return Program(from_name(dev_name, n_cols=COLS), rt, workers=workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gdn_conv", description="Build the GDN conv input")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(gdn_conv, opts, compile_kwargs={},
                   device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
