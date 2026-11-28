#!/usr/bin/env python3
# gdn_mm.py -*- Python -*-
#
# The prefill gated delta rule on the mmul for
# Qwen3.5's recurrent layers: 16 heads, S = 128. Every core runs
# kernels/gdn-mm.cc and holds one head's state for half of the value columns
# for the whole ubatch:
#
#   core (c, 2 + i): head 2c + i // 2, value half i % 2
#
# A column's input is one stream the MemTile splits to its four cores, its
# output one stream joined from them. A call is data-driven: the first
# object of each core's stream is a header (the chunks), then the state in
# (4 objects), then a chunk an object; out come the chunks' outputs and the
# state (8 objects). The host builds the instruction stream; the one compiled
# here is a sample of the same shape.
#
#   python3 gdn_mm.py -d npu2 --xclbin-path build/bin/gdn_mm_c8.xclbin \
#       --insts-path build/bin/gdn_mm_c8.insts.bin
#
# Per-call layouts (the host's, probes/gdn_design_check.py checks them):
#   X  per column, per object, the four cores' input objects (IN_N bf16)
#   O  per column, per object, the four cores' output objects (O_N f32)

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

COLS, C, DK, DV = 8, 16, 128, 64
IN_N = 2 * C * DK + C * DV + (4 * C + 2 * C * C + 32) * 2     # bf16 an input object
O_N = C * DV                                                   # f32 an output object
N_STATE_IN, N_STATE_OUT = 4, 8
bf16 = ml_dtypes.bfloat16

_here = Path(__file__).resolve().parent
_src = kernelsrc.load(_here / "gdn-mm.cc")
_flags = [f"-DGDN_C={C}"]
_obj = "gdnmm_" + hashlib.md5((_src + str(_flags)).encode()).hexdigest()[:8] + ".o"

in_ty = np.ndarray[(IN_N,), np.dtype[bf16]]
in4_ty = np.ndarray[(4 * IN_N,), np.dtype[bf16]]
o_ty = np.ndarray[(O_N,), np.dtype[np.float32]]
o4_ty = np.ndarray[(4 * O_N,), np.dtype[np.float32]]
st_ty = np.ndarray[(2 * DK * DV,), np.dtype[bf16]]
n_ty = np.ndarray[(8,), np.dtype[np.int32]]


def core_of(col: int, i: int):
    """(head, value half) of the core on row 2 + i."""
    return 2 * col + i // 2, i % 2


def build(dev_name: str = "npu2"):
    """The workers and the object fifos' endpoints; the runtime sequence is the
    caller's. Endpoints: x (a column each), o (a column each)."""
    mk = lambda name, types: ExternalFunction(name, object_file_name=_obj, source_string=_src,
                                              arg_types=types, include_dirs=_include_dirs(),
                                              compile_flags=_flags)
    k_hdr = mk("gdn_hdr", [in_ty, n_ty])
    k_sin = mk("gdn_state_in", [in_ty, st_ty, np.int32])
    k_sout = mk("gdn_state_out", [st_ty, o_ty, np.int32])   # 8 value columns an object
    k_chunk = mk("gdn_chunk", [in_ty, st_ty, o_ty])

    x_col, x_core, o_col, o_core = [], [], [], []
    for col in range(COLS):
        f = ObjectFifo(in4_ty, name=f"x{col}", depth=2)
        x_col.append(f)
        x_core.append(f.cons().split(offsets=[i * IN_N for i in range(4)], obj_types=[in_ty] * 4,
                                     depths=[1] * 4, names=[f"x{col}_{i}" for i in range(4)],
                                     tile=Tile(col, 1)))
        f = ObjectFifo(o4_ty, name=f"o{col}", depth=2)
        o_col.append(f)
        # the MemTile turns each object's 8 x 8 tiles into rows: a chunk's
        # output leaves as C tokens' value halves (the state objects are
        # stored so that they come out as ggml's rows)
        o_core.append(f.prod().join(offsets=[i * O_N for i in range(4)], obj_types=[o_ty] * 4,
                                    depths=[1] * 4, names=[f"o{col}_{i}" for i in range(4)],
                                    tile=Tile(col, 1),
                                    dims_from_stream=[[(C // 8, 8 * DV), (DV // 8, 8), (8, DV), (8, 1)]] * 4))

    def core(x_in, o_out, hdr, sin, sout, chunk, st, cnt):
        hh = x_in.acquire(1)
        hdr(hh, cnt)
        x_in.release(1)
        for part in range(N_STATE_IN):
            xx = x_in.acquire(1)
            sin(xx, st, part)
            x_in.release(1)
        for _ in range_(cnt[0]):
            xx = x_in.acquire(1)
            oo = o_out.acquire(1)
            chunk(xx, st, oo)
            o_out.release(1)
            x_in.release(1)
        for part in range(N_STATE_OUT):
            oo = o_out.acquire(1)
            sout(st, oo, part)
            o_out.release(1)

    workers = []
    for col in range(COLS):
        for i in range(4):
            workers.append(Worker(core, fn_args=[x_core[col][i].cons(), o_core[col][i].prod(),
                                                 k_hdr, k_sin, k_sout, k_chunk,
                                                 Buffer(st_ty, name=f"st{col}_{i}"),
                                                 Buffer(n_ty, name=f"n{col}_{i}")],
                                  tile=Tile(col, 2 + i), stack_size=0xA80))
    eps = ([f.prod(tile=Tile(col, 0)) for col, f in enumerate(x_col)]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(o_col)])
    return workers, eps


@iron.jit
def gdn(x: In, o: Out, *, NCH: CompileTime[int] = 4, dev_name: CompileTime[str] = "npu2"):
    NX = 1 + N_STATE_IN + NCH
    NO = NCH + N_STATE_OUT
    X_ELEMS = COLS * NX * 4 * IN_N
    O_ELEMS = COLS * NO * 4 * O_N
    workers, eps = build(dev_name)

    def seq(x_h, o_h, *e):
        xi, oo = e[:COLS], e[COLS:]
        per_x, per_o = NX * 4 * IN_N, NO * 4 * O_N
        for col in range(COLS):
            xi[col].fill(x_h, tap=TensorAccessPattern([1, X_ELEMS], col * per_x,
                                                      [per_x // 64, 64], [64, 1]))
        for col in range(COLS):
            oo[col].drain(o_h, tap=TensorAccessPattern([1, O_ELEMS], col * per_o,
                                                       [per_o // 64, 64], [64, 1]), wait=True)

    rt = Runtime(seq, [np.ndarray[(X_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(O_ELEMS,), np.dtype[np.float32]]] + eps)
    return Program(from_name(dev_name, n_cols=COLS), rt, workers=workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gdn_mm", description="Build the prefill GDN")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(gdn, opts, compile_kwargs={},
                   device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
