#!/usr/bin/env python3
# attn_mm.py -*- Python -*-
#
# Prefill attention on the mmul for Qwen3.5's
# full-attention layers: 8 query heads, 2 KV heads, D = 256, causal. Every
# core runs kernels/attn-mm.cc.
#
#   KV head group g (0, 1) owns columns 4g .. 4g + 3. A column's rows 2, 3
#   are a pair and rows 4, 5 another; a pair's two cores hold the two halves
#   of D and trade their halves of the scores through shared memory. The
#   pair pidx = (column - 4g) * 2 + (row - 2) // 2 of a group owns 32 rows:
#   the group's 4 query heads at 8 positions, pidx * 8 .. + 7 of each pass.
#   A pass is 64 positions of both groups.
#   Each D half of a group's K/V is one stream, broadcast by the MemTile to
#   the 8 cores that hold that half: all 8 pairs take the same key tiles in
#   lockstep, keys 0 to the pass's last position.
#
# A call is data-driven: the first object of each core's Q stream is a header
# (fa_hdr: the passes, the ubatch's first position), and a core works out
# its positions and a pass's key tiles itself, so one artifact serves every
# length. The host builds the instruction stream; the one compiled here is a
# sample of the same shape.
#
#   python3 attn_mm.py -d npu2 --xclbin-path build/bin/attn_mm_c8.xclbin \
#       --insts-path build/bin/attn_mm_c8.insts.bin
#
# Per-call layouts (the host's, probes/fa_design_check.py checks them):
#   Q   per column: the header object, then per pass the column's four
#       cores' Q half^T, (d/8, row/8, 8, 8) bf16, Q scaled by log2(e) / 16
#   KV  per group and D half, per 32-key tile: K half (key/8, d/8, 8, 8) then
#       V half^T (d/8, key/8, 8, 8), bf16; each pass streams tiles 0 .. its
#       last
#   O   per column, per pass, the four cores' O half as rows (32 x 128) f32

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

COLS, R, D, DH, NK = 8, 32, 256, 128, 32
PASS = 64                                # positions a pass
bf16 = ml_dtypes.bfloat16

_here = Path(__file__).resolve().parent
_src = kernelsrc.load(_here / "attn-mm.cc")
_flags = [f"-DFA_R={R}", f"-DFA_DH={DH}", f"-DFA_NK={NK}"]
_obj = "attmm_" + hashlib.md5((_src + str(_flags)).encode()).hexdigest()[:8] + ".o"

q_ty = np.ndarray[(R * DH,), np.dtype[bf16]]
q4_ty = np.ndarray[(4 * R * DH,), np.dtype[bf16]]
kv_ty = np.ndarray[(NK * DH,), np.dtype[bf16]]      # half a tile: K or V^T
s_ty = np.ndarray[(R * NK,), np.dtype[np.float32]]
p_ty = np.ndarray[(R * NK,), np.dtype[bf16]]
o_ty = np.ndarray[(R * DH,), np.dtype[np.float32]]
o4_ty = np.ndarray[(4 * R * DH,), np.dtype[np.float32]]
ml_ty = np.ndarray[(2 * R,), np.dtype[np.float32]]
n_ty = np.ndarray[(8,), np.dtype[np.int32]]


def core_of(col: int, i: int):
    """(group g, pair index pidx, D half h) of the core on row 2 + i."""
    g = col // 4
    return g, (col - 4 * g) * 2 + i // 2, i % 2


def kv_col(g: int, h: int) -> int:
    """The shim column the K/V stream of group g, half h enters on."""
    return 4 * g + h


def build(dev_name: str = "npu2"):
    """The workers and the object fifos' endpoints; the runtime sequence is the
    caller's. Endpoints: q (a column each), kv ([g][h]), o (a column each)."""
    mk = lambda name, types: ExternalFunction(name, object_file_name=_obj, source_string=_src,
                                              arg_types=types, include_dirs=_include_dirs(),
                                              compile_flags=_flags)
    k_hdr = mk("fa_hdr", [q_ty, n_ty])
    k_pass = mk("fa_pass", [o_ty, ml_ty, n_ty, np.int32])
    k_scores = mk("fa_scores", [q_ty, kv_ty, s_ty])
    k_update = mk("fa_update_p", [s_ty, s_ty, o_ty, p_ty, ml_ty])
    k_pv = mk("fa_pv", [p_ty, kv_ty, o_ty])
    k_end = mk("fa_end", [o_ty, ml_ty])

    q_col, q_core = [], []
    for col in range(COLS):
        f = ObjectFifo(q4_ty, name=f"q{col}", depth=2)
        q_col.append(f)
        q_core.append(f.cons().split(offsets=[i * R * DH for i in range(4)],
                                     obj_types=[q_ty] * 4, depths=[1] * 4,
                                     names=[f"q{col}_{i}" for i in range(4)],
                                     tile=Tile(col, 1)))
    kv_in, kv_b = [[None] * 2 for _ in range(2)], [[None] * 2 for _ in range(2)]
    for g in range(2):
        for h in range(2):
            f = ObjectFifo(kv_ty, name=f"kv{g}{h}", depth=2)
            kv_in[g][h] = f
            kv_b[g][h] = f.cons().forward(obj_type=kv_ty, depth=2, name=f"kvb{g}{h}",
                                          tile=Tile(kv_col(g, h), 1))
    o_col, o_core = [], []
    for col in range(COLS):
        f = ObjectFifo(o4_ty, name=f"o{col}", depth=1)
        o_col.append(f)
        # O half^T's (8, 8) tiles, [d/8][row/8][d % 8][row % 8], leave the
        # MemTile as rows: each of the core's 32 rows its 128 values
        o_core.append(f.prod().join(offsets=[i * R * DH for i in range(4)],
                                    obj_types=[o_ty] * 4, depths=[1] * 4,
                                    names=[f"o{col}_{i}" for i in range(4)], tile=Tile(col, 1),
                                    dims_from_stream=[[(DH // 8, 8), (R // 8, 8 * DH), (8, 1), (8, DH)]] * 4))

    def core(q_in, kv, o_out, x_own, x_nb, hdr, pas, scores, update, pv, end, pb, ml, cnt, pidx):
        hq = q_in.acquire(1)
        hdr(hq, cnt)
        q_in.release(1)
        for _ in range_(cnt[0]):
            qq = q_in.acquire(1)
            oo = o_out.acquire(1)
            pas(oo, ml, cnt, pidx)
            for _ in range_(cnt[1]):
                # the tile's K, then its V^T: each object is released as
                # soon as it is used, so the stream brings the next one in
                kk = kv.acquire(1)
                so = x_own.acquire(1)
                scores(qq, kk, so)
                kv.release(1)
                x_own.release(1)
                sn = x_nb.acquire(1)
                update(so, sn, oo, pb, ml)
                x_nb.release(1)
                vv = kv.acquire(1)
                pv(pb, vv, oo)
                kv.release(1)
            end(oo, ml)
            o_out.release(1)
            q_in.release(1)

    workers = []
    for col in range(COLS):
        x = [[ObjectFifo(s_ty, name=f"x{col}_{pr}_{h}", depth=1) for h in range(2)]
             for pr in range(2)]
        for i in range(4):
            g, pidx, h = core_of(col, i)
            pr = i // 2
            workers.append(Worker(core, fn_args=[
                q_core[col][i].cons(), kv_b[g][h].cons(), o_core[col][i].prod(),
                x[pr][h].prod(), x[pr][1 - h].cons(), k_hdr, k_pass, k_scores, k_update,
                k_pv, k_end, Buffer(p_ty, name=f"p{col}_{i}"),
                Buffer(ml_ty, name=f"ml{col}_{i}"), Buffer(n_ty, name=f"n{col}_{i}"), pidx],
                tile=Tile(col, 2 + i), stack_size=0xE00))
    eps = ([f.prod(tile=Tile(col, 0)) for col, f in enumerate(q_col)]
           + [kv_in[g][h].prod(tile=Tile(kv_col(g, h), 0)) for g in range(2) for h in range(2)]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(o_col)])
    return workers, eps


def tiles_of(p0: int, p: int) -> int:
    """Key tiles pass p streams."""
    return (p0 + p * PASS + PASS + NK - 1) // NK


@iron.jit
def attn(q: In, kv: In, o: Out, *, NPASS: CompileTime[int] = 2, P0: CompileTime[int] = 0,
         dev_name: CompileTime[str] = "npu2"):
    Q_ELEMS = COLS * (1 + NPASS) * 4 * R * DH
    T_MAX = tiles_of(P0, NPASS - 1)
    KV_OBJ = 2 * NK * DH                 # a tile: two stream objects
    KV_ELEMS = 4 * T_MAX * KV_OBJ
    O_ELEMS = COLS * NPASS * 4 * R * DH
    workers, eps = build(dev_name)

    def seq(q_h, kv_h, o_h, *e):
        qi, ki, oo = e[:COLS], e[COLS:COLS + 4], e[COLS + 4:]
        per_q = (1 + NPASS) * 4 * R * DH
        for col in range(COLS):
            qi[col].fill(q_h, tap=TensorAccessPattern([1, Q_ELEMS], col * per_q,
                                                      [1, per_q], [0, 1]))
        for p in range(NPASS):
            n = tiles_of(P0, p) * KV_OBJ
            for s in range(4):
                ki[s].fill(kv_h, tap=TensorAccessPattern([1, KV_ELEMS], s * T_MAX * KV_OBJ,
                                                         [n // 1024, 1024], [1024, 1]))
        per_o = NPASS * 4 * R * DH
        for col in range(COLS):
            oo[col].drain(o_h, tap=TensorAccessPattern([1, O_ELEMS], col * per_o,
                                                       [per_o // 1024, 1024], [1024, 1]),
                          wait=True)

    rt = Runtime(seq, [np.ndarray[(Q_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(KV_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(O_ELEMS,), np.dtype[np.float32]]] + eps)
    return Program(from_name(dev_name, n_cols=COLS), rt, workers=workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(prog="attn_mm", description="Build the prefill attention")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(attn, opts, compile_kwargs={},
                   device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
