#!/usr/bin/env python3
# pgemm.py -*- Python -*-
#
# The prefill GEMM: C (M x N) = A (M x K) @ W
# with W the decode GEMV's packed weight tiles, in the decode's own order,
# expanded on the cores (gemm-expand.cc) and multiplied by the bfp16 mmul -
# every one of the 32 cores doing both (one expander feeding one multiplier
# is 1.7x slower a core: the expander, ~2.5 us a 64-row step, is the bound).
#
#   core (c, row), row 2..5: the decode's GEMV core (c, r = (row - 2) // 2) -
#   it gets that core's tiles - and M half mb = (row - 2) % 2 of 64 rows. The
#   two cores of a tile are neighbours: each expands every other K step of
#   it (mb 0 the even ones) into its own memory and the other reads it there,
#   so a core expands half the steps it multiplies (1.82 -> 1.33 ms at 512 x
#   1024 x 6144). The odd core multiplies its own step of a pair first, so
#   its A stream carries each pair swapped. Each A stream feeds 16 cores.
#
# A call is data-driven: the first object of each A stream is a header
# (ggml_xdna_pg_hdr_p) with the output blocks a core computes, the K tiles a
# block, the tile format, the K steps a tile and the pairs of them, so one
# artifact serves every
# M, K, N and format, and nothing is poked into core memory at a fixed
# address. The host builds the call's instruction stream (xdna-pgemm.cpp);
# the one compiled here is only a sample of the same shape.
#
#   python3 pgemm.py -d npu2 --xclbin-path build/bin/pgemm_c8.xclbin \
#       --insts-path build/bin/pgemm_c8.insts.bin
#
# (the sample stream compiled here leaves the weights out: it only exists
# because the artifact needs one, and does not run)
#
# Per-call layouts (the host's):
#   W  per column [chunk][group][r][K tile] tiles of TB bytes. The weights'
#      path is explicit - shim MM2S 1 -> the MemTile's slot buffer (S2MM 5)
#      -> MM2S 4 + r -> the two cores of GEMV tile r (their S2MM 1) - and its
#      MemTile side has no program here: the host sets it up per call, a
#      chunk's tiles taken in once and sent once per M block (xdna-pgemm.cpp)
#   A  a stream per mb: the header object, then per chunk and M block the
#      block's K steps (a_steps order), each (64 x 64) bf16 row-major - the
#      MemTile tiles them into the mmul's (8, 8) sub-tiles, so the host sends
#      A as it is (the header goes through the same tiling)
#   C  row-major (M x N_out). A core's 64 weight columns of chunk ch are two
#      32-column parts h; its block leaves as objects of 64 rows x 32 columns,
#      part h (mode 0) or the SwiGLU of gate part 0 and up part 1 (mode 1).
#      Object s of a column's block - s = 2 ch + h, or ch - is rows blk *
#      128 .., columns s * 512 + column * 64 + r * 32 .., sent as [GEMV tile
#      r][128 rows][32 columns] (the MemTile reorders the cores' sub-tiles):
#      the host packs a weight's rows in that order, so one shim descriptor
#      a block, iterated over s, writes C straight into the output

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import ml_dtypes
import numpy as np

import kernelsrc

import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import Buffer, CompileTime, In, Lock, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.dataflow import Flow, PacketFlow, TileDma
from aie.iron.dataflow.tile_dma import Acquire, Bd, DmaChannel, Release
from aie.dialects._aie_enum_gen import DMAChannelDir, WireBundle
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

import gemv_q4 as gq

COLS, NC, KS, MB = 8, 64, 64, 64       # columns, a core's columns, K a step, M a core
# The W path's fixed resources, the host's to program (xdna-pgemm.cpp)
MT_CTRL_PKT = 26                       # the MemTiles' controller packet id
WLOCK0     = 48                        # MemTile locks: slot s's (empty, full) = 48 + 2 s, + 1
W_SHIM_CH  = 1                         # shim MM2S channel of W
W_MT_S2MM  = 5                         # MemTile S2MM channel W arrives on
W_MT_MM2S  = 4                         # MemTile MM2S channels: GEMV tile r on 4 + r
W_CORE_CH  = 1                         # core S2MM channel W lands on
WS_ADDR    = 0x24000                   # the MemTile slot buffer: past the A objects of
WS_BYTES   = 33 * 10752                # columns 6 and 7, to the end of the 512 KB
CHUNK = COLS * 2 * NC                  # a GEMV chunk: 1024 output columns
TB = gq.tile_bytes("q4g32", 256, NC)   # 10,752 B, both formats
HDR = MB * KS                          # the header object is one A object
bf16 = ml_dtypes.bfloat16

_here = Path(__file__).resolve().parent
_src = kernelsrc.load(_here / "gemm-expand.cc")
# the bfp16 mmul is (8, 8, 8): akernels.mm().mac_dims misreports bf16's
_flags = ["-DMAC_T=8", f"-DEXP_STEP={KS}", "-DEXP_UNROLL=8"]
_obj = "pgexp_" + hashlib.md5((_src + str(_flags)).encode()).hexdigest()[:8] + ".o"

t_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
w2_ty = np.ndarray[(2 * TB,), np.dtype[np.uint8]]
a_ty = np.ndarray[(MB * KS,), np.dtype[bf16]]
b_ty = np.ndarray[(KS * NC,), np.dtype[bf16]]
acc_ty = np.ndarray[(MB * NC,), np.dtype[np.float32]]
c_ty = np.ndarray[(MB * NC // 2,), np.dtype[np.float32]]        # 64 rows x 32 columns
c4_ty = np.ndarray[(4 * MB * NC // 2,), np.dtype[np.float32]]
n_ty = np.ndarray[(8,), np.dtype[np.int32]]
ws_ty = np.ndarray[(WS_BYTES,), np.dtype[np.uint8]]


class _Hold(TileDma):
    """Resolves buffers no program here touches (the host's DMA uses them)
    without emitting a DMA region of its own."""

    def __init__(self, tile, buffers):
        super().__init__(tile, [])
        self._bufs = list(buffers)

    def all_buffers_and_locks(self):
        return self._bufs, []

    def resolve(self, loc=None, ip=None):
        pass


def core_of(i: int):
    """(GEMV core r, M half mb) of the core on row 2 + i."""
    return i // 2, i % 2


def build(dev_name: str = "npu2"):
    """The workers and the object fifos' endpoints; the runtime sequence is the
    caller's. Endpoints: w (a column each), a (mb 0, 1), c (a column each)."""
    mm = akernels.mm(MB, KS, NC, input_dtype=bf16, output_dtype=np.float32, vectorized=True,
                     emulate_bf16_mmul_with_bfp16=True)
    mk = lambda name, types: ExternalFunction(name, object_file_name=_obj, source_string=_src,
                                              arg_types=types, include_dirs=_include_dirs(),
                                              compile_flags=_flags)
    k_hdr = mk("ggml_xdna_pg_hdr_p", [a_ty, n_ty, np.int32])
    k_exp = mk("ggml_xdna_expand_pair", [t_ty, b_ty, n_ty])
    k_out = mk("ggml_xdna_pg_out", [acc_ty, c_ty, n_ty])

    # W: shim -> the MemTile's slot buffer -> the cores, the MemTile's side
    # programmed by the host per call (xdna-pgemm.cpp), so a chunk's tiles
    # are read from DDR once and replayed for every M block
    flows, locks, dmas = [], [], []
    w_core = []
    for col in range(COLS):
        mt, sh = Tile(col, 1), Tile(col, 0)
        wsb = Buffer(ws_ty, name=f"ws{col}", tile=mt, address=WS_ADDR)
        dmas.append(_Hold(mt, [wsb]))
        for k in range(4):
            locks.append(Lock(mt, lock_id=WLOCK0 + k, init=0, name=f"wl{col}_{k}"))
        flows.append(Flow(sh, mt, src_channel=W_SHIM_CH, dst_channel=W_MT_S2MM))
        # the MemTile's task tokens, which the host waits on to keep its
        # channels' queues short (the compiler gives MemTiles packet id 26)
        flows.append(PacketFlow(MT_CTRL_PKT, mt, sh, src_port=WireBundle.TileControl, src_channel=0,
                                dst_port=WireBundle.South, dst_channel=0, keep_pkt_header=True))
        per = []
        for i in range(4):
            r, mb = core_of(i)
            ct = Tile(col, 2 + i)
            flows.append(Flow(mt, ct, src_channel=W_MT_MM2S + r, dst_channel=W_CORE_CH))
            wb = Buffer(t_ty, name=f"wb{col}_{i}", tile=ct)
            wp = Lock(ct, init=1, name=f"wp{col}_{i}")
            wc = Lock(ct, init=0, name=f"wc{col}_{i}")
            locks += [wp, wc]
            dmas.append(TileDma(ct, [DmaChannel(DMAChannelDir.S2MM, W_CORE_CH,
                                                [Bd(wb, acquires=[Acquire(wp)], releases=[Release(wc)])])]))
            per.append((wb, wp, wc))
        w_core.append(per)
    # A: the host's row-major (64 x 64) K steps, tiled into the mmul's (8, 8)
    # sub-tiles by the MemTile on the way to the cores
    a_src = [ObjectFifo(a_ty, name=f"as{mb}", depth=2) for mb in range(2)]
    a_f = [a_src[mb].cons().forward(tile=Tile(6 + mb, 1), obj_type=a_ty, depth=2, name=f"a{mb}",
                                    dims_to_stream=[(MB // 8, 8 * KS), (KS // 8, 8), (8, KS), (8, 1)])
           for mb in range(2)]
    c_col, c_core = [], []
    for col in range(COLS):
        f = ObjectFifo(c4_ty, name=f"c{col}", depth=1)
        c_col.append(f)
        # A core's C object is 64 rows x 32 columns in (8, 8) sub-tiles; the
        # MemTile writes it as plain rows, the column's four in core order:
        # [GEMV tile r][M half mb][64 rows][32], a block of 128 rows of the
        # two tiles' 32-column parts that a shim descriptor puts straight
        # into the output
        c_core.append(f.prod().join(offsets=[i * MB * NC // 2 for i in range(4)],
                                    obj_types=[c_ty] * 4, depths=[1] * 4,
                                    names=[f"c{col}_{i}" for i in range(4)], tile=Tile(col, 1),
                                    dims_from_stream=[[(MB // 8, 8 * NC // 2), (NC // 16, 8),
                                                       (8, NC // 2), (8, 1)]] * 4))

    # (a stack of 0xA00 still fits but runs 1.3-3.5x slower: 0x800 it stays)
    # The block accumulates in the core's own buffer, so the next block runs
    # while the last one's C objects leave: two column halves (mode 0) or
    # their SwiGLU (mode 1), header word 5 of them
    def core(wb, wp, wc, a_in, c_out, x_own, x_nb, zero, mul, khdr, kexp, kout, acc, cnt, par):
        h = a_in.acquire(1)
        khdr(h, cnt, par)
        a_in.release(1)
        for _ in range_(cnt[0]):
            zero(acc)
            for _ in range_(cnt[1]):
                wc.acquire(1)
                t = wb
                for _ in range_(cnt[7]):
                    xo = x_own.acquire(1)
                    kexp(t, xo, cnt)
                    aa = a_in.acquire(1)
                    mul(aa, xo, acc)
                    a_in.release(1)
                    x_own.release(1)
                    xn = x_nb.acquire(1)
                    aa = a_in.acquire(1)
                    mul(aa, xn, acc)
                    a_in.release(1)
                    x_nb.release(1)
                # the tile's last pair: the tile goes back once expanded, so
                # the next one moves in while this pair multiplies
                xo = x_own.acquire(1)
                kexp(t, xo, cnt)
                wp.release(1)
                aa = a_in.acquire(1)
                mul(aa, xo, acc)
                a_in.release(1)
                x_own.release(1)
                xn = x_nb.acquire(1)
                aa = a_in.acquire(1)
                mul(aa, xn, acc)
                a_in.release(1)
                x_nb.release(1)
            for _ in range_(cnt[5]):
                cc = c_out.acquire(1)
                kout(acc, cc, cnt)
                c_out.release(1)

    workers = []
    for col in range(COLS):
        x = [[ObjectFifo(b_ty, name=f"x{col}_{r}_{p}", depth=1) for p in range(2)]
             for r in range(2)]
        for i in range(4):
            r, mb = core_of(i)
            workers.append(Worker(core, fn_args=[
                *w_core[col][i], a_f[mb].cons(), c_core[col][i].prod(),
                x[r][mb].prod(), x[r][1 - mb].cons(), mm.zero, mm, k_hdr, k_exp, k_out,
                Buffer(acc_ty, name=f"acc{col}_{i}"), Buffer(n_ty, name=f"n{col}_{i}"), mb],
                tile=Tile(col, 2 + i), stack_size=0x800))
    eps = ([f.prod(tile=Tile(6 + mb, 0)) for mb, f in enumerate(a_src)]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(c_col)])
    return workers, eps, (flows, locks, dmas)


def add_explicit(rt, extra):
    flows, locks, dmas = extra
    for f in flows:
        rt.add_flow(f)
    for lk in locks:
        rt.add_lock(lk)
    for d in dmas:
        rt.add_tile_dma(d)


def sizes(M: int, K: int, N: int, q8: bool, glu: bool = False):
    k_tile = 128 if q8 else 256
    return dict(NT=K // k_tile, STEPS=k_tile // KS, NSTEP=K // KS, NCH=N // CHUNK,
                NMB=M // (2 * MB), OBJS=1 if glu else 2)


@iron.jit
def pgemm(w: In, a: In, c: Out, *, M: CompileTime[int] = 512, K: CompileTime[int] = 1024,
          N: CompileTime[int] = 2048, Q8: CompileTime[int] = 0, GLU: CompileTime[int] = 0,
          dev_name: CompileTime[str] = "npu2"):
    z = sizes(M, K, N, bool(Q8), bool(GLU))
    NT, NSTEP, NCH, NMB = z["NT"], z["NSTEP"], z["NCH"], z["NMB"]
    W_BYTES = COLS * NCH * NT * 2 * TB
    A_ELEMS = 2 * HDR + NMB * 2 * NSTEP * MB * KS
    N_OUT = N // 2 if GLU else N
    C_ELEMS = M * N_OUT
    OBJS = z["OBJS"]
    workers, eps, extra = build(dev_name)

    def seq(w_h, a_h, c_h, *e):
        # (the sample leaves W out: the host programs the MemTile side)
        ai, co = e[:2], e[2:]
        # (a zero stride only in the outermost dimension: the M blocks are
        # fills of their own, the chunks' repeat of A is a zero stride)
        n = NSTEP * MB * KS
        for mb in range(2):
            ai[mb].fill(a_h, tap=TensorAccessPattern([1, A_ELEMS], mb * HDR, [1, HDR], [0, 1]))
        for blk in range(NMB):
            for mb in range(2):
                ai[mb].fill(a_h, tap=TensorAccessPattern([1, A_ELEMS],
                                                         2 * HDR + (blk * 2 + mb) * n,
                                                         [NCH, 1, n // 512, 512],
                                                         [0, n, 512, 1]))
        # a column's block is NCH * OBJS objects, object s the columns
        # s * 512 + column * 64 + r * 32 .. + 32 of the block's 128 rows
        for blk in range(NMB):
            for col in range(COLS):
                co[col].drain(c_h, tap=TensorAccessPattern([1, C_ELEMS],
                                                           blk * 2 * MB * N_OUT + col * NC,
                                                           [NCH * OBJS, 2, 2 * MB, NC // 2],
                                                           [CHUNK // 2, NC // 2, N_OUT, 1]),
                              wait=blk == NMB - 1)

    rt = Runtime(seq, [np.ndarray[(W_BYTES,), np.dtype[np.uint8]],
                       np.ndarray[(A_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(C_ELEMS,), np.dtype[np.float32]]] + eps)
    add_explicit(rt, extra)
    return Program(from_name(dev_name, n_cols=COLS), rt, workers=workers).resolve_program()


def a_steps(mb: int, nstep: int) -> list:
    """The K steps of M half `mb` in the order its cores take them: the odd
    expander multiplies its own step of a pair first."""
    if mb == 0:
        return list(range(nstep))
    return [s ^ 1 for s in range(nstep)]


def header(M: int, K: int, N: int, q8: bool, glu: bool = False) -> np.ndarray:
    """The header object for this call, as bf16 elements."""
    z = sizes(M, K, N, q8, glu)
    h = np.zeros(HDR * 2 // 4, np.int32)
    h[:8] = [z["NMB"] * z["NCH"], z["NT"], 1 if q8 else 0, z["STEPS"], z["STEPS"] // 2,
             z["OBJS"], 1 if glu else 0, z["STEPS"] // 2 - 1]
    return h.view(bf16)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pgemm", description="Build the prefill GEMM")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(pgemm, opts, compile_kwargs={},
                   device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
