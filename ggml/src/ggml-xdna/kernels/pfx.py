#!/usr/bin/env python3
# pfx.py -*- Python -*-
#
# One prefill design for EmbeddingGemma 2: the prefill GEMM of pgemm.py on all
# 32 cores, the prefill attention of attn-mm.cc on six columns and the row ops
# (norms, rope, residual adds, GEGLU) on the other two, each a mode of the
# cores' one program, so a layer runs without a hardware-context switch.
#
#   A core's program is pgemm's core loop, then the attention or the row-op
#   loop. Header words 0-7 are pgemm's; word 8 gates the attention (its calls),
#   word 9 the row command (a row-op core's C objects), word 10 in an attention
#   command the A objects the row-op cores let pass, word 11 in a row command
#   the A objects the attention cores let pass, word 12 a row-op core's setup
#   objects, word 13 in a GEMM the x objects a block takes on W before its K
#   loop, as the accumulator's up half (GLU_X: the GLU of the gate and x, the
#   weight's up half zero). pgemm's host writes 0 there, so the backend runs
#   its GEMMs through pgemm's runner, with the W path's pair ids set, when
#   GGML_XDNA_PFX=1 loads this artifact (pfx_c8) in pgemm_c8's place.
#   One program for every mode does not fit a core's 16 KB of program memory
#   (probes/pfx_s1.py), so the row-op columns are whole columns: the attention
#   pairs rows (2, 3) and (4, 5) of a column.
#   The GEMM keeps the tile formats and outputs the model uses: 8-bit tiles,
#   plain f32 and the bf16 GLU into the next projection's A.
#
#   python3 pfx.py -d npu2 --xclbin-path build/bin/pfx_c8.xclbin --insts-path build/bin/pfx_c8.insts.bin
#
# Data paths: pgemm's (see pgemm.py), the W path packet-switched. A0 reaches
# every pair's first core and A1 its second, the cores of head-dim halves a and
# b, so the attention's K and V^T tiles of one KV head come on A for every pair
# at once; each core's Q comes on W under its own packet id. O accumulates in
# the GEMM's accumulator and leaves as its two column halves, two C objects a
# pass; the pair's score halves cross through the GEMM's neighbour fifos.
# A row command's blocks come on W under each row-op core's own packet id, one
# head of 8 positions, and their cos and sin on the A broadcast.
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

import kernelsrc
import pgemm as pg
import attn_mm as am

import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

COLS, MB, KS, NC, HDR = pg.COLS, pg.MB, pg.KS, pg.NC, pg.HDR
HDR_WORDS = 16
FA_WORDS = 8
ATTN_ON, ROW_ON = 8, 9                  # header words that gate the modes
A_DRAIN = 10                            # attention commands: A objects a row-op core lets pass (row ops are off)
ATTN_DRAIN = 11                         # row commands: A objects an attention core lets pass
ROW_SETUP = 12                          # row commands: setup objects a row-op core takes on W
X_ON = 13                               # GEMMs: x objects a block takes on W (0 or 1)
ROW_BLOCKS = 2                          # row blocks a C object
ROW_COLS = (3, 7)                       # the row-op columns
STACK = {"attn": 0xE00, "row": 0xB00}
# PFX_STACK_PROBE=1 builds every core with a stack too small, so aiecc's check reports what each one needs.
STACK_PROBE = 0x100 if os.environ.get("PFX_STACK_PROBE") == "1" else 0

bf16 = pg.bf16
t_ty, a_ty, b_ty, acc_ty, c_ty, c4_ty = pg.t_ty, pg.a_ty, pg.b_ty, pg.acc_ty, pg.c_ty, pg.c4_ty
n_ty = np.ndarray[(HDR_WORDS,), np.dtype[np.int32]]
nfa_ty = np.ndarray[(FA_WORDS,), np.dtype[np.int32]]
ml_ty = am.ml_ty

_here = Path(__file__).resolve().parent

# attn-mm.cc includes IRON's mm.cc, whose extern "C" wrappers the GEMM's mm object defines too. The attention calls none
# of them, so its copies are renamed apart and the linker drops them.
ATTN_MM_RENAMES = ["-Dmatmul_scalar_bf16_f32=pfx_fa_matmul_scalar_bf16_f32", "-Dzero_f32=pfx_fa_zero_f32",
                   "-Dzero_scalar_f32=pfx_fa_zero_scalar_f32"]
# The tile's exponents go in the unused half of the core's own score object (the GEMM's B buffer, twice the scores).
ATTN_FLAGS = ["-DFA_X_IN_SCORES"]

Q4_EXPAND = """            if (q8) {
                expand_group8<true>(blk, d8, d8 + LANE, sup, out, gs * GROUP);
            } else {
                expand_group8<false>(blk, d8, d8 + LANE, sup, out, gs * GROUP);
            }
"""
Q8_EXPAND = """            expand_group8<true>(blk, d8, d8 + LANE, sup, out, gs * GROUP);
"""
F32_GLU = """    if (mode == 2) {
        aie::set_rounding(aie::rounding_mode::conv_even);
        silu_mul<true>(acc, out, gelu);
    } else {
        silu_mul<false>(acc, out, gelu);
    }
"""
BF16_GLU = """    aie::set_rounding(aie::rounding_mode::conv_even);
    silu_mul<true>(acc, out, gelu);
"""
HDR_COPY_PG = """    for (int i = 0; i < 8; i++) {
        cnt[i] = h[i];"""
HDR_COPY_PFX = f"""    for (int i = 0; i < {HDR_WORDS}; i++) {{
        cnt[i] = h[i];"""


def replace_once(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"expected one occurrence of: {old.splitlines()[0]}"
    return src.replace(old, new)


def gemm_source() -> str:
    """pgemm's expander for 8-bit tiles, with the bf16 GLU output only and the whole header."""
    src = replace_once(pg._src, Q4_EXPAND, Q8_EXPAND)
    return replace_once(replace_once(src, F32_GLU, BF16_GLU), HDR_COPY_PG, HDR_COPY_PFX)


def external(src: str, flags: list, tag: str):
    obj = tag + "_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
    return lambda name, types: ExternalFunction(name, object_file_name=obj, source_string=src, arg_types=types,
                                                include_dirs=_include_dirs(), compile_flags=flags)


def kernels() -> dict:
    gemm = external(gemm_source(), pg._flags + ["-DPG_LOAD_X"], "pfxexp")
    attn = external(am._src, am._flags + ATTN_MM_RENAMES + ATTN_FLAGS, "pfxatt")
    row = external(kernelsrc.load(_here / "rowop.cc"), [], "pfxrow")
    mm = akernels.mm(MB, KS, NC, input_dtype=bf16, output_dtype=np.float32, vectorized=True,
                     emulate_bf16_mmul_with_bfp16=True)
    return dict(
        mm=mm, zero=mm.zero,
        hdr=gemm("ggml_xdna_pg_hdr_p", [a_ty, n_ty, np.int32]),
        exp=gemm("ggml_xdna_expand_pair", [t_ty, b_ty, n_ty]),
        out=gemm("ggml_xdna_pg_out", [acc_ty, c_ty, n_ty]),
        load_x=gemm("ggml_xdna_pg_load_x", [t_ty, acc_ty]),
        fa_hdr=attn("fa_hdr", [a_ty, nfa_ty]),
        fa_pass=attn("fa_pass", [acc_ty, ml_ty, nfa_ty, t_ty]),
        fa_scores=attn("fa_scores", [t_ty, a_ty, b_ty]),
        fa_update=attn("fa_update_p", [b_ty, b_ty, acc_ty, c_ty, ml_ty]),
        fa_pv=attn("fa_pv", [c_ty, a_ty, acc_ty]),
        fa_end=attn("fa_end", [acc_ty, ml_ty]),
        row_setup=row("ggml_xdna_row_setup", [t_ty, acc_ty]),
        row=row("ggml_xdna_rowop", [t_ty, a_ty, c_ty, acc_ty]),
    )


def role_of(col: int) -> str:
    return "row" if col in ROW_COLS else "attn"


def stack_of(role: str) -> int:
    return STACK_PROBE or STACK[role]


def gemm_mode(p):
    """pgemm's core loop."""
    a_in, x_own, x_nb, wb, wp, wc, cnt, acc = (p["a_in"], p["x_own"], p["x_nb"], p["wb"], p["wp"], p["wc"], p["cnt"],
                                               p["acc"])

    def pair(release_w: bool):
        xo = x_own.acquire(1)
        p["exp"](wb, xo, cnt)
        if release_w:
            wp.release(1)
        aa = a_in.acquire(1)
        p["mm"](aa, xo, acc)
        a_in.release(1)
        x_own.release(1)
        xn = x_nb.acquire(1)
        aa = a_in.acquire(1)
        p["mm"](aa, xn, acc)
        a_in.release(1)
        x_nb.release(1)

    for _ in range_(cnt[0]):
        p["zero"](acc)
        for _ in range_(cnt[X_ON]):
            wc.acquire(1)
            p["load_x"](wb, acc)
            wp.release(1)
        for _ in range_(cnt[1]):
            wc.acquire(1)
            for _ in range_(cnt[7]):
                pair(False)
            # the tile's last pair: the tile goes back once expanded
            pair(True)
        for _ in range_(cnt[5]):
            cc = p["c_out"].acquire(1)
            p["out"](acc, cc, cnt)
            p["c_out"].release(1)


def attn_mode(p):
    """attn_mm's core loop on pfx's paths: a pass's Q in the W buffer, its K and V^T tiles on A. The weights P live in
    the pass's first C object until O replaces them."""
    a_in, x_own, x_nb, wb, wp, wc, acc, ml, cfa, c_out = (p["a_in"], p["x_own"], p["x_nb"], p["wb"], p["wp"], p["wc"],
                                                          p["acc"], p["ml"], p["cfa"], p["c_out"])
    for _ in range_(p["cnt"][ATTN_ON]):
        hq = a_in.acquire(1)
        p["fa_hdr"](hq, cfa)
        a_in.release(1)
        for _ in range_(cfa[0]):
            wc.acquire(1)
            pb = c_out.acquire(1)
            p["fa_pass"](acc, ml, cfa, wb)
            for _ in range_(cfa[1]):
                kk = a_in.acquire(1)
                so = x_own.acquire(1)
                p["fa_scores"](wb, kk, so)
                a_in.release(1)
                x_own.release(1)
                sn = x_nb.acquire(1)
                p["fa_update"](so, sn, acc, pb, ml)
                x_nb.release(1)
                vv = a_in.acquire(1)
                p["fa_pv"](pb, vv, acc)
                a_in.release(1)
            wp.release(1)
            p["fa_end"](acc, ml)
            # O half^T (128 x 32 f32) is the accumulator's two column halves: two C objects
            p["out"](acc, pb, p["cnt"])
            c_out.release(1)
            cc = c_out.acquire(1)
            p["out"](acc, cc, p["cnt"])
            c_out.release(1)


def drain_a(p, word: int):
    """Header word `word`'s count of A objects let pass: the A broadcast reaches every core of a parity."""
    for _ in range_(p["cnt"][word]):
        p["a_in"].acquire(1)
        p["a_in"].release(1)


def attn_drain(p):
    """A row-op core's part of an attention command."""
    drain_a(p, A_DRAIN)


def attn_tail_drain(p):
    """An attention core's part of a row command: the blocks' cos and sin objects."""
    drain_a(p, ATTN_DRAIN)


def row_mode(p):
    """The row command: the setup object on W, then a block a step, its rows on W and its cos and sin on A, into
    half of a C object."""
    a_in, wb, wp, wc, acc = p["a_in"], p["wb"], p["wp"], p["wc"], p["acc"]
    for _ in range_(p["cnt"][ROW_SETUP]):
        wc.acquire(1)
        p["row_setup"](wb, acc)
        wp.release(1)
    for _ in range_(p["cnt"][ROW_ON]):
        cc = p["c_out"].acquire(1)
        for _ in range_(ROW_BLOCKS):
            tt = a_in.acquire(1)
            wc.acquire(1)
            p["row"](wb, tt, cc, acc)
            wp.release(1)
            a_in.release(1)
        p["c_out"].release(1)


GEMM_ARGS = ["wb", "wp", "wc", "a_in", "c_out", "x_own", "x_nb", "zero", "mm", "hdr", "exp", "out", "load_x", "acc",
             "cnt", "par"]
ROLE_ARGS = {"attn": ["fa_hdr", "fa_pass", "fa_scores", "fa_update", "fa_pv", "fa_end", "ml", "cfa"],
             "row": ["row_setup", "row"]}
ROLE_MODES = {"attn": [attn_mode, attn_tail_drain], "row": [attn_drain, row_mode]}


def make_core(role: str):
    names = GEMM_ARGS + ROLE_ARGS[role]

    def core(*args):
        p = dict(zip(names, args))
        h = p["a_in"].acquire(1)
        p["hdr"](h, p["cnt"], p["par"])
        p["a_in"].release(1)
        gemm_mode(p)
        for mode in ROLE_MODES[role]:
            mode(p)

    return core, names


def a_fifos():
    """pgemm's A streams: the host's row-major (64 x 64) K steps, tiled into the mmul's sub-tiles by the MemTile."""
    a_src = [ObjectFifo(a_ty, name=f"as{mb}", depth=2) for mb in range(2)]
    a_f = [a_src[mb].cons().forward(tile=Tile(6 + mb, 1), obj_type=a_ty, depth=2, name=f"a{mb}",
                                    dims_to_stream=[(MB // 8, 8 * KS), (KS // 8, 8), (8, KS), (8, 1)])
           for mb in range(2)]
    return a_src, a_f


def c_fifos():
    """pgemm's C streams: a column's four cores joined by the MemTile into plain rows."""
    c_col, c_core = [], []
    for col in range(COLS):
        f = ObjectFifo(c4_ty, name=f"c{col}", depth=1)
        c_col.append(f)
        c_core.append(f.prod().join(offsets=[i * MB * NC // 2 for i in range(4)], obj_types=[c_ty] * 4,
                                    depths=[1] * 4, names=[f"c{col}_{i}" for i in range(4)], tile=Tile(col, 1),
                                    dims_from_stream=[[(MB // 8, 8 * NC // 2), (NC // 16, 8), (8, NC // 2),
                                                       (8, 1)]] * 4))
    return c_col, c_core


def core_args(k, col, i, w_core, a_f, c_core, x, role):
    r, mb = pg.core_of(i)
    wb, wp, wc = w_core[col][i]
    v = dict(k, wb=wb, wp=wp, wc=wc, a_in=a_f[mb].cons(), c_out=c_core[col][i].prod(), x_own=x[r][mb].prod(),
             x_nb=x[r][1 - mb].cons(), acc=Buffer(acc_ty, name=f"acc{col}_{i}"),
             cnt=Buffer(n_ty, name=f"n{col}_{i}"), par=mb)
    if role == "attn":
        v.update(ml=Buffer(ml_ty, name=f"ml{col}_{i}"), cfa=Buffer(nfa_ty, name=f"fa{col}_{i}"))
    return v


def build(dev_name: str = "npu2"):
    """The workers and the object fifos' endpoints, as pgemm.build: a (mb 0, 1), c (a column each)."""
    k = kernels()
    flows, locks, dmas = [], [], []
    w_core = []
    for col in range(COLS):
        f, lk, d, per = pg.w_path_column(col, w_packet=True)
        flows += f
        locks += lk
        dmas += d
        w_core.append(per)
    a_src, a_f = a_fifos()
    c_col, c_core = c_fifos()
    workers = []
    for col in range(COLS):
        role = role_of(col)
        core, names = make_core(role)
        x = [[ObjectFifo(b_ty, name=f"x{col}_{r}_{q}", depth=1) for q in range(2)] for r in range(2)]
        for i in range(4):
            v = core_args(k, col, i, w_core, a_f, c_core, x, role)
            workers.append(Worker(core, fn_args=[v[n] for n in names], tile=Tile(col, 2 + i),
                                  stack_size=stack_of(role)))
    eps = ([f.prod(tile=Tile(6 + mb, 0)) for mb, f in enumerate(a_src)]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(c_col)])
    return workers, eps, (flows, locks, dmas)


@iron.jit
def pfx(w: In, a: In, c: Out, *, M: CompileTime[int] = 512, K: CompileTime[int] = 1024, N: CompileTime[int] = 2048,
        dev_name: CompileTime[str] = "npu2"):
    # the sample stream is pgemm's own for the same shape (8-bit tiles, plain output)
    return pg.program_of(build(dev_name), M=M, K=K, N=N, Q8=1, GLU=0, dev_name=dev_name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pfx", description="Build the merged prefill design")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(pfx, opts, compile_kwargs={}, device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
