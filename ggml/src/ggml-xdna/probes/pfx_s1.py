#!/usr/bin/env python3
# Program-memory spike for one merged prefill design: pgemm's data paths and core
# loop, with the attention (attn-mm.cc) and the row ops (rowop-s1.cc) as
# header-gated modes. Each column pair takes one role set, so one build sizes
# them all:
#
#   columns 0-1  GEMM, attention, row ops
#   columns 2-3  GEMM, attention
#   columns 4-5  GEMM, row ops
#   columns 6-7  GEMM
#
# The GEMM keeps only what EmbeddingGemma 2's Q8_0 file uses: the 8-bit
# expander and the bf16 GLU output. The W tile and C object buffers are shrunk
# so the cores fit their data memory (that is the L1 spike's question); a
# buffer's size changes no instruction. The stream only exists because the
# artifact needs one, and does not run.
#
#   python3 probes/pfx_s1.py -d npu2 --xclbin-path pfx_s1.xclbin --insts-path pfx_s1.insts.bin
#   python3 probes/xclbin_core_programs.py pfx_s1.xclbin
#
# Columns 0-1 carry all three modes unless PFX_S1_ALL_MODES=0.
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "kernels"))

import kernelsrc  # noqa: E402
import pgemm as pg  # noqa: E402
import attn_mm as am  # noqa: E402

import aie.iron as iron  # noqa: E402
import aie.iron.kernels as akernels  # noqa: E402
from aie.iron import Buffer, CompileTime, In, Lock, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.dataflow import Flow, PacketFlow, TileDma  # noqa: E402
from aie.iron.dataflow.tile_dma import Acquire, Bd, DmaChannel, Release  # noqa: E402
from aie.dialects._aie_enum_gen import DMAChannelDir, WireBundle  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import Tile, from_name  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.iron.kernels._common import _include_dirs  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402
from aie.utils.hostruntime.argparse import add_compile_args  # noqa: E402
from aie.utils.hostruntime.cli import run_design_cli  # noqa: E402

COLS, MB, KS, NC, HDR = pg.COLS, pg.MB, pg.KS, pg.NC, pg.HDR
WB_BYTES = 1024                 # the W tile buffer, shrunk from pgemm's TB
C_ELEMS = 512                   # a C object, shrunk from 64 x 32
HDR_WORDS = 16
FA_WORDS = 8
STACK_GEMM = 0x800              # pgemm's
STACK_ATTN = 0xE00              # attn_mm's
STACK_ROW = 0xB00               # the row ops need 2752 B (aiecc's stack check)
ATTN_ON, ROW_ON = 8, 9          # header words that gate the modes
# PFX_S1_ALL_MODES=0 gives columns 0-1 GEMM and attention: a core that overflows stops the build before the others link.
ALL_MODES = os.environ.get("PFX_S1_ALL_MODES", "1") != "0"
ROLES = {0: (True, ALL_MODES), 1: (True, ALL_MODES), 2: (True, False), 3: (True, False),
         4: (False, True), 5: (False, True), 6: (False, False), 7: (False, False)}
# PFX_S1_TRIM=1: no GLU epilogue in the GEMM (GEGLU runs as a row op) and the header is pgemm's own kernel copying all
# HDR_WORDS words. PFX_S1_ROW_OPT adds one flag to the row ops' build (e.g. -Os).
TRIM = os.environ.get("PFX_S1_TRIM", "0") == "1"
ROW_OPT = [os.environ["PFX_S1_ROW_OPT"]] if os.environ.get("PFX_S1_ROW_OPT") else []

bf16 = pg.bf16
a_ty, b_ty, acc_ty = pg.a_ty, pg.b_ty, pg.acc_ty
wb_ty = np.ndarray[(WB_BYTES,), np.dtype[np.uint8]]
c_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]
c4_ty = np.ndarray[(4 * C_ELEMS,), np.dtype[np.float32]]
n_ty = np.ndarray[(HDR_WORDS,), np.dtype[np.int32]]
nfa_ty = np.ndarray[(FA_WORDS,), np.dtype[np.int32]]
pb_ty, ml_ty = am.p_ty, am.ml_ty


# attn-mm.cc includes IRON's mm.cc, whose extern "C" wrappers the GEMM's mm object defines too. The attention calls none
# of them, so its copies are renamed apart and the linker drops them.
ATTN_MM_RENAMES = ["-Dmatmul_scalar_bf16_f32=pfx_fa_matmul_scalar_bf16_f32", "-Dzero_f32=pfx_fa_zero_f32",
                   "-Dzero_scalar_f32=pfx_fa_zero_scalar_f32"]


def replace_once(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"expected one occurrence of: {old.splitlines()[0]}"
    return src.replace(old, new)


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
GLU_TAIL = """    const bool gelu = (cnt[6] & 4) != 0;
""" + BF16_GLU
HDR_COPY_PG = """    for (int i = 0; i < 8; i++) {
        cnt[i] = h[i];"""
HDR_COPY_PFX = f"""    for (int i = 0; i < {HDR_WORDS}; i++) {{
        cnt[i] = h[i];"""


def gemm_source() -> str:
    """pgemm's expander with only the 8-bit tiles and the bf16 GLU output; trimmed, no GLU and the merged header."""
    src = replace_once(replace_once(pg._src, Q4_EXPAND, Q8_EXPAND), F32_GLU, BF16_GLU)
    if TRIM:
        src = replace_once(replace_once(src, GLU_TAIL, ""), HDR_COPY_PG, HDR_COPY_PFX)
    return src


def external(src: str, flags: list, tag: str):
    obj = tag + "_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
    return lambda name, types: ExternalFunction(name, object_file_name=obj, source_string=src, arg_types=types,
                                                include_dirs=_include_dirs(), compile_flags=flags)


def kernels() -> dict:
    gemm = external(gemm_source(), pg._flags, "pfxexp")
    hdr = external(kernelsrc.load(_here / "pfx-hdr-s1.cc"), [], "pfxhdr")
    attn = external(am._src, am._flags + ATTN_MM_RENAMES, "pfxatt")
    row = external(kernelsrc.load(_here / "rowop-s1.cc"), ROW_OPT, "pfxrow")
    mm = akernels.mm(MB, KS, NC, input_dtype=bf16, output_dtype=np.float32, vectorized=True,
                     emulate_bf16_mmul_with_bfp16=True)
    return dict(
        mm=mm,
        hdr=gemm("ggml_xdna_pg_hdr_p", [a_ty, n_ty, np.int32]) if TRIM else hdr("pfx_hdr", [a_ty, n_ty, np.int32]),
        exp=gemm("ggml_xdna_expand_pair", [wb_ty, b_ty, n_ty]),
        out=gemm("ggml_xdna_pg_out", [acc_ty, c_ty, n_ty]),
        fa_hdr=attn("fa_hdr", [a_ty, nfa_ty]),
        fa_pass=attn("fa_pass", [acc_ty, ml_ty, nfa_ty, a_ty]),
        fa_scores=attn("fa_scores", [a_ty, wb_ty, b_ty]),
        fa_update=attn("fa_update_p", [b_ty, b_ty, acc_ty, pb_ty, ml_ty]),
        fa_pv=attn("fa_pv", [pb_ty, wb_ty, acc_ty]),
        fa_end=attn("fa_end", [acc_ty, ml_ty]),
        row=row("ggml_xdna_rowop", [a_ty, wb_ty, c_ty, n_ty]),
    )


def gemm_mode(p):
    """pgemm's core loop, word for word."""
    a_in, x_own, x_nb, wb, cnt, acc = p["a_in"], p["x_own"], p["x_nb"], p["wb"], p["cnt"], p["acc"]
    wp, wc = p["wp"], p["wc"]
    mul, kexp = p["mm"], p["exp"]

    def pair(release_w: bool):
        xo = x_own.acquire(1)
        kexp(wb, xo, cnt)
        if release_w:
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

    for _ in range_(cnt[0]):
        p["zero"](acc)
        for _ in range_(cnt[1]):
            wc.acquire(1)
            for _ in range_(cnt[7]):
                pair(False)
            pair(True)
        for _ in range_(cnt[5]):
            cc = p["c_out"].acquire(1)
            p["out"](acc, cc, cnt)
            p["c_out"].release(1)


def attn_mode(p):
    """attn_mm's core loop on pgemm's paths: Q on A, K and V^T through the W tile, O in the accumulator, out as C."""
    a_in, x_own, x_nb, wp, wc, wb, acc, ml, pb, cfa = (p["a_in"], p["x_own"], p["x_nb"], p["wp"], p["wc"], p["wb"],
                                                       p["acc"], p["ml"], p["pb"], p["cfa"])
    for _ in range_(p["cnt"][ATTN_ON]):
        hq = a_in.acquire(1)
        p["fa_hdr"](hq, cfa)
        a_in.release(1)
        for _ in range_(cfa[0]):
            qq = a_in.acquire(1)
            p["fa_pass"](acc, ml, cfa, qq)
            for _ in range_(cfa[1]):
                wc.acquire(1)
                so = x_own.acquire(1)
                p["fa_scores"](qq, wb, so)
                wp.release(1)
                x_own.release(1)
                sn = x_nb.acquire(1)
                p["fa_update"](so, sn, acc, pb, ml)
                x_nb.release(1)
                wc.acquire(1)
                p["fa_pv"](pb, wb, acc)
                wp.release(1)
            p["fa_end"](acc, ml)
            for _ in range_(cfa[2]):
                cc = p["c_out"].acquire(1)
                p["out"](acc, cc, p["cnt"])
                p["c_out"].release(1)
            a_in.release(1)


def row_mode(p):
    """One block of rows a step: the rows on A, the second input on the W tile, the bf16 result as C."""
    for _ in range_(p["cnt"][ROW_ON]):
        aa = p["a_in"].acquire(1)
        p["wc"].acquire(1)
        cc = p["c_out"].acquire(1)
        p["row"](aa, p["wb"], cc, p["cnt"])
        p["c_out"].release(1)
        p["wp"].release(1)
        p["a_in"].release(1)


GEMM_ARGS = ["wb", "wp", "wc", "a_in", "c_out", "x_own", "x_nb", "zero", "mm", "hdr", "exp", "out", "acc", "cnt", "par"]
ATTN_ARGS = ["fa_hdr", "fa_pass", "fa_scores", "fa_update", "fa_pv", "fa_end", "pb", "ml", "cfa"]
ROW_ARGS = ["row"]


def arg_names(attn: bool, row: bool) -> list:
    return GEMM_ARGS + (ATTN_ARGS if attn else []) + (ROW_ARGS if row else [])


def stack_of(attn: bool, row: bool) -> int:
    if attn:
        return STACK_ATTN
    return STACK_ROW if row else STACK_GEMM


def make_core(attn: bool, row: bool):
    names = arg_names(attn, row)

    def core(*args):
        p = dict(zip(names, args))
        h = p["a_in"].acquire(1)
        p["hdr"](h, p["cnt"], p["par"])
        p["a_in"].release(1)
        gemm_mode(p)
        if attn:
            attn_mode(p)
        if row:
            row_mode(p)

    return core


def w_path():
    """pgemm's explicit W path with the shrunk tile buffer: (flows, locks, dmas) and each core's (wb, wp, wc)."""
    flows, locks, dmas, w_core = [], [], [], []
    for col in range(COLS):
        mt, sh = Tile(col, 1), Tile(col, 0)
        dmas.append(pg._Hold(mt, [Buffer(pg.ws_ty, name=f"ws{col}", tile=mt, address=pg.WS_ADDR)]))
        for k in range(4):
            locks.append(Lock(mt, lock_id=pg.WLOCK0 + k, init=0, name=f"wl{col}_{k}"))
        flows.append(Flow(sh, mt, src_channel=pg.W_SHIM_CH, dst_channel=pg.W_MT_S2MM))
        flows.append(PacketFlow(pg.MT_CTRL_PKT, mt, sh, src_port=WireBundle.TileControl, src_channel=0,
                                dst_port=WireBundle.South, dst_channel=0, keep_pkt_header=True))
        per = []
        for i in range(4):
            r, _ = pg.core_of(i)
            ct = Tile(col, 2 + i)
            flows.append(Flow(mt, ct, src_channel=pg.W_MT_MM2S + r, dst_channel=pg.W_CORE_CH))
            wb = Buffer(wb_ty, name=f"wb{col}_{i}", tile=ct)
            wp = Lock(ct, init=1, name=f"wp{col}_{i}")
            wc = Lock(ct, init=0, name=f"wc{col}_{i}")
            locks += [wp, wc]
            dmas.append(TileDma(ct, [DmaChannel(DMAChannelDir.S2MM, pg.W_CORE_CH,
                                                [Bd(wb, acquires=[Acquire(wp)], releases=[Release(wc)])])]))
            per.append((wb, wp, wc))
        w_core.append(per)
    return (flows, locks, dmas), w_core


def build():
    k = kernels()
    extra, w_core = w_path()
    a_src = [ObjectFifo(a_ty, name=f"as{mb}", depth=2) for mb in range(2)]
    a_f = [a_src[mb].cons().forward(tile=Tile(6 + mb, 1), obj_type=a_ty, depth=2, name=f"a{mb}")
           for mb in range(2)]
    c_col, c_core = [], []
    for col in range(COLS):
        f = ObjectFifo(c4_ty, name=f"c{col}", depth=1)
        c_col.append(f)
        c_core.append(f.prod().join(offsets=[i * C_ELEMS for i in range(4)], obj_types=[c_ty] * 4,
                                    depths=[1] * 4, names=[f"c{col}_{i}" for i in range(4)], tile=Tile(col, 1)))
    workers = []
    for col in range(COLS):
        attn, row = ROLES[col]
        x = [[ObjectFifo(b_ty, name=f"x{col}_{r}_{q}", depth=1) for q in range(2)] for r in range(2)]
        for i in range(4):
            r, mb = pg.core_of(i)
            wb, wp, wc = w_core[col][i]
            v = dict(wb=wb, wp=wp, wc=wc, a_in=a_f[mb].cons(), c_out=c_core[col][i].prod(),
                     x_own=x[r][mb].prod(), x_nb=x[r][1 - mb].cons(), zero=k["mm"].zero, mm=k["mm"],
                     hdr=k["hdr"], exp=k["exp"], out=k["out"], acc=Buffer(acc_ty, name=f"acc{col}_{i}"),
                     cnt=Buffer(n_ty, name=f"n{col}_{i}"), par=mb, row=k["row"])
            if attn:
                v.update({n: k[n] for n in ATTN_ARGS[:6]}, pb=Buffer(pb_ty, name=f"pb{col}_{i}"),
                         ml=Buffer(ml_ty, name=f"ml{col}_{i}"), cfa=Buffer(nfa_ty, name=f"fa{col}_{i}"))
            workers.append(Worker(make_core(attn, row), fn_args=[v[n] for n in arg_names(attn, row)],
                                  tile=Tile(col, 2 + i), stack_size=stack_of(attn, row)))
    eps = ([f.prod(tile=Tile(6 + mb, 0)) for mb, f in enumerate(a_src)]
           + [f.cons(tile=Tile(col, 0)) for col, f in enumerate(c_col)])
    return workers, eps, extra


@iron.jit
def pfx_s1(w: In, a: In, c: Out, *, dev_name: CompileTime[str] = "npu2"):
    A_ELEMS = 2 * HDR
    C_TOTAL = COLS * 4 * C_ELEMS
    workers, eps, extra = build()

    def seq(w_h, a_h, c_h, *e):
        ai, co = e[:2], e[2:]
        for mb in range(2):
            ai[mb].fill(a_h, tap=TensorAccessPattern([1, A_ELEMS], mb * HDR, [1, HDR], [0, 1]))
        for col in range(COLS):
            co[col].drain(c_h, tap=TensorAccessPattern([1, C_TOTAL], col * 4 * C_ELEMS, [1, 4 * C_ELEMS], [0, 1]),
                          wait=True)

    rt = Runtime(seq, [np.ndarray[(COLS * 2 * WB_BYTES,), np.dtype[np.uint8]],
                       np.ndarray[(A_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(C_TOTAL,), np.dtype[np.float32]]] + eps)
    pg.add_explicit(rt, extra)
    return Program(from_name(dev_name, n_cols=COLS), rt, workers=workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(prog="pfx_s1", description="Program-memory spike of the merged prefill design")
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(pfx_s1, opts, compile_kwargs={}, device=lambda o: from_name(o.dev, n_cols=COLS))


if __name__ == "__main__":
    main()
