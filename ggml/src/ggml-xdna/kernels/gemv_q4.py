#!/usr/bin/env python3
"""Decode GEMV over the packed NPU weight formats, with the group parameters
applied to the accumulator (gemv_q4.cc).

  rm -rf ~/.npu/cache && python3 gemv_q4.py -d npu2 --fmt q4g32
  rm -rf ~/.npu/cache && python3 gemv_q4.py -d npu2 --fmt q8g16 -K 2048 -N 4096 --n-core 64

The q8g16 line pins --n-core 64: the shape-derived 128 doubles the weight tile,
and its double buffering plus the activation tiles then overruns the 56 KB L1
budget.

The core program depends only on (format, N_CORE, K_TILE), so one artifact per
format serves every shape and the backend builds its own instruction stream
(xdna_gemv_seq_build) instead of this one. The artifact is what matters at
runtime; the stream IRON generates here only serves the standalone check below.

The per-dispatch counts - K tiles and output chunks - arrive as the first
object of the activation stream, not through a runtime-parameter register. A
register is written by the instruction stream while the core is already
running, and a core that does not stop at a barrier reads it before the write
lands: with a register the second dispatch of a different shape ran with the
previous one's tile count. An object the core acquires is ordered against the
DMA by construction.

The cache wipe is not optional while editing the kernel: the compiled object is
keyed by the design, not by the source text, so an edited .cc is silently
ignored and the old object is relinked.

Each core owns N/(cols*rows) columns and streams the whole K in K_TILE chunks,
accumulating in f32. The weight stream is ordered [col][tile][row] so a column
reads one contiguous run, and the activation tile is broadcast to the four
cores of a column.
"""

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
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.iron.device import from_name, Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

from wfmt import Q4_GROUP, split_pairs

# The decode GEMV's 8-bit groups are 32 values (the prologue's tile cache
# is 64-byte strided for them), not wfmt's 16 (the prefill's
# dequant GEMM, which does not use them): every type it packs has per-32
# parameters but Q6_K, and the f32 rescale a group costs is half the kernel's
# arithmetic on that form, so a group of 16 paid it twice for nothing on Q5_K.
Q8_GROUP = 32


def group_size(fmt: str) -> int:
    return Q4_GROUP if fmt == "q4g32" else Q8_GROUP

CORES_PER_COL = 4
# Which compute rows of a column the design places its cores on. Rows 2..5 are
# the compute rows of an AIE2P column (0 is shim, 1 is mem). The default takes
# all of them; a per-column list lets the design sit alongside another one in
# the same array - the fused layer's GDN core occupies one row per column, and
# which row that is differs across the columns.
ROWS_ALL = "2,3,4,5"
# Columns one multiply covers. AIE2P's int8 datapath is 64 lanes wide, so a
# core at least that many columns wide builds at 64 and does twice the work per
# multiply; a narrower one has to stay at 32.
def vec_for(n_core: int) -> int:
    return 64 if n_core >= 64 else 32


VEC = 32
L1_BUDGET = 56 * 1024

# One artifact serves both code widths: a 4-bit tile spans twice the K of an
# 8-bit one, so both are the same number of bytes and the same design streams
# them. The activation tile is sized for the wider one and carries the width in
# its last word.
K_TILES = {"q4g32": 512, "q8g16": 256}
ACT_TILE = 2112


def tile_bytes(fmt: str, k_tile: int, n_core: int) -> int:
    vec = vec_for(n_core)
    """Packed size of one (k_tile x n_core) weight tile. Equal for both code
    widths by construction, which is what lets one design stream either."""
    lg = n_core // vec
    ng = k_tile // group_size(fmt)
    # A block is codes plus this group's int8 scale and min per column; the
    # bf16 pair the two of them scale is per super-block and sits after every
    # block of the tile. A q4g32 tile spans twice the values of the q8g16 tile
    # it shares an artifact with, so both are given the larger record count and
    # the two sizes stay equal.
    nsup = max(1, (k_tile if fmt == "q4g32" else 2 * k_tile) // 256)
    if fmt != "q4g32":
        # 32-value 8-bit groups make the tile smaller than the q4g32 one; it
        # is padded to that size, the one object size the design streams.
        return tile_bytes("q4g32", 2 * k_tile, n_core)
    code = Q4_GROUP * vec // 2
    return lg * (ng * (code + 4 * vec) + nsup * 4 * vec * 2)




# The epilogue's quantizer (store_quant in gemv-q4.cc) runs in vector lanes:
# its scalar form pulled ~3 KB of software float into a program memory that
# attention mode needs.
# silu is exp from the exponent bits and a polynomial, and it shares its
# reciprocal with the quantizer (recip_f32): 1.6 KB more for the same reason.
def _kernel_flags(fmt: str, k_tile: int, n_core: int) -> list:
    # The kernel's tile size has to be the one the design streams. Taking it
    # from the table instead of the argument silently mismatched whenever
    # --k-tile was given, which reads as a device fault rather than a build
    # mistake.
    kt = {"q4g32": K_TILES["q4g32"], "q8g16": K_TILES["q8g16"]}
    kt[fmt] = k_tile
    return [f"-DK_TILE_Q4={kt['q4g32']}", f"-DK_TILE_Q8={kt['q8g16']}",
             f"-DACT_TILE={ACT_TILE}", f"-DN_CORE={n_core}",
             f"-DQ4_GROUP={Q4_GROUP}", f"-DQ8_GROUP={Q8_GROUP}",
             f"-DGEMV_VEC={vec_for(n_core)}",
             # ACT_RAW builds the per-tile quantizer into the core (gemv_q4.cc),
             # so a producer on the array can hand this dispatch numbers rather
             # than codes. It costs about a kilobyte of a program memory that is
             # already full, so it is off unless a design asks for it.
             f"-DACT_RAW={int(__import__('os').environ.get('ACT_RAW', '0'))}"]


# The prologue tile's own streams (act-att.cc): a side input from DDR and an
# output to DDR - the attention layer's work and the layer boundary's rows,
# a row quantized once and its tiles copied, the gates on its last tile -
# output to DDR, on the shim channels the layer leaves free.
PRO_SIDE_TY = np.ndarray[(512,), np.dtype[np.int32]]
PRO_EMIT_TY = np.ndarray[(256,), np.dtype[np.int32]]
PRO_CNT_TY = np.ndarray[(2,), np.dtype[np.int32]]
PRO_SIDE_COL = 6
PRO_EMIT_COL = 5


def _kernels(fmt: str, k_tile: int, n_core: int):
    here = Path(__file__).resolve().parent
    raw = (here / "gemv-q4.cc").read_text()
    src = kernelsrc.inline(raw)
    flags = _kernel_flags(fmt, k_tile, n_core)

    # The prologue is a build of its own: its quantizer is about a kilobyte of
    # program memory and the GEMV cores have none spare, so it is linked into
    # the tile that runs it and nothing else. ACT_PRO in the flags is what
    # keeps the two objects apart (ExternalFunction names an object by its
    # digest, and the digest covers the flags).
    pro_flags = [f for f in flags if not f.startswith("-DACT_RAW")] + \
                ["-DACT_RAW=0", "-DACT_PRO=1"]
    digest = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    # The attention layer's prologue work (act-att.cc) shares the object. The
    # two are inlined together, so a header they both include lands once.
    psrc = kernelsrc.inline(raw + "\n" + (here / "act-att.cc").read_text())
    pdigest = hashlib.sha256((psrc + "\0".join(pro_flags)).encode()).hexdigest()[:8]
    w_ty = np.ndarray[(tile_bytes(fmt, k_tile, n_core),), np.dtype[np.uint8]]
    a_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    o_ty = np.ndarray[(n_core,), np.dtype[np.float32]]
    a_raw_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    pmk = lambda name, tys: ExternalFunction(
        name, object_file_name=f"actpro_{n_core}_{pdigest}.o",
        source_string=psrc, arg_types=tys, compile_flags=pro_flags)
    pro_tile = (pmk("ggml_xdna_act_pro", [a_raw_ty, a_raw_ty, a_raw_ty]),
                pmk("ggml_xdna_act_cnt", [a_raw_ty, PRO_CNT_TY]),
                pmk("ggml_xdna_act_side", [PRO_SIDE_TY, a_raw_ty]),
                pmk("ggml_xdna_act_emit", [PRO_EMIT_TY, a_raw_ty]))
    gemv = ExternalFunction(
        "ggml_xdna_gemv",
        object_file_name=f"gemv_{n_core}_{digest}.o",
        source_string=src,
        arg_types=[w_ty, a_ty, o_ty],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )
    zsrc = (Path(__file__).resolve().parent / "gemv-zero.cc").read_text()
    zdigest = hashlib.sha256((zsrc + "\0".join(flags)).encode()).hexdigest()[:8]
    zero = ExternalFunction(
        "ggml_xdna_gemv_zero",
        object_file_name=f"gemv_zero_{n_core}_{zdigest}.o",
        source_string=zsrc,
        arg_types=[o_ty],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )
    return gemv, zero, pro_tile


def _kernels_lead(fmt: str, k_tile: int, n_core: int):
    """The kernels of the lead core of a two-row pair (build_gemv's PAIR).

    The lead's output object carries both cores' blocks, so its GEMV and zero
    take a buffer twice n_core long. The C code writes n_core values through a
    pointer either way; what differs is the buffer type the design declares,
    and one symbol cannot be declared with two types - so the same source is
    built again under another name."""
    kdir = Path(__file__).resolve().parent
    src = kernelsrc.load(kdir / "gemv-q4.cc")
    zsrc = (kdir / "gemv-zero.cc").read_text()
    msrc = (kdir / "gemv-merge.cc").read_text()
    flags = _kernel_flags(fmt, k_tile, n_core)
    gflags = flags + ["-Dggml_xdna_gemv=ggml_xdna_gemv_lead"]
    zflags = flags + ["-Dggml_xdna_gemv_zero=ggml_xdna_gemv_zero_lead"]
    dg = lambda s_, f_: hashlib.sha256((s_ + "\0".join(f_)).encode()).hexdigest()[:8]
    w_ty = np.ndarray[(tile_bytes(fmt, k_tile, n_core),), np.dtype[np.uint8]]
    a_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    o_ty = np.ndarray[(n_core,), np.dtype[np.float32]]
    o2_ty = np.ndarray[(2 * n_core,), np.dtype[np.float32]]
    gemv_l = ExternalFunction(
        "ggml_xdna_gemv_lead",
        object_file_name=f"gemvl_{n_core}_{dg(src, gflags)}.o",
        source_string=src, arg_types=[w_ty, a_ty, o2_ty],
        include_dirs=_include_dirs(), compile_flags=gflags)
    zero_l = ExternalFunction(
        "ggml_xdna_gemv_zero_lead",
        object_file_name=f"gemv_zerol_{n_core}_{dg(zsrc, zflags)}.o",
        source_string=zsrc, arg_types=[o2_ty],
        include_dirs=_include_dirs(), compile_flags=zflags)
    merge = ExternalFunction(
        "ggml_xdna_gemv_merge",
        object_file_name=f"gemv_merge_{n_core}_{dg(msrc, flags)}.o",
        source_string=msrc, arg_types=[o_ty, o2_ty],
        include_dirs=_include_dirs(), compile_flags=flags)
    return gemv_l, zero_l, merge


# The decode-attention kernels of the pool cores (attn-dec.cc): one object per
# core, so the three functions share its state. The lead's emit writes into
# its two-block output object, the partner's into the hand-over buffer.
AD_PIECES = 33
# att_chunk (mmul) tiles its chunk into a per-core scratch (bank 3) as bf16 for the
# QK and PV matrix multiplies; word 514 of the first q tile skips the
# arithmetic (a diagnostic of the DMA). The eighths of a converted row are
# stored with constant extract indices.
AD_SCR = 2 * 32 * 64


def _kernels_att(fmt: str, k_tile: int, n_core: int, lead: bool):
    kdir = Path(__file__).resolve().parent
    src = kernelsrc.load(kdir / "attn-dec.cc")
    sfx = "_lead" if lead else ""
    flags = [f"-DACT_TILE={ACT_TILE}", f"-DN_CORE={n_core}"]
    if lead:
        flags += ["-Dggml_xdna_att_q=ggml_xdna_att_q_lead",
                  "-Dggml_xdna_att_chunk=ggml_xdna_att_chunk_lead",
                  "-Dggml_xdna_att_emit=ggml_xdna_att_emit_lead"]
    dg = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    obj = f"attdec{sfx}_{n_core}_{dg}.o"
    w_ty = np.ndarray[(tile_bytes(fmt, k_tile, n_core),), np.dtype[np.uint8]]
    a_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    e_ty = np.ndarray[((2 if lead else 1) * n_core,), np.dtype[np.float32]]
    mk = lambda name, tys: ExternalFunction(
        name + sfx, object_file_name=obj, source_string=src, arg_types=tys,
        include_dirs=_include_dirs(), compile_flags=flags)
    s_ty = np.ndarray[(AD_SCR,), np.dtype[ml_dtypes.bfloat16]]
    return (mk("ggml_xdna_att_q", [a_ty]), mk("ggml_xdna_att_chunk", [w_ty, s_ty]),
            mk("ggml_xdna_att_emit", [e_ty]))


def _core_lead_att(w_in, a_in, p_in, out, gemv, zero, merge, att_q, att_chunk, att_emit, scr):
    # _core_lead, and after it the attention mode: header words 2 and 3 are
    # the W objects to take (the core's index object and its chunks) and 1 to
    # run it. A projection's header has both zero, so it never enters here.
    hdr = a_in.acquire(1)
    n_tiles = hdr[0]
    n_out = hdr[1]
    n_att = hdr[2]
    att_on = hdr[3]
    a_in.release(1)
    for _ in range_(n_out):
        o = out.acquire(1)
        zero(o)
        for _ in range_(n_tiles):
            w = w_in.acquire(1)
            a = a_in.acquire(1)
            gemv(w, a, o)
            w_in.release(1)
            a_in.release(1)
        p = p_in.acquire(1)
        merge(p, o)
        p_in.release(1)
        out.release(1)
    for _ in range_(att_on):
        for _ in range_(2):
            a = a_in.acquire(1)
            att_q(a)
            a_in.release(1)
        for _ in range_(n_att):
            w = w_in.acquire(1)
            att_chunk(w, scr)
            w_in.release(1)
        for _ in range_(AD_PIECES):
            o = out.acquire(1)
            att_emit(o)
            p = p_in.acquire(1)
            merge(p, o)
            p_in.release(1)
            out.release(1)


def _core_part_att(w_in, a_in, out, gemv, zero, att_q, att_chunk, att_emit, scr):
    hdr = a_in.acquire(1)
    n_tiles = hdr[0]
    n_out = hdr[1]
    n_att = hdr[2]
    att_on = hdr[3]
    a_in.release(1)
    for _ in range_(n_out):
        o = out.acquire(1)
        zero(o)
        for _ in range_(n_tiles):
            w = w_in.acquire(1)
            a = a_in.acquire(1)
            gemv(w, a, o)
            w_in.release(1)
            a_in.release(1)
        out.release(1)
    for _ in range_(att_on):
        for _ in range_(2):
            a = a_in.acquire(1)
            att_q(a)
            a_in.release(1)
        for _ in range_(n_att):
            w = w_in.acquire(1)
            att_chunk(w, scr)
            w_in.release(1)
        for _ in range_(AD_PIECES):
            o = out.acquire(1)
            att_emit(o)
            out.release(1)


# 3072 bytes of core stack, not the 1024 default. The epilogue keeps a
# 64-float buffer of its result so it can quantize it by groups, and that
# overflows both 1024 and 2048. Neither failure is loud: at 1024 the core stops
# without writing anything and every value reads as zero, at 2048 the result is
# wrong by a factor. The size is worth measuring rather than raising blindly -
# at 4096 the eight-column design loses every odd output chunk to a uniform
# NaN, while 1024, 2048, 3072 and 8192 are exact for it.


def _rows_map(spec: str, cols: int) -> list:
    """Parse a row placement: one comma list for every column, or one per
    column separated by "|". Every column must offer the same number of rows,
    since a column's cores are the unit the output fifo joins."""
    parts = spec.split("|")
    if len(parts) == 1:
        parts = parts * cols
    if len(parts) != cols:
        raise ValueError(f"row map has {len(parts)} columns, need {cols}")
    rows = [[int(r) for r in p.split(",")] for p in parts]
    n = len(rows[0])
    for c, r in enumerate(rows):
        if len(r) != n:
            raise ValueError(f"column {c} places {len(r)} cores, column 0 places {n}")
        if any(x < 2 or x > 5 for x in r) or len(set(r)) != n:
            raise ValueError(f"column {c} rows {r} are not distinct rows in 2..5")
    return rows


def _core(w_in, a_in, out, gemv, zero):
    # The first activation object carries the counts for this dispatch.
    hdr = a_in.acquire(1)
    n_tiles = hdr[0]
    n_out = hdr[1]
    a_in.release(1)
    for _ in range_(n_out):
        o = out.acquire(1)
        zero(o)
        for _ in range_(n_tiles):
            w = w_in.acquire(1)
            a = a_in.acquire(1)
            gemv(w, a, o)
            w_in.release(1)
            a_in.release(1)
        out.release(1)


def _core_lead(w_in, a_in, p_in, out, gemv, zero, merge):
    # The lead of a two-row pair: its own block into the first half of the
    # output object, then the partner's block - handed over through the memory
    # the two tiles share - copied into the second half. One object leaves the
    # column instead of two, which is the point: a MemTile takes four streams
    # from the north, and the column's stages already spend three of them.
    hdr = a_in.acquire(1)
    n_tiles = hdr[0]
    n_out = hdr[1]
    a_in.release(1)
    for _ in range_(n_out):
        o = out.acquire(1)
        zero(o)
        for _ in range_(n_tiles):
            w = w_in.acquire(1)
            a = a_in.acquire(1)
            gemv(w, a, o)
            w_in.release(1)
            a_in.release(1)
        p = p_in.acquire(1)
        merge(p, o)
        p_in.release(1)
        out.release(1)


# The design body, separable from the Program it is wrapped in so the same
# array configuration can be built on its own or alongside another design in
# one xclbin (fused_layer.py). Returns what a Program needs: the workers, the
# runtime argument list and the sequence over it.
def build_gemv(FMT: str, K: int, N: int, K_TILE: int, COLS: int,
               N_CORE: int = 0, ROWS_MAP: str = ROWS_ALL, OUT_GROUP: int = 0,
               PAIR: bool = False, ATT: bool = False):
    # "2,3,4,5" for every column, or one such list per column separated by "|".
    rows_map = _rows_map(ROWS_MAP, COLS)
    ROWS = len(rows_map[0])
    # PAIR: a column's two cores leave it as one stream. The second row's core
    # hands its block to the first through the memory the two tiles share, and
    # the first sends both - the same object, in the same order, a MemTile join
    # of the two rows would make, so the host sees an ordinary two-row pool.
    if PAIR and ROWS != 2:
        raise ValueError(f"PAIR needs two rows a column, got {ROWS}")
    if PAIR and any(abs(r[0] - r[1]) != 1 for r in rows_map):
        raise ValueError(f"PAIR needs the two rows adjacent: {ROWS_MAP}")
    n_cores = COLS * ROWS
    # N_CORE fixes the core program; K and N only set the runtime counts. The
    # standalone path derives them from the shape it is asked to check.
    n_core = N_CORE or (N // n_cores)
    if n_core % VEC:
        raise ValueError(f"n_core ({n_core}) must be a multiple of {VEC}")
    if N % (n_cores * n_core):
        raise ValueError(f"N ({N}) must be a multiple of {n_cores * n_core}")
    if K % K_TILE:
        raise ValueError(f"K ({K}) must be a multiple of K_TILE ({K_TILE})")
    n_out = N // (n_cores * n_core)
    NT = K // K_TILE
    tb = tile_bytes(FMT, K_TILE, n_core)
    ab = ACT_TILE

    # L1 holds the double-buffered weight and activation tiles plus the f32
    # accumulator. Overrunning it fails deep inside the MLIR pipeline with an
    # unhelpful tile error, so it is checked here.
    l1 = 2 * (tb + ab) + n_core * 4
    if l1 > L1_BUDGET:
        raise ValueError(
            f"L1 needs {l1} B for K_TILE={K_TILE} n_core={n_core} ({FMT}), "
            f"over {L1_BUDGET}; lower --k-tile or raise --cols")

    gemv, zero, pro_tile = _kernels(FMT, K_TILE, n_core)

    if ab % 4:
        raise ValueError(f"activation tile ({ab} B) must be a multiple of 4")
    w_ty = np.ndarray[(tb,), np.dtype[np.uint8]]
    w_col_ty = np.ndarray[(ROWS * tb,), np.dtype[np.uint8]]
    # int32 elements so the core can read the count header from the first
    # object; the kernel casts the payload back to bytes.
    a_ty = np.ndarray[(ab // 4,), np.dtype[np.int32]]
    o_ty = np.ndarray[(n_core,), np.dtype[np.float32]]
    o_col_ty = np.ndarray[(ROWS * n_core,), np.dtype[np.float32]]

    w_all_ty = np.ndarray[(COLS * n_out * NT * ROWS * tb,), np.dtype[np.uint8]]
    # The header object, then the K tiles once per output chunk. The repeats
    # are laid out in DDR rather than described by a zero stride, because a
    # descriptor's length has to match what its three innermost dimensions
    # walk; a few kilobytes of duplicated activation is nothing next to the
    # weights, and it buys a single descriptor for the whole stream.
    a_all_ty = np.ndarray[((1 + n_out * NT) * ab // 4,), np.dtype[np.int32]]
    o_all_ty = np.ndarray[(N,), np.dtype[np.float32]]

    # One activation stream for the whole array, not one per column. Every
    # column is fed the same tiles, a shim tile has only two MM2S channels, and
    # spending one of them per column on a broadcast of a few kilobytes leaves
    # nothing for another design sharing the array. The weights - the traffic
    # that decides the speed - keep a channel per column.
    a_shim = ObjectFifo(a_ty, name="inA", depth=2)

    # GEMV_W_DEPTH: how far ahead of the cores a column's weights may run.
    # The fixed cost of a dispatch is the pipeline fill - with the arithmetic
    # stubbed out, two phases in one stream overlap by 94 us and no further,
    # because at depth 2 a stream can only be two objects ahead. The buffer is
    # a MemTile's, and a MemTile has 512 KB against an object's 21.5, so this
    # is where the fill can be paid before the phase that needs it.
    # Only where a column has one core: the four-row standalone design splits
    # a column's object four ways, and the extra descriptors that needs push
    # its output join past the MemTile's budget.
    # In the merged design (PAIR) four, so ssm_out's weights and the FFN's
    # first stream in while the core's stages hold the array - except on the
    # last column, whose MemTile's descriptors the core's stages have taken
    # (one more object there does not place; five anywhere does not either).
    wdepth = 2
    wdepths = [4] * (COLS - 1) + [2] if PAIR else [wdepth] * COLS
    w_shim, o_shim = [], []
    w_core = []
    for c in range(COLS):
        wf = ObjectFifo(w_col_ty, name=f"inW{c}", depth=wdepths[c])
        w_core.append(wf.cons().split(
            offsets=[i * tb for i in range(ROWS)],
            obj_types=[w_ty] * ROWS,
            depths=[2] * ROWS,
            names=[f"inW{c}_{i}" for i in range(ROWS)],
        ))
        w_shim.append(wf)




    # OUT_GROUP columns share one output stream, joined in a MemTile. A core's
    # output is half a kilobyte a chunk where its weights are tens of
    # kilobytes a tile, so the join costs nothing and gives the array back a
    # shim channel per column it removes - which is what the gdn block's state
    # stream, 512 KB through a single channel, needs.
    OG = OUT_GROUP or 1
    OG = max(1, min(OG, COLS))
    o_core = [None] * COLS
    # In a pair a column contributes one part, twice a core's block.
    PR = 1 if PAIR else ROWS
    p_ty = o_col_ty if PAIR else o_ty
    for g0 in range(0, COLS, OG):
        og = min(OG, COLS - g0)
        grp_ty = np.ndarray[(og * ROWS * n_core,), np.dtype[np.float32]]
        of = ObjectFifo(grp_ty, name=f"outO{g0}", depth=2)
        parts = of.prod().join(
            offsets=[k * (ROWS // PR) * n_core for k in range(og * PR)],
            obj_types=[p_ty] * (og * PR),
            depths=[2] * (og * PR),
            names=[f"outO{g0}_{k}" for k in range(og * PR)],
        )
        for ci in range(og):
            o_core[g0 + ci] = parts[ci * PR:(ci + 1) * PR]
        o_shim.append(of)

    # ACT_RAW: a tile of its own between the broadcast and the cores. It turns
    # the numbers the dispatch before it left in DDR - the projection's output,
    # the residual and gamma - into the codes the cores read, so the host has
    # nothing to do between a layer's two dispatches and they can go into one
    # runlist. It goes on a tile because the quantizer does not fit in the
    # cores' program memory, and it costs no DMA channel: the array's stream
    # switches carry its output to the cores the way the broadcast already
    # reaches them.
    act_pro = int(__import__("os").environ.get("ACT_RAW", "0"))
    a_cons = a_shim
    pro_workers = []
    if act_pro:
        a_q = ObjectFifo(a_ty, name="actQ", depth=2)
        pro_col = 0
        pro_row = 3

        # One object in, one out. The worker body is wrapped in an outer loop
        # by IRON, so this keeps no count of its own - and a count that drifts
        # from what the stream pushes is a deadlock, not a wrong number. The
        # tile carries everything the prologue needs: the row length for the
        # norm and the flag that ends its reduction.
        # A second input, from a buffer the host owns alone: the residual,
        # gamma and the words describing the tile. The tiles themselves are
        # written only by the dispatch before this one. Keeping the two in
        # separate buffers is what makes the order of the writes stop
        # mattering - host and array writes to one buffer hold in one order
        # only, and a layer in a single dispatch needs the other one.
        # With the side streams: before its output a tile may take side
        # objects and emit objects of its own, as many as the tile says (the
        # counts are the kernel's, so a tile that does not ask takes none).
        p_side = ObjectFifo(PRO_SIDE_TY, name="proS", depth=2)
        p_emit = ObjectFifo(PRO_EMIT_TY, name="proE", depth=2)

        def _pro1(a_in, a_out, s_in, e_out, ktile, kcnt, kside, kemit, cnt):
            # One buffer for both: the tile is its own host half. Correct
            # wherever the host writes the whole tile, which is every stream
            # that does not yet feed the second input.
            i_ = a_in.acquire(1)
            kcnt(i_, cnt)
            for _ in range_(cnt[0]):
                s_ = s_in.acquire(1)
                kside(s_, i_)
                s_in.release(1)
            for _ in range_(cnt[1]):
                e_ = e_out.acquire(1)
                kemit(e_, i_)
                e_out.release(1)
            o_ = a_out.acquire(1)
            ktile(i_, i_, o_)
            a_in.release(1)
            a_out.release(1)

        pro_workers = [Worker(_pro1, fn_args=[a_shim.cons(), a_q.prod(),
                                              p_side.cons(), p_emit.prod(),
                                              *pro_tile,
                                              Buffer(PRO_CNT_TY, name="proC")],
                              tile=Tile(pro_col, pro_row),
                              stack_size=0x1300,
                              # act-att.cc's state: the combine's o and its
                              # output, the q tile, the K/V rows, the gates
                              data_size=33792)]
        a_cons = a_q

    if PAIR:
        gemv_l, zero_l, merge = _kernels_lead(FMT, K_TILE, n_core)
        workers = list(pro_workers)
        if ATT:
            att_l = _kernels_att(FMT, K_TILE, n_core, True)
            att_p = _kernels_att(FMT, K_TILE, n_core, False)
        for c in range(COLS):
            # Adjacent tiles, so the hand-over is shared memory, not a stream.
            pj = ObjectFifo(o_ty, name=f"pj{c}", depth=2)
            if ATT:
                workers.append(Worker(
                    _core_lead_att, fn_args=[w_core[c][0].cons(), a_cons.cons(),
                                             pj.cons(), o_core[c][0].prod(),
                                             gemv_l, zero_l, merge, *att_l,
                                             Buffer(np.ndarray[(AD_SCR,), np.dtype[ml_dtypes.bfloat16]],
                                                    name=f"adscr{c}_0", mem_bank=3)],
                    tile=Tile(c, rows_map[c][0]), stack_size=4928))
                workers.append(Worker(
                    _core_part_att, fn_args=[w_core[c][1].cons(), a_cons.cons(),
                                             pj.prod(), gemv, zero, *att_p,
                                             Buffer(np.ndarray[(AD_SCR,), np.dtype[ml_dtypes.bfloat16]],
                                                    name=f"adscr{c}_1", mem_bank=3)],
                    tile=Tile(c, rows_map[c][1]), stack_size=4928))
                continue
            workers.append(Worker(
                _core_lead, fn_args=[w_core[c][0].cons(), a_cons.cons(),
                                     pj.cons(), o_core[c][0].prod(),
                                     gemv_l, zero_l, merge],
                tile=Tile(c, rows_map[c][0]), stack_size=4928))
            workers.append(Worker(
                _core, fn_args=[w_core[c][1].cons(), a_cons.cons(),
                                pj.prod(), gemv, zero],
                tile=Tile(c, rows_map[c][1]), stack_size=4928))
    else:
        workers = pro_workers + [
            Worker(_core, fn_args=[w_core[c][i].cons(), a_cons.cons(),
                                   o_core[c][i].prod(), gemv, zero],
                   tile=Tile(c, rows_map[c][i]),
                   stack_size=4928)
            for c in range(COLS) for i in range(ROWS)
        ]

    def flat_tap(total, offset, count):
        return TensorAccessPattern([1, total], offset, [1, count], [0, 1])

    # A shim tile has one AXI port for all of its channels, so what decides
    # the weight bandwidth is how many *tiles* carry a stream, not how many
    # channels. Left to the placer the eight landed on six - two columns
    # carried two apiece - and the FFN measured 19 GB/s against the array's
    # 28.6. One per column is the whole point of eight streams, so they are
    # pinned, and the recurrent design's fills are pinned around them
    # (attn_gdn_gated.py) to leave each column a channel.
    w_prods = [w_shim[c].prod(tile=Tile(c, 0)) for c in range(COLS)]
    # The broadcast goes on the column whose shim the recurrent design leaves
    # a second channel on. Pinned rather than left to the placer because the
    # backend's hand-built stream has to know which column carries it.
    a_prod = a_shim.prod(tile=Tile(4, 0))
    o_conss = [o.cons() for o in o_shim]
    if act_pro:
        o_conss = o_conss + [p_side.prod(tile=Tile(PRO_SIDE_COL, 0)),
                             p_emit.cons(tile=Tile(PRO_EMIT_COL, 0))]
    a_words = (1 + n_out * NT) * ab // 4

    def seq(a_w, a_a, a_o, *rest):
        # With the prologue a fourth buffer comes first: the residual rows the
        # layers hand each other (xdna-gemv.h XDNA_RES_*), which its side
        # streams read and write.
        a_r, wp, ap, op = rest if act_pro else (None, *rest)
        # One descriptor per stream per column, whatever the chunk count. A
        # shim tile has sixteen buffer descriptors for all of its channels, and
        # a descriptor per chunk needs 1 + (1 + n_out*NT) + n_out of them: past
        # four chunks that overruns the pool and the runtime reuses a
        # descriptor that has not drained yet, which loses a whole output chunk.
        # Folding the repeats into the access pattern keeps it at four per
        # column and costs nothing - the hardware walks the same addresses.
        span = n_out * NT * ROWS * tb
        # Explicit task groups rather than the default one, because a design
        # merged with another (fused_layer.py) may not mix the two.
        gi = TaskGroup()
        for c in range(COLS):
            wp[c].fill(a_w, tap=flat_tap(COLS * span, c * span, span), group=gi)
        # The header and every chunk's tiles are already consecutive.
        ap.fill(a_a, tap=flat_tap(a_words, 0, a_words), group=gi)
        if act_pro:
            # The prologue's side streams, for the allocation only: the
            # backend builds its own streams.
            op[-2].fill(a_r, tap=flat_tap(4096, 0, 512), group=gi)
        gi.finish()
        go = TaskGroup()
        if act_pro:
            op[-1].drain(a_r, tap=flat_tap(4096, 0, 256), wait=True, group=go)
        for gi_, g0 in enumerate(range(0, COLS, OG)):
            og = min(OG, COLS - g0)
            # The group's slice of every chunk: chunks are one array pass apart
            # in the output, the slice inside a chunk is contiguous.
            op[gi_].drain(a_o, tap=TensorAccessPattern(
                [1, N], g0 * ROWS * n_core,
                [n_out, og * ROWS * n_core], [n_cores * n_core, 1]),
                wait=True, group=go)
        go.finish()

    if act_pro:
        r_ty = np.ndarray[(4096,), np.dtype[np.float32]]
        return workers, [w_all_ty, a_all_ty, o_all_ty, r_ty, w_prods, a_prod, o_conss], seq
    return workers, [w_all_ty, a_all_ty, o_all_ty, w_prods, a_prod, o_conss], seq


# See fused_layer.py: the kernel sources have to be named, or a cached
# artifact can be served for a design whose C++ has changed.
@iron.jit(source_files=["gemv-q4.cc", "gemv-zero.cc"])
def gemv_q4(
    weights: In,
    acts: In,
    out: Out,
    *,
    FMT: CompileTime[str],
    K: CompileTime[int],
    N: CompileTime[int],
    K_TILE: CompileTime[int],
    COLS: CompileTime[int],
    N_CORE: CompileTime[int] = 0,
    ROWS_MAP: CompileTime[str] = ROWS_ALL,
):
    workers, rt_args, seq = build_gemv(FMT, K, N, K_TILE, COLS, N_CORE, ROWS_MAP)
    rt = Runtime(seq, rt_args)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def pack_weight_tile(fmt: str, codes: np.ndarray, d8: np.ndarray, m8: np.ndarray,
                     dS: np.ndarray, mS: np.ndarray) -> np.ndarray:
    """codes [k_tile, n_core] ints, d8/m8 [k_tile//group, n_core] int8,
    dS/mS [nsup, n_core] f32 - the super-block pair the int8s scale."""
    k_tile, n_core = codes.shape
    VEC = vec_for(n_core)
    grp = group_size(fmt)
    sbg = 256 // grp
    ng = k_tile // grp
    nsup = max(1, (k_tile if fmt == "q4g32" else 2 * k_tile) // 256)
    blocks, sups = [], []
    for j in range(n_core // VEC):
        cs = slice(j * VEC, (j + 1) * VEC)
        for g in range(ng):
            blk = codes[g * grp:(g + 1) * grp, cs].reshape(-1)
            if fmt == "q4g32":
                blocks.append(((blk[0::2].astype(np.uint8) & 0xF) |
                               ((blk[1::2].astype(np.uint8) & 0xF) << 4)))
            else:
                blocks.append(blk.astype(np.int8).view(np.uint8))
            blocks.append(d8[g, cs].astype(np.float32)
                          .astype(ml_dtypes.bfloat16).view(np.uint16).view(np.uint8))
            blocks.append(m8[g, cs].astype(np.float32)
                          .astype(ml_dtypes.bfloat16).view(np.uint16).view(np.uint8))
        for sp in range(nsup):
            r = min(sp, dS.shape[0] - 1)
            par = np.concatenate([split_pairs(dS[r:r + 1, cs]).reshape(-1),
                                  split_pairs(mS[r:r + 1, cs]).reshape(-1)])
            sups.append(par.view(np.uint8))
    t = np.concatenate(blocks + sups)
    return np.concatenate([t, np.zeros(tile_bytes(fmt, k_tile, n_core) - t.size, np.uint8)])


def quantize_act(fmt: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f32 row -> int8 codes and one scale per group. A group scale is local
    to the values a core holds, which is what lets a chained operator quantize
    its own slice without a reduction across cores."""
    grp = group_size(fmt)
    g = x.reshape(-1, grp)
    amax = np.abs(g).max(axis=1)
    d = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(g / d[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), d


def pack_act_tile(fmt: str, a: np.ndarray, d: np.ndarray,
                  epi: int = 0, n_core: int = 0) -> np.ndarray:
    """int8 codes, then a f32 sum and a f32 scale per group, then the flags and
    the code width. With n_core the tile is written the way a core emits it -
    blocks of n_core/2 codes followed by that block's sums and scales - which
    is what the epilogue produces and what flag bit 1 selects."""
    grp = group_size(fmt)
    sums = a.astype(np.int64).reshape(-1, grp).sum(axis=1).astype(np.float32)
    out = np.zeros(ACT_TILE, dtype=np.uint8)
    if n_core:
        per = n_core // 2                 # values a core's block holds
        gpb = per // grp                  # groups in one block
        blk = n_core * 4                  # bytes of a core's output object
        for b in range(len(a) // per):
            o = b * blk
            out[o:o + per] = a[b * per:(b + 1) * per].astype(np.int8).view(np.uint8)
            par = np.concatenate([sums[b * gpb:(b + 1) * gpb],
                                  d[b * gpb:(b + 1) * gpb].astype(np.float32)])
            out[o + per:o + per + par.nbytes] = par.view(np.uint8)
        epi |= 2
    else:
        body = np.concatenate([a.astype(np.int8).view(np.uint8),
                               sums.view(np.uint8), d.astype(np.float32).view(np.uint8)])
        out[:body.size] = body
    out[ACT_TILE - 8:ACT_TILE - 4].view(np.int32)[0] = epi
    out[ACT_TILE - 4:].view(np.int32)[0] = 0 if fmt == "q4g32" else 1
    return out


def _run_and_verify(opts) -> None:
    fmt, K, N, COLS = opts.fmt, opts.K, opts.N, opts.cols
    K_TILE = opts.k_tile or K_TILES[fmt]
    ROWS = len(_rows_map(opts.rows, opts.cols)[0])
    n_cores = COLS * ROWS
    n_core = opts.n_core or (N // n_cores)
    n_out = N // (n_cores * n_core)
    NT = K // K_TILE
    grp = group_size(fmt)
    ng = K // grp

    rng = np.random.default_rng(0)
    lo, hi = (0, 16) if fmt == "q4g32" else (-32, 32)
    codes = rng.integers(lo, hi, size=(K, N)).astype(np.int32)
    # Generated the way the format stores it: an int8 scale and min per group
    # over a bf16 pair per super-block, which is how every ggml type this packs
    # from is already built. So the reference below is exact, not approximate.
    sbg = 256 // grp
    nsb = max(1, ng // sbg)
    d8 = rng.integers(1, 64, size=(ng, N)).astype(np.int32)
    m8 = rng.integers(0, 64, size=(ng, N)).astype(np.int32)
    dS = (rng.standard_normal((nsb, N)) * 0.002).astype(np.float32)
    mS = (rng.standard_normal((nsb, N)) * 0.005).astype(np.float32)
    # Real activations, quantized the way the backend does: one int8 scale
    # for the whole row. The reference is the exact f32 product, so the check
    # covers the activation quantization too.
    af = (rng.standard_normal(K) * 0.7).astype(np.float32)
    af[rng.integers(0, K, size=max(1, K // 128))] *= 12.0     # outliers
    a, d_a = quantize_act(fmt, af)
    # The reference multiplies the activation the device actually holds - int8
    # codes times their group scale - not the f32 it came from. Against the f32
    # the check measures the activation quantization instead of the kernel, and
    # with a group of 32 carrying an outlier that is 4.5e-2, which says nothing
    # about whether the weights arrived correctly.
    aq = (a.reshape(-1, grp).astype(np.float32) * d_a[:, None]).reshape(-1)

    # Reference with the parameters the device actually sees: the bf16 pair
    # times the int8, which is exactly what the kernel computes.
    def _pair(x):
        p = split_pairs(x)
        return (p[0::2].astype(np.float32) + p[1::2].astype(np.float32))
    dSe = _pair(dS)
    mSe = _pair(mS)
    sup_of = np.minimum(np.arange(ng) // sbg, nsb - 1)
    d = (dSe[sup_of] * d8).astype(np.float32)
    m = (mSe[sup_of] * m8).astype(np.float32)
    gidx = np.arange(K) // grp
    w = codes.astype(np.float32) * d[gidx] + m[gidx]
    # The lane group is the kernel's vector width, which follows the core
    # width: the epilogue pairs gate with up inside one of them.
    VEC = vec_for(n_core)
    half = VEC // 2
    colmap = np.arange(N)
    if opts.epilogue:
        # The fold happens inside each 32-lane group of a core's accumulator,
        # not once per core: a core that owns more than 32 columns holds
        # several groups and pairs gate with up within each. So the map walks
        # groups, which is the same thing when n_core is 32 and the right thing
        # when it is not.
        colmap = np.empty(N, dtype=np.int64)
        for u in range(N // VEC):
            for i in range(VEC):
                colmap[u * VEC + i] = (u * half + (i % half) +
                                       (0 if i < half else N // 2))

    ref = (aq.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
    if opts.epilogue:
        g = ref[: N // 2].astype(np.float64)
        u = ref[N // 2:].astype(np.float64)
        f = g / (1.0 + np.exp(-g))
        ref = (f * u).astype(np.float32)

    tb = tile_bytes(fmt, K_TILE, n_core)
    wbuf = np.empty(COLS * n_out * NT * ROWS * tb, dtype=np.uint8)
    # Column major, then output chunk, then K tile, then row: that is the order
    # a column's single linear descriptor delivers them in.
    for c in range(COLS):
        for oc in range(n_out):
            for t in range(NT):
                for i in range(ROWS):
                    n0 = oc * n_cores * n_core + (c * ROWS + i) * n_core
                    ks = slice(t * K_TILE, (t + 1) * K_TILE)
                    cols = colmap[n0:n0 + n_core]
                    g0 = t * (K_TILE // grp)
                    g1 = g0 + K_TILE // grp
                    off = (((c * n_out + oc) * NT + t) * ROWS + i) * tb
                    s0 = min(g0 // sbg, nsb - 1)
                    s1 = max(s0 + 1, min(g1 // sbg, nsb))
                    wbuf[off:off + tb] = pack_weight_tile(
                        fmt, codes[ks][:, cols], d8[g0:g1][:, cols],
                        m8[g0:g1][:, cols], dS[s0:s1][:, cols], mS[s0:s1][:, cols])

    hdr = np.zeros(ACT_TILE // 4, dtype=np.int32)
    hdr[0] = NT
    hdr[1] = n_out
    hdr[ACT_TILE // 4 - 1] = 0 if fmt == "q4g32" else 1
    tiles = [pack_act_tile(fmt, a[t * K_TILE:(t + 1) * K_TILE],
                           d_a[t * (K_TILE // grp):(t + 1) * (K_TILE // grp)],
                           int(opts.epilogue and t == NT - 1),
                       0)
             for t in range(NT)]
    abuf = np.concatenate([hdr.view(np.uint8)] + tiles * n_out)

    a_w = iron.tensor((wbuf.size,), dtype=np.uint8, device="npu")
    a_a = iron.tensor((abuf.size // 4,), dtype=np.int32, device="npu")
    a_o = iron.zeros((N,), dtype=np.float32, device="npu")
    np.copyto(a_w.numpy(), wbuf)
    a_w._sync_to_device()
    np.copyto(a_a.numpy(), abuf.view(np.int32))
    a_a._sync_to_device()

    gemv_q4(a_w, a_a, a_o, **_compile_kwargs(opts))
    got = a_o.numpy().astype(np.float32)
    if opts.epilogue:
        # Each 32-lane group carries 16 valid floats and a zeroed tail.
        got = got.reshape(-1, VEC)[:, :half].reshape(-1)

    if opts.iters > 0:
        import time
        for _ in range(2):
            gemv_q4(a_w, a_a, a_o, **_compile_kwargs(opts))
        times = []
        for _ in range(opts.iters):
            t0 = time.perf_counter()
            gemv_q4(a_w, a_a, a_o, **_compile_kwargs(opts))
            times.append(time.perf_counter() - t0)
        wall = min(times) * 1e3
        wbytes = COLS * n_out * NT * ROWS * tb
        print(f"bench: min {wall:.3f} ms/dispatch, {wbytes / 1e6:.2f} MB weights "
              f"-> {wbytes / wall / 1e6:.1f} GB/s")

    bad = int(np.count_nonzero(~np.isfinite(got)))
    if bad:
        idx = np.nonzero(~np.isfinite(got))[0]
        print(f"non-finite lanes: {bad}/{got.size}, first at {idx[:8]}")
        print("got[:8] =", got[:8])
        print("ref[:8] =", ref[:8])
    scale = float(np.sqrt((ref ** 2).mean()))
    err = float(np.abs(got - ref).max())
    print(f"gemv {fmt}: K={K} N={N} K_TILE={K_TILE} cores={n_cores} "
          f"n_core={n_core} | max abs err {err:.4g}, rms(ref) {scale:.4g}, "
          f"rel {err / scale:.3e}")
    # What is left is the int8 activation quantization and one f32 rescale per
    # group, which is still far tighter than a bf16 GEMM of the same shape.
    assert_pass(ref, got, rtol=1e-4, atol=2e-3 * scale,
                fail_msg=f"gemv {fmt} mismatch vs NumPy")
    print(f"PASS: NPU gemv {fmt} "
          f"({tb} B/tile, {tb / (K_TILE * n_core):.3f} B/value)")


def _compile_kwargs(opts) -> dict:
    return {"FMT": opts.fmt, "K": opts.K, "N": opts.N,
            "K_TILE": opts.k_tile or K_TILES[opts.fmt], "COLS": opts.cols,
            "N_CORE": getattr(opts, "n_core", 0) or 0,
            "ROWS_MAP": getattr(opts, "rows", ROWS_ALL) or ROWS_ALL}


def main() -> None:
    parser = argparse.ArgumentParser(prog="gemv_q4",
                                     description="Build/run the packed-weight decode GEMV")
    add_compile_args(parser)
    parser.add_argument("--fmt", choices=["q4g32", "q8g16"], default="q4g32")
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("--k-tile", type=int, default=0,
                        help="K per streamed tile (0 = the format's default)")
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", default=ROWS_ALL,
                        help="compute rows per column, e.g. '3,5' for every "
                             "column or '3,5|2,5|...' one list per column")
    parser.add_argument("--epilogue", action="store_true",
                        help="close silu(gate)*up on the cores (N is the "
                             "concatenated gate and up)")
    parser.add_argument("--n-core", type=int, default=0,
                        help="columns per core; fixes the core program "
                             "(0 = derive from N)")
    parser.add_argument("--iters", type=int, default=0,
                        help="time this many dispatches after verifying")
    opts = parser.parse_args()

    run_design_cli(
        gemv_q4,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=o.cols),
    )
    import design_tag
    design_tag.stamp(getattr(opts, "xclbin_path", None),
                     getattr(opts, "insts_path", None),
                     getattr(opts, "dev", "") or "")


if __name__ == "__main__":
    main()
