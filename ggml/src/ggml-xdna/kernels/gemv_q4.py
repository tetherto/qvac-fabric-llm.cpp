#!/usr/bin/env python3
"""Decode GEMV over the packed NPU weight formats, with the group parameters
applied to the accumulator (gemv_q4.cc).

  rm -rf ~/.npu/cache && python3 gemv_q4.py -d npu2 --fmt q4g32
  rm -rf ~/.npu/cache && python3 gemv_q4.py -d npu2 --fmt q8g16 -K 2048 -N 4096 --n-core 64

The q8 example needs --n-core 64: the shape alone derives 128, and two
int8 code planes plus the parameter planes then ask for more L1 than a
core has.

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

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.iron.device import from_name, Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

from wfmt import Q4_GROUP, Q8_GROUP, group_size, split_pairs

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
    code = Q4_GROUP * vec // 2 if fmt == "q4g32" else Q8_GROUP * vec
    return lg * (ng * (code + 4 * vec) + nsup * 4 * vec * 2)


def _kernels(fmt: str, k_tile: int, n_core: int):
    src = (Path(__file__).resolve().parent / "gemv-q4.cc").read_text()
    # The kernel's tile size has to be the one the design streams. Taking it
    # from the table instead of the argument silently mismatched whenever
    # --k-tile was given, which reads as a device fault rather than a build
    # mistake.
    kt = {"q4g32": K_TILES["q4g32"], "q8g16": K_TILES["q8g16"]}
    kt[fmt] = k_tile
    flags = [f"-DK_TILE_Q4={kt['q4g32']}", f"-DK_TILE_Q8={kt['q8g16']}",
             f"-DACT_TILE={ACT_TILE}", f"-DN_CORE={n_core}",
             f"-DQ4_GROUP={Q4_GROUP}", f"-DQ8_GROUP={Q8_GROUP}",
             f"-DGEMV_VEC={vec_for(n_core)}",
             # ACT_RAW builds the per-tile quantizer into the core (gemv_q4.cc),
             # so a producer on the array can hand this dispatch numbers rather
             # than codes. It costs about a kilobyte of a program memory that is
             # already full, so it is off unless a design asks for it.
             f"-DACT_RAW={int(__import__('os').environ.get('ACT_RAW', '0'))}"]

    # The prologue is a build of its own: its quantizer is about a kilobyte of
    # program memory and the GEMV cores have none spare, so it is linked into
    # the tile that runs it and nothing else. ACT_PRO in the flags is what
    # keeps the two objects apart (ExternalFunction names an object by its
    # digest, and the digest covers the flags).
    pro_flags = [f for f in flags if not f.startswith("-DACT_RAW")] + \
                ["-DACT_RAW=0", "-DACT_PRO=1"]
    digest = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    pdigest = hashlib.sha256((src + "\0".join(pro_flags)).encode()).hexdigest()[:8]
    w_ty = np.ndarray[(tile_bytes(fmt, k_tile, n_core),), np.dtype[np.uint8]]
    a_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    o_ty = np.ndarray[(n_core,), np.dtype[np.float32]]
    a_raw_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
    pro_tile = ExternalFunction(
        "ggml_xdna_act_pro",
        object_file_name=f"actpro_{n_core}_{pdigest}.o",
        source_string=src,
        arg_types=[a_raw_ty, a_raw_ty, a_raw_ty],
        compile_flags=pro_flags)
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


# The design body, separable from the Program it is wrapped in so the same
# array configuration can be built on its own or alongside another design in
# one xclbin (fused_layer.py). Returns what a Program needs: the workers, the
# runtime argument list and the sequence over it.
def build_gemv(FMT: str, K: int, N: int, K_TILE: int, COLS: int,
               N_CORE: int = 0, ROWS_MAP: str = ROWS_ALL, OUT_GROUP: int = 0):
    # "2,3,4,5" for every column, or one such list per column separated by "|".
    rows_map = _rows_map(ROWS_MAP, COLS)
    ROWS = len(rows_map[0])
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
    wdepth = 2
    w_shim, o_shim = [], []
    w_core = []
    for c in range(COLS):
        wf = ObjectFifo(w_col_ty, name=f"inW{c}", depth=wdepth)
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
    for g0 in range(0, COLS, OG):
        og = min(OG, COLS - g0)
        grp_ty = np.ndarray[(og * ROWS * n_core,), np.dtype[np.float32]]
        of = ObjectFifo(grp_ty, name=f"outO{g0}", depth=2)
        parts = of.prod().join(
            offsets=[k * n_core for k in range(og * ROWS)],
            obj_types=[o_ty] * (og * ROWS),
            depths=[2] * (og * ROWS),
            names=[f"outO{g0}_{k}" for k in range(og * ROWS)],
        )
        for ci in range(og):
            o_core[g0 + ci] = parts[ci * ROWS:(ci + 1) * ROWS]
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
        def _pro1(a_in, a_out, ktile):
            # One buffer for both: the tile is its own host half. Correct
            # wherever the host writes the whole tile, which is every stream
            # that does not yet feed the second input.
            i_ = a_in.acquire(1)
            o_ = a_out.acquire(1)
            ktile(i_, i_, o_)
            a_in.release(1)
            a_out.release(1)

        pro_workers = [Worker(_pro1, fn_args=[a_shim.cons(), a_q.prod(), pro_tile],
                              tile=Tile(pro_col, pro_row),
                              stack_size=0x1000)]
        a_cons = a_q

    workers = pro_workers + [
        Worker(_core, fn_args=[w_core[c][i].cons(), a_cons.cons(),
                               o_core[c][i].prod(), gemv, zero],
               tile=Tile(c, rows_map[c][i]),
               stack_size=3072)
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
    a_words = (1 + n_out * NT) * ab // 4

    def seq(a_w, a_a, a_o, wp, ap, op):
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
        gi.finish()
        go = TaskGroup()
        for gi_, g0 in enumerate(range(0, COLS, OG)):
            og = min(OG, COLS - g0)
            # The group's slice of every chunk: chunks are one array pass apart
            # in the output, the slice inside a chunk is contiguous.
            op[gi_].drain(a_o, tap=TensorAccessPattern(
                [1, N], g0 * ROWS * n_core,
                [n_out, og * ROWS * n_core], [n_cores * n_core, 1]),
                wait=True, group=go)
        go.finish()

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
    ng = k_tile // grp
    nsup = max(1, (k_tile if fmt == "q4g32" else 2 * k_tile) // 256)
    blocks, sups = [], []
    for j in range(n_core // VEC):
        cs = slice(j * VEC, (j + 1) * VEC)
        for g in range(ng):
            blk = codes[g * grp:(g + 1) * grp, cs].reshape(-1)
            if fmt == "q4g32":
                blocks.append(((blk[0::2].astype(np.uint8) & 0xF)
                               | ((blk[1::2].astype(np.uint8) & 0xF) << 4)))
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
    return np.concatenate(blocks + sups)


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
                colmap[u * VEC + i] = (u * half + (i % half)
                                       + (0 if i < half else N // 2))

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
