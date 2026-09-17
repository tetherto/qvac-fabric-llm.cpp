#!/usr/bin/env python3
#
# The whole recurrent layer of the Qwen3.5 decode in one xclbin: the fused
# conv+norm+gdn+gated core (attn_gdn_gated.py) and the decode GEMV
# (gemv_q4.py) configured side by side in the same array.
#
# Why one artifact. A hardware context is per xclbin, and the cost of arriving
# at a dispatch whose context was not the last one to run measures 2640 us on
# this design - against 128 us for the same dispatch repeated while its context
# is resident. A layer that alternates between a core artifact and a GEMV
# artifact pays that twice per layer, 18 layers a token, which is about half of
# the decode. Configured together there is nothing to alternate between.
#
# How they fit. An AIE2P column has four compute rows (2..5); the array is
# eight columns. The core takes exactly one row in each column - conv on row 2
# of columns 0-1, norm on row 3 of columns 2-3, the gated epilogue on row 4 of
# column 2 and gdn on row 4 of columns 4-7 - so two further rows are free in
# every column, though not the same two. The GEMV takes those, sixteen cores of
# 64 output columns each, which is the same 1024 columns per pass as the
# standalone eight-column design and the same weight bandwidth, since every
# column still streams.
#
# Getting the two to fit was entirely a question of DMA channels, not compute
# tiles. The array has 16 shim channels in each direction and 6 per MemTile,
# and each design on its own used most of them. What made it fit:
#
#   - the core carries one stream for the whole gdn block and one for the whole
#     norm stage instead of one per column (13/15 shim channels -> 6/8);
#   - the GEMV broadcasts its one activation to every core instead of feeding
#     each column separately (16 MM2S -> 9);
#   - and the GEMV runs one core per column holding 128 output columns, so the
#     shim feeds each core directly: no weight split, no output join, and no
#     MemTile channels at all. A pass still covers 1024 columns and every
#     column still streams, so the weight bandwidth is unchanged.
#
# What the backend needs next: the compiler places the GEMV's shim endpoints
# wherever they fit, which in the merged design is not one per column - the
# core has already taken, for instance, every one of column 0's output
# channels. Pinning them does not place. So the mapping is read back out of the
# built artifact: its GEMV phase's 8 weight fills, 1 activation fill and 8
# output drains, in the order the design issues them, with the column,
# direction and
# channel of each. The core's phase is everything before that.

from __future__ import annotations

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, Program, Runtime
from aie.iron.device import from_name
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

import attn_gdn_gated as ag
import gemv_q4 as gq

# The rows the core leaves free, per column. Kept next to the core's own
# placement constants: if those move, this must move with them.
# One compute row per column - row 5, which the core never uses - and 128
# output columns per core. A pass still covers 1024 columns and every column
# still streams, but with a single core per column the shim feeds it directly:
# no weight split and no output join, so the GEMV needs no MemTile channels at
# all, and those are what the two designs run out of.
ROWS_FREE = "5"

COLS = 8
N_CORE = 128

# Columns sharing one output stream. A core's output is half a kilobyte a
# chunk against tens of kilobytes of weights a tile, so joining the outputs in
# a MemTile costs nothing and hands a shim channel per column back to the
# core's stages - which is what the gdn block's state stream needs, 512 KB
# through a single channel each way.
OUT_GROUP = 2


# The kernel sources are named so their mtimes reach the artifact hash. IRON
# hashes the generator and the files it is told about, not the C++ a design
# happens to read at generation time, and a cached artifact built from an
# older kernel is indistinguishable from a logic bug in the new one.
@iron.jit(source_files=["gemv-q4.cc", "gemv-zero.cc"])
def fused_layer(
    feed: In,
    x: In,
    pkvb: In,
    state: In,
    azg: In,
    gated_out: Out,
    weights: In,
    acts: In,
    out: Out,
    *,
    dev_name: CompileTime[str] = "npu2",
    FMT: CompileTime[str] = "q4g32",
    K: CompileTime[int] = 1024,
    N: CompileTime[int] = 7168,
    K_TILE: CompileTime[int] = 256,
):
    # The merged design pins its own post endpoints (the placer's choice);
    # the standalone attn_gdn_gated pins (1,0)/(2,0) instead. The prologue tile
    # that turns the host's numbers into the activation the next dispatch reads
    # is part of this design, not a choice: the backend always writes those
    # numbers, so an artifact without it cannot be driven. Both are set here
    # rather than read from the environment, so the source alone decides what
    # this design is.
    import os as _os
    _os.environ["POST_FILL_COL"] = "auto"
    _os.environ["POST_DRN_COL"] = "auto"
    _os.environ["ACT_RAW"] = "1"
    cw, ca, cseq = ag.build_core(dev_name)
    gw, ga, gseq = gq.build_gemv(FMT, K, N, K_TILE, COLS, N_CORE, ROWS_FREE,
                                 OUT_GROUP)

    # Each half consumes exactly its own runtime arguments, in the order they
    # were concatenated, so the merged sequence just splits the list.
    n_core_args = len(ca)

    def seq(*a):
        cseq(*a[:n_core_args])
        gseq(*a[n_core_args:])

    rt = Runtime(seq, ca + ga)
    return Program(from_name(dev_name, n_cols=COLS), rt, cw + gw).resolve_program()


# With one core per column holding 128 output columns, a weight tile is four
# times what it is at 32, so the K tile has to come down to keep L1 inside its
# budget. These are the sizes the backend bakes too.
K_TILES = {"q4g32": 256, "q8g16": 128}

# The kernel bakes a tile size for each code width, and only the width being
# built is taken from the argument - so both have to be the merged layer's,
# not just the one this invocation compiles for.
gq.K_TILES = dict(K_TILES)


def _compile_kwargs(opts) -> dict:
    return {"FMT": opts.fmt, "K": opts.K, "N": opts.N,
            "K_TILE": opts.k_tile or K_TILES[opts.fmt]}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fused_layer",
        description="Build the whole recurrent layer as one xclbin")
    add_compile_args(parser)
    parser.add_argument("--fmt", choices=["q4g32", "q8g16"], default="q4g32")
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("--k-tile", type=int, default=0)
    parser.add_argument("--elf-path", default=None,
                        help="compile as a self-contained full ELF to this path")
    opts = parser.parse_args()

    # The same design as a full ELF: the sequence becomes the control code of
    # the ELF instead of an instruction stream the host binds, and the buffers
    # are plain kernel arguments, so the six a patched stream is limited to no
    # longer bound the design.
    if opts.elf_path:
        import shutil
        spec = fused_layer.specialize(full_elf=True, dev_name=opts.dev or "npu2",
                                      **_compile_kwargs(opts))
        elf_path, _ = spec.compile(elf_path=opts.elf_path)
        if str(elf_path) != opts.elf_path:
            shutil.copy(elf_path, opts.elf_path)
        print("compiled full ELF", opts.elf_path,
              "kernel", spec.compilable._full_elf_kernel_name)
        return

    run_design_cli(
        fused_layer,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: from_name(o.dev, n_cols=COLS),
    )
    import design_tag
    design_tag.stamp(getattr(opts, "xclbin_path", None),
                     getattr(opts, "insts_path", None),
                     getattr(opts, "dev", "") or "")


if __name__ == "__main__":
    main()
