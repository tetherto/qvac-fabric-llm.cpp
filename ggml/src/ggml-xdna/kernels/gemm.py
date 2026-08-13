#!/usr/bin/env python3
# gemm.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Minimal IRON design for the ggml-xdna GEMM kernel (C = A @ B, bf16 -> f32).
#
# Compile-only mode (builds .xclbin + .insts.bin artifacts):
#
#   python3 gemm.py -M 32 -K 64 -N 128 -d npu2 \
#       --xclbin-path build/bin/gemm.xclbin \
#       --insts-path  build/bin/gemm.insts.bin
#
# The per-core loop counts (K_div_k, n_tiles_per_core) are runtime parameters
# written by the sequence into per-core RTP buffers before the DMA starts, so
# the compiled xclbin is (K, N)-independent: the same xclbin can be reused
# across dimension variants by swapping only the instruction stream.
#
# The instruction stream always starts with zero(C), so cross-call accumulation
# on the device is not possible. The C++ backend accumulates partial results
# in host memory.
#
# Reference: kernels/ggml-xdna-gemm.py in the qvac-fabric-llm.cpp_a repo.
# This file is a trimmed-down copy: bf16_f32 only, npu2 only, row-major A/B/C.

from __future__ import annotations

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
    WorkerRuntimeBarrier, str_to_dtype, ceildiv,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import (
    TensorTiler2D, TensorAccessPattern,
)
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
import aie.iron.kernels as akernels


# bf16 -> f32 only.
DTYPE_COMBOS = {
    "bf16_f32": {"label": "bf16->f32"},
}

# Micro-kernel MAC dimensions (r, s, t) for NPU2 native bf16.
MICROKERNEL_MAC_DIM = (4, 8, 8)


def _validate(opts) -> None:
    r, s, t = MICROKERNEL_MAC_DIM

    if opts.M % (2 * r) != 0:
        raise SystemExit(f"-M ({opts.M}) must be a multiple of 2*r ({2 * r})")
    if opts.K % s != 0:
        raise SystemExit(f"-K ({opts.K}) must be a multiple of s ({s})")
    if opts.N % (2 * t) != 0:
        raise SystemExit(f"-N ({opts.N}) must be a multiple of 2*t ({2 * t})")

    n_aie_rows = opts.n_aie_rows
    n_aie_cols = opts.n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < n_aie_rows else 1
    mem_tile_m_A = opts.tile_m * n_A_tiles_per_shim
    mem_tile_m_C = opts.tile_m * n_aie_rows
    mem_tile_n = opts.tile_n * n_aie_cols

    if opts.M % mem_tile_m_A != 0:
        raise SystemExit(f"-M ({opts.M}) must be a multiple of mem_tile_m_A ({mem_tile_m_A})")
    if opts.M % mem_tile_m_C != 0:
        raise SystemExit(f"-M ({opts.M}) must be a multiple of mem_tile_m_C ({mem_tile_m_C})")
    if opts.K % opts.tile_k != 0:
        raise SystemExit(f"-K ({opts.K}) must be a multiple of -k ({opts.tile_k})")
    if opts.N % mem_tile_n != 0:
        raise SystemExit(f"-N ({opts.N}) must be a multiple of mem_tile_n ({mem_tile_n})")

    if opts.tile_m % (2 * r) != 0:
        raise SystemExit(f"--tile-m ({opts.tile_m}) must be a multiple of 2*r ({2 * r})")
    if opts.tile_k % s != 0:
        raise SystemExit(f"--tile-k ({opts.tile_k}) must be a multiple of s ({s})")
    if opts.tile_n % (2 * t) != 0:
        raise SystemExit(f"--tile-n ({opts.tile_n}) must be a multiple of 2*t ({2 * t})")


def _compile_kwargs(opts) -> dict:
    dtype_in_str, dtype_out_str = opts.dtype.split("_")  # "bf16_f32" -> "bf16", "f32"
    return {
        "M":            opts.M,
        "K":            opts.K,
        "N":            opts.N,
        "m":            opts.tile_m,
        "k":            opts.tile_k,
        "n":            opts.tile_n,
        "n_aie_cols":   opts.n_aie_cols,
        "n_aie_rows":   opts.n_aie_rows,
        "dtype_in_str": dtype_in_str,
        "dtype_out_str": dtype_out_str,
        "dev_name":     opts.dev,
    }


@iron.jit
def bf16_f32_gemm(
    *,
    M:                       CompileTime[int],
    K:                       CompileTime[int],
    N:                       CompileTime[int],
    m:                       CompileTime[int],
    k:                       CompileTime[int],
    n:                       CompileTime[int],
    n_aie_cols:              CompileTime[int],
    n_aie_rows:              CompileTime[int] = 4,
    dtype_in_str:            CompileTime[str],
    dtype_out_str:           CompileTime[str],
    dev_name:                CompileTime[str] = "npu2",
):
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    # With n_aie_cols > n_aie_rows the A shim/mem tiles are capped at
    # n_aie_rows: there are only n_aie_rows row tiles of A to feed.
    n_shim_mem_A = min(n_aie_cols, n_aie_rows)
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < n_aie_rows else 1

    mem_tile_m_A = m * n_A_tiles_per_shim
    mem_tile_m_C = m * n_aie_rows
    mem_tile_n = n * n_aie_cols

    if dev_name == "npu1" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_name == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Matmul micro-kernel from the IRON kernel library.
    _matmul_kernel = akernels.mm(
        m, k, n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        vectorized=True,
    )
    r, s, t = _matmul_kernel.mac_dims

    assert M % mem_tile_m_A == 0, "A must be tileable into (m * n_A_tiles_per_shim, k)-sized blocks"
    assert K % k == 0
    assert N % mem_tile_n == 0, "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert M % mem_tile_m_C == 0, "C must be tileable into (m * n_aie_rows, n)-sized blocks"

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # Reduce to 1 only if CDO generation runs out of program memory (slow).
    fifo_depth = 2

    dev_ty = from_name(dev_name, n_cols=n_aie_cols)

    # Tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(mem_tile_m_A * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(mem_tile_m_C * n,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    zero_kernel = _matmul_kernel.zero
    matmul_kernel = _matmul_kernel
    fifo_depth_out = fifo_depth

    # AIE tiles: rows 0-1 mem tiles, rows 2-5 compute cores.
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A: L3-L2, then split along rows to the L2-L1 fifos.
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                tile=Tile(
                    2 * i if n_aie_cols == 8 else i, 1
                ),  # alternate columns in full 4x8 NPU2 case
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B: L3-L2, then forward to L2-L1 (row-major [k x n] tiles).
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                tile=Tile(col, 1),
            )
        )

        # Output C: L1-L2 (join along rows), then L2-L3 (row-major [m x n] tiles).
        dims_to_stream = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=dims_to_stream,
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth_out] * n_aie_rows,
                tile=Tile(col, 1),
            )
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[row][col] = c_tmp_fifos[row]

    # Per-variant tile counts. The host always submits the full baked M/K/N
    # block (inputs are zero-padded). The loop counts are passed to the cores
    # as runtime parameters (RTP) written by the runtime sequence, so a single
    # xclbin serves every (K, N) variant; only the instruction stream (DMA
    # tiling + RTP values) differs between variants.
    K_div_k = K // k
    n_c_col_tiles_per_core = N // mem_tile_n
    n_c_row_tiles_per_core = M // mem_tile_m_C
    n_tiles_per_core = n_c_row_tiles_per_core * n_c_col_tiles_per_core

    # Per-core RTP buffers (K_div_k, n_tiles_per_core) and a lock barrier so a
    # core never reads its counts before the sequence has written them.
    rtps = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp{row}_{col}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]
    barriers = [
        [WorkerRuntimeBarrier() for col in range(n_aie_cols)]
        for row in range(n_aie_rows)
    ]

    # Tasks for each worker to perform
    def core_fn(in_a, in_b, out_c, zero, matmul, rtp, barrier):
        barrier.wait_for_value(1)
        rtp_K_div_k = rtp[0]
        rtp_n_tiles_per_core = rtp[1]
        for _ in range_(rtp_n_tiles_per_core):
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(rtp_K_div_k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)

            out_c.release(1)

    # Set up compute tiles
    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
                rtps[row][col],
                barriers[row][col],
            ],
            tile=Tile(*core_tiles[row][col]),
            stack_size=0xD00,
        ),
    )

    # Define tensor access patterns (tiling) for A, B, and C
    A_tiles = TensorTiler2D.group_tiler(
        (M, K),  # Size of A matrix
        (mem_tile_m_A, k),  # Size of A (smallest) tile
        (1, K_div_k),  # Size of "group" of tiles
        # Repeat data so can distribute across whole column
        pattern_repeat=n_c_col_tiles_per_core,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N),  # Size of B matrix
        (k, n),  # Size of B tile
        # Number of tiles per transfer in each dimension (whole col, partial row)
        tile_group_repeats=(K_div_k, n_c_col_tiles_per_core),
        # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True,  # Send all tiles in column before moving on to next column
        prune_step=False,
    )

    # Shim-side fifo handles, registered with the Runtime so their shim
    # endpoints are bound before resolution.
    A_prods = [
        A_l3l2_fifos[i].prod(tile=Tile(2 * i if n_aie_cols == 8 else i, 0))
        for i in range(n_shim_mem_A)
    ]
    B_prods = [B_l3l2_fifos[col].prod(tile=Tile(col, 0)) for col in range(n_aie_cols)]
    C_conss = [C_l2l3_fifos[col].cons(tile=Tile(col, 0)) for col in range(n_aie_cols)]

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    tb_max_n_rows = 4

    def seq_fn(A, B, C, *args):
        # args: A_prods (n_shim_mem_A), B_prods (n_aie_cols), C_conss (n_aie_cols),
        #       rtps (n_aie_rows x n_aie_cols), barriers (n_aie_rows x n_aie_cols)
        arg_iter = iter(args)
        a_prods = [next(arg_iter) for _ in range(n_shim_mem_A)]
        b_prods = [next(arg_iter) for _ in range(n_aie_cols)]
        c_conss = [next(arg_iter) for _ in range(n_aie_cols)]
        rtps_seq = [[next(arg_iter) for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]
        barriers_seq = [[next(arg_iter) for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]

        # Program the per-core loop counts (K_div_k, n_tiles_per_core) and
        # release the barrier so the cores start with the values for this
        # variant. Written before any DMA so a core never reads stale counts.
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps_seq[row][col][0] = K_div_k
                rtps_seq[row][col][1] = n_tiles_per_core
                barriers_seq[row][col].set(1)

        # Task groups determine when to sync/await/free DMA runtime ops.
        for tb in range(ceildiv(n_c_row_tiles_per_core, tb_max_n_rows)):
            for pingpong in [0, 1]:
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, n_c_row_tiles_per_core - row_base]
                )
                if current_tb_n_rows <= 0:
                    # For small input sizes, we may not even need a "pong" iteration
                    break
                tg = TaskGroup()
                for col in range(n_aie_cols):
                    # C output transfer: one (m*n_aie_rows)-x-n sub-tile per
                    # column, evenly spaced, repeated current_tb_n_rows times
                    # for the next contiguous blocks of rows.
                    C_row_offset = row_base * mem_tile_m_C * N
                    C_col_offset = col * n
                    C_offset = C_col_offset + C_row_offset
                    C_sizes = [
                        current_tb_n_rows,
                        N // mem_tile_n,
                        mem_tile_m_C,
                        n,
                    ]
                    C_strides = [mem_tile_m_C * N, mem_tile_n, N, 1]
                    C_tile = TensorAccessPattern(
                        (M, N),
                        offset=C_offset,
                        sizes=C_sizes,
                        strides=C_strides,
                    )

                    c_conss[col].drain(C, tap=C_tile, wait=True, group=tg)

                    for tile_row in range(current_tb_n_rows):
                        # A input transfer: one (m*n_A_tiles_per_shim)-sized
                        # row sub-tile per column, repeated N//n//n_aie_cols.
                        if col < n_shim_mem_A:
                            tile_offset = (
                                (row_base + tile_row) * n_shim_mem_A + col
                            ) % len(A_tiles)
                            a_prods[col].fill(A, tap=A_tiles[tile_offset], group=tg)

                        # B input transfer: one (k x n) column sub-tile per column.
                        b_prods[col].fill(B, tap=B_tiles[col], group=tg)
                tg.finish()

    # Runtime sequence arguments: the three host buffers first, then the
    # shim fifo handles, then the per-core RTP buffers and barriers in a
    # fixed order matched by seq_fn's unpacking.
    rt_args: list = [A_ty, B_ty, C_ty]
    rt_args.extend(A_prods)
    rt_args.extend(B_prods)
    rt_args.extend(C_conss)
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            rt_args.append(rtps[row][col])
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            rt_args.append(barriers[row][col])

    rt = Runtime(seq_fn, rt_args)
    my_program = Program(dev_ty, rt, [w for row in workers for w in row])

    module = my_program.resolve_program()
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gemm",
        description="Build the ggml-xdna NPU GEMM kernel (C = A @ B, bf16->f32)",
    )
    add_compile_args(parser)
    parser.add_argument("-M", type=int, default=32,
                        help="Total M dimension (rows, multiple of 4*tile_m)")
    parser.add_argument("-K", type=int, default=64,
                        help="Total K dimension (inner dim, multiple of tile_k)")
    parser.add_argument("-N", type=int, default=128,
                        help="Total N dimension (cols, multiple of tile_n*n_aie_cols)")

    # Tile dimensions (sub-block sizes)
    parser.add_argument("--tile-m", type=int, default=8, dest="tile_m",
                        help="Per-core M tile (default: 8)")
    parser.add_argument("--tile-k", type=int, default=16, dest="tile_k",
                        help="Per-core K tile (default: 16)")
    parser.add_argument("--tile-n", type=int, default=16, dest="tile_n",
                        help="Per-core N tile (default: 16)")

    # AIE array configuration
    parser.add_argument("--n-aie-cols", type=int, default=8, dest="n_aie_cols",
                        help="Number of AIE columns (1-8, default: 8)")
    parser.add_argument("--n-aie-rows", type=int, default=4, dest="n_aie_rows",
                        help="Number of AIE core rows used (1-4, default: 4)")

    parser.add_argument("--dtype", choices=sorted(DTYPE_COMBOS), default="bf16_f32")
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    run_design_cli(
        bf16_f32_gemm,
        opts,
        compile_kwargs=_compile_kwargs,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
