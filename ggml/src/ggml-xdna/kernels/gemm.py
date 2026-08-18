#!/usr/bin/env python3
# gemm.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Minimal IRON design for the ggml-xdna GEMM kernel (C = A @ B, bf16 -> f32).
#
# Compile-only mode (builds .xclbin + .insts.bin artifacts):
#
#   python3 gemm.py -M 32 -K 1024 -N 2048 -d npu2 \
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
import sys

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
        "emulate_bf16_mmul_with_bfp16": 0 if getattr(opts, "no_bfp16", False) else 1,
        "trace_size":   getattr(opts, "trace_size", 0),
        "trace_rows":   tuple(getattr(opts, "trace_rows", None) or ()),
        "trace_cols":   tuple(getattr(opts, "trace_cols", None) or ()),
        "trace_egress": getattr(opts, "trace_egress", 0),
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
    emulate_bf16_mmul_with_bfp16: CompileTime[int] = 1,
    trace_size:              CompileTime[int] = 0,
    trace_rows:              CompileTime[tuple] = (),
    trace_cols:              CompileTime[tuple] = (),
    trace_egress:            CompileTime[int] = 0,
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

    # Per-variant tile counts. The host always submits the full baked M/K/N
    # block (inputs are zero-padded). The loop counts are passed to the cores
    # as runtime parameters (RTP) written by the runtime sequence, so a single
    # xclbin serves every (K, N) variant; only the instruction stream (DMA
    # tiling + RTP values) differs between variants.
    K_div_k = K // k
    n_c_col_tiles_per_core = N // mem_tile_n
    n_c_row_tiles_per_core = M // mem_tile_m_C

    if dev_name == "npu1" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_name == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Matmul micro-kernel from the IRON kernel library. bfp16 emulation uses
    # the r=8 mmul (2x throughput); --no-bfp16 builds the native bf16 r=4
    # kernel (tile_m 8) used for the decode M-block.
    _matmul_kernel = akernels.mm(
        m, k, n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        vectorized=True,
        emulate_bf16_mmul_with_bfp16=bool(emulate_bf16_mmul_with_bfp16),
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
    # We only transfer 4 rows of tiles at once before starting a new transfer block.
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

    if trace_size > 0:
        from aie.utils.trace.events import CoreEvent

        rows = list(trace_rows) or list(range(n_aie_rows))
        cols = list(trace_cols) or list(range(n_aie_cols))
        traced = [workers[r][c] for r in rows for c in cols]
        # The trace overlay assigns one packet ID per traced tile; the 5-bit
        # packet-id field (with ID 0 reserved) caps the set at 31 tiles.
        if len(traced) > 31:
            print("gemm: trace capped to 31 tiles (hardware packet-id limit); "
                  "use --trace-rows/--trace-cols to select a subset",
                  file=sys.stderr)
            traced = traced[:31]
        my_program.enable_trace(
            trace_size,
            workers=traced,
            coretile_events=[
                CoreEvent.INSTR_EVENT_0,
                CoreEvent.INSTR_EVENT_1,
                CoreEvent.INSTR_VECTOR,
                CoreEvent.MEMORY_STALL,
                CoreEvent.STREAM_STALL,
                CoreEvent.LOCK_STALL,
                CoreEvent.ACTIVE,
            ],
            egress_shim_col=trace_egress,
        )

    module = my_program.resolve_program()
    return module


def _run_gemm(design, opts) -> None:
    """Compile the GEMM kernel and run it standalone on the NPU.

    Mirrors the ggml-xdna C++ submission scheme (xdna-ops.cpp): the host K
    dimension is tiled into GEMM_K_MAX-wide blocks, each block submitted as
    its own kernel run, and A/C buffers are ping-ponged so a block's C
    readback overlaps the next block's NPU execution. Synthetic bf16 data is
    used; pass --verify to compare against a numpy reference.

    With --hang the kernel is compiled at K > 1024 (K_div_k > 64) and run as a
    single submission with a watchdog; a background thread keeps snapshots of
    the trace buffer so the state just before a stall is preserved.
    """
    import os
    import time
    from pathlib import Path

    import pyxrt as xrt
    from ml_dtypes import bfloat16

    if opts.dev is None:
        from aie.utils import get_current_device
        from aie.utils.compile import resolve_target_arch

        rtdev = get_current_device()
        if rtdev is None:
            sys.exit("--run: no NPU runtime device detected (pass --dev npu2)")
        opts.dev = "npu2" if resolve_target_arch(rtdev) == "aie2p" else "npu"

    _validate(opts)

    GEMM_K_MAX = 1024

    Mk = opts.M
    N = opts.N
    kernel_K = opts.K
    host_K = opts.host_K if opts.host_K is not None else opts.K
    host_M = opts.host_M if opts.host_M is not None else opts.M

    n_m_blocks = ceildiv(host_M, Mk)
    n_k_blocks = ceildiv(host_K, kernel_K)

    if opts.hang and n_k_blocks != 1:
        sys.exit("--hang expects a single full-K run (set -K > 1024, not --host-K tiling)")
    if opts.hang and host_M > Mk:
        sys.exit("--hang expects host M <= baked M block (decode-style, --host-M <= %d)" % Mk)

    # --- compile ------------------------------------------------------------
    os.makedirs(opts.workdir, exist_ok=True)
    base = os.path.join(opts.workdir, "gemm_M%d_K%d_N%d" % (Mk, kernel_K, N))
    if opts.trace_size > 0:
        base += "_tr%d" % opts.trace_size
    xclbin_path = base + ".xclbin"
    insts_path = base + ".insts.bin"

    spec = design.specialize(**_compile_kwargs(opts))
    xclbin_path, insts_path = spec.compile(xclbin_path=xclbin_path, inst_path=insts_path)

    physical_mlir = None
    try:
        kdir = spec.compilable._kernel_dir
        if kdir is not None:
            pm = kdir / "input_with_addresses.mlir"
            if pm.exists():
                physical_mlir = str(pm)
    except Exception:
        pass

    # --- device -------------------------------------------------------------
    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(ctx, xb.get_kernels()[0].get_name())

    insts = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
    insts_bo = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    np.frombuffer(insts_bo.map(), dtype=np.uint32)[:] = insts
    insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    insts_bytes = int(insts.nbytes)

    to_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    from_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def alloc(nbytes):
        return xrt.bo(dev, nbytes, xrt.bo.host_only, 0)

    a_bos = [alloc(Mk * kernel_K * np.dtype(bfloat16).itemsize) for _ in range(2)]
    c_bos = [alloc(Mk * N * np.dtype(np.float32).itemsize) for _ in range(2)]
    b_bos = [alloc(kernel_K * N * np.dtype(bfloat16).itemsize) for _ in range(n_k_blocks)]

    trace_bo = None
    if opts.trace_size > 0:
        trace_bo = alloc(opts.trace_size)
        if n_k_blocks > 1:
            sys.exit("--trace-size with host K tiling: trace only a single K-block (set --host-K <= 1024)")

    # --- data ---------------------------------------------------------------
    rng = np.random.default_rng(7)
    A = rng.standard_normal((host_M, host_K)).astype(np.float32).astype(bfloat16)
    B = rng.standard_normal((host_K, N)).astype(np.float32).astype(bfloat16)

    for kb in range(n_k_blocks):
        k0 = kb * kernel_K
        kc = min(kernel_K, host_K - k0)
        blk = np.frombuffer(b_bos[kb].map(), dtype=bfloat16).reshape(kernel_K, N)
        blk.fill(bfloat16(0))
        blk[0:kc, :] = B[k0:k0 + kc, :]
        b_bos[kb].sync(to_dev)

    def fill_a(bo, mb, kb):
        m0 = mb * Mk
        mc = min(Mk, host_M - m0)
        k0 = kb * kernel_K
        kc = min(kernel_K, host_K - k0)
        a = np.frombuffer(bo.map(), dtype=bfloat16).reshape(Mk, kernel_K)
        a.fill(bfloat16(0))
        a[0:mc, 0:kc] = A[m0:m0 + mc, k0:k0 + kc]
        bo.sync(to_dev)

    # --- submission loop ----------------------------------------------------
    trace_words = None

    def run_once():
        nonlocal trace_words
        c_host = np.zeros((n_m_blocks * Mk, N), dtype=np.float32)
        bank = 0
        pending = -1
        pend_mb = 0
        run_prev = None
        stats = np.zeros(5)  # pack, sync, submit, wait, read

        for kb in range(n_k_blocks):
            for mb in range(n_m_blocks):
                t0 = time.perf_counter()
                fill_a(a_bos[bank], mb, kb)
                stats[0] += time.perf_counter() - t0

                t0 = time.perf_counter()
                a_bos[bank].sync(to_dev)
                stats[1] += time.perf_counter() - t0

                args = [a_bos[bank], b_bos[kb], c_bos[bank]]
                if trace_bo is not None:
                    args += [trace_bo]
                t0 = time.perf_counter()
                run = kernel(3, insts_bo, insts_bytes, *args)
                stats[2] += time.perf_counter() - t0

                if pending >= 0:
                    t0 = time.perf_counter()
                    st = run_prev.wait()
                    stats[3] += time.perf_counter() - t0
                    if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                        sys.exit("--run: kernel wait state %s" % str(st))
                    t0 = time.perf_counter()
                    c_bos[pending].sync(from_dev)
                    c = np.frombuffer(c_bos[pending].map(), dtype=np.float32).copy().reshape(Mk, N)
                    stats[4] += time.perf_counter() - t0
                    c_host[pend_mb * Mk:(pend_mb + 1) * Mk, :] += c
                run_prev = run
                pend_mb = mb
                pending = bank
                bank = 1 - bank

        if pending >= 0:
            t0 = time.perf_counter()
            st = run_prev.wait()
            stats[3] += time.perf_counter() - t0
            if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                sys.exit("--run: kernel wait state %s" % str(st))
            t0 = time.perf_counter()
            c_bos[pending].sync(from_dev)
            c = np.frombuffer(c_bos[pending].map(), dtype=np.float32).copy().reshape(Mk, N)
            stats[4] += time.perf_counter() - t0
            c_host[pend_mb * Mk:(pend_mb + 1) * Mk, :] += c

        if trace_bo is not None:
            trace_bo.sync(from_dev)
            trace_words = np.frombuffer(trace_bo.map(), dtype=np.uint32).copy()

        return c_host, stats

    def run_hang():
        """Single full-K (K > 1024) run with a watchdog and a trace saver.

        The run normally wedges the shared hw_context; the trace BO is a
        host-only buffer written by the shim DMA directly into system memory,
        so polling its mapping preserves the events emitted just before the
        stall.
        """
        import threading

        fill_a(a_bos[0], 0, 0)
        a_bos[0].sync(to_dev)
        args = [a_bos[0], b_bos[0], c_bos[0]]
        if trace_bo is not None:
            args += [trace_bo]
        run = kernel(3, insts_bo, insts_bytes, *args)

        done = [False]
        result = [None]
        started = time.perf_counter()

        def waiter():
            result[0] = run.wait()
            done[0] = True

        threading.Thread(target=waiter, daemon=True).start()

        snapshots = []
        while not done[0]:
            if trace_bo is not None:
                snapshots.append(np.frombuffer(trace_bo.map(), dtype=np.uint32).copy())
            if time.perf_counter() - started > opts.hang_timeout:
                break
            time.sleep(0.05)

        elapsed = time.perf_counter() - started
        if done[0]:
            st = result[0]
            if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                sys.exit("--hang: kernel returned %s" % str(st))
            print("--hang: run completed in %.3f ms (no wedge at K=%d)" % (elapsed * 1000, kernel_K))
            if trace_bo is not None and snapshots:
                _save_trace(snapshots[-1])
            return

        print("--hang: run did not complete within %.0f s - context WEDGE at K=%d (K_div_k=%d)"
              % (opts.hang_timeout, kernel_K, kernel_K // opts.tile_k))
        if trace_bo is not None and snapshots:
            _save_trace(snapshots[-1])
        sys.exit(2)

    def _save_trace(words):
        from aie.utils.trace import TraceConfig

        tcfg = TraceConfig(trace_size=opts.trace_size, trace_file=opts.trace_file)
        tcfg.write_trace(words)
        print("trace      : %d words -> %s" % (len(words), opts.trace_file))
        if not physical_mlir:
            print("  note: no input_with_addresses.mlir found; cannot parse")
            return
        try:
            from aie.utils.trace.parse import parse_trace
            from aie.utils.trace.utils import print_cycles_summary
            import json as _json

            buf = tcfg.read_trace()
            with open(physical_mlir) as f:
                mlir = f.read()
            events = parse_trace(buf, mlir, colshift=0)
            json_path = opts.trace_file + ".json"
            with open(json_path, "w") as f:
                _json.dump(events, f)
            print("  parsed   : -> %s" % json_path)
            print_cycles_summary(json_path)
        except Exception as e:
            print("  parse/summary failed: %r" % e)
        print("  commands : python -m aie.utils.trace.parse --input %s --mlir %s --output trace.json --colshift 0"
              % (opts.trace_file, physical_mlir))

    if opts.hang:
        run_hang()
        return

    # warmup + iters
    for _ in range(max(opts.warmup, 0)):
        run_once()
    iters = max(opts.iters, 1)
    c_host = None
    stats_sum = np.zeros(5)
    t_wall = time.perf_counter()
    for _ in range(iters):
        c_host, stats = run_once()
        stats_sum += stats
    wall = (time.perf_counter() - t_wall) / iters * 1000.0
    stats_avg = stats_sum / iters * 1000.0

    # --- report -------------------------------------------------------------
    flops = 2.0 * host_M * host_K * N
    print("shape      : M=%d K=%d N=%d (kernel M=%d K=%d, %d K-block(s) x %d M-block(s))"
          % (host_M, host_K, N, Mk, kernel_K, n_k_blocks, n_m_blocks))
    print("avg wall   : %.3f ms/iter -> %.1f GFLOP/s, %.1f tokens/s" % (wall, flops / wall / 1e6, 1000.0 / wall))
    print("  pack     : %.3f ms (%.1f%%)" % (stats_avg[0], 100 * stats_avg[0] / wall))
    print("  sync A   : %.3f ms (%.1f%%)" % (stats_avg[1], 100 * stats_avg[1] / wall))
    print("  submit   : %.3f ms (%.1f%%)" % (stats_avg[2], 100 * stats_avg[2] / wall))
    print("  wait NPU : %.3f ms (%.1f%%)" % (stats_avg[3], 100 * stats_avg[3] / wall))
    print("  read C   : %.3f ms (%.1f%%)" % (stats_avg[4], 100 * stats_avg[4] / wall))

    if not opts.no_verify:
        C_ref = A.astype(np.float32) @ B.astype(np.float32)
        C = c_host[0:host_M, :]
        err = np.max(np.abs(C - C_ref))
        print("verify     : max abs err %.4g (bf16 tolerance ~1e-2)" % err)

    if trace_words is not None:
        _save_trace(trace_words)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gemm",
        description="Build the ggml-xdna NPU GEMM kernel (C = A @ B, bf16->f32)",
    )
    add_compile_args(parser)
    parser.add_argument("-M", type=int, default=32,
                        help="Total M dimension (rows, multiple of 4*tile_m)")
    parser.add_argument("-K", type=int, default=1024,
                        help="Total K dimension (inner dim, multiple of tile_k). "
                             "1024 = GEMM_K_MAX; set > 1024 with --hang to probe the K_div_k > 64 boundary")
    parser.add_argument("-N", type=int, default=2048,
                        help="Total N dimension (cols, multiple of tile_n*n_aie_cols)")

    # Tile dimensions (sub-block sizes)
    parser.add_argument("--tile-m", type=int, default=8, dest="tile_m",
                        help="Per-core M tile (default: 8)")
    parser.add_argument("--tile-k", type=int, default=64, dest="tile_k",
                        help="Per-core K tile (default: 64; larger = fewer C reloads)")
    # bfp16 emulation uses the r=8 mmul (2x throughput, tile_m % 16 == 0);
    # --no-bfp16 builds the native bf16 r=4 kernel (tile_m 8) for decode.
    parser.add_argument("--no-bfp16", action="store_true", dest="no_bfp16",
                        help="Use the native bf16 r=4 mmul (tile-m 8, M 32)")
    parser.add_argument("--tile-n", type=int, default=64, dest="tile_n",
                        help="Per-core N tile (default: 64; wider = longer B DDR bursts)")

    # AIE array configuration
    parser.add_argument("--n-aie-cols", type=int, default=8, dest="n_aie_cols",
                        help="Number of AIE columns (1-8, default: 8)")
    parser.add_argument("--n-aie-rows", type=int, default=4, dest="n_aie_rows",
                        help="Number of AIE core rows used (1-4, default: 4)")

    parser.add_argument("--dtype", choices=sorted(DTYPE_COMBOS), default="bf16_f32")

    # Standalone run mode
    parser.add_argument("--run", action="store_true",
                        help="Compile and run the kernel on the NPU (no ggml backend)")
    parser.add_argument("--host-M", type=int, default=None, dest="host_M",
                        help="Host matrix M rows (decode=1, prefill=32+); default = -M")
    parser.add_argument("--host-K", type=int, default=None, dest="host_K",
                        help="Host matrix K (tiled into -K blocks); default = -K")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--workdir", type=str, default="build/bin",
                        help="Directory for compiled artifacts (default: build/bin)")
    parser.add_argument("--trace-size", type=int, default=0, dest="trace_size",
                        help="AIE trace buffer size in bytes (0 = off)")
    parser.add_argument("--trace-file", type=str, default="trace.txt", dest="trace_file")
    parser.add_argument("--trace-rows", type=int, nargs="*", default=None, dest="trace_rows",
                        help="Compute rows to trace (0..n_aie_rows-1); default all")
    parser.add_argument("--trace-cols", type=int, nargs="*", default=None, dest="trace_cols",
                        help="Compute cols to trace (0..n_aie_cols-1); default all")
    parser.add_argument("--trace-egress", type=int, default=0, dest="trace_egress",
                        help="Shim column the trace packets are routed to (default: 0)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the numpy reference check")
    parser.add_argument("--hang", action="store_true",
                        help="Single full-K run with K > 1024 and a watchdog (probe the documented wedge)")
    parser.add_argument("--hang-timeout", type=float, default=15.0, dest="hang_timeout",
                        help="Seconds to wait for the --hang run before declaring a wedge")
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    if opts.run or opts.hang:
        _run_gemm(bf16_f32_gemm, opts)
    else:
        run_design_cli(
            bf16_f32_gemm,
            opts,
            compile_kwargs=_compile_kwargs,
            validate=_validate,
        )


if __name__ == "__main__":
    main()
