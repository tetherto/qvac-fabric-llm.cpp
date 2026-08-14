#!/usr/bin/env python3
"""Pure DDR read-bandwidth probe for the XDNA shim DMA.

Streams a [K x N] bf16 buffer from host (DDR) through the shim tile into the
mem tile via ObjectFifo, with no compute. The shim reads the buffer in
CONTIGUOUS 64KB bursts (sequential pattern). Lets us compare against the
strided 4D pattern the GEMM's B DMA uses (sizes=[8,64,16,32] strides=
[256,32768,2048,1], 64-byte inner bursts).

Usage: python bstream.py -K 1024 -N 2048 -d npu2 [--iters 20 --warmup 3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
    ceildiv,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args
import pyxrt as xrt

CHUNK = 16384  # bytes per BD, sequential


@iron.jit
def bstream(
    *,
    K: CompileTime[int],
    N: CompileTime[int],
    cols: CompileTime[int],
    strided: CompileTime[int],
    row512: CompileTime[int],
    dev_name: CompileTime[str] = "npu2",
):
    nbytes = K * N * 2
    slice_bytes = nbytes // cols   # per-column slice
    n_chunks = slice_bytes // CHUNK
    b_ty = np.ndarray[(nbytes,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]

    # Strided mode replicates the GEMM B BD geometry: a 4D box
    # sizes=[4,4,16,32] strides=[256,32768,2048,1] (bf16 elements) = 16KB per
    # BD with 64-byte inner bursts and 4KB row jumps. Mirrors the compiled
    # B_L3L2 BD (sizes=[8,64,16,32], same strides) at mem-tile-slot size.
    s_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    per_col = N // cols
    box_rows = 4 * 16 + 16  # 80 rows covered per 4D box (K-dim)
    n_boxes = -(-K // box_rows)  # ceil
    total_read = box_rows * 32 * 2 * n_boxes * cols

    # "row" mode: 512B bursts (the full per-column row segment: 256 bf16),
    # rows 4KB apart. One 16KB fill = 32 rows.
    row_tap_rows = 32
    n_row_fills = -(-K // row_tap_rows)

    workers = []
    rt_args = [b_ty]
    seq_args = []

    for col in range(cols):
        if strided:
            of = ObjectFifo(s_ty, name="b%d" % col, depth=1)
            of_l2l1 = of.cons().forward(obj_type=s_ty, name="b%d_l2l1" % col, tile=Tile(col, 1))

            def core_fn(inp, n=n_boxes):
                for _ in range_(n):
                    inp.acquire(1)
                    inp.release(1)
        elif row512:
            of = ObjectFifo(s_ty, name="b%d" % col, depth=1)
            of_l2l1 = of.cons().forward(obj_type=s_ty, name="b%d_l2l1" % col, tile=Tile(col, 1))

            def core_fn(inp, n=n_row_fills):
                for _ in range_(n):
                    inp.acquire(1)
                    inp.release(1)
        else:
            of = ObjectFifo(c_ty, name="b%d" % col, depth=1)
            of_l2l1 = of.cons().forward(obj_type=c_ty, name="b%d_l2l1" % col, tile=Tile(col, 1))

            def core_fn(inp, n=n_chunks):
                for _ in range_(n):
                    inp.acquire(1)
                    inp.release(1)

        workers.append(Worker(core_fn, [of_l2l1.cons()]))
        prod = of.prod(tile=Tile(col, 0))
        seq_args.append((prod, col))
        rt_args.append(prod)

    def seq_fn(B, *args):
        for prod, col in seq_args:
            if strided:
                # One 4D-stride BD per 80-row box, 16KB slot, grouped to reuse
                # the shim's 16 BD slots.
                for base in range(0, n_boxes, 4):
                    tg = TaskGroup()
                    for rb in range(base, min(base + 4, n_boxes)):
                        prod.fill(
                            B,
                            tap=TensorAccessPattern(
                                (K * N,),
                                offset=(rb * box_rows) * N + col * per_col,
                                sizes=[4, 4, 16, 32],
                                strides=[256, 32768, 2048, 1],
                            ),
                            group=tg,
                        )
                    tg.finish()
            elif row512:
                # 512B bursts: 32 rows x 256 bf16 per fill, rows N apart.
                for base in range(0, n_row_fills, 4):
                    tg = TaskGroup()
                    for rf in range(base, min(base + 4, n_row_fills)):
                        prod.fill(
                            B,
                            tap=TensorAccessPattern(
                                (K * N,),
                                offset=(rf * row_tap_rows) * N + col * per_col,
                                sizes=[row_tap_rows, per_col],
                                strides=[N, 1],
                            ),
                            group=tg,
                        )
                    tg.finish()
            else:
                col_off = col * slice_bytes
                for base in range(0, n_chunks, 8):
                    tg = TaskGroup()
                    for c in range(base, min(base + 8, n_chunks)):
                        prod.fill(
                            B,
                            tap=TensorAccessPattern(
                                (nbytes,), offset=col_off + c * CHUNK, sizes=[CHUNK], strides=[1]
                            ),
                            group=tg,
                        )
                    tg.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=cols), rt, workers)
    return prog.resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(prog="bstream")
    add_compile_args(parser)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cols", type=int, default=8,
                        help="Shim columns reading their B slice in parallel")
    parser.add_argument("--strided", action="store_true",
                        help="Read with the GEMM B pattern (64B bursts, 4KB jumps) instead of contiguous")
    parser.add_argument("--row512", action="store_true",
                        help="Read per-column row segments as 512B bursts (32 bf16 -> full 256-col slice)")
    parser.add_argument("--workdir", type=str, default="build/bin")
    opts = parser.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"

    kw = {"K": opts.K, "N": opts.N, "cols": opts.cols,
          "strided": 1 if opts.strided else 0,
          "row512": 1 if opts.row512 else 0,
          "dev_name": opts.dev}
    os.makedirs(opts.workdir, exist_ok=True)
    tag = "_strided" if opts.strided else ("_row512" if opts.row512 else "")
    base = os.path.join(opts.workdir, "bstream_K%d_N%d_c%d%s" % (
        opts.K, opts.N, opts.cols, tag))
    xclbin_path, insts_path = bstream.specialize(**kw).compile(
        xclbin_path=base + ".xclbin", inst_path=base + ".insts.bin")

    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(ctx, xb.get_kernels()[0].get_name())

    insts = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
    insts_bo = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    np.frombuffer(insts_bo.map(), dtype=np.uint32)[:] = insts
    insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    nbytes = opts.K * opts.N * 2
    if opts.strided:
        n_fills = -(-opts.K // 80)
    elif opts.row512:
        n_fills = -(-opts.K // 32)
    else:
        n_fills = opts.K * opts.N * 2 // CHUNK // opts.cols
    read_bytes = n_fills * CHUNK * opts.cols
    b_bo = xrt.bo(dev, nbytes, xrt.bo.host_only, 0)
    np.frombuffer(b_bo.map(), dtype=np.uint8)[:] = np.zeros(nbytes, dtype=np.uint8)
    b_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def once():
        run = kernel(3, insts_bo, int(insts.nbytes), b_bo)
        st = run.wait()
        if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit("wait state %s" % str(st))

    for _ in range(max(opts.warmup, 0)):
        once()
    n = max(opts.iters, 1)
    t0 = time.perf_counter()
    for _ in range(n):
        once()
    dt = (time.perf_counter() - t0) / n
    bw = read_bytes / dt / 1e9
    mode = "strided (64B bursts / 4KB jumps)" if opts.strided else (
        "512B row bursts" if opts.row512 else "sequential contiguous")
    print("stream    : K=%d N=%d buf=%.2f MiB, %d cols, %s"
          % (opts.K, opts.N, nbytes / 1048576, opts.cols, mode))
    print("avg       : %.3f ms/run -> %.1f GB/s aggregate read (%.2f MiB read)"
          % (dt * 1000, bw, read_bytes / 1048576))


if __name__ == "__main__":
    main()
