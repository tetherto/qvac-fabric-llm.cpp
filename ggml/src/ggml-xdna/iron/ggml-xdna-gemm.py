#!/usr/bin/env python3
# ggml-xdna-gemm.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON design for the ggml-xdna MUL_MAT kernel.

  run+verify   python3 ggml-xdna-gemm.py --MB 16
  compile-only python3 ggml-xdna-gemm.py -d npu2 --MB 16 \
                   --xclbin-path ggml-xdna-gemm-npu2-bf16-8-16.xclbin \
                   --insts-path  ggml-xdna-gemm-npu2-bf16-8-16.insts.bin

One submit:

    C[MB*GR*M, GC*N] = A[MB*GR*M, KT*K] @ B[KT*K, GC*N]

with KT fixed at 16 (K=1024). The host splits larger K into 1024-chunks and
accumulates C, and loops over output column blocks of GC*N (NB=1). A K that
does not fill the last chunk is zero-padded by the host.

B is staged once per column as a single MemTile object (KT*K*N bf16), then
forwarded to L1 as tile-sized pieces and replayed MB times via set_iter_count.
Per-tile MemTile depth=KT plus set_iter_count deadlocks; the asymmetric
chain object is what makes L2 reuse work.

Artifact name: ggml-xdna-gemm-npu2-bf16-<KT>-<MB>
"""

from __future__ import annotations

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernels import mm as mm_kernel
from aie.utils.hostruntime.argparse import add_compile_args
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.device import from_name
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

MAC_R, MAC_S, MAC_T = 8, 8, 8

CORES_PER_COL = 4
MAX_COLS = 8
MEMTILE_BYTES = 512 * 1024

TILE_M, TILE_K, TILE_N = 32, 64, 64
# Fixed K depth: 16 tiles = 1024. One MemTile object per column (128 KiB).
DEFAULT_KT = 16


def swizzle_a(a, r=MAC_R, s=MAC_S):
    M, K = a.shape
    return a.reshape(M // r, r, K // s, s).transpose(0, 2, 1, 3).reshape(-1)


def swizzle_b(b, s=MAC_S, t=MAC_T):
    K, N = b.shape
    return b.reshape(K // s, s, N // t, t).transpose(0, 2, 1, 3).reshape(-1)


def unswizzle_c(c, M, N, r=MAC_R, t=MAC_T):
    return c.reshape(M // r, N // t, r, t).transpose(0, 2, 1, 3).reshape(M, N)


def _core(of_a, of_b, of_c, zero_fn, matmul_fn, k_tiles):
    elem_out = of_c.acquire(1)
    zero_fn(elem_out)
    for _ in range_(k_tiles):
        elem_in_a = of_a.acquire(1)
        elem_in_b = of_b.acquire(1)
        matmul_fn(elem_in_a, elem_in_b, elem_out)
        of_a.release(1)
        of_b.release(1)
    of_c.release(1)


@iron.jit
def ggml_xdna_gemm(
    input_a: In,
    input_b: In,
    output_c: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    KT: CompileTime[int],
    MB: CompileTime[int],
    GR: CompileTime[int],
    GC: CompileTime[int],
):
    matmul = mm_kernel(
        dim_m=M,
        dim_k=K,
        dim_n=N,
        input_dtype=bfloat16,
        output_dtype=np.float32,
        vectorized=True,
        emulate_bf16_mmul_with_bfp16=True,
        b_col_maj=False,
        c_col_maj=False,
    )
    zero = matmul.zero

    a_tile_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    b_tile_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    b_chain_ty = np.ndarray[(KT * K * N,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    a_ty = np.ndarray[(GR * MB * KT * M * K,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(GC * KT * K * N,), np.dtype[bfloat16]]
    c_host_ty = np.ndarray[(GC * MB * GR * M * N,), np.dtype[np.float32]]

    of_a = [ObjectFifo(a_tile_ty, name=f"inA{i}", depth=2) for i in range(GR)]

    of_b_ddr = []
    of_b_l2 = []
    for j in range(GC):
        # One MemTile-resident chain object, split to L1 tiles and replayed MB
        # times. Per-tile depth=KT + set_iter_count deadlocks; asymmetric chain
        # + forward(repeat_count=MB) is the working L2-reuse pattern.
        ddr = ObjectFifo(b_chain_ty, name=f"inB{j}", depth=1)
        l2 = ddr.cons(depth=1).forward(
            name=f"inB{j}_l1",
            obj_type=b_tile_ty,
            depth=2,
            repeat_count=MB,
        )
        of_b_ddr.append(ddr)
        of_b_l2.append(l2)

    of_c = []
    of_c_shim = []
    for j in range(GC):
        if GR == 1:
            f = ObjectFifo(c_ty, name=f"outC{j}", depth=1)
            of_c.append([f])
            of_c_shim.append(f)
            continue

        c_wide_ty = np.ndarray[(GR * M * N,), np.dtype[np.float32]]
        c_shim = ObjectFifo(c_wide_ty, name=f"outC{j}", depth=1)
        of_c.append(c_shim.prod().join(
            offsets=[i * M * N for i in range(GR)],
            obj_types=[c_ty] * GR,
            depths=[1] * GR,
        ))
        of_c_shim.append(c_shim)

    workers = [Worker(
        _core,
        fn_args=[of_a[i].cons(), of_b_l2[j].cons(depth=2), of_c[j][i].prod(),
                 zero, matmul, KT],
        while_true=True,
    ) for j in range(GC) for i in range(GR)]

    def slice_tap(total, offset, count):
        return TensorAccessPattern([1, total], offset, [1, count], [0, 1])

    a_stride = MB * KT * M * K
    b_stride = KT * K * N
    c_stride = MB * GR * M * N

    rt = Runtime()
    with rt.sequence(a_ty, b_ty, c_host_ty) as (a, b, c):
        for w in workers:
            rt.start(w)
        for i in range(GR):
            rt.fill(of_a[i].prod(), a,
                    tap=slice_tap(GR * a_stride, i * a_stride, a_stride))
        for j in range(GC):
            rt.fill(of_b_ddr[j].prod(), b,
                    tap=slice_tap(GC * b_stride, j * b_stride, b_stride))
            rt.drain(of_c_shim[j].cons(), c,
                     tap=slice_tap(GC * c_stride, j * c_stride, c_stride),
                     wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts) -> dict:
    return {"M": TILE_M, "K": TILE_K, "N": TILE_N,
            "KT": opts.KT, "MB": opts.MB,
            "GR": opts.grid[0], "GC": opts.grid[1]}


def _validate(opts) -> None:
    gr, gc = opts.grid
    if gr < 1 or gc < 1:
        raise SystemExit("--grid dimensions must be >= 1")
    if gr > CORES_PER_COL:
        raise SystemExit(f"grid rows map to cores within a column, so <= {CORES_PER_COL}")
    if gc > MAX_COLS:
        raise SystemExit(f"grid columns map to device columns, so <= {MAX_COLS}")
    if opts.KT < 1 or opts.KT > 16:
        raise SystemExit("--KT must be in 1..16 (MemTile BD-ID budget for chain+C)")
    if opts.MB < 1 or opts.MB > 256:
        raise SystemExit("--MB must be in 1..256")

    l1 = TILE_M * TILE_N * 4 + 2 * 2 * (TILE_M * TILE_K + TILE_K * TILE_N)
    if l1 > 60 * 1024:
        raise SystemExit(f"tile needs {l1} B of L1, over budget")

    # One B chain object + C join wide buffer share the MemTile.
    b_chain = opts.KT * TILE_K * TILE_N * 2
    c_wide = gr * TILE_M * TILE_N * 4
    if b_chain + c_wide > MEMTILE_BYTES:
        raise SystemExit(f"MemTile needs {b_chain + c_wide} B, over budget")


def _report(opts) -> None:
    gr, gc = opts.grid
    m, k, n, kt, mb = TILE_M, TILE_K, TILE_N, opts.KT, opts.MB
    macs = gr * gc * mb * kt * m * k * n
    dma = 2 * (gr * mb * kt * m * k + gc * kt * k * n) + gr * gc * mb * m * n * 4
    print(f"geometry: C[{mb * gr * m}, {gc * n}] = A[{mb * gr * m}, {kt * k}] @ "
          f"B[{kt * k}, {gc * n}]")
    print(f"          B staged as one L2 chain ({kt * k * n * 2 / 1024:.0f} KiB/col), "
          f"replayed x{mb}")
    print(f"          {macs / 1e6:.1f} MMAC and {dma / 1024 / 1024:.2f} MiB per submit, "
          f"{macs / dma:.1f} MAC/byte")


def _run_and_verify(opts) -> None:
    M, K, N, KT, MB = TILE_M, TILE_K, TILE_N, opts.KT, opts.MB
    GR, GC = opts.grid
    rng = np.random.default_rng(0)

    a = rng.standard_normal((MB * GR * M, KT * K), dtype=np.float32).astype(bfloat16)
    b = rng.standard_normal((KT * K, GC * N), dtype=np.float32).astype(bfloat16)

    input_a = iron.tensor((GR * MB * KT * M * K,), dtype=bfloat16, device="npu")
    input_b = iron.tensor((GC * KT * K * N,), dtype=bfloat16, device="npu")
    output_c = iron.zeros((GC * MB * GR * M * N,), dtype=np.float32, device="npu")

    a_parts = []
    for i in range(GR):
        for mb in range(MB):
            m0 = (mb * GR + i) * M
            for kt in range(KT):
                a_parts.append(swizzle_a(a[m0:m0 + M, kt * K:(kt + 1) * K]))
    np.copyto(input_a.numpy(), np.concatenate(a_parts))

    b_parts = []
    for j in range(GC):
        for kt in range(KT):
            b_parts.append(swizzle_b(b[kt * K:(kt + 1) * K, j * N:(j + 1) * N]))
    np.copyto(input_b.numpy(), np.concatenate(b_parts))

    ggml_xdna_gemm(input_a, input_b, output_c, **_compile_kwargs(opts))

    flat_c = output_c.numpy()
    actual = np.zeros((MB * GR * M, GC * N), dtype=np.float32)
    w = 0
    for j in range(GC):
        for mb in range(MB):
            for i in range(GR):
                tile = unswizzle_c(flat_c[w * M * N:(w + 1) * M * N], M, N)
                m0 = (mb * GR + i) * M
                actual[m0:m0 + M, j * N:(j + 1) * N] = tile
                w += 1

    expected = a.astype(np.float32) @ b.astype(np.float32)
    # bf16/BFP16 matmul noise grows with K and with more row blocks in one submit
    rtol, atol = 2e-1, 6e-1 * np.sqrt(KT * K / 32.0)

    if opts.verbose:
        _report(opts)

    assert_pass(expected.reshape(-1), actual.reshape(-1), rtol=rtol, atol=atol,
                fail_msg="ggml-xdna GEMM mismatch vs NumPy")
    err = np.abs(actual.reshape(-1) - expected.reshape(-1))
    print(f"PASS: NPU bf16 GEMM matches NumPy "
          f"(C[{MB * GR * M}, {GC * N}], K={KT * K}, MB={MB}), "
          f"max abs err {err.max():.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-gemm",
        description="Build/run the ggml-xdna NPU MUL_MAT kernel",
    )
    add_compile_args(parser)
    parser.add_argument("--KT", type=int, default=DEFAULT_KT,
                        help="K chunks per submit; K = KT*%d (default: %%(default)s). "
                             "Host splits larger K." % TILE_K)
    parser.add_argument("--MB", type=int, default=16,
                        help="row blocks per submit; M = MB*GR*%d. B is staged "
                             "in L2 and replayed this many times "
                             "(default: %%(default)s)" % TILE_M)
    parser.add_argument("--grid", type=int, nargs=2, default=[4, 8],
                        metavar=("ROWS", "COLS"),
                        help="worker grid (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    def device(o):
        return from_name(o.dev, n_cols=o.grid[1])

    run_design_cli(
        ggml_xdna_gemm,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_validate,
        device=device,
    )


if __name__ == "__main__":
    main()
