#!/usr/bin/env python3
"""Standalone harness for the packed-weight expansion kernel (dequant_b.cc).

Proves the on-chip expansion of the NPU weight formats (xdna-quant.h) into the
bf16 B tile the mmul micro-kernel consumes, before it is folded into the GEMM
design. One core expands MB tiles per dispatch.

  rm -rf ~/.npu/cache && python3 dequant_b.py -d npu2 --fmt q4g32
  rm -rf ~/.npu/cache && python3 dequant_b.py -d npu2 --fmt q8g16

The cache wipe is not optional while editing the kernel: the compiled object
is keyed by the design, not by the source text, so an edited .cc is silently
ignored and the old object is relinked.

Layout contract (must match dequant_b.cc and the host packer in xdna-ops.cpp):
one (K_TILE x N_TILE) tile is stored sub-tile major in the mmul's own order -
(K_TILE/S) x (N_TILE/T) sub-tiles of S*T values, (si, ti) row-major - followed
by the f32 group-parameter planes. q4g32 carries a scale and a min plane,
q8g16 only a scale plane.
"""

from __future__ import annotations

import argparse

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

from wfmt import (dequant_fn, group_size, pack_tile, tile_bytes,
                  unpack_reference)

K_TILE = 64
N_TILE = 64


def _core(of_in, of_out, fn, MB):
    for _ in range_(MB):
        a = of_in.acquire(1)
        o = of_out.acquire(1)
        fn(a, o)
        of_in.release(1)
        of_out.release(1)


@iron.jit
def dequant_b(
    packed: In,
    out: Out,
    *,
    FMT: CompileTime[str],
    MB: CompileTime[int] = 8,
):
    tb = tile_bytes(FMT, K_TILE, N_TILE)
    fn = dequant_fn(FMT, K_TILE, N_TILE)

    in_ty = np.ndarray[(tb,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(K_TILE * N_TILE,), np.dtype[ml_dtypes.bfloat16]]
    in_all_ty = np.ndarray[(MB * tb,), np.dtype[np.uint8]]
    out_all_ty = np.ndarray[(MB * K_TILE * N_TILE,), np.dtype[ml_dtypes.bfloat16]]

    of_in = ObjectFifo(in_ty, name="inPK", depth=2)
    of_out = ObjectFifo(out_ty, name="outBF", depth=2)
    worker = Worker(_core, fn_args=[of_in.cons(), of_out.prod(), fn, MB])

    def seq(a_in, a_out, pin, cout):
        pin.fill(a_in, tap=TensorAccessPattern([1, MB * tb], 0, [1, MB * tb], [0, 1]))
        cout.drain(a_out,
                   tap=TensorAccessPattern([1, MB * K_TILE * N_TILE], 0,
                                           [1, MB * K_TILE * N_TILE], [0, 1]),
                   wait=True)

    rt = Runtime(seq, [in_all_ty, out_all_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _run_and_verify(opts) -> None:
    fmt, MB = opts.fmt, opts.MB
    rng = np.random.default_rng(0)
    ng = K_TILE // group_size(fmt)

    packed = []
    want = []
    for _ in range(MB):
        if fmt == "q4g32":
            codes = rng.integers(0, 16, size=(K_TILE, N_TILE), dtype=np.int64)
            m = rng.standard_normal((ng, N_TILE), dtype=np.float32) * 0.05
        else:
            codes = rng.integers(-32, 32, size=(K_TILE, N_TILE), dtype=np.int64)
            m = None
        d = (rng.standard_normal((ng, N_TILE), dtype=np.float32) * 0.01).astype(np.float32)
        packed.append(pack_tile(fmt, codes, d, m))
        want.append(unpack_reference(fmt, codes, d, m))

    a_in = iron.tensor((MB * tile_bytes(fmt, K_TILE, N_TILE),), dtype=np.uint8, device="npu")
    a_out = iron.zeros((MB * K_TILE * N_TILE,), dtype=ml_dtypes.bfloat16, device="npu")
    buf = a_in.numpy()
    np.copyto(buf, np.concatenate(packed))
    a_in._sync_to_device()

    dequant_b(a_in, a_out, FMT=fmt, MB=MB)

    got = a_out.numpy().astype(np.float32)
    ref = np.concatenate(want).astype(ml_dtypes.bfloat16).astype(np.float32)
    exact = np.concatenate(want).astype(np.float32)
    bad = np.nonzero(got != ref)[0]
    if bad.size:
        print(f"mismatch {bad.size}/{ref.size}; first 8 offending lanes "
              f"(exact f32 -> bf16 ref vs NPU):")
        for i in bad[:8]:
            print(f"  idx={i:6d} exact={exact[i]:.9g} ref={ref[i]:.9g} got={got[i]:.9g} "
                  f"ref_bits={np.float32(ref[i]).view(np.uint32):#010x} "
                  f"got_bits={np.float32(got[i]).view(np.uint32):#010x} "
                  f"exact_bits={exact[i].view(np.uint32):#010x}")
    assert_pass(ref, got, rtol=0, atol=0,
                fail_msg=f"dequant_b {fmt} mismatch vs NumPy")
    print(f"PASS: NPU dequant {fmt} K_TILE={K_TILE} N_TILE={N_TILE} MB={MB} "
          f"({tile_bytes(fmt, K_TILE, N_TILE)} B/tile vs {K_TILE * N_TILE * 2} B bf16)")


def _compile_kwargs(opts) -> dict:
    return {"FMT": opts.fmt, "MB": opts.MB}


def main() -> None:
    parser = argparse.ArgumentParser(prog="dequant_b",
                                     description="Build/run the packed-weight expansion kernel")
    add_compile_args(parser)
    parser.add_argument("--fmt", choices=["q4g32", "q8g16"], default="q4g32")
    parser.add_argument("--MB", type=int, default=8)
    opts = parser.parse_args()

    run_design_cli(
        dequant_b,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=1),
    )


if __name__ == "__main__":
    main()
