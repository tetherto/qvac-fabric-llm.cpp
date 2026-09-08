#!/usr/bin/env python3
# ggml-xdna-mul.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Derived from Xilinx/mlir-aie programming_examples/basic/vector_vector_add
#
"""IRON design for the ggml-xdna MUL kernel.

Two modes:

  run+verify   python3 ggml-xdna-mul.py -n 1024              (JIT, checks vs NumPy)
  compile-only python3 ggml-xdna-mul.py -n 1024 -d npu2 \
                   --xclbin-path ggml-xdna-mul-npu2-1024.xclbin \
                   --insts-path  ggml-xdna-mul-npu2-1024.insts.bin

This builds a second xclbin next to the ADD one. Both stay resident on the NPU
at the same time, each in its own hw_context, so dispatching between them never
reloads a design.

``num_elements`` is baked into the artifacts and *is* the host ABI: the backend
submits exactly this many elements per NPU run and chunks larger tensors.
"""

from __future__ import annotations

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform_binary
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

DTYPES = {
    "f32": np.float32,
    "i32": np.int32,
}


@iron.jit
def ggml_xdna_mul(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: CompileTime[int],
    dtype: CompileTime[type],
    tile_size: CompileTime[int],
):
    """output[i] = input0[i] * input1[i] for a fixed-length vector."""
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary(lambda a, b: a * b, tensor_ty, tile_size=tile_size)


def _compile_kwargs(opts) -> dict:
    return {
        "num_elements": opts.num_elements,
        "dtype": DTYPES[opts.dtype],
        "tile_size": opts.tile_size,
    }


def _validate(opts) -> None:
    if opts.num_elements % opts.tile_size != 0:
        raise SystemExit(
            f"--num-elements ({opts.num_elements}) must be a multiple of "
            f"--tile-size ({opts.tile_size})"
        )


def _run_and_verify(opts) -> None:
    dtype = DTYPES[opts.dtype]
    n = opts.num_elements

    if np.issubdtype(dtype, np.integer):
        input0 = iron.randint(0, 100, (n,), dtype=dtype, device="npu")
        input1 = iron.randint(0, 100, (n,), dtype=dtype, device="npu")
    else:
        input0 = iron.rand(n, dtype=dtype, device="npu")
        input1 = iron.rand(n, dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    ggml_xdna_mul(input0, input1, output, **_compile_kwargs(opts))

    expected = input0.numpy() * input1.numpy()
    actual = output.numpy()

    if opts.verbose:
        print(f"{'i':>4}  {'a':>12} * {'b':>12} = {'c':>12}  (ref)")
        print("-" * 60)
        for i in range(min(8, n)):
            print(
                f"{i:4d}  {input0[i]:12} * {input1[i]:12} = "
                f"{output[i]:12}  ({expected[i]})"
            )

    if np.issubdtype(dtype, np.integer):
        assert_pass(expected, actual, fail_msg="ggml-xdna MUL mismatch vs NumPy")
    else:
        # AIE f32 multiply rounds slightly differently than the host FPU (~1 ULP).
        assert_pass(expected, actual, rtol=1e-6,
                    fail_msg="ggml-xdna MUL mismatch vs NumPy")
    print(f"PASS: NPU MUL matches NumPy (n={n}, dtype={opts.dtype})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-mul",
        description="Build/run the ggml-xdna NPU MUL kernel",
    )
    add_compile_args(parser)
    parser.add_argument("-n", "--num-elements", type=int, default=1024,
                        help="elements per NPU run, baked into the artifacts (default: %(default)s)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="DMA tile size; must divide --num-elements (default: %(default)s)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="f32",
                        help="element type (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    run_design_cli(
        ggml_xdna_mul,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
