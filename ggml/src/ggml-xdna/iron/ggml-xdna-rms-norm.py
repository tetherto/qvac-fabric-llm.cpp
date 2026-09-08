#!/usr/bin/env python3
# ggml-xdna-rms-norm.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON design for the ggml-xdna RMS_NORM kernel.

  run+verify   python3 ggml-xdna-rms-norm.py -n 1024
  compile-only python3 ggml-xdna-rms-norm.py -n 1024 -d npu2 \
                   --xclbin-path ggml-xdna-rms-norm-npu2-f32-1024.xclbin \
                   --insts-path  ggml-xdna-rms-norm-npu2-f32-1024.insts.bin

Unlike ADD and MUL this is not elementwise: the row has to be summed before any
output can be produced, so it cannot be expressed as a ``transform_binary``
lambda and needs an AIE kernel with accumulator state.

``num_elements`` is one row (ggml's ne0) and is baked into the artifacts. The
host ABI is three buffers - x, eps, y - in that order. eps travels in its own
one-element buffer instead of being a compile-time constant so the same xclbin
serves any model.

This is a reference implementation: scalar, correct, and not fast. It exists so
the RMS_NORM op and the ADD+RMS_NORM+MUL fusion path can be exercised end to
end; replacing it with a vectorised kernel changes nothing above it.
"""

from __future__ import annotations

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

KERNEL_SOURCE = r"""
#include <aie_api/aie.hpp>
#include <stdint.h>

// y[i] = x[i] / sqrt(sum(x^2)/n + eps)
extern "C" void ggml_xdna_rms_norm_f32(const float * x, const float * eps, float * y, int32_t n) {
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }

    // aie::invsqrt is an approximation; exact 1/sqrtf would pull in libm
    const float scale = aie::invsqrt(sum / (float) n + eps[0]);

    for (int32_t i = 0; i < n; i++) {
        y[i] = x[i] * scale;
    }
}
"""


def _rms_norm_program(num_elements: int):
    """Whole-row design: one kernel call per submission, three buffers."""
    row_ty = np.ndarray[(num_elements,), np.dtype[np.float32]]
    eps_ty = np.ndarray[(1,), np.dtype[np.float32]]

    kernel = ExternalFunction(
        "ggml_xdna_rms_norm_f32",
        source_string=KERNEL_SOURCE,
        arg_types=[row_ty, eps_ty, row_ty, np.int32],
        include_dirs=_include_dirs(),
    )

    of_x = ObjectFifo(row_ty, name="x")
    of_eps = ObjectFifo(eps_ty, name="eps")
    of_y = ObjectFifo(row_ty, name="y")

    def core_body(of_x, of_eps, of_y, kernel):
        e = of_eps.acquire(1)
        x = of_x.acquire(1)
        y = of_y.acquire(1)
        kernel(x, e, y, num_elements)
        of_x.release(1)
        of_eps.release(1)
        of_y.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_eps.cons(), of_y.prod(), kernel])

    rt = Runtime()
    with rt.sequence(row_ty, eps_ty, row_ty) as (a_x, a_eps, a_y):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_eps.prod(), a_eps)
        rt.drain(of_y.cons(), a_y, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def ggml_xdna_rms_norm(x: In, eps: In, y: Out, *, num_elements: CompileTime[int]):
    return _rms_norm_program(num_elements)


def _compile_kwargs(opts) -> dict:
    return {"num_elements": opts.num_elements}


def _run_and_verify(opts) -> None:
    n = opts.num_elements

    x = iron.rand(n, dtype=np.float32, device="npu")
    eps = iron.zeros(1, dtype=np.float32, device="npu")
    eps[0] = opts.eps
    y = iron.zeros_like(x)

    ggml_xdna_rms_norm(x, eps, y, **_compile_kwargs(opts))

    xn = x.numpy().astype(np.float64)
    expected = (xn / np.sqrt(np.mean(xn * xn) + opts.eps)).astype(np.float32)
    actual = y.numpy()

    if opts.verbose:
        print(f"{'i':>4}  {'x':>14}  {'y':>14}  {'ref':>14}")
        print("-" * 54)
        for i in range(min(8, n)):
            print(f"{i:4d}  {x[i]:14.6f}  {actual[i]:14.6f}  {expected[i]:14.6f}")

    assert_pass(expected, actual, rtol=1e-4,
                fail_msg="ggml-xdna RMS_NORM mismatch vs NumPy")
    print(f"PASS: NPU RMS_NORM matches NumPy (n={n}, eps={opts.eps})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-rms-norm",
        description="Build/run the ggml-xdna NPU RMS_NORM kernel",
    )
    add_compile_args(parser)
    parser.add_argument("-n", "--num-elements", type=int, default=1024,
                        help="row length (ggml ne0), baked into the artifacts (default: %(default)s)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="epsilon used by the verify path only (default: %(default)s)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="accepted for symmetry with the elementwise designs; unused")
    parser.add_argument("--dtype", choices=["f32"], default="f32")
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    run_design_cli(
        ggml_xdna_rms_norm,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
