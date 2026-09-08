#!/usr/bin/env python3
# ggml-xdna-add-rms-mul.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON design for the ggml-xdna ADD + RMS_NORM + MUL fused kernel.

  run+verify   python3 ggml-xdna-add-rms-mul.py -n 1024
  compile-only python3 ggml-xdna-add-rms-mul.py -n 1024 -d npu2 \
                   --xclbin-path ggml-xdna-add-rms-mul-npu2-f32-1024.xclbin \
                   --insts-path  ggml-xdna-add-rms-mul-npu2-f32-1024.insts.bin

This is the pre-norm transformer block: residual add, row RMS normalisation,
then multiply by the learned scale. The design writes two outputs in one
submission - the residual sum and the normalised, scaled row.

The host ABI uses three DMA buffers to fit the core tile's channel budget:

  ab_eps : [a[n], b[n], eps]            (2*n + 1 floats)
  weight : [w[n]]
  out    : [add_out[n], mul_out[n]]     (2*n floats)
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

// ab_eps = [a[n], b[n], eps]
// out    = [add_out[n], mul_out[n]]
extern "C" void ggml_xdna_add_rms_mul_f32(
    const float * ab_eps, const float * weight, float * out, int32_t n) {
    const float * a = ab_eps;
    const float * b = ab_eps + n;
    const float eps = ab_eps[2 * n];
    float * add_out = out;
    float * mul_out = out + n;

    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        const float s = a[i] + b[i];
        add_out[i] = s;
        sum += s * s;
    }

    const float scale = aie::invsqrt(sum / (float) n + eps);

    for (int32_t i = 0; i < n; i++) {
        mul_out[i] = add_out[i] * scale * weight[i];
    }
}
"""


def _add_rms_mul_program(num_elements: int):
    ab_ty = np.ndarray[(2 * num_elements + 1,), np.dtype[np.float32]]
    row_ty = np.ndarray[(num_elements,), np.dtype[np.float32]]
    out_ty = np.ndarray[(2 * num_elements,), np.dtype[np.float32]]

    kernel = ExternalFunction(
        "ggml_xdna_add_rms_mul_f32",
        source_string=KERNEL_SOURCE,
        arg_types=[ab_ty, row_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )

    of_ab = ObjectFifo(ab_ty, name="ab")
    of_w = ObjectFifo(row_ty, name="w")
    of_out = ObjectFifo(out_ty, name="out")

    def core_body(of_ab, of_w, of_out, kernel):
        ab = of_ab.acquire(1)
        w = of_w.acquire(1)
        out = of_out.acquire(1)
        kernel(ab, w, out, num_elements)
        of_ab.release(1)
        of_w.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_ab.cons(), of_w.cons(), of_out.prod(), kernel])

    rt = Runtime()
    with rt.sequence(ab_ty, row_ty, out_ty) as (a_ab, a_w, a_out):
        rt.start(worker)
        rt.fill(of_ab.prod(), a_ab)
        rt.fill(of_w.prod(), a_w)
        rt.drain(of_out.cons(), a_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def ggml_xdna_add_rms_mul(
    ab_eps: In,
    weight: In,
    out: Out,
    *,
    num_elements: CompileTime[int],
):
    return _add_rms_mul_program(num_elements)


def _compile_kwargs(opts) -> dict:
    return {"num_elements": opts.num_elements}


def _run_and_verify(opts) -> None:
    n = opts.num_elements

    a = iron.rand(n, dtype=np.float32, device="npu")
    b = iron.rand(n, dtype=np.float32, device="npu")
    weight = iron.rand(n, dtype=np.float32, device="npu")
    ab_eps = iron.zeros(2 * n + 1, dtype=np.float32, device="npu")
    ab_eps[:n] = a
    ab_eps[n:2 * n] = b
    ab_eps[2 * n] = opts.eps
    out = iron.zeros(2 * n, dtype=np.float32, device="npu")

    ggml_xdna_add_rms_mul(ab_eps, weight, out, **_compile_kwargs(opts))

    an = a.numpy().astype(np.float64)
    bn = b.numpy().astype(np.float64)
    wn = weight.numpy().astype(np.float64)
    expected_add = (an + bn).astype(np.float32)
    sum_ab = an + bn
    scale = 1.0 / np.sqrt(np.mean(sum_ab * sum_ab) + opts.eps)
    expected_mul = (sum_ab * scale * wn).astype(np.float32)

    add_out = out.numpy()[:n]
    mul_out = out.numpy()[n:]

    if opts.verbose:
        print(f"{'i':>4}  {'add':>12}  {'mul':>12}  {'ref_mul':>12}")
        print("-" * 48)
        for i in range(min(8, n)):
            print(f"{i:4d}  {add_out[i]:12.6f}  {mul_out[i]:12.6f}  {expected_mul[i]:12.6f}")

    assert_pass(expected_add, add_out, rtol=1e-4,
                fail_msg="ggml-xdna ADD_RMS_MUL add_out mismatch vs NumPy")
    assert_pass(expected_mul, mul_out, rtol=1e-4,
                fail_msg="ggml-xdna ADD_RMS_MUL mul_out mismatch vs NumPy")
    print(f"PASS: NPU ADD_RMS_MUL matches NumPy (n={n}, eps={opts.eps})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-add-rms-mul",
        description="Build/run the ggml-xdna NPU ADD+RMS_NORM+MUL fused kernel",
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
        ggml_xdna_add_rms_mul,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
