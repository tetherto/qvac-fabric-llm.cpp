#!/usr/bin/env python3
# ggml-xdna-conv.py -*- Python -*-
#
"""Causal depthwise conv1d for GGML_OP_SSM_CONV (Qwen3.5 GDN layers).

  compile-only python3 ggml-xdna-conv.py -d npu2 \\
    --C 16 --T 256 --KW 4 --cols 8 \\
    --xclbin-path ggml-xdna-conv-npu2-c16-t256-kw4.xclbin \\
    --insts-path  ggml-xdna-conv-npu2-c16-t256-kw4.insts.bin

Host ABI (2 BOs, f32):

  xw  : per-core [C][XROW] x, then [C][KW] w. XROW = round_up(T+KW-1, 16).
  out : per-core [C][T] channel-major; the host reorders to token-major.

With --token-major a core owns a column's whole channel slice and a quarter of
its tokens, so neither side reorders:

  xw  : per-core [T/ROWS+KW-1][ROWS*C] x, then [KW][ROWS*C] w.
  out : per-core [T/ROWS][ROWS*C]; the four concatenate into ggml's own
        {d_inner, n_t} for that column.

Both streams are ordered [col][tile][core], and a dispatch streams MB tiles
through L1 rather than one.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

DEFAULT_C = 16
DEFAULT_T = 256
DEFAULT_KW = 4
DEFAULT_COLS = 8
DEFAULT_MB = 8
CORES_PER_COL = 4
MAX_COLS = 8
L1_BUDGET = 36 * 1024
VLEN = 16


def _xrow(T: int, KW: int) -> int:
    n = T + KW - 1
    return (n + VLEN - 1) // VLEN * VLEN


@contextlib.contextmanager
def _fill(t):
    """Tensor.overwrite() is gone from mlir-aie; write .data and sync by hand."""
    if hasattr(t, "overwrite"):
        with t.overwrite() as buf:
            yield buf
        return
    yield t.data
    t._sync_to_device()


def _conv_fn(C: int, T: int, KW: int, tokmaj: int = 0, bf16: int = 0):
    src = (Path(__file__).resolve().parent / "conv.cc").read_text()
    flags = [f"-DDIM_C={C}", f"-DDIM_T={T}", f"-DDIM_KW={KW}"]
    if tokmaj:
        flags.append("-DCONV_TOKEN_MAJOR")
    if bf16:
        flags.append("-DCONV_BF16")
    digest = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    # Token-major rows are whole vectors of channels, so they never pad.
    xrow = T + KW - 1 if tokmaj else _xrow(T, KW)
    dt = np.dtype[bfloat16] if bf16 else np.dtype[np.float32]
    xw_ty = np.ndarray[(C * (xrow + KW),), dt]
    o_ty = np.ndarray[(C * T,), dt]
    return ExternalFunction(
        "ggml_xdna_conv_apply",
        object_file_name=f"ggml_xdna_conv_{digest}.o",
        source_string=src,
        arg_types=[xw_ty, o_ty],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def _core(of_xw, of_o, conv_fn, MB):
    for _ in range_(MB):
        xw = of_xw.acquire(1)
        o = of_o.acquire(1)
        conv_fn(xw, o)
        of_xw.release(1)
        of_o.release(1)


@iron.jit
def ggml_xdna_conv(
    xw: In,
    out: Out,
    *,
    C: CompileTime[int],
    T: CompileTime[int],
    KW: CompileTime[int],
    COLS: CompileTime[int],
    MB: CompileTime[int] = 1,
    TOKMAJ: CompileTime[int] = 0,
    BF16: CompileTime[int] = 0,
):
    ROWS = CORES_PER_COL
    n_cores = COLS * ROWS
    c_col = ROWS * C
    c_tot = n_cores * C

    if TOKMAJ:
        # The column, not the core, owns a channel slice; its four cores divide
        # the tokens. Slicing the other way would hand each core a strided view
        # of the column buffer, and a strided mem-tile split deadlocks the
        # design. This way every descriptor on every level is one flat run.
        if T % ROWS:
            raise ValueError(f"--T {T} must be a multiple of {ROWS}")
        k_C, k_T = c_col, T // ROWS
        xrow = k_T + KW - 1
    else:
        k_C, k_T = C, T
        xrow = _xrow(T, KW)
    tile_in = k_C * (xrow + KW)
    tile_out = k_C * k_T

    esz = 2 if BF16 else 4
    l1 = (tile_in + tile_out) * esz
    if l1 > L1_BUDGET:
        raise ValueError(f"L1 needs {l1} B, over {L1_BUDGET}")

    conv_fn = _conv_fn(k_C, k_T, KW, TOKMAJ, BF16)
    dt = np.dtype[bfloat16] if BF16 else np.dtype[np.float32]
    xw_ty = np.ndarray[(tile_in,), dt]
    o_ty = np.ndarray[(tile_out,), dt]
    xw_col_ty = np.ndarray[(ROWS * tile_in,), dt]
    o_col_ty = np.ndarray[(ROWS * tile_out,), dt]
    xw_all_ty = np.ndarray[(MB * n_cores * tile_in,), dt]
    o_all_ty = np.ndarray[(MB * T * c_tot,), dt]

    of_xw = []
    of_o = []
    of_xw_shim = []
    of_o_shim = []

    for j in range(COLS):
        xw_shim = ObjectFifo(xw_col_ty, name=f"inXW{j}", depth=1)
        of_xw.append(xw_shim.cons().split(
            offsets=[i * tile_in for i in range(ROWS)],
            obj_types=[xw_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"inXW{j}_{i}" for i in range(ROWS)],
        ))
        of_xw_shim.append(xw_shim)

        o_shim = ObjectFifo(o_col_ty, name=f"outO{j}", depth=1)
        of_o.append(o_shim.prod().join(
            offsets=[i * tile_out for i in range(ROWS)],
            obj_types=[o_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"outO{j}_{i}" for i in range(ROWS)],
        ))
        of_o_shim.append(o_shim)

    workers = []
    for j in range(COLS):
        for i in range(ROWS):
            workers.append(Worker(
                _core,
                fn_args=[of_xw[j][i].cons(), of_o[j][i].prod(), conv_fn, MB],
            ))

    def slice_tap(total, offset, count):
        return TensorAccessPattern([1, total], offset, [1, count], [0, 1])

    # A column owns MB * ROWS consecutive tiles, so both its streams are one
    # contiguous descriptor and the split hands round mb's ROWS tiles to its
    # cores. Neither level ever strides: channel-major cores each own a flat
    # [C][XROW] slab, and token-major ones own a flat [k_T][c_col] slab whose
    # ROWS-way concatenation is already ggml's {d_inner, n_t}.
    xw_col_span = MB * ROWS * tile_in
    o_col_span = MB * ROWS * tile_out

    p_xw = [of_xw_shim[j].prod() for j in range(COLS)]
    c_o = [of_o_shim[j].cons() for j in range(COLS)]

    def seq(a_xw, a_out, px, co):
        for j in range(COLS):
            px[j].fill(a_xw, tap=slice_tap(MB * n_cores * tile_in,
                                           j * xw_col_span, xw_col_span))
        for j in range(COLS):
            co[j].drain(a_out, tap=slice_tap(MB * T * c_tot,
                                             j * o_col_span, o_col_span),
                        wait=j == COLS - 1)

    rt = Runtime(seq, [xw_all_ty, o_all_ty, p_xw, c_o])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _compile_kwargs(opts) -> dict:
    return {
        "C": opts.C, "T": opts.T, "KW": opts.KW, "COLS": opts.cols,
        "MB": opts.MB, "TOKMAJ": int(opts.token_major),
        "BF16": int(getattr(opts, "bf16", 0)),
    }


def _run_and_verify(opts) -> None:
    C, T, KW, COLS, MB = opts.C, opts.T, opts.KW, opts.cols, opts.MB
    ROWS = CORES_PER_COL
    n_cores = COLS * ROWS
    c_col = ROWS * C
    c_tot = n_cores * C
    if opts.token_major:
        k_C, k_T = c_col, T // ROWS
        xrow = k_T + KW - 1
    else:
        k_C, k_T = C, T
        xrow = _xrow(T, KW)
    tile_in = k_C * (xrow + KW)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((MB, c_tot, T + KW - 1), dtype=np.float32)
    w = rng.standard_normal((MB, c_tot, KW), dtype=np.float32)
    bf16 = int(getattr(opts, "bf16", 0))
    dt = np.dtype[bfloat16] if bf16 else np.dtype[np.float32]
    if bf16:
        # Quantize the operands before the reference so the check measures the
        # kernel and not the bf16 rounding of its inputs.
        x = x.astype(bfloat16).astype(np.float32)
        w = w.astype(bfloat16).astype(np.float32)

    ref = np.zeros((MB, c_tot, T), dtype=np.float32)
    for b in range(MB):
        for c in range(c_tot):
            for t in range(T):
                ref[b, c, t] = float(np.dot(x[b, c, t:t + KW], w[b, c]))

    xw = np.zeros((MB * n_cores * tile_in,), dtype=np.float32)
    for b in range(MB):
        for j in range(COLS):
            for i in range(ROWS):
                base = (j * MB * ROWS + b * ROWS + i) * tile_in
                blk = xw[base:base + tile_in]
                if opts.token_major:
                    # The core's own token window, channels contiguous:
                    # [k_T+KW-1][c_col] of x then [KW][c_col] of w.
                    v = blk.reshape(xrow + KW, c_col)
                    ch = slice(j * c_col, (j + 1) * c_col)
                    t0 = i * k_T
                    v[:xrow] = x[b, ch, t0:t0 + xrow].T
                    v[xrow:] = w[b, ch].T
                else:
                    k = j * ROWS + i
                    for c in range(C):
                        src = x[b, k * C + c]
                        blk[c * xrow:c * xrow + src.shape[0]] = src
                        blk[C * xrow + c * KW:C * xrow + (c + 1) * KW] = w[b, k * C + c]

    # mlir-aie's tensor factory wants the ml_dtypes scalar, not a np.dtype.
    iron_dt = bfloat16 if bf16 else np.float32
    a_xw = iron.tensor((MB * n_cores * tile_in,), dtype=iron_dt, device="npu")
    a_out = iron.zeros((MB * T * c_tot,), dtype=iron_dt, device="npu")
    with _fill(a_xw) as buf:
        np.copyto(buf, xw.astype(iron_dt))

    ggml_xdna_conv(a_xw, a_out, **_compile_kwargs(opts))
    if opts.token_major:
        got = a_out.numpy().astype(np.float32).reshape(COLS, MB, ROWS, k_T, c_col)
        want = ref.reshape(MB, COLS, c_col, ROWS, k_T).transpose(1, 0, 3, 4, 2)
    else:
        got = a_out.numpy().astype(np.float32).reshape(COLS, MB, ROWS, C, T)
        want = ref.reshape(MB, COLS, ROWS, C, T).transpose(1, 0, 2, 3, 4)
    tol = 2e-2 if bf16 else 1e-4
    assert_pass(want.reshape(-1), got.reshape(-1), rtol=tol, atol=tol,
                fail_msg="ggml-xdna conv mismatch vs NumPy")
    print(f"PASS: NPU conv C={C} T={T} KW={KW} cols={COLS} cores={n_cores} "
          f"MB={MB} token_major={int(opts.token_major)} bf16={bf16}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-conv",
        description="Build/run the ggml-xdna NPU SSM_CONV kernel",
    )
    add_compile_args(parser)
    parser.add_argument("--C", type=int, default=DEFAULT_C)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--KW", type=int, default=DEFAULT_KW)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--MB", type=int, default=DEFAULT_MB)
    parser.add_argument("--token-major", action="store_true",
                        help="x as [XROW][C] and out as [T][C], so the host "
                             "neither gathers nor scatters across channels")
    parser.add_argument("--bf16", action="store_true",
                        help="pack xw and out as bf16 (halves every BO stream; "
                             "the accumulators stay f32)")
    opts = parser.parse_args()
    if opts.cols < 1 or opts.cols > MAX_COLS:
        raise SystemExit(f"--cols must be in 1..{MAX_COLS}")

    def device(o):
        return from_name(o.dev, n_cols=o.cols)

    run_design_cli(
        ggml_xdna_conv,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device,
    )


if __name__ == "__main__":
    main()
