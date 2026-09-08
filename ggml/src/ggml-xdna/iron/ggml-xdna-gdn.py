#!/usr/bin/env python3
# ggml-xdna-gdn.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON design for GGML_OP_GATED_DELTA_NET (chunked prefill, scalar gate).

Qwen3.5-0.8B: head dim S=128. Full state is 64 KiB (over L1), so one Worker
keeps a ROWS=32 strip (16 KiB) resident. Two artifact modes:

  --workers 1  (legacy): host issues NS=S/ROWS submits per head/chunk with v
               slid so v[0:ROWS] match the active strip.
  --workers 4  (w4): one submit fans the tok stream to NS workers; each keeps
               its own state strip. Host state/attn are full-width.

Host ABI (3 DMA buffers), workers=1:

  tok   : CS*(3*S+2); v already slid to the active strip
  state : ROWS*S, updated in place
  out   : attn[CS*ROWS]

Host ABI, workers=4 (NS=4):

  tok   : CS*(3*S+2); v is the full S-wide vector
  state : NS*ROWS*S (full SxS, row-major strips), updated in place
  out   : attn[CS*S]

Build:
  python3 ggml-xdna-gdn.py -d npu2 --S 128 --ROWS 32 --CS 64 --workers 4 \\
      --xclbin-path ggml-xdna-gdn-npu2-s128-r32-cs64-w4.xclbin \\
      --insts-path  ggml-xdna-gdn-npu2-s128-r32-cs64-w4.insts.bin
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

DEFAULT_S = 128
DEFAULT_ROWS = 32
DEFAULT_CS = 64
L1_BUDGET = 60 * 1024

KERNEL_SOURCE = r"""
#include <stdint.h>

// attn_col is CS*ROWS; this call writes rows for token t at attn_col[t*ROWS+r].
// For workers=1, v already starts at the active strip (j0=0). For workers>1
// the kernel indexes v[j0 + r] while state_strip is that worker's local strip.
extern "C" void ggml_xdna_gdn_strip_f32(
    float * state_strip,
    const float * tok_t,
    float * attn_col,
    int32_t S,
    int32_t ROWS,
    int32_t t,
    int32_t j0,
    float scale) {
    const float * q = tok_t;
    const float * k = tok_t + S;
    const float * v = tok_t + 2 * S + j0;
    const float eg = tok_t[3 * S];
    const float beta = tok_t[3 * S + 1];
    float * attn_rows = attn_col + t * ROWS;

    for (int32_t r = 0; r < ROWS; ++r) {
        float * row = state_strip + r * S;
        float dot_k = 0.0f;
        for (int32_t i = 0; i < S; ++i) {
            row[i] *= eg;
            dot_k += row[i] * k[i];
        }
        const float delta = (v[r] - dot_k) * beta;
        float dot_q = 0.0f;
        for (int32_t i = 0; i < S; ++i) {
            row[i] += k[i] * delta;
            dot_q += row[i] * q[i];
        }
        attn_rows[r] = scale * dot_q;
    }
}
"""


def _gdn_program(S: int, ROWS: int, CS: int, workers: int, fuse_pre: int):
    del fuse_pre
    NS = S // ROWS
    if workers not in (1, NS):
        raise ValueError(f"workers must be 1 or {NS} (S/ROWS), got {workers}")

    tok_elems = 3 * S + 2
    strip_elems = ROWS * S
    state_elems = workers * strip_elems
    attn_elems = CS * ROWS * workers

    tok_t_ty = np.ndarray[(tok_elems,), np.dtype[np.float32]]
    strip_ty = np.ndarray[(strip_elems,), np.dtype[np.float32]]
    attn_strip_ty = np.ndarray[(CS * ROWS,), np.dtype[np.float32]]
    tok_pack_ty = np.ndarray[(CS * tok_elems,), np.dtype[np.float32]]
    state_ty = np.ndarray[(state_elems,), np.dtype[np.float32]]
    attn_ty = np.ndarray[(attn_elems,), np.dtype[np.float32]]

    # Per-worker L1: state strip + tok double-buffer + attn column.
    l1_worker = 2 * strip_elems * 4 + 2 * tok_elems * 4 + CS * ROWS * 4
    if l1_worker > L1_BUDGET:
        raise ValueError(f"L1 needs {l1_worker} B, over {L1_BUDGET}")

    strip_fn = ExternalFunction(
        "ggml_xdna_gdn_strip_f32",
        source_string=KERNEL_SOURCE,
        arg_types=[
            strip_ty,
            tok_t_ty,
            attn_strip_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.float32,
        ],
        include_dirs=_include_dirs(),
    )
    scale = float(1.0 / math.sqrt(S))

    def make_body(j0: int):
        def body(of_tok, of_sin, of_sout, of_attn, strip_fn):
            sin = of_sin.acquire(1)
            sout = of_sout.acquire(1)
            for i in range_(strip_elems):
                sout[i] = sin[i]
            of_sin.release(1)

            attn = of_attn.acquire(1)
            for t in range_(CS):
                tok = of_tok.acquire(1)
                strip_fn(sout, tok, attn, S, ROWS, t, j0, scale)
                of_tok.release(1)
            of_attn.release(1)
            of_sout.release(1)

        return body

    if workers == 1:
        of_sin = ObjectFifo(strip_ty, name="sin", depth=1)
        of_sout = ObjectFifo(strip_ty, name="sout", depth=1)
        of_tok = ObjectFifo(
            tok_pack_ty,
            name="tok",
            depth=1,
            consumer_obj_type=tok_t_ty,
        )
        of_attn = ObjectFifo(attn_strip_ty, name="attn", depth=1)

        worker = Worker(
            make_body(0),
            fn_args=[
                of_tok.cons(depth=2),
                of_sin.cons(),
                of_sout.prod(),
                of_attn.prod(),
                strip_fn,
            ],
        )

        rt = Runtime()
        with rt.sequence(tok_pack_ty, strip_ty, attn_ty) as (tok, st, out):
            rt.start(worker)
            rt.fill(of_tok.prod(), tok)
            rt.fill(of_sin.prod(), st)
            rt.drain(of_attn.cons(), out, wait=False)
            rt.drain(of_sout.cons(), st, wait=True)
        return Program(iron.get_current_device(), rt).resolve_program()

    # Multi-worker: broadcast tok; MemTile join/split keeps shim DMA to 4 channels.
    of_tok = ObjectFifo(
        tok_pack_ty,
        name="tok",
        depth=1,
        consumer_obj_type=tok_t_ty,
    )

    of_sin_shim = ObjectFifo(state_ty, name="sin", depth=1)
    of_sin = of_sin_shim.cons().split(
        offsets=[w * strip_elems for w in range(workers)],
        obj_types=[strip_ty] * workers,
        depths=[1] * workers,
        names=[f"sin{w}" for w in range(workers)],
    )

    of_attn_shim = ObjectFifo(
        np.ndarray[(attn_elems,), np.dtype[np.float32]],
        name="attn",
        depth=1,
    )
    of_attn = of_attn_shim.prod().join(
        offsets=[w * CS * ROWS for w in range(workers)],
        obj_types=[attn_strip_ty] * workers,
        depths=[1] * workers,
        names=[f"attn{w}" for w in range(workers)],
    )

    of_sout_shim = ObjectFifo(state_ty, name="sout", depth=1)
    of_sout = of_sout_shim.prod().join(
        offsets=[w * strip_elems for w in range(workers)],
        obj_types=[strip_ty] * workers,
        depths=[1] * workers,
        names=[f"sout{w}" for w in range(workers)],
    )

    worker_list = [
        Worker(
            make_body(w * ROWS),
            fn_args=[
                of_tok.cons(depth=2),
                of_sin[w].cons(),
                of_sout[w].prod(),
                of_attn[w].prod(),
                strip_fn,
            ],
        )
        for w in range(workers)
    ]

    rt = Runtime()
    with rt.sequence(tok_pack_ty, state_ty, attn_ty) as (tok, st, out):
        for w in worker_list:
            rt.start(w)
        rt.fill(of_tok.prod(), tok)
        rt.fill(of_sin_shim.prod(), st)
        rt.drain(of_attn_shim.cons(), out, wait=False)
        rt.drain(of_sout_shim.cons(), st, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def ggml_xdna_gdn(
    tok: In,
    state: InOut,
    out: Out,
    *,
    S: CompileTime[int],
    ROWS: CompileTime[int],
    CS: CompileTime[int],
    workers: CompileTime[int],
    fuse_pre: CompileTime[int],
):
    return _gdn_program(S, ROWS, CS, workers, fuse_pre)


def _compile_kwargs(opts) -> dict:
    return {
        "S": opts.S,
        "ROWS": opts.ROWS,
        "CS": opts.CS,
        "workers": opts.workers,
        "fuse_pre": opts.fuse_pre,
    }


def _ref_strip(q, k, v, g, beta, state_strip, S, ROWS, CS, j0=0):
    scale = 1.0 / math.sqrt(S)
    s = state_strip.copy()
    attn = np.empty((CS, ROWS), dtype=np.float32)
    for t in range(CS):
        s *= math.exp(float(g[t]))
        delta = (v[t, j0:j0 + ROWS] - s @ k[t]) * float(beta[t])
        s += np.outer(delta, k[t])
        attn[t] = scale * (s @ q[t])
    return attn, s


def _run_and_verify(opts) -> None:
    S, ROWS, CS, workers = opts.S, opts.ROWS, opts.CS, opts.workers
    NS = S // ROWS
    tok_elems = 3 * S + 2
    rng = np.random.default_rng(0)

    q = rng.standard_normal((CS, S), dtype=np.float32)
    k = rng.standard_normal((CS, S), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.standard_normal((CS, S), dtype=np.float32)
    g = -np.abs(rng.standard_normal(CS, dtype=np.float32)) * 0.01
    beta = np.clip(np.abs(rng.standard_normal(CS, dtype=np.float32)), 0.01, 1.0)
    state = rng.standard_normal((NS * ROWS, S), dtype=np.float32) * 0.01

    pack = np.zeros(CS * tok_elems, dtype=np.float32)
    for t in range(CS):
        base = t * tok_elems
        pack[base:base + S] = q[t]
        pack[base + S:base + 2 * S] = k[t]
        if workers == 1:
            pack[base + 2 * S:base + 2 * S + ROWS] = v[t, :ROWS]
        else:
            pack[base + 2 * S:base + 3 * S] = v[t]
        pack[base + 3 * S] = np.exp(g[t])
        pack[base + 3 * S + 1] = beta[t]

    state_elems = workers * ROWS * S
    attn_elems = CS * ROWS * workers
    tok = iron.tensor((CS * tok_elems,), dtype=np.float32, device="npu")
    st = iron.tensor((state_elems,), dtype=np.float32, device="npu")
    out = iron.zeros((attn_elems,), dtype=np.float32, device="npu")
    tok[:] = pack
    st[:] = state.reshape(-1)[:state_elems]

    ggml_xdna_gdn(tok, st, out, **_compile_kwargs(opts))

    actual = out.numpy()
    updated_state = st.numpy()
    if workers == 1:
        attn_ref, state_ref = _ref_strip(q, k, v, g, beta, state[:ROWS], S, ROWS, CS, 0)
        attn = actual.reshape(CS, ROWS)
        new_state = updated_state.reshape(ROWS, S)
        assert_pass(attn_ref.reshape(-1), attn.reshape(-1), rtol=2e-3, atol=2e-3,
                    fail_msg="ggml-xdna GDN attn mismatch")
        assert_pass(state_ref.reshape(-1), new_state.reshape(-1), rtol=2e-3, atol=2e-3,
                    fail_msg="ggml-xdna GDN state mismatch")
    else:
        attn_ref = np.empty((NS, CS, ROWS), dtype=np.float32)
        state_ref = np.empty_like(state)
        for ns in range(NS):
            a, s = _ref_strip(q, k, v, g, beta, state[ns * ROWS:(ns + 1) * ROWS],
                              S, ROWS, CS, ns * ROWS)
            attn_ref[ns] = a
            state_ref[ns * ROWS:(ns + 1) * ROWS] = s
        # Host layout: attn strips concatenated as worker-major CS*ROWS blocks.
        attn = actual.reshape(NS, CS, ROWS)
        new_state = updated_state.reshape(NS * ROWS, S)
        if opts.verbose:
            print(f"attn max abs: {np.max(np.abs(attn_ref - attn)):.6g}")
            print(f"state max abs: {np.max(np.abs(state_ref - new_state)):.6g}")
        assert_pass(attn_ref.reshape(-1), attn.reshape(-1), rtol=2e-3, atol=2e-3,
                    fail_msg="ggml-xdna GDN full attn mismatch")
        assert_pass(state_ref.reshape(-1), new_state.reshape(-1), rtol=2e-3, atol=2e-3,
                    fail_msg="ggml-xdna GDN full state mismatch")
    print(f"PASS: NPU GDN matches NumPy (S={S}, ROWS={ROWS}, CS={CS}, workers={workers})")


def _device(opts):
    from aie.iron.device import from_name
    # Multi-worker needs one column (and MemTile DMA budget) per strip worker.
    cols = opts.workers if opts.workers > 1 else 1
    return from_name(opts.dev, n_cols=cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-gdn",
        description="Build/run the ggml-xdna NPU GATED_DELTA_NET kernel",
    )
    add_compile_args(parser)
    parser.add_argument("--S", type=int, default=DEFAULT_S)
    parser.add_argument("--ROWS", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--CS", type=int, default=DEFAULT_CS)
    parser.add_argument("--workers", type=int, default=1,
                        help="1 = single strip (legacy); S/ROWS = full state in one submit")
    parser.add_argument("--fuse-pre", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()

    if opts.S % opts.ROWS:
        raise SystemExit("--S must be divisible by --ROWS")
    if opts.workers not in (1, opts.S // opts.ROWS):
        raise SystemExit(f"--workers must be 1 or {opts.S // opts.ROWS}")

    l1 = 2 * opts.ROWS * opts.S * 4 + 2 * (3 * opts.S + 2) * 4 + opts.CS * opts.ROWS * 4
    if l1 > L1_BUDGET:
        raise SystemExit(f"L1 needs {l1} B; reduce --ROWS or --CS")

    run_design_cli(
        ggml_xdna_gdn,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=_device,
    )


if __name__ == "__main__":
    main()
