#!/usr/bin/env python3
# ggml-xdna-gdn-prefill.py -*- Python -*-
#
"""Qwen3.5 GDN prefill dataflow (bf16, 4x8 workers).

GGML_OP_GATED_DELTA_NET is only the semantic match. 32 tiles: 8 columns x 4
cores, one 64-row strip per core. MemTile DMA cannot hold two 4-core joins, so
attn and the updated state share one packed output. The strip lives in L1
across NC token chunks of one TXN (host pads unused chunks).

Each column has one tok stream (2 heads); strip workers multicast it and skip
the other head (no host NS replay).

  compile-only python3 ggml-xdna-gdn-prefill.py -d npu2 \\
    --xclbin-path ggml-xdna-gdn-prefill-npu2-s128-h16-cs64.xclbin \\
    --insts-path  ggml-xdna-gdn-prefill-npu2-s128-h16-cs64.insts.bin

Host ABI (3 BOs, bf16):

  tok    : [NC][H][CS][3*S+2]   q | k | v | eg | beta
  packed : [H][NS][ROWS*S + CS*ROWS]  in (state prefix) / out (state|attn)
           state strip is transposed, [S][ROWS]
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction, Kernel
from aie.iron.kernels._common import _include_dirs
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

DEFAULT_S = 128
DEFAULT_H = 16
DEFAULT_CS = 64
DEFAULT_COLS = 8
DEFAULT_NC = 1
ROWS = 64
CORES_PER_COL = 4
L1_BUDGET = 60 * 1024
MEMTILE_BYTES = 512 * 1024


def _fns(S: int, CS: int):
    src = (Path(__file__).resolve().parent / "gdn-prefill.cc").read_text()
    flags = [f"-DDH={S}", f"-DROWS={ROWS}", "-DVEC=16"]
    digest = hashlib.sha256((src + "".join(flags)).encode()).hexdigest()[:8]
    tok_t_ty = np.ndarray[(3 * S + 2,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(ROWS * S + CS * ROWS,), np.dtype[bfloat16]]
    token = ExternalFunction(
        "ggml_xdna_gdn_token_bf16",
        object_file_name=f"ggml_xdna_gdn_prefill_{digest}.o",
        source_string=src,
        arg_types=[packed_ty, tok_t_ty, np.int32, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )
    copy = Kernel("ggml_xdna_gdn_copy_strip", token.object_file_name, [packed_ty, packed_ty])
    return token, copy


def _make_core(j0: int, h_local: int, hpc: int, NC: int):
    def core(of_tok, of_sin, of_out, token_fn, copy_fn, CS):
        for _ in range_(NC):
            sin = of_sin.acquire(1)
            packed = of_out.acquire(1)
            copy_fn(packed, sin)
            of_sin.release(1)
            for _ in range_(h_local * CS):
                tok = of_tok.acquire(1)
                of_tok.release(1)
            for t in range_(CS):
                tok = of_tok.acquire(1)
                token_fn(packed, tok, t, j0)
                of_tok.release(1)
            for _ in range_((hpc - 1 - h_local) * CS):
                tok = of_tok.acquire(1)
                of_tok.release(1)
            of_out.release(1)

    return core


@iron.jit
def ggml_xdna_gdn_prefill(
    tok: In,
    state: In,
    out: Out,
    *,
    S: CompileTime[int],
    H: CompileTime[int],
    CS: CompileTime[int],
    COLS: CompileTime[int],
    NC: CompileTime[int],
):
    if H % COLS:
        raise ValueError("H must be divisible by COLS")
    if NC != 1:
        # the state fifo is filled once, so a core asking for it NC times would
        # hang. Production chains chunks by binding chunk c's output BO view as
        # chunk c+1's state input, which keeps NC=1.
        raise ValueError("NC must be 1; chunks are chained host-side")
    if S % ROWS:
        raise ValueError("S must be divisible by ROWS")
    hpc = H // COLS
    n_strips = S // ROWS
    gr = hpc * n_strips
    if gr > CORES_PER_COL:
        raise ValueError(f"need {gr} cores/col, max {CORES_PER_COL}")
    tok_elems = 3 * S + 2
    packed_n = ROWS * S + CS * ROWS

    l1 = 2 * packed_n * 2 + 2 * tok_elems * 2
    if l1 > L1_BUDGET:
        raise ValueError(f"L1 needs {l1} B, over {L1_BUDGET}")

    token_fn, copy_fn = _fns(S, CS)

    tok_t_ty = np.ndarray[(tok_elems,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(packed_n,), np.dtype[bfloat16]]

    tok_col_n = hpc * CS * tok_elems
    packed_col_n = gr * packed_n

    tok_col_ty = np.ndarray[(tok_col_n,), np.dtype[bfloat16]]
    packed_col_ty = np.ndarray[(packed_col_n,), np.dtype[bfloat16]]

    tok_ty = np.ndarray[(NC * H * CS * tok_elems,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(H * n_strips * packed_n,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(NC * H * n_strips * packed_n,), np.dtype[bfloat16]]

    l2 = tok_col_n * 2 + packed_col_n * 2 + packed_col_n * 2
    if l2 > MEMTILE_BYTES:
        raise ValueError(f"MemTile needs {l2} B, over {MEMTILE_BYTES}")

    of_tok_shim = []
    of_sin = []
    of_out = []
    of_sin_shim = []
    of_out_shim = []

    for c in range(COLS):
        of_tok_shim.append(ObjectFifo(
            tok_col_ty,
            name=f"tok{c}",
            depth=1,
            consumer_obj_type=tok_t_ty,
        ))

        sin_shim = ObjectFifo(packed_col_ty, name=f"sin{c}", depth=1)
        of_sin.append(sin_shim.cons().split(
            offsets=[i * packed_n for i in range(gr)],
            obj_types=[packed_ty] * gr,
            depths=[1] * gr,
            names=[f"sin{c}_{i}" for i in range(gr)],
        ))
        of_sin_shim.append(sin_shim)

        out_shim = ObjectFifo(packed_col_ty, name=f"out{c}", depth=1)
        of_out.append(out_shim.prod().join(
            offsets=[i * packed_n for i in range(gr)],
            obj_types=[packed_ty] * gr,
            depths=[1] * gr,
            names=[f"out{c}_{i}" for i in range(gr)],
        ))
        of_out_shim.append(out_shim)

    workers = []
    for c in range(COLS):
        tok_cons = [of_tok_shim[c].cons(depth=2) for _ in range(gr)]
        for i in range(gr):
            h_local = i // n_strips
            j0 = (i % n_strips) * ROWS
            workers.append(Worker(
                _make_core(j0, h_local, hpc, NC),
                fn_args=[
                    tok_cons[i],
                    of_sin[c][i].cons(),
                    of_out[c][i].prod(),
                    token_fn, copy_fn, CS,
                ],
            ))

    def slice_tap(total, offset, count):
        return TensorAccessPattern([1, total], offset, [1, count], [0, 1])

    tok_chunk = H * CS * tok_elems
    packed_chunk = H * n_strips * packed_n

    p_sin = [of_sin_shim[c].prod() for c in range(COLS)]
    p_tok = [of_tok_shim[c].prod() for c in range(COLS)]
    c_out = [of_out_shim[c].cons() for c in range(COLS)]

    def seq(a_tok, a_st, a_out, ps, pt, co):
        for c in range(COLS):
            ps[c].fill(a_st,
                       tap=slice_tap(packed_chunk, c * packed_col_n, packed_col_n))
        for k in range(NC):
            last = k == NC - 1
            for c in range(COLS):
                pt[c].fill(a_tok,
                           tap=slice_tap(tok_chunk * NC, k * tok_chunk + c * tok_col_n, tok_col_n))
                co[c].drain(a_out,
                            tap=slice_tap(packed_chunk * NC, k * packed_chunk + c * packed_col_n, packed_col_n),
                            wait=last and c == COLS - 1)

    rt = Runtime(seq, [tok_ty, state_ty, out_ty, p_sin, p_tok, c_out])
    return Program(iron.get_current_device(), rt,
                   workers=workers).resolve_program()


def _compile_kwargs(opts) -> dict:
    return {"S": opts.S, "H": opts.heads, "CS": opts.CS, "COLS": opts.cols, "NC": opts.chunks}


def _ref(q, k, v, g, beta, state):
    S = q.shape[-1]
    CS = q.shape[0]
    scale = 1.0 / math.sqrt(S)
    s = state.copy()
    attn = np.empty((CS, S), dtype=np.float32)
    for t in range(CS):
        s *= math.exp(float(g[t]))
        delta = (v[t] - s @ k[t]) * float(beta[t])
        s += np.outer(delta, k[t])
        attn[t] = scale * (s @ q[t])
    return attn, s


def _run_and_verify(opts) -> None:
    S, H, CS, NC = opts.S, opts.heads, opts.CS, opts.chunks
    n_strips = S // ROWS
    tok_elems = 3 * S + 2
    packed_n = ROWS * S + CS * ROWS
    rng = np.random.default_rng(0)

    q = rng.standard_normal((NC, H, CS, S), dtype=np.float32)
    k = rng.standard_normal((NC, H, CS, S), dtype=np.float32)
    q /= np.linalg.norm(q, axis=3, keepdims=True)
    k /= np.linalg.norm(k, axis=3, keepdims=True)
    v = rng.standard_normal((NC, H, CS, S), dtype=np.float32)
    g = -np.abs(rng.standard_normal((NC, H, CS), dtype=np.float32)) * 0.01
    beta = np.clip(np.abs(rng.standard_normal((NC, H, CS), dtype=np.float32)), 0.01, 1.0)
    state = rng.standard_normal((H, S, S), dtype=np.float32) * 0.01

    pack = np.zeros((NC, H, CS, tok_elems), dtype=np.float32)
    for ck in range(NC):
        for h in range(H):
            for t in range(CS):
                pack[ck, h, t, :S] = q[ck, h, t]
                pack[ck, h, t, S:2 * S] = k[ck, h, t]
                pack[ck, h, t, 2 * S:3 * S] = v[ck, h, t]
                pack[ck, h, t, 3 * S] = np.exp(g[ck, h, t])
                pack[ck, h, t, 3 * S + 1] = beta[ck, h, t]

    tok = iron.tensor((NC * H * CS * tok_elems,), dtype=bfloat16, device="npu")
    st = iron.zeros((H * n_strips * packed_n,), dtype=bfloat16, device="npu")
    out = iron.zeros((NC * H * n_strips * packed_n,), dtype=bfloat16, device="npu")
    # Writes go through the overwrite() borrow, not numpy(): numpy() is the
    # read path (it reconciles from the device) and a write through it is
    # never recorded dirty, so it never reaches the device.
    with tok.overwrite() as _buf:
        np.copyto(_buf, pack.reshape(-1).astype(bfloat16))
    st_f = np.zeros((H, n_strips, packed_n), dtype=np.float32)
    for h in range(H):
        for ns in range(n_strips):
            st_f[h, ns, :ROWS * S] = state[h, ns * ROWS:(ns + 1) * ROWS].T.reshape(-1)
    with st.overwrite() as _buf:
        np.copyto(_buf, st_f.reshape(-1).astype(bfloat16))

    ggml_xdna_gdn_prefill(tok, st, out, **_compile_kwargs(opts))

    packed = out.numpy().astype(np.float32).reshape(NC, H, n_strips, packed_n)
    attn_ref = np.empty((NC, H, CS, S), dtype=np.float32)
    state_cur = state.copy()
    for ck in range(NC):
        for h in range(H):
            a, s = _ref(q[ck, h], k[ck, h], v[ck, h], g[ck, h], beta[ck, h], state_cur[h])
            attn_ref[ck, h] = a
            state_cur[h] = s

    attn_got = np.empty_like(attn_ref)
    state_got = np.empty_like(state_cur)
    for ck in range(NC):
        for h in range(H):
            for ns in range(n_strips):
                p = packed[ck, h, ns]
                attn_got[ck, h, :, ns * ROWS:(ns + 1) * ROWS] = p[ROWS * S:].reshape(CS, ROWS)
                if ck == NC - 1:
                    state_got[h, ns * ROWS:(ns + 1) * ROWS] = p[:ROWS * S].reshape(S, ROWS).T

    atol, rtol = 8e-2, 8e-2
    if opts.verbose:
        print(f"attn max abs: {np.max(np.abs(attn_ref - attn_got)):.6g}")
        print(f"state max abs: {np.max(np.abs(state_cur - state_got)):.6g}")
    assert_pass(attn_ref.reshape(-1), attn_got.reshape(-1), rtol=rtol, atol=atol,
                fail_msg="gdn-prefill attn mismatch")
    assert_pass(state_cur.reshape(-1), state_got.reshape(-1), rtol=rtol, atol=atol,
                fail_msg="gdn-prefill state mismatch")
    print(f"PASS: GDN prefill bf16 S={S} H={H} CS={CS} NC={NC} cols={opts.cols}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ggml-xdna-gdn-prefill")
    add_compile_args(parser)
    parser.add_argument("--S", type=int, default=DEFAULT_S)
    parser.add_argument("--heads", type=int, default=DEFAULT_H)
    parser.add_argument("--CS", type=int, default=DEFAULT_CS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--chunks", type=int, default=DEFAULT_NC)
    parser.add_argument("-v", "--verbose", action="store_true")
    opts = parser.parse_args()
    if opts.heads % opts.cols:
        raise SystemExit("--heads must be divisible by --cols")
    if opts.S % ROWS:
        raise SystemExit(f"--S must be divisible by {ROWS}")
    if opts.chunks < 1:
        raise SystemExit("--chunks must be >= 1")

    def device(o):
        return from_name(o.dev, n_cols=o.cols)

    run_design_cli(
        ggml_xdna_gdn_prefill,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device,
    )


if __name__ == "__main__":
    main()
