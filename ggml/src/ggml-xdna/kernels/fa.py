#!/usr/bin/env python3
# fa.py -*- Python -*-
#
"""Causal flash attention for the Qwen3.5 full-attention prefill layers.

  compile-only python3 fa.py -d npu2 --cols 8 --NMB 8 --NJ 256 \\
      --xclbin-path fa.xclbin --insts-path fa.insts.bin

  verify      python3 fa.py -d npu2 --cols 8 --KVH 2 --NMB 1 --NJ 64 --NCHUNK 4

One column owns one query head; its four cores each own a block of MT queries
and stream a chunk of NJ key tiles past it. The key range is walked in chunks,
one dispatch each, with the accumulator left where it is in L1 between them -
which is why NMB must be 1 when there is more than one chunk: with several
rounds per dispatch they share the one output buffer, and the next chunk would
resume from whichever round wrote it last.

So a dispatch covers COLS heads x ROWS*MT queries x NJ*JT keys, and the host
issues ceil(M / ROWS*MT) * ceil(n_kv / NJ*JT) of them per layer.

Host ABI (3 BOs):

  q  : [col][mb][row][hdr 16 x i32 | D*MT bf16]  the query block TRANSPOSED,
       behind a per-chunk header (zero, normalise, first tile, keys cached)
  kv : [kvhead][tile][K JT*D | V JT*D] bf16, keys/values as the cache has them
  o  : [col][mb][row][(D+2)*MT]     f32, O transposed then the softmax m and l

Q is transposed because the kernel keeps the queries on the vector lanes (see
fa.cc); that is what lets K and V stream in unchanged, which is the traffic
that matters. The trailing m/l of an output object are scratch - the host
reads the first D*MT floats.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path

import numpy as np

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
from ml_dtypes import bfloat16

DEFAULT_D = 256
DEFAULT_MT = 32
DEFAULT_JT = 8
DEFAULT_COLS = 8
DEFAULT_KVH = 2
CORES_PER_COL = 4
MAX_COLS = 8
L1_BUDGET = 60 * 1024


@contextlib.contextmanager
def _fill(t):
    if hasattr(t, "overwrite"):
        with t.overwrite() as buf:
            yield buf
        return
    yield t.data
    t._sync_to_device()


QHDR = 16   # int32 of chunk header ahead of a query block (64-byte aligned)


def _fa_fns(D: int, MT: int, JT: int, ROWS: int, scale: float):
    src = (Path(__file__).resolve().parent / "fa.cc").read_text()
    flags = [f"-DFA_D={D}", f"-DFA_MT={MT}", f"-DFA_JT={JT}",
             f"-DFA_ROWS={ROWS}", f"-DFA_SCALE={scale!r}f", f"-DFA_QHDR={QHDR}"]
    digest = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    acc_ty = np.ndarray[((D + 2) * MT,), np.dtype[np.float32]]
    q_ty = np.ndarray[(QHDR + D * MT // 2,), np.dtype[np.int32]]
    kv_ty = np.ndarray[(2 * JT * D,), np.dtype[bfloat16]]
    step = ExternalFunction(
        "ggml_xdna_fa_step",
        object_file_name=f"ggml_xdna_fa_{digest}.o",
        source_string=src,
        arg_types=[acc_ty, q_ty, kv_ty, np.int32, np.int32, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )
    zero = Kernel("ggml_xdna_fa_zero", step.object_file_name, [acc_ty])
    norm = Kernel("ggml_xdna_fa_norm", step.object_file_name, [acc_ty])
    return step, zero, norm


def _make_core(row: int, NMB: int, NJ: int):
    # Every query object opens with the chunk header (see fa.cc): zero,
    # normalise, first tile, keys cached before the batch. These change per
    # chunk, so they cannot be runtime parameters - an RTP is written by the
    # runtime sequence and so is baked into the instruction stream - and they
    # cannot be a stream of their own either, since a core has two input DMA
    # channels and K/V needs one. The zero and the normalise are loops that run
    # 0 or 1 times: an `if` on a runtime value is not expressible here, a
    # range_ of it is.
    def core(q_in, kv_in, o_out, zero_fn, step_fn, norm_fn):
        for b in range_(NMB):
            q = q_in.acquire(1)
            o = o_out.acquire(1)
            for _ in range_(q[0]):
                zero_fn(o)
            for t in range_(NJ):
                kv = kv_in.acquire(1)
                step_fn(o, q, kv, t, b, row)
                kv_in.release(1)
            for _ in range_(q[1]):
                norm_fn(o)
            o_out.release(1)
            q_in.release(1)

    return core


@iron.jit
def ggml_xdna_fa(
    q: In,
    kv: In,
    out: Out,
    *,
    D: CompileTime[int],
    MT: CompileTime[int],
    JT: CompileTime[int],
    COLS: CompileTime[int],
    KVH: CompileTime[int],
    NMB: CompileTime[int],
    NJ: CompileTime[int],
    SCALE_INV: CompileTime[int] = 16,
):
    ROWS = CORES_PER_COL
    if COLS % KVH:
        raise ValueError("COLS must be a multiple of KVH")
    acc_n = (D + 2) * MT
    q_n = QHDR + D * MT // 2      # int32: header then the bf16 query block
    kv_n = 2 * JT * D

    # A core holds one query block, one accumulator and KVDEPTH K/V tiles.
    # MT=32 fills the 512-bit vector, which is why the accumulator is 33 KB
    # and the K/V fifo cannot also be double buffered.
    KVDEPTH = 2 if MT <= 16 else 1
    l1 = q_n * 4 + acc_n * 4 + KVDEPTH * kv_n * 2
    if l1 > L1_BUDGET:
        raise ValueError(f"L1 needs {l1} B, over {L1_BUDGET}")

    scale = 1.0 / float(SCALE_INV)
    step_fn, zero_fn, norm_fn = _fa_fns(D, MT, JT, ROWS, scale)

    q_ty = np.ndarray[(q_n,), np.dtype[np.int32]]
    kv_ty = np.ndarray[(kv_n,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(acc_n,), np.dtype[np.float32]]
    q_col_ty = np.ndarray[(ROWS * q_n,), np.dtype[np.int32]]
    o_col_ty = np.ndarray[(ROWS * acc_n,), np.dtype[np.float32]]
    q_all_ty = np.ndarray[(COLS * NMB * ROWS * q_n,), np.dtype[np.int32]]
    kv_all_ty = np.ndarray[(KVH * NJ * kv_n,), np.dtype[bfloat16]]
    o_all_ty = np.ndarray[(COLS * NMB * ROWS * acc_n,), np.dtype[np.float32]]

    of_q, of_o = [], []
    q_shims, kv_shims, o_shims = [], [], []
    for c in range(COLS):
        qs = ObjectFifo(q_col_ty, name=f"inQ{c}", depth=1)
        of_q.append(qs.cons().split(
            offsets=[i * q_n for i in range(ROWS)],
            obj_types=[q_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"inQ{c}_{i}" for i in range(ROWS)],
        ))
        q_shims.append(qs)

        # Not split: every core of the column reads the same keys, so this is
        # one broadcast and the column touches its head's K/V once a round.
        kv_shims.append(ObjectFifo(kv_ty, name=f"inKV{c}", depth=KVDEPTH))

        os_ = ObjectFifo(o_col_ty, name=f"outO{c}", depth=1)
        of_o.append(os_.prod().join(
            offsets=[i * acc_n for i in range(ROWS)],
            obj_types=[acc_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"outO{c}_{i}" for i in range(ROWS)],
        ))
        o_shims.append(os_)

    workers = []
    for c in range(COLS):
        for i in range(ROWS):
            workers.append(Worker(
                _make_core(i, NMB, NJ),
                fn_args=[of_q[c][i].cons(), kv_shims[c].cons(),
                         of_o[c][i].prod(), zero_fn, step_fn, norm_fn],
                stack_size=0x1000,
            ))

    def slice_tap(total, offset, count):
        return TensorAccessPattern([1, total], offset, [1, count], [0, 1])

    q_col_span = NMB * ROWS * q_n
    o_col_span = NMB * ROWS * acc_n
    kv_head_span = NJ * kv_n
    heads_per_kv = COLS // KVH

    p_q = [q_shims[c].prod() for c in range(COLS)]
    p_kv = [kv_shims[c].prod() for c in range(COLS)]
    c_o = [o_shims[c].cons() for c in range(COLS)]

    def seq(a_q, a_kv, a_o, pq, pkv, co):
        for c in range(COLS):
            pq[c].fill(a_q, tap=slice_tap(COLS * q_col_span, c * q_col_span,
                                          q_col_span))
        for c in range(COLS):
            # One fill per round: a shim BD cannot express the replay itself -
            # a zero stride is rejected in any of its three dimensions ("BD
            # length does not match ... lowest three dimensions"). Four rounds
            # is the most that runs; past that the dispatch times out, which
            # is why MT is 32 and four rounds already cover a 512-token batch.
            for _ in range(NMB):
                pkv[c].fill(a_kv, tap=slice_tap(
                    KVH * kv_head_span,
                    (c // heads_per_kv) * kv_head_span, kv_head_span))
        for c in range(COLS):
            co[c].drain(a_o, tap=slice_tap(COLS * o_col_span, c * o_col_span,
                                           o_col_span),
                        wait=c == COLS - 1)

    rt = Runtime(seq, [q_all_ty, kv_all_ty, o_all_ty, p_q, p_kv, c_o])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _compile_kwargs(opts) -> dict:
    return {
        "D": opts.D, "MT": opts.MT, "JT": opts.JT, "COLS": opts.cols,
        "KVH": opts.KVH, "NMB": opts.NMB, "NJ": opts.NJ,
        "SCALE_INV": opts.scale_inv,
    }


def _reference(qf, kf, vf, scale, npast):
    """f32 causal attention, the ggml FLASH_ATTN_EXT semantics."""
    H, M, D = qf.shape
    KVH, NK, _ = kf.shape
    hpk = H // KVH
    out = np.zeros((H, M, D), dtype=np.float32)
    for h in range(H):
        k = kf[h // hpk]
        v = vf[h // hpk]
        for i in range(M):
            s = scale * (k @ qf[h, i])
            s[np.arange(NK) > npast + i] = -np.inf
            s -= s.max()
            p = np.exp(s)
            out[h, i] = (p / p.sum()) @ v
    return out


def _run_and_verify(opts) -> None:
    D, MT, JT, COLS, KVH = opts.D, opts.MT, opts.JT, opts.cols, opts.KVH
    NMB, NJ, NCHUNK = opts.NMB, opts.NJ, opts.NCHUNK
    if NCHUNK > 1 and NMB != 1:
        raise SystemExit("chained chunks need --NMB 1: the rounds of a "
                         "dispatch share one accumulator buffer")
    ROWS = CORES_PER_COL
    H, M = COLS, NMB * ROWS * MT
    NK = NCHUNK * NJ * JT
    NPAST = NK - M
    if NPAST < 0:
        raise SystemExit("n_kv (NCHUNK*NJ*JT) must be at least the batch")
    scale = 1.0 / float(opts.scale_inv)

    rng = np.random.default_rng(0)
    qf = rng.standard_normal((H, M, D), dtype=np.float32) * 0.5
    kf = rng.standard_normal((KVH, NK, D), dtype=np.float32) * 0.5
    vf = rng.standard_normal((KVH, NK, D), dtype=np.float32) * 0.5
    # Compare against the reference computed on what the device actually sees.
    ref = _reference(qf.astype(bfloat16).astype(np.float32),
                     kf.astype(bfloat16).astype(np.float32),
                     vf.astype(bfloat16).astype(np.float32), scale, NPAST)

    acc_n = (D + 2) * MT
    q_n = QHDR + D * MT // 2
    q_h = np.zeros((COLS, NMB, ROWS, q_n), dtype=np.int32)
    q_bf = q_h.view(bfloat16).reshape(COLS, NMB, ROWS, 2 * q_n)
    for c in range(COLS):
        for b in range(NMB):
            for r in range(ROWS):
                m0 = (b * ROWS + r) * MT
                q_bf[c, b, r, 2 * QHDR:2 * QHDR + D * MT] = \
                    qf[c, m0:m0 + MT].T.reshape(-1).astype(bfloat16)

    a_q = iron.tensor((COLS * NMB * ROWS * q_n,), dtype=np.int32, device="npu")
    a_o = iron.zeros((COLS * NMB * ROWS * acc_n,), dtype=np.float32, device="npu")

    for ch in range(NCHUNK):
        q_h[:, :, :, 0] = int(ch == 0)
        q_h[:, :, :, 1] = int(ch == NCHUNK - 1)
        q_h[:, :, :, 2] = ch * NJ
        q_h[:, :, :, 3] = NPAST
        kv_h = np.zeros((KVH, NJ, 2, JT, D), dtype=bfloat16)
        for g in range(KVH):
            for t in range(NJ):
                j0 = (ch * NJ + t) * JT
                kv_h[g, t, 0] = kf[g, j0:j0 + JT].astype(bfloat16)
                kv_h[g, t, 1] = vf[g, j0:j0 + JT].astype(bfloat16)
        a_kv = iron.tensor((KVH * NJ * 2 * JT * D,), dtype=bfloat16, device="npu")
        with _fill(a_q) as buf:
            np.copyto(buf, q_h.reshape(-1))
        with _fill(a_kv) as buf:
            np.copyto(buf, kv_h.reshape(-1))
        ggml_xdna_fa(a_q, a_kv, a_o, **_compile_kwargs(opts))

    got_raw = a_o.numpy().reshape(COLS, NMB, ROWS, acc_n)
    got = np.zeros((H, M, D), dtype=np.float32)
    for c in range(COLS):
        for b in range(NMB):
            for r in range(ROWS):
                m0 = (b * ROWS + r) * MT
                got[c, m0:m0 + MT] = got_raw[c, b, r, :D * MT].reshape(D, MT).T

    assert_pass(ref.reshape(-1), got.reshape(-1), rtol=2e-2, atol=2e-2,
                fail_msg="ggml-xdna flash attention mismatch vs NumPy")
    print(f"PASS: NPU flash attention D={D} MT={MT} JT={JT} heads={H} "
          f"kvheads={KVH} M={M} n_kv={NK} chunks={NCHUNK} npast={NPAST}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ggml-xdna-fa",
        description="Build/run the ggml-xdna NPU flash attention kernel",
    )
    add_compile_args(parser)
    parser.add_argument("--D", type=int, default=DEFAULT_D)
    parser.add_argument("--MT", type=int, default=DEFAULT_MT)
    parser.add_argument("--JT", type=int, default=DEFAULT_JT)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--KVH", type=int, default=DEFAULT_KVH)
    parser.add_argument("--NMB", type=int, default=1)
    parser.add_argument("--NJ", type=int, default=8)
    parser.add_argument("--NCHUNK", type=int, default=1,
                        help="key chunks; n_kv = NCHUNK*NJ*JT, one dispatch each")
    parser.add_argument("--scale-inv", type=int, default=16,
                        help="1/scale; 16 is 1/sqrt(256)")
    opts = parser.parse_args()
    if opts.cols < 1 or opts.cols > MAX_COLS:
        raise SystemExit(f"--cols must be in 1..{MAX_COLS}")

    def device(o):
        return from_name(o.dev, n_cols=o.cols)

    run_design_cli(
        ggml_xdna_fa,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device,
    )


if __name__ == "__main__":
    main()
