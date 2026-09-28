#!/usr/bin/env python3
"""Reference check of the fused recurrent core's norm and state-update stages.

  python3 fused_core_check.py -d npu2 --workdir build/bin

Two checks, both against what ggml's CPU backend computes for the same
inputs, and the script exits nonzero if either fails:

norm   attn-norm.cc (ggml_xdna_attn_norm) on one design core, over 16 heads
       whose q and k are chosen to cover the cases the norm has to get right:
       both zero, one zero, ordinary, large, one-hot and constant. Every one
       of a head's eight pkv chunks
       [kn | qn | v16 | eg | b | scale] is compared: kn and qn against
       ggml_l2_norm in fp32, the copied fields bit for bit.

state  The norm's device output fed straight to gdn-v.cc (gdn_v.py's design)
       for three tokens, the state carried between them in the device BO and
       starting from a nonzero one, against ggml's CPU gated delta rule
       (ggml_compute_forward_gated_delta_net: S *= exp(g); delta = (v - S k)
       beta; S += k delta; o = S q / sqrt(S_v)) run in fp32 on the fp32
       inputs. The device keeps the state in bf16, so it is also compared
       against gdn_v.py's bf16 emulation of the kernel's own op sequence, to
       tell a kernel error from bf16 drift.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cn  # noqa: E402
import gdn_v  # noqa: E402

S_V = attn_cn.S_V
N_VH = attn_cn.N_VH
CHUNK = attn_cn.CHUNK
N_OBJ = attn_cn.N_OBJ
PKV_N = attn_cn.PKV_N
HEAD_NORM = attn_cn.HEAD_NORM
PKVB_N = attn_cn.PKVB_N
L2_EPS = 1e-6          # Qwen3.5's f_norm_rms_eps, what ggml_l2_norm gets
SCALE = np.float32(1.0 / np.sqrt(S_V))


@iron.jit
def norm_check(*, dev_name: CompileTime[str] = "npu2"):
    HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
    PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]
    kern = iron.ExternalFunction(name="ggml_xdna_attn_norm",
                                 source_string=attn_cn._norm_src(),
                                 arg_types=[HN_T, PKVB_T],
                                 compile_flags=["-O2", "-DNDEBUG"], inline=True)

    i3 = ObjectFifo(HN_T, name="ni3", depth=2)
    i2 = i3.cons().forward(obj_type=HN_T, name="ni2", tile=Tile(0, 1))
    o23 = ObjectFifo(PKVB_T, name="no23", depth=2)
    o12 = o23.prod().join([0], obj_types=[PKVB_T], names=["no12"],
                          depths=[2], tile=Tile(0, 1))[0]

    def norm_fn(ic, oc, k):
        for _ in range_(N_VH):
            x = ic.acquire(1)
            o = oc.acquire(1)
            k(x, o)
            ic.release(1)
            oc.release(1)

    workers = [Worker(norm_fn, [i2.cons(), o12.prod(), kern],
                      tile=Tile(0, 2), stack_size=0x3000)]

    IN_g = np.ndarray[(N_VH * HEAD_NORM,), np.dtype[np.float32]]
    OUT_g = np.ndarray[(N_VH * PKVB_N,), np.dtype[np.float32]]

    def seq_fn(IN, OUT, inf, outf):
        gi = TaskGroup()
        inf.fill(IN, tap=TensorAccessPattern((N_VH * HEAD_NORM,), offset=0,
                                             sizes=[N_VH * HEAD_NORM], strides=[1]),
                 group=gi)
        gi.finish()
        go = TaskGroup()
        outf.drain(OUT, tap=TensorAccessPattern((N_VH * PKVB_N,), offset=0,
                                                sizes=[N_VH * PKVB_N], strides=[1]),
                   wait=True, group=go)
        go.finish()

    rt = Runtime(seq_fn, [IN_g, OUT_g, i3.prod(tile=Tile(0, 0)), o23.cons(tile=Tile(0, 0))])
    return Program(from_name(dev_name, n_cols=1), rt, workers).resolve_program()


# ---- what ggml's CPU backend computes ------------------------------------

def ggml_l2_norm(x: np.ndarray) -> np.ndarray:
    """ggml_compute_forward_l2_norm_f32: x / max(|x|, eps), in fp32."""
    x = x.astype(np.float32)
    s = np.float32(np.sum(x * x, dtype=np.float32))
    return x * np.float32(1.0 / max(np.sqrt(s), np.float32(L2_EPS)))


def ggml_gdn_token(S, q, k, v, g, beta):
    """One token of ggml_compute_forward_gated_delta_net_one_chunk for one head
    (scalar gate), fp32. S [S_v, S_v] row j = value dim. Returns (S', o)."""
    S = S.astype(np.float32) * np.float32(np.exp(np.float32(g)))
    delta = (v.astype(np.float32) - S @ k.astype(np.float32)) * np.float32(beta)
    S = S + np.outer(delta, k.astype(np.float32)).astype(np.float32)
    o = (S @ q.astype(np.float32)) * SCALE
    return S.astype(np.float32), o.astype(np.float32)


# ---- inputs ----------------------------------------------------------------

def head_cases(rng, t):
    """16 heads of [q | k | v] with the norm's edge cases in the first eight;
    the second eight are ordinary, so the state update sees real work on
    every chunk. `t` varies them per token."""
    q = rng.standard_normal((N_VH, S_V)).astype(np.float32)
    k = rng.standard_normal((N_VH, S_V)).astype(np.float32)
    q[0] = 0.0
    k[0] = 0.0                                   # both zero
    q[1] = 0.0                                   # q zero, k not
    q[3] *= 1.0e3                                # large
    k[3] *= 1.0e3
    q[4] = 0.0
    q[4][(5 + t) % S_V] = 1.0                    # one-hot
    k[4] = 0.0
    k[4][(17 + 3 * t) % S_V] = -2.0
    q[5] = 0.25                                  # constant
    k[5] = -0.5
    v = rng.standard_normal((N_VH, S_V)).astype(np.float32)
    g = -rng.uniform(0.01, 2.0, N_VH).astype(np.float32)       # log decay
    beta = rng.uniform(0.05, 0.95, N_VH).astype(np.float32)
    return q, k, v, g, beta


def norm_input(q, k, v, g, beta):
    x = np.zeros((N_VH, HEAD_NORM), dtype=np.float32)
    x[:, 0:S_V] = q
    x[:, S_V:2 * S_V] = k
    x[:, 2 * S_V:3 * S_V] = v
    x[:, 3 * S_V] = np.exp(g)
    x[:, 3 * S_V + 1] = beta
    x[:, 3 * S_V + 2] = SCALE
    return x


# ---- device ----------------------------------------------------------------

class Kernel:
    def __init__(self, xrt, dev, xclbin_path, insts_path):
        self.xrt = xrt
        self.dev = dev
        xb = xrt.xclbin(str(xclbin_path))
        dev.register_xclbin(xb)
        self.ctx = xrt.hw_context(dev, xb.get_uuid())
        self.kernel = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        insts = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
        self.insts_bo = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, self.kernel.group_id(1))
        np.frombuffer(self.insts_bo.map(), dtype=np.uint32)[:] = insts
        self.insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.n_words = int(insts.size)     # argument 2 counts 32-bit words

    def bo(self, arr):
        xrt = self.xrt
        bo = xrt.bo(self.dev, arr.nbytes, xrt.bo.host_only, 0)
        self.put(bo, arr)
        return bo

    def put(self, bo, arr):
        xrt = self.xrt
        np.frombuffer(bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def get(self, bo, dtype, n):
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(bo.map(), dtype=dtype)[:n].copy()

    def run(self, *bos):
        r = self.kernel(3, self.insts_bo, self.n_words, *bos)
        if r.wait() != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit("fused_core_check: a run did not complete")


def compile_design(fn, wd, stem, **kw):
    xclbin_path = os.path.join(wd, stem + ".xclbin")
    insts_path = os.path.join(wd, stem + ".insts.bin")
    t0 = time.time()
    xclbin_path, insts_path = fn.specialize(**kw).compile(xclbin_path=xclbin_path,
                                                          inst_path=insts_path)
    print(f"compiled {xclbin_path} ({time.time() - t0:.0f}s)")
    return xclbin_path, insts_path


# ---- checks ----------------------------------------------------------------

def check_norm(pkv_dev, x, failures, tol):
    """pkv_dev [N_VH, N_OBJ, PKV_N] from the device; x the norm input."""
    q, k = x[:, 0:S_V], x[:, S_V:2 * S_V]
    worst = 0.0
    for h in range(N_VH):
        qn, kn = ggml_l2_norm(q[h]), ggml_l2_norm(k[h])
        for j in range(N_OBJ):
            o = pkv_dev[h, j]
            if not np.all(np.isfinite(o)):
                failures.append(f"norm head {h} chunk {j}: non-finite output")
                continue
            for name, got, ref in (("kn", o[0:S_V], kn), ("qn", o[S_V:2 * S_V], qn)):
                err = float(np.abs(got - ref).max())
                worst = max(worst, err)
                if not err <= tol:
                    failures.append(f"norm head {h} chunk {j} {name}: max abs err {err:.3e}")
            exact = np.concatenate([o[2 * S_V:2 * S_V + CHUNK], o[3 * S_V:3 * S_V + 3]])
            want = np.concatenate([x[h, 2 * S_V + j * CHUNK:2 * S_V + (j + 1) * CHUNK],
                                   x[h, 3 * S_V:3 * S_V + 3]])
            if not np.array_equal(exact.view(np.uint32), want.view(np.uint32)):
                failures.append(f"norm head {h} chunk {j}: v16/eg/b/scale not copied bit for bit")
        if h in (0, 1) and np.any(pkv_dev[h, :, S_V:2 * S_V] != 0.0):
            failures.append(f"norm head {h}: zero q did not give exactly zero qn")
        if h == 0 and np.any(pkv_dev[h, :, 0:S_V] != 0.0):
            failures.append("norm head 0: zero k did not give exactly zero kn")
    return worst


def dev_state_to_f32(raw):
    """Device state BO (bf16, [head][chunk][16 rows][128]) -> [N_VH, S_V, S_V]."""
    return np.asarray(raw, dtype=np.float32).reshape(N_VH, N_OBJ * CHUNK, S_V)


def f32_to_dev_state(S, bf16):
    return np.ascontiguousarray(S.reshape(-1)).astype(bf16)


def main():
    ap = argparse.ArgumentParser(prog="fused_core_check")
    add_compile_args(ap)
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--norm-tol", type=float, default=8e-3,
                    help="max abs error of a unit-vector element (bf16 input rounding)")
    ap.add_argument("--state-tol", type=float, default=0.05,
                    help="relative error against ggml's fp32 recurrence (bf16 state)")
    ap.add_argument("--sim-tol", type=float, default=2e-3,
                    help="relative error against the bf16 emulation of the kernel")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.tag is None:
        opts.tag = time.strftime("corecheck_%H%M%S")
    wd = os.path.join(opts.workdir, opts.tag)
    os.makedirs(wd, exist_ok=True)

    norm_x, norm_i = compile_design(norm_check, wd, "norm_check", dev_name=opts.dev)
    gdn_x, gdn_i = compile_design(gdn_v.ggml_xdna_gdn_v, wd, "gdn", ncol=gdn_v.NCOL,
                                  dev_name=opts.dev)

    import pyxrt as xrt
    from ml_dtypes import bfloat16

    dev = xrt.device(0)
    nk = Kernel(xrt, dev, norm_x, norm_i)
    gk = Kernel(xrt, dev, gdn_x, gdn_i)

    rng = np.random.default_rng(291)
    failures = []

    # A nonzero starting state, so the first update already has a history to
    # decay and correct. Both sides start from the same bf16 values.
    S0 = (0.1 * rng.standard_normal((N_VH, S_V, S_V))).astype(np.float32)
    S0 = np.asarray(S0.astype(bfloat16), dtype=np.float32)
    S_cpu = S0.copy()
    S_sim = S0.copy()

    in_bo = nk.bo(np.zeros(N_VH * HEAD_NORM, dtype=np.float32))
    pkv_bo = nk.bo(np.zeros(N_VH * PKVB_N, dtype=np.float32))
    state_bo = gk.bo(f32_to_dev_state(S0, bfloat16))
    attn_bo = gk.bo(np.zeros(N_VH * S_V, dtype=np.float32))

    worst_norm = worst_state = worst_attn = worst_sim = 0.0
    for t in range(opts.tokens):
        q, k, v, g, beta = head_cases(rng, t)
        x = norm_input(q, k, v, g, beta)

        nk.put(in_bo, x)
        nk.run(in_bo, pkv_bo)
        pkv = nk.get(pkv_bo, np.float32, N_VH * PKVB_N).reshape(N_VH, N_OBJ, PKV_N)
        worst_norm = max(worst_norm, check_norm(pkv, x, failures, opts.norm_tol))

        # The gdn stage takes the norm's output as it is: chained exactly as
        # the fused layer chains them.
        gk.put(pkv_bo, pkv.reshape(-1))
        gk.run(pkv_bo, state_bo, state_bo, attn_bo)
        attn_dev = gk.get(attn_bo, np.float32, N_VH * S_V).reshape(N_VH, S_V)
        S_dev = dev_state_to_f32(gk.get(state_bo, bfloat16, N_VH * S_V * S_V))

        attn_cpu = np.zeros((N_VH, S_V), dtype=np.float32)
        attn_sim = np.zeros((N_VH, S_V), dtype=np.float32)
        for h in range(N_VH):
            S_cpu[h], attn_cpu[h] = ggml_gdn_token(S_cpu[h], ggml_l2_norm(q[h]),
                                                   ggml_l2_norm(k[h]), v[h], g[h], beta[h])
            for j in range(N_OBJ):
                rows = slice(j * CHUNK, (j + 1) * CHUNK)
                S_sim[h, rows], attn_sim[h, rows] = gdn_v.sim_chunk_bf16(pkv[h, j], S_sim[h, rows])

        def rel(a, b):
            return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))

        e_state, e_attn = rel(S_dev, S_cpu), rel(attn_dev, attn_cpu)
        e_sim = max(rel(S_dev, S_sim), rel(attn_dev, attn_sim))
        worst_state, worst_attn = max(worst_state, e_state), max(worst_attn, e_attn)
        worst_sim = max(worst_sim, e_sim)
        print(f"token {t}: state rel err {e_state:.3e}, attn rel err {e_attn:.3e} "
              f"(vs ggml fp32); vs bf16 emulation {e_sim:.3e}")
        if not e_state <= opts.state_tol:
            failures.append(f"token {t}: state rel err {e_state:.3e} over {opts.state_tol}")
        if not e_attn <= opts.state_tol:
            failures.append(f"token {t}: attn rel err {e_attn:.3e} over {opts.state_tol}")
        if not e_sim <= opts.sim_tol:
            failures.append(f"token {t}: bf16 emulation rel err {e_sim:.3e} over {opts.sim_tol}")
        if np.any(attn_dev[0:2] != 0.0):
            failures.append(f"token {t}: a zero q did not give exactly zero attn")

    print(f"norm: worst max abs err {worst_norm:.3e} (tolerance {opts.norm_tol:.1e})")
    print(f"state: worst rel err {worst_state:.3e}, attn {worst_attn:.3e} "
          f"(tolerance {opts.state_tol:.1e}); vs bf16 emulation {worst_sim:.3e} "
          f"(tolerance {opts.sim_tol:.1e})")
    if failures:
        for f in failures[:40]:
            print("FAIL", f)
        sys.exit(f"fused_core_check: {len(failures)} failures")
    print("fused_core_check: PASS")


if __name__ == "__main__":
    main()
