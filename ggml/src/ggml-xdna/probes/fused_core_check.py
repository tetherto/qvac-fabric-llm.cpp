#!/usr/bin/env python3
"""Reference check of the fused core's norm stage and its state update.

`attn-norm.cc` and `gdn-v.cc` are the two device kernels of the fused recurrent
layer that no standalone harness covers: `fused_layer.py` compiles them into the
merged design and only the model exercises them end to end. This probe compiles
each one on its own and compares it with the same math in NumPy.

  python3 probes/fused_core_check.py -d npu2

norm: 12 heads whose q and k cover zero, one zero, ordinary, large, one-hot and
constant vectors; all eight pkv chunks of every head are compared - kn and qn
against the fp32 L2 norm (the device rounds q/k and the inverse norm to bf16, so
the tolerance covers that), v16/eg/b/scale bit for bit. A zero q must give
exactly zero qn.

state: one head's device norm output is fed to gdn-v for three tokens, starting
from a nonzero state and carrying the device rows between tokens, against the
fp32 gated-delta rule written in gdn-v.cc's header comment.
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
import ml_dtypes
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile, from_name

KERNELS = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(KERNELS))
import fused_layer as fl  # noqa: E402

bf16 = ml_dtypes.bfloat16

S_V = fl.S_V
CHUNK = fl.CHUNK
N_OBJ = fl.N_OBJ
PKV_N = fl.PKV_N
PKVB_N = fl.PKVB_N
HEAD_NORM = fl.HEAD_NORM
ROWS = fl.ROWS
O_V = fl.O_V
O_EG = fl.O_EG

HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]
PKV_T = np.ndarray[(PKV_N,), np.dtype[np.float32]]
ROWS_T = np.ndarray[(ROWS,), np.dtype[bf16]]
ATT_T = np.ndarray[(CHUNK,), np.dtype[np.float32]]

NORM_TOL = 5.0e-2  # the bf16 rounding of q/k and of the inverse norm
STATE_TOL = 5.0e-2  # the kernel's bf16 vectors with fp32 accumulators


def norm_kernel():
    src = fl._norm_src("ggml_xdna_attn_norm", -1, -1)
    return iron.ExternalFunction(name="ggml_xdna_attn_norm", source_string=src,
                                 arg_types=[HN_T, PKVB_T], compile_flags=["-O2", "-DNDEBUG"],
                                 inline=True)


def gdn_kernel():
    return iron.ExternalFunction(name="ggml_xdna_gdn_v", source_string=fl._kernel_src(),
                                 arg_types=[PKV_T, ROWS_T, ROWS_T, ATT_T],
                                 compile_flags=["-O2", "-DNDEBUG"], inline=True)


def norm_design():
    @iron.jit
    def norm(inp: In, outp: Out):
        kfn = norm_kernel()
        fa = ObjectFifo(HN_T, name="fa", depth=1)
        fb = ObjectFifo(PKVB_T, name="fb", depth=1)

        def body(ai, bo, k):
            a = ai.acquire(1)
            b = bo.acquire(1)
            k(a, b)
            ai.release(1)
            bo.release(1)

        w = Worker(body, fn_args=[fa.cons(), fb.prod(), kfn], tile=Tile(0, 2), stack_size=0xD00)

        def seq(a_h, b_h, ai, bo):
            ai.fill(a_h)
            bo.drain(b_h, wait=True)

        rt = Runtime(seq, [HN_T, PKVB_T, fa.prod(), fb.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return norm


def gdn_design():
    @iron.jit
    def gdn(pkv: In, rows: In, sout: Out, attn: Out):
        kfn = gdn_kernel()
        fp = ObjectFifo(PKV_T, name="fp", depth=1)
        fr = ObjectFifo(ROWS_T, name="fr", depth=1)
        fs = ObjectFifo(ROWS_T, name="fs", depth=1)
        ft = ObjectFifo(ATT_T, name="ft", depth=1)

        def body(pi, ri, so, to, k):
            p = pi.acquire(1)
            r = ri.acquire(1)
            s = so.acquire(1)
            t = to.acquire(1)
            k(p, r, s, t)
            pi.release(1)
            ri.release(1)
            so.release(1)
            to.release(1)

        w = Worker(body, fn_args=[fp.cons(), fr.cons(), fs.prod(), ft.prod(), kfn], tile=Tile(0, 2),
                   stack_size=0x3000)

        def seq(p_h, r_h, s_h, t_h, pi, ri, so, to):
            pi.fill(p_h)
            ri.fill(r_h)
            so.drain(s_h, wait=True)
            to.drain(t_h, wait=True)

        rt = Runtime(seq, [PKV_T, ROWS_T, ROWS_T, ATT_T, fp.prod(), fr.prod(), fs.cons(), ft.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return gdn


def run_norm(design, hn):
    a = iron.tensor((HEAD_NORM,), dtype=np.float32, device="npu")
    b = iron.zeros((PKVB_N,), dtype=np.float32, device="npu")
    np.copyto(a.numpy(), hn)
    a._sync_to_device()
    design(a, b)
    b._sync_from_device()
    return b.numpy().copy()


def run_gdn(design, pkv, rows):
    p = iron.tensor((PKV_N,), dtype=np.float32, device="npu")
    r = iron.tensor((ROWS,), dtype=bf16, device="npu")
    s = iron.zeros((ROWS,), dtype=bf16, device="npu")
    t = iron.zeros((CHUNK,), dtype=np.float32, device="npu")
    np.copyto(p.numpy(), pkv)
    np.copyto(r.numpy(), rows.astype(bf16))
    p._sync_to_device()
    r._sync_to_device()
    design(p, r, s, t)
    s._sync_from_device()
    t._sync_from_device()
    return s.numpy().copy(), t.numpy().copy()


def ref_norm(hn):
    """The fp32 L2 norm of the head's q and k."""
    q = hn[0:S_V].astype(np.float64)
    k = hn[S_V:2 * S_V].astype(np.float64)
    qn = (q / np.sqrt((q * q).sum())).astype(np.float32) if q.any() else np.zeros(S_V, np.float32)
    kn = (k / np.sqrt((k * k).sum())).astype(np.float32) if k.any() else np.zeros(S_V, np.float32)
    return qn, kn


def ref_gdn(pkv, rows):
    """gdn-v.cc's header comment in fp32, state row-major A[j][i]."""
    kb = pkv[0:S_V].astype(np.float32).astype(bf16).astype(np.float32)
    qb = pkv[S_V:2 * S_V].astype(np.float32).astype(bf16).astype(np.float32)
    v16 = pkv[O_V:O_V + CHUNK].astype(np.float32)
    eg, b, scale = float(pkv[O_EG]), float(pkv[O_EG + 1]), float(pkv[O_EG + 2])
    R = rows.astype(np.float32).reshape(CHUNK, S_V)
    kq = float(np.dot(kb, qb))
    sout = np.empty((CHUNK, S_V), np.float32)
    attn = np.empty(CHUNK, np.float32)
    for j in range(CHUNK):
        dotk = float(np.dot(R[j], kb))
        dj = (v16[j] - eg * dotk) * b
        dotq = float(np.dot(R[j], qb))
        sout[j] = R[j] * eg + dj * kb
        attn[j] = (eg * dotq + dj * kq) * scale
    return sout.astype(bf16), attn


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30)))


def make_heads(seed=3):
    rng = np.random.default_rng(seed)
    zero = np.zeros(S_V, np.float32)
    one = np.zeros(S_V, np.float32)
    one[0] = 1.0
    one[S_V // 2] = 1.0
    heads = []
    for name, q, k in [("zero q", zero, rng.standard_normal(S_V).astype(np.float32)),
                       ("one zero", rng.standard_normal(S_V).astype(np.float32), zero),
                       ("ordinary", rng.standard_normal(S_V).astype(np.float32),
                        rng.standard_normal(S_V).astype(np.float32)),
                       ("large", rng.standard_normal(S_V).astype(np.float32) * 50.0,
                        rng.standard_normal(S_V).astype(np.float32) * 50.0),
                       ("one-hot", one, rng.standard_normal(S_V).astype(np.float32)),
                       ("constant", np.ones(S_V, np.float32) * 0.3, np.ones(S_V, np.float32) * 0.3)]:
        for extra in range(2):
            hn = np.zeros(HEAD_NORM, np.float32)
            hn[0:S_V] = q
            hn[S_V:2 * S_V] = k
            hn[2 * S_V:3 * S_V] = rng.standard_normal(S_V).astype(np.float32)
            hn[O_EG] = float(rng.uniform(0.5, 1.5))
            hn[O_EG + 1] = float(rng.uniform(0.2, 1.0))
            hn[O_EG + 2] = float(rng.uniform(0.05, 0.2))
            heads.append((f"{name}#{extra}", hn))
    return heads


def main():
    ap = argparse.ArgumentParser(prog="fused_core_check")
    ap.add_argument("-d", "--dev", default="npu2")
    args = ap.parse_args()

    iron.set_current_device(from_name(args.dev, n_cols=1))
    norm = norm_design()
    gdn = gdn_design()

    heads = make_heads()
    failures = 0
    for name, hn in heads:
        got = run_norm(norm, hn).reshape(N_OBJ, PKV_N)
        qn_ref, kn_ref = ref_norm(hn)
        for j in range(N_OBJ):
            o = got[j]
            kn, qn = o[0:S_V], o[S_V:2 * S_V]
            if rel(kn, kn_ref) > NORM_TOL or rel(qn, qn_ref) > NORM_TOL:
                print(f"FAIL norm {name} chunk {j}: kn rel {rel(kn, kn_ref):.3e} qn rel {rel(qn, qn_ref):.3e}")
                failures += 1
            if not np.array_equal(o[O_V:O_V + CHUNK], hn[2 * S_V + j * CHUNK:2 * S_V + (j + 1) * CHUNK]):
                print(f"FAIL norm {name} chunk {j}: v16 is not the input's slice")
                failures += 1
            if not (o[O_EG] == np.float32(hn[O_EG]) and o[O_EG + 1] == np.float32(hn[O_EG + 1]) and
                    o[O_EG + 2] == np.float32(hn[O_EG + 2])):
                print(f"FAIL norm {name} chunk {j}: eg/b/scale changed")
                failures += 1
        if not np.any(hn[0:S_V]) and np.any(got[:, S_V:2 * S_V] != 0.0):
            print(f"FAIL norm {name}: a zero q did not give exactly zero qn")
            failures += 1
    print(f"norm: {len(heads)} heads x {N_OBJ} chunks, every kn/qn within {NORM_TOL:.1e}, "
          f"v16/eg/b/scale exact; {failures} failure(s)")

    # One state update per token, three tokens, carrying the device rows.
    rng = np.random.default_rng(7)
    pkv = run_norm(norm, heads[4][1])[0:PKV_N]
    rows = (rng.standard_normal(ROWS) * 0.1).astype(bf16)
    worst_s, worst_a = 0.0, 0.0
    for tok in range(3):
        want_s, want_a = ref_gdn(pkv, rows)
        got_s, got_a = run_gdn(gdn, pkv, rows)
        rows = got_s
        worst_s = max(worst_s, rel(got_s.astype(np.float32).reshape(CHUNK, S_V), want_s.astype(np.float32)))
        worst_a = max(worst_a, rel(got_a, want_a))
        print(f"state token {tok + 1}: rows rel {rel(got_s.astype(np.float32).reshape(CHUNK, S_V), want_s.astype(np.float32)):.3e}, "
              f"attn rel {rel(got_a, want_a):.3e}")
    print(f"state: worst rows rel {worst_s:.3e}, worst attn rel {worst_a:.3e} "
          f"(tolerance {STATE_TOL:.1e})")
    if worst_s > STATE_TOL or worst_a > STATE_TOL:
        failures += 1

    if failures:
        print(f"fused_core_check: FAIL ({failures} failure(s))")
        return 1
    print("fused_core_check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
