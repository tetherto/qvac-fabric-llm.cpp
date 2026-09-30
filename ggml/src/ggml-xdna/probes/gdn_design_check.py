# The prefill gated delta rule on the whole array (kernels/gdn_mm.py) against
# ggml's token recurrence: 16 heads, S = 128, NCH chunks of 16 tokens.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/gdn_design_check.py [chunks] [gate scale] [corr]
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gdn_mm as gm  # noqa: E402

NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
# "corr": a chunk's keys close to each other and beta near 1 - where the
# triangular inverse's doubling form blew up
CORR = len(sys.argv) > 3 and sys.argv[3] == "corr"
bf16 = ml_dtypes.bfloat16
COLS, C, DK, DV, IN_N, O_N = gm.COLS, gm.C, gm.DK, gm.DV, gm.IN_N, gm.O_N
H, D, T = 16, 128, NCH * gm.C
NX, NO = 1 + gm.N_STATE_IN + NCH, NCH + gm.N_STATE_OUT


def blk(x):
    r, c = x.shape
    return x.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


def unblk(x, r, c):
    return x.reshape(r // 8, c // 8, 8, 8).transpose(0, 2, 1, 3).reshape(r, c)


def scalars(g, beta):
    """A head's per-chunk factor region (as bf16 words: what the core works
    its factors out of - the gates, beta, log2 sigma before and after the
    renormalization, the shift) and its final log2 sigma."""
    g2 = g * np.log2(np.e)
    n_fx = 4 * C + 2 * C * C + 32
    out, ls = [], 0.0
    for c in range(len(g) // C):
        ls_new = ls + np.float32(g2[c * C:(c + 1) * C]).sum()
        shift = int(-ls_new) if ls_new < -32 else 0
        f = np.zeros(n_fx, np.float32)
        f[:C] = g[c * C:(c + 1) * C]
        f[C:2 * C] = beta[c * C:(c + 1) * C]
        f[2 * C] = ls
        f[2 * C + 1] = ls_new + shift             # sigma' after the shift, before the update
        f.view(np.int32)[2 * C + 2] = shift
        out.append(f.view(bf16))
        ls = ls_new + shift
    return out, ls


rng = np.random.default_rng(5)
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
q = nrm(rng.standard_normal((T, H, D))); k = nrm(rng.standard_normal((T, H, D)))
if CORR:
    base = rng.standard_normal((T // 16, H, D))
    k = nrm(np.repeat(base, 16, axis=0) + 0.1 * rng.standard_normal((T, H, D)))
v = rng.standard_normal((T, H, D)); beta = 1 / (1 + np.exp(-rng.standard_normal((T, H)) - (3.0 if CORR else 0.0)))
g = -np.log1p(np.exp(rng.standard_normal((T, H)))) * GS
S0 = rng.standard_normal((H, D, D)) * 0.1
qb, kb, vb = (x.astype(np.float32).astype(bf16) for x in (q, k, v))

xin = np.zeros((COLS, NX, 4, IN_N), bf16)
lss = np.zeros(H)
for col in range(COLS):
    for i in range(4):
        hd, h = gm.core_of(col, i)
        fx, lss[hd] = scalars(g[:, hd], beta[:, hd])
        hdr = np.zeros(IN_N // 2, np.int32); hdr[0] = NCH
        hdr[1] = np.array([2.0 ** lss[hd]], np.float32).view(np.int32)[0]
        xin[col, 0, i] = hdr.view(bf16)
        # the state as ggml keeps it: the core's value columns, each a row of
        # the 128 key rows, 16 an object
        sT = np.ascontiguousarray(S0[hd][:, h * DV:(h + 1) * DV].T.astype(np.float32))
        for p in range(4):
            xin[col, 1 + p, i, :4096] = sT[p * 16:(p + 1) * 16].reshape(-1).view(bf16)
        for c in range(NCH):
            t = slice(c * C, (c + 1) * C)
            xin[col, 5 + c, i] = np.concatenate([blk(kb[t, hd]), blk(qb[t, hd]),
                                                 blk(vb[t, hd, h * DV:(h + 1) * DV]), fx[c]])

x_t = iron.tensor((xin.size,), dtype=bf16, device="npu")
o_t = iron.zeros((COLS * NO * 4 * O_N,), dtype=np.float32, device="npu")
np.copyto(x_t.numpy(), xin.reshape(-1))
x_t._sync_to_device()
run = lambda: gm.gdn(x_t, o_t, NCH=NCH)
run()
o_t._sync_from_device()
ob = o_t.numpy().reshape(COLS, NO, 4, O_N)
got = np.zeros((T, H, D)); S_got = np.zeros((H, D, D))
for col in range(COLS):
    for i in range(4):
        hd, h = gm.core_of(col, i)
        for c in range(NCH):
            got[c * C:(c + 1) * C, hd, h * DV:(h + 1) * DV] = ob[col, c, i].reshape(C, DV)
        # out as ggml keeps it, sigma applied: 8 value columns an object
        st = ob[col, NCH:NCH + 8, i].reshape(DV, D).astype(np.float64)
        S_got[hd][:, h * DV:(h + 1) * DV] = st.T

ref = np.zeros((T, H, D)); S = S0.copy()
for hd in range(H):
    for t in range(T):
        S[hd] *= np.exp(g[t, hd]); delta = beta[t, hd] * (v[t, hd] - S[hd].T @ k[t, hd])
        S[hd] += np.outer(k[t, hd], delta)
        ref[t, hd] = (S[hd].T @ q[t, hd]) / np.sqrt(D)
e = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
# the floor: the same recurrence on the bf16-rounded inputs the array gets
rr = np.zeros((T, H, D)); Sb = S0.copy()
qf, kf, vf = (x.astype(np.float64) for x in (qb, kb, vb))
for hd in range(H):
    for t in range(T):
        Sb[hd] *= np.exp(g[t, hd]); delta = beta[t, hd] * (vf[t, hd] - Sb[hd].T @ kf[t, hd])
        Sb[hd] += np.outer(kf[t, hd], delta)
        rr[t, hd] = (Sb[hd].T @ qf[t, hd]) / np.sqrt(D)
print(f"  floor (bf16 inputs, exact arithmetic): out rel {e(rr, ref):.3e}; array against it {e(got, rr):.3e}")
print(f"{NCH} chunks ({T} tokens) x {H} heads: out rel {e(got, ref):.3e}  state rel {e(S_got, S):.3e}")
print("PASS" if e(got, ref) < 1e-2 and e(S_got, S) < 1e-2 else "FAIL")
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    run()
    ts.append(time.perf_counter() - t0)
print(f"  min {min(ts) * 1e3:.3f} ms a call")
