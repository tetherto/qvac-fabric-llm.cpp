# One head of the prefill gated delta rule on the mmul against ggml's token
# recurrence: two cores, each holding the state for half of the value columns
# (kernels/gdn-mm.cc).
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/gdn_pair_check.py [chunks] [gate scale]
import hashlib
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.helpers.taplib import TensorAccessPattern as Tap

here = Path(__file__).resolve().parent.parent / "kernels"
bf16 = ml_dtypes.bfloat16
NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
C, DK, DV, D = 16, 128, 64, 128
T = NCH * C
IN_N = 2 * C * DK + C * DV + (4 * C + 2 * C * C + 32) * 2   # bf16 an input object (99 x 64)
O_N = C * DV                                   # f32 an output object
NIN = 1 + 4 + NCH
NOUT = NCH + 8

src = (here / "gdn-mm.cc").read_text()
flags = [f"-DGDN_C={C}"]
obj = "tgdn_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
in_ty = np.ndarray[(IN_N,), np.dtype[bf16]]
o_ty = np.ndarray[(O_N,), np.dtype[np.float32]]
st_ty = np.ndarray[(2 * DK * DV,), np.dtype[bf16]]
n_ty = np.ndarray[(8,), np.dtype[np.int32]]


@iron.jit
def pair(x: In, o: Out):
    mk = lambda name, types: ExternalFunction(name, object_file_name=obj, source_string=src,
                                              arg_types=types, include_dirs=_include_dirs(),
                                              compile_flags=flags)
    k_hdr = mk("gdn_hdr", [in_ty, n_ty])
    k_sin = mk("gdn_state_in", [in_ty, st_ty, np.int32])
    k_sout = mk("gdn_state_out", [st_ty, o_ty, np.int32])
    k_chunk = mk("gdn_chunk", [in_ty, st_ty, o_ty])
    fi = [ObjectFifo(in_ty, name=f"i{h}", depth=1) for h in range(2)]
    fo = [ObjectFifo(o_ty, name=f"o{h}", depth=1) for h in range(2)]

    def core(x_in, o_out, hdr, sin, sout, chunk, st, cnt):
        hh = x_in.acquire(1)
        hdr(hh, cnt)
        x_in.release(1)
        for part in range(4):
            xx = x_in.acquire(1)
            sin(xx, st, part)
            x_in.release(1)
        for _ in range_(cnt[0]):
            xx = x_in.acquire(1)
            oo = o_out.acquire(1)
            chunk(xx, st, oo)
            o_out.release(1)
            x_in.release(1)
        for part in range(8):
            oo = o_out.acquire(1)
            sout(st, oo, part)
            o_out.release(1)

    ws = [Worker(core, fn_args=[fi[h].cons(), fo[h].prod(), k_hdr, k_sin, k_sout, k_chunk,
                                Buffer(st_ty, name=f"st{h}"), Buffer(n_ty, name=f"n{h}")],
                 tile=Tile(0, 2 + h), stack_size=0xA00) for h in range(2)]

    def seq(x_h, o_h, a0, a1, b0, b1):
        for h, e in enumerate((a0, a1)):
            e.fill(x_h, tap=Tap([1, 2 * NIN * IN_N], h * NIN * IN_N, [NIN * IN_N // 64, 64], [64, 1]))
        for h, e in enumerate((b0, b1)):
            e.drain(o_h, tap=Tap([1, 2 * NOUT * O_N], h * NOUT * O_N, [NOUT * O_N // 64, 64], [64, 1]),
                    wait=True)

    rt = Runtime(seq, [np.ndarray[(2 * NIN * IN_N,), np.dtype[bf16]],
                       np.ndarray[(2 * NOUT * O_N,), np.dtype[np.float32]],
                       fi[0].prod(tile=Tile(0, 0)), fi[1].prod(tile=Tile(0, 0)),
                       fo[0].cons(tile=Tile(0, 0)), fo[1].cons(tile=Tile(0, 0))])
    return Program(iron.get_current_device(), rt, workers=ws).resolve_program()


def blk(x):
    r, c = x.shape
    return x.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


def unblk(x, r, c):
    return x.reshape(r // 8, c // 8, 8, 8).transpose(0, 2, 1, 3).reshape(r, c)


rng = np.random.default_rng(0)
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
q = nrm(rng.standard_normal((T, D)))
k = nrm(rng.standard_normal((T, D)))
v = rng.standard_normal((T, D))
beta = 1 / (1 + np.exp(-rng.standard_normal(T)))
g = -np.log1p(np.exp(rng.standard_normal(T))) * GS
S0 = rng.standard_normal((D, D)) * 0.1
qb, kb, vb = (x.astype(np.float32).astype(bf16) for x in (q, k, v))

# the scalars, as the host works them out: exponents of 2 for every factor,
# and when the state is renormalized
L2S = np.log2(1 / np.sqrt(D))
g2 = g * np.log2(np.e)
fxs, ls = [], 0.0
for c in range(NCH):
    gam = np.cumsum(g2[c * C:(c + 1) * C])
    ls_new = ls + gam[-1]
    shift = int(-ls_new) if ls_new < -32 else 0
    lsn = ls_new + shift                      # sigma' after the shift, before the update
    bt = beta[c * C:(c + 1) * C]
    Dm = 2.0 ** (gam[:, None] - gam[None, :])
    lo_ = np.arange(C)[None, :] < np.arange(C)[:, None]
    le_ = np.arange(C)[None, :] <= np.arange(C)[:, None]
    sc = 1 / np.sqrt(D)
    f = np.concatenate([2.0 ** (gam + ls), 2.0 ** (gam + ls) * sc, 2.0 ** (gam[-1] - gam - lsn), bt,
                        blk(np.where(lo_, bt[:, None] * Dm, 0.0)), blk(np.where(le_, Dm * sc, 0.0))]).astype(np.float32)
    tail = np.zeros(32, np.int32)
    tail[0] = shift
    fxs.append(np.concatenate([f.view(np.int32), tail]).view(bf16))
    ls = ls_new + shift
xin = np.zeros((2, NIN, IN_N), bf16)
for h in range(2):
    hdr = np.zeros(IN_N // 2, np.int32)
    hdr[0] = NCH
    xin[h, 0] = hdr.view(bf16)
    s = S0[:, h * DV:(h + 1) * DV].astype(np.float32)
    shi = s.astype(bf16)
    slo = (s - shi.astype(np.float32)).astype(bf16)
    parts = np.concatenate([blk(shi), blk(slo)])
    for p in range(4):
        xin[h, 1 + p, :4096] = parts[p * 4096:(p + 1) * 4096]
    for c in range(NCH):
        t = slice(c * C, (c + 1) * C)
        xin[h, 5 + c] = np.concatenate([blk(kb[t]), blk(qb[t]), blk(vb[t, h * DV:(h + 1) * DV]),
                                        fxs[c]])

x_t = iron.tensor((xin.size,), dtype=bf16, device="npu")
o_t = iron.zeros((2 * NOUT * O_N,), dtype=np.float32, device="npu")
np.copyto(x_t.numpy(), xin.reshape(-1))
x_t._sync_to_device()
pair(x_t, o_t)
o_t._sync_from_device()
ob = o_t.numpy().reshape(2, NOUT, O_N)
got = np.zeros((T, D))
S_got = np.zeros((D, D))
for h in range(2):
    for c in range(NCH):
        got[c * C:(c + 1) * C, h * DV:(h + 1) * DV] = unblk(ob[h, c], C, DV)
    st = ob[h, NCH:NCH + 8].reshape(-1).view(bf16).astype(np.float64)
    S_got[:, h * DV:(h + 1) * DV] = 2.0 ** ls * (unblk(st[:DK * DV], DK, DV) + unblk(st[DK * DV:], DK, DV))

S = S0.copy()
ref = np.zeros((T, D))
for t in range(T):
    S *= np.exp(g[t])
    delta = beta[t] * (v[t] - S.T @ k[t])
    S += np.outer(k[t], delta)
    ref[t] = (S.T @ q[t]) / np.sqrt(D)
e = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
print(f"{NCH} chunks ({T} tokens), gate x{GS}: out rel {e(got, ref):.3e}  state rel {e(S_got, S):.3e}")
print("PASS" if e(got, ref) < 1e-2 and e(S_got, S) < 1e-2 else "FAIL")
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    pair(x_t, o_t)
    ts.append(time.perf_counter() - t0)
print(f"  min {min(ts) * 1e6:.0f} us a call")
