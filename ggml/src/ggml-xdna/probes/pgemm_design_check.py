# The prefill GEMM as built for the backend (kernels/pgemm.py: header-driven
# cores) against numpy - probes/pgemm_check.py's data, the design's own
# layouts. On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/pgemm_design_check.py q4g32|q8g16 [M K N [glu]]
# glu: the weight is a gate/up pair (a core's columns 32 gate then 32 up) and
# C their SwiGLU, N / 2 columns
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gemv_q4  # noqa: E402
import pgemm as pg  # noqa: E402

FMT = sys.argv[1] if len(sys.argv) > 1 else "q4g32"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
K = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
GLU = len(sys.argv) > 5 and sys.argv[5] == "glu"
N_OUT = N // 2 if GLU else N
Q8 = FMT != "q4g32"
bf16 = ml_dtypes.bfloat16
COLS, NC, KS, MB, R, S, T = pg.COLS, pg.NC, pg.KS, pg.MB, 8, 8, 8
K_TILE = 128 if Q8 else 256
NT = K // K_TILE
NSTEP = K // KS
CHUNK = pg.CHUNK
NCH = N // CHUNK
NMB = M // (2 * MB)
TB = pg.TB
W_BYTES = COLS * NCH * NT * 2 * TB
A_ELEMS = 2 * pg.HDR + NMB * 2 * NSTEP * MB * KS
C_ELEMS = M * N_OUT

rng = np.random.default_rng(2)
ng = K_TILE // 32
Wfull = np.zeros((K, N))          # mode 0: the output's columns; glu: [gate | up]
wbuf = np.zeros(W_BYTES, np.uint8)


def pr(x):
    p = gemv_q4.split_pairs(x)
    return p[0::2].astype(np.float64) + p[1::2].astype(np.float64)


# the decode order: [column][chunk][K tile][row], core (c, r) of chunk ch
# owning columns ch * 1024 + (c * 2 + r) * 64 .. + 64
for col in range(COLS):
    for ch in range(NCH):
        for t in range(NT):
            for r in range(2):
                lo, hi = (0, 16) if not Q8 else (-128, 128)
                codes = rng.integers(lo, hi, size=(K_TILE, NC)).astype(np.int32)
                d8 = rng.integers(1, 64, size=(ng, NC)).astype(np.int32)
                m8 = rng.integers(0, 64, size=(ng, NC)).astype(np.int32)
                dS = (rng.standard_normal((1, NC)) * 0.002).astype(np.float32)
                mS = (rng.standard_normal((1, NC)) * 0.005).astype(np.float32)
                off = (((col * NCH + ch) * NT + t) * 2 + r) * TB
                wbuf[off:off + TB] = gemv_q4.pack_weight_tile(FMT, codes, d8, m8, dS, mS)
                g = np.arange(K_TILE) // 32
                w = codes * (pr(dS)[0][None, :] * d8[g]) + pr(mS)[0][None, :] * m8[g]
                # the core's part h (32 columns) is output columns
                # (2 ch + h) * 512 + (col * 2 + r) * 32; glu: gate part 0 and
                # up part 1 of output ch * 512 + (col * 2 + r) * 32
                for h in range(2):
                    if GLU:
                        n0 = h * N_OUT + ch * 512 + (col * 2 + r) * 32
                    else:
                        n0 = (2 * ch + h) * 512 + (col * 2 + r) * 32
                    Wfull[t * K_TILE:(t + 1) * K_TILE, n0:n0 + 32] = w[:, h * 32:(h + 1) * 32]
A = rng.standard_normal((M, K)).astype(np.float32).astype(bf16)
# A as [M block][mb][K step][(64/r, KS/s, r, s) blocks]
# row-major (64 x 64) steps: the MemTile tiles them
ablk = A.reshape(NMB, 2, MB, NSTEP, KS).transpose(0, 1, 3, 2, 4).copy()
for mb in range(2):
    ablk[:, mb] = ablk[:, mb][:, pg.a_steps(mb, NSTEP)]

w_t = iron.tensor((W_BYTES,), dtype=np.uint8, device="npu")
a_t = iron.tensor((A_ELEMS,), dtype=bf16, device="npu")
c_t = iron.zeros((C_ELEMS,), dtype=np.float32, device="npu")
hdr = pg.header(M, K, N, Q8, GLU)
np.copyto(w_t.numpy(), wbuf)
# the header through the MemTile's tiling: bf16 j of it sent where the
# tiling reads output j from


def untiled(h):
    o = np.zeros_like(h)
    for j in range(64):
        mr, kb, r, e = j // 512, (j // 64) % 8, (j // 8) % 8, j % 8
        o[(mr * 8 + r) * 64 + kb * 8 + e] = h[j]
    return o


hdr = untiled(hdr)
np.copyto(a_t.numpy(), np.concatenate([hdr, hdr, ablk.reshape(-1)]))
w_t._sync_to_device()
a_t._sync_to_device()
run = lambda: pg.pgemm(w_t, a_t, c_t, M=M, K=K, N=N, Q8=1 if Q8 else 0, GLU=1 if GLU else 0)
run()
c_t._sync_from_device()
# C is row-major (M x N_OUT)
got = c_t.numpy().reshape(M, N_OUT).astype(np.float64)
ref = A.astype(np.float64) @ Wfull.astype(np.float32).astype(bf16).astype(np.float64)
if GLU:
    gate, up = ref[:, :N_OUT], ref[:, N_OUT:]
    ref = gate / (1.0 + np.exp(-gate)) * up
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"{FMT} M={M} K={K} N={N}{' glu' if GLU else ''}: rel rms err {rel:.3e}")
for blk in range(NMB):
    rows = slice(blk * 2 * MB, (blk + 1) * 2 * MB)
    e = got[rows] - ref[rows]
    print(f"  blk {blk}: {np.sqrt((e ** 2).mean() / (ref[rows] ** 2).mean()):.1e}")
print("PASS" if rel < 2e-2 else "FAIL")
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    run()
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"  min {ts[0] * 1e3:.3f} ms a call, {2 * M * K * N / ts[0] / 1e12:.2f} TFLOP/s")
