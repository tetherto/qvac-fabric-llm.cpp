# The prefill attention on the whole array (kernels/attn_mm.py) against numpy:
# Qwen3.5's full-attention shape (8 query heads, 2 KV heads, D = 256), causal,
# NPASS passes of 64 positions after P0 positions of context.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/fa_design_check.py [NPASS] [P0]
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import attn_mm as am  # noqa: E402

NPASS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
P0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
bf16 = ml_dtypes.bfloat16
COLS, R, D, DH, NK, PASS = am.COLS, am.R, am.D, am.DH, am.NK, am.PASS
NTOK = NPASS * PASS
T_MAX = am.tiles_of(P0, NPASS - 1)
NKEY = T_MAX * NK


def blk(x):
    """(rows x cols) -> the mmul's (rows/8, cols/8, 8, 8) sub-tiles."""
    r, c = x.shape
    return x.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


rng = np.random.default_rng(4)
Q = rng.standard_normal((NTOK, 8, D)).astype(np.float32)            # token, head, d
K = rng.standard_normal((NKEY, 2, D)).astype(np.float32).astype(bf16)
V = rng.standard_normal((NKEY, 2, D)).astype(np.float32).astype(bf16)
Qs = (Q * np.float32(np.log2(np.e) / 16)).astype(bf16)

# Q: per column the header object, then per pass the 4 cores' Q half^T
qbuf = np.zeros((COLS, 1 + NPASS, 4, R * DH), bf16)
hdr = np.zeros(R * DH // 2, np.int32)
hdr[:2] = [NPASS, P0]
qbuf[:, 0, :, :] = hdr.view(bf16)
for col in range(COLS):
    for i in range(4):
        g, pidx, h = am.core_of(col, i)
        for p in range(NPASS):
            rows = np.zeros((R, DH), np.float32)
            for hh in range(4):
                for j in range(8):
                    t = p * PASS + pidx * 8 + j
                    rows[hh * 8 + j] = Qs[t, 4 * g + hh, h * DH:(h + 1) * DH]
            qbuf[col, 1 + p, i] = blk(rows.T.astype(bf16))
# KV: [g][h][tile] of [K half | V half^T]
kvbuf = np.zeros((2, 2, T_MAX, 2 * NK * DH), bf16)
for g in range(2):
    for h in range(2):
        for t in range(T_MAX):
            kk = K[t * NK:(t + 1) * NK, g, h * DH:(h + 1) * DH]
            vv = V[t * NK:(t + 1) * NK, g, h * DH:(h + 1) * DH]
            kvbuf[g, h, t] = np.concatenate([blk(kk), blk(vv.T)])
# the streams in endpoint order: (g, h) = (0,0), (0,1), (1,0), (1,1)
q_t = iron.tensor((qbuf.size,), dtype=bf16, device="npu")
kv_t = iron.tensor((kvbuf.size,), dtype=bf16, device="npu")
o_t = iron.zeros((COLS * NPASS * 4 * R * DH,), dtype=np.float32, device="npu")
np.copyto(q_t.numpy(), qbuf.reshape(-1))
np.copyto(kv_t.numpy(), kvbuf.reshape(-1))
q_t._sync_to_device()
kv_t._sync_to_device()
run = lambda: am.attn(q_t, kv_t, o_t, NPASS=NPASS, P0=P0)
run()
o_t._sync_from_device()
ob = o_t.numpy().reshape(COLS, NPASS, 4, R, DH)             # a core's rows, as the MemTile leaves them
got = np.zeros((NTOK, 8, D))
for col in range(COLS):
    for i in range(4):
        g, pidx, h = am.core_of(col, i)
        for p in range(NPASS):
            oT = ob[col, p, i].T                                        # (d, row)
            for hh in range(4):
                for j in range(8):
                    t = p * PASS + pidx * 8 + j
                    got[t, 4 * g + hh, h * DH:(h + 1) * DH] = oT[:, hh * 8 + j]

ref = np.zeros((NTOK, 8, D))
Kd, Vd = K.astype(np.float64), V.astype(np.float64)
for qh in range(8):
    g = qh // 4
    S = Q[:, qh, :].astype(np.float64) @ Kd[:, g, :].T / 16.0          # (tok, key)
    pos = P0 + np.arange(NTOK)
    S[np.arange(NKEY)[None, :] > pos[:, None]] = -np.inf
    P = np.exp(S - S.max(1, keepdims=True))
    ref[:, qh, :] = (P / P.sum(1, keepdims=True)) @ Vd[:, g, :]
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"passes {NPASS} P0 {P0} ({NTOK} positions, {NKEY} keys): rel rms err {rel:.3e}")
print("PASS" if rel < 3e-2 else "FAIL")
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    run()
    ts.append(time.perf_counter() - t0)
macs = 8 * sum((P0 + t + 1) for t in range(NTOK)) * D * 2
print(f"  min {min(ts) * 1e3:.3f} ms a call, {macs / min(ts) / 1e12:.2f} TMAC/s useful")
