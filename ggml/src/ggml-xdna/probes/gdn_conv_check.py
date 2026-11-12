# The GDN conv input on the array (kernels/gdn_conv.py) against numpy: conv
# along the tokens, SiLU, L2 norm of K and Q, as ggml computes them. On the
# bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/gdn_conv_check.py [NCH [NTOK]]
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gdn_conv as gc  # noqa: E402

NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else NCH * gc.C
bf16 = ml_dtypes.bfloat16
C, KW, D, DV, ROWS, IN_N, OBJ = gc.C, gc.KW, gc.D, gc.DV, gc.ROWS, gc.IN_N, gc.OBJ
H, CH = 16, 6144
EPS = 1e-6

rng = np.random.default_rng(3)
X = rng.standard_normal((NCH * C + KW - 1, CH)).astype(np.float32)      # row 0 = token -3
W = (rng.standard_normal((CH, KW)) * 0.5).astype(np.float32)


def head_ch(h):
    return np.concatenate([h * D + np.arange(D), 2048 + h * D + np.arange(D), 4096 + h * D + np.arange(D)])


xbuf = np.zeros((H, NCH, IN_N), np.float32)
hbuf = np.zeros((H, 2, IN_N), np.float32)
for h in range(H):
    ch = head_ch(h)
    for k in range(NCH):
        xbuf[h, k] = X[k * C:k * C + ROWS][:, ch].reshape(-1)
    for p in range(2):
        w = hbuf[h, p].view(np.int32)
        w[0] = NCH // 2
        w[1] = p
        w[2] = NTOK
        w[3] = np.array([EPS * EPS], np.float32).view(np.int32)[0]
        hbuf[h, p, 64:64 + KW * 3 * D] = W[ch].T.reshape(-1)

x_t = iron.tensor((xbuf.size,), dtype=np.float32, device="npu")
h_t = iron.tensor((hbuf.size,), dtype=np.float32, device="npu")
o_t = iron.zeros((gc.COLS * NCH * 2 * gc.OUT_N,), dtype=bf16, device="npu")
np.copyto(x_t.numpy(), xbuf.reshape(-1))
np.copyto(h_t.numpy(), hbuf.reshape(-1))
x_t._sync_to_device()
h_t._sync_to_device()
run = lambda: gc.gdn_conv(x_t, h_t, o_t, NCH_=NCH)
run()
o_t._sync_from_device()

# reference (f64 conv from f32, as ggml: taps in order; silu; l2 per token)
conv = np.zeros((NCH * C, CH))
for i in range(KW):
    conv += X[i:i + NCH * C].astype(np.float64) * W[:, i].astype(np.float64)[None, :]
y = conv / (1.0 + np.exp(-conv))
y[NTOK:] = 0.0


def untile(buf, rows, cols):
    return buf.reshape(rows // 8, cols // 8, 8, 8).transpose(0, 2, 1, 3).reshape(rows, cols)


got = o_t.numpy().astype(np.float64).reshape(gc.COLS, NCH, 4, OBJ)
err, ref2 = 0.0, 0.0
worst = 0.0
for col in range(gc.COLS):
    for k in range(NCH):
        for i in range(4):
            h, half = 2 * col + i // 2, i % 2
            ch = head_ch(h)
            yy = y[k * C:(k + 1) * C][:, ch]
            q, kk, v = yy[:, :D], yy[:, D:2 * D], yy[:, 2 * D:]
            nq = np.maximum(np.sqrt((q ** 2).sum(1, keepdims=True)), EPS)
            nk = np.maximum(np.sqrt((kk ** 2).sum(1, keepdims=True)), EPS)
            o = got[col, k, i]
            gk = untile(o[:C * D], C, D)
            gq = untile(o[C * D:2 * C * D], C, D)
            gv = untile(o[2 * C * D:], C, DV)
            for a, b in ((gk, kk / nk), (gq, q / nq), (gv, v[:, half * DV:(half + 1) * DV])):
                err += ((a - b) ** 2).sum()
                ref2 += (b ** 2).sum()
                worst = max(worst, np.abs(a - b).max())
rel = np.sqrt(err / ref2)
print(f"NCH {NCH} NTOK {NTOK}: rel rms err {rel:.3e}, max abs {worst:.3e}")
print("PASS" if rel < 8e-3 else "FAIL")
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    run()
    ts.append(time.perf_counter() - t0)
print(f"  min {min(ts) * 1e3:.3f} ms a call ({NCH} chunks of 16 heads)")
