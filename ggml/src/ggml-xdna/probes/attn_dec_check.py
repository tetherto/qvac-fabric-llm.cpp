# Standalone check of kernels/attn-dec.cc on one core against numpy.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/attn_dec_check.py [chunks] [q scale]
# A fresh NPU_CACHE_HOME per kernel edit: iron.jit reuses a cached artifact.
import sys
from pathlib import Path
import numpy as np
import ml_dtypes
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

# The kernel includes the shared headers (xdna-math.h, xdna-vec.h), which do not
# compile beside the source string: kernelsrc pastes them in, as the designs do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
import kernelsrc  # noqa: E402

ACT_TILE = 2112
TB = 10752
D, H, G, KVH, P = 256, 8, 4, 2, 5
NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
QS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
# Pass/fail: each head's output against the f32 softmax over the bf16-rounded
# K and V (a bf16 kernel lands near 2e-2), and its log-sum-exp.
REL_TOL = 5e-2
LSE_TOL = 1e-1
PIECES = 33

src = kernelsrc.load(Path(__file__).resolve().parent.parent / "kernels" / "attn-dec.cc")
flags = [f"-DACT_TILE={ACT_TILE}", "-DN_CORE=64"]
a_ty = np.ndarray[(ACT_TILE // 4,), np.dtype[np.int32]]
w_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
e_ty = np.ndarray[(64,), np.dtype[np.float32]]
s_ty = np.ndarray[(4096,), np.dtype[ml_dtypes.bfloat16]]


@iron.jit
def att(qin: In, win: In, out: Out, *, NW: CompileTime[int]):
    mk = lambda name, tys: ExternalFunction(
        name, object_file_name=f"tatt_{__import__('hashlib').md5((src + str(flags)).encode()).hexdigest()[:8]}.o", source_string=src, arg_types=tys,
        include_dirs=_include_dirs(), compile_flags=flags)
    fq, fc, fe = mk("ggml_xdna_att_q", [a_ty]), mk("ggml_xdna_att_chunk", [w_ty, s_ty]), mk("ggml_xdna_att_emit", [e_ty])
    qf = ObjectFifo(a_ty, name="qf", depth=2)
    wf = ObjectFifo(w_ty, name="wf", depth=2)
    of = ObjectFifo(e_ty, name="of", depth=2)

    def core(q_in, w_in, o_out, fq, fc, fe, scr):
        for _ in range_(2):
            a = q_in.acquire(1)
            fq(a)
            q_in.release(1)
        for _ in range_(NW):
            w = w_in.acquire(1)
            fc(w, scr)
            w_in.release(1)
        for _ in range_(PIECES):
            o = o_out.acquire(1)
            fe(o)
            o_out.release(1)

    wk = Worker(core, fn_args=[qf.cons(), wf.cons(), of.prod(), fq, fc, fe,
                               Buffer(s_ty, name="scr", mem_bank=3)], stack_size=4928)

    def seq(q_h, w_h, o_h, qi, wi, oo):
        qi.fill(q_h)
        wi.fill(w_h)
        oo.drain(o_h, wait=True)

    rt = Runtime(seq, [np.ndarray[(2 * ACT_TILE // 4,), np.dtype[np.int32]],
                       np.ndarray[(NW * TB,), np.dtype[np.uint8]],
                       np.ndarray[(PIECES * 64,), np.dtype[np.float32]],
                       qf.prod(), wf.prod(), of.cons()])
    return Program(iron.get_current_device(), rt, workers=[wk]).resolve_program()


rng = np.random.default_rng(1)
npos = 5 * 16 * NCH
n_valid = 5 * 16 * (NCH - 1) + 3
q = (rng.standard_normal((H, D)) * QS / 16).astype(np.float32)
K = rng.standard_normal((npos, KVH, D)).astype(np.float16)
V = rng.standard_normal((npos, KVH, D)).astype(np.float16)

qa = np.zeros((2, ACT_TILE), np.uint8)
for t in range(2):
    qb = np.zeros(1024, ml_dtypes.bfloat16)
    for hh in range(4):
        for d in range(D):
            qb[(d // 8) * 32 + (d % 8) * 4 + hh] = q[4 * t + hh, d]
    qa[t, :2048] = qb.view(np.uint8)
    qa[t, 2048:2060] = np.array([n_valid, t, 0], np.int32).view(np.uint8)

w = np.zeros((1 + NCH, TB), np.uint8)
w[0, :4] = np.array([0], np.int32).view(np.uint8)
covered = []
for j in range(NCH):
    k = 16 * j
    ch = np.zeros(2 * P * KVH * D, np.float16)
    kk = K[5 * k:5 * k + 5].reshape(-1)
    vv = V[5 * k:5 * k + 5].reshape(-1)
    ch[:kk.size] = kk
    ch[P * KVH * D:P * KVH * D + vv.size] = vv
    w[1 + j, :ch.nbytes] = ch.view(np.uint8)
    covered += [p for p in range(5 * k, 5 * k + 5) if p < n_valid]

qt = iron.tensor(qa.view(np.int32).reshape(-1), dtype=np.int32, device="npu")
wt = iron.tensor(w.reshape(-1), dtype=np.uint8, device="npu")
ot = iron.zeros(PIECES * 64, dtype=np.float32, device="npu")
att(qt, wt, ot, NW=1 + NCH)
st = ot.numpy()

qbf = q.astype(ml_dtypes.bfloat16).astype(np.float32)
worst = 0
failed = []
for h in range(H):
    g = h // 4
    Kc = K[covered, g].astype(ml_dtypes.bfloat16).astype(np.float32)
    Vc = V[covered, g].astype(ml_dtypes.bfloat16).astype(np.float32)
    s = Kc @ qbf[h]
    m = s.max()
    e = np.exp(s - m)
    ref = (e @ Vc) / e.sum()
    mi = g * 16 + h % 4
    mm, ll = st[mi], st[32 + mi]
    base = 64 + g * 1024 + (h % 4) * 8
    o = np.array([st[base + (d // 8) * 32 + d % 8] for d in range(D)])
    got = o / ll if ll > 0 else o
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    lse_ref = m + np.log(e.sum())
    lse = mm + np.log(ll) if ll > 0 else float("nan")
    worst = max(worst, rel)
    print(f"h{h} m={mm:.3f}/{m:.3f} lse={lse:.4f}/{lse_ref:.4f} rel={rel:.3e}")
    if not (np.all(np.isfinite(got)) and rel < REL_TOL and abs(lse - lse_ref) < LSE_TOL):
        failed.append(h)
print("worst", worst)
if failed:
    sys.exit(f"attn_dec_check: FAIL on heads {failed} (rel tolerance {REL_TOL}, lse tolerance {LSE_TOL})")
print("attn_dec_check: PASS")
