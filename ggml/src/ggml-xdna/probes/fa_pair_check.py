# One pair of the prefill attention on the mmul against numpy: 32 rows (4 query
# heads x 8 positions q0 .. q0 + 7 of one KV head), causal over keys 0 .. q0 + 7,
# each core holding one half of D = 256 and the two exchanging their halves of
# the scores through shared memory (kernels/attn-mm.cc), everything transposed.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/fa_pair_check.py [q0]
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
Q0 = int(sys.argv[1]) if len(sys.argv) > 1 else 40
R, D, DH, NK = 32, 256, 128, 16
NT = (Q0 + 8 + NK - 1) // NK            # key tiles up to the last position
NKEY = NT * NK

src = (here / "attn-mm.cc").read_text()
flags = [f"-DFA_R={R}", f"-DFA_DH={DH}", f"-DFA_NK={NK}"]
obj = "tfa_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
q_ty = np.ndarray[(R * DH,), np.dtype[bf16]]
kv_ty = np.ndarray[(2 * NK * DH,), np.dtype[bf16]]
s_ty = np.ndarray[(R * NK,), np.dtype[np.float32]]
p_ty = np.ndarray[(R * NK,), np.dtype[bf16]]
o_ty = np.ndarray[(R * DH,), np.dtype[np.float32]]
ml_ty = np.ndarray[(2 * R,), np.dtype[np.float32]]


@iron.jit
def pair(q: In, kv: In, o: Out):
    mk = lambda name, types: ExternalFunction(name, object_file_name=obj, source_string=src,
                                              arg_types=types, include_dirs=_include_dirs(),
                                              compile_flags=flags)
    k_begin = mk("fa_begin", [o_ty, ml_ty])
    k_scores = mk("fa_scores", [q_ty, kv_ty, s_ty])
    k_update = mk("fa_update", [s_ty, s_ty, o_ty, p_ty, ml_ty, np.int32])
    k_pv = mk("fa_pv", [p_ty, kv_ty, o_ty])
    k_end = mk("fa_end", [o_ty, ml_ty])
    fq = [ObjectFifo(q_ty, name=f"q{h}", depth=1) for h in range(2)]
    fkv = [ObjectFifo(kv_ty, name=f"kv{h}", depth=2) for h in range(2)]
    fo = [ObjectFifo(o_ty, name=f"o{h}", depth=1) for h in range(2)]
    fx = [ObjectFifo(s_ty, name=f"x{h}", depth=1) for h in range(2)]

    def core(q_in, kv_in, o_out, x_own, x_nb, begin, scores, update, pv, end, pb, ml):
        qq = q_in.acquire(1)
        oo = o_out.acquire(1)
        begin(oo, ml)
        for _ in range_(NT):
            kk = kv_in.acquire(1)
            so = x_own.acquire(1)
            scores(qq, kk, so)
            x_own.release(1)
            sn = x_nb.acquire(1)
            update(so, sn, oo, pb, ml, Q0)
            x_nb.release(1)
            pv(pb, kk, oo)
            kv_in.release(1)
        end(oo, ml)
        o_out.release(1)
        q_in.release(1)

    ws = [Worker(core, fn_args=[fq[h].cons(), fkv[h].cons(), fo[h].prod(), fx[h].prod(),
                                fx[1 - h].cons(), k_begin, k_scores, k_update, k_pv, k_end,
                                Buffer(p_ty, name=f"p{h}"), Buffer(ml_ty, name=f"ml{h}")],
                 tile=Tile(0, 2 + h), stack_size=0xC00) for h in range(2)]

    def seq(q_h, kv_h, o_h, *e):
        qi, ki, oo = e[0:2], e[2:4], e[4:6]
        for h in range(2):
            qi[h].fill(q_h, tap=Tap([1, 2 * R * DH], h * R * DH, [1, R * DH], [0, 1]))
            ki[h].fill(kv_h, tap=Tap([1, 2 * NT * 2 * NK * DH], h * NT * 2 * NK * DH, [1, NT * 2 * NK * DH], [0, 1]))
        for h in range(2):
            oo[h].drain(o_h, tap=Tap([1, 2 * R * DH], h * R * DH, [1, R * DH], [0, 1]), wait=True)

    rt = Runtime(seq, [np.ndarray[(2 * R * DH,), np.dtype[bf16]],
                       np.ndarray[(2 * NT * 2 * NK * DH,), np.dtype[bf16]],
                       np.ndarray[(2 * R * DH,), np.dtype[np.float32]]]
                 + [f.prod(tile=Tile(1, 0)) for f in fq]
                 + [f.prod(tile=Tile(0, 0)) for f in fkv]
                 + [f.cons(tile=Tile(0, 0)) for f in fo])
    return Program(iron.get_current_device(), rt, workers=ws).resolve_program()


def blk(x):
    """(rows x cols) -> the mmul's (rows/8, cols/8, 8, 8) sub-tiles."""
    r, c = x.shape
    return x.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


rng = np.random.default_rng(3)
Q = rng.standard_normal((4, 8, D)).astype(np.float32).astype(bf16)       # head, pos, d
K = rng.standard_normal((NKEY, D)).astype(np.float32).astype(bf16)
V = rng.standard_normal((NKEY, D)).astype(np.float32).astype(bf16)
Qr = Q.reshape(R, D)                                                     # row = head * 8 + i
# the kernel takes Q scaled by log2(e) / sqrt(D): a score is then a power of 2
Qs = (Qr.astype(np.float32) * np.float32(np.log2(np.e) / 16)).astype(bf16)
qb = np.concatenate([blk(Qs[:, h * DH:(h + 1) * DH].T) for h in range(2)])
kvb = []
for h in range(2):
    for t in range(NT):
        kvb.append(blk(K[t * NK:(t + 1) * NK, h * DH:(h + 1) * DH]))
        kvb.append(blk(V[t * NK:(t + 1) * NK, h * DH:(h + 1) * DH].T))
kvb = np.concatenate(kvb)

q_t = iron.tensor(qb.shape, dtype=bf16, device="npu")
kv_t = iron.tensor(kvb.shape, dtype=bf16, device="npu")
o_t = iron.zeros((2 * R * DH,), dtype=np.float32, device="npu")
np.copyto(q_t.numpy(), qb)
np.copyto(kv_t.numpy(), kvb)
q_t._sync_to_device()
kv_t._sync_to_device()
pair(q_t, kv_t, o_t)
o_t._sync_from_device()
got = np.concatenate([o_t.numpy()[h * R * DH:(h + 1) * R * DH]
                      .reshape(DH // 8, R // 8, 8, 8).transpose(0, 2, 1, 3).reshape(DH, R).T
                      for h in range(2)], axis=1)

S = Qr.astype(np.float64) @ K.astype(np.float64).T * 0.0625             # (R, NKEY)
pos = Q0 + np.arange(R) % 8
S[np.arange(NKEY)[None, :] > pos[:, None]] = -np.inf
P = np.exp(S - S.max(1, keepdims=True))
ref = (P / P.sum(1, keepdims=True)) @ V.astype(np.float64)
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"q0={Q0} keys={NKEY}: rel rms err {rel:.3e}, max |err| {np.abs(got - ref).max():.3e}")
print("PASS" if rel < 2e-2 else "FAIL")
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    pair(q_t, kv_t, o_t)
    ts.append(time.perf_counter() - t0)
print(f"  min {min(ts) * 1e6:.0f} us a call, {NT} tiles")
