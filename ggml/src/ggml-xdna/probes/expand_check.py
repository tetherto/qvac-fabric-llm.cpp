# One-core check of the prefill GEMM's expander (kernels/gemm-expand.cc)
# against numpy: a random decode GEMV tile, built by the decode's own packer
# (gemv_q4.pack_weight_tile), expanded one K step at a time into the mmul's
# (kb, nb, si, ti) bf16 order.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/expand_check.py q4g32|q8g16 [t]
import hashlib
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gemv_q4  # noqa: E402

FMT = sys.argv[1] if len(sys.argv) > 1 else "q4g32"
Q8 = FMT != "q4g32"
N_CORE = 64
K_TILE = 128 if Q8 else 256
STEPS = K_TILE // 128
TB = gemv_q4.tile_bytes(FMT, K_TILE, N_CORE)
NT = 3   # tiles through the core

T = int(sys.argv[2]) if len(sys.argv) > 2 else 4          # the mmul's t
src = (here / "gemm-expand.cc").read_text()
flags = [f"-DMAC_T={T}"]
obj = "texp_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
t_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
o_ty = np.ndarray[(128 * 64,), np.dtype[ml_dtypes.bfloat16]]


@iron.jit
def expand(tiles: In, out: Out):
    k = ExternalFunction("ggml_xdna_expand", object_file_name=obj, source_string=src,
                         arg_types=[t_ty, o_ty, np.int32, np.int32],
                         include_dirs=_include_dirs(), compile_flags=flags)
    ft = ObjectFifo(t_ty, name="ft", depth=2)
    fo = ObjectFifo(o_ty, name="fo", depth=2)

    def body(t_in, o_out, kexp):
        for _ in range(NT):
            t = t_in.acquire(1)
            for s in range(STEPS):
                o = o_out.acquire(1)
                kexp(t, o, s, 1 if Q8 else 0)
                o_out.release(1)
            t_in.release(1)

    w = Worker(body, fn_args=[ft.cons(), fo.prod(), k], stack_size=0x800)

    def seq(t_h, o_h, ti, oo):
        ti.fill(t_h)
        oo.drain(o_h, wait=True)

    rt = Runtime(seq, [np.ndarray[(NT * TB,), np.dtype[np.uint8]],
                       np.ndarray[(NT * STEPS * 128 * 64,), np.dtype[ml_dtypes.bfloat16]],
                       ft.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


rng = np.random.default_rng(0)
ng = K_TILE // 32
tiles, refs = [], []
for _ in range(NT):
    lo, hi = (0, 16) if not Q8 else (-128, 128)
    codes = rng.integers(lo, hi, size=(K_TILE, N_CORE)).astype(np.int32)
    d8 = rng.integers(1, 64, size=(ng, N_CORE)).astype(np.int32)
    m8 = rng.integers(0, 64, size=(ng, N_CORE)).astype(np.int32)
    dS = (rng.standard_normal((1, N_CORE)) * 0.002).astype(np.float32)
    mS = (rng.standard_normal((1, N_CORE)) * 0.005).astype(np.float32)
    tiles.append(gemv_q4.pack_weight_tile(FMT, codes, d8, m8, dS, mS))

    def pair(x):
        p = gemv_q4.split_pairs(x)
        return p[0::2].astype(np.float64) + p[1::2].astype(np.float64)
    dSe, mSe = pair(dS)[0], pair(mS)[0]
    g = np.arange(K_TILE) // 32
    w = codes * (dSe[None, :] * d8[g]) + mSe[None, :] * m8[g]     # (K_TILE, 64)
    for s in range(STEPS):
        ws = w[s * 128:(s + 1) * 128]                               # (128, 64)
        # (kb, si, nb, ti) -> (kb, nb, si, ti)
        refs.append(ws.reshape(16, 8, 64 // T, T).transpose(0, 2, 1, 3).reshape(-1))

t_buf = iron.tensor((NT * TB,), dtype=np.uint8, device="npu")
o_buf = iron.zeros((NT * STEPS * 128 * 64,), dtype=ml_dtypes.bfloat16, device="npu")
np.copyto(t_buf.numpy(), np.concatenate(tiles))
t_buf._sync_to_device()
expand(t_buf, o_buf)
o_buf._sync_from_device()
got = o_buf.numpy().astype(np.float64)
ref = np.concatenate(refs)
ref_bf = ref.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
# within one bf16 ulp of the value, or - near zero, where the biased form's
# cancellation shows - within 3e-4 of the tile's rms (an order under what
# rounding a typical weight to bf16 costs)
rms = np.sqrt((ref ** 2).mean())
err = np.abs(got - ref_bf)
ulp = err / np.maximum(np.maximum(np.abs(ref_bf) * 2.0 ** -7, 3e-4 * rms), 1e-30)
bad = int((ulp > 1.01).sum())
print(f"  rel rms err {np.sqrt(((got - ref) ** 2).mean()) / rms:.3e}")
print(f"{FMT}: {got.size} values, max |err| {np.abs(got - ref).max():.3e}, "
      f"rms(ref) {np.sqrt((ref ** 2).mean()):.3e}, beyond 1 bf16 ulp: {bad}")
idx = np.nonzero(ulp > 1.01)[0]
for i in idx[:12]:
    st, rem = divmod(i, 128 * 64)
    kb, rem = divmod(rem, 512); nb, rem = divmod(rem, 8 * T); si, ti = divmod(rem, T)
    print(f"  tile/step {st} row {kb*8+si} col {nb*T+ti}: got {got[i]:.5g} ref {ref[i]:.5g} ref_bf {ref_bf[i]:.5g}")
print("PASS" if bad == 0 else "FAIL")
