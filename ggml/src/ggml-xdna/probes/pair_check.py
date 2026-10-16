# One expander/multiplier pair of the prefill GEMM (FLM_PREFILL_PLAN.md, step
# 1, build order 2) against numpy: C (M x 64) = A (M x K) @ W (K x 64), W the
# decode GEMV tiles of one core, expanded on tile (0, 2) (gemm-expand.cc) and
# handed over in shared memory to the mmul on its neighbour (0, 3).
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/pair_check.py q4g32|q8g16 [K] [bfp16]
import hashlib
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gemv_q4  # noqa: E402

FMT = sys.argv[1] if len(sys.argv) > 1 else "q4g32"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
BFP = len(sys.argv) > 3 and sys.argv[3] == "bfp16"
# "one": expand and multiply on the same core, the full design's shape
ONE = len(sys.argv) > 4 and sys.argv[4] == "one"
Q8 = FMT != "q4g32"
N_CORE, M = 64, 64
KS = int(__import__("os").environ.get("KS", "128"))   # K a step: 64 or 128
K_TILE = 128 if Q8 else 256
STEPS = K_TILE // KS
NT = K // K_TILE
TB = gemv_q4.tile_bytes(FMT, K_TILE, N_CORE)
bf16 = ml_dtypes.bfloat16

src = (here / "gemm-expand.cc").read_text()
t_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
b_ty = np.ndarray[(KS * N_CORE,), np.dtype[bf16]]
a_ty = np.ndarray[(M * KS,), np.dtype[bf16]]
c_ty = np.ndarray[(M * N_CORE,), np.dtype[np.float32]]


def make_mm():
    return akernels.mm(M, KS, N_CORE, input_dtype=bf16, output_dtype=np.float32,
                       vectorized=True, emulate_bf16_mmul_with_bfp16=BFP)


# Not mm().mac_dims: it reports (4, 8, 4) for bf16 where the kernel is built
# (4, 8, 8) - probes/mm_check.py; bfp16 emulation is (8, 8, 8) (gemm.py).
R, S, T = (8, 8, 8) if BFP else (4, 8, 8)
assert S == 8 and T in (4, 8), "the expander writes 8 x 4 or 8 x 8 sub-tiles"
flags = [f"-DMAC_T={T}", f"-DEXP_STEP={KS}"]
obj = "texp_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"


@iron.jit
def pair(tiles: In, a: In, c: Out):
    mm = make_mm()   # inside the traced function, so the design compiles it
    kexp = ExternalFunction("ggml_xdna_expand", object_file_name=obj, source_string=src,
                            arg_types=[t_ty, b_ty, np.int32, np.int32],
                            include_dirs=_include_dirs(), compile_flags=flags)
    ft = ObjectFifo(t_ty, name="ft", depth=2)
    fb = None if ONE else ObjectFifo(b_ty, name="fb", depth=2)   # (0,2) -> (0,3): shared memory
    fa = ObjectFifo(a_ty, name="fa", depth=2)
    fc = ObjectFifo(c_ty, name="fc", depth=1)

    def expander(t_in, b_out, k):
        for _ in range(NT):
            t = t_in.acquire(1)
            for s in range(STEPS):
                b = b_out.acquire(1)
                k(t, b, s, 1 if Q8 else 0)
                b_out.release(1)
            t_in.release(1)

    def multiplier(a_in, b_in, c_out, zero, mul):
        cc = c_out.acquire(1)
        zero(cc)
        for _ in range(NT * STEPS):
            aa = a_in.acquire(1)
            bb = b_in.acquire(1)
            mul(aa, bb, cc)
            a_in.release(1)
            b_in.release(1)
        c_out.release(1)

    def both(t_in, a_in, c_out, k, zero, mul, bb):
        cc = c_out.acquire(1)
        zero(cc)
        for _ in range(NT):
            t = t_in.acquire(1)
            for s in range(STEPS):
                k(t, bb, s, 1 if Q8 else 0)
                aa = a_in.acquire(1)
                mul(aa, bb, cc)
                a_in.release(1)
            t_in.release(1)
        c_out.release(1)

    if ONE:
        from aie.iron import Buffer
        wo = Worker(both, fn_args=[ft.cons(), fa.cons(), fc.prod(), kexp, mm.zero, mm,
                                   Buffer(b_ty, name="bexp")], tile=Tile(0, 2),
                    stack_size=int(__import__("os").environ.get("STK", "0x800"), 0))
    else:
        we = Worker(expander, fn_args=[ft.cons(), fb.prod(), kexp], tile=Tile(0, 2),
                    stack_size=0x800)
        wm = Worker(multiplier, fn_args=[fa.cons(), fb.cons(), fc.prod(), mm.zero, mm],
                    tile=Tile(0, 3))

    def seq(t_h, a_h, c_h, ti, ai, co):
        ti.fill(t_h)
        ai.fill(a_h)
        co.drain(c_h, wait=True)

    rt = Runtime(seq, [np.ndarray[(NT * TB,), np.dtype[np.uint8]],
                       np.ndarray[(NT * STEPS * M * KS,), np.dtype[bf16]],
                       np.ndarray[(M * N_CORE,), np.dtype[np.float32]],
                       ft.prod(), fa.prod(), fc.cons()])
    return Program(iron.get_current_device(), rt,
                   workers=[wo] if ONE else [we, wm]).resolve_program()


rng = np.random.default_rng(1)
ng = K_TILE // 32
tiles, wcols = [], []
for _ in range(NT):
    lo, hi = (0, 16) if not Q8 else (-128, 128)
    codes = rng.integers(lo, hi, size=(K_TILE, N_CORE)).astype(np.int32)
    d8 = rng.integers(1, 64, size=(ng, N_CORE)).astype(np.int32)
    m8 = rng.integers(0, 64, size=(ng, N_CORE)).astype(np.int32)
    dS = (rng.standard_normal((1, N_CORE)) * 0.002).astype(np.float32)
    mS = (rng.standard_normal((1, N_CORE)) * 0.005).astype(np.float32)
    tiles.append(gemv_q4.pack_weight_tile(FMT, codes, d8, m8, dS, mS))

    def pr(x):
        p = gemv_q4.split_pairs(x)
        return p[0::2].astype(np.float64) + p[1::2].astype(np.float64)
    g = np.arange(K_TILE) // 32
    wcols.append(codes * (pr(dS)[0][None, :] * d8[g]) + pr(mS)[0][None, :] * m8[g])
W = np.concatenate(wcols).astype(np.float32).astype(bf16).astype(np.float64)   # (K, 64)
A = rng.standard_normal((M, K)).astype(np.float32).astype(bf16)

# A in the mmul's own order, a K step at a time: (m/r, k/s, r, s) blocks
a_steps = []
for s in range(K // KS):
    blk = A[:, s * KS:(s + 1) * KS].reshape(M // R, R, KS // S, S).transpose(0, 2, 1, 3)
    a_steps.append(blk.reshape(-1))

t_buf = iron.tensor((NT * TB,), dtype=np.uint8, device="npu")
a_buf = iron.tensor((K * M,), dtype=bf16, device="npu")
c_buf = iron.zeros((M * N_CORE,), dtype=np.float32, device="npu")
np.copyto(t_buf.numpy(), np.concatenate(tiles))
np.copyto(a_buf.numpy(), np.concatenate(a_steps))
t_buf._sync_to_device()
a_buf._sync_to_device()
pair(t_buf, a_buf, c_buf)
c_buf._sync_from_device()
# C comes back in (m/r, n/t, r, t) blocks
got = c_buf.numpy().reshape(M // R, N_CORE // T, R, T).transpose(0, 2, 1, 3).reshape(M, N_CORE)
ref = A.astype(np.float64) @ W
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"{FMT} K={K} {'bfp16' if BFP else 'bf16'}: rel rms err {rel:.3e}, "
      f"max |err| {np.abs(got - ref).max():.3e}")
print("PASS" if rel < (2e-2 if BFP else 2e-3) else "FAIL")

# timing: the same call repeated; a step is 64 x 128 x 64 MACs
import time  # noqa: E402
ITERS = 20
ts = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    pair(t_buf, a_buf, c_buf)
    ts.append(time.perf_counter() - t0)
ts.sort()
steps = NT * STEPS
print(f"  {steps} steps: min {ts[0] * 1e6:.0f} us a call, "
      f"{ts[0] * 1e6 / steps:.1f} us a step, "
      f"{2 * M * K * N_CORE / ts[0] / 1e9:.1f} GFLOP/s for the pair")
