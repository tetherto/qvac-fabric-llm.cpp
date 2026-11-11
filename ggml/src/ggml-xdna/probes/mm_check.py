# The mmul alone (akernels.mm) with A and B given in the sub-tile orders the
# prefill GEMM's expander assumes, to pin those orders down.
#   NPU_CACHE_HOME=$(mktemp -d) python probes/mm_check.py [a_order] [b_order] [c_order]
# orders: "blk" = the (outer, outer, r|s, s|t) sub-tile order, "row" = row-major
import sys
import ml_dtypes
import numpy as np
import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile

bf16 = ml_dtypes.bfloat16
M, K, N = 64, 128, 64
AO = sys.argv[1] if len(sys.argv) > 1 else "blk"
BO = sys.argv[2] if len(sys.argv) > 2 else "blk"
CO = sys.argv[3] if len(sys.argv) > 3 else "blk"
R, S, T = akernels.mm(M, K, N, input_dtype=bf16, output_dtype=np.float32).mac_dims
if len(sys.argv) > 4:   # override: r s t
    R, S, T = (int(x) for x in sys.argv[4:7])
a_ty = np.ndarray[(M * K,), np.dtype[bf16]]
b_ty = np.ndarray[(K * N,), np.dtype[bf16]]
c_ty = np.ndarray[(M * N,), np.dtype[np.float32]]


@iron.jit
def one(a: In, b: In, c: Out):
    mm = akernels.mm(M, K, N, input_dtype=bf16, output_dtype=np.float32, vectorized=True)
    fa = ObjectFifo(a_ty, name="fa", depth=1)
    fb = ObjectFifo(b_ty, name="fb", depth=1)
    fc = ObjectFifo(c_ty, name="fc", depth=1)

    def body(ai, bi, co, zero, mul):
        cc = co.acquire(1)
        zero(cc)
        aa = ai.acquire(1)
        bb = bi.acquire(1)
        mul(aa, bb, cc)
        ai.release(1)
        bi.release(1)
        co.release(1)

    w = Worker(body, fn_args=[fa.cons(), fb.cons(), fc.prod(), mm.zero, mm], tile=Tile(0, 2))

    def seq(a_h, b_h, c_h, ai, bi, co):
        ai.fill(a_h)
        bi.fill(b_h)
        co.drain(c_h, wait=True)

    rt = Runtime(seq, [a_ty, b_ty, c_ty, fa.prod(), fb.prod(), fc.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(np.float32).astype(bf16)
B = rng.standard_normal((K, N)).astype(np.float32).astype(bf16)
a = A.reshape(M // R, R, K // S, S).transpose(0, 2, 1, 3).reshape(-1) if AO == "blk" else A.reshape(-1)
b = B.reshape(K // S, S, N // T, T).transpose(0, 2, 1, 3).reshape(-1) if BO == "blk" else B.reshape(-1)
ab = iron.tensor((M * K,), dtype=bf16, device="npu")
bb = iron.tensor((K * N,), dtype=bf16, device="npu")
cb = iron.zeros((M * N,), dtype=np.float32, device="npu")
np.copyto(ab.numpy(), a)
np.copyto(bb.numpy(), b)
ab._sync_to_device()
bb._sync_to_device()
one(ab, bb, cb)
cb._sync_from_device()
g = cb.numpy()
got = g.reshape(M // R, N // T, R, T).transpose(0, 2, 1, 3).reshape(M, N) if CO == "blk" else g.reshape(M, N)
ref = A.astype(np.float64) @ B.astype(np.float64)
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"mac_dims {R, S, T} A {AO} B {BO} C {CO}: rel rms err {rel:.3e}")
