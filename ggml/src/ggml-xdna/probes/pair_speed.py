# Per-call cost of the prefill GEMM's two kernels on their own, from L1, no
# data movement: the mmul (64 x 128 x 64 bf16 -> f32) and the expander (one
# K step of a tile), each called REP times inside one core's loop.
#   NPU_CACHE_HOME=$(mktemp -d) python probes/pair_speed.py mm|exp [q8]
import hashlib
import sys
import time
from pathlib import Path
import ml_dtypes
import numpy as np
import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

WHAT = sys.argv[1] if len(sys.argv) > 1 else "mm"
Q8 = len(sys.argv) > 2 and sys.argv[2] == "q8"
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 256
bf16 = ml_dtypes.bfloat16
M, KS, N = (int(x) for x in __import__("os").environ.get("MKN", "64,128,64").split(","))
BFP = __import__("os").environ.get("BFP", "0") == "1"
TB = 10752
here = Path(__file__).resolve().parent.parent / "kernels"
src = (here / "gemm-expand.cc").read_text()
flags = ["-DMAC_T=8", f"-DEXP_UNROLL={__import__('os').environ.get('UNR', '1')}"] + __import__('os').environ.get('XF', '').split()
obj = "tspd_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
t_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
a_ty = np.ndarray[(M * KS,), np.dtype[bf16]]
b_ty = np.ndarray[(KS * N,), np.dtype[bf16]]
c_ty = np.ndarray[(M * N,), np.dtype[np.float32]]


@iron.jit
def spd(x: In, o: Out):
    fo = ObjectFifo(c_ty, name="fo", depth=1)
    if WHAT == "mm":
        mm = akernels.mm(M, KS, N, input_dtype=bf16, output_dtype=np.float32, vectorized=True,
                         emulate_bf16_mmul_with_bfp16=BFP)
        fx = ObjectFifo(a_ty, name="fx", depth=1)
        fy = ObjectFifo(b_ty, name="fy", depth=1)

        def body(xi, yi, oo, zero, mul):
            cc = oo.acquire(1)
            aa = xi.acquire(1)
            bb = yi.acquire(1)
            zero(cc)
            for _ in range_(REP):
                mul(aa, bb, cc)
            xi.release(1)
            yi.release(1)
            oo.release(1)
        w = Worker(body, fn_args=[fx.cons(), fy.cons(), fo.prod(), mm.zero, mm], tile=Tile(0, 2))

        def seq(x_h, o_h, xi, yi, oo):
            xi.fill(x_h, tap=None)
            yi.fill(x_h)
            oo.drain(o_h, wait=True)
        rt = Runtime(seq, [a_ty, c_ty, fx.prod(), fy.prod(), fo.cons()])
    else:
        k = ExternalFunction("ggml_xdna_expand", object_file_name=obj, source_string=src,
                             arg_types=[t_ty, b_ty, np.int32, np.int32],
                             include_dirs=_include_dirs(), compile_flags=flags)
        fx = ObjectFifo(t_ty, name="fx", depth=1)

        def body(xi, oo, kexp, b):
            t = xi.acquire(1)
            oo.acquire(1)
            for _ in range_(REP):
                kexp(t, b, 0, 1 if Q8 else 0)
            xi.release(1)
            oo.release(1)
        from aie.iron import Buffer
        w = Worker(body, fn_args=[fx.cons(), fo.prod(), k, Buffer(b_ty, name="bexp")],
                   tile=Tile(0, 2), stack_size=0x800)

        def seq(x_h, o_h, xi, oo):
            xi.fill(x_h)
            oo.drain(o_h, wait=True)
        rt = Runtime(seq, [t_ty, c_ty, fx.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


x = iron.zeros((M * KS if WHAT == "mm" else TB,), dtype=bf16 if WHAT == "mm" else np.uint8,
               device="npu")
o = iron.zeros((M * N,), dtype=np.float32, device="npu")
spd(x, o)
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    spd(x, o)
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"{WHAT}{' q8' if Q8 else ''} {M}x{KS}x{N}{' bfp16' if BFP else ''}: {REP} calls, "
      f"min {ts[0] * 1e6:.0f} us -> {ts[0] * 1e6 / REP:.2f} us a call, "
      f"{M * KS * N * REP / ts[0] / 1e9:.1f} GMAC/s")
