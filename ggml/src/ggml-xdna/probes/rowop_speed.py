# Per-row cost of the merged design's row ops (probes/rowop-s1.cc) on one core, from L1, no data movement:
# each case calls ggml_xdna_rowop REP times inside one core's loop, at two REP counts, and the slope between them is
# the cost of one call (the command's own floor cancels out). Sizes are EmbeddingGemma 2's.
#   NPU_CACHE_HOME=$(mktemp -d) python probes/rowop_speed.py
import hashlib
import struct
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "kernels"))
import kernelsrc  # noqa: E402

N_MAX = 2048                       # floats a row buffer holds (n_ff)
HDR_WORDS = 16
REPS = (32, 256)
RUNS = 10
MODE_WORD, SIZE_WORD, ROT_WORD, SCALE_WORD, ROW_WORD, SCRATCH_WORD = 10, 11, 12, 13, 14, 15
CASES = {                          # name: (mode, row size, rotated dims)
    "R1 head norm + rope, D 256": (1, 256, 256),
    "R1 head norm, D 256": (1, 256, 0),
    "R2 post-norm + add + pre-norm, 512": (2, 512, 0),
    "R4 gelu(gate) * x, 2048": (4, 2048, 0),
}
bf16 = ml_dtypes.bfloat16
x_ty = np.ndarray[(N_MAX,), np.dtype[np.float32]]
y_ty = np.ndarray[(4 * N_MAX,), np.dtype[np.float32]]
o_ty = np.ndarray[(N_MAX,), np.dtype[bf16]]
n_ty = np.ndarray[(HDR_WORDS,), np.dtype[np.int32]]
src = kernelsrc.load(here / "rowop-s1.cc")
obj = "rowspd_" + hashlib.md5(src.encode()).hexdigest()[:8] + ".o"


def header(mode: int, n: int, n_rot: int) -> np.ndarray:
    h = np.zeros(HDR_WORDS, np.int32)
    h[MODE_WORD], h[SIZE_WORD], h[ROT_WORD], h[ROW_WORD], h[SCRATCH_WORD] = mode, n, n_rot, 0, 3 * N_MAX
    h[SCALE_WORD] = struct.unpack("<i", struct.pack("<f", 1.0))[0]
    return h


@iron.jit
def spd(x: In, o: Out, *, MODE: CompileTime[int] = 2, N: CompileTime[int] = 512, ROT: CompileTime[int] = 0,
        REP: CompileTime[int] = 32):
    k = ExternalFunction("ggml_xdna_rowop", object_file_name=obj, source_string=src,
                         arg_types=[x_ty, y_ty, o_ty, n_ty], include_dirs=_include_dirs())
    fx = ObjectFifo(x_ty, name="fx", depth=1)
    fo = ObjectFifo(o_ty, name="fo", depth=1)

    def body(xi, oo, row, y, cnt):
        xx = xi.acquire(1)
        out = oo.acquire(1)
        for _ in range_(REP):
            row(xx, y, out, cnt)
        xi.release(1)
        oo.release(1)

    w = Worker(body, fn_args=[fx.cons(), fo.prod(), k, Buffer(y_ty, name="y", initial_value=np.full(4 * N_MAX, 0.5,
                                                                                                   np.float32)),
                              Buffer(n_ty, name="cnt", initial_value=header(MODE, N, ROT))],
               tile=Tile(0, 2), stack_size=0xB00)

    def seq(x_h, o_h, xi, oo):
        xi.fill(x_h)
        oo.drain(o_h, wait=True)

    rt = Runtime(seq, [x_ty, o_ty, fx.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def best_us(mode: int, n: int, rot: int, rep: int, x, o) -> float:
    spd(x, o, MODE=mode, N=n, ROT=rot, REP=rep)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        spd(x, o, MODE=mode, N=n, ROT=rot, REP=rep)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e6


def main() -> None:
    x = iron.tensor(np.linspace(-2.0, 2.0, N_MAX, dtype=np.float32), dtype=np.float32, device="npu")
    o = iron.zeros((N_MAX,), dtype=bf16, device="npu")
    for name, (mode, n, rot) in CASES.items():
        lo, hi = (best_us(mode, n, rot, r, x, o) for r in REPS)
        per = (hi - lo) / (REPS[1] - REPS[0])
        print(f"{name}: {REPS[0]} calls {lo:.0f} us, {REPS[1]} calls {hi:.0f} us -> {per:.2f} us a row "
              f"({n / per:.0f} values/us)")


if __name__ == "__main__":
    main()
