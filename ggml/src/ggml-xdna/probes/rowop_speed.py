# Per-row cost of the merged design's row command kernel (kernels/rowop.cc) on one core, from L1, no data movement:
# each case sets the core up once and calls ggml_xdna_rowop REP times inside one core's loop, at two REP counts, and the
# slope between them is the cost of one call, a block of 8 rows (the command's own floor cancels out). Sizes are
# EmbeddingGemma 2's.
#   NPU_CACHE_HOME=$(mktemp -d) python probes/rowop_speed.py
import hashlib
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

D = 256
BLOCK_ROWS = 8
SETUP_BYTES = 10752                # a core's W buffer, which the setup object fills
C_BF16 = 4096                      # a C object: two blocks
ACC_F32 = 4096
PARAM_WORDS = 16
MODE_R1_BLOCK = 5
RMS_EPS = 1e-6
STACK = 0xB00                      # kernels/pfx.py's row-op stack
REPS = (32, 256)
RUNS = 10
CASES = {                          # name: rotated dims
    "R1 block norm + rope, 8 x D 256": 256,
    "R1 block norm, 8 x D 256": 0,
}
bf16 = ml_dtypes.bfloat16
x_ty = np.ndarray[(BLOCK_ROWS * D,), np.dtype[np.float32]]
s_ty = np.ndarray[(SETUP_BYTES // 4,), np.dtype[np.int32]]
t_ty = np.ndarray[(BLOCK_ROWS * D,), np.dtype[np.float32]]      # cos [8][128], then sin
o_ty = np.ndarray[(C_BF16,), np.dtype[bf16]]
a_ty = np.ndarray[(ACC_F32,), np.dtype[np.float32]]
src = kernelsrc.load(here.parent / "kernels" / "rowop.cc")
obj = "rowspd_" + hashlib.md5(src.encode()).hexdigest()[:8] + ".o"


def setup_object(n_rot: int) -> np.ndarray:
    """kernels/rowop.cc's parameters, and a norm weight of 1 as its bf16 split (hi 1, lo 0)."""
    s = np.zeros(SETUP_BYTES // 4, np.int32)
    s[:5] = [MODE_R1_BLOCK, D, n_rot, np.float32(1.0).view(np.int32), np.float32(RMS_EPS).view(np.int32)]
    gs = np.zeros(2 * D, np.uint16)
    gs[:D] = np.float32(1.0).view(np.uint32) >> 16
    s[PARAM_WORDS:PARAM_WORDS + D] = gs.view(np.int32)
    return s


@iron.jit
def spd(x: In, o: Out, *, ROT: CompileTime[int] = 256, REP: CompileTime[int] = 32):
    mk = lambda name, types: ExternalFunction(name, object_file_name=obj, source_string=src, arg_types=types,
                                              include_dirs=_include_dirs())
    k_setup = mk("ggml_xdna_row_setup", [s_ty, a_ty])
    k_row = mk("ggml_xdna_rowop", [x_ty, t_ty, o_ty, a_ty])
    fx = ObjectFifo(x_ty, name="fx", depth=1)
    fo = ObjectFifo(o_ty, name="fo", depth=1)

    def body(xi, oo, setup, row, sb, tab, acc):
        xx = xi.acquire(1)
        out = oo.acquire(1)
        setup(sb, acc)
        for _ in range_(REP):
            row(xx, tab, out, acc)
        xi.release(1)
        oo.release(1)

    w = Worker(body, fn_args=[fx.cons(), fo.prod(), k_setup, k_row,
                              Buffer(s_ty, name="setup", initial_value=setup_object(ROT)),
                              Buffer(t_ty, name="tab", initial_value=np.full(BLOCK_ROWS * D, 0.5, np.float32)),
                              Buffer(a_ty, name="acc")],
               tile=Tile(0, 2), stack_size=STACK)

    def seq(x_h, o_h, xi, oo):
        xi.fill(x_h)
        oo.drain(o_h, wait=True)

    rt = Runtime(seq, [x_ty, o_ty, fx.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def best_us(rot: int, rep: int, x, o) -> float:
    spd(x, o, ROT=rot, REP=rep)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        spd(x, o, ROT=rot, REP=rep)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e6


def main() -> None:
    x = iron.tensor(np.linspace(-2.0, 2.0, BLOCK_ROWS * D, dtype=np.float32), dtype=np.float32, device="npu")
    o = iron.zeros((C_BF16,), dtype=bf16, device="npu")
    for name, rot in CASES.items():
        lo, hi = (best_us(rot, r, x, o) for r in REPS)
        per = (hi - lo) / (REPS[1] - REPS[0]) / BLOCK_ROWS
        print(f"{name}: {REPS[0]} calls {lo:.0f} us, {REPS[1]} calls {hi:.0f} us -> {per:.3f} us a row "
              f"({D / per:.0f} values/us)")


if __name__ == "__main__":
    main()
