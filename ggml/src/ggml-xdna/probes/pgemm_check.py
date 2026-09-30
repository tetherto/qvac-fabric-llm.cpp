# The prefill GEMM on the whole array against
# numpy: C (M x N) = A (M x K) @ W (K x N), W the decode GEMV's packed tiles in
# the decode's own order, every one of the 32 cores expanding its tiles and
# running the bfp16 mmul.
#
#   core (c, row), row 2..5: the GEMV core (c, r = (row - 2) % 2) - the tiles
#   it gets in the decode - and M block mb = (row - 2) // 2 of 64 rows; so the
#   two M blocks share every weight tile (the MemTile sends each half of a
#   column's object to two cores) and each A block goes to 16 cores.
#
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/pgemm_check.py q4g32|q8g16 [M K N]
import hashlib
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
import aie.iron.kernels as akernels
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs
from aie.helpers.taplib import TensorAccessPattern

here = Path(__file__).resolve().parent.parent / "kernels"
sys.path.insert(0, str(here))
import gemv_q4  # noqa: E402

FMT = sys.argv[1] if len(sys.argv) > 1 else "q4g32"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 128
K = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
Q8 = FMT != "q4g32"
bf16 = ml_dtypes.bfloat16
COLS, NC, KS = 8, 64, 64               # columns, a core's columns, K a step
R, S, T = 8, 8, 8                      # the bfp16 mmul's sub-tiles
MB = 64                                # M rows a core
K_TILE = 128 if Q8 else 256
STEPS = K_TILE // KS
NT = K // K_TILE
NSTEP = K // KS
CHUNK = COLS * 2 * NC                  # 1024 output columns a GEMV chunk
NCH = N // CHUNK
NMB = M // (2 * MB)
TB = gemv_q4.tile_bytes(FMT, K_TILE, NC)
assert M % (2 * MB) == 0 and N % CHUNK == 0 and K % K_TILE == 0

src = (here / "gemm-expand.cc").read_text()
flags = [f"-DMAC_T={T}", f"-DEXP_STEP={KS}", "-DEXP_UNROLL=8"] + __import__("os").environ.get("XF", "").split()
obj = "tpg_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
t_ty = np.ndarray[(TB,), np.dtype[np.uint8]]
w2_ty = np.ndarray[(2 * TB,), np.dtype[np.uint8]]
a_ty = np.ndarray[(MB * KS,), np.dtype[bf16]]
b_ty = np.ndarray[(KS * NC,), np.dtype[bf16]]
c_ty = np.ndarray[(MB * NC,), np.dtype[np.float32]]
c4_ty = np.ndarray[(4 * MB * NC,), np.dtype[np.float32]]
W_BYTES = COLS * NCH * NT * 2 * TB
A_ELEMS = NMB * 2 * NSTEP * MB * KS
C_ELEMS = NMB * NCH * COLS * 4 * MB * NC


@iron.jit
def pgemm(w: In, a: In, c: Out):
    mm = akernels.mm(MB, KS, NC, input_dtype=bf16, output_dtype=np.float32, vectorized=True,
                     emulate_bf16_mmul_with_bfp16=True)
    kexp = ExternalFunction("ggml_xdna_expand", object_file_name=obj, source_string=src,
                            arg_types=[t_ty, b_ty, np.int32, np.int32],
                            include_dirs=_include_dirs(), compile_flags=flags)
    # weights: a column's object is both rows' tiles; halves to rows r, r+2
    w_col, w_half = [], []
    for col in range(COLS):
        f = ObjectFifo(w2_ty, name=f"w{col}", depth=2)
        w_col.append(f)
        w_half.append(f.cons().split(offsets=[0, TB], obj_types=[t_ty, t_ty], depths=[2, 2],
                                     names=[f"w{col}_0", f"w{col}_1"], tile=Tile(col, 1)))
    a_f = [ObjectFifo(a_ty, name=f"a{mb}", depth=2) for mb in range(2)]
    c_col, c_core = [], []
    for col in range(COLS):
        f = ObjectFifo(c4_ty, name=f"c{col}", depth=1)
        c_col.append(f)
        c_core.append(f.prod().join(offsets=[i * MB * NC for i in range(4)],
                                    obj_types=[c_ty] * 4, depths=[1] * 4,
                                    names=[f"c{col}_{i}" for i in range(4)], tile=Tile(col, 1)))

    def core(w_in, a_in, c_out, zero, mul, k, bb):
        for _ in range_(NMB * NCH):
            cc = c_out.acquire(1)
            zero(cc)
            for _ in range_(NT):
                t = w_in.acquire(1)
                for s in range(STEPS):
                    k(t, bb, s, 1 if Q8 else 0)
                    aa = a_in.acquire(1)
                    mul(aa, bb, cc)
                    a_in.release(1)
                w_in.release(1)
            c_out.release(1)

    workers = []
    for col in range(COLS):
        for i in range(4):
            r, mb = i % 2, i // 2
            workers.append(Worker(core, fn_args=[w_half[col][r].cons(), a_f[mb].cons(),
                                                 c_core[col][i].prod(), mm.zero, mm, kexp,
                                                 Buffer(b_ty, name=f"b{col}_{i}")],
                                  tile=Tile(col, 2 + i), stack_size=0xC00))

    def seq(w_h, a_h, c_h, *eps):
        wi, ai, co = eps[:COLS], eps[COLS:COLS + 2], eps[COLS + 2:]
        # one fill a stream, repeats by zero strides; every dimension under the
        # shim's 10-bit sizes
        # (a zero stride only in the outermost dimension: the M blocks are
        # fills of their own, the chunks' repeat of A is a zero stride)
        span = NCH * NT * 2 * TB
        q = 2 * TB // 512
        n = NSTEP * MB * KS
        for blk in range(NMB):
            for col in range(COLS):
                wi[col].fill(w_h, tap=TensorAccessPattern([1, W_BYTES], col * span,
                                                          [NCH * NT, q, 512],
                                                          [2 * TB, 512, 1]))
            for mb in range(2):
                ai[mb].fill(a_h, tap=TensorAccessPattern([1, A_ELEMS], (blk * 2 + mb) * n,
                                                         [NCH, 1, n // 512, 512],
                                                         [0, n, 512, 1]))
        per = 4 * MB * NC
        for col in range(COLS):
            co[col].drain(c_h, tap=TensorAccessPattern([1, C_ELEMS], col * per,
                                                       [NMB * NCH, per // 512, 512],
                                                       [COLS * per, 512, 1]), wait=True)

    rt = Runtime(seq, [np.ndarray[(W_BYTES,), np.dtype[np.uint8]],
                       np.ndarray[(A_ELEMS,), np.dtype[bf16]],
                       np.ndarray[(C_ELEMS,), np.dtype[np.float32]]] +
                 [f.prod(tile=Tile(col, 0)) for col, f in enumerate(w_col)] +
                 [f.prod(tile=Tile(6 + mb, 0)) for mb, f in enumerate(a_f)] +
                 [f.cons(tile=Tile(col, 0)) for col, f in enumerate(c_col)])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


rng = np.random.default_rng(2)
ng = K_TILE // 32
Wfull = np.zeros((K, N))
wbuf = np.zeros(W_BYTES, np.uint8)


def pr(x):
    p = gemv_q4.split_pairs(x)
    return p[0::2].astype(np.float64) + p[1::2].astype(np.float64)


# the decode order: [column][chunk][K tile][row], core (c, r) of chunk ch
# owning columns ch * 1024 + (c * 2 + r) * 64 .. + 64
for col in range(COLS):
    for ch in range(NCH):
        for t in range(NT):
            for r in range(2):
                lo, hi = (0, 16) if not Q8 else (-128, 128)
                codes = rng.integers(lo, hi, size=(K_TILE, NC)).astype(np.int32)
                d8 = rng.integers(1, 64, size=(ng, NC)).astype(np.int32)
                m8 = rng.integers(0, 64, size=(ng, NC)).astype(np.int32)
                dS = (rng.standard_normal((1, NC)) * 0.002).astype(np.float32)
                mS = (rng.standard_normal((1, NC)) * 0.005).astype(np.float32)
                off = (((col * NCH + ch) * NT + t) * 2 + r) * TB
                wbuf[off:off + TB] = gemv_q4.pack_weight_tile(FMT, codes, d8, m8, dS, mS)
                g = np.arange(K_TILE) // 32
                n0 = ch * CHUNK + (col * 2 + r) * NC
                Wfull[t * K_TILE:(t + 1) * K_TILE, n0:n0 + NC] = \
                    codes * (pr(dS)[0][None, :] * d8[g]) + pr(mS)[0][None, :] * m8[g]
A = rng.standard_normal((M, K)).astype(np.float32).astype(bf16)
# A as [M block][mb][K step][(64/r, KS/s, r, s) blocks]
ablk = A.reshape(NMB, 2, MB // R, R, NSTEP, KS // S, S).transpose(0, 1, 4, 2, 5, 3, 6)

w_t = iron.tensor((W_BYTES,), dtype=np.uint8, device="npu")
a_t = iron.tensor((A_ELEMS,), dtype=bf16, device="npu")
c_t = iron.zeros((C_ELEMS,), dtype=np.float32, device="npu")
np.copyto(w_t.numpy(), wbuf)
np.copyto(a_t.numpy(), ablk.reshape(-1))
w_t._sync_to_device()
a_t._sync_to_device()
pgemm(w_t, a_t, c_t)
c_t._sync_from_device()
# C as [M block][chunk][column][row i][(64/r, 64/t, r, t) blocks]
g6 = c_t.numpy().reshape(NMB, NCH, COLS, 4, MB // R, NC // T, R, T)
got = np.zeros((M, N))
for blk in range(NMB):
    for ch in range(NCH):
        for col in range(COLS):
            for i in range(4):
                r, mb = i % 2, i // 2
                rows = blk * 2 * MB + mb * MB
                n0 = ch * CHUNK + (col * 2 + r) * NC
                got[rows:rows + MB, n0:n0 + NC] = \
                    g6[blk, ch, col, i].transpose(0, 2, 1, 3).reshape(MB, NC)
ref = A.astype(np.float64) @ Wfull.astype(np.float32).astype(bf16).astype(np.float64)
rel = np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean())
print(f"{FMT} M={M} K={K} N={N}: rel rms err {rel:.3e}")
for blk in range(NMB):
    for ch in range(NCH):
        line = []
        for col in range(COLS):
            for i in range(4):
                r, mb = i % 2, i // 2
                rows = slice(blk * 2 * MB + mb * MB, blk * 2 * MB + mb * MB + MB)
                n0 = ch * CHUNK + (col * 2 + r) * NC
                e = got[rows, n0:n0 + NC] - ref[rows, n0:n0 + NC]
                line.append(f"{np.sqrt((e ** 2).mean() / (ref[rows, n0:n0 + NC] ** 2).mean()):.0e}")
        print(f"  blk {blk} chunk {ch}: " + " ".join(line))
print("PASS" if rel < 2e-2 else "FAIL")
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    pgemm(w_t, a_t, c_t)
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"  min {ts[0] * 1e3:.3f} ms a call, {2 * M * K * N / ts[0] / 1e12:.2f} TFLOP/s")
