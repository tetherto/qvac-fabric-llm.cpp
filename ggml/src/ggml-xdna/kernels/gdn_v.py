#!/usr/bin/env python3
# gdn_v.py -*- Python -*-
#
# Gated-delta-net recurrent step, bf16-vectorized, replacing the scalar gdn
# stage of attn_cg.xclbin (conv + norm + gdn).  This xclbin is the gdn stage
# ONLY: it consumes the per-(head,chunk) pkv objects that the norm stage of the
# conv+norm kernel produces (identical layout to attn_cg: one 387-float fp32
# object per chunk = kn(128)|qn(128)|v16(16, at +256)|eg|b|scale at +384) plus
# the persistent 16x128 state rows of a chunk in BF16, and writes the updated
# state rows back (BF16) plus the 16 attn values (fp32, head-major gather).
#
# Row math (literal ggml / attn_cg scalar form, state row-major A[j][i]):
#   dotk[j] = sum_i row[j,i]*k[i]
#   dj      = (v[j] - eg*dotk[j]) * b
#   row'[j,i] = row[j,i]*eg + dj*k[i]                (bf16 vectors)
#   attn[j]   = (eg*dotq[j] + dj*(k.q)) * scale      with dotq[j]=sum_i row[j,i]*q[i]
#
# The decay eg factors out of the two 128-dots, so the matvec products are
# taken over the raw rows and eg/dj applied after.  k.q is a per-chunk 128-dot.
# All heavy elementwise math is bf16 x bf16 vector (32 lanes) with fp32
# accumulator lanes (aie::mul / aie::mac), exactly the op set that is
# validated on this IRON stack; the fp32 values that only touch a few scalars
# per row (eg, b, scale, delta) stay fp32.  State is persisted in bf16, so the
# device state BO is half of the fp32 size (512 KiB).
#
# Numeric check: fused_core_check.py drives this design with the norm stage's
# device output for three tokens, the state carried in the device BO, against
# ggml's CPU recurrence and against sim_chunk_bf16 below:
#   python fused_core_check.py -d npu2 --workdir <fresh>
#
# Geometry mirrors attn_cg: NCOL=8 shim columns, a worker per column consumes
# N_VH*N_OBJ/NCOL = 16 chunk objects.  Per (head,chunk) object sizes: pkv
# 387 fp32 (1548 B), state 2048 bf16 (4096 B), state' 4096 B, attn 16 fp32.

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

try:
    from ml_dtypes import bfloat16
except Exception:
    bfloat16 = None   # design host dtype helper; real imports in --run

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

S_V = 128
N_VH = 16
CHUNK = 16                 # state rows per (head,chunk) object
N_OBJ = S_V // CHUNK       # 8 chunks per head
ROWS = CHUNK * S_V         # 2048 rows-values per chunk object
PKV_N = 3 * S_V + 3        # kn(128)|qn(128)|v16@256|eg,b,scale@384 = 387
O_V = 2 * S_V              # 256
O_EG = 3 * S_V             # 384
NCOL = 8
N_OBJ_PER_COL = N_VH * N_OBJ // NCOL   # 16


GDN_V_SRC = Path(__file__).resolve().parent / "gdn-v.cc"


def _kernel_src():
    src = GDN_V_SRC.read_text()
    for k, v in (("S_V", S_V), ("CHUNK", CHUNK), ("O_V", O_V), ("O_EG", O_EG)):
        src = src.replace(f"@{k}@", str(v))
    return src


@iron.jit
def ggml_xdna_gdn_v(*, ncol: CompileTime[int], dev_name: CompileTime[str] = "npu2"):
    PKV_T = np.ndarray[(PKV_N,), np.dtype[np.float32]]
    SIN_T = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    SOUT_T = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    ATT_T = np.ndarray[(CHUNK,), np.dtype[np.float32]]

    kern = iron.ExternalFunction(name="ggml_xdna_gdn_v", source_string=_kernel_src(),
                                 arg_types=[PKV_T, SIN_T, SOUT_T, ATT_T],
                                 compile_flags=["-O2", "-DNDEBUG"], inline=True)

    PKV_g = np.ndarray[(N_VH * N_OBJ * PKV_N,), np.dtype[np.float32]]
    SIN_g = np.ndarray[(N_VH * N_OBJ * ROWS,), np.dtype[bfloat16]]
    SOUT_g = np.ndarray[(N_VH * N_OBJ * ROWS,), np.dtype[bfloat16]]
    ATTN_g = np.ndarray[(N_VH * S_V,), np.dtype[np.float32]]

    workers = []
    rt_args = [PKV_g, SIN_g, SOUT_g, ATTN_g]

    for col in range(ncol):
        p3 = ObjectFifo(PKV_T, name=f"vp3_{col}", depth=2)
        p2 = p3.cons().forward(obj_type=PKV_T, name=f"vp2_{col}", tile=Tile(col, 1))
        s3 = ObjectFifo(SIN_T, name=f"vs3_{col}", depth=2)
        s2 = s3.cons().forward(obj_type=SIN_T, name=f"vs2_{col}", tile=Tile(col, 1))
        o23 = ObjectFifo(SOUT_T, name=f"vo23_{col}", depth=2)
        o12 = o23.prod().join([0], obj_types=[SOUT_T], names=[f"vo12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        a23 = ObjectFifo(ATT_T, name=f"va23_{col}", depth=2)
        a12 = a23.prod().join([0], obj_types=[ATT_T], names=[f"va12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def gdn_fn(pc, sc, oc, ac, k, nobj=N_OBJ_PER_COL):
            for _ in range_(nobj):
                pp = pc.acquire(1)
                si = sc.acquire(1)
                so = oc.acquire(1)
                ao = ac.acquire(1)
                k(pp, si, so, ao)
                oc.release(1)
                ac.release(1)
                pc.release(1)
                sc.release(1)

        workers.append(Worker(gdn_fn, [p2.cons(), s2.cons(), o12.prod(),
                                       a12.prod(), kern],
                              tile=Tile(col, 2), stack_size=0x3000))
        rt_args += [p3.prod(tile=Tile(col, 0)),
                    s3.prod(tile=Tile(col, 0)),
                    o23.cons(tile=Tile(col, 0)),
                    a23.cons(tile=Tile(col, 0))]

    def seq_fn(PKV, SIN, SOUT, ATTN, *fifos):
        it = iter(fifos)
        pprod = []
        sprod = []
        ocons = []
        aconss = []
        for col in range(ncol):
            pprod.append(next(it))
            sprod.append(next(it))
            ocons.append(next(it))
            aconss.append(next(it))
        # chunk global index c = head*N_OBJ + j ; column = c % ncol
        for col in range(ncol):
            for c in range(col, N_VH * N_OBJ, ncol):
                head = c // N_OBJ
                j = c % N_OBJ
                gi = TaskGroup()
                pprod[col].fill(PKV, tap=TensorAccessPattern(
                    (N_VH * N_OBJ * PKV_N,),
                    offset=head * N_OBJ * PKV_N + j * PKV_N,
                    sizes=[PKV_N], strides=[1]), group=gi)
                sprod[col].fill(SIN, tap=TensorAccessPattern(
                    (N_VH * N_OBJ * ROWS,), offset=c * ROWS,
                    sizes=[ROWS], strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                ocons[col].drain(SOUT, tap=TensorAccessPattern(
                    (N_VH * N_OBJ * ROWS,), offset=c * ROWS,
                    sizes=[ROWS], strides=[1]), wait=True, group=go)
                aconss[col].drain(ATTN, tap=TensorAccessPattern(
                    (N_VH * S_V,), offset=head * S_V + j * CHUNK,
                    sizes=[CHUNK], strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=ncol), rt, workers)
    return prog.resolve_program()


# ---- host fp32 scalar reference of one chunk (the attn_cg gdnc formula) ----
def scalar_chunk(pkv, rows_f32, dtype=np.float32):
    """pkv [387] fp32, rows_f32 [16,128] fp32. Returns (rows' f32 [16,128],
    attn[16] f32) computed in the literal attn_cg.gdnc order."""
    kn = pkv[0:S_V]
    qn = pkv[S_V:2 * S_V]
    v16 = pkv[O_V:O_V + CHUNK]
    eg = dtype(pkv[O_EG])
    b = dtype(pkv[O_EG + 1])
    scale = dtype(pkv[O_EG + 2])
    rows = rows_f32.astype(dtype)
    sout = np.zeros_like(rows)
    attn = np.zeros(CHUNK, dtype=dtype)
    for j in range(CHUNK):
        dotk = dtype(0)
        for i in range(S_V):
            dotk += dtype(rows[j, i] * eg) * dtype(kn[i])
        dj = (dtype(v16[j]) - dotk) * b
        dotq = dtype(0)
        for i in range(S_V):
            val = dtype(rows[j, i] * eg) + dj * dtype(kn[i])
            sout[j, i] = val
            dotq += val * dtype(qn[i])
        attn[j] = dotq * scale
    return sout, attn


def main():
    ap = argparse.ArgumentParser(prog="ggml_xdna_gdn_v")
    add_compile_args(ap)
    ap.add_argument("--cols", type=int, default=NCOL)
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reuse", action="store_true")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.tag is None:
        opts.tag = time.strftime("gdn_%H%M%S")
    wd = os.path.join(opts.workdir, opts.tag)
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "gdn")
    xclbin_path = base + ".xclbin"
    insts_path = base + ".insts.bin"
    if opts.reuse and os.path.exists(xclbin_path) and os.path.exists(insts_path):
        print("reuse", xclbin_path)
        return
    t0 = time.time()
    spec = ggml_xdna_gdn_v.specialize(ncol=opts.cols, dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(xclbin_path=xclbin_path,
                                           inst_path=insts_path)
    print(f"compiled {xclbin_path} ({time.time()-t0:.0f}s)")


def sim_chunk_bf16(pkv, rows_f32):
    """Host emulation of the exact ggml_xdna_gdn_v kernel op sequence (per-32-lane fp32
    accumulate then bf16 store). rows_f32 [16,128] fp32."""
    from ml_dtypes import bfloat16
    kn = np.asarray(pkv[0:S_V].astype(bfloat16), dtype=np.float32)
    qn = np.asarray(pkv[S_V:2 * S_V].astype(bfloat16), dtype=np.float32)
    v16 = pkv[O_V:O_V + CHUNK]
    eg = pkv[O_EG]
    b = pkv[O_EG + 1]
    scale = pkv[O_EG + 2]
    kb = kn.reshape(4, 32)
    qb = qn.reshape(4, 32)
    kq = 0.0
    for blk in range(4):
        kq += float(np.sum(kb[blk] * qb[blk]))
    rows = np.asarray(rows_f32.astype(bfloat16), dtype=np.float32).reshape(CHUNK, S_V)
    sout = np.zeros((CHUNK, S_V), dtype=np.float32)
    attn = np.zeros(CHUNK, dtype=np.float32)
    eg_b = float(np.float32(bfloat16(eg)))
    for j in range(CHUNK):
        r = rows[j].reshape(4, 32)
        dotk = 0.0
        dotq = 0.0
        for blk in range(4):
            dotk += float(np.sum(r[blk] * kb[blk]))
            dotq += float(np.sum(r[blk] * qb[blk]))
        dj = (float(v16[j]) - eg * dotk) * b
        attn[j] = scale * (eg * dotq + dj * kq)
        dj_b = float(np.float32(bfloat16(dj)))
        acc = r * eg_b + dj_b * kb     # fp32 lanes of exact bf16 products
        sout[j] = np.asarray(acc.astype(bfloat16), dtype=np.float32).reshape(S_V)
    return sout, attn


if __name__ == "__main__":
    main()
