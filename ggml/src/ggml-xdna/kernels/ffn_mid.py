#!/usr/bin/env python3
# ffn_mid.py -*- Python -*-
#
# S1 core, standalone: compute the FFN-mid column of a decode token entirely
# on the NPU from ggml Q4_K gate/up weights (block-dequant gemv, silu in-core):
#   mid[m] = silu( hff . W_gate[:,m] ) * ( hff . W_up[:,m] ),   m in [0,3584)
# hff is the (already RMS-normed) [1024] hidden input, host-fed f32.  The two
# Q4_K dequant-gemvs and the silu/gate multiply run in one AIE core.
#
# Each tile object = [ hff (K*4 f32) | TILE gate columns (Q4_K) | TILE up
# columns (Q4_K) ]; a gate/up column pair at mid index m is
# blocks_k*144 bytes each.  The kernel dequants both columns in-core (exact
# ggml Q4_K port from proj_qK.py) and dots them with hff.
#
# silu is computed with the gdn_layer.py convh in-core trick (no expf on AIE):
#   silu(a) = a * 0.5 * (1 + tanh(a/2))    via aie::tanh<bfloat16>
# (silu.cc / swiglu_mm.cc epilogue verbatim).  The chain narrows the f32 dots
# to bf16 and rounds every multiply/add to bf16, so mid matches an exact e^-x
# silu only to ~1e-2 abs (bf16, expected for FFN; documented).  The oracle
# mirrors the same bf16-tanh path, so the residual kernel-vs-reference diff is
# just a few bf16 ulps from the AIE tanh approximation (measured max ~1e-2,
# median ~1e-4).
#
# Geometry: 8 AIE columns x 448 mid columns; per-tile object sized to fit L1
# at input fifo depth 1 (~20 KB proven limit, see proj_qK.py).
#
# Usage:
#   python ffn_mid.py -d npu2 --workdir /tmp/opencode/f1 --run
#   python ffn_mid.py -d npu2 --workdir /tmp/opencode/f1 --tile 8 --run

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proj_qK import fp16, gsm, oracle_col, BLK, SC_OFF, load_reader  # noqa: E402

from ml_dtypes import bfloat16  # noqa: E402

K = 1024                 # hidden (gate/up row) dim = hff width
N_MID = 3584             # gate/up output (mid) dim
CB = 576                 # Q4_K bytes of one output column: (K/256)*144
HK = K * 4               # hff f32 bytes


def geom(ncol: int, n_out: int, tile: int, k: int):
    """Layout constants for a (ncol, n_out, tile, k) build.  gate/up fixed
    Q4_K, so col_bytes is 144 bytes per 256-row ggml block."""
    if n_out % ncol:
        raise SystemExit(f"N={n_out} not divisible by ncol={ncol}")
    if k % BLK:
        raise SystemExit(f"K={k} not a multiple of BLK={BLK}")
    cols_per_col = n_out // ncol
    if cols_per_col % tile:
        raise SystemExit(f"cols/col={cols_per_col} not divisible by tile={tile}")
    blocks_k = k // BLK
    col_bytes = blocks_k * 144
    nt = cols_per_col // tile
    obj_bytes = k * 4 + 2 * tile * col_bytes   # hff + gate cols + up cols
    return dict(
        ncol=ncol, n_out=n_out, tile=tile, k=k, blocks_k=blocks_k,
        cols_per_col=cols_per_col, nt=nt, col_bytes=col_bytes,
        obj_bytes=obj_bytes, tile_qbytes=tile * col_bytes,
    )


_FP16F_SRC = """static float fp16f(uint16_t h) {
    uint32_t s=(h>>15)&1u,e=(h>>10)&0x1fu,m=h&0x3ffu,f;
    if(e==0){ if(m==0) f=s<<31; else { int e2=-14; uint32_t mm=m;
        while((mm&0x400u)==0){mm<<=1;e2--;} mm&=0x3ffu;
        f=(s<<31)|((uint32_t)(e2+127)<<23)|(mm<<13); } }
    else if(e==31) f=(s<<31)|(0xffu<<23)|(m<<13);
    else f=(s<<31)|((uint32_t)((int)e-15+127)<<23)|(m<<13);
    float o; memcpy(&o,&f,4); return o;
}
"""


def _kernel_src(g):
    """C for the fused gate/up Q4_K dequant-gemv + bf16-tanh silu epilogue.
    The gemv body is the proj_qK.py Q4_K port; the silu/mul epilogue is the
    gdn_layer.py convh / swiglu_mm.cc bf16 silu path verbatim."""
    return """#include <stdint.h>
#include <string.h>
#include <aie_api/aie.hpp>
using namespace aie;
""" + _FP16F_SRC + """static void gsm4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;} else {
        *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); } }
static float deqdot4(const uint8_t * col, const float * h) {
    // Q4_K column gemv, exact ggml dequantize_row_q4_K order (proj_qK.py)
    float acc = 0.0f;
    int el = 0;
    for (int b = 0; b < %(blocks_k)d; ++b) {
        const uint8_t * blk = col + b * %(qb)d;
        const float d = fp16f((uint16_t)(blk[0] | (blk[1] << 8)));
        const float mn = fp16f((uint16_t)(blk[2] | (blk[3] << 8)));
        const uint8_t * sc = blk + %(sc_off)d;
        const uint8_t * ql = blk + %(ql_off)d;
        int is = 0;
        for (int j2 = 0; j2 < %(blk)d; j2 += 64) {
            uint8_t s0, m0; gsm4(is + 0, sc, &s0, &m0);
            const float d1 = d * s0, m1 = mn * m0;
            uint8_t s1, m1b; gsm4(is + 1, sc, &s1, &m1b);
            const float d2 = d * s1, m2 = mn * m1b;
            for (int l = 0; l < 32; ++l) {
                float v = d1 * (ql[l] & 0xF) - m1;
                acc += v * h[el++];
            }
            for (int l = 0; l < 32; ++l) {
                float v = d2 * (ql[l] >> 4) - m2;
                acc += v * h[el++];
            }
            ql += 32; is += 2;
        }
    }
    return acc;
}
extern "C" void ffnmid(const uint8_t * obj, float * out) {
    const float * h = (const float *)(obj + 0);
    const uint8_t * wg = obj + %(h_bytes)d;
    const uint8_t * wu = wg + %(tile)d * %(col_bytes)d;
    alignas(64) float g32[32] = {0.0f};
    alignas(64) float u32[32] = {0.0f};
    for (int c = 0; c < %(tile)d; ++c) {
        g32[c] = deqdot4(wg + c * %(col_bytes)d, h);
        u32[c] = deqdot4(wu + c * %(col_bytes)d, h);
    }
    // vector-only bf16 epilogue, silu(a) = a*0.5*(1+tanh(a/2)) via
    // aie::tanh<bfloat16> (silu.cc / swiglu_mm.cc epilogue, gdn_layer.py
    // convh verbatim).  All operands stay in vectors: keeping fp32/bf16
    // staging arrays live across the aie store_v/load_v ops lets this
    // backend overlap them (u32 got clobbered), so read every operand once
    // and emit scalar lanes straight from the result float vector.
    aie::vector<float, 32> gfv = aie::load_v<32>(g32);
    aie::vector<float, 32> ufv = aie::load_v<32>(u32);
    aie::accum<accfloat, 32> ga;
    aie::accum<accfloat, 32> ua;
    ga.from_vector(gfv, 0);
    ua.from_vector(ufv, 0);
    aie::vector<bfloat16, 32> input = ga.to_vector<bfloat16>();
    aie::vector<bfloat16, 32> up_v = ua.to_vector<bfloat16>();
    const auto reg_one = aie::broadcast<bfloat16, 32>(1.0f);
    const auto reg_half16 = aie::broadcast<bfloat16, 16>(0.5f);
    const auto reg_half32 = aie::broadcast<bfloat16, 32>(0.5f);
    auto half_lo = aie::mul(input.extract<16>(0), reg_half16);
    auto half_hi = aie::mul(input.extract<16>(1), reg_half16);
    auto tanh_lo = aie::tanh<bfloat16>(half_lo.to_vector<float>());
    auto tanh_hi = aie::tanh<bfloat16>(half_hi.to_vector<float>());
    aie::vector<bfloat16, 32> tanh_half_x = aie::concat(tanh_lo, tanh_hi);
    aie::vector<bfloat16, 32> sig =
        aie::mul(aie::add(tanh_half_x, reg_one), reg_half32).to_vector<bfloat16>();
    aie::vector<bfloat16, 32> silu_v = aie::mul(input, sig).to_vector<bfloat16>();
    aie::vector<float, 32> pf = aie::mul(silu_v, up_v).to_vector<float>();
    for (int c = 0; c < %(tile)d; ++c) out[c] = pf[c];
}
""" % dict(
        blocks_k=g["blocks_k"], qb=144, sc_off=SC_OFF, ql_off=16,
        blk=BLK, h_bytes=g["k"] * 4, tile=g["tile"],
        col_bytes=g["col_bytes"])
    return src


@iron.jit
def ffn_mid(*, ncol: CompileTime[int] = 8, n_out: CompileTime[int] = N_MID,
            tile: CompileTime[int] = 14, k: CompileTime[int] = K,
            dev_name: CompileTime[str] = "npu2"):
    g = geom(ncol, n_out, tile, k)
    OBJ_T = np.ndarray[(g["obj_bytes"],), np.dtype[np.uint8]]
    OUT_T = np.ndarray[(tile,), np.dtype[np.float32]]
    # -fno-unroll-loops: a fully-unrolled 1024-el Q4_K dot x2 (gate+up) plus
    # the bf16 epilogue exceeds the 16 KB core program memory; rolling the
    # dequant-dot loops keeps the per-core text ~4 KB.
    kern = iron.ExternalFunction(
        name="ffnmid", source_string=_kernel_src(g),
        arg_types=[OBJ_T, OUT_T],
        compile_flags=["-O2", "-DNDEBUG", "-fno-unroll-loops"], inline=True)

    OBJ_g = np.ndarray[(ncol * g["nt"] * g["obj_bytes"],), np.dtype[np.uint8]]
    OUT_g = np.ndarray[(n_out,), np.dtype[np.float32]]
    workers = []
    rt_args = [OBJ_g, OUT_g]
    for col in range(ncol):
        s3 = ObjectFifo(OBJ_T, name=f"s3_{col}", depth=1)
        s2 = s3.cons().forward(obj_type=OBJ_T, name=f"s2_{col}", tile=Tile(col, 1))
        o23 = ObjectFifo(OUT_T, name=f"o23_{col}", depth=1)
        o12 = o23.prod().join([0], obj_types=[OUT_T], names=[f"o12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def core_fn(sc, oc, k, n=g["nt"]):
            for _ in range_(n):
                si = sc.acquire(1)
                so = oc.acquire(1)
                k(si, so)
                oc.release(1)
                sc.release(1)

        workers.append(Worker(core_fn, [s2.cons(), o12.prod(), kern], tile=Tile(col, 2)))
        rt_args += [s3.prod(tile=Tile(col, 0)), o23.cons(tile=Tile(col, 0))]

    def seq_fn(OBJ, OUT, *fifos):
        it = iter(fifos)
        for _col in range(ncol):
            sp = next(it)
            oc = next(it)
            for t in range(g["nt"]):
                gi = TaskGroup()
                sp.fill(OBJ, tap=TensorAccessPattern(
                    (ncol * g["nt"] * g["obj_bytes"],),
                    offset=(_col * g["nt"] + t) * g["obj_bytes"],
                    sizes=[g["obj_bytes"]], strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                oc.drain(OUT, tap=TensorAccessPattern(
                    (n_out,), offset=_col * g["cols_per_col"] + t * tile,
                    sizes=[tile], strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=ncol), rt, workers)
    return prog.resolve_program()


# object budget for the gate+up tile at input fifo depth 1 (proj_qK.py
# proved 26624 B on npu2; keep some headroom for the bf16 epilogue).
OBJ_LIMIT = 20 * 1024


def pick_tile(ncol: int, n_out: int, k: int) -> int:
    """Largest tile (<=32) that splits n_out over ncol evenly and keeps the
    per-object bytes (hff + TILE gate + TILE up columns) under OBJ_LIMIT."""
    if n_out % ncol:
        raise SystemExit(f"N={n_out} not divisible by ncol={ncol}")
    cpc = n_out // ncol
    col_bytes = (k // BLK) * 144
    for tile in range(min(32, cpc), 0, -1):
        if cpc % tile:
            continue
        if k * 4 + 2 * tile * col_bytes <= OBJ_LIMIT:
            return tile
    raise SystemExit(f"K={k}: no tile keeps object under {OBJ_LIMIT} B")


def bf16_of(x):
    """Round an fp64 array through f32 -> bf16, returned as fp64."""
    return x.astype(np.float32).astype(bfloat16).astype(np.float64)


def oracle_mid_bf16(g64, u64):
    """Reference mid via the SAME bf16-tanh-silu chain as the kernel
    (aie::mul/add round every step to bf16; tanh output is bf16).  The final
    silu*up product is kept fp32, matching the kernel's .to_vector<float>()
    store.  Residual error vs the kernel is a few bf16 ulps from the AIE tanh
    approximation (see silu gap note in the header)."""
    gb = bf16_of(g64)
    ub = bf16_of(u64)
    th = bf16_of(np.tanh(gb * 0.5))
    sig = bf16_of((th + 1.0) * 0.5)
    silu = bf16_of(gb * sig)
    return silu * ub


def main():
    ap = argparse.ArgumentParser(prog="ffn_mid")
    add_compile_args(ap)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--model", default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-q4km.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"

    reader = load_reader(opts.model)
    names = ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"]
    tensors = {n: next(t for t in reader.tensors if t.name == n) for n in names}
    for n, t in tensors.items():
        if int(t.tensor_type) != 12:
            sys.exit(f"tensor {n} type={t.tensor_type} is not Q4_K")
        if tuple(int(x) for x in t.shape) != (K, N_MID):
            sys.exit(f"tensor {n} shape={t.shape} != {(K, N_MID)}")

    def colbytes(t):
        return (K // BLK) * 144

    if opts.tile is None:
        opts.tile = pick_tile(opts.cols, N_MID, K)
    g = geom(opts.cols, N_MID, opts.tile, K)
    print(f"gate/up Q4_K K={K} N={N_MID} cols={opts.cols} tile={opts.tile} "
          f"nt={g['nt']} obj_bytes={g['obj_bytes']}")

    os.makedirs(opts.workdir, exist_ok=True)
    base = os.path.join(opts.workdir, "ffn_mid")
    spec = ffn_mid.specialize(ncol=opts.cols, n_out=N_MID, tile=opts.tile,
                              k=K, dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(
        xclbin_path=base + ".xclbin", inst_path=base + ".insts.bin")
    if not opts.run:
        print("compiled", xclbin_path)
        return

    import pyxrt as xrt
    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(ctx, xb.get_kernels()[0].get_name())
    insts = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
    insts_bo = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    np.frombuffer(insts_bo.map(), dtype=np.uint32)[:] = insts
    insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def mk_bo(arr, ro=False):
        bo = xrt.bo(dev, arr.nbytes, xrt.bo.host_only, 0)
        if not ro:
            np.frombuffer(bo.map(), dtype=np.uint8)[:] = \
                np.asarray(arr).view(np.uint8).reshape(-1)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    qbytes = {}
    for n, t in tensors.items():
        raw = np.ascontiguousarray(np.asarray(t.data).reshape(-1)).view(np.uint8)
        qbytes[n] = np.frombuffer(raw, np.uint8)
    qg = qbytes[names[0]]
    qu = qbytes[names[1]]
    cb = colbytes(tensors[names[0]])
    rng = np.random.default_rng(7)
    h = rng.standard_normal(K).astype(np.float32)
    h64 = h.astype(np.float64)

    feed = np.zeros(opts.cols * g["nt"] * g["obj_bytes"], dtype=np.uint8)
    hb = h.view(np.uint8)
    wu0 = g["k"] * 4 + opts.tile * cb
    for col in range(opts.cols):
        for tt in range(g["nt"]):
            o = feed[(col * g["nt"] + tt) * g["obj_bytes"]:]
            o[0:g["k"] * 4] = hb
            base_m = col * g["cols_per_col"] + tt * opts.tile
            for i in range(opts.tile):
                m = base_m + i
                o[g["k"] * 4 + i * cb: g["k"] * 4 + (i + 1) * cb] = \
                    qg[m * cb:(m + 1) * cb]
                o[wu0 + i * cb: wu0 + (i + 1) * cb] = qu[m * cb:(m + 1) * cb]
    out = np.zeros(N_MID, dtype=np.float32)
    feed_bo = mk_bo(feed)
    out_bo = mk_bo(out, ro=True)
    run = kernel(3, insts_bo, int(insts.nbytes), feed_bo, out_bo)
    if run.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        sys.exit("run failed")
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(out_bo.map(), dtype=np.float32)[:N_MID].copy()

    # oracle: exact fp64 dequant + dot (proj_qK.py), then bf16-tanh silu chain
    gcol = lambda n: qg[n * cb:(n + 1) * cb]
    ucol = lambda n: qu[n * cb:(n + 1) * cb]
    gq = np.array([oracle_col(dict(wtype=4, qb=144, ql_off=16, qh_off=None,
                                   blocks_k=g["blocks_k"]), gcol(n), h64)
                   for n in range(N_MID)])
    uq = np.array([oracle_col(dict(wtype=4, qb=144, ql_off=16, qh_off=None,
                                   blocks_k=g["blocks_k"]), ucol(n), h64)
                   for n in range(N_MID)])
    mid_bf16 = oracle_mid_bf16(gq, uq)
    silu_ex = lambda x: x / (1.0 + np.exp(-x))
    mid_exact = silu_ex(gq) * uq

    kern64 = out.astype(np.float64)
    err = np.abs(kern64 - mid_bf16)
    err_exact = np.abs(kern64 - mid_exact)
    print("out[:6]  =", out[:6])
    print("ref[:6]  =", mid_bf16[:6])
    print("nan count =", np.isnan(out).sum(), "/", out.size)
    print(f"ffn_mid: mid max_err (bf16-tanh oracle) = {err.max():.3e}  "
          f"median = {np.median(err):.3e}")
    print(f"ffn_mid: mid max_err (exact e^-x silu)   = {err_exact.max():.3e}  "
          f"(documented bf16-silu gap)")
    print(f"ffn_mid: |ref|max = {np.abs(mid_bf16).max():.3e}  "
          f"frac |err|>1e-3 = {np.mean(err > 1e-3):.3e}")


if __name__ == "__main__":
    main()
