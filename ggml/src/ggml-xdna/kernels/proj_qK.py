#!/usr/bin/env python3
# proj_qK.py -*- Python -*-
#
# S1 core, standalone: ggml K-family block-dequant gemv projection for ONE
# decode token, with the weight type (Q4_K / Q5_K / Q6_K) and h dim K baked
# at compile time.
#   out[n] = sum_k h[k] * W[k, n]   (n = output column, W in ggml Q4_K/Q5_K/Q6_K)
#
# Geometry is fully compile-time: N (output columns), ncol (AIE columns) and
# the per-object tile of output columns are parameters of the IRON design
# function (specialize() keys the baked kernel + FIFO sizes). Each tile object
# = [ h(K f32) | TILE-col Q-type bytes (TILE * BLOCKS_K * QB) ]; the kernel
# dequants in-core (exact ggml port) and dots with h.  K is taken from the
# tensor shape, so any multiple of 256 is supported.
#
# Q5_K block (176 B): d(2) dmin(2) scales(12) qh(32) qs(128)   qs=blk+48
# Q4_K block (144 B): d(2) dmin(2) scales(12) qs(128)          qs=blk+16
# Q6_K block (210 B): ql(128) qh(64) scales(16,int8) d(2,blk+208)
#
# Validated on real Qwen3.5 weights: blk.0.attn_qkv (Q5_K, N=6144),
# blk.0.ffn_gate (Q4_K, N=3584), blk.0.ffn_down (Q6_K, K=3584, N=1024) and
# blk.0.ssm_out (Q5_K, K=2048, N=1024).
#
# Usage:
#   python proj_qK.py -d npu2 --workdir /tmp/opencode/c1 --wtype 5 --run
#   python proj_qK.py -d npu2 --workdir /tmp/opencode/c1 --wtype 4 --run
#   python proj_qK.py -d npu2 --workdir /tmp/opencode/c1 --wtype 6 --run

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

BLK = 256         # K-family ggml block size
SC_OFF = 4        # Q4/Q5 scales offset inside a block

# wtype -> ggml type code and per-column geometry of one ggml block
QTYPE = {4: 12, 5: 13, 6: 14}
WBLK = {
    4: dict(tag="q4", qb=144, ql_off=16, qh_off=None),
    5: dict(tag="q5", qb=176, ql_off=48, qh_off=16),
    6: dict(tag="q6", qb=210, ql_off=0, qh_off=128),
}


def geom(wtype: int, ncol: int, n_out: int, tile: int, k: int):
    """Resolve every layout constant for a (wtype, ncol, n_out, tile, k) build."""
    w = WBLK[wtype]
    if k % BLK:
        raise SystemExit(f"K={k} not a multiple of BLK={BLK}")
    blocks_k = k // BLK
    if n_out % ncol:
        raise SystemExit(f"N={n_out} not divisible by ncol={ncol}")
    cols_per_col = n_out // ncol
    if cols_per_col % tile:
        raise SystemExit(f"cols/col={cols_per_col} not divisible by tile={tile}")
    col_bytes = blocks_k * w["qb"]
    return dict(
        wtype=wtype, tag=w["tag"], qb=w["qb"], ql_off=w["ql_off"],
        qh_off=w["qh_off"], blocks_k=blocks_k, cols_per_col=cols_per_col,
        nt=cols_per_col // tile, tile=tile, col_bytes=col_bytes,
        tile_qbytes=tile * col_bytes,
        obj_bytes=k * 4 + tile * col_bytes,   # h(f32) + TILE columns of blocks
        ncol=ncol, n_out=n_out, k=k,
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


def _kernel_src_q45(g):
    """C for a Q4_K / Q5_K kernel. projq4 / projq5 are emitted from one
    template; the qh (Q5 high-bit) pieces are dropped for Q4."""
    if g["qh_off"] is not None:
        qh_decl = "            const uint8_t * qh = blk + %d;\n" % g["qh_off"]
        u_init = "            uint8_t u1 = 1, u2 = 2;\n"
        u_adv = " u1 <<= 2; u2 <<= 2;"
        v_lo = "d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1"
        v_hi = "d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2"
    else:
        qh_decl = ""
        u_init = ""
        u_adv = ""
        v_lo = "d1 * (ql[l] & 0xF) - m1"
        v_hi = "d2 * (ql[l] >> 4) - m2"

    src = """#include <stdint.h>
#include <string.h>
""" + _FP16F_SRC + """static void gsm4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;} else {
        *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); } }
extern "C" void proj%(tag)s(const uint8_t * obj, float * out) {
    const float * h = (const float *)(obj + 0);
    const uint8_t * wb = obj + %(w_off)d;
    for (int c = 0; c < %(tile)d; ++c) {
        const uint8_t * col = wb + c * %(col_bytes)d;
        float acc = 0.0f;
        int el = 0;
        for (int b = 0; b < %(blocks_k)d; ++b) {
            const uint8_t * blk = col + b * %(qb)d;
            const float d = fp16f((uint16_t)(blk[0] | (blk[1] << 8)));
            const float mn = fp16f((uint16_t)(blk[2] | (blk[3] << 8)));
            const uint8_t * sc = blk + %(sc_off)d;
%(qh_decl)s            const uint8_t * ql = blk + %(ql_off)d;
%(u_init)s            int is = 0;
            for (int j2 = 0; j2 < %(blk)d; j2 += 64) {
                uint8_t s0, m0; gsm4(is + 0, sc, &s0, &m0);
                const float d1 = d * s0, m1 = mn * m0;
                uint8_t s1, m1b; gsm4(is + 1, sc, &s1, &m1b);
                const float d2 = d * s1, m2 = mn * m1b;
                for (int l = 0; l < 32; ++l) {
                    float v = %(v_lo)s;
                    acc += v * h[el++];
                }
                for (int l = 0; l < 32; ++l) {
                    float v = %(v_hi)s;
                    acc += v * h[el++];
                }
                ql += 32; is += 2;%(u_adv)s
            }
        }
        out[c] = acc;
    }
}
""" % dict(
        tag=g["tag"], w_off=g["k"] * 4, tile=g["tile"], col_bytes=g["col_bytes"],
        blocks_k=g["blocks_k"], qb=g["qb"], sc_off=SC_OFF, ql_off=g["ql_off"],
        blk=BLK, qh_decl=qh_decl, u_init=u_init, u_adv=u_adv, v_lo=v_lo,
        v_hi=v_hi)
    return src


def _kernel_src_q6(g):
    """C for a Q6_K kernel. 256-el ggml block = ql(128) qh(64) sc(16 int8)
    d(fp16 at +208). Exact port of dequantize_row_q6_K (ggml-quants.c): two
    halves of 128 elements, each producing four 32-el lanes (q1..q4) which
    are accumulated against h in element order."""
    return """#include <stdint.h>
#include <string.h>
""" + _FP16F_SRC + """extern "C" void projq6(const uint8_t * obj, float * out) {
    const float * h = (const float *)(obj + 0);
    const uint8_t * wb = obj + %(w_off)d;
    for (int c = 0; c < %(tile)d; ++c) {
        const uint8_t * col = wb + c * %(col_bytes)d;
        float acc = 0.0f;
        int el = 0;
        for (int b = 0; b < %(blocks_k)d; ++b) {
            const uint8_t * blk = col + b * %(qb)d;
            const float d = fp16f((uint16_t)(blk[%(d_off)d] | (blk[%(d_off)d + 1] << 8)));
            const uint8_t * ql = blk + %(ql_off)d;
            const uint8_t * qh = blk + %(qh_off)d;
            const int8_t  * sc = (const int8_t *)(blk + %(sc_off)d);
            for (int hf = 0; hf < 2; ++hf) {
                for (int l = 0; l < 32; ++l) {
                    int is = l >> 4;
                    int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    acc += (d * sc[is + 0]) * q1 * h[el++];
                }
                for (int l = 0; l < 32; ++l) {
                    int is = l >> 4;
                    int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    acc += (d * sc[is + 2]) * q2 * h[el++];
                }
                for (int l = 0; l < 32; ++l) {
                    int is = l >> 4;
                    int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    acc += (d * sc[is + 4]) * q3 * h[el++];
                }
                for (int l = 0; l < 32; ++l) {
                    int is = l >> 4;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += (d * sc[is + 6]) * q4 * h[el++];
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
        out[c] = acc;
    }
}
""" % dict(
        tag=g["tag"], w_off=g["k"] * 4, tile=g["tile"], col_bytes=g["col_bytes"],
        blocks_k=g["blocks_k"], qb=g["qb"], sc_off=192, ql_off=g["ql_off"],
        qh_off=g["qh_off"], d_off=208)
    return src


def _kernel_src(g):
    return _kernel_src_q45(g) if g["wtype"] <= 5 else _kernel_src_q6(g)


@iron.jit
def proj_qK(*, ncol: CompileTime[int] = 8, wtype: CompileTime[int] = 5,
            n_out: CompileTime[int] = 6144, tile: CompileTime[int] = 32,
            k: CompileTime[int] = 1024, dev_name: CompileTime[str] = "npu2"):
    g = geom(wtype, ncol, n_out, tile, k)
    OBJ_T = np.ndarray[(g["obj_bytes"],), np.dtype[np.uint8]]
    OUT_T = np.ndarray[(tile,), np.dtype[np.float32]]
    kern = iron.ExternalFunction(
        name="proj" + g["tag"], source_string=_kernel_src(g),
        arg_types=[OBJ_T, OUT_T],
        compile_flags=["-O2", "-DNDEBUG"], inline=True)

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


def fp16(u):
    s = (u >> 15) & 1; e = (u >> 10) & 0x1f; m = u & 0x3ff
    if e == 0:
        return 0.0 if m == 0 else float((-1) ** s) * 2.0 ** -24 * m
    return float((-1) ** s) * 2.0 ** (e - 15) * (1 + m / 1024.0)


def gsm(sc, j):
    if j < 4:
        return sc[j] & 63, sc[j + 4] & 63
    return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))


def oracle_col_q6(g, chunk, h64):
    """Exact numpy dequant of one Q6_K output column, fp64. Port of
    dequantize_row_q6_K: per 256-block two 128-halves x four 32-el lanes."""
    nblk = g["blocks_k"]
    q = np.empty(nblk * BLK, dtype=np.float64)
    for b in range(nblk):
        blk = chunk[b * g["qb"]:(b + 1) * g["qb"]]
        d = fp16(int.from_bytes(blk[208:210], 'little'))
        ql = blk[0:128].astype(np.int64)
        qh = blk[128:192].astype(np.int64)
        sc = blk[192:208].astype(np.int8).astype(np.int64)
        for hf in range(2):
            qlo = ql[hf * 64:]
            qho = qh[hf * 32:]
            sco = sc[hf * 8:]
            for l in range(32):
                is_ = l >> 4
                q1 = ((qlo[l] & 0xF) | (((qho[l] >> 0) & 3) << 4)) - 32
                q2 = ((qlo[l + 32] & 0xF) | (((qho[l] >> 2) & 3) << 4)) - 32
                q3 = ((qlo[l] >> 4) | (((qho[l] >> 4) & 3) << 4)) - 32
                q4 = ((qlo[l + 32] >> 4) | (((qho[l] >> 6) & 3) << 4)) - 32
                base = b * BLK + hf * 128
                q[base + l] = d * sco[is_ + 0] * q1
                q[base + l + 32] = d * sco[is_ + 2] * q2
                q[base + l + 64] = d * sco[is_ + 4] * q3
                q[base + l + 96] = d * sco[is_ + 6] * q4
    return q @ h64


def oracle_col(g, chunk, h64):
    """Exact numpy dequant of one output column (g['blocks_k'] blocks), fp64."""
    if g["wtype"] == 6:
        return oracle_col_q6(g, chunk, h64)
    nblk = g["blocks_k"]
    q = np.empty(nblk * BLK, dtype=np.float64)
    qh_off = g["qh_off"]
    for b in range(nblk):
        blk = chunk[b * g["qb"]:(b + 1) * g["qb"]]
        d = fp16(int.from_bytes(blk[0:2], 'little'))
        mn = fp16(int.from_bytes(blk[2:4], 'little'))
        sc = blk[SC_OFF:SC_OFF + 12].astype(np.int64)
        ql = blk[g["ql_off"]:g["ql_off"] + 128].astype(np.int64)
        qh = blk[qh_off:qh_off + 32] if qh_off is not None else None
        for gi in range(BLK // 64):
            a, b0 = gsm(sc, 2 * gi + 0); d1 = d * a; m1 = mn * b0
            c, e = gsm(sc, 2 * gi + 1); d2 = d * c; m2 = mn * e
            lane = ql[gi * 32:(gi + 1) * 32]
            lo = lane & 0xF
            hi = lane >> 4
            if qh is not None:
                hb = qh.astype(np.int64)   # 32 bytes, same window each group
                lo = lo + 16 * ((hb >> (2 * gi)) & 1)
                hi = hi + 16 * ((hb >> (2 * gi + 1)) & 1)
            v = np.empty(64, dtype=np.float64)
            v[:32] = d1 * lo - m1
            v[32:] = d2 * hi - m2
            off = b * BLK + gi * 64
            q[off:off + 64] = v
    return q @ h64


def load_reader(model):
    repo = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(repo, "gguf-py")):
        repo = os.path.dirname(repo)
    sys.path.insert(0, os.path.join(repo, "gguf-py"))
    from gguf.gguf_reader import GGUFReader
    return GGUFReader(model)


# Largest per-tile object ever proven to fit the AIE L1 at input fifo depth 1
# (Q5_K K=1024 tile=32 => 26624 B). Auto tile keeps new builds under it.
OBJ_LIMIT = 26 * 1024


def pick_tile(wtype: int, ncol: int, n_out: int, k: int) -> int:
    """Largest tile (<=32) that splits n_out over ncol evenly and keeps the
    per-object bytes (h f32 + TILE cols of blocks) under OBJ_LIMIT."""
    w = WBLK[wtype]
    if n_out % ncol:
        raise SystemExit(f"N={n_out} not divisible by ncol={ncol}")
    cpc = n_out // ncol
    blk_bytes = (k // BLK) * w["qb"]
    for tile in range(min(32, cpc), 0, -1):
        if cpc % tile:
            continue
        if k * 4 + tile * blk_bytes <= OBJ_LIMIT:
            return tile
    raise SystemExit(f"K={k}: no tile keeps object under {OBJ_LIMIT} B; "
                     f"h alone is {k * 4} B - a separate h fifo is needed")


def main():
    ap = argparse.ArgumentParser(prog="proj_qK")
    add_compile_args(ap)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--wtype", type=int, default=5, choices=(4, 5, 6))
    ap.add_argument("--wten", default=None,
                    help="gguf tensor to project (default by --wtype)")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--model", default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-q4km.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.wten is None:
        opts.wten = {5: "blk.0.attn_qkv.weight",
                     4: "blk.0.ffn_gate.weight",
                     6: "blk.0.ffn_down.weight"}[opts.wtype]

    reader = load_reader(opts.model)
    t = next(tt for tt in reader.tensors if tt.name == opts.wten)
    k = int(t.shape[0])
    if int(t.tensor_type) != QTYPE[opts.wtype]:
        sys.exit(f"tensor {opts.wten} type={t.tensor_type} "
                 f"does not match --wtype {opts.wtype} (Q{opts.wtype}_K)")
    n_out = int(t.shape[1])
    if opts.tile is None:
        opts.tile = pick_tile(opts.wtype, opts.cols, n_out, k)
    g = geom(opts.wtype, opts.cols, n_out, opts.tile, k)
    print(f"wtype=Q{opts.wtype}_K tensor={opts.wten} K={k} N={n_out} "
          f"cols={opts.cols} tile={opts.tile} nt={g['nt']} "
          f"obj_bytes={g['obj_bytes']}")

    os.makedirs(opts.workdir, exist_ok=True)
    base = os.path.join(opts.workdir, f"proj_qK_q{opts.wtype}")
    spec = proj_qK.specialize(ncol=opts.cols, wtype=opts.wtype, n_out=n_out,
                              tile=opts.tile, k=k, dev_name=opts.dev)
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

    raw = np.ascontiguousarray(np.asarray(t.data).reshape(-1)).view(np.uint8)
    qbytes = np.frombuffer(raw, np.uint8)
    rng = np.random.default_rng(7)
    h = rng.standard_normal(k).astype(np.float32)
    h64 = h.astype(np.float64)

    # build feed: per (col,tile): [h f32 | TILE cols of blocks]
    feed = np.zeros(opts.cols * g["nt"] * g["obj_bytes"], dtype=np.uint8)
    for col in range(opts.cols):
        for tt in range(g["nt"]):
            o = feed[(col * g["nt"] + tt) * g["obj_bytes"]:]
            o[0:k * 4] = h.view(np.uint8)
            wf = k * 4
            base_n = col * g["cols_per_col"] + tt * opts.tile
            for i in range(opts.tile):
                n = base_n + i
                o[wf + i * g["col_bytes"]: wf + (i + 1) * g["col_bytes"]] = \
                    qbytes[n * g["col_bytes"]:(n + 1) * g["col_bytes"]]
    out = np.zeros(n_out, dtype=np.float32)
    feed_bo = mk_bo(feed)
    out_bo = mk_bo(out, ro=True)
    run = kernel(3, insts_bo, int(insts.nbytes), feed_bo, out_bo)
    if run.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        sys.exit("run failed")
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(out_bo.map(), dtype=np.float32)[:n_out].copy()

    ref = np.array([oracle_col(g, qbytes[n * g["col_bytes"]:(n + 1) * g["col_bytes"]], h64)
                    for n in range(n_out)])
    print("ref[:6] =", ref[:6])
    print("out[:6] =", out[:6])
    print("nan count =", np.isnan(out).sum(), "/", out.size)
    err = np.abs(np.nan_to_num(out.astype(np.float64), nan=1e30) - ref).max()
    rel = err / max(np.abs(ref).max(), 1e-12)
    print(f"proj_qK: out max_err={err:.3e} rel={rel:.3e} "
          f"|ref|max={np.abs(ref).max():.3e}")


if __name__ == "__main__":
    main()
