#!/usr/bin/env python3
# attn_so_mmul.py -*- Python -*-
#
# ssm_out on the native int8 x int4 MMUL, in the ffn_layer_mmul.py "stage A"
# shape: ONE 64x64 grid tile per (col, group, slab) feed object, single output
# matrix (no gate/up dual), raw int32 acc drained per group, host rescale.
#
#   gated[2048] (host-computed attn*rms*gamma*silu(z)) -> int8 codes (d_a)
#      -> per (col 0..7, group of 64 out cols, k-slab of 64 rows) object:
#           [ A int8 padded slab 256 B | W int4 tile 2048 B ]  (OBJ 2304 B)
#      -> worker accumulates a per-(col,group) int32 acc over the 32 slabs
#         and drains it raw to DDR; host rescales acc*d_a*d_w[n] and adds the
#         attn residual -> h_attn (the attn_layer scalar ssm_out replacement).
#
# W = Q5_K blk.0.ssm_out [K=2048, N=1024], dequantized exactly once on the
# host (dequantize_row_q5_K semantics, proj_qK.py oracle) and re-quantized into
# a plain int4 grid (0.5 B/val) with a per-output-column scale d_w = amax/7,
# exactly like xdna-fln-mmul / ffn_layer_mmul.  Max-rel on out vs the exact
# scalar Q5_K oracle ~0.16 (measured on the real tensor + real gated vector),
# inside the FFN w4a8 band (0.12-0.2); per-64-slab scales are worse (rel 1.05).
#
# Layout constants mirror ffn_layer_mmul.py / gemm_w4a8.py:
#   object B tile: nibble (kz,nt,kr,nc) -> byte (kz*4+nt)*128 + kr*8 + nc/2
#     low nibble first; tile covers k rows [t*64, +64) x n cols [n0, +64).
#   object A slab: real element k (0..63) of the int8 gated row at byte
#     (k>>4)*64 + (k&15) of the 256-B region (the "16k per kz row" layout the
#     ffn stage-A/bmac A tiles consume; rows beyond col 15 stay zero).
#   acc: per (col, grp) [256] int32 raw, output col n at
#     raw[(c>>4)*64 + (c&15)] (gather_out), then * d_a * d_w[n].
#
# Usage:
#   python attn_so_mmul.py -d npu2 --workdir build/bin --compile   (xclbin only)
#   python attn_so_mmul.py --workdir /tmp/opencode/so1 --run
#   python attn_so_mmul.py --workdir /tmp/opencode/so1            (host-only)
from __future__ import annotations

import argparse
import os
import sys
import time
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
import ffn_layer_mmul as ffm  # noqa: E402
from proj_qK import oracle_col, BLK, load_reader  # noqa: E402

K_SO = 2048               # ssm_out row dim = gated vector length (16 heads*128)
N_SO = 1024               # ssm_out col dim = D_OUT
GRP = 64                  # output columns per group
NCOL = 8                  # AIE columns
CPC = N_SO // NCOL        # 128 out cols per column
NG = CPC // GRP           # 2 groups per column
SB = K_SO // 64           # 32 k-slabs per group

A_SZ = 256                # padded int8 slab (4 x 64)
B_SZ = 2048               # int4 tile bytes (64k x 64n)
OBJ = A_SZ + B_SZ         # 2304 B per feed object
NT = NG * SB              # objects per column

RES = dict(ncol=NCOL, cpc=CPC, ng=NG, sb=SB, nt=NT, obj=OBJ,
           acc_bytes=NG * 256 * 4, feed_bytes=NCOL * NT * OBJ)


_HEAD = """#include <stdint.h>
#include <aie_api/aie.hpp>
using namespace aie;
typedef aie::mmul<4, 16, 16, int8, int4> MMUL;
"""


def _zero_src():
    return _HEAD + """
extern "C" void sozero(int32_t * acc) {
    for (int i = 0; i < 256; i++) acc[i] = 0;
}
"""


def _mac_src():
    # A slab int8 at obj+0 (256 B padded), W int4 tile at obj+256.  Mirror of
    # ffn_layer_mmul._amac_src's single (gate) grid; acc is the per-group
    # [4 nt x 64] int32 held across the SB slabs of the group.
    return _HEAD + """
extern "C" void somac(const uint8_t * obj, int32_t * acc) {
    const int8_t * A = (const int8_t *)(obj + 0);
    const uint8_t * W = obj + %(B_OFF)d;
    for (int nt = 0; nt < 4; ++nt) {
        MMUL C0(aie::load_v<64>(acc + nt * 64));
        for (int kz = 0; kz < 4; ++kz) {
            auto A0 = aie::load_v<64>(A + kz * 64);
            const uint8_t * bs = W + (kz * 4 + nt) * 128;
            auto B0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(bs));
            C0.mac(A0, B0);
        }
        aie::store_v(acc + nt * 64, C0.template to_vector<int32_t>());
    }
}
""" % dict(B_OFF=A_SZ)


@iron.jit
def attn_so_mmul(*, ncol: CompileTime[int] = NCOL,
                 dev_name: CompileTime[str] = "npu2"):
    FEED_T = np.ndarray[(OBJ,), np.dtype[np.uint8]]
    ACC_T = np.ndarray[(256,), np.dtype[np.int32]]
    zero_k = iron.ExternalFunction(name="sozero", source_string=_zero_src(),
                                   arg_types=[ACC_T], compile_flags=["-O2", "-DNDEBUG"],
                                   inline=True)
    mac_k = iron.ExternalFunction(name="somac", source_string=_mac_src(),
                                  arg_types=[FEED_T, ACC_T],
                                  compile_flags=["-O2", "-DNDEBUG"], inline=True)

    FEED_g = np.ndarray[(ncol * NT * OBJ,), np.dtype[np.uint8]]
    ACC_g = np.ndarray[(ncol * NG * 256,), np.dtype[np.int32]]

    workers = []
    rt_args = [FEED_g, ACC_g]
    for col in range(ncol):
        f3 = ObjectFifo(FEED_T, name=f"sf3_{col}", depth=2)
        f2 = f3.cons().forward(obj_type=FEED_T, name=f"sf2_{col}", tile=Tile(col, 1))
        a23 = ObjectFifo(ACC_T, name=f"sa23_{col}", depth=1)
        a12 = a23.prod().join([0], obj_types=[ACC_T], names=[f"sa12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def so_core(fc, acp, kz_, km, ng=NG, ns=SB):
            for _ in range_(ng):
                ac = acp.acquire(1)
                kz_(ac)
                for _ in range_(ns):
                    oi = fc.acquire(1)
                    km(oi, ac)
                    fc.release(1)
                acp.release(1)

        workers.append(Worker(so_core, [f2.cons(), a12.prod(), zero_k, mac_k],
                              tile=Tile(col, 2), stack_size=0x3000))
        rt_args += [f3.prod(tile=Tile(col, 0)),
                    a23.cons(tile=Tile(col, 0))]

    def seq_fn(FEED, ACC, *fifos):
        it = iter(fifos)
        fill = []; ad = []
        for _c in range(ncol):
            fill.append(next(it)); ad.append(next(it))
        for c in range(ncol):
            tg = TaskGroup()
            fill[c].fill(FEED, tap=TensorAccessPattern(
                (ncol * NT * OBJ,), offset=c * NT * OBJ,
                sizes=[NT * OBJ], strides=[1]), group=tg)
            tg.finish()
        for grp in range(NG):
            for c in range(ncol):
                go = TaskGroup()
                ad[c].drain(ACC, tap=TensorAccessPattern(
                    (ncol * NG * 256,), offset=(c * NG + grp) * 256,
                    sizes=[256], strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=ncol), rt, workers)
    return prog.resolve_program()


# ---------------------------------------------------------------------------
# host pack + numpy integer model (mirrors xdna-fln-mmul / ffn_layer_mmul)
# ---------------------------------------------------------------------------

def pack_a_slab(aq, t, buf, off):
    """int8 gated slab t (64 vals) into the padded A region at buf[off:+256]."""
    for k in range(64):
        buf[off + (k >> 4) * 64 + (k & 15)] = aq[t * 64 + k]


def build_feed(codes_w, dw, aq):
    """Full per-token feed: resident W int4 tiles + per-token int8 A slabs.
    codes_w: column-major int8 int4 codes [K*N]; aq: int8 gated codes [K]."""
    codes2 = codes_w.reshape(K_SO, N_SO, order="F")
    feed = np.zeros(NCOL * NT * OBJ, dtype=np.uint8)
    for c in range(NCOL):
        for grp in range(NG):
            n0 = c * CPC + grp * GRP
            for t in range(SB):
                o = feed[(c * NT + grp * SB + t) * OBJ:]
                ffm.pack_tile4(codes2, t * 64, n0, o, A_SZ)
                pack_a_slab(aq, t, o, 0)
    return feed


def requant_W(Wd):
    """per-column int4 requant of the dequantized [K x N] fp64 weight matrix."""
    codes = np.zeros((K_SO * N_SO,), dtype=np.int8)
    dw = np.empty(N_SO, dtype=np.float32)
    for n in range(N_SO):
        amax = np.abs(Wd[:, n]).max()
        d = amax / 7.0 if amax > 0 else 1.0
        q = np.sign(Wd[:, n] / d) * np.floor(np.abs(Wd[:, n] / d) + 0.5)
        codes[n * K_SO:(n + 1) * K_SO] = np.clip(q, -8, 7).astype(np.int8)
        dw[n] = np.float32(d)
    return codes, dw


def dequant_so(qbytes, cb):
    """Exact fp64 Q5_K dequant of blk.0.ssm_out [K=2048, N=1024]."""
    W = np.empty((K_SO, N_SO), dtype=np.float64)
    for n in range(N_SO):
        chunk = qbytes[n * cb:(n + 1) * cb]
        col = np.empty(K_SO, dtype=np.float64)
        for b in range(K_SO // BLK):
            blk = chunk[b * 176:(b + 1) * 176]
            d = proj_fp16(int.from_bytes(blk[0:2], 'little'))
            mn = proj_fp16(int.from_bytes(blk[2:4], 'little'))
            sc = blk[4:16].astype(np.int64)
            ql = blk[48:176].astype(np.int64)
            qh = blk[16:48].astype(np.int64)
            for gi in range(4):
                a_, b_ = gsm_ok(sc, 2 * gi + 0); d1 = d * a_; m1 = mn * b_
                c_, e_ = gsm_ok(sc, 2 * gi + 1); d2 = d * c_; m2 = mn * e_
                lane = ql[gi * 32:(gi + 1) * 32]
                lo = lane & 0xF
                hi = lane >> 4
                hb = qh
                lo = lo + 16 * ((hb >> (2 * gi)) & 1)
                hi = hi + 16 * ((hb >> (2 * gi + 1)) & 1)
                v = np.empty(64, dtype=np.float64)
                v[:32] = d1 * lo - m1
                v[32:] = d2 * hi - m2
                col[b * BLK + gi * 64:b * BLK + gi * 64 + 64] = v
        W[:, n] = col
    return W


def proj_fp16(u):
    s = (u >> 15) & 1; e = (u >> 10) & 0x1f; m = u & 0x3ff
    if e == 0:
        return 0.0 if m == 0 else float((-1) ** s) * 2.0 ** -24 * m
    return float((-1) ** s) * 2.0 ** (e - 15) * (1 + m / 1024.0)


def gsm_ok(sc, j):
    if j < 4:
        return sc[j] & 63, sc[j + 4] & 63
    return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))


def int8_act(g):
    """host quant of the gated vector: round-half-away, d_a = amax/127."""
    amax = np.abs(g).max()
    d = amax / 127.0 if amax > 0 else 1.0
    q = np.sign(g / d) * np.floor(np.abs(g / d) + 0.5)
    return np.clip(q, -128, 127).astype(np.int8), np.float32(d)


def gather_out(raw, d_a, dw):
    """Host rescale of the drained raw int32: out[n] = acc*d_a*dw[n]."""
    out = np.zeros(N_SO, dtype=np.float64)
    codes = raw.reshape(NCOL, NG, 256)
    for c in range(NCOL):
        for grp in range(NG):
            blk = codes[c, grp]
            for cc in range(GRP):
                n = c * CPC + grp * GRP + cc
                acc = int(blk[(cc >> 4) * 64 + (cc & 15)])
                out[n] = acc * float(d_a) * float(dw[n])
    return out


def main():
    ap = argparse.ArgumentParser(prog="attn_so_mmul")
    add_compile_args(ap)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--model",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-bf16.gguf")
    ap.add_argument("--qmodel",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-q4km.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"

    if opts.compile:
        os.makedirs(opts.workdir, exist_ok=True)
        base = os.path.join(opts.workdir, "attn_so_mmul")
        spec = attn_so_mmul.specialize(ncol=NCOL, dev_name=opts.dev)
        xclbin_path, insts_path = spec.compile(xclbin_path=base + ".xclbin",
                                               inst_path=base + ".insts.bin")
        print("compiled", xclbin_path)
        import design_tag
        design_tag.stamp(xclbin_path, insts_path, opts.dev or "")
        return

    # --- real weights + a real gated activation ---------------------------
    qreader = load_reader(opts.qmodel)
    so_qb = np.ascontiguousarray(np.asarray(
        next(t for t in qreader.tensors if t.name == "blk.0.ssm_out.weight").data
    ).reshape(-1)).view(np.uint8)
    if int(next(t for t in qreader.tensors
               if t.name == "blk.0.ssm_out.weight").tensor_type) != 13:
        sys.exit("ssm_out is not Q5_K")
    cb = (K_SO // BLK) * 176
    Wd = dequant_so(so_qb, cb)
    codes_w, dw = requant_W(Wd)

    reader = load_reader(opts.model)
    import ref_delta_layer as rdl
    from check_real import load_embed
    W = rdl.load_layer(reader, 0)
    h = load_embed(reader, 5).astype(np.float32)
    conv_state = np.zeros((rdl.DCONV - 1, rdl.QKVD), dtype=np.float32)
    S = np.zeros((rdl.N_VH, rdl.S_V, rdl.S_V), dtype=np.float32)

    def gated_of(h, cs, S):
        cur = rdl.rms_norm(h, W["attn_norm"])
        qkv = cur @ W["wqkv"]
        z = cur @ W["z_gate"]
        alpha = cur @ W["alpha"]
        beta_raw = cur @ W["beta"]
        gate = rdl.softplus(alpha + W["dt"]) * W["a"]
        beta = 1.0 / (1.0 + np.exp(-beta_raw))
        conv_in = np.concatenate([cs, qkv[None, :]], axis=0)
        cs = conv_in[-(rdl.DCONV - 1):].copy()
        x = rdl.silu(rdl.ssm_conv(conv_in, W["conv"]))
        q_flat = x[0:2048].reshape(16, 128)
        k_flat = x[2048:4096].reshape(16, 128)
        v_flat = x[4096:6144].reshape(16, 128)
        q = np.stack([rdl.l2_norm_vec(q_flat[hh]) for hh in range(16)])
        k = np.stack([rdl.l2_norm_vec(k_flat[hh]) for hh in range(16)])
        attn, S = rdl.gdn_step(q, k, v_flat, gate, beta, S)
        zz = z.reshape(16, 128)
        attn_n = np.stack([rdl.rms_norm(attn[hh], W["ssm_norm"])
                           for hh in range(16)])
        return (attn_n * rdl.silu(zz)).reshape(K_SO).astype(np.float64), cs, S

    g64 = None
    for step in range(4):
        g64, conv_state, S = gated_of(h, conv_state, S)
        if step < 3:
            h = h + (np.random.default_rng(step).standard_normal(rdl.D) * 0.1).astype(np.float32)

    # exact scalar Q5_K oracle (what attn_layer computes today)
    g5 = dict(wtype=5, qb=176, ql_off=48, qh_off=16, blocks_k=K_SO // BLK)
    out_oracle = np.array([oracle_col(g5, so_qb[n * cb:(n + 1) * cb], g64)
                           for n in range(N_SO)])
    aq, da = int8_act(g64)
    # numpy integer model of the device (row0 int8 x int4, then rescale)
    codes2 = codes_w.reshape(K_SO, N_SO, order="F")
    out_model = ((aq.astype(np.int64) @ codes2.astype(np.int64)).astype(np.float64)
                 * float(da) * dw.astype(np.float64))
    e = np.abs(out_model - out_oracle)
    print(f"[host] gated |max|={np.abs(g64).max():.3e} rms="
          f"{np.sqrt(np.mean(g64 * g64)):.3e} d_a={da:.3e}")
    print(f"[host] int4 model out vs scalar oracle: max_abs={e.max():.3e} "
          f"rel={e.max() / np.abs(out_oracle).max():.4f}")

    if not opts.run:
        print("host-only ok (no NPU run)")
        return

    import pyxrt as xrt
    os.makedirs(opts.workdir, exist_ok=True)
    base = os.path.join(opts.workdir, "attn_so_mmul")
    spec = attn_so_mmul.specialize(ncol=NCOL, dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(xclbin_path=base + ".xclbin",
                                           inst_path=base + ".insts.bin")
    print("compiled", xclbin_path)

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
                np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    feed = build_feed(codes_w, dw, aq)
    feed_bo = mk_bo(feed)
    acc_bo = mk_bo(np.zeros(NCOL * NG * 256, np.int32), ro=True)

    def once():
        r = kernel(3, insts_bo, int(insts.nbytes), feed_bo, acc_bo)
        return r

    r = once()
    if r.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        sys.exit("run failed")
    raw = np.frombuffer(acc_bo.map(), dtype=np.int32)[:NCOL * NG * 256].copy()
    out = gather_out(raw, da, dw)
    e_dev = np.abs(out - out_oracle)
    e_mod = np.abs(out - out_model)
    print(f"[npu ] out vs scalar oracle: max_abs={e_dev.max():.3e} "
          f"rel={e_dev.max() / np.abs(out_oracle).max():.4f}")
    print(f"[npu ] out vs int4 model : max_abs={e_mod.max():.3e}")

    for _ in range(max(opts.warmup, 0)):
        rr = once(); rr.wait()
    ts = []
    for _ in range(max(opts.iters, 1)):
        t1 = time.perf_counter()
        rr = once(); rr.wait()
        ts.append(time.perf_counter() - t1)
    print(f"[npu ] attn_so_mmul: min={min(ts)*1e3:.2f}ms "
          f"mean={np.mean(ts)*1e3:.2f}ms  (N={len(ts)})")


if __name__ == "__main__":
    main()
