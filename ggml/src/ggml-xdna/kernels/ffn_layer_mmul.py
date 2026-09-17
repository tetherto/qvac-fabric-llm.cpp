#!/usr/bin/env python3
# ffn_layer_mmul.py -*- Python -*-
#
# Ph1: the fused decode FFN for ONE token vectorized on the native AIE2P
# int8 x int4 MMUL (w4a8 style), replacing the scalar ExternalFunction gemv of
# ffn_layer.py.  Weights are dequantized once on the host with the exact
# ggml dequantize_row_q4_K / q6_K tables (dequant_col_vec), re-quantized into
# a plain int4 grid (0.5 B/val, per-output-column scale d_w) and streamed per
# token from DDR.  Activations are int8 (per-row scale d_a).  bf16 weights are
# NOT used anywhere (byte/val budget).
#
#   hff[1024] int8
#      -> stage A (cols 0..n_a-1): mmul gate/up int4 grids + in-core bf16-tanh
#         silu = mid f32[3584] into DDR mid_bo  (silu port: ffn_mid verbatim)
#      -> stage B (cols n_a..7): each column re-quantizes mid to int8 in-core
#         (one global d_aB) and mmuls it against the down int4 grid -> raw
#         int32 [4][64] per 64-col group into DDR; the host rescales by
#         d_aB*d_w and adds residual (xdna-ops.cpp w4a8_finalize pattern).
#
# Tile geometry (64 output cols per group; K chunked into 64-row slabs):
#   stage A: group = 64 mid cols over K=1024 = 16 slab objects of 64 k-rows;
#     object = [512-B scale header sg|su | A int8 4x64 padded | gate int4 |
#               up int4] = 4864 B.  gate/up int32 acc is held in one object
#     across the group (azero/amac), the last slab is held so the epilogue
#     (afin) reads the scale header and emits silu(g)*u = mid f32.
#   stage B: group = 64 out cols over K=3584 = 56 slab objects of
#     [down int4 2048 | u16 slab tag] = 2056 B; A tiles are read from the held,
#     in-place-quantized mid codes object (bquant writes padded codes).
#
# MMUL B layout follows gemm_w4a8.py: element 64k x 64n, nibble (kz,nt,kr,nc)
# at (kz*4+nt)*128 + kr*8 + nc/2, low nibble first.  A tile = 64 int8 =
# [row0 16k | rows1-3 zero] per kz at kz*64 within the slab region.
#
# Usage:
#   python ffn_layer_mmul.py -d npu2 --workdir build/bin --compile   (xclbin only)
#   python ffn_layer_mmul.py -d npu2 --workdir /tmp/opencode/f --run   (fused)
#   python ffn_layer_mmul.py -d npu2 --stage a --workdir /tmp/opencode/f --run
#   python ffn_layer_mmul.py --workdir /tmp/opencode/f                 (host-only)
#
# Accuracy (scheme (a), w4a8-style, chosen over bf16 per the Ph1 brief):
# re-quantizing the dequantized Q4_K/Q6_K columns into a single per-column
# int4 scale costs ~0.13-0.22 max-rel on mid/out vs the exact scalar Q4_K/Q6_K
# oracle (measured host-side on blk.0; identical to the xdna-ops.cpp w4a8 path
# the backend ships, whose measured max-rel is 0.20).  At the h_out (residual)
# level the deviation is ~1e-2 abs, inside the persist-harness band.  The
# device kernel matches the integer w4a8 model to ~1e-2 (aie tanh gap only).
#
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffn_mid import _kernel_src as _mid_src, geom as _mid_geom  # noqa: E402
from ffn_mid import oracle_mid_bf16  # noqa: E402
import proj_qK  # noqa: E402
from proj_qK import oracle_col, BLK, load_reader  # noqa: E402

K = 1024                 # hidden (gate/up row) dim = hff width
K_DOWN = 3584            # down row dim = mid length
N_MID = 3584
N_OUT = 1024

# stage A element layout (bytes); the 512-B scale header rides in every
# feed object (sg[64] su[64] f32), read by the epilogue from the held object
A_HDR = 512
A_AOF = 512              # A int8 4x64 padded (256)
A_BGO = 768              # gate int4 2048
A_BUO = 2816             # up int4 2048
A_OBJ = 4864
# stage B element layout
B_TAG = 2048             # u16 slab tag after the 2048 B int4 bytes
B_OBJ = 2056

GRP_A = 64               # stage A group cols (one mid-drain / one acc)
GRP_B = 64               # stage B group cols
S_A = K // 64            # stage A slabs per group (16)
S_B = K_DOWN // 64       # stage B slabs per group (56)


def resolve(n_a: int, n_b: int):
    """Column-split layout for the fused design.  Every output-dim chunk is
    whole groups of GRP (64) cols, so per column:
      stage A owns N_MID/n_a cols  -> n_gA = N_MID/n_a/GRP groups of 16 slabs
      stage B owns N_OUT/n_b cols  -> n_gB = N_OUT/n_b/GRP groups of 56 slabs
    """
    if n_a + n_b != 8 and not (n_b == 0 and 0 < n_a <= 8):
        raise SystemExit(f"n_a + n_b = {n_a + n_b} != 8")
    if N_MID % (n_a * GRP_A) or (n_b and N_OUT % (n_b * GRP_B)):
        raise SystemExit("output dims not divisible by the (ncol x 64) grid")
    ng_b = N_OUT // n_b // GRP_B if n_b else 0
    return dict(
        n_a=n_a, n_b=n_b,
        ng_a=N_MID // n_a // GRP_A, ng_b=ng_b,
        cpc_a=N_MID // n_a, cpc_b=N_OUT // n_b if n_b else 0,
        nt_a=S_A * (N_MID // n_a // GRP_A),
        nt_b=S_B * ng_b if n_b else 0,
        mid_bytes=K_DOWN * 4,
        obj_a=A_OBJ, obj_b=B_OBJ,
    )


# ----------------------------------------------------------------------------
# device C
# ----------------------------------------------------------------------------

def _acc_off(mat, c):
    """row0 index of output col c inside a [mat][nt][row][c16] acc: mat*1024 +
    (c>>4)*64 + (c&15)."""
    return mat * 256 + (c >> 4) * 64 + (c & 15)


_HEAD = """#include <stdint.h>
#include <string.h>
#include <aie_api/aie.hpp>
using namespace aie;
typedef aie::mmul<4, 16, 16, int8, int4> MMUL;
"""


def _azero_src():
    return _HEAD + """
extern "C" void azero(int32_t * acc) {
    for (int i = 0; i < 512; i++) acc[i] = 0;
}
"""


def _amac_src():
    return _HEAD + """
extern "C" void amac(const uint8_t * obj, int32_t * acc) {
    const int8_t * A = (const int8_t *)(obj + %(A_AOF)d);
    for (int nt = 0; nt < 4; ++nt) {
        MMUL Cg(aie::load_v<64>(acc + nt * 64));
        MMUL Cu(aie::load_v<64>(acc + 256 + nt * 64));
        for (int kz = 0; kz < 4; ++kz) {
            auto A0 = aie::load_v<64>(A + kz * 64);
            const uint8_t * bg = obj + %(A_BGO)d + (kz * 4 + nt) * 128;
            const uint8_t * bu = obj + %(A_BUO)d + (kz * 4 + nt) * 128;
            auto Bg = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(bg));
            auto Bu = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(bu));
            Cg.mac(A0, Bg);
            Cu.mac(A0, Bu);
        }
        aie::store_v(acc + nt * 64, Cg.template to_vector<int32_t>());
        aie::store_v(acc + 256 + nt * 64, Cu.template to_vector<int32_t>());
    }
}
""" % dict(A_AOF=A_AOF, A_BGO=A_BGO, A_BUO=A_BUO)


def _afin_src():
    return _HEAD + """
extern "C" void afin(const uint8_t * obj, const int32_t * acc, float * mid) {
    const float * sg = (const float *)(obj + 0);
    const float * su = (const float *)(obj + 256);
    alignas(64) float g[64], u[64];
    for (int c = 0; c < 64; ++c) {
        g[c] = sg[c] * (float)acc[(c >> 4) * 64 + (c & 15)];
        u[c] = su[c] * (float)acc[256 + (c >> 4) * 64 + (c & 15)];
    }
    for (int half = 0; half < 2; ++half) {
        float * g32 = g + half * 32;
        float * u32 = u + half * 32;
        aie::vector<float, 32> gfv = aie::load_v<32>(g32);
        aie::vector<float, 32> ufv = aie::load_v<32>(u32);
        aie::accum<accfloat, 32> ga, ua;
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
        for (int c = 0; c < 32; ++c) mid[half * 32 + c] = pf[c];
    }
}
"""


def _bquant_src():
    return _HEAD + """
extern "C" void bquant(float * mid, float * aux) {
    float amax = 0.0f;
    for (int k = 0; k < %(K_D)d; ++k) {
        float a = mid[k]; if (a < 0.0f) a = -a;
        if (a > amax) amax = a;
    }
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    int8_t * codes = (int8_t *)mid;
    for (int k = 0; k < %(K_D)d; ++k) {
        float v = mid[k] / d;
        if (v > 127.0f) v = 127.0f; else if (v < -128.0f) v = -128.0f;
        int q = (int)v;
        if (v - (float)q >= 0.5f) q++; else if (v - (float)q <= -0.5f) q--;
        const int t = k >> 6, kz = (k >> 4) & 3, k16 = k & 15;
        codes[t * 256 + kz * 64 + k16] = (int8_t)q;
        if ((k & 63) == 63)
            for (int z = t * 256 + 16; z < t * 256 + 256; z += 64)
                for (int b = 0; b < 48; ++b) codes[z + b] = 0;
    }
    aux[0] = d;
}
""" % dict(K_D=K_DOWN)


def _bzero_src():
    return _HEAD + """
extern "C" void bzero(int32_t * acc) {
    for (int i = 0; i < 256; i++) acc[i] = 0;
}
"""


def _bmac_src():
    return _HEAD + """
extern "C" void bmac(const uint8_t * obj, const float * midf, int32_t * acc) {
    const int8_t * codes = (const int8_t *)midf;
    const int t = obj[%(B_TAG)d] | (obj[%(B_TAG)d + 1] << 8);
    const int8_t * A = codes + t * 256;
    for (int nt = 0; nt < 4; ++nt) {
        MMUL C0(aie::load_v<64>(acc + nt * 64));
        for (int kz = 0; kz < 4; ++kz) {
            auto A0 = aie::load_v<64>(A + kz * 64);
            const uint8_t * bs = obj + (kz * 4 + nt) * 128;
            auto B0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(bs));
            C0.mac(A0, B0);
        }
        aie::store_v(acc + nt * 64, C0.template to_vector<int32_t>());
    }
}
""" % dict(B_TAG=B_TAG)



# IRON designs
# ----------------------------------------------------------------------------

def _kern_a(g):
    ca = ["-O2", "-DNDEBUG"]
    FEED_T = np.ndarray[(g["obj_a"],), np.dtype[np.uint8]]
    ACC_T = np.ndarray[(512,), np.dtype[np.int32]]
    MOUT_T = np.ndarray[(GRP_A,), np.dtype[np.float32]]
    return dict(
        azero=iron.ExternalFunction(name="azero", source_string=_azero_src(),
                                    arg_types=[ACC_T], compile_flags=ca, inline=True),
        amac=iron.ExternalFunction(name="amac", source_string=_amac_src(),
                                   arg_types=[FEED_T, ACC_T],
                                   compile_flags=ca, inline=True),
        afin=iron.ExternalFunction(name="afin", source_string=_afin_src(),
                                   arg_types=[FEED_T, ACC_T, MOUT_T],
                                   compile_flags=["-O2", "-DNDEBUG",
                                                  "-fno-unroll-loops"],
                                   inline=True),
    )


def _kern_b(g):
    ca = ["-O2", "-DNDEBUG"]
    MID_T = np.ndarray[(K_DOWN,), np.dtype[np.float32]]
    AUX_T = np.ndarray[(8,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(256,), np.dtype[np.int32]]
    BFEED_T = np.ndarray[(g["obj_b"],), np.dtype[np.uint8]]
    return dict(
        bquant=iron.ExternalFunction(name="bquant", source_string=_bquant_src(),
                                     arg_types=[MID_T, AUX_T],
                                     compile_flags=ca, inline=True),
        bzero=iron.ExternalFunction(name="bzero", source_string=_bzero_src(),
                                    arg_types=[OUT_T], compile_flags=ca, inline=True),
        bmac=iron.ExternalFunction(name="bmac", source_string=_bmac_src(),
                                   arg_types=[BFEED_T, MID_T, OUT_T],
                                   compile_flags=ca, inline=True),
    )


@iron.jit
def ffn_a_mmul(*, n_a: CompileTime[int] = 4, dev_name: CompileTime[str] = "npu2"):
    """Stage A only xclbin: mmul gate/up + in-core silu -> mid f32 (bring-up)."""
    g = resolve(n_a, 0)
    FEED_T = np.ndarray[(g["obj_a"],), np.dtype[np.uint8]]
    ACC_T = np.ndarray[(512,), np.dtype[np.int32]]
    MOUT_T = np.ndarray[(GRP_A,), np.dtype[np.float32]]
    ka = _kern_a(g)

    workers = []
    rt_args = []
    for col in range(g["n_a"]):
        f3 = ObjectFifo(FEED_T, name=f"af3_{col}", depth=2)
        f2 = f3.cons().forward(obj_type=FEED_T, name=f"af2_{col}", tile=Tile(col, 1))
        a23 = ObjectFifo(ACC_T, name=f"aa23_{col}", depth=1)
        a12 = a23.prod().join([0], obj_types=[ACC_T], names=[f"aa12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        m23 = ObjectFifo(MOUT_T, name=f"am23_{col}", depth=1)
        m12 = m23.prod().join([0], obj_types=[MOUT_T], names=[f"am12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def a_core(fc, acp, mp, kz, km, kf, ng=g["ng_a"], ns=S_A):
            for _ in range_(ng):
                ac = acp.acquire(1)
                kz(ac)
                for _ in range_(ns - 1):
                    oi = fc.acquire(1)
                    km(oi, ac)
                    fc.release(1)
                ol = fc.acquire(1)          # last slab held for the epilogue
                km(ol, ac)
                om = mp.acquire(1)
                kf(ol, ac, om)
                fc.release(1)
                mp.release(1)
                acp.release(1)

        workers.append(Worker(a_core, [f2.cons(), a12.prod(), m12.prod(),
                                       ka["azero"], ka["amac"], ka["afin"]],
                              tile=Tile(col, 2), stack_size=0x3000))
        rt_args += [f3.prod(tile=Tile(col, 0)),
                    a23.cons(tile=Tile(col, 0)), m23.cons(tile=Tile(col, 0))]

    FEED_g = np.ndarray[(g["n_a"] * g["nt_a"] * g["obj_a"],), np.dtype[np.uint8]]
    MID_g = np.ndarray[(K_DOWN,), np.dtype[np.float32]]
    ACC_g = np.ndarray[(g["n_a"] * 512,), np.dtype[np.int32]]

    def seq_fn(FEED, MID, ACC, *fifos):
        it = iter(fifos)
        fill = []; ad = []; md = []
        for c in range(g["n_a"]):
            fill.append(next(it)); ad.append(next(it)); md.append(next(it))
        for c in range(g["n_a"]):
            tg = TaskGroup()
            fill[c].fill(FEED, tap=TensorAccessPattern(
                (g["n_a"] * g["nt_a"] * g["obj_a"],),
                offset=c * g["nt_a"] * g["obj_a"],
                sizes=[g["nt_a"] * g["obj_a"]], strides=[1]), group=tg)
            tg.finish()
        for grp in range(g["ng_a"]):
            for c in range(g["n_a"]):
                go = TaskGroup()
                md[c].drain(MID, tap=TensorAccessPattern(
                    (K_DOWN,), offset=c * g["cpc_a"] + grp * GRP_A,
                    sizes=[GRP_A], strides=[1]), wait=True, group=go)
                ad[c].drain(ACC, tap=TensorAccessPattern(
                    (g["n_a"] * 512,), offset=c * 512, sizes=[512],
                    strides=[1]), wait=True, group=go)
                go.finish()

    rt_args = [FEED_g, MID_g, ACC_g] + rt_args
    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=g["n_a"]), rt, workers)
    return prog.resolve_program()


@iron.jit
def ffn_ab_mmul(*, n_a: CompileTime[int] = 4, n_b: CompileTime[int] = 4,
                dev_name: CompileTime[str] = "npu2"):
    """Fused xclbin: stage A (mmul gate/up + silu -> mid f32) on cols 0..n_a-1,
    stage B (in-core mid requant + mmul down -> raw int32) on cols n_a..7."""
    g = resolve(n_a, n_b)
    FEED_T = np.ndarray[(g["obj_a"],), np.dtype[np.uint8]]
    ACC_T = np.ndarray[(512,), np.dtype[np.int32]]
    MOUT_T = np.ndarray[(GRP_A,), np.dtype[np.float32]]
    BFEED_T = np.ndarray[(g["obj_b"],), np.dtype[np.uint8]]
    MID_T = np.ndarray[(K_DOWN,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(256,), np.dtype[np.int32]]
    AUX_T = np.ndarray[(8,), np.dtype[np.float32]]
    ka = _kern_a(g)
    kb = _kern_b(g)

    FEED_g = np.ndarray[(g["n_a"] * g["nt_a"] * g["obj_a"],), np.dtype[np.uint8]]
    MID_g = np.ndarray[(K_DOWN,), np.dtype[np.float32]]
    ACC_g = np.ndarray[(g["n_a"] * 512,), np.dtype[np.int32]]
    BFEED_g = np.ndarray[(g["n_b"] * g["nt_b"] * g["obj_b"],), np.dtype[np.uint8]]
    OUT_g = np.ndarray[(g["n_b"] * g["ng_b"] * 256,), np.dtype[np.int32]]
    AUX_g = np.ndarray[(g["n_b"] * 8,), np.dtype[np.float32]]

    workers = []
    rt_args = [FEED_g, MID_g, ACC_g, BFEED_g, OUT_g, AUX_g]

    # ---- stage A ----
    for col in range(g["n_a"]):
        f3 = ObjectFifo(FEED_T, name=f"af3_{col}", depth=2)
        f2 = f3.cons().forward(obj_type=FEED_T, name=f"af2_{col}", tile=Tile(col, 1))
        a23 = ObjectFifo(ACC_T, name=f"aa23_{col}", depth=1)
        a12 = a23.prod().join([0], obj_types=[ACC_T], names=[f"aa12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        m23 = ObjectFifo(MOUT_T, name=f"am23_{col}", depth=1)
        m12 = m23.prod().join([0], obj_types=[MOUT_T], names=[f"am12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def a_core(fc, acp, mp, kz, km, kf, ng=g["ng_a"], ns=S_A):
            for _ in range_(ng):
                ac = acp.acquire(1)
                kz(ac)
                for _ in range_(ns - 1):
                    oi = fc.acquire(1)
                    km(oi, ac)
                    fc.release(1)
                ol = fc.acquire(1)
                km(ol, ac)
                om = mp.acquire(1)
                kf(ol, ac, om)
                fc.release(1)
                mp.release(1)
                acp.release(1)

        workers.append(Worker(a_core, [f2.cons(), a12.prod(), m12.prod(),
                                       ka["azero"], ka["amac"], ka["afin"]],
                              tile=Tile(col, 2), stack_size=0x3000))
        rt_args += [f3.prod(tile=Tile(col, 0)),
                    a23.cons(tile=Tile(col, 0)), m23.cons(tile=Tile(col, 0))]

    # ---- stage B ----
    bbar = [WorkerRuntimeBarrier() for _ in range(g["n_b"])]
    for bi in range(g["n_b"]):
        col = g["n_a"] + bi
        m3 = ObjectFifo(MID_T, name=f"bm3_{col}", depth=1)
        m2 = m3.cons().forward(obj_type=MID_T, name=f"bm2_{col}", tile=Tile(col, 1))
        e3 = ObjectFifo(BFEED_T, name=f"be3_{col}", depth=2)
        e2 = e3.cons().forward(obj_type=BFEED_T, name=f"be2_{col}", tile=Tile(col, 1))
        o23 = ObjectFifo(OUT_T, name=f"bo23_{col}", depth=1)
        o12 = o23.prod().join([0], obj_types=[OUT_T], names=[f"bo12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        u23 = ObjectFifo(AUX_T, name=f"bu23_{col}", depth=1)
        u12 = u23.prod().join([0], obj_types=[AUX_T], names=[f"bu12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def b_core(mc, ec, op, up, kq, kz, km, bar, ng=g["ng_b"], ns=S_B):
            bar.wait_for_value(1)
            mi = mc.acquire(1)
            ax = up.acquire(1)
            kq(mi, ax)
            up.release(1)
            for _ in range_(ng):
                oo = op.acquire(1)
                kz(oo)
                for _ in range_(ns):
                    ei = ec.acquire(1)
                    km(ei, mi, oo)
                    ec.release(1)
                op.release(1)
            mc.release(1)

        workers.append(Worker(b_core, [m2.cons(), e2.cons(), o12.prod(), u12.prod(),
                                       kb["bquant"], kb["bzero"], kb["bmac"],
                                       bbar[bi]], tile=Tile(col, 2), stack_size=0x3000))
        rt_args += [m3.prod(tile=Tile(col, 0)), e3.prod(tile=Tile(col, 0)),
                    o23.cons(tile=Tile(col, 0)), u23.cons(tile=Tile(col, 0)),
                    bbar[bi]]

    def seq_fn(FEED, MID, ACC, BFEED, OUTG, AUX, *fifos):
        it = iter(fifos)
        a_fill = []; a_ad = []; a_md = []
        for _ in range(g["n_a"]):
            a_fill.append(next(it))
            a_ad.append(next(it)); a_md.append(next(it))
        b_mfill = []; b_efill = []; b_od = []; b_ud = []; b_bar = []
        for _ in range(g["n_b"]):
            b_mfill.append(next(it)); b_efill.append(next(it))
            b_od.append(next(it)); b_ud.append(next(it)); b_bar.append(next(it))

        for c in range(g["n_a"]):
            tg = TaskGroup()
            a_fill[c].fill(FEED, tap=TensorAccessPattern(
                (g["n_a"] * g["nt_a"] * g["obj_a"],),
                offset=c * g["nt_a"] * g["obj_a"],
                sizes=[g["nt_a"] * g["obj_a"]], strides=[1]), group=tg)
            tg.finish()
        for grp in range(g["ng_a"]):
            for c in range(g["n_a"]):
                go = TaskGroup()
                a_md[c].drain(MID, tap=TensorAccessPattern(
                    (K_DOWN,), offset=c * g["cpc_a"] + grp * GRP_A,
                    sizes=[GRP_A], strides=[1]), wait=True, group=go)
                a_ad[c].drain(ACC, tap=TensorAccessPattern(
                    (g["n_a"] * 512,), offset=c * 512, sizes=[512],
                    strides=[1]), wait=True, group=go)
                go.finish()

        for bi in range(g["n_b"]):
            b_bar[bi].set(1)

        for bi in range(g["n_b"]):
            tg = TaskGroup()
            b_mfill[bi].fill(MID, tap=TensorAccessPattern(
                (K_DOWN,), offset=0, sizes=[K_DOWN], strides=[1]), group=tg)
            tg.finish()
        for bi in range(g["n_b"]):
            go = TaskGroup()
            b_ud[bi].drain(AUX, tap=TensorAccessPattern(
                (g["n_b"] * 8,), offset=bi * 8, sizes=[8], strides=[1]),
                wait=True, group=go)
            go.finish()
        for bi in range(g["n_b"]):
            tg = TaskGroup()
            b_efill[bi].fill(BFEED, tap=TensorAccessPattern(
                (g["n_b"] * g["nt_b"] * g["obj_b"],),
                offset=bi * g["nt_b"] * g["obj_b"],
                sizes=[g["nt_b"] * g["obj_b"]], strides=[1]), group=tg)
            tg.finish()
        for grp in range(g["ng_b"]):
            for bi in range(g["n_b"]):
                go = TaskGroup()
                b_od[bi].drain(OUTG, tap=TensorAccessPattern(
                    (g["n_b"] * g["ng_b"] * 256,),
                    offset=(bi * g["ng_b"] + grp) * 256, sizes=[256],
                    strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=g["n_a"] + g["n_b"]), rt, workers)
    return prog.resolve_program()



# host pack + numpy kernel emulation
# ----------------------------------------------------------------------------

def dequant_col_vec(wtype, chunk, blocks_k):
    """Exact fp64 dequant of one [K] column from ggml Q4_K/Q6_K blocks."""
    qb = 144 if wtype == 4 else 210
    q = np.empty(blocks_k * BLK, dtype=np.float64)
    for b in range(blocks_k):
        blk = chunk[b * qb:(b + 1) * qb]
        if wtype == 4:
            d = proj_qK.fp16(int.from_bytes(blk[0:2], 'little'))
            mn = proj_qK.fp16(int.from_bytes(blk[2:4], 'little'))
            sc = blk[4:16].astype(np.int64)
            ql = blk[16:144].astype(np.int64)
            for gi in range(4):
                a, b0 = proj_qK.gsm(sc, 2 * gi + 0); d1 = d * a; m1 = mn * b0
                c, e = proj_qK.gsm(sc, 2 * gi + 1); d2 = d * c; m2 = mn * e
                lane = ql[gi * 32:(gi + 1) * 32]
                v = np.empty(64, dtype=np.float64)
                v[:32] = d1 * (lane & 0xF) - m1
                v[32:] = d2 * (lane >> 4) - m2
                q[b * BLK + gi * 64: b * BLK + gi * 64 + 64] = v
        else:
            d = proj_qK.fp16(int.from_bytes(blk[208:210], 'little'))
            ql = blk[0:128].astype(np.int64)
            qh = blk[128:192].astype(np.int64)
            sc = blk[192:208].astype(np.int8).astype(np.int64)
            for hf in range(2):
                qlo = ql[hf * 64:]; qho = qh[hf * 32:]; sco = sc[hf * 8:]
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
    return q


def dequant_mat(wtype, qbytes, K, N):
    """Exact fp64 dequant of a Q4_K/Q6_K [K x N] tensor, one col at a time."""
    qb = 144 if wtype == 4 else 210
    cb = (K // BLK) * qb
    W = np.empty((K, N), dtype=np.float64)
    for n in range(N):
        W[:, n] = dequant_col_vec(wtype, qbytes[n * cb:(n + 1) * cb], K // BLK)
    return W


def quant_w_col(W):
    """per-column (full K) int4 requant: codes int8 -8..7, d_w per column."""
    amax = np.abs(W).max(axis=0, keepdims=True)
    d_w = np.where(amax > 0, amax / 7.0, 1.0)
    codes = np.clip(np.round(W / d_w), -8, 7).astype(np.int8)
    return codes, d_w.ravel().astype(np.float32)


def quant_a_row(h):
    amax = np.abs(h).max()
    d = amax / 127.0 if amax > 0 else 1.0
    return np.clip(np.round(h / d), -128, 127).astype(np.int8), np.float32(d)


def pack_tile4(codes, k0, n0, buf, off):
    """One 64(k) x 64(n) int4 element at buf[off:off+2048], w4a8 nibble order:
    (kz,nt,kr,nc) -> byte (kz*4+nt)*128 + kr*8 + nc/2 low nibble first."""
    for kz in range(4):
        for nt in range(4):
            base = (kz * 4 + nt) * 128
            for kr in range(16):
                for nc in range(16):
                    v = int(codes[k0 + kz * 16 + kr, n0 + nt * 16 + nc]) & 0xF
                    o = off + base + kr * 8 + nc // 2
                    if nc % 2 == 0:
                        buf[o] |= v
                    else:
                        buf[o] |= (v << 4)


def build_a_feed(g, hq, da, codes_g, dw_g, codes_u, dw_u):
    """Per (col, group, slab) A element object with a 512-B scale header."""
    feed = np.zeros(g["n_a"] * g["nt_a"] * A_OBJ, dtype=np.uint8)
    n_g = g["ng_a"]
    for c in range(g["n_a"]):
        for grp in range(n_g):
            m0 = c * g["cpc_a"] + grp * GRP_A
            hdr = np.concatenate([
                (da * dw_g[m0:m0 + GRP_A]).astype(np.float32),
                (da * dw_u[m0:m0 + GRP_A]).astype(np.float32),
            ]).view(np.uint8)
            for t in range(S_A):
                o = feed[(c * g["nt_a"] + grp * S_A + t) * A_OBJ:]
                o[0:A_HDR] = hdr
                at = o[A_AOF:A_AOF + 256]
                at[:] = 0
                for k in range(64):
                    at[(k >> 4) * 64 + (k & 15)] = hq[t * 64 + k]
                pack_tile4(codes_g, t * 64, m0, o, A_BGO)
                pack_tile4(codes_u, t * 64, m0, o, A_BUO)
    return feed


def build_b_feed(g, codes_d):
    """down element objects: B int4 at 0..2048, u16 slab tag at 2048."""
    n = g["n_b"] * g["nt_b"]
    feed = np.zeros(n * B_OBJ, dtype=np.uint8)
    for bi in range(g["n_b"]):
        for grp in range(g["ng_b"]):
            n0 = bi * g["cpc_b"] + grp * GRP_B
            for t in range(S_B):
                o = feed[(bi * g["nt_b"] + grp * S_B + t) * B_OBJ:]
                pack_tile4(codes_d, t * 64, n0, o, 0)
                o[B_TAG] = t & 0xFF
                o[B_TAG + 1] = (t >> 8) & 0xFF
    return feed


def mid_from_feed(g, feed, hq, da, dw_g, dw_u, hff_ref):
    """Host emulation of stage A: per (col, grp) reproduce azero/amac/afin
    with the SAME integer math (row0 only) + bf16-tanh silu."""
    mid = np.zeros(N_MID, dtype=np.float32)
    for c in range(g["n_a"]):
        for grp in range(g["ng_a"]):
            m0 = c * g["cpc_a"] + grp * GRP_A
            acc_g = np.zeros(GRP_A, dtype=np.int64)
            acc_u = np.zeros(GRP_A, dtype=np.int64)
            for t in range(S_A):
                o = feed[(c * g["nt_a"] + grp * S_A + t) * A_OBJ:]
                wg = unpack_tile4(o[A_BGO:A_BGO + 2048])
                wu = unpack_tile4(o[A_BUO:A_BUO + 2048])
                a = hq[t * 64:(t + 1) * 64].astype(np.int64)
                acc_g += a @ wg
                acc_u += a @ wu
            o0 = feed[(c * g["nt_a"] + grp * S_A + 0) * A_OBJ:]
            sg = o0[0:256].view(np.float32)[:64]
            su = o0[256:512].view(np.float32)[:64]
            gv = (acc_g * sg).astype(np.float32)
            uv = (acc_u * su).astype(np.float32)
            mid[m0:m0 + GRP_A] = bf16_silu_mul(gv, uv)
    return mid


def unpack_tile4(buf):
    """Decode one 64x64 int4 element to int64 codes (host emu of the pack)."""
    w = np.zeros((64, 64), dtype=np.int64)
    for kz in range(4):
        for nt in range(4):
            base = (kz * 4 + nt) * 128
            for kr in range(16):
                for nc in range(16):
                    b = int(buf[base + kr * 8 + nc // 2])
                    v = (b & 0xF) if nc % 2 == 0 else ((b >> 4) & 0xF)
                    w[kz * 16 + kr, nt * 16 + nc] = v - 16 if v >= 8 else v
    return w


def bf16_silu_mul(g32, u32):
    """bf16-tanh silu chain * up, fp64->f32 (matches oracle_mid_bf16)."""
    gb = g32.astype(np.float32).astype(_bf16()).astype(np.float64)
    ub = u32.astype(np.float32).astype(_bf16()).astype(np.float64)
    th = np.tanh(gb * 0.5)
    thb = th.astype(np.float32).astype(_bf16()).astype(np.float64)
    sig = ((thb + 1.0) * 0.5).astype(np.float32).astype(_bf16()).astype(np.float64)
    silu = (gb * sig).astype(np.float32).astype(_bf16()).astype(np.float64)
    return (silu * ub).astype(np.float32)


def _bf16():
    from ml_dtypes import bfloat16
    return bfloat16


def out_from_down(g, codes_d, dw_d, mid32):
    """Host emulation of stage B (bquant global scale, bmac int acc, host
    rescale d_aB*d_w).  Return out fp64 and the raw int32 rows."""
    amax = np.abs(mid32).max()
    d_aB = np.float32(amax / 127.0 if amax > 0 else 1.0)
    # device rounding (half away from zero)
    mq = np.empty(K_DOWN, dtype=np.int64)
    mf = mid32.astype(np.float64) / float(d_aB)
    mq[:] = np.trunc(mf)
    frac = mf - mq
    mq[frac >= 0.5] += 1
    mq[frac <= -0.5] -= 1
    mq = np.clip(mq, -128, 127)
    out = np.zeros(N_OUT, dtype=np.float64)
    codes_d2 = codes_d.astype(np.int64)
    for n in range(N_OUT):
        c = codes_d2[:, n]
        acc = int(mq @ c)
        out[n] = acc * float(d_aB) * float(dw_d[n])
    return out, mq, float(d_aB)


def oracle_stage_a(h64, qg, qu):
    g4 = dict(wtype=4, qb=144, ql_off=16, qh_off=None, blocks_k=K // BLK)
    cb = (K // BLK) * 144
    gq = np.array([oracle_col(g4, qg[m * cb:(m + 1) * cb], h64) for m in range(N_MID)])
    uq = np.array([oracle_col(g4, qu[m * cb:(m + 1) * cb], h64) for m in range(N_MID)])
    return gq, uq


def oracle_stage_b(qd, mid64):
    q6 = dict(wtype=6, qb=210, ql_off=0, qh_off=128, blocks_k=K_DOWN // BLK)
    cb = (K_DOWN // BLK) * 210
    return np.array([oracle_col(q6, qd[n * cb:(n + 1) * cb], mid64)
                     for n in range(N_OUT)])


def compile_design(opts, g):
    import time
    os.makedirs(opts.workdir, exist_ok=True)
    tag = "a" if opts.stage == "a" else "ab"
    base = os.path.join(opts.workdir, f"ffn_mmul_{tag}")
    if opts.stage == "a":
        spec = ffn_a_mmul.specialize(n_a=g["n_a"], dev_name=opts.dev)
    else:
        spec = ffn_ab_mmul.specialize(n_a=g["n_a"], n_b=g["n_b"], dev_name=opts.dev)
    t0 = time.time()
    xclbin_path, insts_path = spec.compile(xclbin_path=base + ".xclbin",
                                           inst_path=base + ".insts.bin")
    print(f"compiled {xclbin_path} ({time.time()-t0:.0f}s)")
    import design_tag
    design_tag.stamp(xclbin_path, insts_path, opts.dev or "")
    return xclbin_path, insts_path


def main():
    ap = argparse.ArgumentParser(prog="ffn_layer_mmul")
    add_compile_args(ap)
    ap.add_argument("--na", type=int, default=4)
    ap.add_argument("--nb", type=int, default=4)
    ap.add_argument("--stage", default="ab", choices=["a", "ab"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-q4km.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    g = resolve(opts.na, opts.nb)
    print(f"stage A: cols={g['n_a']} groups={g['ng_a']} slabs={S_A} "
          f"obj={A_OBJ} B  | stage B: cols={g['n_b']} groups={g['ng_b']} "
          f"slabs={S_B} obj={B_OBJ} B")

    if opts.compile:
        compile_design(opts, g)
        return

    reader = load_reader(opts.model)
    names = ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]
    qbytes = {}
    for n in names:
        t = next(tt for tt in reader.tensors if tt.name == n)
        raw = np.ascontiguousarray(np.asarray(t.data).reshape(-1)).view(np.uint8)
        qbytes[n] = np.frombuffer(raw, np.uint8)
    if (int(next(t for t in reader.tensors if t.name == names[0]).tensor_type),
            int(next(t for t in reader.tensors if t.name == names[1]).tensor_type)) != (12, 12):
        sys.exit("gate/up are not Q4_K")
    if int(next(t for t in reader.tensors if t.name == names[2]).tensor_type) != 14:
        sys.exit("down is not Q6_K")

    # host weights -> int4 grid (per-column d_w), exact Q4_K/Q6_K dequant
    gW = dequant_mat(4, qbytes[names[0]], K, N_MID)
    uW = dequant_mat(4, qbytes[names[1]], K, N_MID)
    dW = dequant_mat(6, qbytes[names[2]], K_DOWN, N_OUT)
    codes_g, dw_g = quant_w_col(gW)
    codes_u, dw_u = quant_w_col(uW)
    codes_d, dw_d = quant_w_col(dW)

    rng = np.random.default_rng(opts.seed)
    h = rng.standard_normal(K).astype(np.float32)
    h64 = h.astype(np.float64)
    hq, da = quant_a_row(h)

    feed_a = build_a_feed(g, hq, da, codes_g, dw_g, codes_u, dw_u)
    feed_b = build_b_feed(g, codes_d)

    # ---- host emulation of the device kernels (pack + integer math) ----
    mid_emu = mid_from_feed(g, feed_a, hq, da, dw_g, dw_u, h)
    # w4a8-model reference: integer gemv then bf16 silu chain
    from ffn_mid import oracle_mid_bf16
    gm = (hq.astype(np.int64) @ codes_g.astype(np.int64)).astype(np.float64)
    um = (hq.astype(np.int64) @ codes_u.astype(np.int64)).astype(np.float64)
    mid_model = oracle_mid_bf16(gm * da * dw_g.astype(np.float64),
                                um * da * dw_u.astype(np.float64))
    mid_oracle = oracle_mid_bf16(*oracle_stage_a(h64, qbytes[names[0]],
                                                 qbytes[names[1]]))
    print(f"[host] emu mid vs w4a8-model: {np.abs(mid_emu.astype(np.float64) - mid_model).max():.2e}")
    print(f"[host] w4a8-model mid vs scalar oracle: "
          f"{np.abs(mid_model - mid_oracle).max():.2e}  "
          f"(rel {np.abs(mid_model - mid_oracle).max() / np.abs(mid_oracle).max():.2e})")

    # stage B host emulation against the model
    out_emu, _, d_aB_emu = out_from_down(g, codes_d, dw_d,
                                         mid_model.astype(np.float32))
    out_oracle = oracle_stage_b(qbytes[names[2]], mid_oracle)
    print(f"[host] w4a8 out vs scalar oracle (bf16-mid): "
          f"{np.abs(out_emu - out_oracle).max():.2e}  "
          f"(rel {np.abs(out_emu - out_oracle).max() / np.abs(out_oracle).max():.2e})")
    if not opts.run:
        print("host-only ok (no NPU run)")
        return

    import pyxrt as xrt
    import time
    xclbin_path, insts_path = compile_design(opts, g)

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

    af_bo = mk_bo(feed_a)
    bf_bo = mk_bo(feed_b)
    mid_bo = mk_bo(np.zeros(K_DOWN, np.float32), ro=True)
    acc_bo = mk_bo(np.zeros(g["n_a"] * 512, np.int32), ro=True)
    out_bo = mk_bo(np.zeros(g["n_b"] * g["ng_b"] * 256, np.int32), ro=True)
    aux_bo = mk_bo(np.zeros(max(g["n_b"], 1) * 8, np.float32), ro=True)
    out0 = np.zeros(max(g["n_b"] * g["ng_b"], 1) * 256, np.int32)

    def once():
        if opts.stage == "a":
            r = kernel(3, insts_bo, int(insts.nbytes), af_bo, mid_bo, acc_bo)
        else:
            r = kernel(3, insts_bo, int(insts.nbytes), af_bo, mid_bo, acc_bo,
                       bf_bo, out_bo, aux_bo)
        return r

    r = once()
    if r.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        sys.exit("run failed")

    mid = np.frombuffer(mid_bo.map(), dtype=np.float32)[:K_DOWN].copy()
    if opts.stage == "a":
        print("[npu ] stage-a xclbin: mid drained OK; timing below")
        for _ in range(max(opts.warmup, 0)):
            rr = once(); rr.wait()
        ts = []
        for _ in range(max(opts.iters, 1)):
            t1 = time.perf_counter(); rr = once(); rr.wait()
            ts.append(time.perf_counter() - t1)
        print(f"[npu ] stage-a mmul: min={min(ts)*1e3:.2f}ms mean={np.mean(ts)*1e3:.2f}ms")
        err = np.abs(mid.astype(np.float64) - mid_model).max()
        print(f"[npu ] stage-a mid max_err vs w4a8-model = {err:.3e}")
        return
    out_raw = np.frombuffer(out_bo.map(), dtype=np.int32)[:g["n_b"] * g["ng_b"] * 256].copy()
    aux = np.frombuffer(aux_bo.map(), dtype=np.float32)[:g["n_b"] * 8].copy()

    err_mid_model = np.abs(mid.astype(np.float64) - mid_model).max()
    err_mid_or = np.abs(mid.astype(np.float64) - mid_oracle).max()
    print("mid[:6] =", mid[:6])
    print("ref[:6] =", mid_oracle[:6])
    print(f"[npu ] mid max_err vs w4a8-model = {err_mid_model:.3e}  "
          f"vs scalar oracle = {err_mid_or:.3e}")
    d_aB_dev = float(aux[0])
    d_aB_emu2 = float(np.abs(mid_model.astype(np.float32)).max()) / 127.0 if \
        np.abs(mid_model).max() > 0 else 1.0
    print(f"[npu ] d_aB dev = {d_aB_dev:.5e}  emu = {d_aB_emu2:.5e}")
    # down rescale with the device d_aB, host d_w, from the drained raw int32
    def gather_out():
        out = np.zeros(N_OUT, dtype=np.float64)
        codes = codes_d.astype(np.int64)
        for bi in range(g["n_b"]):
            for grp in range(g["ng_b"]):
                blk = out_raw[(bi * g["ng_b"] + grp) * 256:
                              (bi * g["ng_b"] + grp + 1) * 256]
                for c in range(GRP_B):
                    n = bi * g["cpc_b"] + grp * GRP_B + c
                    acc = int(blk[(c >> 4) * 64 + (c & 15)])
                    out[n] = acc * float(d_aB_dev) * float(dw_d[n])
        return out
    out = gather_out()
    out_err = np.abs(out - out_oracle).max()
    print(f"[npu ] out max_err vs scalar oracle = {out_err:.3e}  "
          f"(rel {out_err / np.abs(out_oracle).max():.3e})")

    # ---- timing ----
    for _ in range(max(opts.warmup, 0)):
        rr = once(); rr.wait()
    ts = []
    for _ in range(max(opts.iters, 1)):
        t1 = time.perf_counter()
        rr = once(); rr.wait()
        ts.append(time.perf_counter() - t1)
    print(f"[npu ] fused ffn mmul: min={min(ts)*1e3:.2f}ms "
          f"mean={np.mean(ts)*1e3:.2f}ms  (N={len(ts)})")
    print("nan mid =", np.isnan(mid).sum(), " nan out =", np.isnan(out).sum())


if __name__ == "__main__":
    main()
