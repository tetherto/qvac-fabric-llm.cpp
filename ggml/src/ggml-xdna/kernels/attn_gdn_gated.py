#!/usr/bin/env python3
# attn_gdn_gated.py -*- Python -*-
#
# A-half of the fused recurrent layer: the merged conv+norm+gdn design of
# attn_gdn_txn.py PLUS the rec_gated epilogue tile, all in ONE xclbin / ONE
# per-token run, with no host attn round trip.  Channel verdict (probe #1): a
# shim column exposes 2 MM2S + 2 S2MM DMA channels, and the rec_gated tile as
# written takes 2 MM2S fills (az, gamma) + 1 S2MM drain.  The decoded
# attn_gdn_txn stream shows only the norm columns (2-3) have spare channels,
# and they have exactly one free MM2S (norm x fill owns the other) and one free
# S2MM.  The allocator therefore rejects a norm-column gated tile fed by az +
# gamma fills:
#
#   tile (2, 0) requires 0 input/1 output DMA channels, but only
#   1 input/0 output available
#
# So the gated tile on a norm column is a SINGLE MM2S chain (gamma folded into
# the per-head object) + a single S2MM drain.  The azg head object is [attn 128
# | z 128 | gamma 128 | hh] (3*S_V+1 fp32), and - key for the backend - the azg
# BO is BOTH the gdn attn drain target and the gated fill source in the same
# run: gdn chunk (head,j) S2MM-drains its 16 attn values into azg at
# head*AZG_N + j*CHUNK (the 8 chunks of a head fill azg[0:128] contiguously),
# then the gated phase MM2S-fills the 16 azg heads from the same BO and drains
# one OUT object (gated f32 scratch + aq int8 codes + d_a).  The host never
# sees attn: it only uploads z into azg[head*AZG_N+128..] per token (z is known
# before the gdn stage) and gamma/hh lanes once per layer, then reads aq + d_a
# out of arg5.
#
# Artifacts (fresh workdir per compile):
#   python attn_gdn_gated.py -d npu2 --workdir build/bin --tag ""
#   -> build/bin/attn_gdn_gated.{xclbin,insts.bin}
#
# Run BO set (6): arg0 feed, arg1 x, arg2 pkvb, arg3 state (bf16), arg4 azg
# (16 x [attn|z|gamma|hh], 385 fp32 each; attn lanes device-written, z/gamma/hh
# lanes host-seeded), arg5 out (10244 B: gated f32 scratch + aq + d_a).

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import hashlib

import numpy as np

try:
    from ml_dtypes import bfloat16
except Exception:
    bfloat16 = None

import aie.iron as iron
from aie.iron import (
    CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cn  # noqa: E402
import gdn_v  # noqa: E402
import rec_gated
import rec_post  # noqa: E402
import design_tag

# ---- stage geometry (attn_cn + gdn_v + rec_gated constants) ----------------
CH    = attn_cn.CH          # 6144
S_V   = attn_cn.S_V         # 128
N_VH  = attn_cn.N_VH        # 16
N_OBJ = attn_cn.N_OBJ       # 8 chunks per head
PKV_N = attn_cn.PKV_N       # 387
PKVB_N = attn_cn.PKVB_N     # 8 * 387 per head
FEED_N = attn_cn.FEED_N     # 1024
F_H   = attn_cn.F_H
F_Q   = attn_cn.F_Q
F_W   = attn_cn.F_W
HEAD_NORM = attn_cn.HEAD_NORM  # 387

NC_CONV = 2                 # conv columns (cols 0-1)
CN_CONV = attn_cn.CN        # 24 conv feed groups per conv column
NC_NORM = 2                 # norm columns (cols 2-3)
HP_NORM = N_VH // NC_NORM   # 8 heads per norm column
NC_GDN  = 4                 # gdn columns (cols 4-7)
GDN_COL0 = NC_CONV + NC_NORM
N_OBJ_GDN = N_VH * N_OBJ // NC_GDN   # 32 chunk objects per gdn column

STATE_N = N_VH * N_OBJ * gdn_v.ROWS  # 128 * 2048 bf16 rows-values

GATED_COL = NC_CONV                  # 2 (first norm column; row 4 worker)

# 3*S_V+1 is what a head needs (attn | z | gamma | hh); the stride is rounded
# up to a multiple of the 16-lane vector so that a head inside a multi-head
# object still starts on a vector boundary. At 385 every second head did not,
# and the loads there came back with the wrong lanes - quietly, as a degraded
# model rather than an error.
AZG_N  = ((3 * S_V + 1 + 15) // 16) * 16     # 400 fp32 per head
HG     = int(__import__("os").environ.get("GATED_HEADS", "1"))  # heads per object
K_G    = N_VH * S_V                  # 2048
# ssm_out's GEMV activation, then the gated f32 scratch, then aq and d_a. The
# activation is written here so the projection can read it in this dispatch
# rather than after a round trip through the host.
ACT_TILE_G = 2112
ACT_OFF_G  = 10496                    # after the scratch, aq and d_a
# GATED_FMT selects the activation layout the gated stage writes: 0 = the
# 4-bit form (tiles of 256, groups of 32), 1 = the 8-bit form (tiles of 128,
# groups of 16). Which one a model needs is decided by its ssm_out weights -
# Q4_K is the 4-bit form, Q5_K/Q6_K the 8-bit - so the design is built per
# model and the tag covers the flag.
GATED_FMT = int(__import__("os").environ.get("GATED_FMT", "1"))
ACTN   = (1 + K_G // (128 if GATED_FMT == 1 else 256)) * ACT_TILE_G
OUTN   = ACT_OFF_G + ACTN

_GAHEAD = """#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;
"""


def _gated_stub() -> int:
    """GGML/GEMV-style diagnostic: 1 drops the per-head epilogue, 2 drops the
    final int8 quantization, 3 drops both. The results are wrong; the point is
    what the stage costs without them."""
    return int(__import__("os").environ.get("GATED_STUB", "0"))


def _gated_h2_src():
    # azg per head = [attn 128][z 128][gamma 128][hh] (gamma replicated per
    # head by the host so the tile needs a single MM2S chain).  Same epilogue
    # math as rec_gated._gated_src's gated_head, gamma read from the object.
    return _GAHEAD + """
#if ATTN_ONCHIP
extern "C" void gated_h2(uint8_t * out, const float * azg4,
                         const float * att_lo, const float * att_hi) {
#else
extern "C" void gated_h2(uint8_t * out, const float * azg4) {
#endif
#if GATED_STUB & 1
    (void)out; (void)azg4;
    return;
#else
    // One object carries HG heads. The stage is one tile walking them in
    // order, and with a head per object the fifo handshake, not the
    // arithmetic, was what it spent its time on: stubbing both of its kernels
    // left the stage exactly as expensive.

    for (int hj = 0; hj < HG; hj++) {
    const float * azg = azg4 + hj * AZG_N;
    const int hh = (int)azg[384];
    float * gbuf = (float *)out + hh * 128;
#if ATTN_ONCHIP
    // The gdn block hands its attn over on chip, a round at a time, so a head
    // arrives as two objects of NC_GDN chunks. Copying them into one local
    // array costs eight vector moves and leaves the rest of the kernel - and
    // the DDR layout it falls back to - exactly as it was.
    alignas(64) float aloc[128];
    for (int i = 0; i < 64; i += 16) {
        aie::store_v(aloc + i,      aie::load_v<16>(att_lo + i));
        aie::store_v(aloc + 64 + i, aie::load_v<16>(att_hi + i));
    }
    const float * a = aloc;
#else
    const float * a = azg;
#endif
    const float * z = azg + 128;
    const float * gamma = azg + 256;
    const auto bc_h = aie::broadcast<float, 16>(0.5f);
    const auto bc1 = aie::broadcast<bfloat16, 16>(1.0f);
    // Vector sum of squares; the scalar loop this replaces was 128 dependent
    // float adds per head.
    aie::vector<float, 16> sq = aie::zeros<float, 16>();
    for (int i = 0; i < 128; i += 16) {
        auto v = aie::load_v<16>(a + i);
        sq = aie::add(sq, aie::mul(v, v).to_vector<float>());
    }
    alignas(64) float sql[16];
    aie::store_v(sql, sq);
    float ms = 0.0f;
    for (int i = 0; i < 16; i++) ms += sql[i];
    const float rsc = 1.0f / aie::sqrt(ms / 128.0f + 1e-6f);
    const auto bc_r = aie::broadcast<float, 16>(rsc);
    for (int i = 0; i < 128; i += 16) {
        auto a16 = aie::load_v<16>(a + i);
        auto z16 = aie::load_v<16>(z + i);
        auto g16 = aie::load_v<16>(gamma + i);
        auto t16 = aie::mul(z16, bc_h).to_vector<float>();
        auto thb = aie::tanh(t16);
        auto th  = aie::mul(thb, bc1).to_vector<float>();
        auto thh = aie::mul(th, bc_h).to_vector<float>();
        auto sig = aie::add(thh, bc_h);
        auto silu = aie::mul(z16, sig).to_vector<float>();
        auto g1 = aie::mul(a16, bc_r).to_vector<float>();
        auto g2 = aie::mul(g1, g16).to_vector<float>();
        aie::vector<float, 16> g = aie::mul(g2, silu).to_vector<float>();
        aie::store_v(gbuf + i, g);
    }
    }
#endif
}
"""


# ---- merged conv + norm + gdn + gated design (park workers) ----------------

# The design body, separable from the Program it is wrapped in so the same
# array configuration can be built on its own or alongside another design in
# one xclbin (fused_layer.py). Returns what a Program needs: the workers, the
# runtime argument list and the sequence over it.
def build_core(dev_name: str = "npu2"):
    # CONV_GPO: feed groups an object carries. Four is the most L1 holds - a
    # feed object is 16 KB and the fifo is double-buffered - and the most the
    # x layout allows, since a round has to stay inside one q/k/v region.
    # Measured 229 / 216 / 204 us for 1 / 2 / 4, so the object size is part of
    # the stage's cost but not the whole of it.
    gpo = int(__import__("os").environ.get("CONV_GPO", "4"))
    # GATED_ATTN_ONCHIP: the gdn block's attn reaches the gated tile through
    # the array instead of DDR. A gated head is two gdn rounds, so the tile
    # takes two objects per head; see gated_fn.
    att_onchip = int(__import__("os").environ.get("GATED_ATTN_ONCHIP", "1"))
    # ACT_SPLIT: the ssm_out activation leaves the gated tile on a fifo of its
    # own rather than in the tail of its output object, so the stream can drain
    # it with a plainly patched descriptor. The gated output's own drain needs
    # the bit31 patch form, and a window of what that writes is not readable
    # again later in the same dispatch; the conv drains, which are patched
    # plainly, are read back by the norm fills in the same stream every token.
    act_split = int(__import__("os").environ.get("GATED_ACT_SPLIT", "1"))
    # CONV_SLOT: floats of a feed slot the object actually carries. The full
    # 1024 is history (384), the token's qkv (128) and the conv weights (512),
    # and the weights do not change between tokens. Setting it to 512 streams
    # only the half that does - the kernel then multiplies by garbage, so this
    # is a diagnostic that prices the stage's bytes and nothing else.
    fslot = int(__import__("os").environ.get("CONV_SLOT", "0")) or FEED_N
    FEED_T = np.ndarray[(gpo * fslot,), np.dtype[np.float32]]
    X_T = np.ndarray[(gpo * S_V,), np.dtype[np.float32]]
    HIST_T = np.ndarray[(gpo * 3 * S_V,), np.dtype[np.float32]]
    HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
    PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]
    PKV_T = np.ndarray[(PKV_N,), np.dtype[np.float32]]
    PKV4_T = np.ndarray[(NC_GDN * PKV_N,), np.dtype[np.float32]]   # one gdn round
    ROWS_T = np.ndarray[(gdn_v.ROWS,), np.dtype[bfloat16]]
    ATT_T = np.ndarray[(gdn_v.CHUNK,), np.dtype[np.float32]]
    ATT4_T = np.ndarray[(NC_GDN * gdn_v.CHUNK,), np.dtype[np.float32]]

    # STAGE_STUB is a bitmask - 1 conv, 2 norm, 4 gdn - that empties a stage's
    # kernel so the stage's floor, the part that is data movement rather than
    # arithmetic, can be measured on its own.
    stage_stub = int(__import__("os").environ.get("STAGE_STUB", "0"))
    sflags = ["-O2", "-DNDEBUG", f"-DSTAGE_STUB={stage_stub}",
              f"-DGDN_NOREDUCE={int(__import__('os').environ.get('GDN_NOREDUCE', '0'))}"]

    def _sobj(name: str, src: str, flags: list) -> str:
        # Every kernel here is named for its flags as well as its source.
        # Without that the object compiled under an earlier flag set is reused
        # and the design runs the wrong kernel silently - which cost three
        # wrong conclusions about the gated stage before it was noticed.
        d = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
        return f"{name}_{d}.ll"

    conv_src = attn_cn._conv_src(gpo, fslot)
    norm_src = attn_cn._norm_src()
    # PKV_ONCHIP: the norm stage hands its chunks to the gdn cores through a
    # MemTile instead of writing them to DDR for the gdn fill to read back.
    # Two kernels because a gdn round is four of a head's eight chunks and an
    # ObjectFifo object is what a core releases at once.
    onchip = bool(int(__import__("os").environ.get("PKV_ONCHIP", "1")))
    # One norm core per gdn core, sitting directly above it, emitting exactly
    # the two chunks of each head that its gdn core takes - chunk gi and gi+4,
    # because a gdn round is NC_GDN consecutive chunks and a head is two
    # rounds. A core has two input DMA channels and the gdn core needs one for
    # its state, so this is the only shape that leaves it one for the chunks.
    norm_src_j = [[attn_cn._norm_src(f"normj{gi}_{half}", -1, half * NC_GDN + gi)
                   for half in range(2)] for gi in range(NC_GDN)]
    gdn_src  = gdn_v._kernel_src()
    conv_k = iron.ExternalFunction(name="convh", source_string=conv_src,
                                   arg_types=[FEED_T, X_T, HIST_T],
                                   object_file_name=_sobj("convh", conv_src, sflags),
                                   compile_flags=sflags, inline=True)
    norm_k = iron.ExternalFunction(name="normh", source_string=norm_src,
                                   arg_types=[HN_T, PKVB_T],
                                   object_file_name=_sobj("normh", norm_src, sflags),
                                   compile_flags=sflags, inline=True)
    norm_kj = [[iron.ExternalFunction(
                    name=f"normj{gi}_{half}", source_string=norm_src_j[gi][half],
                    arg_types=[HN_T, PKV_T],
                    object_file_name=_sobj(f"normj{gi}_{half}",
                                           norm_src_j[gi][half], sflags),
                    compile_flags=sflags, inline=True)
                for half in range(2)] for gi in range(NC_GDN)]
    gdn_k = iron.ExternalFunction(name="gdn_v", source_string=gdn_src,
                                  arg_types=[PKV_T, ROWS_T, ROWS_T, ATT_T],
                                  object_file_name=_sobj("gdn_v", gdn_src, sflags),
                                  compile_flags=sflags, inline=True)

    AZG_T = np.ndarray[(HG * AZG_N,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(ACT_OFF_G if act_split else OUTN,), np.dtype[np.uint8]]
    ACTB_T = np.ndarray[(ACTN,), np.dtype[np.uint8]]
    gflags = ["-O2", "-DNDEBUG", f"-DGATED_STUB={_gated_stub()}",
              f"-DHG={HG}", f"-DAZG_N={AZG_N}", f"-DATTN_ONCHIP={int(att_onchip)}",
              f"-DK_GATE={K_G}", f"-DACT_TILE={ACT_TILE_G}",
              f"-DACT_OFF={ACT_OFF_G}", f"-DACT_SPLIT={int(act_split)}",
              f"-DGATED_ACT={int(__import__('os').environ.get('GATED_ACT', '1'))}",
              f"-DGATED_FMT={GATED_FMT}"]

    def _obj(name: str, src: str) -> str:
        # The object file has to be named for the flags as well as the source.
        # Without that the compiled object of an earlier flag set is reused and
        # the design silently runs the wrong kernel - here that showed up as
        # only the first head of every object being written, which reads as a
        # slightly worse model rather than as an error.
        d = hashlib.sha256((src + "\0".join(gflags)).encode()).hexdigest()[:8]
        return f"{name}_{d}.ll"

    # The transition to the FFN: residual, norm, gamma and the quantized
    # activation the next projection reads, on a tile of its own inside this
    # design. Standalone the same tile runs fine (the xclbin's own sequence
    # completes over and over); a second context is what broke, so the tile
    # lives here, in the one context everything else uses.
    POST_D  = rec_post.D
    POST_NT = POST_D // (128 if GATED_FMT == 1 else 256)
    POSTI_T = np.ndarray[(3 * POST_D + 4,), np.dtype[np.float32]]
    POSTO_T = np.ndarray[(POST_D * 4 + (1 + POST_NT) * ACT_TILE_G,),
                         np.dtype[np.uint8]]
    post_src = rec_post._kernel_src(POST_D, 256, ACT_TILE_G)
    pflags = ["-O2", "-DNDEBUG", f"-DGATED_FMT={GATED_FMT}",
              f"-DPOST_STUB={int(__import__('os').environ.get('POST_STUB', '0'))}"]
    post_k = iron.ExternalFunction(
        name="post_norm", source_string=post_src,
        arg_types=[POSTI_T, POSTO_T],
        object_file_name=_sobj("post_norm", post_src, pflags),
        compile_flags=pflags, inline=True)

    h2_src = _gated_h2_src()
    fin_src = rec_gated._gated_src()
    kh = iron.ExternalFunction(name="gated_h2", source_string=h2_src,
                               arg_types=([OUT_T, AZG_T, ATT4_T, ATT4_T]
                                          if att_onchip else [OUT_T, AZG_T]),
                               object_file_name=_obj("gated_h2", h2_src),
                               compile_flags=gflags, inline=True)
    kf = iron.ExternalFunction(name="gated_fin", source_string=fin_src,
                               arg_types=([OUT_T, ACTB_T] if act_split
                                          else [OUT_T]),
                               object_file_name=_obj("gated_fin", fin_src),
                               compile_flags=gflags, inline=True)

    FEED_g = np.ndarray[(NC_CONV * CN_CONV * FEED_N,), np.dtype[np.float32]]
    X_g = np.ndarray[(N_VH * HEAD_NORM,), np.dtype[np.float32]]
    PKVB_g = np.ndarray[(N_VH * PKVB_N,), np.dtype[np.float32]]
    STATE_g = np.ndarray[(STATE_N,), np.dtype[bfloat16]]
    AZG_g = np.ndarray[(N_VH * AZG_N,), np.dtype[np.float32]]
    OUT_g = np.ndarray[(OUTN,), np.dtype[np.uint8]]

    workers = []
    rt_args = [FEED_g, X_g, PKVB_g, STATE_g, AZG_g, OUT_g]

    def group_base(col, g):
        return col * CN_CONV * S_V + g * S_V

    def head_region(ach):
        return (ach % 2048) // S_V, ach // 2048

    # ---- conv: cols 0..NC_CONV-1, row 2 ----
    # CONV_DEPTH: how many of a column's 24 group objects can be in flight.
    # The stage's time is neither its arithmetic (emptying the kernel changes
    # nothing) nor its bytes (295 KB is 23 us at the array's per-channel rate),
    # so what is left is the latency of a group's round trip - shim, MemTile,
    # core, back - and how many of those overlap is exactly this number.
    cdepth = int(__import__("os").environ.get("CONV_DEPTH", "2"))
    # The shim feeds the conv cores directly. Each of the stage's three fifos
    # has one producer and one consumer, so the forward and the single-offset
    # joins only moved every byte through the MemTile on its way between the
    # shim and the core; dropping them takes the stage from 245 us to 230 and
    # gives the MemTiles of the conv columns back. CONV_DIRECT=0 restores the
    # hop. It is not where the stage's time goes - it is 6% of it - but it is
    # the difference between this path and the GEMV's, whose cores the shim
    # also feeds directly and which reaches 4.3 GB/s a channel against a conv
    # column's 1.8.
    cdirect = bool(int(__import__("os").environ.get("CONV_DIRECT", "1")))
    # CONV_SPREAD=1 sends the x and history writes to columns the GEMV's
    # joined outputs freed - all of them idle while conv runs - instead of the
    # column the feed streams in on. A shim tile has one port for all of its
    # channels, which is what made the gdn block's second state stream
    # pointless, so this was the obvious next thing to try here. It is off
    # because it changes nothing at all: 202 us against 204, 37.15 t/s against
    # 37.27. Whatever holds this stage to about 1.2 GB/s where the tile does
    # 4.3, it is not the port either.
    cspread = bool(int(__import__("os").environ.get("CONV_SPREAD", "0")))
    CONV_X_COL = [5, 4] if cspread else list(range(NC_CONV))
    CONV_H_COL = [5, 7] if cspread else list(range(NC_CONV))
    for col in range(NC_CONV):
        f3 = ObjectFifo(FEED_T, name=f"agf3_{col}", depth=cdepth)
        x23 = ObjectFifo(X_T, name=f"agx23_{col}", depth=cdepth)
        h23 = ObjectFifo(HIST_T, name=f"agh23_{col}", depth=cdepth)
        if cdirect:
            f2, x12, h12 = f3, x23, h23
        else:
            f2 = f3.cons().forward(obj_type=FEED_T, name=f"agf2_{col}",
                                   tile=Tile(col, 1), depth=cdepth)
            x12 = x23.prod().join([0], obj_types=[X_T], names=[f"agx12_{col}"],
                                  depths=[cdepth], tile=Tile(col, 1))[0]
            h12 = h23.prod().join([0], obj_types=[HIST_T], names=[f"agh12_{col}"],
                                  depths=[cdepth], tile=Tile(col, 1))[0]

        def conv_fn(fc, xp, hc, k, n=CN_CONV // gpo):
            for _ in range_(n):
                f = fc.acquire(1)
                x = xp.acquire(1)
                ho = hc.acquire(1)
                k(f, x, ho)
                xp.release(1)
                hc.release(1)
                fc.release(1)

        workers.append(Worker(
            conv_fn, [f2.cons(), x12.prod(), h12.prod(), conv_k],
            tile=Tile(col, 2), stack_size=0xD00))
        rt_args += [f3.prod(tile=Tile(col, 0)),
                    x23.cons(tile=Tile(CONV_X_COL[col], 0)),
                    h23.cons(tile=Tile(CONV_H_COL[col], 0))]

    # CORE_MIN=1: conv stage only - the smallest useful design, for bisecting
    # the full-ELF load stall against PDI size.
    if int(__import__("os").environ.get("CORE_MIN", "0")):
        def seq_fn_min(FEED, X, PKVB, STATE, AZG, OUT, *fifos):
            it = iter(fifos)
            def nxt():
                return next(it)
            cfeed, cxdrain, chist = [], [], []
            for _col in range(NC_CONV):
                cfeed.append(nxt())
                cxdrain.append(nxt())
                chist.append(nxt())
            for col in range(NC_CONV):
                for g in range(0, CN_CONV, gpo):
                    ach = group_base(col, g)
                    h, r = head_region(ach)
                    gi = TaskGroup()
                    cfeed[col].fill(FEED, tap=TensorAccessPattern(
                        (NC_CONV * CN_CONV * FEED_N,),
                        offset=(col * CN_CONV + g) * FEED_N,
                        sizes=[gpo, fslot], strides=[FEED_N, 1]), group=gi)
                    gi.finish()
                    go = TaskGroup()
                    cxdrain[col].drain(X, tap=TensorAccessPattern(
                        (N_VH * HEAD_NORM,), offset=h * HEAD_NORM + r * S_V,
                        sizes=[gpo, S_V], strides=[HEAD_NORM, 1]), wait=True,
                        group=go)
                    chist[col].drain(FEED, tap=TensorAccessPattern(
                        (NC_CONV * CN_CONV * FEED_N,),
                        offset=(col * CN_CONV + g) * FEED_N + F_H,
                        sizes=[gpo, 3 * S_V], strides=[FEED_N, 1]), wait=True,
                        group=go)
                    go.finish()
        return workers, rt_args, seq_fn_min

    # ---- norm: cols NC_CONV..NC_CONV+NC_NORM-1, row 3 ----
    # One stream in and one out for the whole stage rather than one per column.
    # The heads interleave across the columns instead of splitting into blocks
    # - column ci takes heads ci, ci+NC_NORM, ... - so a round is NC_NORM
    # adjacent heads, which is contiguous in both the x input and the pkvb
    # output. The kernel does not care which head it is handed, and the gdn
    # stage still reads pkvb head-major.
    ncol0 = NC_CONV
    HN2_T   = np.ndarray[(NC_NORM * HEAD_NORM,), np.dtype[np.float32]]
    PKVB2_T = np.ndarray[(NC_NORM * PKVB_N,), np.dtype[np.float32]]

    n2 = None
    nb = None
    if onchip:
        # One head per object, broadcast to the four norm cores: each needs the
        # whole head to normalise q and k before emitting its own chunks.
        n3 = ObjectFifo(HN_T, name="agn3", depth=2)
        nb = [n3.cons() for _ in range(NC_GDN)]
    else:
        n3 = ObjectFifo(HN2_T, name="agn3", depth=2)
        n2 = n3.cons().split(offsets=[i * HEAD_NORM for i in range(NC_NORM)],
                             obj_types=[HN_T] * NC_NORM, depths=[2] * NC_NORM,
                             names=[f"agn2_{ncol0 + i}" for i in range(NC_NORM)],
                             tile=Tile(ncol0, 1))
    # On chip the stage is four cores, one over each gdn core, each taking the
    # whole head and emitting only the two chunks its neighbour consumes. The
    # x input becomes a broadcast rather than a split, which is the same shim
    # channel and one more MemTile read.
    pkv_on = []
    no23 = None
    if onchip:
        for gi in range(NC_GDN):
            pkv_on.append(ObjectFifo(PKV_T, name=f"agpk{gi}", depth=2))
    else:
        no23 = ObjectFifo(PKVB2_T, name="agno23", depth=2)
        no12 = no23.prod().join(offsets=[i * PKVB_N for i in range(NC_NORM)],
                                obj_types=[PKVB_T] * NC_NORM, depths=[2] * NC_NORM,
                                names=[f"agno12_{ncol0 + i}" for i in range(NC_NORM)],
                                tile=Tile(ncol0 + 1, 1))

    def norm_fn(nc, oc, k, nh=HP_NORM):
        for _ in range_(nh):
            ni = nc.acquire(1)
            oo = oc.acquire(1)
            k(ni, oo)
            oc.release(1)
            nc.release(1)

    def norm_fnj(nc, oc, k0, k1, nh=N_VH):
        # Every head, two chunks of each: the per-head normalisation is
        # repeated for the second chunk, which is 128 values against the round
        # trip through DDR it replaces.
        for _ in range_(nh):
            ni = nc.acquire(1)
            o0 = oc.acquire(1)
            k0(ni, o0)
            oc.release(1)
            o1 = oc.acquire(1)
            k1(ni, o1)
            oc.release(1)
            nc.release(1)

    if onchip:
        for gi in range(NC_GDN):
            workers.append(Worker(
                norm_fnj, [nb[gi], pkv_on[gi].prod(),
                           norm_kj[gi][0], norm_kj[gi][1]],
                tile=Tile(GDN_COL0 + gi, 3), stack_size=0xD00))
    else:
        for ci in range(NC_NORM):
            workers.append(Worker(
                norm_fn, [n2[ci].cons(), no12[ci].prod(), norm_k],
                tile=Tile(ncol0 + ci, 3), stack_size=0xD00))
    rt_args += [n3.prod(tile=Tile(ncol0, 0))]
    if not onchip:
        rt_args += [no23.cons(tile=Tile(ncol0 + 1, 0))]

    # ---- gdn: cols GDN_COL0..7, row 4 ----
    # One stream for all four gdn columns instead of one each. A shim tile has
    # two DMA channels in each direction, so four columns of independent
    # streams take eight of the array's sixteen - more than the whole design
    # can afford if anything else is to share the array. Four consecutive
    # chunks are contiguous in every buffer they touch (the chunk index maps
    # linearly onto (head, j), and four divides the eight chunks of a head), so
    # one object carries a round of four and a MemTile hands one chunk to each
    # column. Column gi still sees chunks gi, gi+4, ... in the same order it
    # did before.
    gcol0 = GDN_COL0
    PKV4_T  = np.ndarray[(NC_GDN * PKV_N,), np.dtype[np.float32]]
    ROWS4_T = np.ndarray[(NC_GDN * gdn_v.ROWS,), np.dtype[bfloat16]]

    # One stream per MemTile, not all four on one: a MemTile has six DMA
    # channels each way, and a four-way split or join needs five of them. So
    # each of the four streams sits on its own gdn column, which also decides
    # which column's shim carries it - the stream builder mirrors this.
    PKV_COL, STATE_COL, SOUT_COL, AZG_COL = (gcol0 + i for i in range(4))
    # The state is the block's whole cost: 512 KB in and 512 KB out a layer,
    # and one shim channel moves it in 119 us where the arithmetic hides
    # underneath. Two streams each way, each serving half the columns, halve
    # that; the channels come from joining the GEMV's outputs (fused_layer.py).
    # A round is still NC_GDN consecutive chunks, so a group's half of it is
    # contiguous in the state buffer.
    # Off by default, and now for a second reason. It used to place badly: a
    # shim tile has one port for all of its channels, so what a second stream
    # has to find is a light *tile*, and two channels of the same two tiles
    # gave 209 us against 177, the best placement left 36.4 t/s against 37.5.
    # The norm stage moving on chip freed four tiles that would have fixed
    # that, but SG=2 no longer runs at all: every placement tried (drains on
    # the attn and pkv columns, and on the two the norm stage freed) times out
    # in the gdn block, with pkv on chip and in DDR alike, at every batch
    # depth. The design places and routes; the split is what stops. Whatever
    # broke it arrived with something else, and the design's own validation
    # path cannot arbitrate - it imports ref_delta_layer, which is not in the
    # tree. Left wired up because the state is still the block's whole cost,
    # 512 KB each way a layer, and this is the only way to halve it.
    SG = int(__import__("os").environ.get("GDN_STATE_STREAMS", "1"))
    GG = NC_GDN // SG                      # gdn columns to a state stream
    ROWSG_T = np.ndarray[(GG * gdn_v.ROWS,), np.dtype[bfloat16]]
    # Second stream on the column whose shim still has a free channel in each
    # direction, and on the MemTiles the halved split and join leave room in.
    # A shim tile has one port for all of its channels, so the four streams go
    # to four tiles and no tile carries both directions. The norm stage moving
    # on chip freed the columns this needs.
    S_COLS = [STATE_COL, SOUT_COL][:SG]          # fill: cols 5 and 6
    O_COLS = ([SOUT_COL] if SG == 1 else [AZG_COL, PKV_COL])  # drain: 3, 4

    p3 = None
    if onchip:
        # Two producers, alternating two rounds each: head h is rounds 2h and
        # 2h+1, and the norm columns take even and odd heads.
        p2 = [[pkv_on[i]] for i in range(NC_GDN)]
    else:
        p3 = ObjectFifo(PKV4_T, name="agp3", depth=2)
        p2 = [[f] for f in p3.cons().split(
            offsets=[i * PKV_N for i in range(NC_GDN)],
            obj_types=[PKV_T] * NC_GDN, depths=[2] * NC_GDN,
            names=[f"agp2_{gcol0 + i}" for i in range(NC_GDN)],
            tile=Tile(PKV_COL, 1))]
    s3, s2 = [], []
    for g in range(SG):
        sf = ObjectFifo(ROWSG_T, name=f"ags3_{g}", depth=2)
        s2 += sf.cons().split(
            offsets=[i * gdn_v.ROWS for i in range(GG)],
            obj_types=[ROWS_T] * GG, depths=[2] * GG,
            names=[f"ags2_{g}_{i}" for i in range(GG)],
            tile=Tile(S_COLS[g], 1))
        s3.append(sf)
    o23, o12 = [], []
    for g in range(SG):
        of = ObjectFifo(ROWSG_T, name=f"ago23_{g}", depth=2)
        o12 += of.prod().join(
            offsets=[i * gdn_v.ROWS for i in range(GG)],
            obj_types=[ROWS_T] * GG, depths=[2] * GG,
            names=[f"ago12_{g}_{i}" for i in range(GG)],
            tile=Tile(O_COLS[g], 1))
        o23.append(of)
    # Deeper on chip: the gated tile holds two objects at once and the gdn
    # block must not stall behind it.
    a23 = ObjectFifo(ATT4_T, name="aga23",
                     depth=int(__import__("os").environ.get("ATT_DEPTH", "3"))
                     if att_onchip else 2)
    a12 = a23.prod().join(offsets=[i * gdn_v.CHUNK for i in range(NC_GDN)],
                          obj_types=[ATT_T] * NC_GDN, depths=[2] * NC_GDN,
                          names=[f"aga12_{gcol0 + i}" for i in range(NC_GDN)],
                          tile=Tile(AZG_COL, 1))

    def gdn_fn(pc, sc, oc, ac, k, nobj=N_OBJ_GDN):
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

    def gdn_fn2(pa, pb, sc, oc, ac, k, npair=N_OBJ_GDN // 4):
        # Two rounds from one norm column, then two from the other: that is the
        # order the rounds come in, because a head is two rounds and the
        # columns take alternate heads.
        for _ in range_(npair):
            for _ in range_(2):
                pp = pa.acquire(1)
                si = sc.acquire(1)
                so = oc.acquire(1)
                ao = ac.acquire(1)
                k(pp, si, so, ao)
                oc.release(1)
                ac.release(1)
                pa.release(1)
                sc.release(1)
            for _ in range_(2):
                pp = pb.acquire(1)
                si = sc.acquire(1)
                so = oc.acquire(1)
                ao = ac.acquire(1)
                k(pp, si, so, ao)
                oc.release(1)
                ac.release(1)
                pb.release(1)
                sc.release(1)

    for gi in range(NC_GDN):
        workers.append(Worker(
            gdn_fn, [p2[gi][0].cons(), s2[gi].cons(),
                     o12[gi].prod(), a12[gi].prod(), gdn_k],
            tile=Tile(gcol0 + gi, 4), stack_size=0x3000))
    if not onchip:
        rt_args += [p3.prod(tile=Tile(PKV_COL, 0))]
    rt_args += [s3[g].prod(tile=Tile(S_COLS[g], 0)) for g in range(SG)]
    rt_args += [o23[g].cons(tile=Tile(O_COLS[g], 0)) for g in range(SG)]
    if not att_onchip:
        rt_args += [a23.cons(tile=Tile(AZG_COL, 0))]

    # ---- gated: col GATED_COL (= first norm column), row 4 ----
    gcol = GATED_COL
    # The stage is one tile consuming its heads one after another, and what it
    # spends its time on is the handshake, not the arithmetic - stubbing both
    # of its kernels leaves it exactly as expensive. A deeper queue lets the
    # fill run ahead of the tile instead of taking turns with it.
    az_depth = int(__import__("os").environ.get("GATED_DEPTH", "2"))
    az3 = ObjectFifo(AZG_T, name="gaz3", depth=az_depth)
    az2 = az3.cons().forward(obj_type=AZG_T, name="gaz2", tile=Tile(gcol, 1),
                             depth=az_depth)
    gt23 = ObjectFifo(OUT_T, name="ggo23", depth=1)
    gt12 = gt23.prod().join([0], obj_types=[OUT_T], names=["ggo12"],
                            depths=[1], tile=Tile(gcol, 1))[0]
    ga23 = ga12 = None
    if act_split:
        ga23 = ObjectFifo(ACTB_T, name="gga23", depth=1)
        ga12 = ga23.prod().join([0], obj_types=[ACTB_T], names=["gga12"],
                                depths=[1], tile=Tile(gcol, 1))[0]

    def gated_fn(ac, oc, kh_, kf_, nh=N_VH // HG):
        o = oc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            kh_(o, a)
            ac.release(1)
        kf_(o)
        oc.release(1)

    def gated_fn_on_s(ac, tc, oc, qc, kh_, kf_, nh=N_VH // HG):
        # The activation leaves on its own fifo, so the epilogue writes two
        # objects and the stream drains them with two descriptors.
        o = oc.acquire(1)
        q = qc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            t = tc.acquire(2)
            kh_(o, a, t[0], t[1])
            tc.release(2)
            ac.release(1)
        kf_(o, q)
        qc.release(1)
        oc.release(1)

    def gated_fn_on(ac, tc, oc, kh_, kf_, nh=N_VH // HG):
        # A head is NC_GDN * 2 chunks and a gdn round is NC_GDN, so the attn
        # of one head is two objects. They are taken together rather than one
        # per call because the head's scale is a reduction over all 128.
        o = oc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            t = tc.acquire(2)
            kh_(o, a, t[0], t[1])
            tc.release(2)
            ac.release(1)
        kf_(o)
        oc.release(1)

    # GATED_DIAG: diagnostic worker variants that consume the gated tile's
    # inputs without calling the compute kernels (kf still runs so the
    # activation drain has data). 1 waits on both inputs, 2 waits on az only
    # and leaves attn unconsumed (gdn then blocks on its attn fifo - the
    # state drains stop, which is itself the answer). Whichever completes
    # names the input the full worker waits on forever.
    gdiag = int(__import__("os").environ.get("GATED_DIAG", "0"))

    def gated_fn_diag1(ac, tc, oc, qc, kh_, kf_, nh=N_VH // HG):
        # waits on both inputs, calls no compute kernel
        o = oc.acquire(1)
        q = qc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            t = tc.acquire(2)
            tc.release(2)
            ac.release(1)
        kf_(o, q)
        qc.release(1)
        oc.release(1)

    def gated_fn_diag2(ac, tc, oc, qc, kh_, kf_, nh=N_VH // HG):
        # waits on az only; attn stays unconsumed
        o = oc.acquire(1)
        q = qc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            ac.release(1)
        kf_(o, q)
        qc.release(1)
        oc.release(1)

    def gated_fn_diag3(ac, tc, oc, qc, kh_, kf_, nh=N_VH // HG):
        # consumes both inputs, calls no kernel at all
        o = oc.acquire(1)
        q = qc.acquire(1)
        for _ in range_(nh):
            a = ac.acquire(1)
            t = tc.acquire(2)
            tc.release(2)
            ac.release(1)
        qc.release(1)
        oc.release(1)

    if att_onchip and act_split and gdiag == 1:
        workers.append(Worker(gated_fn_diag1,
                              [az2.cons(), a23.cons(), gt12.prod(),
                               ga12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    elif att_onchip and act_split and gdiag == 2:
        workers.append(Worker(gated_fn_diag2,
                              [az2.cons(), a23.cons(), gt12.prod(),
                               ga12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    elif att_onchip and act_split and gdiag == 3:
        workers.append(Worker(gated_fn_diag3,
                              [az2.cons(), a23.cons(), gt12.prod(),
                               ga12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    elif att_onchip and act_split:
        workers.append(Worker(gated_fn_on_s,
                              [az2.cons(), a23.cons(), gt12.prod(),
                               ga12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    elif att_onchip:
        workers.append(Worker(gated_fn_on,
                              [az2.cons(), a23.cons(), gt12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    else:
        workers.append(Worker(gated_fn, [az2.cons(), gt12.prod(), kh, kf],
                              tile=Tile(gcol, 4), stack_size=0x3000))
    # The fill is on the attn column's shim, not this one: the GEMV's eight
    # weight streams want a tile each (gemv_q4.py) and this column's other
    # MM2S carries the norm x. The route crosses columns, which costs nothing
    # - it is 32 KB once a layer. GATED_AZ_COL overrides the fill column
    # (the activation drain must then stay off it too - see GACT_COL).
    az_col = int(__import__("os").environ.get("GATED_AZ_COL",
                                              str(AZG_COL)))
    rt_args += [az3.prod(tile=Tile(az_col, 0)),
                gt23.cons(tile=Tile(gcol, 0))]
    if act_split:
        # GACT_COL: the activation drain's shim column. Defaults to the attn
        # column (7) - what the C++ stream builder (gated_act_col) expects;
        # a different column must be mirrored there or the drain token never
        # arrives. The azg fill shares the column, so the builder keeps its
        # descriptor apart (fill bd 0 / drain bd 1).
        gact_col = int(__import__("os").environ.get("GACT_COL",
                                                    str(AZG_COL)))
        rt_args += [ga23.cons(tile=Tile(gact_col, 0))]

    # ---- post: col POST_COL, row 2 ----
    # POST_TILE=0 drops the in-design post tile entirely (worker, fifos and
    # its fill/drain) - the diagnostic switch for the full-ELF sequence,
    # which the C++ backend never exercised with this tile enabled (there the
    # transition runs as its own design).
    post_tile = bool(int(__import__("os").environ.get("POST_TILE", "1")))
    POST_COL = NC_CONV + 1
    pi3 = ObjectFifo(POSTI_T, name="pni3", depth=1)
    pi2 = pi3.cons().forward(obj_type=POSTI_T, name="pni2", depth=1)
    po23 = ObjectFifo(POSTO_T, name="pno23", depth=1)
    po12 = po23.prod().join([0], obj_types=[POSTO_T], names=["pno12"],
                            depths=[1])[0]

    def post_fn(ic, oc, k):
        for _ in range_(1):
            i = ic.acquire(1)
            o = oc.acquire(1)
            k(i, o)
            oc.release(1)
            ic.release(1)

    if post_tile:
        workers.append(Worker(post_fn, [pi2.cons(), po12.prod(), post_k],
                              stack_size=0x3000))
        # The endpoints are pinned: the fill on (1,0) MM2S ch1, the drain on
        # (2,0) S2MM ch1 - free in this design. The fused projection's first
        # weight stream drives (0,0) MM2S ch1, so the fill must not sit
        # there or the post dispatch after a fused dispatch never completes.
        # The host-built stream (xdna-rec.cpp POST_FILL_COL / POST_DRN_COL)
        # agrees with these. POST_FILL_COL=auto / POST_DRN_COL=auto restore
        # the placer's choice (the merged fused_layer design pins its own
        # endpoints).
        _pf = __import__("os").environ.get("POST_FILL_COL", "1")
        _pd = __import__("os").environ.get("POST_DRN_COL", "2")
        if _pf in ("", "auto") and _pd in ("", "auto"):
            rt_args += [pi3.prod(), po23.cons()]
        elif _pf in ("", "auto"):
            rt_args += [pi3.prod(), po23.cons(tile=Tile(int(_pd), 0))]
        elif _pd in ("", "auto"):
            rt_args += [pi3.prod(tile=Tile(int(_pf), 0)), po23.cons()]
        else:
            rt_args += [pi3.prod(tile=Tile(int(_pf), 0)),
                        po23.cons(tile=Tile(int(_pd), 0))]

    # IRON reference seq mirrors the python mode0 col-major order + gated phase.
    # CORE_STOP: end the sequence after phase N (1 conv, 2 norm, 3 azg fill,
    # 4 gdn rounds) - the phase-bisection switch for the full-ELF stall.
    stop_at = int(__import__("os").environ.get("CORE_STOP", "0"))

    def seq_fn(FEED, X, PKVB, STATE, AZG, OUT, *fifos):
        it = iter(fifos)

        def nxt():
            return next(it)

        cfeed = []
        cxdrain = []
        chist = []
        for _col in range(NC_CONV):
            cfeed.append(nxt())
            cxdrain.append(nxt())
            chist.append(nxt())
        nfill = nxt()
        ndrain = None if onchip else nxt()
        gfillp = None if onchip else nxt()
        gfills = [nxt() for _ in range(SG)]
        gdrain = [nxt() for _ in range(SG)]
        gattn = None if att_onchip else nxt()
        gzfill = nxt()
        gout = nxt()
        gact = nxt() if act_split else None
        pfill = nxt() if post_tile else None
        pdrain = nxt() if post_tile else None


        # conv cols col-major, a round being gpo consecutive feed groups: they
        # are contiguous in the feed buffer and land on gpo consecutive heads
        # of the same region in x.
        for col in range(NC_CONV):
            for g in range(0, CN_CONV, gpo):
                ach = group_base(col, g)
                h, r = head_region(ach)
                gi = TaskGroup()
                cfeed[col].fill(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN_CONV * FEED_N,),
                    offset=(col * CN_CONV + g) * FEED_N,
                    sizes=[gpo, fslot], strides=[FEED_N, 1]), group=gi)
                gi.finish()
                go = TaskGroup()
                cxdrain[col].drain(X, tap=TensorAccessPattern(
                    (N_VH * HEAD_NORM,), offset=h * HEAD_NORM + r * S_V,
                    sizes=[gpo, S_V], strides=[HEAD_NORM, 1]), wait=True, group=go)
                chist[col].drain(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN_CONV * FEED_N,),
                    offset=(col * CN_CONV + g) * FEED_N + F_H,
                    sizes=[gpo, 3 * S_V], strides=[FEED_N, 1]), wait=True, group=go)
                go.finish()
        if stop_at == 1:
            return

        # norm: a round of NC_NORM adjacent heads, one per column - or one head
        # broadcast to every core when the chunks stay on chip.
        for h in range(N_VH if onchip else HP_NORM):
            gi = TaskGroup()
            nfill.fill(X, tap=TensorAccessPattern(
                (N_VH * HEAD_NORM,),
                offset=h * (HEAD_NORM if onchip else NC_NORM * HEAD_NORM),
                sizes=[HEAD_NORM if onchip else NC_NORM * HEAD_NORM],
                strides=[1]), group=gi)
            gi.finish()
            if ndrain is not None:
                go = TaskGroup()
                ndrain.drain(PKVB, tap=TensorAccessPattern(
                    (N_VH * PKVB_N,), offset=h * NC_NORM * PKVB_N,
                    sizes=[NC_NORM * PKVB_N], strides=[1]), wait=True, group=go)
                go.finish()
        if stop_at == 2:
            return

        # gated phase: the z, gamma and head index of every head. With attn
        # coming over the array this fill has to be in flight *before* the gdn
        # rounds, not after them - the gated tile consumes a head's attn as
        # the block produces it, and would otherwise sit on the first head
        # while the attn fifo backs up into the gdn cores.
        tg = TaskGroup()
        gzfill.fill(AZG, tap=TensorAccessPattern(
            (N_VH * AZG_N,), offset=0, sizes=[N_VH * AZG_N], strides=[1]),
            group=tg)
        tg.finish()
        if stop_at == 3:
            return

        # gdn: a round of NC_GDN consecutive chunks per transfer, which the
        # MemTile hands out one chunk per column. Chunk gi + r*NC_GDN still
        # lands on column gi, exactly as when each column had its own stream.
        for r in range(N_VH * N_OBJ // NC_GDN):
            c0 = r * NC_GDN
            head = c0 // N_OBJ
            j = c0 % N_OBJ
            gi2 = TaskGroup()
            if gfillp is not None:
                gfillp.fill(PKVB, tap=TensorAccessPattern(
                    (N_VH * PKVB_N,), offset=head * PKVB_N + j * PKV_N,
                    sizes=[NC_GDN * PKV_N], strides=[1]), group=gi2)
            for g in range(SG):
                gfills[g].fill(STATE, tap=TensorAccessPattern(
                    (STATE_N,), offset=(c0 + g * GG) * gdn_v.ROWS,
                    sizes=[GG * gdn_v.ROWS], strides=[1]), group=gi2)
            gi2.finish()
            go = TaskGroup()
            for g in range(SG):
                gdrain[g].drain(STATE, tap=TensorAccessPattern(
                    (STATE_N,), offset=(c0 + g * GG) * gdn_v.ROWS,
                    sizes=[GG * gdn_v.ROWS], strides=[1]), wait=True, group=go)
            if gattn is not None:
                gattn.drain(AZG, tap=TensorAccessPattern(
                    (N_VH * AZG_N,), offset=head * AZG_N + j * gdn_v.CHUNK,
                    sizes=[NC_GDN * gdn_v.CHUNK], strides=[1]), wait=True,
                    group=go)
            go.finish()
        if stop_at == 4:
            return

        # The post tile's own transfers, filling and draining windows of the
        # gated output buffer the backend points them at.
        if pfill is not None:
            tp = TaskGroup()
            pfill.fill(OUT, tap=TensorAccessPattern(
                (OUTN,), offset=0, sizes=[3 * POST_D + 4], strides=[1]),
                group=tp)
            tp.finish()
        if pdrain is not None:
            gp = TaskGroup()
            pdrain.drain(OUT, tap=TensorAccessPattern(
                (OUTN,), offset=0,
                sizes=[POST_D + (1 + POST_NT) * ACT_TILE_G // 4], strides=[1]),
                wait=True, group=gp)
            gp.finish()

        go = TaskGroup()
        gout.drain(OUT, tap=TensorAccessPattern(
            (OUTN,), offset=0,
            sizes=[ACT_OFF_G if act_split else OUTN], strides=[1]),
            wait=True, group=go)
        if gact is not None:
            gact.drain(OUT, tap=TensorAccessPattern(
                (OUTN,), offset=ACT_OFF_G, sizes=[ACTN], strides=[1]),
                wait=True, group=go)
        go.finish()

    return workers, rt_args, seq_fn


@iron.jit
def attn_gdn_gated(*, dev_name: CompileTime[str] = "npu2"):
    workers, rt_args, seq_fn = build_core(dev_name)
    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=GDN_COL0 + NC_GDN), rt, workers)
    # IRON_TRACE: route hardware tracing to a DDR buffer (appended as a
    # runtime argument) - the diagnostic for which wait of the full-ELF
    # sequence never satisfies.
    if int(__import__("os").environ.get("IRON_TRACE", "0")):
        # IRON_TRACE_ONE: trace a single worker (index into `workers`) -
        # the full trace overlay of all workers does not route on this
        # array (the pathfinder's mastersets collide with the design's
        # stream-switch connections), and one flow at a time is enough to
        # find which wait never satisfies.
        one = __import__("os").environ.get("IRON_TRACE_ONE")
        traced = [workers[int(one)]] if one is not None else list(workers)
        for w in traced:
            if w.trace is None:
                w.trace = 1
        # The trace egress needs a shim DMA of its own: column 0 carries the
        # conv fills, 4-7 the gdn transfers, so 2 (a norm column, no shim
        # endpoints) is the free one.
        egress = int(__import__("os").environ.get("IRON_TRACE_COL", "2"))
        # IRON_TRACE_STALLS: trace only the three stall events - which of
        # them fires every cycle tells what a stalled worker waits on
        # (memory / stream / lock).
        stalls = bool(int(__import__("os").environ.get(
            "IRON_TRACE_STALLS", "0")))
        stall_events = None
        if stalls:
            from aie.dialects.aie import CoreEventAIE2P
            stall_events = [CoreEventAIE2P.MEMORY_STALL,
                            CoreEventAIE2P.STREAM_STALL,
                            CoreEventAIE2P.LOCK_STALL]
        prog.enable_trace(
            trace_size=262144, workers=traced, egress_shim_col=egress,
            coretile_events=stall_events)
    return prog.resolve_program()


# ---- TXN builder (mirror of the C++ xdna-seq-attn.cpp fused schedule) -------

SZ_W = {0x00: 6, 0x01: 12, 0x03: 7, 0x80: 4, 0x81: 12}
SHIM_BD_BASE = 0x1D000
SHIM_PUSHQ_BASE = 0x1D204
SHIM_TOKEN_BASE = 0x1D200
NCOL_MERGED = GDN_COL0 + NC_GDN   # 8


def tile_reg(col, row, reg):
    return ((col & 0x7F) << 25) | ((row & 0x1F) << 20) | (reg & 0xFFFFF)


class Txn:
    def __init__(self, n_cols):
        self.n_cols = n_cols
        self.ops = []

    def write(self, col, row, reg, val):
        self.ops += [0x00, 0, tile_reg(col, row, reg), 0, val, SZ_W[0x00] * 4]

    def blockwrite(self, col, row, bd, blen, boff, d0, d1, d2, it):
        self.ops += [0x01, 0, tile_reg(col, row, SHIM_BD_BASE + bd * 0x20),
                     SZ_W[0x01] * 4, blen, boff, 0, d0, d1, d2, it,
                     (1 << 25)]

    def ddr_patch(self, col, row, bd, arg_idx, arg_off):
        self.ops += [0x81, SZ_W[0x81] * 4, 0, 0, 0, 0,
                     tile_reg(col, row, SHIM_BD_BASE + bd * 0x20 + 4), 0,
                     arg_idx, 0, arg_off, 0]

    def maskwrite(self, col, row, reg, val, mask):
        self.ops += [0x03, 0, tile_reg(col, row, reg), 0, val, mask,
                     SZ_W[0x03] * 4]

    def push_queue(self, col, row, bd, dma_dir, channel, issue_token, repeat=0):
        reg = SHIM_PUSHQ_BASE + (0x8 if channel > 0 else 0) + \
              (0x10 if dma_dir else 0)
        val = bd | (repeat << 16)
        if issue_token:
            val |= 0x80000000
        self.write(col, row, reg, val)

    def issue_token(self, col, row, dma_dir, channel):
        reg = SHIM_TOKEN_BASE + channel * 0x8 + (0x10 if dma_dir else 0)
        self.maskwrite(col, row, reg, 0xF << 8, 0x1F00)

    def wait_token(self, col, row, dma_dir, channel):
        self.ops += [0x80, SZ_W[0x80] * 4,
                     (row & 0xFF) << 8 | (col & 0xFF) << 16 |
                     (1 if dma_dir else 0),
                     (channel & 0xFF) << 24 | 0x00010100]

    def build(self):
        ops = self.ops
        ninstr = 0
        i = 0
        while i < len(ops):
            op = ops[i]
            if op in SZ_W:
                ninstr += 1
                i += SZ_W[op]
            else:
                raise SystemExit(f"attn_gdn_gated: malformed op stream at word {i}")
        out = [(6 << 24) | (4 << 16) | (1 << 8) | 0,
               self.n_cols | (1 << 8),
               ninstr,
               (4 + len(ops)) * 4]
        out += ops
        return np.array(out, dtype=np.uint32)


# Per-fifo endpoint -> shim (dma_dir, channel, bd_id, arg) maps matching the
# IRON-compiled attn_gdn_gated columns (verified against the decoded stream):
# conv cols 0-1 (MM2S ch0 feed, S2MM ch1 x, S2MM ch0 hist), norm cols 2-3
# (MM2S ch0 x, S2MM ch0 pkvb), gdn cols 4-7 (MM2S ch0 pkv, MM2S ch1 state,
# S2MM ch1 state, S2MM ch0 attn->azg) and the gated tile on col 2 (MM2S ch1
# azg fill, S2MM ch1 out drain). MM2S = 1 (host -> device), S2MM = 0.
EP_CV_FEED = {"dir": 1, "ch": 0, "bd": 0, "arg": 0}
EP_CV_X    = {"dir": 0, "ch": 1, "bd": 0, "arg": 1}
EP_CV_HIST = {"dir": 0, "ch": 0, "bd": 1, "arg": 0}
EP_NM_X    = {"dir": 1, "ch": 0, "bd": 0, "arg": 1}
EP_NM_PKVB = {"dir": 0, "ch": 0, "bd": 0, "arg": 2}
EP_GD_PKV  = {"dir": 1, "ch": 0, "bd": 0, "arg": 2}
EP_GD_STATE= {"dir": 1, "ch": 1, "bd": 1, "arg": 3}
EP_GD_OUT  = {"dir": 0, "ch": 1, "bd": 0, "arg": 3}
EP_GD_AZG  = {"dir": 0, "ch": 0, "bd": 1, "arg": 4}
EP_GT_AZG  = {"dir": 1, "ch": 1, "bd": 0, "arg": 4}
EP_GT_OUT  = {"dir": 0, "ch": 1, "bd": 0, "arg": 5}


def ga_head_region(ga):
    return ga % 16, ga // 16


def build_attn_gdn_gated_stream(mode=4):
    """Per-token TXN stream for attn_gdn_gated.xclbin (6 BOs: arg0 feed, arg1
    x, arg2 pkvb, arg3 state bf16, arg4 azg, arg5 out). Byte offsets:
      conv/hist/norm identical to attn_gdn_txn;
      gdn azg chunk c (head,j): attn lanes at (head*AZG_N + j*CHUNK)*4;
      gated: azg heads filled from arg4 (attn lanes written by gdn above),
      out drained to arg5.
    Phases run in time: conv cols0-1, norm cols2-3, gdn cols4-7, gated col2.
    mode 0: col-major per phase; 1: slot-major; 2/4: phased groups."""
    t = Txn(NCOL_MERGED)

    def conv_fill(col, ga):
        e = EP_CV_FEED
        t.blockwrite(col, 0, e["bd"], FEED_N, ga * FEED_N * 4, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(col, 0, e["bd"], e["arg"], ga * FEED_N * 4)
        t.push_queue(col, 0, e["bd"], e["dir"], e["ch"], False)

    def conv_drain_issue(col, ga):
        x = EP_CV_X
        h, r = ga_head_region(ga)
        xoff = (h * HEAD_NORM + r * S_V) * 4
        t.blockwrite(col, 0, x["bd"], S_V, xoff, 0, 0xC0000000, 0x2000000, 0)
        t.ddr_patch(col, 0, x["bd"], x["arg"], xoff)
        t.issue_token(col, 0, x["dir"], x["ch"])
        t.push_queue(col, 0, x["bd"], x["dir"], x["ch"], True)
        hh = EP_CV_HIST
        t.blockwrite(col, 0, hh["bd"], 3 * S_V, ga * FEED_N * 4, 0,
                     0xC0000000, 0x2000000, 0)
        t.ddr_patch(col, 0, hh["bd"], hh["arg"], ga * FEED_N * 4)
        t.issue_token(col, 0, hh["dir"], hh["ch"])
        t.push_queue(col, 0, hh["bd"], hh["dir"], hh["ch"], True)

    def conv_wait(col):
        t.wait_token(col, 0, EP_CV_X["dir"], EP_CV_X["ch"])
        t.wait_token(col, 0, EP_CV_HIST["dir"], EP_CV_HIST["ch"])

    def norm_fill(col, head):
        e = EP_NM_X
        t.blockwrite(col, 0, e["bd"], HEAD_NORM, head * HEAD_NORM * 4, 0,
                     0xC0000000, 0x2000000, 0)
        t.ddr_patch(col, 0, e["bd"], e["arg"], head * HEAD_NORM * 4)
        t.push_queue(col, 0, e["bd"], e["dir"], e["ch"], False)

    def norm_drain_issue(col, head):
        e = EP_NM_PKVB
        t.blockwrite(col, 0, e["bd"], PKVB_N, head * PKVB_N * 4, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(col, 0, e["bd"], e["arg"], head * PKVB_N * 4)
        t.issue_token(col, 0, e["dir"], e["ch"])
        t.push_queue(col, 0, e["bd"], e["dir"], e["ch"], True)

    def norm_wait(col):
        t.wait_token(col, 0, EP_NM_PKVB["dir"], EP_NM_PKVB["ch"])

    def gdn_fill(col, c):
        e = EP_GD_PKV
        t.blockwrite(col, 0, e["bd"], PKV_N, c * PKV_N * 4, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(col, 0, e["bd"], e["arg"], c * PKV_N * 4)
        t.push_queue(col, 0, e["bd"], e["dir"], e["ch"], False)
        st = EP_GD_STATE
        t.blockwrite(col, 0, st["bd"], gdn_v.ROWS // 2, c * gdn_v.ROWS * 2, 0,
                     0xC0000000, 0x2000000, 0)
        t.ddr_patch(col, 0, st["bd"], st["arg"], c * gdn_v.ROWS * 2)
        t.push_queue(col, 0, st["bd"], st["dir"], st["ch"], False)

    def gdn_drain(col, c):
        ou = EP_GD_OUT
        t.blockwrite(col, 0, ou["bd"], gdn_v.ROWS // 2, c * gdn_v.ROWS * 2, 0,
                     0xC0000000, 0x2000000, 0)
        t.ddr_patch(col, 0, ou["bd"], ou["arg"], c * gdn_v.ROWS * 2)
        t.issue_token(col, 0, ou["dir"], ou["ch"])
        t.push_queue(col, 0, ou["bd"], ou["dir"], ou["ch"], True)
        az = EP_GD_AZG
        head = c // N_OBJ
        j = c % N_OBJ
        aoff = (head * AZG_N + j * gdn_v.CHUNK) * 4
        t.blockwrite(col, 0, az["bd"], gdn_v.CHUNK, aoff, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(col, 0, az["bd"], az["arg"], aoff)
        t.issue_token(col, 0, az["dir"], az["ch"])
        t.push_queue(col, 0, az["bd"], az["dir"], az["ch"], True)
        t.wait_token(col, 0, ou["dir"], ou["ch"])
        t.wait_token(col, 0, az["dir"], az["ch"])

    def gated_phase():
        g = EP_GT_AZG
        t.blockwrite(GATED_COL, 0, g["bd"], N_VH * AZG_N, 0, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(GATED_COL, 0, g["bd"], g["arg"], 0)
        t.push_queue(GATED_COL, 0, g["bd"], g["dir"], g["ch"], False)
        o = EP_GT_OUT
        t.blockwrite(GATED_COL, 0, o["bd"], OUTN // 4, 0, 0, 0xC0000000,
                     0x2000000, 0)
        # arg5 out: IRON patches the drain address with bit31 set; without it
        # the firmware drains the wrong target and the out BO stays zero.
        t.ddr_patch(GATED_COL, 0, o["bd"], o["arg"], 0x80000000)
        t.issue_token(GATED_COL, 0, o["dir"], o["ch"])
        t.push_queue(GATED_COL, 0, o["bd"], o["dir"], o["ch"], True)
        t.wait_token(GATED_COL, 0, o["dir"], o["ch"])

    gp = NC_CONV * CN_CONV // NC_CONV  # 24 conv groups per conv col
    hp = N_VH // NC_NORM          # 8 heads per norm col
    slots = N_VH * N_OBJ // NC_GDN  # 32 chunk objects per gdn col
    group = 1 if mode == 1 else 2

    # conv phase (cols 0..1)
    if mode == 0:
        for col in range(NC_CONV):
            for s in range(gp):
                t.blockwrite(col, 0, EP_CV_FEED["bd"], FEED_N,
                             (col * gp + s) * FEED_N * 4, 0, 0xC0000000,
                             0x2000000, 0)
                t.ddr_patch(col, 0, EP_CV_FEED["bd"], EP_CV_FEED["arg"],
                            (col * gp + s) * FEED_N * 4)
                t.push_queue(col, 0, EP_CV_FEED["bd"], 1, 0, False)
                conv_drain_issue(col, col * gp + s)
                conv_wait(col)
    else:
        for g0 in range(0, gp, group):
            for s in range(g0, min(g0 + group, gp)):
                for col in range(NC_CONV):
                    conv_fill(col, col * gp + s)
            for s in range(g0, min(g0 + group, gp)):
                for col in range(NC_CONV):
                    conv_drain_issue(col, col * gp + s)
            for s in range(g0, min(g0 + group, gp)):
                for col in range(NC_CONV):
                    conv_wait(col)

    # norm phase (cols 2..3)
    if mode == 0:
        for j in range(NC_NORM):
            for hs in range(hp):
                t.blockwrite(NC_CONV + j, 0, EP_NM_X["bd"], HEAD_NORM,
                             (j * hp + hs) * HEAD_NORM * 4, 0, 0xC0000000,
                             0x2000000, 0)
                t.ddr_patch(NC_CONV + j, 0, EP_NM_X["bd"], EP_NM_X["arg"],
                            (j * hp + hs) * HEAD_NORM * 4)
                t.push_queue(NC_CONV + j, 0, EP_NM_X["bd"], 1, 0, False)
                norm_drain_issue(NC_CONV + j, j * hp + hs)
                norm_wait(NC_CONV + j)
    else:
        for h0 in range(0, hp, group):
            for hs in range(h0, min(h0 + group, hp)):
                for j in range(NC_NORM):
                    norm_fill(NC_CONV + j, j * hp + hs)
            for hs in range(h0, min(h0 + group, hp)):
                for j in range(NC_NORM):
                    norm_drain_issue(NC_CONV + j, j * hp + hs)
            for hs in range(h0, min(h0 + group, hp)):
                for j in range(NC_NORM):
                    norm_wait(NC_CONV + j)

    # gdn phase (cols 4..7)
    if mode == 0:
        for gi in range(NC_GDN):
            for s in range(slots):
                gdn_fill(GDN_COL0 + gi, gi + s * NC_GDN)
                gdn_drain(GDN_COL0 + gi, gi + s * NC_GDN)
    else:
        for g0 in range(0, slots, group):
            for s in range(g0, min(g0 + group, slots)):
                for gi in range(NC_GDN):
                    gdn_fill(GDN_COL0 + gi, gi + s * NC_GDN)
            for s in range(g0, min(g0 + group, slots)):
                for gi in range(NC_GDN):
                    gdn_drain(GDN_COL0 + gi, gi + s * NC_GDN)

    # gated phase (col 2), after every gdn drain waited
    gated_phase()
    return t.build()


def main():
    ap = argparse.ArgumentParser(prog="attn_gdn_gated")
    add_compile_args(ap)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ntok", type=int, default=3)
    ap.add_argument("--model",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-bf16.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.tag is None:
        opts.tag = time.strftime("agg_%H%M%S")
    wd = os.path.join(opts.workdir, opts.tag)
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "attn_gdn_gated")
    t0 = time.time()
    spec = attn_gdn_gated.specialize(dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(xclbin_path=base + ".xclbin",
                                           inst_path=base + ".insts.bin")
    print(f"compiled {xclbin_path} ({time.time()-t0:.0f}s)")
    design_tag.stamp(xclbin_path, insts_path, opts.dev or "")
    print("xclbin", xclbin_path)
    print(f"tag={opts.tag}")
    if not opts.run:
        return
    run_validation(opts, xclbin_path)


def run_validation(opts, xclbin_path):
    """Standalone device check of the fused kernel vs the fp64 scalar layer
    trajectory (gdn_v/check_real semantics).  Drives attn_gdn_gated.xclbin with
    the host-built mode-2 stream, uploads feed/x/azg like xdna-rec.cpp, and for
    each token diffs (a) the azg attn lanes against the scalar gdn attn and
    (b) the device aq/d_a against rec_gated.host_gated(attn_scalar, z, gamma)."""
    import pyxrt as xrt
    from ml_dtypes import bfloat16

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "/home/asherstnev/Work/qvac-fabric-llm.cpp/ggml/src/"
                       "ggml-xdna/kernels")
    from ref_delta_layer import N_KH, QKVD, D, load_layer, rms_norm, silu
    from check_real import load_reader, load_embed
    from rec_gated import host_gated, N_VH as _NVR

    if os.environ.get("AGTX_IRON_WORDS"):
        ip = Path(str(xclbin_path)).with_suffix(".insts.bin")
        if os.environ.get("AGTX_IRON_FILE"):
            ip = Path(os.environ["AGTX_IRON_FILE"])
        words = np.frombuffer(ip.read_bytes(), dtype=np.uint32)
        print(f"using IRON words ({len(words)})")
    else:
        words = build_attn_gdn_gated_stream(mode=2)
        print(f"using host words ({len(words)})")

    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(ctx, xb.get_kernels()[0].get_name())
    insts_bo = xrt.bo(dev, words.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    np.frombuffer(insts_bo.map(), dtype=np.uint32)[:] = words
    insts_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def mk_bo(arr, ro=False):
        bo = xrt.bo(dev, arr.nbytes, xrt.bo.host_only, 0)
        if not ro:
            np.frombuffer(bo.map(), dtype=np.uint8)[:] = \
                np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    reader = load_reader(opts.model)
    W = load_layer(reader, 0)
    scale = np.float32(1.0 / np.sqrt(S_V))
    gamma = W["ssm_norm"].astype(np.float64)

    tokens = list(range(5, 5 + opts.ntok))
    if opts.real:
        h_in = [load_embed(reader, tk).astype(np.float32) for tk in tokens]
    else:
        rng = np.random.default_rng(3)
        h_in = [rng.standard_normal(D).astype(np.float32) for _ in tokens]

    def proj(h):
        cur = rms_norm(h, W["attn_norm"])
        qkv = cur @ W["wqkv"]
        z = cur @ W["z_gate"]
        alpha = cur @ W["alpha"]
        beta_raw = cur @ W["beta"]
        gate = np.log1p(np.exp(-np.abs(alpha + W["dt"]))) + \
            np.maximum(alpha + W["dt"], 0.0)
        gate = gate * W["a"]
        beta = 1.0 / (1.0 + np.exp(-beta_raw))
        return qkv, z, np.exp(gate).astype(np.float32), beta.astype(np.float32)

    Wcv = W["conv"]

    def block_chan_base(b):
        return (b // CN_CONV) * (CN_CONV * S_V) + (b % CN_CONV) * S_V

    feed0 = np.zeros(NC_CONV * CN_CONV * FEED_N, dtype=np.float32)
    for b in range(NC_CONV * CN_CONV):
        for ch in range(S_V):
            ach = block_chan_base(b) + ch
            feed0[b * FEED_N + F_W + 4 * ch: b * FEED_N + F_W + 4 * ch + 4] = \
                Wcv[:, ach]

    feed_bo = mk_bo(feed0)
    x_bo = mk_bo(np.zeros(N_VH * HEAD_NORM, dtype=np.float32))
    pkvb_bo = mk_bo(np.zeros(N_VH * PKVB_N, dtype=np.float32))
    state_bo = mk_bo(np.zeros(STATE_N, dtype=bfloat16))
    azg = np.zeros(N_VH * AZG_N, dtype=np.float32)
    for h in range(N_VH):
        azg[h * AZG_N + 2 * S_V:h * AZG_N + 3 * S_V] = gamma[:S_V]
        azg[h * AZG_N + 3 * S_V] = float(h)
    azg_bo = mk_bo(azg, ro=True)
    out_bo = mk_bo(np.zeros(OUTN, dtype=np.uint8), ro=True)

    hist = np.zeros((3, QKVD), dtype=np.float32)
    feed = feed0.copy()

    def step_dev(qkv, z, eg, beta_s):
        nonlocal feed, hist
        for b in range(NC_CONV * CN_CONV):
            ach0 = block_chan_base(b)
            hslice = feed[b * FEED_N + F_H: b * FEED_N + F_H + 3 * S_V]
            for tap in range(3):
                hslice[tap::3] = hist[tap, ach0:ach0 + S_V]
            feed[b * FEED_N + F_Q: b * FEED_N + F_Q + S_V] = \
                qkv[ach0:ach0 + S_V]
        np.frombuffer(feed_bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(feed).view(np.uint8).reshape(-1)
        feed_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        X = np.zeros(N_VH * HEAD_NORM, dtype=np.float32)
        for hh in range(N_VH):
            X[hh * HEAD_NORM + 3 * S_V:(hh + 1) * HEAD_NORM] = \
                [eg[hh], beta_s[hh], scale]
        np.frombuffer(x_bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(X).view(np.uint8).reshape(-1)
        x_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for h in range(N_VH):
            azg[h * AZG_N + S_V:h * AZG_N + 2 * S_V] = \
                z[h * S_V:(h + 1) * S_V].astype(np.float32)
        np.frombuffer(azg_bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(azg).view(np.uint8).reshape(-1)
        azg_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = kernel(3, insts_bo, int(words.nbytes), feed_bo, x_bo, pkvb_bo,
                     state_bo, azg_bo, out_bo)
        if run.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit("attn_gdn_gated run failed")
        conv_in = np.concatenate([hist, qkv[None, :]], axis=0)
        hist = conv_in[1:, :].copy()

    from gdn_v import scalar_chunk
    S_ref = np.zeros((N_VH, S_V, S_V), dtype=np.float32)
    print(f"== attn_gdn_gated: {len(tokens)} tokens layer 0 (tokens {tokens}) ==")
    for t in range(len(tokens)):
        qkv, z, eg, beta_s = proj(h_in[t])
        conv_in = np.concatenate([hist, qkv[None, :]], axis=0)
        step_dev(qkv, z, eg, beta_s)

        # scalar reference: host conv (same history) + fp32 gdn recursion
        x = silu(np.sum(Wcv * conv_in, axis=0).astype(np.float32))
        qf = x[0:2048].reshape(N_KH, S_V)
        kf = x[2048:4096].reshape(N_KH, S_V)
        vf = x[4096:6144].reshape(N_VH, S_V)
        qn = np.stack([q / np.linalg.norm(q) for q in qf])
        kn = np.stack([k / np.linalg.norm(k) for k in kf])
        attn_ref = np.zeros((N_VH, S_V), dtype=np.float32)
        A_new = np.zeros_like(S_ref)
        for h in range(N_VH):
            p = np.zeros(PKV_N, dtype=np.float32)
            p[0:S_V] = kn[h]
            p[S_V:2 * S_V] = qn[h]
            p[3 * S_V:3 * S_V + 3] = [eg[h], beta_s[h], scale]
            for j in range(N_OBJ):
                pp = p.copy()
                pp[2 * S_V:2 * S_V + gdn_v.CHUNK] = vf[h][j * gdn_v.CHUNK:(j + 1) * gdn_v.CHUNK]
                sout, attn = scalar_chunk(
                    pp, S_ref[h, j * gdn_v.CHUNK:(j + 1) * gdn_v.CHUNK])
                A_new[h, j * gdn_v.CHUNK:(j + 1) * gdn_v.CHUNK] = sout
                attn_ref[h, j * gdn_v.CHUNK:(j + 1) * gdn_v.CHUNK] = attn
        S_ref = A_new

        # device readbacks
        azg_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        azg_dev = np.frombuffer(azg_bo.map(), dtype=np.float32)[:N_VH * AZG_N]
        attn_dev = np.zeros(N_VH * S_V, dtype=np.float32)
        for h in range(N_VH):
            attn_dev[h * S_V:(h + 1) * S_V] = \
                azg_dev[h * AZG_N:h * AZG_N + S_V]
        out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        ob = np.frombuffer(out_bo.map(), dtype=np.uint8)[:]
        aq_dev = ob[K_G * 4:K_G * 4 + K_G].copy().view(np.int8)
        da_dev = np.frombuffer(ob[K_G * 4 + K_G:K_G * 4 + K_G + 4].copy(),
                               dtype=np.float32)[0]

        attn_sc = attn_ref.reshape(-1)
        ea = np.abs(attn_dev - attn_sc).max()
        _, aq_ref, da_ref = host_gated(attn_sc.astype(np.float64),
                                       z.astype(np.float64), gamma)
        nmis = int(np.count_nonzero(aq_dev != aq_ref))
        scratch = np.frombuffer(ob[:K_G * 4].copy(), dtype=np.float32)
        print(f"t={t}: attn max_abs vs scalar = {ea:.3e}   "
              f"d_a dev={da_dev:.6e} ref={da_ref:.6e}   "
              f"aq mismatches={nmis}/{K_G}")
        print(f"    scratch max={np.abs(scratch).max():.3e} "
              f"rms={np.sqrt(np.mean(scratch*scratch)):.3e} "
              f"aq_dev nonzero={int(np.count_nonzero(aq_dev))}")
    print(f"== done ({opts.ntok} tokens) ==")


if __name__ == "__main__":
    main()
