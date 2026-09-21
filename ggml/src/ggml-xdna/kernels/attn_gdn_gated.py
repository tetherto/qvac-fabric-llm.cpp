#!/usr/bin/env python3
# attn_gdn_gated.py -*- Python -*-
#
# A-half of the fused recurrent layer: the merged conv+norm+gdn design PLUS the
# rec_gated epilogue tile, all in ONE xclbin / ONE per-token run, with no host
# attn round trip.  Channel verdict (probe #1): a shim column exposes 2 MM2S +
# 2 S2MM DMA channels, and the rec_gated tile as written takes 2 MM2S fills
# (az, gamma) + 1 S2MM drain.  The decoded stream shows only the norm columns
# (2-3) have spare channels,
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

import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cn  # noqa: E402
import gdn_v  # noqa: E402
import rec_gated  # noqa: E402
import rec_post  # noqa: E402

# ---- stage geometry (attn_cn + gdn_v + rec_gated constants) -------------------------
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

H2_SRC = Path(__file__).resolve().parent / "gated-h2.cc"


def _gated_h2_src():
    return H2_SRC.read_text()


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

    sflags = ["-O2", "-DNDEBUG"]

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
    conv_k = iron.ExternalFunction(name="ggml_xdna_attn_conv", source_string=conv_src,
                                   arg_types=[FEED_T, X_T, HIST_T],
                                   object_file_name=_sobj("ggml_xdna_attn_conv", conv_src, sflags),
                                   compile_flags=sflags, inline=True)
    norm_k = iron.ExternalFunction(name="ggml_xdna_attn_norm", source_string=norm_src,
                                   arg_types=[HN_T, PKVB_T],
                                   object_file_name=_sobj("ggml_xdna_attn_norm", norm_src, sflags),
                                   compile_flags=sflags, inline=True)
    norm_kj = [[iron.ExternalFunction(
                name=f"normj{gi}_{half}", source_string=norm_src_j[gi][half],
                arg_types=[HN_T, PKV_T],
                object_file_name=_sobj(f"normj{gi}_{half}",
                                       norm_src_j[gi][half], sflags),
                compile_flags=sflags, inline=True)
                for half in range(2)] for gi in range(NC_GDN)]
    gdn_k = iron.ExternalFunction(name="ggml_xdna_gdn_v", source_string=gdn_src,
                                  arg_types=[PKV_T, ROWS_T, ROWS_T, ATT_T],
                                  object_file_name=_sobj("ggml_xdna_gdn_v", gdn_src, sflags),
                                  compile_flags=sflags, inline=True)

    AZG_T = np.ndarray[(HG * AZG_N,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(ACT_OFF_G if act_split else OUTN,), np.dtype[np.uint8]]
    ACTB_T = np.ndarray[(ACTN,), np.dtype[np.uint8]]
    gflags = ["-O2", "-DNDEBUG",
              f"-DHG={HG}", f"-DAZG_N={AZG_N}", f"-DATTN_ONCHIP={int(att_onchip)}",
              f"-DK_GATE={K_G}", f"-DACT_TILE={ACT_TILE_G}",
              f"-DACT_OFF={ACT_OFF_G}", f"-DACT_SPLIT={int(act_split)}",
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
    pflags = ["-O2", "-DNDEBUG", f"-DGATED_FMT={GATED_FMT}"]
    post_k = iron.ExternalFunction(
        name="ggml_xdna_post_norm", source_string=post_src,
        arg_types=[POSTI_T, POSTO_T],
        object_file_name=_sobj("ggml_xdna_post_norm", post_src, pflags),
        compile_flags=pflags, inline=True)

    h2_src = _gated_h2_src()
    fin_src = rec_gated._gated_src()
    kh = iron.ExternalFunction(name="ggml_xdna_gated_h2", source_string=h2_src,
                               arg_types=([OUT_T, AZG_T, ATT4_T, ATT4_T]
                                          if att_onchip else [OUT_T, AZG_T]),
                               object_file_name=_obj("ggml_xdna_gated_h2", h2_src),
                               compile_flags=gflags, inline=True)
    kf = iron.ExternalFunction(name="ggml_xdna_gated_fin", source_string=fin_src,
                               arg_types=([OUT_T, ACTB_T] if act_split
                                          else [OUT_T]),
                               object_file_name=_obj("ggml_xdna_gated_fin", fin_src),
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

    if att_onchip and act_split:
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
    az_col = int(__import__("os").environ.get("GATED_AZ_COL", str(AZG_COL)))
    rt_args += [az3.prod(tile=Tile(az_col, 0)),
                gt23.cons(tile=Tile(gcol, 0))]
    if act_split:
        # GACT_COL: the activation drain's shim column. Defaults to the attn
        # column (7) - what the C++ stream builder (gated_act_col) expects;
        # a different column must be mirrored there or the drain token never
        # arrives. The azg fill shares the column, so the builder keeps its
        # descriptor apart (fill bd 0 / drain bd 1).
        gact_col = int(__import__("os").environ.get("GACT_COL", str(AZG_COL)))
        rt_args += [ga23.cons(tile=Tile(gact_col, 0))]

    # ---- post: col POST_COL, row 2 ----
    # The FFN transition is always part of this design: it turns the
    # projection's output, the residual and gamma the host wrote into the
    # activation the next dispatch reads, which is what lets a layer's two
    # dispatches go into one stream.
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

    workers.append(Worker(post_fn, [pi2.cons(), po12.prod(), post_k],
                          stack_size=0x3000))
    # The endpoints are pinned: the fill on (1,0) MM2S ch1, the drain on
    # (2,0) S2MM ch1 - free in this design. The fused projection's first
    # weight stream drives (0,0) MM2S ch1, so the fill must not sit
    # there or the post dispatch after a fused dispatch never completes.
    # The host-built stream (xdna-rec.cpp POST_FILL_COL / POST_DRN_COL)
    # agrees with these; "auto" restores the placer's choice (the merged
    # fused_layer design pins its own endpoints).
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
        pfill = nxt()
        pdrain = nxt()

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
    return prog.resolve_program()
