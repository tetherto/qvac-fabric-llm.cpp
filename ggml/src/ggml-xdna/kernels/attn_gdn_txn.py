#!/usr/bin/env python3
# attn_gdn_txn.py -*- Python -*-
#
# conv+norm+gdn in ONE xclbin / ONE per-token TXN stream / ONE run.  The
# conv/norm/gdn workers are persistent park workers (no IRON barrier, no IRON
# runtime gating): conv on cols 0-1 row2 (24 feed groups each), norm on cols
# 2-3 row3 (8 heads each), bf16-vector gdn on cols 4-7 row4 (32 chunk objects
# each).  The host-built per-token TXN stream phases:
#   phase A conv fills/drains (feed arg0 -> x arg1 + hist arg0),
#   phase B norm (x arg1 -> pkvb arg2),
#   phase C gdn (pkvb arg2 + state arg3 -> state arg3 + attn arg4).
# Run BO set: {feed, x, pkvb, state, attn}.
#
# Artifacts (fresh workdir per compile):
#   python attn_gdn_txn.py -d npu2 --workdir build/bin --tag ""
#   -> build/bin/attn_gdn_txn.{xclbin,insts.bin}
#
# Geometry: conv feed blocks are linear 0..47 (48 blocks) placed in the feed BO
# at float ga*FEED_N; block ga's x slice goes to X at ((ga%16)*HEAD_NORM +
# (ga//16)*S_V) floats and its shifted history back to feed float ga*FEED_N.
# norm head h reads X h*HEAD_NORM and writes pkvb h*PKVB_N.  gdn chunk c (c =
# head*8+j) reads pkv from pkvb c*PKV_N floats, state from arg3 c*ROWS bf16,
# writes state back and attn at (head*S_V + j*CHUNK) floats.  The 5 BOs and
# per-chunk/head/block offsets are identical to rec_full.xclbin.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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

# ---- stage geometry (attn_cn + gdn_v constants, 1:1 with rec_full) ----
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


# ---- merged conv + norm + gdn design (barrier-free persistent workers) ------

@iron.jit
def attn_gdn_txn(*, dev_name: CompileTime[str] = "npu2"):
    FEED_T = np.ndarray[(FEED_N,), np.dtype[np.float32]]
    X_T = np.ndarray[(S_V,), np.dtype[np.float32]]
    HIST_T = np.ndarray[(3 * S_V,), np.dtype[np.float32]]
    HN_T = np.ndarray[(HEAD_NORM,), np.dtype[np.float32]]
    PKVB_T = np.ndarray[(PKVB_N,), np.dtype[np.float32]]
    PKV_T = np.ndarray[(PKV_N,), np.dtype[np.float32]]
    ROWS_T = np.ndarray[(gdn_v.ROWS,), np.dtype[bfloat16]]
    ATT_T = np.ndarray[(gdn_v.CHUNK,), np.dtype[np.float32]]

    conv_k = iron.ExternalFunction(name="convh", source_string=attn_cn._conv_src(),
                                   arg_types=[FEED_T, X_T, HIST_T],
                                   compile_flags=["-O2", "-DNDEBUG"], inline=True)
    norm_k = iron.ExternalFunction(name="normh", source_string=attn_cn._norm_src(),
                                   arg_types=[HN_T, PKVB_T],
                                   compile_flags=["-O2", "-DNDEBUG"], inline=True)
    gdn_k = iron.ExternalFunction(name="gdn_v", source_string=gdn_v._kernel_src(),
                                  arg_types=[PKV_T, ROWS_T, ROWS_T, ATT_T],
                                  compile_flags=["-O2", "-DNDEBUG"], inline=True)

    FEED_g = np.ndarray[(NC_CONV * CN_CONV * FEED_N,), np.dtype[np.float32]]
    X_g = np.ndarray[(N_VH * HEAD_NORM,), np.dtype[np.float32]]
    PKVB_g = np.ndarray[(N_VH * PKVB_N,), np.dtype[np.float32]]
    STATE_g = np.ndarray[(STATE_N,), np.dtype[bfloat16]]
    ATTN_g = np.ndarray[(N_VH * S_V,), np.dtype[np.float32]]

    workers = []
    rt_args = [FEED_g, X_g, PKVB_g, STATE_g, ATTN_g]

    def group_base(col, g):
        return col * CN_CONV * S_V + g * S_V

    def head_region(ach):
        return (ach % 2048) // S_V, ach // 2048

    # ---- conv: cols 0..NC_CONV-1, row 2 ----
    for col in range(NC_CONV):
        f3 = ObjectFifo(FEED_T, name=f"agf3_{col}", depth=2)
        f2 = f3.cons().forward(obj_type=FEED_T, name=f"agf2_{col}",
                               tile=Tile(col, 1))
        x23 = ObjectFifo(X_T, name=f"agx23_{col}", depth=2)
        x12 = x23.prod().join([0], obj_types=[X_T], names=[f"agx12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        h23 = ObjectFifo(HIST_T, name=f"agh23_{col}", depth=2)
        h12 = h23.prod().join([0], obj_types=[HIST_T], names=[f"agh12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def conv_fn(fc, xp, hc, k, n=CN_CONV):
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
                    x23.cons(tile=Tile(col, 0)),
                    h23.cons(tile=Tile(col, 0))]

    # ---- norm: cols NC_CONV..NC_CONV+NC_NORM-1, row 3 ----
    for ci in range(NC_NORM):
        col = NC_CONV + ci
        n3 = ObjectFifo(HN_T, name=f"agn3_{col}", depth=2)
        n2 = n3.cons().forward(obj_type=HN_T, name=f"agn2_{col}",
                               tile=Tile(col, 1))
        o23 = ObjectFifo(PKVB_T, name=f"agno23_{col}", depth=2)
        o12 = o23.prod().join([0], obj_types=[PKVB_T], names=[f"agno12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

        def norm_fn(nc, oc, k, nh=HP_NORM):
            for _ in range_(nh):
                ni = nc.acquire(1)
                oo = oc.acquire(1)
                k(ni, oo)
                oc.release(1)
                nc.release(1)

        workers.append(Worker(
            norm_fn, [n2.cons(), o12.prod(), norm_k],
            tile=Tile(col, 3), stack_size=0xD00))
        rt_args += [n3.prod(tile=Tile(col, 0)),
                    o23.cons(tile=Tile(col, 0))]

    # ---- gdn: cols GDN_COL0..7, row 4 ----
    for gi in range(NC_GDN):
        col = GDN_COL0 + gi
        p3 = ObjectFifo(PKV_T, name=f"agp3_{col}", depth=2)
        p2 = p3.cons().forward(obj_type=PKV_T, name=f"agp2_{col}",
                               tile=Tile(col, 1))
        s3 = ObjectFifo(ROWS_T, name=f"ags3_{col}", depth=2)
        s2 = s3.cons().forward(obj_type=ROWS_T, name=f"ags2_{col}",
                               tile=Tile(col, 1))
        o23 = ObjectFifo(ROWS_T, name=f"ago23_{col}", depth=2)
        o12 = o23.prod().join([0], obj_types=[ROWS_T], names=[f"ago12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]
        a23 = ObjectFifo(ATT_T, name=f"aga23_{col}", depth=2)
        a12 = a23.prod().join([0], obj_types=[ATT_T], names=[f"aga12_{col}"],
                              depths=[2], tile=Tile(col, 1))[0]

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

        workers.append(Worker(gdn_fn, [p2.cons(), s2.cons(), o12.prod(),
                                       a12.prod(), gdn_k],
                              tile=Tile(col, 4), stack_size=0x3000))
        rt_args += [p3.prod(tile=Tile(col, 0)),
                    s3.prod(tile=Tile(col, 0)),
                    o23.cons(tile=Tile(col, 0)),
                    a23.cons(tile=Tile(col, 0))]

    # IRON reference seq mirrors the python mode0 col-major order.
    def seq_fn(FEED, X, PKVB, STATE, ATTN, *fifos):
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
        nfill = []
        ndrain = []
        for _ci in range(NC_NORM):
            nfill.append(nxt())
            ndrain.append(nxt())
        gfillp = []
        gfills = []
        gdrain = []
        gattn = []
        for _gi in range(NC_GDN):
            gfillp.append(nxt())
            gfills.append(nxt())
            gdrain.append(nxt())
            gattn.append(nxt())

        # conv cols col-major
        for col in range(NC_CONV):
            for g in range(CN_CONV):
                ach = group_base(col, g)
                h, r = head_region(ach)
                gi = TaskGroup()
                cfeed[col].fill(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN_CONV * FEED_N,),
                    offset=(col * CN_CONV + g) * FEED_N, sizes=[FEED_N],
                    strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                cxdrain[col].drain(X, tap=TensorAccessPattern(
                    (N_VH * HEAD_NORM,), offset=h * HEAD_NORM + r * S_V,
                    sizes=[S_V], strides=[1]), wait=True, group=go)
                chist[col].drain(FEED, tap=TensorAccessPattern(
                    (NC_CONV * CN_CONV * FEED_N,),
                    offset=(col * CN_CONV + g) * FEED_N + F_H,
                    sizes=[3 * S_V], strides=[1]), wait=True, group=go)
                go.finish()

        # norm cols col-major
        for ci in range(NC_NORM):
            for hs in range(HP_NORM):
                head = ci * HP_NORM + hs
                gi = TaskGroup()
                nfill[ci].fill(X, tap=TensorAccessPattern(
                    (N_VH * HEAD_NORM,), offset=head * HEAD_NORM,
                    sizes=[HEAD_NORM], strides=[1]), group=gi)
                gi.finish()
                go = TaskGroup()
                ndrain[ci].drain(PKVB, tap=TensorAccessPattern(
                    (N_VH * PKVB_N,), offset=head * PKVB_N,
                    sizes=[PKVB_N], strides=[1]), wait=True, group=go)
                go.finish()

        # gdn cols col-major (chunk c assigned to col gi iff c%4==gi)
        for gi in range(NC_GDN):
            for c in range(gi, N_VH * N_OBJ, NC_GDN):
                head = c // N_OBJ
                j = c % N_OBJ
                gi2 = TaskGroup()
                gfillp[gi].fill(PKVB, tap=TensorAccessPattern(
                    (N_VH * PKVB_N,), offset=head * PKVB_N + j * PKV_N,
                    sizes=[PKV_N], strides=[1]), group=gi2)
                gfills[gi].fill(STATE, tap=TensorAccessPattern(
                    (STATE_N,), offset=c * gdn_v.ROWS,
                    sizes=[gdn_v.ROWS], strides=[1]), group=gi2)
                gi2.finish()
                go = TaskGroup()
                gdrain[gi].drain(STATE, tap=TensorAccessPattern(
                    (STATE_N,), offset=c * gdn_v.ROWS,
                    sizes=[gdn_v.ROWS], strides=[1]), wait=True, group=go)
                gattn[gi].drain(ATTN, tap=TensorAccessPattern(
                    (N_VH * S_V,), offset=head * S_V + j * gdn_v.CHUNK,
                    sizes=[gdn_v.CHUNK], strides=[1]), wait=True, group=go)
                go.finish()

    rt = Runtime(seq_fn, rt_args)
    prog = Program(from_name(dev_name, n_cols=GDN_COL0 + NC_GDN), rt, workers)
    return prog.resolve_program()




# ---- TXN builder (mirror of the C++ xdna-seq-attn-gdn-txn.cpp) --------------

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
                raise SystemExit(f"attn_gdn_txn: malformed op stream at word {i}")
        out = [(6 << 24) | (4 << 16) | (1 << 8) | 0,
               self.n_cols | (1 << 8),
               ninstr,
               (4 + len(ops)) * 4]
        out += ops
        return np.array(out, dtype=np.uint32)


# Per-fifo endpoint -> shim (dma_dir, channel, bd_id, arg) maps, matching the
# IRON-compiled attn_gdn columns (verified against the decoded stream). MM2S =
# 1 (fill from DDR, host -> device), S2MM = 0 (drain to DDR).
EP_CV_FEED = {"dir": 1, "ch": 0, "bd": 0, "arg": 0}
EP_CV_X    = {"dir": 0, "ch": 1, "bd": 0, "arg": 1}
EP_CV_HIST = {"dir": 0, "ch": 0, "bd": 1, "arg": 0}
EP_NM_X    = {"dir": 1, "ch": 0, "bd": 0, "arg": 1}
EP_NM_PKVB = {"dir": 0, "ch": 0, "bd": 0, "arg": 2}
EP_GD_PKV  = {"dir": 1, "ch": 0, "bd": 0, "arg": 2}
EP_GD_STATE= {"dir": 1, "ch": 1, "bd": 1, "arg": 3}
EP_GD_OUT  = {"dir": 0, "ch": 1, "bd": 0, "arg": 3}
EP_GD_ATTN = {"dir": 0, "ch": 0, "bd": 1, "arg": 4}

# block ga (0..47) x slice -> X BO head ga%16, region ga//16; hist -> feed block
def ga_head_region(ga):
    return ga % 16, ga // 16


def build_attn_gdn_txn_stream(mode=2):
    """Per-token TXN stream for attn_gdn_txn.xclbin.  Byte offsets into the 5
    arg BOs (arg0 feed, arg1 X, arg2 pkvb, arg3 state bf16, arg4 attn):
      conv feed block ga   : ga*FEED_N*4  (hist writeback also at ga*FEED_N*4)
      conv x slice ga      : ((ga%16)*HEAD_NORM + (ga//16)*S_V)*4
      norm X head hh       : hh*HEAD_NORM*4
      norm pkvb head hh    : hh*PKVB_N*4
      gdn pkv chunk c      : c*PKV_N*4    (pkvb is contiguous per chunk)
      gdn state chunk c    : c*ROWS*2     (in place)
      gdn attn chunk c     : ((c//8)*S_V + (c%8)*CHUNK)*4
    Phases run in time: conv cols0-1, then norm cols2-3, then gdn cols4-7.
    mode 0: IRON order (col-major per phase); mode 1: slot-major per phase;
    mode 2: phased (groups of 2 within each phase's columns)."""
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
        at = EP_GD_ATTN
        head = c // N_OBJ
        j = c % N_OBJ
        aoff = (head * S_V + j * gdn_v.CHUNK) * 4
        t.blockwrite(col, 0, at["bd"], gdn_v.CHUNK, aoff, 0, 0xC0000000,
                     0x2000000, 0)
        t.ddr_patch(col, 0, at["bd"], at["arg"], aoff)
        t.issue_token(col, 0, at["dir"], at["ch"])
        t.push_queue(col, 0, at["bd"], at["dir"], at["ch"], True)
        t.wait_token(col, 0, ou["dir"], ou["ch"])
        t.wait_token(col, 0, at["dir"], at["ch"])

    gp = NC_CONV * CN_CONV // NC_CONV  # 24 conv groups per conv col
    hp = N_VH // NC_NORM          # 8 heads per norm col
    slots = N_VH * N_OBJ // NC_GDN  # 32 chunk objects per gdn col
    group = 1 if mode == 1 else 2

    def conv_slot(col, s):
        conv_fill(col, col * gp + s)
        conv_drain_issue(col, col * gp + s)
        conv_wait(col)

    def norm_slot(j, hs):
        norm_fill(NC_CONV + j, j * hp + hs)
        norm_drain_issue(NC_CONV + j, j * hp + hs)
        norm_wait(NC_CONV + j)

    def gdn_slot(gi, s):
        gdn_fill(GDN_COL0 + gi, gi + s * NC_GDN)
        gdn_drain(GDN_COL0 + gi, gi + s * NC_GDN)

    # conv phase (cols 0..1)
    if mode == 0:
        for col in range(NC_CONV):
            for s in range(gp):
                conv_slot(col, s)
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
                norm_slot(j, hs)
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
                gdn_slot(gi, s)
    else:
        for g0 in range(0, slots, group):
            for s in range(g0, min(g0 + group, slots)):
                for gi in range(NC_GDN):
                    gdn_fill(GDN_COL0 + gi, gi + s * NC_GDN)
            for s in range(g0, min(g0 + group, slots)):
                for gi in range(NC_GDN):
                    gdn_drain(GDN_COL0 + gi, gi + s * NC_GDN)
    return t.build()


# ---- CLI / standalone validation ---------------------------------------------

def mk_bo(dev, arr, ro=False):
    import pyxrt as xrt
    bo = xrt.bo(dev, arr.nbytes, xrt.bo.host_only, 0)
    if not ro:
        np.frombuffer(bo.map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return bo


def load_kernel(dev, xclbin_path, words):
    import pyxrt as xrt
    xb = xrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kern = xrt.kernel(ctx, xb.get_kernels()[0].get_name())
    ibo = xrt.bo(dev, words.nbytes, xrt.bo.cacheable, kern.group_id(1))
    np.frombuffer(ibo.map(), dtype=np.uint32)[:] = words
    ibo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return kern, ibo, int(words.nbytes)


class Trial:
    """One kernel/insts-words candidate over `ntok` tokens with a fresh feed
    (conv history) and bf16 state, mirroring the rec_full.py host semantics."""

    def __init__(self, dev, name, xclbin, words):
        self.name = name
        self.kern, self.ibo, self.nb = load_kernel(dev, xclbin, words)
        self.feed = None
        self.feed_bo = None
        self.x_bo = None
        self.pkvb_bo = None
        self.state_bo = None
        self.attn_bo = None

    def setup(self, feed0):
        import pyxrt as xrt
        self.feed = feed0.copy()
        self.feed_bo = mk_bo(self.kern.get_device(), self.feed) if False else None

    def free(self):
        pass


def main():
    ap = argparse.ArgumentParser(prog="attn_gdn_txn")
    add_compile_args(ap)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--workdir", type=str, default="build/bin")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ntok", type=int, default=5)
    ap.add_argument("--model",
                    default="/home/asherstnev/.cache/llama.cpp/qwen3.5-0.8b-bf16.gguf")
    opts = ap.parse_args()
    if opts.dev is None:
        opts.dev = "npu2"
    if opts.tag is None:
        opts.tag = time.strftime("agtx_%H%M%S")
    wd = os.path.join(opts.workdir, opts.tag)
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "attn_gdn_txn")
    xclbin_path = base + ".xclbin"
    insts_path = base + ".insts.bin"
    t0 = time.time()
    spec = attn_gdn_txn.specialize(dev_name=opts.dev)
    xclbin_path, insts_path = spec.compile(xclbin_path=xclbin_path,
                                           inst_path=insts_path)
    print(f"compiled {xclbin_path} ({time.time()-t0:.0f}s)")
    print("xclbin", xclbin_path)
    print(f"tag={opts.tag}")
    if not opts.run:
        return
    run_validation(opts, xclbin_path, insts_path)


def run_validation(opts, xclbin_path, insts_path):
    import pyxrt as xrt
    from ml_dtypes import bfloat16
    from ref_delta_layer import N_KH, QKVD, D, load_layer, rms_norm, silu
    from check_real import load_reader, load_embed

    # words: IRON reference, then the three host-built schedules
    iref = np.frombuffer(Path(insts_path).read_bytes(), dtype=np.uint32)
    words = [("IRON", iref)]
    for m in (0, 1, 2):
        words.append((f"m{m}", build_attn_gdn_txn_stream(mode=m)))
    print(f"IRON words {len(iref)}  " + "  ".join(
        f"{n}={len(w)}" for n, w in words))
    for n, w in words:
        print(f"  {n}==IRON: {np.array_equal(w, iref)}")

    dev = xrt.device(0)
    reader = load_reader(opts.model)
    W = load_layer(reader, 0)
    scale = np.float32(1.0 / np.sqrt(S_V))

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

    T = len(tokens)
    projs = [proj(h_in[t]) for t in range(T)]
    Wcv = W["conv"]

    def block_chan_base(b):
        return (b // CN_CONV) * (CN_CONV * S_V) + (b % CN_CONV) * S_V

    feed0 = np.zeros(NC_CONV * CN_CONV * FEED_N, dtype=np.float32)
    for b in range(NC_CONV * CN_CONV):
        for ch in range(S_V):
            ach = block_chan_base(b) + ch
            feed0[b * FEED_N + F_W + 4 * ch: b * FEED_N + F_W + 4 * ch + 4] = \
                Wcv[:, ach]

    # reference device trajectory on rec_full.xclbin (byte-identical kernel
    # bodies/geometry), run side by side with the merged TXN candidates.
    ref_x = os.path.join(opts.workdir, "rec_full.xclbin")
    ref_i = os.path.join(opts.workdir, "rec_full.insts.bin")
    ref_words = None
    if Path(ref_x).exists() and Path(ref_i).exists():
        ref_words = np.frombuffer(Path(ref_i).read_bytes(), dtype=np.uint32)
    else:
        # fall back to the split pair via two chained launches?  the merged
        # xclbin is only compared against rec_full; without it we cannot prove
        # byte equality on device, so fail loudly.
        sys.exit("attn_gdn_txn --run: build/bin/rec_full.xclbin + insts "
                 "required as the byte reference")

    xclbin = Path(xclbin_path)

    def make_trial(name, kwords):
        kern, ibo, nb = load_kernel(dev, xclbin, kwords)
        feed_bo = mk_bo(dev, feed0)
        x_bo = mk_bo(dev, np.zeros(N_VH * HEAD_NORM, dtype=np.float32))
        pkvb_bo = mk_bo(dev, np.zeros(N_VH * PKVB_N, dtype=np.float32))
        state_bo = mk_bo(dev, np.zeros(STATE_N, dtype=bfloat16))
        attn_bo = mk_bo(dev, np.zeros(N_VH * S_V, dtype=np.float32))
        hist = np.zeros((3, QKVD), dtype=np.float32)
        return dict(kern=kern, ibo=ibo, nb=nb, feed_bo=feed_bo, x_bo=x_bo,
                    pkvb_bo=pkvb_bo, state_bo=state_bo, attn_bo=attn_bo,
                    hist=hist, feed=feed0.copy())

    def step(tr, qkv, eg, beta_s):
        feed = tr["feed"]
        for b in range(NC_CONV * CN_CONV):
            ach0 = block_chan_base(b)
            hist = feed[b * FEED_N + F_H: b * FEED_N + F_H + 3 * S_V]
            for tap in range(3):
                hist[tap::3] = tr["hist"][tap, ach0:ach0 + S_V]
            feed[b * FEED_N + F_Q: b * FEED_N + F_Q + S_V] = \
                qkv[ach0:ach0 + S_V]
        np.frombuffer(tr["feed_bo"].map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(feed).view(np.uint8).reshape(-1)
        tr["feed_bo"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        X = np.zeros(N_VH * HEAD_NORM, dtype=np.float32)
        for hh in range(N_VH):
            X[hh * HEAD_NORM + 3 * S_V:(hh + 1) * HEAD_NORM] = \
                [eg[hh], beta_s[hh], scale]
        np.frombuffer(tr["x_bo"].map(), dtype=np.uint8)[:] = \
            np.ascontiguousarray(X).view(np.uint8).reshape(-1)
        tr["x_bo"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = tr["kern"](3, tr["ibo"], tr["nb"], tr["feed_bo"], tr["x_bo"],
                         tr["pkvb_bo"], tr["state_bo"], tr["attn_bo"])
        if run.wait() != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit(f"attn_gdn_txn: run failed ({tr['name']})")
        conv_in = np.concatenate([tr["hist"], qkv[None, :]], axis=0)
        tr["hist"] = conv_in[1:, :].copy()

    def read(tr):
        for bo in (tr["feed_bo"], tr["x_bo"], tr["pkvb_bo"], tr["attn_bo"],
                   tr["state_bo"]):
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return {
            "feed": np.frombuffer(tr["feed_bo"].map(), dtype=np.float32)[:feed0.size].copy(),
            "x": np.frombuffer(tr["x_bo"].map(), dtype=np.float32)[:N_VH * HEAD_NORM].copy(),
            "pkvb": np.frombuffer(tr["pkvb_bo"].map(), dtype=np.float32)[:N_VH * PKVB_N].copy(),
            "attn": np.frombuffer(tr["attn_bo"].map(), dtype=np.float32)[:N_VH * S_V].copy(),
            "state": np.asarray(np.frombuffer(tr["state_bo"].map(),
                                              dtype=bfloat16)[:STATE_N],
                                dtype=np.float32),
        }

    # reference: rec_full kernel (IRON words)
    ref_kern, ref_ibo, ref_nb = load_kernel(dev, ref_x, ref_words)
    ref = dict(kern=ref_kern, ibo=ref_ibo, nb=ref_nb,
               feed_bo=mk_bo(dev, feed0),
               x_bo=mk_bo(dev, np.zeros(N_VH * HEAD_NORM, dtype=np.float32)),
               pkvb_bo=mk_bo(dev, np.zeros(N_VH * PKVB_N, dtype=np.float32)),
               state_bo=mk_bo(dev, np.zeros(STATE_N, dtype=bfloat16)),
               attn_bo=mk_bo(dev, np.zeros(N_VH * S_V, dtype=np.float32)),
               hist=np.zeros((3, QKVD), dtype=np.float32), feed=feed0.copy(),
               name="rec_full")
    trials = [("rec_full", ref)] + \
             [(n, make_trial(n, w)) for n, w in words]

    fields = ["feed", "x", "pkvb", "attn", "state"]
    worst_all = {}
    for t in range(T):
        qkv, z, eg, beta_s = projs[t]
        for tr in trials:
            step(tr[1], qkv, eg, beta_s)
        base = read(ref)
        for name, tr in trials[1:]:
            d = read(tr)
            worst = {f: np.abs(np.asarray(base[f], dtype=np.float32) -
                               np.asarray(d[f], dtype=np.float32)).max()
                     for f in fields}
            worst_all[name] = max(worst.values())
            print(f"t={t} {name}: " + "  ".join(
                f"{k}={v:.2e}" for k, v in worst.items()))
    for name in worst_all:
        print(f"  worst |{name}| vs rec_full: {worst_all[name]:.2e}")
    print("nan state:", int(np.isnan(read(ref)["state"]).sum()))
    print(f"== {T} tokens: attn_gdn_txn byte-vs rec_full done ==")


if __name__ == "__main__":
    main()
