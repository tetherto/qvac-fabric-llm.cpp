# One-core check of the prologue's attention work (act-att.cc) against numpy.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/act_att_check.py prep|combine|pass
# A fresh NPU_CACHE_HOME per kernel edit: iron.jit reuses a cached artifact.
import hashlib
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

MODE = sys.argv[1] if len(sys.argv) > 1 else "prep"
ACT = 2112
AW = ACT // 4
D, H, NROT = 256, 8, 64
here = Path(__file__).resolve().parent.parent / "kernels"
src = (here / "gemv-q4.cc").read_text() + "\n" + (here / "act-att.cc").read_text()
flags = ["-DK_TILE_Q4=256", "-DK_TILE_Q8=128", f"-DACT_TILE={ACT}", "-DN_CORE=64",
         "-DQ4_GROUP=32", "-DQ8_GROUP=16", "-DGEMV_VEC=64", "-DACT_RAW=0", "-DACT_PRO=1"]
obj = "tpro_" + hashlib.md5((src + str(flags)).encode()).hexdigest()[:8] + ".o"
a_ty = np.ndarray[(AW,), np.dtype[np.int32]]
s_ty = np.ndarray[(512,), np.dtype[np.int32]]
e_ty = np.ndarray[(256,), np.dtype[np.int32]]
c_ty = np.ndarray[(2,), np.dtype[np.int32]]


@iron.jit
def pro(main: In, side: In, emit: Out, out: Out, *, NM: CompileTime[int],
        NS: CompileTime[int], NE: CompileTime[int]):
    mk = lambda n, t: ExternalFunction(n, object_file_name=obj, source_string=src,
                                       arg_types=t, include_dirs=_include_dirs(),
                                       compile_flags=flags)
    k_pro = mk("ggml_xdna_act_pro", [a_ty, a_ty, a_ty])
    k_cnt = mk("ggml_xdna_act_cnt", [a_ty, c_ty])
    k_side = mk("ggml_xdna_act_side", [s_ty, a_ty])
    k_emit = mk("ggml_xdna_act_emit", [e_ty, a_ty])
    fm = ObjectFifo(a_ty, name="fm", depth=2)
    fs = ObjectFifo(s_ty, name="fs", depth=2)
    fe = ObjectFifo(e_ty, name="fe", depth=2)
    fo = ObjectFifo(a_ty, name="fo", depth=2)

    def body(a_in, a_out, s_in, e_out, ktile, kcnt, kside, kemit, cnt):
        for _ in range_(NM):
            i_ = a_in.acquire(1)
            kcnt(i_, cnt)
            for _ in range_(cnt[0]):
                s_ = s_in.acquire(1)
                kside(s_, i_)
                s_in.release(1)
            for _ in range_(cnt[1]):
                e_ = e_out.acquire(1)
                kemit(e_, i_)
                e_out.release(1)
            o_ = a_out.acquire(1)
            ktile(i_, i_, o_)
            a_in.release(1)
            a_out.release(1)

    w = Worker(body, fn_args=[fm.cons(), fo.prod(), fs.cons(), fe.prod(), k_pro, k_cnt,
                              k_side, k_emit, Buffer(c_ty, name="cnt")], stack_size=0x1300,
               data_size=28672)

    def seq(m_h, s_h, e_h, o_h, mi, si, eo, oo):
        mi.fill(m_h)
        si.fill(s_h)
        if NE:
            eo.drain(e_h, wait=True)
        oo.drain(o_h, wait=True)

    rt = Runtime(seq, [np.ndarray[(NM * AW,), np.dtype[np.int32]],
                       np.ndarray[(max(NS, 1) * 512,), np.dtype[np.int32]],
                       np.ndarray[(max(NE, 1) * 256,), np.dtype[np.int32]],
                       np.ndarray[(NM * AW,), np.dtype[np.int32]],
                       fm.prod(), fs.prod(), fe.cons(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def run(main, side, ne):
    nm, ns = main.shape[0], side.shape[0]
    m = iron.tensor(main.reshape(-1), dtype=np.int32, device="npu")
    s = iron.tensor(side.reshape(-1) if ns else np.zeros(512, np.int32), dtype=np.int32, device="npu")
    e = iron.zeros(max(ne, 1) * 256, dtype=np.int32, device="npu")
    o = iron.zeros(nm * AW, dtype=np.int32, device="npu")
    pro(m, s, e, o, NM=nm, NS=ns, NE=ne)
    return e.numpy().copy(), o.numpy().copy().reshape(nm, AW)


def fbits(x):
    return np.array([x], np.float32).view(np.int32)[0]


rng = np.random.default_rng(3)
bf = lambda x: x.astype(ml_dtypes.bfloat16).astype(np.float32)


def rms_rope(x, g, cos, sin, eps, scale):
    y = x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps) * g * scale
    y = y.astype(np.float32)
    h = NROT // 2
    a, b = y[:h].copy(), y[h:NROT].copy()
    y[:h] = a * cos[:h] - b * sin[:h]
    y[h:NROT] = a * sin[:h] + b * cos[:h]
    return y


if MODE == "prep":
    eps, scale, n_valid = 1e-6, 1.0 / 16, 777
    gq = rng.standard_normal(D).astype(np.float32) * 0.3 + 1
    gk = rng.standard_normal(D).astype(np.float32) * 0.3 + 1
    ang = 777 * 10000.0 ** (-np.arange(NROT // 2) * 2.0 / NROT)
    cos = np.zeros(128, np.float32); sin = np.zeros(128, np.float32)
    cos[:NROT // 2] = np.cos(ang); sin[:NROT // 2] = np.sin(ang)
    q = rng.standard_normal((H, D)).astype(np.float32) * 3
    k = rng.standard_normal((2, D)).astype(np.float32) * 2
    v = rng.standard_normal((2, D)).astype(np.float32)
    main = np.zeros((2, AW), np.int32)
    for t in range(2):
        main[t, :256] = gq.view(np.int32)
        main[t, 512:518] = [n_valid, t, 0, NROT, fbits(scale), fbits(eps)]
        main[t, 518] = 1
        main[t, AW - 2] = 1 << 5
        main[t, AW - 6] = 5 if t == 0 else 2
        main[t, AW - 7] = 2 if t == 0 else 0
    consts = np.concatenate([gk, cos, sin]).astype(np.float32)
    side = np.stack([consts.view(np.int32),
                     q[0:2].reshape(-1).view(np.int32), q[2:4].reshape(-1).view(np.int32),
                     k.reshape(-1).view(np.int32), v.reshape(-1).view(np.int32),
                     q[4:6].reshape(-1).view(np.int32), q[6:8].reshape(-1).view(np.int32)])
    e, o = run(main, side, 2)
    worst = 0
    for t in range(2):
        qt = o[t, :512].view(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
        for hh in range(4):
            ref = rms_rope(q[4 * t + hh], gq, cos, sin, eps, scale)
            got = np.array([qt[(d // 8) * 32 + (d % 8) * 4 + hh] for d in range(D)])
            rel = np.abs(got - ref).max() / np.abs(ref).max()
            worst = max(worst, rel)
            print(f"q t{t} h{hh} max rel {rel:.2e}")
        print("words", o[t, 512:515])
    kk = e[:256].view(np.float16).astype(np.float32).reshape(2, D)
    vv = e[256:512].view(np.float16).astype(np.float32).reshape(2, D)
    for g in range(2):
        ref = rms_rope(k[g], gk, cos, sin, eps, 1.0)
        rel = np.abs(kk[g] - ref.astype(np.float16).astype(np.float32)).max()
        print(f"k g{g} max abs vs f16 ref {rel:.2e}")
        print(f"v g{g} max abs vs f16 ref {np.abs(vv[g] - v[g].astype(np.float16)).max():.2e}")
    print("worst q", worst)

elif MODE == "combine":
    # sixteen partials in attn-dec.cc's layout, then the gate
    st = np.zeros((16, 2112), np.float32)
    ms = rng.standard_normal((16, 8)).astype(np.float32) * 3
    ls = rng.uniform(0.5, 5, (16, 8)).astype(np.float32)
    ls[15] = 0; ms[15] = -1e30          # a core past the valid positions
    os_ = rng.standard_normal((16, 8, D)).astype(np.float32)
    os_[15] = 0
    for c in range(16):
        for h in range(8):
            g, hh = h // 4, h % 4
            st[c, g * 16 + np.arange(16)[np.arange(16) % 4 == hh]] = ms[c, h]
            st[c, 32 + g * 16 + np.arange(16)[np.arange(16) % 4 == hh]] = ls[c, h]
            for d in range(D):
                st[c, 64 + g * 1024 + (d // 8) * 32 + hh * 8 + d % 8] = os_[c, h, d]
    gate = rng.standard_normal((8, D)).astype(np.float32) * 2
    # as the layer's side stream delivers them: every core's m and l, then
    # every core's o, then the gate
    flat = np.concatenate([st[:, :64].reshape(-1), st[:, 64:].reshape(-1)])
    npart = flat.size // 512
    # the projection's header takes the gate (during the attention, in the
    # layer), then its first tile the partials
    side = np.concatenate([gate.reshape(4, 512).view(np.int32),
                           flat.view(np.int32).reshape(npart, 512)])
    main = np.zeros((9, AW), np.int32)
    main[0, 0:2] = [8, 1]
    main[0, 518] = 3
    main[0, AW - 2] = 1 << 5
    main[0, AW - 6] = 4
    for kk in range(8):
        main[1 + kk, 517] = kk
        main[1 + kk, 518] = 2
        main[1 + kk, 519] = npart
        main[1 + kk, AW - 1] = 0
        main[1 + kk, AW - 2] = 1 << 5
        main[1 + kk, AW - 6] = npart if kk == 0 else 0
    e, o = run(main, side, 0)
    print("header", o[0, :2], "flags", o[0, AW - 2])
    o = o[1:]
    M = ms.max(axis=0)
    w = np.exp(ms - M)
    L = (ls * w).sum(axis=0)
    O = (os_ * w[:, :, None]).sum(axis=0) / L[:, None]
    ref = (O / (1 + np.exp(-gate))).reshape(-1)
    got = np.zeros(2048, np.float32)
    for kk in range(8):
        codes = o[kk].view(np.int8)[:256].astype(np.float32)
        gd = o[kk].view(np.float32)[64 + 8:64 + 16]
        got[kk * 256:(kk + 1) * 256] = codes * np.repeat(gd, 32)
        print("tile", kk, "flags", o[kk, AW - 2], "fmt", o[kk, AW - 1])
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"combine rel {rel:.3e} (int8 group quantization alone ~4e-3)")
    g = ref.reshape(-1, 32)
    dq = np.where(np.abs(g).max(1) > 0, np.abs(g).max(1) / 127, 1)
    qref = (np.clip(np.rint(g / dq[:, None]), -127, 127) * dq[:, None]).reshape(-1)
    print(f"vs the reference quantized the same way: rel {np.linalg.norm(got - qref) / np.linalg.norm(qref):.3e}, "
          f"codes differing {(np.abs(got - qref) > 1e-6 * np.abs(qref).max()).sum()} of 2048")

elif MODE == "gates":
    # a GDN in-projection's row: the gates' dot products and x's tails
    Dr, kt, grp = 1024, 128, 16
    nt = Dr // kt
    eps = 1e-6
    acc = rng.standard_normal(Dr).astype(np.float32) * 3
    res = rng.standard_normal(Dr).astype(np.float32) * 10
    gam = (rng.standard_normal(Dr) * 0.2 + 1).astype(np.float32)
    wab = (rng.standard_normal((32, Dr)) * 0.05).astype(ml_dtypes.bfloat16)
    dt = rng.standard_normal(16).astype(np.float32)
    aa = -np.exp(rng.standard_normal(16)).astype(np.float32)
    sc = np.float32(1 / np.sqrt(128))
    consts = np.zeros(512, np.float32)
    consts[:16], consts[16:32], consts[32] = dt, aa, sc
    main = np.zeros((nt, AW), np.int32)
    for k in range(nt):
        t = main[k]
        t[512] = 32
        t[513:519] = [1, fbits(eps), Dr, 1, k, 4]
        t[AW - 1] = 1
        t[AW - 2] = 1 << 5
        t[AW - 6] = 6 if k == 0 else 0
        t[AW - 7] = 4 if k == 0 else 0
        if k == nt - 1:          # the gates on the last tile
            t[510] = 1
            t[AW - 6] += 33
            t[AW - 7] += 1
    side = np.concatenate([acc.view(np.int32).reshape(2, 512), res.view(np.int32).reshape(2, 512),
                           gam.view(np.int32).reshape(2, 512), consts.view(np.int32).reshape(1, 512),
                           wab.view(np.int32).reshape(32, 512)])
    e, o = run(main, side, 5)
    h = acc + res
    x = (h / np.sqrt(np.mean(h.astype(np.float64) ** 2) + eps) * gam).astype(np.float32)
    ab = wab.astype(np.float32) @ x.astype(ml_dtypes.bfloat16).astype(np.float32)
    z = ab[:16] + dt
    sp = np.where(z > 20, z, np.log1p(np.exp(z)))
    eg = np.exp(sp * aa)
    b = 1 / (1 + np.exp(-ab[16:]))
    tl = e[1024:1024 + 48].view(np.float32).reshape(16, 3)
    print("gates: eg rel", np.abs(tl[:, 0] / eg - 1).max(), "beta rel", np.abs(tl[:, 1] / b - 1).max(),
          "scale", tl[0, 2], sc)

elif MODE in ("row4", "row8"):
    # the layer boundary: h = acc + residual out, rms_norm(h) * gamma in tiles,
    # two chunks (the second replays from the row)
    q8 = MODE == "row8"
    Dr, kt, grp = 1024, (128 if q8 else 256), (16 if q8 else 32)
    nt = Dr // kt
    eps = 1e-6
    acc = rng.standard_normal(Dr).astype(np.float32) * 3
    res = rng.standard_normal(Dr).astype(np.float32) * 10
    gam = (rng.standard_normal(Dr) * 0.2 + 1).astype(np.float32)
    main = np.zeros((2 * nt, AW), np.int32)
    for c in range(2):
        for k in range(nt):
            t = main[c * nt + k]
            t[513:519] = [1, fbits(eps), Dr, 1 if q8 else 0, k, 4]
            t[AW - 1] = 1 if q8 else 0
            t[AW - 2] = (1 << 5) | (1 if k == nt - 1 else 0)
            t[AW - 6] = 6 if (c == 0 and k == 0) else 0
            t[AW - 7] = 4 if (c == 0 and k == 0) else 0
    side = np.concatenate([acc.view(np.int32).reshape(2, 512), res.view(np.int32).reshape(2, 512),
                           gam.view(np.int32).reshape(2, 512)])
    e, o = run(main, side, 4)
    h = acc + res
    hd = e[:1024].view(np.float32)
    ulp = np.abs(hd.view(np.int32).astype(np.int64) - h.view(np.int32).astype(np.int64))
    print("h exact:", np.array_equal(hd, h), "max ulps", ulp.max(), "differing", (ulp > 0).sum(),
          "max rel", (np.abs(hd - h) / np.abs(h)).max())
    x = (h / np.sqrt(np.mean(h.astype(np.float64) ** 2) + eps) * gam).astype(np.float32)
    g = x.reshape(-1, grp)
    dq = np.where(np.abs(g).max(1) > 0, np.abs(g).max(1) / 127, 1).astype(np.float32)
    qref = np.clip(np.rint(g / dq[:, None]), -127, 127).astype(np.int8).reshape(-1)
    worst = 0
    for c in range(2):
        codes = np.concatenate([o[c * nt + k].view(np.int8)[:kt] for k in range(nt)])
        gd = np.concatenate([o[c * nt + k].view(np.float32)[kt // 4 + kt // grp: kt // 4 + 2 * (kt // grp)] for k in range(nt)])
        worst = max(worst, np.abs(codes.astype(int) - qref).max())
        print(f"chunk {c}: codes off by at most {np.abs(codes.astype(int) - qref).max()}, "
              f"scale rel {np.abs(gd / dq - 1).max():.1e}, flags {o[c * nt + nt - 1, AW - 2]}")

else:
    # a tile without bit 5 passes through untouched
    main = rng.integers(-1000, 1000, (3, AW)).astype(np.int32)
    main[:, AW - 2] = 0
    e, o = run(main, np.zeros((0, 512), np.int32), 0)
    print("pass identical:", np.array_equal(o, main))
