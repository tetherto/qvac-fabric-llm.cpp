# One-core check of the prologue's attention work (act-att.cc) against numpy.
# On the bench, in the IRON env:
#   NPU_CACHE_HOME=$(mktemp -d) python probes/act_att_check.py MODE
# MODE: prep|prep16 (8 or 16 query heads), combine|combine16 (one head group or
# two), pass, or a row mode (ROW_MODES): row4|row8 (D 1024),
# row4w|row8w (D 2048), gates|gates4w|gates8w (with the GDN gates), and the
# wider rows row4-D|row8-D|gates4-D|gates8-D for D 2560, 3072, 4096.
# Each mode checks its results against a tolerance and exits nonzero on a miss.
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

# The kernels include the shared headers (xdna-math.h, xdna-vec.h), which do not
# compile beside the source string: kernelsrc pastes them in, as the designs do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
import kernelsrc  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "prep"
failed = []


def check(what, ok):
    if not ok:
        failed.append(what)


ACT = 2112
AW = ACT // 4
D, H, NROT = 256, 8, 64
here = Path(__file__).resolve().parent.parent / "kernels"
src = kernelsrc.inline((here / "gemv-q4.cc").read_text() + "\n" + (here / "act-att.cc").read_text())
flags = ["-DK_TILE_Q4=256", "-DK_TILE_Q8=128", f"-DACT_TILE={ACT}", "-DN_CORE=64",
         "-DQ4_GROUP=32", "-DQ8_GROUP=32", "-DGEMV_VEC=64", "-DACT_RAW=0", "-DACT_PRO=1"]
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
               data_size=34816)

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


SIDE, EMIT, GATE_VALUES, N_GATES = 512, 256, 1024, 32
# mode: (D, 8-bit codes, gates)
ROW_MODES = {"row4": (1024, False, False), "row8": (1024, True, False),
             "row4w": (2048, False, False), "row8w": (2048, True, False),
             "gates": (1024, True, True), "gates4w": (2048, False, True), "gates8w": (2048, True, True)}
for _d in (2560, 3072, 4096):
    for _q8 in (False, True):
        ROW_MODES[f"row{8 if _q8 else 4}-{_d}"] = (_d, _q8, False)
        ROW_MODES[f"gates{8 if _q8 else 4}-{_d}"] = (_d, _q8, True)


def row_case(Dr, q8, gates):
    # The layer boundary as xdna_gemv_row_act lays it out: h = acc + residual
    # out, rms_norm(h) * gamma into the tiles of two chunks (the second a copy
    # of the first's), and with gates x's tails from the last tile of the last
    # chunk.
    kt, grp = (128, 32) if q8 else (256, 32)
    nt = Dr // kt
    ng = N_GATES if gates else 0
    eps = 1e-6
    acc = rng.standard_normal(Dr).astype(np.float32) * 3
    res = rng.standard_normal(Dr).astype(np.float32) * 10
    gam = (rng.standard_normal(Dr) * 0.2 + 1).astype(np.float32)
    main = np.zeros((2 * nt, AW), np.int32)
    for c in range(2):
        for k in range(nt):
            t = main[c * nt + k]
            t[512] = ng
            t[513:519] = [1, fbits(eps), Dr, int(q8), k, 4]
            t[AW - 1] = int(q8)
            t[AW - 2] = (1 << 5) | (1 if k == nt - 1 else 0)
            if c == 0 and k == 0:
                t[AW - 6] = Dr // SIDE * 3
                t[AW - 7] = Dr // EMIT
            if gates and c == 1 and k == nt - 1:
                t[510] = 1
                t[AW - 6] += 1 + ng * Dr // GATE_VALUES
                t[AW - 7] += 1
    side = [acc.view(np.int32).reshape(-1, SIDE), res.view(np.int32).reshape(-1, SIDE),
            gam.view(np.int32).reshape(-1, SIDE)]
    if gates:
        wab = (rng.standard_normal((ng, Dr)) * 0.05).astype(ml_dtypes.bfloat16)
        dt = rng.standard_normal(16).astype(np.float32)
        aa = -np.exp(rng.standard_normal(16)).astype(np.float32)
        sc = np.float32(1 / np.sqrt(128))
        consts = np.zeros(SIDE, np.float32)
        consts[:16], consts[16:32], consts[32] = dt, aa, sc
        side += [consts.view(np.int32).reshape(1, SIDE), wab.view(np.int32).reshape(-1, SIDE)]
    e, o = run(main, np.concatenate(side), Dr // EMIT + (1 if gates else 0))
    h = acc + res
    hd = e[:Dr].view(np.float32)
    ulp = np.abs(hd.view(np.int32).astype(np.int64) - h.view(np.int32).astype(np.int64))
    print(f"D {Dr}: h exact:", np.array_equal(hd, h), "max ulps", ulp.max(), "differing", (ulp > 0).sum(),
          "max rel", (np.abs(hd - h) / np.abs(h)).max())
    check("h", np.all(np.isfinite(hd)) and (np.abs(hd - h) / np.abs(h)).max() < 1e-5)
    x = (h / np.sqrt(np.mean(h.astype(np.float64) ** 2) + eps) * gam).astype(np.float32)
    g = x.reshape(-1, grp)
    dq = np.where(np.abs(g).max(1) > 0, np.abs(g).max(1) / 127, 1).astype(np.float32)
    qref = np.clip(np.rint(g / dq[:, None]), -127, 127).astype(np.int8).reshape(-1)
    for c in range(2):
        codes = np.concatenate([o[c * nt + k].view(np.int8)[:kt] for k in range(nt)])
        gd = np.concatenate([o[c * nt + k].view(np.float32)[kt // 4 + kt // grp: kt // 4 + 2 * (kt // grp)]
                             for k in range(nt)])
        off = np.abs(codes.astype(int) - qref).max()
        print(f"chunk {c}: codes off by at most {off}, scale rel {np.abs(gd / dq - 1).max():.1e}, "
              f"flags {o[c * nt + nt - 1, AW - 2]}")
        check(f"chunk {c} codes", off <= 1)
        check(f"chunk {c} scale", np.abs(gd / dq - 1).max() < 1e-5)
    if gates:
        ab = wab.astype(np.float32) @ x.astype(ml_dtypes.bfloat16).astype(np.float32)
        z = ab[:16] + dt
        sp = np.where(z > 20, z, np.log1p(np.exp(z)))
        eg = np.exp(sp * aa)
        b = 1 / (1 + np.exp(-ab[16:]))
        tl = e[Dr:Dr + 48].view(np.float32).reshape(16, 3)
        print("gates: eg rel", np.abs(tl[:, 0] / eg - 1).max(), "beta rel", np.abs(tl[:, 1] / b - 1).max(),
              "scale", tl[0, 2], sc)
        check("gates eg", np.abs(tl[:, 0] / eg - 1).max() < 1e-3)
        check("gates beta", np.abs(tl[:, 1] / b - 1).max() < 1e-3)
        check("gates scale", np.all(tl[:, 2] == sc))


def prep_case(heads):
    # The layer's prepare tiles, four query heads a tile: the first takes the
    # consts and every kv head's k and v (two heads an object) before its q,
    # and emits this position's K row, then its V row.
    eps, scale, n_valid = 1e-6, 1.0 / 16, 777
    kvh, nt = heads // 4, heads // 4
    gq = rng.standard_normal(D).astype(np.float32) * 0.3 + 1
    gk = rng.standard_normal(D).astype(np.float32) * 0.3 + 1
    ang = 777 * 10000.0 ** (-np.arange(NROT // 2) * 2.0 / NROT)
    cos = np.zeros(128, np.float32)
    sin = np.zeros(128, np.float32)
    cos[:NROT // 2] = np.cos(ang)
    sin[:NROT // 2] = np.sin(ang)
    q = rng.standard_normal((heads, D)).astype(np.float32) * 3
    k = rng.standard_normal((kvh, D)).astype(np.float32) * 2
    v = rng.standard_normal((kvh, D)).astype(np.float32)
    main = np.zeros((nt, AW), np.int32)
    for t in range(nt):
        main[t, :256] = gq.view(np.int32)
        main[t, 512:518] = [n_valid, t % 2, 0, NROT, fbits(scale), fbits(eps)]
        main[t, 518] = 1
        main[t, 519] = kvh if t == 0 else 0
        main[t, AW - 2] = 1 << 5
        main[t, AW - 6] = 1 + kvh + 2 if t == 0 else 2
        main[t, AW - 7] = kvh if t == 0 else 0
    consts = np.concatenate([gk, cos, sin]).astype(np.float32)
    side = np.concatenate([consts.view(np.int32).reshape(1, 512), k.reshape(-1, 512).view(np.int32),
                           v.reshape(-1, 512).view(np.int32), q.reshape(-1, 512).view(np.int32)])
    e, o = run(main, side, kvh)
    worst = 0
    for t in range(nt):
        qt = o[t, :512].view(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
        for hh in range(4):
            ref = rms_rope(q[4 * t + hh], gq, cos, sin, eps, scale)
            got = np.array([qt[(d // 8) * 32 + (d % 8) * 4 + hh] for d in range(D)])
            rel = np.abs(got - ref).max() / np.abs(ref).max()
            worst = max(worst, rel)
            print(f"q t{t} h{hh} max rel {rel:.2e}")
            check(f"q t{t} h{hh}", np.all(np.isfinite(got)) and rel < 1e-2)
        print("words", o[t, 512:515])
        check(f"words t{t}", list(o[t, 512:515]) == [n_valid, t % 2, 0])
    kk = e[:kvh * 128].view(np.float16).astype(np.float32).reshape(kvh, D)
    vv = e[kvh * 128:kvh * 256].view(np.float16).astype(np.float32).reshape(kvh, D)
    for g in range(kvh):
        ref = rms_rope(k[g], gk, cos, sin, eps, 1.0)
        rel = np.abs(kk[g] - ref.astype(np.float16).astype(np.float32)).max()
        print(f"k g{g} max abs vs f16 ref {rel:.2e}")
        print(f"v g{g} max abs vs f16 ref {np.abs(vv[g] - v[g].astype(np.float16)).max():.2e}")
        check(f"k g{g}", rel == 0)
        check(f"v g{g}", np.array_equal(vv[g], v[g].astype(np.float16).astype(np.float32)))
    print("worst q", worst)


def combine_case(groups):
    # Per head group sixteen partials in attn-dec.cc's layout; the header takes
    # every head's gate, the first tile of each group that group's partials.
    nh = 8 * groups
    gate = rng.standard_normal((nh, D)).astype(np.float32) * 2
    side = [gate.reshape(-1, 512).view(np.int32)]
    refs = []
    npart = 0
    for _ in range(groups):
        st = np.zeros((16, 2112), np.float32)
        ms = rng.standard_normal((16, 8)).astype(np.float32) * 3
        ls = rng.uniform(0.5, 5, (16, 8)).astype(np.float32)
        ls[15] = 0
        ms[15] = -1e30          # a core past the valid positions
        os_ = rng.standard_normal((16, 8, D)).astype(np.float32)
        os_[15] = 0
        for c in range(16):
            for h in range(8):
                g, hh = h // 4, h % 4
                st[c, g * 16 + np.arange(16)[np.arange(16) % 4 == hh]] = ms[c, h]
                st[c, 32 + g * 16 + np.arange(16)[np.arange(16) % 4 == hh]] = ls[c, h]
                for d in range(D):
                    st[c, 64 + g * 1024 + (d // 8) * 32 + hh * 8 + d % 8] = os_[c, h, d]
        # every core's m and l, then every core's o
        flat = np.concatenate([st[:, :64].reshape(-1), st[:, 64:].reshape(-1)])
        npart = flat.size // 512
        side.append(flat.view(np.int32).reshape(npart, 512))
        M = ms.max(axis=0)
        w = np.exp(ms - M)
        L = (ls * w).sum(axis=0)
        refs.append((os_ * w[:, :, None]).sum(axis=0) / L[:, None])
    ref = (np.concatenate(refs) / (1 + np.exp(-gate))).reshape(-1)
    main = np.zeros((1 + nh, AW), np.int32)
    main[0, 0:2] = [nh, 1]
    main[0, 518] = 3
    main[0, AW - 2] = 1 << 5
    main[0, AW - 6] = nh // 2
    for kk in range(nh):
        main[1 + kk, 517] = kk
        main[1 + kk, 518] = 2
        main[1 + kk, 519] = npart
        main[1 + kk, AW - 1] = 0
        main[1 + kk, AW - 2] = 1 << 5
        main[1 + kk, AW - 6] = npart if kk % 8 == 0 else 0
    e, o = run(main, np.concatenate(side), 0)
    print("header", o[0, :2], "flags", o[0, AW - 2])
    o = o[1:]
    got = np.zeros(nh * D, np.float32)
    for kk in range(nh):
        codes = o[kk].view(np.int8)[:256].astype(np.float32)
        gd = o[kk].view(np.float32)[64 + 8:64 + 16]
        got[kk * 256:(kk + 1) * 256] = codes * np.repeat(gd, 32)
        print("tile", kk, "flags", o[kk, AW - 2], "fmt", o[kk, AW - 1])
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"combine rel {rel:.3e} (int8 group quantization alone ~4e-3)")
    check("combine", np.all(np.isfinite(got)) and rel < 2e-2)
    g = ref.reshape(-1, 32)
    dq = np.where(np.abs(g).max(1) > 0, np.abs(g).max(1) / 127, 1)
    qref = (np.clip(np.rint(g / dq[:, None]), -127, 127) * dq[:, None]).reshape(-1)
    check("combine vs quantized", np.linalg.norm(got - qref) / np.linalg.norm(qref) < 1e-2)
    print(f"vs the reference quantized the same way: rel {np.linalg.norm(got - qref) / np.linalg.norm(qref):.3e}, "
          f"codes differing {(np.abs(got - qref) > 1e-6 * np.abs(qref).max()).sum()} of {nh * D}")


if MODE in ("prep", "prep16"):
    prep_case(16 if MODE == "prep16" else 8)

elif MODE in ("combine", "combine16"):
    combine_case(2 if MODE == "combine16" else 1)

elif MODE in ROW_MODES:
    row_case(*ROW_MODES[MODE])

else:
    # a tile without bit 5 passes through untouched
    main = rng.integers(-1000, 1000, (3, AW)).astype(np.int32)
    main[:, AW - 2] = 0
    e, o = run(main, np.zeros((0, 512), np.int32), 0)
    print("pass identical:", np.array_equal(o, main))
    check("pass", np.array_equal(o, main))

if failed:
    sys.exit(f"act_att_check {MODE}: FAIL on {', '.join(failed)}")
print(f"act_att_check {MODE}: PASS")
