# The FFN transition as its own design: residual add, RMS norm, gamma and the
# quantized activation tiles the FFN pair reads. A standalone xclbin like the
# IRON reference operators, dispatched between the fused core and the FFN -
# everything the layer needs is then on the array, which is worth more than
# one fewer dispatch.
#
# One column, one core. The object sizes:
#   in   [so_out D f32][residual D f32][gamma D f32][flags i32]
#   out  [hattn D f32][header tile][D/256 tiles]
# where a tile is ACT_TILE bytes of K_TILE int8 codes, a f32 code sum and a
# f32 scale per group of 32, and two trailing flag words.

import argparse
import hashlib
import os
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import from_name, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_tag

D        = 1024
K_TILE   = 256
ACT_TILE = 2112
GROUP    = 32


def _kernel_src() -> str:
    nt = D // K_TILE
    gpt = K_TILE // GROUP
    return f"""#include <stdint.h>
#include <math.h>
#include <aie_api/aie.hpp>
using namespace aie;

#define PD {D}
#if defined(GATED_FMT) && GATED_FMT == 1
#define PNT (PD / 128)
#else
#define PNT {nt}
#endif
#define PKT {K_TILE}
#define PGPT {gpt}
#define PACT {ACT_TILE}

extern "C" void post_norm(const float * in, uint8_t * out) {{
#if POST_STUB
    (void)in; (void)out;
    return;
#else
    event0();
    const float * so  = in;
    const float * res = in + PD;
    const float * gam = in + 2 * PD;
    // What the last tile tells the projection that reads it: close the chunk,
    // and for a pair, quantize the epilogue's output in the next format's
    // grouping. The host knows the weight types; the design does not.
    const int32_t last_flags = ((const int32_t *)in)[3 * PD];
    float * hattn = (float *)out;
    uint8_t * act = out + PD * 4;

    // Pass one: the residual add, kept in hattn - which the next layer reads
    // as its own residual - and the sum of squares along with it.
    aie::vector<float, 16> sq = aie::zeros<float, 16>();
    for (int i = 0; i < PD; i += 16) {{
        auto v = aie::add(aie::load_v<16>(so + i), aie::load_v<16>(res + i));
        aie::store_v(hattn + i, v);
        sq = aie::add(sq, aie::mul(v, v).to_vector<float>());
    }}
    alignas(64) float lanes[16];
    aie::store_v(lanes, sq);
    float ss = 0.0f;
    for (int i = 0; i < 16; i++) {{
        ss += lanes[i];
    }}
    const float scale = 1.0f / aie::sqrt(ss / (float)PD + 1e-6f);
    const auto vscale = aie::broadcast<float, 16>(scale);

    // Pass two: scale by gamma and quantize, a group of 32 at a time. The
    // scale is per group, not per row, because that is what the projection's
    // kernel reads.
    int32_t * hdr = (int32_t *)act;
    hdr[0] = PNT;
    hdr[1] = 1;
    hdr[PACT / 4 - 2] = 0;
#if defined(GATED_FMT) && GATED_FMT == 1
    hdr[PACT / 4 - 1] = 1;
#else
    hdr[PACT / 4 - 1] = 0;
#endif
    const auto absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
#if defined(GATED_FMT) && GATED_FMT == 1
    const auto vlo = aie::broadcast<float, 16>(-128.0f);
#else
    const auto vlo = aie::broadcast<float, 16>(-127.0f);
#endif
    const auto vhi = aie::broadcast<float, 16>(127.0f);
    const auto magic = aie::broadcast<float, 16>(12582912.0f);
    const auto magici = magic.cast_to<int32>();
    for (int t = 0; t < PNT; t++) {{
        uint8_t * tile = act + (1 + t) * PACT;
        int8 * code = (int8 *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
        float * gsum = (float *)(tile + 128);
        float * gd = gsum + 8;
        for (int g = 0; g < 8; g++) {{
            const int base = t * 128 + g * 16;
            auto y0 = aie::mul(aie::mul(aie::load_v<16>(hattn + base), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base)).to_vector<float>();
            const auto a0 = aie::bit_and(y0.cast_to<int32>(), absmask)
                                .cast_to<float>();
            const float ga = aie::reduce_max(a0);
            const float gdv = ga > 0.0f ? ga / 127.0f : 1.0f;
            const auto ginv = aie::broadcast<float, 16>(1.0f / gdv);
            auto q0 = aie::min(aie::max(aie::mul(y0, ginv).to_vector<float>(),
                                        vlo), vhi);
            const auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(),
                                      magici);
            aie::store_v(code + g * 16, aie::pack(aie::pack(i0)));
            gsum[g] = (float) aie::reduce_add(i0);
            gd[g] = gdv;
        }}
#else
        float * gsum = (float *)(tile + PKT);
        float * gd = gsum + PGPT;
        for (int g = 0; g < PGPT; g++) {{
            const int base = t * PKT + g * {GROUP};
            auto y0 = aie::mul(aie::mul(aie::load_v<16>(hattn + base), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base)).to_vector<float>();
            auto y1 = aie::mul(aie::mul(aie::load_v<16>(hattn + base + 16), vscale)
                                   .to_vector<float>(),
                               aie::load_v<16>(gam + base + 16)).to_vector<float>();
            auto a0 = aie::bit_and(y0.cast_to<int32>(), absmask).cast_to<float>();
            auto a1 = aie::bit_and(y1.cast_to<int32>(), absmask).cast_to<float>();
            auto vm = aie::max(a0, a1);
            alignas(64) float ml[16];
            aie::store_v(ml, vm);
            float amax = 0.0f;
            for (int i = 0; i < 16; i++) {{
                if (ml[i] > amax) amax = ml[i];
            }}
            const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
            const auto vinv = aie::broadcast<float, 16>(1.0f / d);
            auto q0 = aie::min(aie::max(aie::mul(y0, vinv).to_vector<float>(),
                                        vlo), vhi);
            auto q1 = aie::min(aie::max(aie::mul(y1, vinv).to_vector<float>(),
                                        vlo), vhi);
            auto i0 = aie::sub(aie::add(q0, magic).cast_to<int32>(), magici);
            auto i1 = aie::sub(aie::add(q1, magic).cast_to<int32>(), magici);
            aie::store_v(code + g * {GROUP}, aie::pack(aie::pack(i0)));
            aie::store_v(code + g * {GROUP} + 16, aie::pack(aie::pack(i1)));
            auto s0 = aie::add(i0, i1);
            alignas(64) int32_t sl[16];
            aie::store_v(sl, s0);
            int sum = 0;
            for (int i = 0; i < 16; i++) {{
                sum += sl[i];
            }}
            gsum[g] = (float)sum;
            gd[g] = d;
        }}
#endif
        int32_t * tw = (int32_t *)tile;
#if defined(GATED_FMT) && GATED_FMT == 1
        tw[PACT / 4 - 1] = 1;
#else
        tw[PACT / 4 - 1] = 0;
#endif
        tw[PACT / 4 - 2] = (t == PNT - 1) ? last_flags : 0;
    }}
    event1();
#endif
}}
"""


def post_norm_design(*, dev_name: str = "npu2"):
    _fmt = int(os.environ.get("GATED_FMT", "1"))
    _nt = D // (128 if _fmt == 1 else K_TILE)
    IN_T  = np.ndarray[(3 * D + 4,), np.dtype[np.float32]]
    OUT_T = np.ndarray[(D * 4 + (1 + _nt) * ACT_TILE,), np.dtype[np.uint8]]

    src = _kernel_src()
    flags = ["-O2", "-DNDEBUG",
             f"-DGATED_FMT={int(os.environ.get('GATED_FMT', '1'))}",
             f"-DPOST_STUB={int(os.environ.get('POST_STUB', '0'))}"]
    k = ExternalFunction(
        name="post_norm", source_string=src,
        arg_types=[IN_T, OUT_T],
        object_file_name=f"post_norm_{hashlib.sha256((src + '\\0'.join(flags)).encode()).hexdigest()[:8]}.ll",
        compile_flags=flags, inline=True)

    pi3 = ObjectFifo(IN_T, name="pni3", depth=1)
    pi2 = pi3.cons().forward(obj_type=IN_T, name="pni2",
                             tile=Tile(0, 1), depth=1)
    po23 = ObjectFifo(OUT_T, name="pno23", depth=1)
    po12 = po23.prod().join([0], obj_types=[OUT_T], names=["pno12"],
                            depths=[1], tile=Tile(0, 1))[0]

    def core_body(ic, oc, kern):
        for _ in range_(1):
            i = ic.acquire(1)
            o = oc.acquire(1)
            kern(i, o)
            oc.release(1)
            ic.release(1)

    workers = [Worker(core_body, [pi2.cons(), po12.prod(), k],
                      tile=Tile(0, 2), stack_size=0x3000)]

    def seq(IN, OUT, ip, oc_):
        tg = TaskGroup()
        ip.fill(IN, tap=TensorAccessPattern(
            (3 * D + 4,), offset=0, sizes=[3 * D + 4], strides=[1]), group=tg)
        oc_.drain(OUT, tap=TensorAccessPattern(
            (D * 4 + (1 + D // K_TILE) * ACT_TILE,), offset=0,
            sizes=[D * 4 + (1 + D // K_TILE) * ACT_TILE], strides=[1]),
            wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [IN_T, OUT_T, pi3.prod(tile=Tile(0, 0)),
                       po23.cons(tile=Tile(0, 0))])
    # Eight columns even though one core works: the array partition then
    # matches the fused layer's, so the dispatch does not reconfigure the
    # array between the two contexts.
    return Program(from_name(dev_name, n_cols=8), rt, workers).resolve_program()


@iron.jit
def post_norm(*, dev_name: iron.CompileTime[str] = "npu2"):
    return post_norm_design(dev_name=dev_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="post_norm",
        description="Build the standalone FFN transition tile")
    add_compile_args(parser)
    parser.add_argument("--elf-path", default=None,
                        help="compile as a self-contained full ELF to this path")
    opts = parser.parse_args()
    if opts.elf_path:
        spec = post_norm.specialize(full_elf=True, dev_name=opts.dev or "npu2")
        elf_path, _ = spec.compile(elf_path=opts.elf_path)
        import shutil
        shutil.copy(elf_path, opts.elf_path)
        print("compiled full ELF", opts.elf_path)
        return
    run_design_cli(
        post_norm,
        opts,
        compile_kwargs={},
        device=lambda o: from_name(o.dev, n_cols=8),
    )
    import design_tag
    design_tag.stamp(getattr(opts, "xclbin_path", None),
                     getattr(opts, "insts_path", None),
                     getattr(opts, "dev", "") or "")


if __name__ == "__main__":
    main()
