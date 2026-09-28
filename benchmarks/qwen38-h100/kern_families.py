#!/usr/bin/env python3
"""Aggregate `nsys stats --report cuda_gpu_kern_sum --format csv` output by kernel family.

Usage: kern_families.py <kern_sum.csv> [--top N]
Prints total GPU time, per-family share and time, and the top-N raw kernels.
"""

import argparse
import csv
import re
import sys

FAMILIES = [
    ("mmq",              re.compile(r"mul_mat_q")),
    ("mmvq",             re.compile(r"mul_mat_vec_q")),
    ("mmf8_gemv",        re.compile(r"mul_mat_vec_f8_e4m3")),
    ("mmf8_mma",         re.compile(r"mul_mat_f8_e4m3_mma")),
    ("mmvf",             re.compile(r"mul_mat_vec_f")),
    ("mmf",              re.compile(r"mul_mat_f\b|mul_mat_f<")),
    ("fmha_cutlass",     re.compile(r"cutlass::fmha|FmhaKernel")),
    ("gated_delta_net",  re.compile(r"gated_delta_net|flashinfergdn|gdn_gates")),
    ("deepgemm",         re.compile(r"sm90_fp8_gemm|deep_gemm")),
    ("cutlass_gemm",     re.compile(r"cutlass|device_kernel")),
    ("f8_quantize",      re.compile(r"quantize_f8_e4m3|dequant_f8_e4m3")),
    ("cublas_gemm",      re.compile(r"nvjet|gemm|sm90_xmma|ampere_")),
    ("dequant_convert",  re.compile(r"dequantize|convert_unary|k_get_rows_float|to_fp16|to_fp32|k_convert")),
    ("quantize_q8_1",    re.compile(r"quantize_q8_1|quantize_mmq")),
    ("fa_vec",           re.compile(r"flash_attn_ext_vec|flash_attn_vec")),
    ("fa_mma",           re.compile(r"flash_attn_ext_f16|flash_attn_mma|flash_attn_ext_mma")),
    ("fa_tile",          re.compile(r"flash_attn_tile|flash_attn_ext_tile")),
    ("fa_combine",       re.compile(r"flash_attn_stream_k_fixup|flash_attn_combine")),
    ("ssm_conv",         re.compile(r"ssm_conv")),
    ("rms_norm",         re.compile(r"rms_norm|group_norm|l2_norm|\bnorm_f32")),
    ("rope",             re.compile(r"rope")),
    ("concat_conv_state", re.compile(r"concat_|conv_state")),
    ("cpy_setrows",      re.compile(r"cpy_|set_rows|k_set_rows|copy")),
    ("binbcast",         re.compile(r"k_bin_bcast|binbcast|k_add|k_mul\b")),
    ("unary_glu",        re.compile(r"unary|glu|silu|gelu|swiglu|sigmoid|softplus|exp")),
    ("softmax",          re.compile(r"soft_max|softmax")),
    ("get_rows",         re.compile(r"get_rows")),
    ("scale_argmax",     re.compile(r"scale|argmax|argsort")),
]


def family_of(name: str) -> str:
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    tcol = "Total Time (ns)"
    ncol = "Name"
    icol = "Instances"
    total = sum(float(r[tcol]) for r in rows)
    fam_t = {}
    fam_n = {}
    for r in rows:
        fam = family_of(r[ncol])
        fam_t[fam] = fam_t.get(fam, 0.0) + float(r[tcol])
        fam_n[fam] = fam_n.get(fam, 0) + int(r[icol])

    print(f"total GPU kernel time: {total/1e6:.1f} ms over {sum(int(r[icol]) for r in rows)} launches")
    print(f"{'family':18s} {'share':>7s} {'time ms':>10s} {'launches':>9s}")
    for fam, t in sorted(fam_t.items(), key=lambda kv: -kv[1]):
        print(f"{fam:18s} {100*t/total:6.1f}% {t/1e6:10.1f} {fam_n[fam]:9d}")

    print()
    print(f"top {args.top} kernels:")
    for r in sorted(rows, key=lambda r: -float(r[tcol]))[: args.top]:
        t = float(r[tcol])
        print(f"{100*t/total:6.1f}% {t/1e6:9.1f} ms {int(r[icol]):7d}x  {r[ncol][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
