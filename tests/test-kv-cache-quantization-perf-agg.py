#!/usr/bin/env python3
"""
Aggregate multiple KV cache quantization performance CSV files.

Reads CSVs produced by test-kv-cache-quantization-perf.sh, groups rows by
their grouping key (config, coopmat_mode, cache_k, cache_v), and computes
aggregated mean/stdev across runs. Outputs a combined CSV and a rendered
text table, appending the list of source files.

Usage:
    python tests/test-kv-cache-quantization-perf-agg.py -o aggregated.csv file1.csv file2.csv ...
    python tests/test-kv-cache-quantization-perf-agg.py -o aggregated.csv kv-perf_*.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import TypedDict

from kv_quant_agg_common import render_table, run_aggregation, safe_float


class PerfGroup(TypedDict):
    pp_means: list[float]
    pp_stdevs: list[float | None]
    tg_means: list[float]
    tg_stdevs: list[float | None]
    file_indices: list[int]
    backfilled: set[int]
    passthrough: dict[str, str] | None
    reps_total: int


LEGEND = """\
Legend:
  CM      — cooperative matrix mode: cm1=coopmat1, cm2=coopmat2, s=scalar, ref
  K / V   — KV cache quantization type for keys / values
  Compr   — compression ratio vs f16 (higher = smaller cache)
  Runs    — number of input files (models) aggregated
  pp avg  — prompt processing throughput (tokens/s), mean across runs
  pp sd   — prompt processing stdev; computed by averaging each run's
             relative stdev (CV = stdev/mean), then multiplying the
             aggregated mean by that average CV
  tg avg  — token generation throughput (tokens/s), mean across runs
  tg sd   — token generation stdev (same CV-based method as pp sd)
  pp/f16  — pp throughput as fraction of the f16 baseline (same CM mode)
  tg/f16  — tg throughput as fraction of the f16 baseline (same CM mode)
  pp/scl  — pp speedup vs scalar mode (same K/V config, coopmat_mode=scalar)
  tg/scl  — tg speedup vs scalar mode (same K/V config, coopmat_mode=scalar)
"""

GROUP_KEYS = ("config", "coopmat_mode", "cache_k", "cache_v")
PASSTHROUGH = ("gpu_device", "model", "mixed", "kv_size_mib", "compression_vs_f16",
               "prompt_len", "gen_len")
AGG_COLS = ("pp_avg", "pp_stdev", "tg_avg", "tg_stdev")
RATIO_COLS = ("pp_vs_f16_x", "tg_vs_f16_x", "pp_speedup_vs_scalar", "tg_speedup_vs_scalar")

OUTPUT_HEADER = [
    "gpu_device", "model", "config", "coopmat_mode", "cache_k", "cache_v",
    "mixed", "kv_size_mib", "compression_vs_f16", "prompt_len", "gen_len",
    "n_runs", "pp_avg", "pp_stdev", "tg_avg", "tg_stdev",
    "pp_vs_f16_x", "tg_vs_f16_x", "pp_speedup_vs_scalar", "tg_speedup_vs_scalar",
]


def combined_mean_stdev(means, stdevs):
    """Combine per-run mean±stdev into an aggregate mean and stdev.

    Averages relative stdev (CV%) across runs, then applies that
    percentage to the aggregated mean.  This avoids inflated stdev
    when aggregating across models with very different absolute scales.
    """
    n = len(means)
    if n == 0:
        return None, None
    grand_mean = sum(means) / n
    if n == 1:
        return grand_mean, stdevs[0] if stdevs[0] is not None else 0.0

    cvs = []
    for m, s in zip(means, stdevs):
        sd = s if s is not None else 0.0
        if m and m > 0:
            cvs.append(sd / m)
        else:
            cvs.append(0.0)
    avg_cv = sum(cvs) / n
    combined_sd = avg_cv * grand_mean
    return grand_mean, combined_sd


def _new_perf_group() -> PerfGroup:
    return {
        "pp_means": [], "pp_stdevs": [],
        "tg_means": [], "tg_stdevs": [],
        "file_indices": [],
        "backfilled": set(),
        "passthrough": None,
        "reps_total": 0,
    }


def aggregate(input_files: list[str]) -> dict[tuple[str, ...], PerfGroup]:
    groups: dict[tuple[str, ...], PerfGroup] = defaultdict(_new_perf_group)

    for file_idx, fpath in enumerate(input_files):
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row.get(k, "").strip().strip('"') for k in GROUP_KEYS)

                pp_avg = safe_float(row.get("pp_avg"))
                pp_sd = safe_float(row.get("pp_stdev"))
                tg_avg = safe_float(row.get("tg_avg"))
                tg_sd = safe_float(row.get("tg_stdev"))

                if pp_avg is None or tg_avg is None:
                    continue

                g = groups[key]
                g["pp_means"].append(pp_avg)
                g["pp_stdevs"].append(pp_sd)
                g["tg_means"].append(tg_avg)
                g["tg_stdevs"].append(tg_sd)
                g["file_indices"].append(file_idx)

                reps = safe_float(row.get("reps"))
                g["reps_total"] += int(reps) if reps else 1

                if g["passthrough"] is None:
                    g["passthrough"] = {k: row.get(k, "").strip().strip('"') for k in PASSTHROUGH}

    _backfill_missing(groups, len(input_files))
    return groups


REF_TO_CM2 = {
    "ref-turbo3": "pq3_0",
    "ref-turbo4": "pq4_0",
}


def _backfill_missing(groups, n_files):
    """Fill missing file entries from a donor group or group average.

    For ref rows, use the same file's cm2 pq3_0/pq4_0 row as the donor
    (ref-turbo3 ≈ pq3_0, ref-turbo4 ≈ pq4_0).  Otherwise fall back to
    the group's own average.
    """
    for key, g in groups.items():
        present = set(g["file_indices"])
        if len(present) >= n_files:
            continue

        config, cm_mode, ck, cv = key

        donor = None
        if cm_mode == "ref":
            dk = REF_TO_CM2.get(ck)
            dv = REF_TO_CM2.get(cv)
            if dk and dv:
                donor = groups.get((config, "coopmat2", dk, dv))
                if donor is None:
                    donor = groups.get((config, "coopmat2", dk, dk))

        for fi in range(n_files):
            if fi in present:
                continue

            filled = False
            if donor is not None:
                try:
                    idx = donor["file_indices"].index(fi)
                    g["pp_means"].append(donor["pp_means"][idx])
                    g["pp_stdevs"].append(donor["pp_stdevs"][idx])
                    g["tg_means"].append(donor["tg_means"][idx])
                    g["tg_stdevs"].append(donor["tg_stdevs"][idx])
                    filled = True
                except ValueError:
                    pass

            if not filled:
                n = len(g["pp_means"])
                g["pp_means"].append(sum(g["pp_means"]) / n)
                pp_sds = [s for s in g["pp_stdevs"] if s is not None]
                g["pp_stdevs"].append(sum(pp_sds) / len(pp_sds) if pp_sds else 0.0)
                g["tg_means"].append(sum(g["tg_means"]) / n)
                tg_sds = [s for s in g["tg_stdevs"] if s is not None]
                g["tg_stdevs"].append(sum(tg_sds) / len(tg_sds) if tg_sds else 0.0)

            g["file_indices"].append(fi)
            g["backfilled"].add(fi)


def _find_f16_group(groups, config, cm_mode):
    """Find the f16/f16 baseline group, falling back across CM modes."""
    for mode in (cm_mode, "coopmat2", "coopmat1", "scalar"):
        g = groups.get((config, mode, "f16", "f16"))
        if g is not None:
            return g
    return None


def _per_file_ratios(group, baseline):
    """Compute per-file ratios by matching on file_index, then average.

    Skips backfilled entries so ratios only reflect real measurements.
    The stdev combines two sources of uncertainty via total variance:
      1. Within-model: error-propagated ratio uncertainty from each file's
         pp_sd/tg_sd (CV of numerator and denominator).
      2. Across-model: variance of the per-file ratio point estimates.
    """
    bl_backfilled = baseline.get("backfilled", set())
    bl_by_file = {}
    for i, fi in enumerate(baseline["file_indices"]):
        if fi not in bl_backfilled:
            bl_by_file[fi] = (baseline["pp_means"][i], baseline["pp_stdevs"][i],
                              baseline["tg_means"][i], baseline["tg_stdevs"][i])

    grp_backfilled = group.get("backfilled", set())
    pp_ratios = []
    pp_ratio_vars = []
    tg_ratios = []
    tg_ratio_vars = []
    for i, fi in enumerate(group["file_indices"]):
        if fi in grp_backfilled or fi not in bl_by_file:
            continue
        bl_pp, bl_pp_sd, bl_tg, bl_tg_sd = bl_by_file[fi]
        g_pp = group["pp_means"][i]
        g_pp_sd = group["pp_stdevs"][i] or 0.0
        g_tg = group["tg_means"][i]
        g_tg_sd = group["tg_stdevs"][i] or 0.0

        if bl_pp and bl_pp > 0:
            r = g_pp / bl_pp
            pp_ratios.append(r)
            cv_num = g_pp_sd / g_pp if g_pp else 0.0
            cv_den = (bl_pp_sd or 0.0) / bl_pp
            pp_ratio_vars.append((r * math.sqrt(cv_num**2 + cv_den**2))**2)
        if bl_tg and bl_tg > 0:
            r = g_tg / bl_tg
            tg_ratios.append(r)
            cv_num = g_tg_sd / g_tg if g_tg else 0.0
            cv_den = (bl_tg_sd or 0.0) / bl_tg
            tg_ratio_vars.append((r * math.sqrt(cv_num**2 + cv_den**2))**2)

    def _combine(ratios, ratio_vars):
        if not ratios:
            return None, None
        n = len(ratios)
        m = sum(ratios) / n
        mean_within_var = sum(ratio_vars) / n
        if n < 2:
            return m, math.sqrt(mean_within_var)
        across_var = sum((r - m) ** 2 for r in ratios) / (n - 1)
        return m, math.sqrt(mean_within_var + across_var)

    pp_m, pp_s = _combine(pp_ratios, pp_ratio_vars)
    tg_m, tg_s = _combine(tg_ratios, tg_ratio_vars)
    return pp_m, pp_s, tg_m, tg_s


def compute_ratios(groups):
    """Recompute vs-f16 and vs-scalar ratios from per-file paired ratios."""
    results = []
    for key in groups:
        config, cm_mode, ck, cv = key
        g = groups[key]

        pp_mean, pp_sd = combined_mean_stdev(g["pp_means"], g["pp_stdevs"])
        tg_mean, tg_sd = combined_mean_stdev(g["tg_means"], g["tg_stdevs"])
        n_runs = len(g["pp_means"])

        def _fmt_ratio(mean, sd):
            if mean is None:
                return ""
            if sd is not None and sd > 0:
                return f"{mean:.2f}\u00b1{sd:.2f}"
            return f"{mean:.2f}"

        pp_vs_f16 = ""
        tg_vs_f16 = ""
        f16 = _find_f16_group(groups, config, cm_mode)
        if f16 is not None:
            pp_m, pp_s, tg_m, tg_s = _per_file_ratios(g, f16)
            pp_vs_f16 = _fmt_ratio(pp_m, pp_s)
            tg_vs_f16 = _fmt_ratio(tg_m, tg_s)

        pp_vs_scalar = ""
        tg_vs_scalar = ""
        scalar_key = (config, "scalar", ck, cv)
        if cm_mode != "scalar" and scalar_key in groups:
            pp_m, pp_s, tg_m, tg_s = _per_file_ratios(g, groups[scalar_key])
            pp_vs_scalar = _fmt_ratio(pp_m, pp_s)
            tg_vs_scalar = _fmt_ratio(tg_m, tg_s)

        pt = g["passthrough"] or {}
        results.append({
            "gpu_device": pt.get("gpu_device", ""),
            "model": pt.get("model", ""),
            "config": config,
            "coopmat_mode": cm_mode,
            "cache_k": ck,
            "cache_v": cv,
            "mixed": pt.get("mixed", ""),
            "kv_size_mib": pt.get("kv_size_mib", ""),
            "compression_vs_f16": pt.get("compression_vs_f16", ""),
            "prompt_len": pt.get("prompt_len", ""),
            "gen_len": pt.get("gen_len", ""),
            "n_runs": str(n_runs),
            "pp_avg": f"{pp_mean:.2f}" if pp_mean is not None else "",
            "pp_stdev": f"{pp_sd:.2f}" if pp_sd is not None else "",
            "tg_avg": f"{tg_mean:.2f}" if tg_mean is not None else "",
            "tg_stdev": f"{tg_sd:.2f}" if tg_sd is not None else "",
            "pp_vs_f16_x": pp_vs_f16,
            "tg_vs_f16_x": tg_vs_f16,
            "pp_speedup_vs_scalar": pp_vs_scalar,
            "tg_speedup_vs_scalar": tg_vs_scalar,
        })

    return results


CM_SHORT = {
    "coopmat1": "cm1",
    "coopmat2": "cm2",
    "scalar": "s",
}


DISPLAY_COLS = [
    ("coopmat_mode", "CM", 5),
    ("cache_k", "K", 10),
    ("cache_v", "V", 10),
    ("compression_vs_f16", "Compr", 7),
    ("n_runs", "Runs", 5),
    ("pp_avg", "pp avg", 10),
    ("pp_stdev", "pp sd", 8),
    ("tg_avg", "tg avg", 10),
    ("tg_stdev", "tg sd", 8),
    ("pp_vs_f16_x", "pp/f16", 11),
    ("tg_vs_f16_x", "tg/f16", 11),
    ("pp_speedup_vs_scalar", "pp/scl", 11),
    ("tg_speedup_vs_scalar", "tg/scl", 11),
]


def main():
    run_aggregation(
        description="Aggregate KV cache quantization performance CSVs",
        title=" KV Cache Quantization Performance — Aggregated Results",
        output_header=OUTPUT_HEADER,
        aggregate=aggregate,
        compute_ratios=compute_ratios,
        render=lambda rows: render_table(
            rows, DISPLAY_COLS,
            transform=lambda k, v: CM_SHORT.get(v, v) if k == "coopmat_mode" else v),
        legend=LEGEND,
    )


if __name__ == "__main__":
    main()
