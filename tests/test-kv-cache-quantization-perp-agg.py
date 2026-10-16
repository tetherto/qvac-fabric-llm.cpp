#!/usr/bin/env python3
"""
Aggregate multiple KV cache quantization perplexity CSV files.

Reads CSVs produced by test-kv-cache-quantization-perp.sh, groups rows by
their grouping key (config, cache_k, cache_v, norm_correction), and computes
aggregated mean/stdev across runs. Outputs a combined CSV and a rendered
text table, appending the list of source files.

Usage:
    python tests/test-kv-cache-quantization-perp-agg.py -o aggregated.csv file1.csv file2.csv ...
    python tests/test-kv-cache-quantization-perp-agg.py -o aggregated.csv kv-perp_*.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import TypedDict

from kv_quant_agg_common import render_table, run_aggregation, safe_float


class PerpGroup(TypedDict):
    ppl_means: list[float]
    ppl_stdevs: list[float | None]
    time_means: list[float]
    time_stdevs: list[float | None]
    passthrough: dict[str, str] | None


class DerivedGroup(TypedDict):
    ppl_means: list[float]
    ppl_stdevs: list[float | None]
    time_means: list[float]
    time_stdevs: list[float | None]
    passthrough: dict[str, str] | None
    mixed: str
    bpw_avgs: list[float]
    cache_k: str
    cache_v: str


GROUP_KEYS = ("config", "cache_k", "cache_v", "norm_correction")
PASSTHROUGH = ("gpu_device", "model", "mixed", "bpw_avg", "n_chunks", "n_ctx_sweep")
REF_PREFIX = "ref-"

OUTPUT_HEADER = [
    "gpu_device", "model", "cache_k", "cache_v", "mixed",
    "norm_correction", "bpw_avg", "n_chunks", "n_ctx_sweep",
    "n_runs", "ppl_mean", "ppl_stdev", "time_mean_s", "time_stdev_s",
    "ppl_vs_f16_pct", "ppl_vs_f16_pct_stdev",
    "time_vs_f16_x", "time_vs_f16_x_stdev",
]


def format_passthrough_value(v):
    v = (v or "").strip().strip('"')
    return v if v else "-"


def normalize_bpw_for_ref(cache_k, cache_v, bpw):
    if cache_k.startswith(REF_PREFIX) or cache_v.startswith(REF_PREFIX):
        return "-"
    return bpw


def normalize_nc_for_ref(cache_k, cache_v, norm_correction):
    if cache_k.startswith(REF_PREFIX) or cache_v.startswith(REF_PREFIX):
        return "on"
    return norm_correction


def _new_derived_group() -> DerivedGroup:
    return {
        "ppl_means": [],
        "ppl_stdevs": [],
        "time_means": [],
        "time_stdevs": [],
        "passthrough": None,
        "mixed": "",
        "bpw_avgs": [],
        "cache_k": "",
        "cache_v": "",
    }


def make_derived_pattern_groups(groups: dict[tuple[str, ...], PerpGroup]) -> dict[tuple[str, ...], DerivedGroup]:
    derived: dict[tuple[str, ...], DerivedGroup] = defaultdict(_new_derived_group)

    for (config, ck, cv, nc), g in groups.items():
        pt = g["passthrough"] or {}
        mixed = pt.get("mixed", "-")
        if ck.startswith("tbq") and cv.startswith("pq"):
            key = (config, "tbq*", "pq*", nc, mixed)
            d = derived[key]
            d["cache_k"] = "tbq*"
            d["cache_v"] = "pq*"
        elif ck.startswith("pq") and cv.startswith("pq"):
            key = (config, "pq*", "pq*", nc, mixed)
            d = derived[key]
            d["cache_k"] = "pq*"
            d["cache_v"] = "pq*"
        else:
            continue

        d["ppl_means"].extend(g["ppl_means"])
        d["ppl_stdevs"].extend(g["ppl_stdevs"])
        d["time_means"].extend(g["time_means"])
        d["time_stdevs"].extend(g["time_stdevs"])
        if d["passthrough"] is None:
            d["passthrough"] = pt
        d["mixed"] = mixed

        try:
            bpw_value = float(pt.get("bpw_avg", ""))
        except (TypeError, ValueError):
            bpw_value = None
        if bpw_value is not None:
            d["bpw_avgs"].append(bpw_value)

    return derived


def format_bpw_or_dash(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"{sum(values) / len(values):.2f}"


def combined_mean_stdev(means, stdevs):
    """Combine per-run mean±stdev into an aggregate mean and stdev.

    Uses the law of total variance:
        combined_var = mean(individual_variances) + var(individual_means)
    """
    n = len(means)
    if n == 0:
        return None, None
    grand_mean = sum(means) / n
    if n == 1:
        return grand_mean, stdevs[0] if stdevs[0] is not None else 0.0

    mean_of_vars = sum((s ** 2 if s is not None else 0.0) for s in stdevs) / n
    var_of_means = sum((m - grand_mean) ** 2 for m in means) / (n - 1)
    combined_sd = math.sqrt(mean_of_vars + var_of_means)
    return grand_mean, combined_sd


def _new_perp_group() -> PerpGroup:
    return {
        "ppl_means": [], "ppl_stdevs": [],
        "time_means": [], "time_stdevs": [],
        "passthrough": None,
    }


def aggregate(input_files: list[str]) -> dict[tuple[str, ...], PerpGroup]:
    groups: dict[tuple[str, ...], PerpGroup] = defaultdict(_new_perp_group)

    for fpath in input_files:
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = row.get("config", "").strip().strip('"')
                cache_k = row.get("cache_k", "").strip().strip('"')
                cache_v = row.get("cache_v", "").strip().strip('"')
                norm_correction = normalize_nc_for_ref(
                    cache_k,
                    cache_v,
                    row.get("norm_correction", "").strip().strip('"'),
                )
                key = (config, cache_k, cache_v, norm_correction)

                ppl_mean = safe_float(row.get("ppl_mean"))
                ppl_sd = safe_float(row.get("ppl_stdev"))
                time_mean = safe_float(row.get("time_mean_s"))
                time_sd = safe_float(row.get("time_stdev_s"))

                if ppl_mean is None:
                    continue

                g = groups[key]
                g["ppl_means"].append(ppl_mean)
                g["ppl_stdevs"].append(ppl_sd)
                if time_mean is not None:
                    g["time_means"].append(time_mean)
                    g["time_stdevs"].append(time_sd)

                if g["passthrough"] is None:
                    passthrough = {k: format_passthrough_value(row.get(k, "")) for k in PASSTHROUGH}
                    passthrough["bpw_avg"] = normalize_bpw_for_ref(
                        cache_k,
                        cache_v,
                        passthrough.get("bpw_avg", "-"),
                    )
                    g["passthrough"] = passthrough

    return groups


def compute_ratios(groups):
    """Recompute vs-f16 ratios from aggregated values."""
    f16_lookup = {}
    for key, g in groups.items():
        config, ck, cv, nc = key
        if ck == "f16" and cv == "f16":
            f16_lookup[(config, nc)] = g

    derived = make_derived_pattern_groups(groups)

    results = []

    for key in groups:
        config, ck, cv, nc = key
        g = groups[key]

        ppl_mean, ppl_sd = combined_mean_stdev(g["ppl_means"], g["ppl_stdevs"])
        n_runs = len(g["ppl_means"])

        time_mean, time_sd = None, None
        if g["time_means"]:
            time_mean, time_sd = combined_mean_stdev(g["time_means"], g["time_stdevs"])

        ppl_vs_f16 = ""
        ppl_vs_f16_sd = ""
        time_vs_f16 = ""
        time_vs_f16_sd = ""

        f16 = f16_lookup.get((config, nc))
        if f16 is None:
            f16 = f16_lookup.get((config, "off"))

        if f16 is not None:
            f16_ppl, _ = combined_mean_stdev(f16["ppl_means"], f16["ppl_stdevs"])
            if f16_ppl and f16_ppl > 0 and ppl_mean is not None:
                pct = ((ppl_mean - f16_ppl) / f16_ppl) * 100
                ppl_vs_f16 = f"{pct:.2f}"
                if ppl_sd is not None:
                    ppl_vs_f16_sd = f"{(ppl_sd / f16_ppl) * 100:.2f}"

            if time_mean is not None and f16["time_means"]:
                f16_time, f16_time_sd = combined_mean_stdev(
                    f16["time_means"], f16["time_stdevs"])
                if f16_time and f16_time > 0:
                    ratio = time_mean / f16_time
                    time_vs_f16 = f"{ratio:.2f}"
                    if time_sd is not None and f16_time_sd is not None and f16_time > 0:
                        a = (time_sd / f16_time) ** 2
                        b = (time_mean * f16_time_sd / (f16_time ** 2)) ** 2
                        time_vs_f16_sd = f"{math.sqrt(a + b):.2f}"

        pt = g["passthrough"] or {}
        results.append({
            "gpu_device": pt.get("gpu_device", ""),
            "model": pt.get("model", ""),
            "cache_k": ck,
            "cache_v": cv,
            "mixed": pt.get("mixed", ""),
            "norm_correction": nc,
            "bpw_avg": pt.get("bpw_avg", ""),
            "n_chunks": pt.get("n_chunks", ""),
            "n_ctx_sweep": pt.get("n_ctx_sweep", ""),
            "n_runs": str(n_runs),
            "ppl_mean": f"{ppl_mean:.4f}" if ppl_mean is not None else "",
            "ppl_stdev": f"{ppl_sd:.4f}" if ppl_sd is not None else "",
            "time_mean_s": f"{time_mean:.2f}" if time_mean is not None else "",
            "time_stdev_s": f"{time_sd:.2f}" if time_sd is not None else "",
            "ppl_vs_f16_pct": ppl_vs_f16,
            "ppl_vs_f16_pct_stdev": ppl_vs_f16_sd,
            "time_vs_f16_x": time_vs_f16,
            "time_vs_f16_x_stdev": time_vs_f16_sd,
        })

    for dkey in derived:
        config, ck, cv, nc, mixed = dkey
        dg = derived[dkey]
        ck_s = dg["cache_k"]
        cv_s = dg["cache_v"]
        ppl_mean, ppl_sd = combined_mean_stdev(dg["ppl_means"], dg["ppl_stdevs"])
        n_runs = len(dg["ppl_means"])

        time_mean, time_sd = None, None
        if dg["time_means"]:
            time_mean, time_sd = combined_mean_stdev(dg["time_means"], dg["time_stdevs"])

        ppl_vs_f16 = ""
        ppl_vs_f16_sd = ""
        time_vs_f16 = ""
        time_vs_f16_sd = ""

        f16 = f16_lookup.get((config, nc))
        if f16 is None:
            f16 = f16_lookup.get((config, "off"))

        if f16 is not None:
            f16_ppl, _ = combined_mean_stdev(f16["ppl_means"], f16["ppl_stdevs"])
            if f16_ppl and f16_ppl > 0 and ppl_mean is not None:
                pct = ((ppl_mean - f16_ppl) / f16_ppl) * 100
                ppl_vs_f16 = f"{pct:.2f}"
                if ppl_sd is not None:
                    ppl_vs_f16_sd = f"{(ppl_sd / f16_ppl) * 100:.2f}"

            if time_mean is not None and f16["time_means"]:
                f16_time, f16_time_sd = combined_mean_stdev(
                    f16["time_means"], f16["time_stdevs"])
                if f16_time and f16_time > 0:
                    ratio = time_mean / f16_time
                    time_vs_f16 = f"{ratio:.2f}"
                    if time_sd is not None and f16_time_sd is not None and f16_time > 0:
                        a = (time_sd / f16_time) ** 2
                        b = (time_mean * f16_time_sd / (f16_time ** 2)) ** 2
                        time_vs_f16_sd = f"{math.sqrt(a + b):.2f}"

        pt = dg["passthrough"] or {}
        bpw = format_bpw_or_dash(dg["bpw_avgs"])
        results.append({
            "gpu_device": pt.get("gpu_device", ""),
            "model": pt.get("model", ""),
            "cache_k": ck_s,
            "cache_v": cv_s,
            "mixed": mixed,
            "norm_correction": nc,
            "bpw_avg": bpw,
            "n_chunks": pt.get("n_chunks", ""),
            "n_ctx_sweep": pt.get("n_ctx_sweep", ""),
            "n_runs": str(n_runs),
            "ppl_mean": f"{ppl_mean:.4f}" if ppl_mean is not None else "",
            "ppl_stdev": f"{ppl_sd:.4f}" if ppl_sd is not None else "",
            "time_mean_s": f"{time_mean:.2f}" if time_mean is not None else "",
            "time_stdev_s": f"{time_sd:.2f}" if time_sd is not None else "",
            "ppl_vs_f16_pct": ppl_vs_f16,
            "ppl_vs_f16_pct_stdev": ppl_vs_f16_sd,
            "time_vs_f16_x": time_vs_f16,
            "time_vs_f16_x_stdev": time_vs_f16_sd,
        })

    return results


DISPLAY_COLS = [
    ("cache_k", "K", 10),
    ("cache_v", "V", 10),
    ("norm_correction", "NC", 4),
    ("bpw_avg", "BPW", 6),
    ("n_runs", "Runs", 5),
    ("ppl_mean", "PPL mean", 10),
    ("ppl_stdev", "PPL sd", 8),
    ("ppl_vs_f16_pct", "vs f16%", 8),
    ("ppl_vs_f16_pct_stdev", "±%", 6),
    ("time_mean_s", "Time(s)", 8),
    ("time_stdev_s", "Time sd", 8),
    ("time_vs_f16_x", "t/f16x", 7),
]


def main():
    run_aggregation(
        description="Aggregate KV cache quantization perplexity CSVs",
        title=" KV Cache Quantization Perplexity — Aggregated Results",
        output_header=OUTPUT_HEADER,
        aggregate=aggregate,
        compute_ratios=compute_ratios,
        render=lambda rows: render_table(rows, DISPLAY_COLS),
    )


if __name__ == "__main__":
    main()
