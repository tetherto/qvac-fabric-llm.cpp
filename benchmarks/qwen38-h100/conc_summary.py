#!/usr/bin/env python3
"""Summarize conc_bench.py results: per concurrency level, means over reps of the aggregate rates; optional B/A ratio.

usage: conc_summary.py <label>=<file-or-glob> [<label>=<file-or-glob> ...]
Every run in the named files is grouped by concurrency; the first label is the ratio base.
"""

import glob
import json
import statistics
import sys


def load(spec):
    label, pattern = spec.split("=", 1)
    runs = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for r in json.load(f)["runs"]:
                s = r["summary"]
                runs.setdefault(s["n"], []).append(s)
    return label, runs


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def fmt(v, nd=0):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def main():
    engines = [load(spec) for spec in sys.argv[1:]]
    keys = [
        ("prefill_agg_tok_s", "prefill agg tok/s", 0),
        ("decode_agg_tok_s", "decode agg tok/s", 1),
        ("output_tok_s", "output tok/s", 1),
        ("total_tok_s", "total tok/s", 0),
        ("ttft_mean_s", "TTFT mean s", 2),
        ("ttft_max_s", "TTFT max s", 2),
        ("stream_decode_tok_s_mean", "per-stream decode tok/s", 1),
        ("t_end_s", "wall s", 1),
        ("streams_at_all_first", "streams at all-first", 1),
    ]
    levels = sorted({n for _, runs in engines for n in runs})
    base_label = engines[0][0]
    for key, title, nd in keys:
        print(f"== {title}")
        head = "| N | " + " | ".join(label for label, _ in engines)
        if len(engines) > 1:
            head += " | " + " | ".join(f"{label}/{base_label}" for label, _ in engines[1:])
        print(head + " |")
        print("|" + "---|" * (1 + len(engines) + max(0, len(engines) - 1)))
        for n in levels:
            cells = [str(n)]
            vals = [mean(runs.get(n, []), key) for _, runs in engines]
            cells += [fmt(v, nd) for v in vals]
            if len(engines) > 1:
                b = vals[0]
                cells += [fmt(v / b, 2) if (v is not None and b) else "-" for v in vals[1:]]
            print("| " + " | ".join(cells) + " |")
        print()
    print("== reps (per level: rep values of prefill agg / decode agg / output tok/s, load before)")
    for label, runs in engines:
        for n in levels:
            rows = runs.get(n, [])
            if rows:
                print(
                    f"{label} n={n}: "
                    + "; ".join(
                        f"{fmt(r['prefill_agg_tok_s'])} / {fmt(r.get('decode_agg_tok_s'), 1)} / "
                        f"{fmt(r['output_tok_s'], 1)} (load {fmt(r.get('load_before'), 1)})"
                        for r in rows
                    )
                )


if __name__ == "__main__":
    main()
