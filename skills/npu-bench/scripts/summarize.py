#!/usr/bin/env python3
"""One run directory of bench_npu.py records -> summary.json (cells, token counts, drift check, run metadata).

usage: summarize.py <run dir> [--status ok|partial|failed] [--reason text]

Reads <run>/results.jsonl (labels <model>|<runtime>|<session>), <run>/run.json (metadata from weekly.sh),
<run>/foreign.log and <run>/errors.log when present. Writes <run>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

SCHEMA = 1
RUNTIMES = ("xdna", "flm")
DRIFT_SESSION = "drift"
DRIFT_RUNTIME = "xdna"
DRIFT_METRICS = ("prefill", "decode", "dj")
DIGITS = {"prefill": 1, "decode": 1, "ttft": 3, "req": 2, "pj": 2, "dj": 4, "tpj": 2}
TOKEN_DIGITS = 1


def mean_sd(values):
    if not values:
        return None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return [statistics.mean(values), sd]


def rounded(cell, digits):
    return None if cell is None else [round(cell[0], digits), round(cell[1], digits)]


def timed_requests(records):
    return [q for r in records if r["kind"] == "full" for q in r["reqs"] if q["ttft"]]


def prompt_tokens(record):
    return sum(q["prompt_tokens"] for q in record["reqs"])


def decode_tokens(record):
    return sum(q["completion_tokens"] - 1 for q in record["reqs"])


def prefill_energy(bursts, tiny_j):
    """J per 1k prompt tokens of each one-token burst, without the fixed cost of one energy window."""
    return mean_sd([(r["energy_j"] - tiny_j) / prompt_tokens(r) * 1000 for r in bursts])


def decode_energy(fulls, tiny_j, prefill_j_per_token):
    """J per generated token: a full request minus the window cost and its prompt at the prefill rate."""
    return mean_sd([(r["energy_j"] - tiny_j - prefill_j_per_token * prompt_tokens(r)) / decode_tokens(r) for r in fulls])


def cell_metrics(records, tiny):
    """records: one (session, ctx) cell; tiny: the tiny-request windows of the same session."""
    fulls = [r for r in records if r["kind"] == "full"]
    bursts = [r for r in records if r["kind"] == "burst"]
    timed = timed_requests(records)
    out = {
        "prefill": mean_sd([q["prompt_tokens"] / q["ttft"] for q in timed]),
        "decode": mean_sd([q["completion_tokens"] / (q["total"] - q["ttft"]) for q in timed]),
        "ttft": mean_sd([q["ttft"] for q in timed]),
        "req": mean_sd([r["energy_j"] for r in fulls]),
        "pj": None, "dj": None, "tpj": None,
    }
    if tiny and bursts:
        tiny_j = statistics.mean(r["energy_j"] for r in tiny)
        pj = prefill_energy(bursts, tiny_j)
        out["pj"] = pj
        dj = decode_energy(fulls, tiny_j, pj[0] / 1000) if pj and fulls else None
        if dj is not None and dj[0] > 0:
            out["dj"] = dj
            out["tpj"] = [1 / dj[0], dj[1] / dj[0] ** 2]
    tokens = statistics.mean(q["completion_tokens"] for q in timed) if timed else None
    return {k: rounded(v, DIGITS[k]) for k, v in out.items()}, tokens


def split_label(label):
    model, runtime, session = label.split("|")
    return model, runtime, session


def winning_sessions(records):
    """(model, runtime, ctx) -> label of the last session that measured it, so a re-run replaces a disturbed cell."""
    win = {}
    for r in records:
        if r["kind"] == "tiny":
            continue
        model, runtime, session = split_label(r["label"])
        if session != DRIFT_SESSION:
            win[(model, runtime, r["ctx"])] = r["label"]
    return win


def records_of(records, label, ctx=None, kind=None):
    return [r for r in records if r["label"] == label and (ctx is None or r["ctx"] == ctx)
            and (kind is None or r["kind"] == kind)]


def ordered_unique(values):
    return list(dict.fromkeys(values))


def build_cells(records, models, contexts):
    win = winning_sessions(records)
    cells = {m: {k: {rt: [None] * len(contexts) for rt in RUNTIMES} for k in DIGITS} for m in models}
    tokens = {m: {rt: [None] * len(contexts) for rt in RUNTIMES} for m in models}
    for (model, runtime, ctx), label in win.items():
        if model not in cells or runtime not in RUNTIMES or ctx not in contexts:
            continue
        i = contexts.index(ctx)
        metrics, tok = cell_metrics(records_of(records, label, ctx), records_of(records, label, kind="tiny"))
        for k, v in metrics.items():
            cells[model][k][runtime][i] = v
        tokens[model][runtime][i] = None if tok is None else round(tok, TOKEN_DIGITS)
    return cells, tokens


def relative_change(cell, base):
    return round(cell[0] / base[0] - 1, 4) if cell and base else None


def drift_checks(records, cells, contexts):
    """Relative change of the drift session (repeated after flm) against the main xdna cell at the same context."""
    out = {}
    for label in ordered_unique(r["label"] for r in records):
        model, runtime, session = split_label(label)
        if session != DRIFT_SESSION or runtime != DRIFT_RUNTIME or model not in cells:
            continue
        ctx = records_of(records, label)[0]["ctx"]
        metrics, _ = cell_metrics(records_of(records, label, ctx), records_of(records, label, kind="tiny"))
        main = {k: cells[model][k][runtime][contexts.index(ctx)] for k in DRIFT_METRICS}
        out[model] = {"ctx": ctx, **{k: relative_change(metrics[k], main[k]) for k in DRIFT_METRICS}}
    return out


def missing_cells(cells, models, contexts):
    return [f"{m}|{rt}|{c}" for m in models for rt in RUNTIMES for i, c in enumerate(contexts)
            if cells[m]["decode"][rt][i] is None]


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_records(path):
    return [json.loads(line) for line in read_lines(path)]


def summarize(run_dir, status, reason):
    records = read_records(os.path.join(run_dir, "results.jsonl"))
    meta_path = os.path.join(run_dir, "run.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    measured = [r for r in records if r["kind"] != "tiny"]
    models = meta.get("models_order") or ordered_unique(split_label(r["label"])[0] for r in measured)
    contexts = meta.get("contexts") or sorted({r["ctx"] for r in measured})
    cells, tokens = build_cells(records, models, contexts)
    missing = missing_cells(cells, models, contexts)
    if status == "ok" and missing:
        status = "partial"
    return {
        "schema": SCHEMA, **{k: v for k, v in meta.items() if k not in ("models_order", "contexts")},
        "status": status, "reason": reason, "contexts": contexts, "models_order": models, "runtimes": list(RUNTIMES),
        "cells": cells, "tokens": tokens, "drift": drift_checks(records, cells, contexts), "missing": missing,
        "foreign": read_lines(os.path.join(run_dir, "foreign.log")),
        "errors": read_lines(os.path.join(run_dir, "errors.log")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--status", default="ok", choices=["ok", "partial", "failed"])
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    summary = summarize(a.run_dir, a.status, a.reason)
    with open(os.path.join(a.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    sys.stdout.write(json.dumps({k: summary[k] for k in ("status", "missing")}) + "\n")


if __name__ == "__main__":
    main()
