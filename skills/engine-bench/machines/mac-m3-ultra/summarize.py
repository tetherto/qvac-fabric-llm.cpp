#!/usr/bin/env python3
"""Summarize one weekly run directory into summary.json (schema 3, shared by the Mac, Strix and Spark benches).

usage: summarize.py <run dir> [--status ok|failed] [--reason TEXT]

Results come from <run dir>/results.jsonl, written by bench_client.py under labels <kind>-<engine>-<mode>[-<profile>]:
conc (no speculation or Splash's DFlash2, C = 1..16), concspec (MTP and DFlash2, C = 1, 4, 8), spec (one request,
four prompts) and depth (one request per prompt length). The profile defaults to "standard"; a machine may add others
(the Spark's "heavy"). llama-bench output is bench-<engine>.jsonl.

Problems are listed, not hidden: `errors` copies <run dir>/errors.jsonl (run_engine.sh's rows: label, step, reason in
"error", time in "at"), and any error other than a failed warm-up makes an "ok" run "partial". `short_by_label` names
every conc, concspec and depth row whose requests stopped before --gen tokens (label, concurrency, mean prompt tokens,
short of total), and `short_outputs` is their sum. `jobs` maps each label to the time its server came up (weekly.log),
so drops, errors and short outputs can be matched to a time window.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict

MACHINE = "mac-m3-ultra"
NAMES = {"up": "llama.cpp master", "fab": "qvac-fabric PR stack", "mlx": "mlx-lm", "omlx": "oMLX",
         "sglang": "SGLang", "splash": "Splash"}
BACKEND = {"up": "metal", "fab": "metal", "mlx": "mlx", "omlx": "mlx", "sglang": "mlx", "splash": "metal"}
WEIGHTS = {"up": "Q4_0 GGUF (pure)", "fab": "Q4_0 GGUF (pure)", "mlx": "MLX 4-bit g64", "omlx": "MLX 4-bit g64",
           "sglang": "MLX 4-bit g64", "splash": "MLX 4-bit g64"}
DRAFT = {("up", "mtp"): "GGUF MTP layer", ("fab", "mtp"): "GGUF MTP layer",
         ("up", "dflash2"): "DFlash2 Q4_0 GGUF", ("fab", "dflash2"): "DFlash2 Q4_0 GGUF",
         ("omlx", "mtp"): "MTP head 4-bit (mlx-community)", ("omlx", "dflash2"): "DFlash2 bf16",
         ("splash", "dflash2"): "DFlash2 bf16, 4-bit at load"}
# the matrix weekly.sh runs: (kind, engine, mode)
EXPECTED = [("conc", e, m) for e, m in (("omlx", "ar"), ("sglang", "ar"), ("splash", "dflash2"), ("fab", "ar"), ("up", "ar"), ("mlx", "ar"))] \
    + [("spec", e, m) for e, m in (("up", "ar"), ("up", "mtp"), ("up", "dflash2"), ("fab", "ar"), ("fab", "mtp"), ("fab", "dflash2"),
                                   ("omlx", "ar"), ("omlx", "mtp"), ("omlx", "dflash2"), ("splash", "dflash2"), ("mlx", "ar"), ("sglang", "ar"))] \
    + [("concspec", e, m) for e in ("up", "fab", "omlx") for m in ("mtp", "dflash2")] \
    + [("depth", e, m) for e, m in (("up", "ar"), ("fab", "ar"), ("omlx", "ar"), ("mlx", "ar"), ("sglang", "ar"), ("splash", "dflash2"),
                                    ("up", "dflash2"), ("fab", "dflash2"), ("omlx", "dflash2"))]
PROMPTS = ["code", "rewrite", "story", "explain"]


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def short_row(r):
    """The requests of one conc or depth row that stopped before --gen tokens."""
    return {"label": r["label"], "c": r.get("conc"), "prompt": round(r["prompt_tok"] / r["n_req"]) if r.get("n_req") else None,
            "short": r["short_outputs"], "of": r.get("n_req")}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def hardware():
    hw = sh("system_profiler SPHardwareDataType")
    chip = re.search(r"Chip: (.+)", hw)
    mem = re.search(r"Memory: (\d+) GB", hw)
    gpu = re.findall(r"Total Number of Cores: (\d+)", sh("system_profiler SPDisplaysDataType"))
    return {
        "chip": chip.group(1).strip() if chip else None,
        "gpu": f"{gpu[0]}-core GPU" if gpu else None,
        "memory_gb": int(mem.group(1)) if mem else None,
        "os": "macOS " + sh("sw_vers -productVersion"),
    }


def versions(run):
    env = {}
    path = os.path.join(run, "engines.env")
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"""(\w+)=['"](.*)['"]$""", line.strip())
            if m:
                env[m.group(1)] = m.group(2)
    keys = {"up": "UP_VER", "fab": "FAB_VER", "mlx": "MLX_VER", "omlx": "OMLX_VER", "sglang": "SGL_VER", "splash": "SPLASH_VER"}
    return [{"engine": e, "name": NAMES[e], "version": env.get(k), "backend": BACKEND[e], "weights": WEIGHTS[e]} for e, k in keys.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--status", default="ok")
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    run = a.run.rstrip("/")

    single = []
    for e in ("up", "fab"):
        row = {"engine": e, "backend": BACKEND[e], "tg128": None, "pp512": None}
        for r in read_jsonl(os.path.join(run, f"bench-{e}.jsonl")):
            if r.get("n_prompt") == 0:
                row["tg128"] = round(r["avg_ts"], 2)
            elif r.get("n_gen") == 0:
                row[f"pp{r['n_prompt']}"] = round(r["avg_ts"], 2)
        if row["tg128"] is not None or row["pp512"] is not None:
            single.append(row)

    conc, depth, seen, short = [], [], set(), []
    spec = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: [0, 0])
    for r in read_jsonl(os.path.join(run, "results.jsonl")):
        parts = r.get("label", "").split("-")
        if len(parts) not in (3, 4):
            continue
        kind, e, m = parts[:3]
        prof = parts[3] if len(parts) == 4 else "standard"
        base = {"engine": e, "backend": BACKEND.get(e), "mode": m, "profile": prof}
        if r.get("mode") == "conc" and kind in ("conc", "concspec", "depth") and r.get("short_outputs"):
            short.append(short_row(r))
        if r.get("mode") == "conc" and kind in ("conc", "concspec"):
            conc.append({**base, "draft": DRAFT.get((e, m), ""), "prompt": round(r["prompt_tok"] / r["n_req"]) if r.get("n_req") else None,
                         "c": r["conc"], "gen": r["agg_gen_tps"], "req": r["per_req_tps_mean"], "ttft": r["ttft_mean_s"], "p90": r["ttft_p90_s"]})
            seen.add((kind, e, m))
        elif r.get("mode") == "conc" and kind == "depth":
            d = round(r["prompt_tok"] / r["n_req"]) if r.get("n_req") else None
            depth.append({**base, "depth": d, "pp": round(d / r["ttft_mean_s"], 1) if d and r.get("ttft_mean_s") else None,
                          "tg": r["per_req_tps_mean"], "ttft": r["ttft_mean_s"]})
            seen.add((kind, e, m))
        elif r.get("mode") == "spec" and kind == "spec":
            # a request that came back empty has no decode rate (client_check.py reports it as an error)
            if r.get("dec_tps") is not None:
                spec[(e, m, prof)][r["prompt"]].append(r["dec_tps"])
            if r.get("draft_n"):
                acc[(e, m, prof)][0] += r["draft_acc"]
                acc[(e, m, prof)][1] += r["draft_n"]
            seen.add((kind, e, m))

    spec_rows = []
    for (e, m, prof), per in spec.items():
        vals = [v for p in per.values() for v in p]
        a2 = acc[(e, m, prof)]
        row = {"engine": e, "backend": BACKEND.get(e), "mode": m, "profile": prof, "draft": DRAFT.get((e, m), ""),
               "mean": round(statistics.mean(vals), 2), "runs": len(vals),
               "acc": round(100 * a2[0] / a2[1], 1) if a2[1] else None}
        row.update({p: round(statistics.mean(per[p]), 1) if per.get(p) else None for p in PROMPTS})
        spec_rows.append(row)

    status = a.status
    missing = [f"{k}:{e}/{m}" for k, e, m in EXPECTED if (k, e, m) not in seen]
    errors = read_jsonl(os.path.join(run, "errors.jsonl"))
    if status == "ok" and (missing or any(x.get("step") != "warm-up" for x in errors)):
        status = "partial"
    started, jobs = None, {}
    log = os.path.join(run, "weekly.log")
    if os.path.exists(log):
        text = open(log, errors="replace").read()
        mm = re.search(r"^start (.+)$", text, re.M)
        started = mm.group(1) if mm else None
        for j in re.finditer(r"^== (\S+) (?:server up|job) (\d\d:\d\d:\d\d)$", text, re.M):
            jobs.setdefault(j.group(1), j.group(2))
    upd = os.path.join(run, "updates.txt")

    summary = {
        "schema": 3,
        "date": os.path.basename(run),
        "machine": MACHINE,
        "status": status,
        "reason": a.reason,
        "missing": missing,
        "started": started,
        "finished": time.strftime("%a %b %d %H:%M:%S %Z %Y"),
        "hardware": hardware(),
        "versions": versions(run),
        "updates": [line.rstrip("\n") for line in open(upd)] if os.path.exists(upd) else [],
        "single": single,
        "conc": sorted(conc, key=lambda r: (r["profile"], r["engine"], r["mode"], r["prompt"] or 0, r["c"])),
        "spec": sorted(spec_rows, key=lambda r: (r["profile"], r["engine"], r["mode"])),
        "depth": sorted(depth, key=lambda r: (r["profile"], r["engine"], r["mode"], r["depth"] or 0)),
        "short_outputs": sum(x["short"] for x in short),
        "short_by_label": short,
        "errors": errors,
        "jobs": jobs,
    }
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("date", "status", "missing")}))  # noqa: NP100


if __name__ == "__main__":
    main()
