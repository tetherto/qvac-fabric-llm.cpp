#!/usr/bin/env python3
"""Summarize one weekly run directory into summary.json (schema 3, shared by the Mac, Strix and Spark benches).
Spark copy: the reference with the Spark's constants, a Linux hardware(), acceptance for vLLM and SGLang from their
/metrics counters, and two extra keys (checks, provenance) that stay out of the shared datasets.

usage: summarize.py <run dir> [--status ok|failed] [--reason TEXT]

Results come from <run dir>/results.jsonl, written by bench_client.py under labels <kind>-<engine>-<mode>[-<profile>]:
conc (no speculation or Splash's DFlash2, C = 1..16), concspec (MTP and DFlash2, C = 1, 4, 8), spec (one request,
four prompts) and depth (one request per prompt length). The profile defaults to "standard"; a machine may add others
(the Spark's "heavy"). llama-bench output is bench-<engine>.jsonl.

Problems are listed, not hidden: `errors` copies <run dir>/errors.jsonl (run_engine.sh's rows: label, step, reason in
"error", time in "at"), and any error other than a failed warm-up makes an "ok" run "partial". `short_by_label` names
every conc, concspec and depth row whose requests stopped before --gen tokens (label, concurrency, mean prompt tokens,
short of total), and `short_outputs` is their sum. `jobs` maps each label to the time its job started (weekly.log), so
drops, errors and short outputs can be matched to a time window.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict

MACHINE = "dgx-spark"
GGUF_W = "NVFP4 GGUF via convert_hf_to_gguf from the ModelOpt checkpoint (attention Q8_0)"
NAMES = {"up": "llama.cpp master", "fab": "qvac-fabric PR stack", "fab0": "qvac-fabric PR stack, deferred off",
         "vllm": "vLLM", "sglang": "SGLang"}
BACKEND = {"up": "cuda", "fab": "cuda", "fab0": "cuda", "vllm": "cuda", "sglang": "cuda"}
WEIGHTS = {"up": GGUF_W, "fab": GGUF_W, "fab0": GGUF_W, "vllm": "NVFP4 ModelOpt", "sglang": "NVFP4 ModelOpt"}
DRAFT = {**{(e, "mtp"): "GGUF MTP layer (Q8_0)" for e in ("up", "fab", "fab0")},
         **{(e, "dflash2"): "DFlash2 Q8_0 GGUF" for e in ("up", "fab", "fab0")},
         **{(e, "mtp"): "model MTP head (bf16)" for e in ("vllm", "sglang")},
         **{(e, "dflash2"): "DFlash2 bf16" for e in ("vllm", "sglang")}}
# the standard matrix weekly.sh runs: (kind, engine, mode); the heavy profile is on demand and never counts as missing
EXPECTED = [("spec", e, m) for e in ("up", "fab", "vllm", "sglang") for m in ("ar", "mtp", "dflash2")] \
    + [("spec", "fab0", "mtp"), ("spec", "fab0", "dflash2")] \
    + [("conc", e, "ar") for e in ("up", "fab", "vllm", "sglang")] \
    + [("concspec", e, m) for e in ("up", "fab", "vllm", "sglang") for m in ("mtp", "dflash2")] \
    + [("depth", e, "ar") for e in ("up", "fab", "vllm", "sglang")]
DRAFT_LEN = {"mtp": 3, "dflash2": 7}  # Spark: tokens drafted per verify step, for SGLang's acceptance
AR_OF = {"fab0": "fab"}  # Spark: the run without speculation a speculative run must reproduce
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
    # Spark: Linux and nvidia-smi instead of system_profiler
    gpu = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").split(", ")
    cuda = re.search(r"CUDA Version: ([0-9.]+)", sh("nvidia-smi"))
    mem = re.search(r"MemTotal:\s+(\d+) kB", open("/proc/meminfo").read())
    osr = re.search(r'PRETTY_NAME="([^"]+)"', open("/etc/os-release").read())
    name = gpu[0] if gpu and gpu[0] else None
    return {
        "chip": f"{name} (Grace Blackwell)" if name else None,
        "gpu": ", ".join(x for x in (name, f"driver {gpu[1]}" if len(gpu) > 1 else None, f"CUDA {cuda.group(1)}" if cuda else None) if x),
        # physical size: MemTotal rounded to the 8 GB the modules come in
        "memory_gb": int(round(int(mem.group(1)) * 1024 / 1e9 / 8) * 8) if mem else None,
        "os": f"{osr.group(1)}, kernel {sh('uname -r')}" if osr else None,
    }


def versions(run):
    env = {}
    path = os.path.join(run, "engines.env")
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"""(\w+)=['"](.*)['"]$""", line.strip())
            if m:
                env[m.group(1)] = m.group(2)
    keys = {"up": "UP_VER", "fab": "FAB_VER", "fab0": "FAB_VER", "vllm": "VLLM_VER", "sglang": "SGL_VER"}
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
    stats = defaultdict(lambda: defaultdict(float))  # Spark: vLLM/SGLang /metrics counter deltas per label
    texts = defaultdict(dict)  # Spark: greedy text crc per (engine, mode, profile) and (prompt, rep)
    for r in read_jsonl(os.path.join(run, "results.jsonl")):
        if r.get("mode") == "specstats":
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    stats[r["label"]][k] += v
            continue
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
            texts[(e, m, prof)][(r["prompt"], r["rep"])] = r.get("text_crc")
            if r.get("draft_n"):
                acc[(e, m, prof)][0] += r["draft_acc"]
                acc[(e, m, prof)][1] += r["draft_n"]
            seen.add((kind, e, m))

    def counters_acc(e, m, label):
        # Spark: acceptance % for vLLM (accepted / drafted tokens) and SGLang (from tokens per verify step)
        t = stats.get(label) or {}
        if e == "vllm" and t.get("draft_tokens"):
            return round(100 * t["accepted"] / t["draft_tokens"], 1)
        if e == "sglang" and t.get("verify") and m in DRAFT_LEN:
            return round(100 * (t["gen"] / t["verify"] - 1) / DRAFT_LEN[m], 1)
        return None

    spec_rows = []
    for (e, m, prof), per in spec.items():
        vals = [v for p in per.values() for v in p]
        a2 = acc[(e, m, prof)]
        sfx = "" if prof == "standard" else f"-{prof}"
        row = {"engine": e, "backend": BACKEND.get(e), "mode": m, "profile": prof, "draft": DRAFT.get((e, m), ""),
               "mean": round(statistics.mean(vals), 2), "runs": len(vals),
               "acc": round(100 * a2[0] / a2[1], 1) if a2[1] else counters_acc(e, m, f"spec-{e}-{m}{sfx}")}
        row.update({p: round(statistics.mean(per[p]), 1) if per.get(p) else None for p in PROMPTS})
        spec_rows.append(row)

    # Spark: greedy speculation should reproduce the same engine's text without it; engines on one weights file should agree
    checks = []

    def compare(key, ref):
        pairs = [(v, texts[ref][k]) for k, v in texts.get(key, {}).items() if k in texts.get(ref, {})]
        if pairs:
            checks.append({"engine": key[0], "mode": key[1], "profile": key[2], "reference": f"{ref[0]} {ref[1]}",
                           "same": sum(1 for x, y in pairs if x == y), "of": len(pairs)})
    for (e, m, prof) in sorted(texts):
        if m != "ar":
            compare((e, m, prof), (AR_OF.get(e, e), "ar", prof))
    for x, y in (("fab", "up"), ("sglang", "vllm"), ("vllm", "up")):
        compare((x, "ar", "standard"), (y, "ar", "standard"))
    concspec_acc = {}
    for lab in sorted({lab for lab in stats if lab.startswith("concspec-")}):
        k = lab.split("-")
        if k[1] in ("vllm", "sglang"):
            concspec_acc[f"{k[1]}/{k[2]}" + ("" if len(k) < 4 else f"/{k[3]}")] = counters_acc(k[1], k[2], lab)

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
        # Spark extras beside schema 3: not in the shared datasets, read by the publish routine's report
        "checks": checks,
        "concspec_acc": concspec_acc,
        "host": sh("hostname"),
        "provenance": (json.load(open(os.path.join(run, "weights.json"))).get("provenance", {})
                       if os.path.exists(os.path.join(run, "weights.json")) else {}),
    }
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("date", "status", "missing")}))  # noqa: NP100


if __name__ == "__main__":
    main()
