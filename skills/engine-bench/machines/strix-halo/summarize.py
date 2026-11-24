#!/usr/bin/env python3
"""Summarize one weekly run directory on qvac-dev-strix-0 into summary.json (schema 3, shared by the Mac, Strix and
Spark benches).

usage: summarize.py <run dir> [--status ok|failed] [--reason TEXT]

Results come from <run dir>/results.jsonl, written by bench_client.py under labels
<kind>-<engine>-<mode>[-<profile>]-<backend>: conc (no speculation, C = 1, 4, 16), concspec (MTP and DFlash2, C = 4),
spec (one request, four prompts) and depth (one request per prompt length). The backend suffix extends the shared
labels because two backends of one engine can run here; the profile defaults to "standard". llama-bench output is
bench-<engine>-<backend>.jsonl. The engine:backend pairs run are CONFIGS in the run's matrix.conf.

Problems are listed, not hidden: `errors` copies <run dir>/errors.jsonl (run_engine.sh's rows: label, step, reason in
"error", time in "at"; skipped cells and notes go to `notes` instead), and any error other than a failed warm-up makes
an "ok" run "partial". `short_by_label` names every conc, concspec and depth row whose requests stopped before --gen
tokens (label, concurrency, mean prompt tokens, short of total), and `short_outputs` is their sum. `jobs` maps each
label to the time its server came up (weekly.log), so drops, errors and short outputs can be matched to a time window.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict

MACHINE = "strix-halo"
NAMES = {"up": "llama.cpp master", "strix": "strix-llama.cpp", "fab": "qvac-fabric PR stack",
         "nath": "Nathanw1014 strix-halo-vulkan", "paoai": "paoai-strix-engine", "gufo": "gufo"}
VER_KEYS = {"up": "UP_VER", "strix": "STRIX_VER", "fab": "FAB_VER", "nath": "NATH_VER", "paoai": "PAOAI_VER",
            "gufo": "GUFO_VER"}
BACKENDS = ("vulkan", "rocm")
WEIGHTS = "UD-Q4_K_XL GGUF @ 4ca7207"
DRAFT = {"mtp": "GGUF MTP layer (in the UD-Q4_K_XL file)", "dflash2": "DFlash2 Q4_K_M GGUF @ 2d9571f"}
UNSUPPORTED = {("gufo", "mtp")}  # gufo documents AR and DFlash2 only for Qwen3.8-27B
MODES = ["ar", "mtp", "dflash2"]
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
    cpu = re.search(r"Model name:\s+(.+)", sh("lscpu"))
    mem = re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read())
    vk = sh("vulkaninfo --summary 2>/dev/null")
    gpu = re.search(r"deviceName\s+=\s+(.+)", vk)
    mesa = re.search(r"driverInfo\s+=\s+(Mesa \S+)", vk)
    return {
        "chip": cpu.group(1).strip() if cpu else None,
        "gpu": (gpu.group(1).strip() if gpu else "Radeon 8060S") + (f", {mesa.group(1)}" if mesa else "") + ", ROCm 7.2.4",
        "memory_gb": round(int(mem.group(1)) / 1024 / 1024) if mem else None,
        "os": sh(". /etc/os-release && echo $PRETTY_NAME") + ", Linux " + sh("uname -r"),
    }


def read_env(run):
    env = {}
    for name in ("engines.env", "engine-opts.conf", "matrix.conf"):
        path = os.path.join(run, name)
        if os.path.exists(path):
            for line in open(path):
                m = re.match(r"""(\w+)=['"]?(.*?)['"]?$""", line.strip())
                if m:
                    env[m.group(1)] = m.group(2)
    return env


def split_label(label):
    """<kind>-<engine>-<mode>[-<profile>]-<backend> -> (kind, engine, mode, profile, backend) or None."""
    parts = label.split("-")
    be = parts.pop() if parts and parts[-1] in BACKENDS else None
    if len(parts) not in (3, 4) or be is None:
        return None
    return parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "standard", be


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--status", default="ok")
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    run = a.run.rstrip("/")
    env = read_env(run)
    configs = [tuple(c.split(":")) for c in env.get("CONFIGS", "").split()]
    extra = {(e, m): env.get(f"SPEC_EXTRA_{e.upper()}_{m.upper()}", "") for e in NAMES for m in ("mtp", "dflash2")}
    extra[("gufo", "dflash2")] = env.get("GUFO_SPEC_EXTRA", "")

    versions = [{"engine": e, "name": NAMES.get(e, e), "version": env.get(VER_KEYS.get(e, "")), "backend": be,
                 "weights": WEIGHTS} for e, be in configs]

    single = []
    for e, be in configs:
        row = {"engine": e, "backend": be, "tg128": None, "pp512": None}
        for r in read_jsonl(os.path.join(run, f"bench-{e}-{be}.jsonl")):
            if r.get("n_prompt") == 0:
                row["tg128"] = round(r["avg_ts"], 2)
            elif r.get("n_gen") == 0:
                row[f"pp{r['n_prompt']}"] = round(r["avg_ts"], 2)
        if row["tg128"] is not None or row["pp512"] is not None:
            single.append(row)

    conc, depth, seen, short = [], [], set(), []
    spec = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: [0, 0])
    heads = defaultdict(dict)  # (engine, backend, mode, profile) -> prompt -> first 60 characters of rep 0
    for r in read_jsonl(os.path.join(run, "results.jsonl")):
        lab = split_label(r.get("label", ""))
        if not lab:
            continue
        kind, e, m, prof, be = lab
        base = {"engine": e, "backend": be, "mode": m, "profile": prof}
        if r.get("mode") == "conc" and kind in ("conc", "concspec", "depth") and r.get("short_outputs"):
            short.append(short_row(r))
        if r.get("mode") == "conc" and kind in ("conc", "concspec"):
            conc.append({**base, "draft": DRAFT.get(m, ""), "prompt": round(r["prompt_tok"] / r["n_req"]) if r.get("n_req") else None,
                         "c": r["conc"], "gen": r["agg_gen_tps"], "req": r["per_req_tps_mean"], "ttft": r["ttft_mean_s"], "p90": r["ttft_p90_s"]})
            seen.add((kind, e, m, be))
        elif r.get("mode") == "conc" and kind == "depth":
            d = round(r["prompt_tok"] / r["n_req"]) if r.get("n_req") else None
            depth.append({**base, "depth": d, "pp": round(d / r["ttft_mean_s"], 1) if d and r.get("ttft_mean_s") else None,
                          "tg": r["per_req_tps_mean"], "ttft": r["ttft_mean_s"]})
            seen.add((kind, e, m, be))
        elif r.get("mode") == "spec" and kind == "spec":
            if r.get("dec_tps"):
                spec[(e, be, m, prof)][r["prompt"]].append(r["dec_tps"])
            if r.get("draft_n"):
                acc[(e, be, m, prof)][0] += r["draft_acc"]
                acc[(e, be, m, prof)][1] += r["draft_n"]
            if r.get("rep") == 0:
                heads[(e, be, m, prof)].setdefault(r["prompt"], r.get("text_head"))
            seen.add((kind, e, m, be))

    # Greedy speculation must reproduce the engine's own AR output, and every engine reads the same weights, so in AR
    # mode it should agree with llama.cpp Vulkan. `differs` lists prompts whose first 60 characters do not match.
    spec_rows = []
    for (e, be, m, prof), per in spec.items():
        vals = [v for p in per.values() for v in p]
        cmp = heads.get(("up", "vulkan", "ar", prof), {}) if m == "ar" else heads.get((e, be, "ar", prof), {})
        diff = [p for p, h in heads[(e, be, m, prof)].items() if p in cmp and cmp[p] != h]
        a2 = acc[(e, be, m, prof)]
        row = {"engine": e, "backend": be, "mode": m, "profile": prof, "draft": DRAFT.get(m, ""),
               "mean": round(statistics.mean(vals), 2), "runs": len(vals),
               "acc": round(100 * a2[0] / a2[1], 1) if a2[1] else None,
               "flags": extra.get((e, m), ""), "differs": ", ".join(diff)}
        row.update({p: round(statistics.mean(per[p]), 1) if per.get(p) else None for p in PROMPTS})
        spec_rows.append(row)

    expected = []
    for e, be in configs:
        for m in MODES:
            if (e, m) in UNSUPPORTED:
                continue
            expected.append(("spec", e, m, be))
            expected.append(("conc" if m == "ar" else "concspec", e, m, be))
        expected.append(("depth", e, "ar", be))
    status = a.status
    missing = [f"{k}:{e}/{m}/{be}" for k, e, m, be in expected if (k, e, m, be) not in seen]
    errors = read_jsonl(os.path.join(run, "errors.jsonl"))
    failures = [x for x in errors if not x.get("skipped") and not x.get("note")]
    if status == "ok" and (missing or any(x.get("step") != "warm-up" for x in failures)):
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
    notes = [f"{x['label']}: {x.get('skipped') or x.get('note')}" for x in errors if x.get("skipped") or x.get("note")]
    notes += [f"{e} {m}: {x}" for (e, m), x in extra.items() if x]
    notes += [f"{k[len('SPEC_EXTRA_'):].lower()}: {v or 'no extra flags'}" for k, v in env.items()
              if k.startswith("SPEC_EXTRA_") and k.count("_") == 4]  # per-kind overrides

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
        "versions": versions,
        "updates": [line.rstrip("\n") for line in open(upd)] if os.path.exists(upd) else [],
        "single": single,
        "conc": sorted(conc, key=lambda r: (r["profile"], r["engine"], r["backend"], r["mode"], r["prompt"] or 0, r["c"])),
        "spec": sorted(spec_rows, key=lambda r: (r["profile"], r["engine"], r["backend"], r["mode"])),
        "depth": sorted(depth, key=lambda r: (r["profile"], r["engine"], r["backend"], r["mode"], r["depth"] or 0)),
        "short_outputs": sum(x["short"] for x in short),
        "short_by_label": short,
        "errors": failures,
        "notes": notes,
        "jobs": jobs,
    }
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("date", "status", "missing")}))  # noqa: NP100


if __name__ == "__main__":
    main()
