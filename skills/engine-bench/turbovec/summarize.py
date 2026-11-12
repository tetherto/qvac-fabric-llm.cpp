"""Writes <run>/summary.json for one TurboVec run from the drivers' JSON files, engines.env, updates.txt, the load
log, errors.jsonl and jobs.jsonl. Runs on the system python3 too (a failed run may have no venv), so it keeps to Python 3.9.

Every result row gets `impl` (qvac-fabric or turbovec) and `search`: "staged" (upstream's default, and fabric's once it has
the 1.1.2 stack) or "full scan" (every vector scanned: fabric with staged search off, or the 0.9 port, which has only that).

usage: summarize.py <run dir> --status ok|failed [--reason TEXT]   (machine.conf from $TV_DIR, default this script's folder)
"""
import argparse
import json
import os
import platform
import time

T = os.environ.get("TV_DIR") or os.path.dirname(os.path.abspath(__file__))
SCHEMA = 1


def conf(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def split_name(name):
    if name.startswith("fabric"):
        full = "base" in name or "staged-off" in name or "staged off" in name
        return "qvac-fabric", "full scan" if full else "staged"
    return "turbovec", "staged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--status", required=True)
    ap.add_argument("--reason")
    args = ap.parse_args()
    run = args.run.rstrip("/")
    mc = conf(os.path.join(T, "machine.conf"))
    env = conf(os.path.join(run, "engines.env"))
    date = os.path.basename(run)

    batch = []
    for r in load_json(os.path.join(run, "batch.json"), []):
        impl, search = split_name(r["name"])
        batch.append({"bits": r["bits"], "threads": r["threads"], "impl": impl, "search": search,
                      "add_ms": round(r["add_ms"], 1), "search_ms": round(r["search_ms"], 2), "qps": r["qps"],
                      "top1": r["top1"], "samples": r["samples"]})
    single = []
    for r in load_json(os.path.join(run, "single.json"), []):
        impl, search = split_name(r["name"])
        single.append({"bits": r["bits"], "k": r["k"], "threads": r["threads"], "impl": impl, "search": search,
                       "p50_us": r["p50_us"], "p95_us": r["p95_us"], "top1": r["top1"], "samples": r["samples"]})
    rec = load_json(os.path.join(run, "recall.json"), {})
    recall, same = [], []
    for r in rec.get("recall", []):
        impl, search = split_name(r["name"])
        recall.append({"bits": r["bits"], "k": r["k"], "impl": impl, "search": search,
                       "r1": r["r1"], "r10": r["r10"], "r64": r["r64"]})
    for s in rec.get("same", []):
        ai, asr = split_name(s["a"])
        bi, bsr = split_name(s["b"])
        same.append({"bits": s["bits"], "k": s["k"], "a": ai, "a_search": asr, "b": bi, "b_search": bsr,
                     "top10": s["top10"], "topk": s["topk"]})

    loads = []
    if os.path.exists(os.path.join(run, "load.log")):
        for line in open(os.path.join(run, "load.log")):
            p = line.split()
            if len(p) == 2:
                try:
                    loads.append(float(p[1]))
                except ValueError:
                    pass
    errors = jsonl(os.path.join(run, "errors.jsonl"))
    missing = [n for n, rows in (("batch", batch), ("single", single), ("recall", recall)) if not rows]
    status = args.status
    if status == "ok" and (errors or missing):
        status = "partial"
    started = open(os.path.join(run, "started")).read().strip() if os.path.exists(os.path.join(run, "started")) else None
    updates = open(os.path.join(run, "updates.txt")).read().strip() if os.path.exists(os.path.join(run, "updates.txt")) else None
    fab_api = env.get("FAB_API") == "1"

    s = {
        "schema": SCHEMA, "date": date, "machine": mc.get("MACHINE"), "kind": "test" if date.endswith("-quick") else "weekly",
        "status": status, "reason": args.reason, "missing": missing, "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {"cpu": mc.get("HW"), "os": f"{platform.system()} {platform.machine()}"},
        "versions": [
            {"impl": "qvac-fabric", "version": env.get("FAB_VER"), "api": fab_api,
             "detail": "1.1.2 stack: threads, staged search (default) and full scan" if fab_api
             else "0.9 port: one thread, full scan only"},
            {"impl": "turbovec", "version": env.get("TV_VER"), "api": True, "detail": "PyPI wheel, staged search (default)"},
        ],
        "updates": updates, "batch": batch, "single": single, "recall": recall, "same": same,
        "load": {"max": max(loads) if loads else None, "mean": round(sum(loads) / len(loads), 2) if loads else None,
                 "samples": len(loads), "threshold": max(2.0, (os.cpu_count() or 16) / 8)},
        "errors": errors, "jobs": jsonl(os.path.join(run, "jobs.jsonl")),
    }
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps({"date": date, "status": status, "missing": missing, "errors": len(errors)}))


if __name__ == "__main__":
    main()
