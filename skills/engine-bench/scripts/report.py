#!/usr/bin/env python3
"""Write a Markdown report from one or more runs/<date>/summary.json files (schema 3, any machine).

usage: report.py <summary.json or run dir>... [--previous <summary.json or run dir>...] [--threshold 5] [-o report.md]

One section per run: hardware, versions, problems (status, missing cells, errors, short outputs, notes), llama-bench,
one request per speculative mode, concurrency with and without speculation, long-prompt concurrency, and prefill and
decode by depth, each per profile. With --previous, each run is compared with the previous run of the same machine:
every metric that moved by more than the threshold is listed with its job's start time, and whether the engine's
version changed. Drops that cluster in one time window, or hit engines whose version did not change, usually mean
another user loaded the shared machine: re-run those jobs before trusting them.
"""
import argparse
import json
import os
import sys

MODES = ["ar", "mtp", "dflash2"]
MODE_NAME = {"ar": "no speculation", "mtp": "MTP", "dflash2": "DFlash2"}
LONG_PROMPT = 2000  # mean prompt tokens per request from which a conc row counts as a long-prompt run


def load(path):
    p = os.path.join(path, "summary.json") if os.path.isdir(path) else path
    with open(p) as f:
        s = json.load(f)
    s["_path"] = p
    return s


def fmt(x, nd=1):
    if x is None or x == "":
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}" if abs(x) < 1000 else f"{x:,.0f}"
    return str(x)


def table(head, rows):
    if not rows:
        return []
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return out + [""]


def buckets(values, tol=0.1):
    """Group real prompt lengths that came from one requested length (they differ by a few tokens per engine)."""
    out = []
    for v in sorted(set(values)):
        if out and v <= out[-1][0] * (1 + tol):
            out[-1][1].append(v)
        else:
            out.append((v, [v]))
    return [(f"~{b[0] / 1000:.1f}k" if b[0] < 10000 else f"~{b[0] / 1000:.0f}k", set(b[1])) for b in out]


class Run:
    def __init__(self, s):
        self.s = s
        self.names = {v["engine"]: v.get("name", v["engine"]) for v in s.get("versions", [])}
        self.vers = {(v["engine"], v.get("backend")): v.get("version", "") for v in s.get("versions", [])}
        rows = s.get("conc", []) + s.get("spec", []) + s.get("depth", []) + s.get("single", [])
        per_engine = {}
        for r in rows:
            per_engine.setdefault(r["engine"], set()).add(r.get("backend"))
        self.multi = {e for e, b in per_engine.items() if len(b) > 1}

    def eng(self, r):
        n = self.names.get(r["engine"], r["engine"])
        return f"{n} ({r.get('backend')})" if r["engine"] in self.multi else n

    def order(self, rows):
        seen = {v["engine"]: i for i, v in enumerate(self.s.get("versions", []))}
        return sorted(rows, key=lambda r: (seen.get(r["engine"], 99), r["engine"], str(r.get("backend")),
                                           MODES.index(r["mode"]) if r.get("mode") in MODES else 9))


def profiles(rows):
    ps = sorted({r.get("profile", "standard") for r in rows}, key=lambda p: (p != "standard", p))
    return [(p, [r for r in rows if r.get("profile", "standard") == p]) for p in ps]


def section(run):
    s, out = run.s, []
    hw = s.get("hardware", {})
    out += [f"## {s.get('machine')} - {s.get('date')} ({s.get('status')})", ""]
    out += [f"- **Hardware:** {hw.get('chip', '?')}; {hw.get('gpu', '?')}; {hw.get('memory_gb', '?')} GB; {hw.get('os', '?')}"]
    out += [f"- **Run:** {s.get('started', '?')} to {s.get('finished', '?')}"]
    if s.get("reason"):
        out += [f"- **Reason:** {s['reason']}"]
    out += [""]
    out += ["### Versions", ""]
    out += table(["engine", "version", "backend", "weights"],
                 [[v.get("name", v["engine"]), v.get("version", ""), v.get("backend", ""), v.get("weights", "")]
                  for v in s.get("versions", [])])

    probs = []
    for m in s.get("missing") or []:
        probs.append(f"- missing: `{m}`")
    for e in s.get("errors") or []:
        where = e.get("step") or ""
        probs.append(f"- error: `{e.get('label')}` {where} {('at ' + e['at']) if e.get('at') else ''}: {e.get('error')}".rstrip())
    for x in s.get("short_by_label") or []:
        probs.append(f"- short outputs: `{x.get('label')}` C={x.get('c')}: {x.get('short')} of {x.get('of')} requests stopped early")
    if s.get("short_outputs") and not s.get("short_by_label"):
        probs.append(f"- short outputs: {s['short_outputs']} requests stopped before the requested length")
    for c in s.get("checks") or []:
        if c.get("same") is not None and c.get("of") and c["same"] < c["of"]:
            probs.append(f"- output check: {c['engine']} {c['mode']} ({c.get('profile')}) matched {c.get('reference')} on "
                         f"{c['same']} of {c['of']} texts")
    for n in s.get("notes") or []:
        probs.append(f"- note: {n}")
    out += ["### Problems", ""] + (probs or ["- none recorded"]) + [""]

    single = run.order(s.get("single", []))
    if single:
        out += ["### llama-bench, one sequence (t/s)", ""]
        out += table(["engine", "pp512", "tg128"], [[run.eng(r), fmt(r.get("pp512")), fmt(r.get("tg128"))] for r in single])

    for prof, rows in profiles(s.get("spec", [])):
        out += [f"### One request, decode t/s by mode ({prof})", ""]
        byeng = {}
        for r in run.order(rows):
            byeng.setdefault((r["engine"], r.get("backend")), {})[r["mode"]] = r
        body = []
        for (e, b), ms in byeng.items():
            cells = []
            for m in MODES:
                r = ms.get(m)
                cells.append("-" if not r else fmt(r.get("mean")) + (f" ({r['acc']:.0f}% acc)" if r.get("acc") else ""))
            body.append([run.eng({"engine": e, "backend": b})] + cells)
        out += table(["engine"] + [MODE_NAME[m] for m in MODES], body)

    for prof, rows in profiles(s.get("conc", [])):
        short = [r for r in rows if (r.get("prompt") or 0) < LONG_PROMPT]
        long_ = [r for r in rows if (r.get("prompt") or 0) >= LONG_PROMPT]
        for title, sub in (("short prompts", short), (f"long prompts (>= {LONG_PROMPT} tokens)", long_)):
            for spec in (False, True):
                part = [r for r in sub if (r["mode"] != "ar") == spec]
                if not part:
                    continue
                cs = sorted({r["c"] for r in part})
                kind = "with speculation" if spec else "without speculation"
                out += [f"### Concurrency {kind}, {title} ({prof}): total t/s (TTFT s)", ""]
                body, keyed = [], {}
                for r in run.order(part):
                    keyed.setdefault((r["engine"], r.get("backend"), r["mode"]), {})[r["c"]] = r
                for (e, b, m), byc in keyed.items():
                    name = run.eng({"engine": e, "backend": b}) + (f" {MODE_NAME[m]}" if spec or m != "ar" else "")
                    body.append([name] + [f"{fmt(byc[c]['gen'])} ({fmt(byc[c].get('ttft'), 2)})" if c in byc else "-"
                                          for c in cs])
                out += table(["engine"] + [f"C={c}" for c in cs], body)

    for prof, rows in profiles(s.get("depth", [])):
        bks = buckets([r["depth"] for r in rows])
        for m in MODES:
            part = [r for r in rows if r["mode"] == m]
            if not part:
                continue
            out += [f"### Prefill / decode t/s by prompt depth, {MODE_NAME[m]} ({prof})", ""]
            keyed = {}
            for r in run.order(part):
                keyed.setdefault((r["engine"], r.get("backend")), {})[r["depth"]] = r
            body = []
            for (e, b), byd in keyed.items():
                cells = []
                for _, members in bks:
                    r = next((byd[d] for d in byd if d in members), None)
                    cells.append(f"{fmt(r['pp'])} / {fmt(r['tg'])}" if r else "-")
                body.append([run.eng({"engine": e, "backend": b})] + cells)
            out += table(["engine"] + [b[0] for b in bks], body)
    return out


def metrics(run):
    """(key, label, value, job label) for every number worth comparing between runs."""
    s, out = run.s, []
    for r in s.get("single", []):
        for k in ("pp512", "tg128"):
            out.append((("single", r["engine"], r.get("backend"), k), f"{run.eng(r)} llama-bench {k}", r.get(k), None))
    for r in s.get("spec", []):
        p = r.get("profile", "standard")
        out.append((("spec", r["engine"], r.get("backend"), r["mode"], p), f"{run.eng(r)} one request {MODE_NAME[r['mode']]}",
                    r.get("mean"), f"spec-{r['engine']}-{r['mode']}"))
    for r in s.get("conc", []):
        p = r.get("profile", "standard")
        lng = (r.get("prompt") or 0) >= LONG_PROMPT
        kind = "concspec" if r["mode"] != "ar" else "conc"
        out.append((("conc", r["engine"], r.get("backend"), r["mode"], p, lng, r["c"]),
                    f"{run.eng(r)} {MODE_NAME[r['mode']]} C={r['c']}{' long' if lng else ''} total", r.get("gen"),
                    f"{kind}-{r['engine']}-{r['mode']}"))
    bks = buckets([r["depth"] for r in s.get("depth", [])])
    for r in s.get("depth", []):
        b = next(name for name, members in bks if r["depth"] in members)
        for k, what in (("pp", "prefill"), ("tg", "decode")):
            out.append((("depth", r["engine"], r.get("backend"), r["mode"], r.get("profile", "standard"), b, k),
                        f"{run.eng(r)} {MODE_NAME[r['mode']]} {what} at {b}", r.get(k), f"depth-{r['engine']}-{r['mode']}"))
    return out


def compare(run, prev, threshold):
    old = {k: v for k, _, v, _ in metrics(prev)}
    jobs = run.s.get("jobs") or {}
    rows = []
    for k, label, v, job in metrics(run):
        o = old.get(k)
        if not o or v is None:
            continue
        d = 100.0 * (v - o) / o
        if abs(d) <= threshold:
            continue
        ver = "changed" if run.vers.get((k[1], k[2])) != prev.vers.get((k[1], k[2])) else "same"
        t = next((jobs[j] for j in jobs if job and (j == job or j.startswith(job + "-"))), "")
        rows.append((abs(d), [label, fmt(o), fmt(v), f"{d:+.1f}%", ver, t]))
    rows.sort(key=lambda x: -x[0])
    out = [f"### Changes over {threshold:g}% since {prev.s.get('date')}", ""]
    if not rows:
        return out + ["- none", ""]
    out += table(["metric", prev.s.get("date"), run.s.get("date"), "change", "version", "job started"], [r for _, r in rows])
    return out + ["Drops that share one time window, or hit an engine whose version did not change, point to load from "
                  "other users rather than code: re-run those jobs on the idle machine before trusting them.", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--previous", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    runs = [Run(load(p)) for p in a.runs]
    prev = {}
    for p in a.previous:
        r = Run(load(p))
        if r.s.get("machine") not in prev or r.s.get("date", "") > prev[r.s["machine"]].s.get("date", ""):
            prev[r.s["machine"]] = r
    out = ["# Qwen3.8-27B engine benchmark report", "",
           "Totals and per-request rates are measured by bench_client.py through each engine's OpenAI-compatible server "
           "(greedy, prefix caching off). Prefill t/s by depth is prompt tokens / time to first token. Numbers from "
           "different machines use different weights formats (see Versions) and are not directly comparable.", ""]
    for run in runs:
        out += section(run)
        if run.s.get("machine") in prev:
            out += compare(run, prev[run.s["machine"]], a.threshold)
    text = "\n".join(out).rstrip() + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
