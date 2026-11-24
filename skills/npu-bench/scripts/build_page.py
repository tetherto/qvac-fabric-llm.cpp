#!/usr/bin/env python3
"""summary.json -> the xdna vs FastFlowLM page: Summary and Per model.

usage: build_page.py <summary.json> -o <page.html>
"""
from __future__ import annotations

import argparse
import json
import os

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")
PLACEHOLDER = "/*PAGE_JSON*/null"


def model_label(model):
    return model[:-1] + "B" if model.endswith("b") else model


def short_sha(sha):
    return sha[:8] if sha else "unknown"


def main_runtimes(s):
    v = s.get("versions", {})
    return [
        {"id": "xdna", "name": "xdna", "src": "qvac-fabric",
         "desc": f"NPU backend, {v.get('xdna', 'unknown build')}, -ngl 99 --device XDNA0, flash attention on."},
        {"id": "flm", "name": "flm", "src": "fastflowlm",
         "desc": f"FastFlowLM {v.get('flm', 'unknown version')} (flm serve). Own kernels and model store."},
    ]


def pct(x):
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def drift_text(s):
    rows = [f"{model_label(m)} prefill {pct(d['prefill'])}, decode {pct(d['decode'])}, decode energy {pct(d['dj'])}"
            for m, d in s.get("drift", {}).items()]
    if not rows:
        return "No drift session ran."
    return "xdna at 1k is repeated after each model's flm session: " + "; ".join(rows) + "."


def foreign_text(s):
    jobs = [line for line in s.get("foreign", []) if not line.endswith("---")]
    if not jobs:
        return "No other GPU, NPU or build job was logged during the run."
    return f"{len(jobs)} lines of other jobs were logged during the run (<code>foreign.log</code>); check them before trusting the affected cells."


def models_text(s):
    files = [f"<code>{m['file']}</code> ({short_sha(m.get('sha256'))})" for m in s.get("models", {}).values()]
    return ", ".join(files) if files else "the GGUF files listed in <code>run.json</code>"


def status_text(s):
    if s.get("status") == "ok":
        return ""
    missing = ", ".join(s.get("missing", [])) or "none"
    return f" Run status: <b>{s.get('status')}</b> {s.get('reason', '')}; cells without a result: {missing}."


def default_text(s):
    date, max_tokens = s.get("date", "unknown date"), s.get("max_tokens", 512)
    v = s.get("versions", {})
    return {
        "intro": [
            "<b>xdna</b> is the qvac-fabric (llama.cpp-based) backend that runs Qwen3.5 on the <b>AMD XDNA NPU</b> inside Strix Halo. It loads a Q4_K_M GGUF, shows up as device <code>XDNA0</code> and is selected with <code>-ngl 99 --device XDNA0</code> (flash attention on). In a build that also has the Vulkan backend, llama.cpp picks the iGPU by default, so <code>--device XDNA0</code> is needed to use the NPU.",
            "<b>FastFlowLM (flm)</b> is the existing NPU runtime, with its own kernels and model store. Both run on the same machine, one after the other, on the same prompts. The boxes below put xdna against flm directly.",
        ],
        "summaryNote": f"Each cell divides one flm value by the matching xdna value (or the other way round for rates), so above 1.00&times; xdna is ahead. Mint is a win, amber a loss, and the colour deepens with the size of the gap. Hover a cell for the raw numbers. Both runtimes were measured in one run on {date}.{status_text(s)}",
        "speedNote": "Bars are the mean over iterations and whiskers one standard deviation. Hover a context group for each runtime's value, spread and multiple of flm on the same context (rates above 1.00&times; are faster, times below 1.00&times; are shorter).",
        "energyNote": "System power is the package power PPT (amdgpu hwmon, which includes the NPU) sampled every 0.1 s, minus the idle baseline (the mean of the 1 s before a window and the last 1 s of its tail). Every window runs until 10 s after its last request, because the NPU stays powered about 5 s after a job and PPT lags about 1 s. J per generated token uses only the decode part of a request, J per 1k prefill tokens only one-token prefill bursts. Lower is cheaper, except tokens per joule.",
        "method": [
            f"<b>Comparison caveat.</b> flm is served by a runtime with its own model store, so it loads its own quantised build, not a GGUF. Charts that include flm compare runtimes and models at once. The xdna runs load {models_text(s)}.",
            f"<b>Runs.</b> <code>bench_npu.py run {s.get('bench_args', '')}</code>: streaming chat requests with <code>ignore_eos</code>, no prompt cache and a unique first line, the same prompts for both runtimes, and a wait (up to 10 min) for Tctl to fall to 47 &deg;C before each context. Prefill rate is prompt tokens / TTFT and TTFT is measured at the client, over the timed requests of each context; decode rate is completion tokens / (total time - TTFT) over the same requests. llama-server: <code>{s.get('server_flags', '')}</code>, with {s.get('omp_policy', 'the OpenMP policy in run.json')}. flm: <code>flm serve</code> with its defaults.",
            "<b>Energy.</b> Prefill J per 1k tokens = (burst energy - one tiny request) / prompt tokens. Decode J per token = (full request - tiny request - its prompt tokens at the prefill rate of energy) / (completion tokens - 1). Whole request = one full request including its 10 s tail. Tokens per joule = 1 / decode J per token; its spread is derived from the decode energy spread.",
            f"<b>flm stops early.</b> When flm ends a request before {max_tokens} tokens, its decode rate is unaffected, but energy cells whose requests averaged fewer than {max_tokens // 2} tokens carry an asterisk with the token count, and a decode energy that computes to zero or below is shown as n/a.",
            f"<b>Checks.</b> {drift_text(s)} {foreign_text(s)}",
            f"<b>Builds.</b> xdna: {v.get('xdna', 'unknown')}. flm: {v.get('flm', 'unknown')}.",
            f"<b>Hardware.</b> {s.get('hardware', 'unknown')}, on {s.get('machine', 'unknown host')}.",
        ],
    }


def page_data(s):
    models = s["models_order"]
    return {
        "models": models, "labels": {m: model_label(m) for m in models}, "contexts": s["contexts"],
        "maxTokens": s.get("max_tokens", 512), "runtimes": main_runtimes(s), "cells": s["cells"],
        "tokens": s["tokens"], "text": default_text(s),
    }


def render(data):
    with open(TEMPLATE) as f:
        page = f.read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return page.replace(PLACEHOLDER, blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    with open(a.summary) as f:
        data = page_data(json.load(f))
    with open(a.out, "w") as f:
        f.write(render(data))


if __name__ == "__main__":
    main()
