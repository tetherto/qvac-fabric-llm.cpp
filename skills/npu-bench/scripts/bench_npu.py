#!/usr/bin/env python3
"""One streaming OpenAI-style client for llama-server and flm serve, with package-power (PPT) energy per window.

sensors - print the PPT and Tctl sensor paths and current readings (pre-flight); fails when one is missing
prep    - build the prompt set (token-exact corpus slices) with llama-server /tokenize and /detokenize
run     - per context: warm-up, timed full requests, one prefill burst, one tiny request; one JSON line per window
"""
from __future__ import annotations

import argparse
import glob
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse

HWMON = "/sys/class/hwmon/hwmon*"
SAMPLE_S = 0.1
# The NPU stays powered ~5 s after its last job (amdxdna autosuspend) and PPT lags ~1 s: every window is
# integrated until 10 s after its last request, and a tiny request measures that fixed per-window cost.
TAIL_S = 10.0
IDLE_S = 1.0
MIN_IDLE_W = 1.0
TEMP_MAX_C = 47.0
TEMP_WAIT_S = 600
TEMPLATE_RESERVE = 32
TINY_PROMPT = "Reply with one word."


def read_text(path):
    with open(path) as f:
        return f.read().strip()


def sensor_path(name, file, label=None):
    """Find a hwmon file by driver name (and label); hwmon numbers change across reboots."""
    for d in sorted(glob.glob(HWMON)):
        if read_text(os.path.join(d, "name")) != name or not os.path.exists(os.path.join(d, file)):
            continue
        label_file = os.path.join(d, file.replace("_input", "_label"))
        if label is None or (os.path.exists(label_file) and read_text(label_file) == label):
            return os.path.join(d, file)
    sys.exit(f"no hwmon sensor {name}/{file} {label or ''}")


def ppt_path():
    return sensor_path("amdgpu", "power1_input", "PPT")


def tctl_path():
    return sensor_path("k10temp", "temp1_input")


def log(msg):
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


def post(url, path, body, timeout=1800):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"{path} HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


class Power:
    def __init__(self):
        self.path = ppt_path()
        self.samples = []
        self.stop = False
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        while not self.stop:
            self.samples.append((time.monotonic(), int(read_text(self.path)) / 1e6))
            time.sleep(SAMPLE_S)

    def window(self, a, b):
        return [(t, w) for t, w in self.samples if a <= t <= b]

    def mean(self, a, b):
        w = [w for _, w in self.window(a, b)]
        return statistics.mean(w) if w else float("nan")

    def energy_above(self, a, b, idle):
        s = self.window(a, b)
        return sum((t1 - t0) * ((w0 + w1) / 2 - idle) for (t0, w0), (t1, w1) in zip(s, s[1:]))


def tctl(path):
    return int(read_text(path)) / 1000


def wait_cool(path, label):
    t0 = time.monotonic()
    while tctl(path) > TEMP_MAX_C and time.monotonic() - t0 < TEMP_WAIT_S:
        time.sleep(2)
    log(f"{label} tctl {tctl(path):.1f} C after {time.monotonic() - t0:.0f} s")


def read_stream(r):
    """Returns (time of the first content chunk, last event with usage or timings, content chunk count)."""
    t_first, last, n_chunks = None, None, 0
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        if ev.get("usage") or ev.get("timings") or last is None:
            last = ev
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ch.get("text"):
                n_chunks += 1
                if t_first is None:
                    t_first = time.monotonic()
    return t_first, last, n_chunks


def stream(url, model, prompt, max_tokens, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    body.update(extra)
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    t_start = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    t_first, last, n_chunks = read_stream(r)
    t_end = time.monotonic()
    c.close()
    return t_start, t_first, t_end, n_chunks, last


def nonce_prompt(slices, ctx, tag):
    if ctx == 0:
        return f"Request {tag}. {TINY_PROMPT}"
    return f"Request {tag}. Continue the following text.\n\n" + slices[str(ctx)]


def cmd_sensors(a):
    p, t = ppt_path(), tctl_path()
    w = int(read_text(p)) / 1e6
    sys.stdout.write(json.dumps({"ppt": p, "ppt_w": w, "tctl": t, "tctl_c": tctl(t)}) + "\n")
    if w < MIN_IDLE_W:
        sys.exit(f"PPT reads {w:.2f} W")


def cmd_prep(a):
    with open(a.corpus) as f:
        text = f.read()
    toks = post(a.server, "/tokenize", {"content": text})["tokens"]
    need = max(a.contexts) - TEMPLATE_RESERVE
    if len(toks) < need:
        sys.exit(f"corpus has {len(toks)} tokens, need {need}")
    slices = {str(c): post(a.server, "/detokenize", {"tokens": toks[: c - TEMPLATE_RESERVE]})["content"] for c in a.contexts}
    with open(a.out, "w") as f:
        json.dump({"corpus_tokens": len(toks), "slices": slices}, f)
    log(f"corpus_tokens {len(toks)}; slices {list(slices)}")


class Runner:
    def __init__(self, a):
        with open(a.prompts) as f:
            self.slices = json.load(f)["slices"]
        self.a = a
        self.extra = json.loads(a.extra)
        self.tctl = tctl_path()
        self.pw = Power()
        self.out = open(a.out, "a")

    def one(self, ctx, max_tokens):
        tag = f"{self.a.label}-{ctx}-{random.getrandbits(48):012x}"
        s, f, e, n, last = stream(self.a.server, self.a.model, nonce_prompt(self.slices, ctx, tag), max_tokens, self.extra)
        usage = (last or {}).get("usage") or {}
        return {"start": s, "ttft": (f - s) if f else None, "total": e - s, "end": e, "chunks": n,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "timings": (last or {}).get("timings")}

    def measure(self, kind, ctx, n, max_tokens):
        """n back-to-back requests as one energy window: 1 s idle before, TAIL_S after the last."""
        t_idle = time.monotonic()
        time.sleep(IDLE_S)
        reqs = [self.one(ctx, max_tokens) for _ in range(n)]
        time.sleep(TAIL_S)
        end = reqs[-1]["end"] + TAIL_S
        idle_pre, idle_post = self.pw.mean(t_idle, reqs[0]["start"]), self.pw.mean(end - IDLE_S, end)
        idle = (idle_pre + idle_post) / 2
        if not idle >= MIN_IDLE_W:
            sys.exit(f"{self.a.label} idle power {idle:.2f} W: the PPT sensor is not read")
        rec = {"label": self.a.label, "ctx": ctx, "kind": kind, "reqs": reqs, "idle_w": idle, "idle_pre": idle_pre,
               "idle_post": idle_post, "tctl": tctl(self.tctl), "energy_j": self.pw.energy_above(reqs[0]["start"], end, idle),
               "mean_w": self.pw.mean(reqs[0]["start"], reqs[-1]["end"])}
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()
        log(f"{self.a.label} ctx {ctx} {kind} x{n} ttft {[round(r['ttft'] or -1, 3) for r in reqs]} "
            f"pt {reqs[0]['prompt_tokens']} ct {reqs[0]['completion_tokens']} E {rec['energy_j']:.1f}")

    def context(self, ctx):
        a = self.a
        wait_cool(self.tctl, a.label)
        for _ in range(a.warmup):
            self.one(ctx, a.max_tokens)
        time.sleep(TAIL_S)
        for _ in range(a.full):
            self.measure("full", ctx, 1, a.max_tokens)
        if a.burst:
            self.measure("burst", ctx, max(a.burst, -(-a.burst_tokens // ctx)), 1)
        for _ in range(a.tiny):
            self.measure("tiny", 0, 1, 1)


def run_contexts(runner, contexts):
    for ctx in contexts:
        runner.context(ctx)


def cmd_run(a):
    runner = Runner(a)
    run_contexts(runner, a.contexts)
    runner.pw.stop = True
    log(f"{a.label} RUN_DONE")


def parse_contexts(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("sensors")
    for name in ("prep", "run"):
        p = sp.add_parser(name)
        p.add_argument("--server", required=True)
        p.add_argument("--contexts", type=parse_contexts, default=[1024, 2048, 4096, 8192, 16384])
        if name == "prep":
            p.add_argument("--corpus", required=True)
            p.add_argument("--out", required=True)
            continue
        p.add_argument("--prompts", required=True)
        p.add_argument("--model", default="default")
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument("--extra", default="{}", help="JSON fields merged into each request")
        p.add_argument("--label", required=True, help="<model>|<runtime>|<session>")
        p.add_argument("--out", required=True)
        p.add_argument("--full", type=int, default=8, help="single timed requests generating --max-tokens")
        p.add_argument("--burst", type=int, default=4, help="back-to-back one-token requests, one energy window")
        p.add_argument("--burst-tokens", type=int, default=8192, help="minimum prompt tokens in a burst")
        p.add_argument("--warmup", type=int, default=2)
        p.add_argument("--tiny", type=int, default=1, help="timed tiny-prompt one-token requests (fixed cost)")
    a = ap.parse_args()
    {"sensors": cmd_sensors, "prep": cmd_prep, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
