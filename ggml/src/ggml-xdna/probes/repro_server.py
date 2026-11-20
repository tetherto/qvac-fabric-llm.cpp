#!/usr/bin/env python3
"""Reproducibility check of the decode path, one fresh server per run.

Every run starts its own llama-server with -np 1 -b 4096 -ub 4096 --flash-attn
off and sends identical greedy requests (n_probs 5), then prints a hash of the
generated text and of every top-5 record. The default path is reproducible when
all requests of a run, and all runs, give one hash per prompt.

  python3 probes/repro_server.py build/bin/llama-server model.gguf

Older XRT is not supported: 2.21.75 has been seen to fail a ~1.2 GB host-memory
BO allocation at startup and its results were not reproducible, so the server
log is scanned for that and for decode errors, and a run that shows either is
reported as failed rather than as a hash.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request


def post(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def request(base, prompt, n_predict):
    d = post(base, "/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "top_k": 1, "seed": 1234, "cache_prompt": False, "n_probs": 5,
    })
    cp = d.get("completion_probabilities") or []
    rec = [(e["id"], round(e["logprob"], 6),
            [(t["id"], round(t["logprob"], 6)) for t in e["top_logprobs"]]) for e in cp]
    h = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    return h, d.get("content", "")


def wait_healthy(base, proc, seconds=180):
    for _ in range(seconds):
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_once(args, run_index):
    base = f"http://127.0.0.1:{args.port}"
    log = open(f"/tmp/xdna-repro-{run_index}.log", "w")
    proc = subprocess.Popen([
        args.server, "-m", args.model, "-ngl", "99", "-c", "4096", "-b", "4096",
        "-ub", "4096", "--flash-attn", "off", "-np", "1", "--seed", "1234",
        "--host", "127.0.0.1", "--port", str(args.port), "--no-webui",
    ], stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_healthy(base, proc):
            print(f"run {run_index}: server did not come up (see /tmp/xdna-repro-{run_index}.log)")
            return None
        a_id = post(base, "/tokenize", {"content": "A"})["tokens"][-1]
        one = [request(base, [a_id], args.n_predict)[0] for _ in range(args.requests)]
        long_ids = post(base, "/tokenize", {"content": args.long_text * args.long_repeat})["tokens"]
        long = [request(base, long_ids, args.n_predict)[0] for _ in range(2)]
    finally:
        proc.terminate()
        log.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = open(f"/tmp/xdna-repro-{run_index}.log", errors="replace").read()
    for bad in ("failed to allocate", "Compute error", "failed to decode", "not built for design tag"):
        if bad in text:
            print(f"run {run_index}: the server log has {bad!r}; see /tmp/xdna-repro-{run_index}.log")
            return None
    print(f"run {run_index}: one-token {sorted(set(one))}  long {sorted(set(long))}")
    return one, long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server", help="path to llama-server")
    ap.add_argument("model", help="path to a model the fused layer covers")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--runs", type=int, default=3, help="fresh server processes")
    ap.add_argument("--requests", type=int, default=6, help="identical requests per prompt")
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--long-repeat", type=int, default=200,
                    help="repetitions of the long prompt line (~2000 tokens)")
    args = ap.parse_args()
    args.long_text = "The quick brown fox jumps over the lazy dog. "

    results = [run_once(args, i + 1) for i in range(args.runs)]
    if any(r is None for r in results):
        print("repro_server: FAIL (a run did not complete)")
        return 1
    one = {h for r in results for h in r[0]}
    long = {h for r in results for h in r[1]}
    if len(one) != 1 or len(long) != 1:
        print(f"repro_server: FAIL (one-token hashes {sorted(one)}, long hashes {sorted(long)})")
        return 1
    print("repro_server: PASS (one hash per prompt across every run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
