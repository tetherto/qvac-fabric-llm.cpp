#!/usr/bin/env python3
"""Concurrency bench for one OpenAI-compatible server: N identical-shape requests sent at once.

Every request carries a unique first line, so no request can hit a prefix cache (server-side or radix) left by another
request or an earlier run; a run fails if the server reports a cache hit. Prompt shape: the 10k prompt file of pd_bench.py,
fixed output length with ignore_eos, temperature 0, streaming with usage.

Per run (one concurrency level, one repetition) the client records every stream chunk's arrival time and reports:
  prefill_agg_tok_s  = sum(prompt_tokens) / t_all_first, t_all_first = when the last request received its first token
  decode_agg_tok_s   = output tokens streamed after t_all_first / (t_end - t_all_first)
  output_tok_s       = sum(output tokens) / t_end;  total_tok_s = (prompt + output tokens) / t_end
  streams_at_all_first = requests still streaming at t_all_first (fewer than N means some finished during others' prefill)
plus per-request TTFT and per-stream decode rates, and the llama.cpp server timings when present (diagnostic only).
Chunk count is not a token count: tokens after t_all_first are chunks after it scaled by completion_tokens / n_chunks.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid


def post(url, payload, timeout=3600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def load_prompt(path):
    with open(path) as f:
        return json.loads(f.readline())["prompt"]


def read_cached(usage, timings):
    if usage:
        details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            return details["cached_tokens"]
        if "cached_tokens" in usage:
            return usage["cached_tokens"]
    if timings and "cache_n" in timings:
        return timings["cache_n"]
    return None


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except OSError:
        return None


def stream_one(base, model, prompt, max_tokens, t0, out, idx):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    rec = {"idx": idx, "t_send": time.perf_counter() - t0, "chunks": [], "error": None}
    try:
        resp = post(base + "/v1/completions", payload)
        prompt_tokens = completion_tokens = cached = timings = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            tm = chunk.get("timings")
            if tm:
                timings = tm
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = usage.get("completion_tokens") or completion_tokens
            hit = read_cached(usage, tm)
            if hit is not None:
                cached = hit
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                rec["chunks"].append(time.perf_counter() - t0)
        rec["t_end"] = time.perf_counter() - t0
        rec["prompt_tokens"] = prompt_tokens
        rec["n_out"] = completion_tokens
        rec["cached_tokens"] = cached
        rec["server_timings"] = timings
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        rec["t_end"] = time.perf_counter() - t0
    out[idx] = rec


def run_level(base, model, prompt, n, max_tokens, rep):
    tag = uuid.uuid4().hex
    prompts = [f"run {tag} rep {rep} req {i}\n" + prompt for i in range(n)]
    out = [None] * n
    load0 = loadavg()
    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=stream_one, args=(base, model, prompts[i], max_tokens, t0, out, i)) for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    load1 = loadavg()

    errs = [r["error"] for r in out if r["error"]]
    if errs:
        raise RuntimeError(f"n={n} rep={rep}: {len(errs)} request(s) failed: {errs[0]}")
    for r in out:
        if not r["n_out"] or not r["prompt_tokens"]:
            raise RuntimeError(f"n={n} rep={rep}: request {r['idx']} reported no usage token counts")
        if r["cached_tokens"]:
            raise RuntimeError(f"n={n} rep={rep}: request {r['idx']} hit {r['cached_tokens']} cached tokens")
        if not r["chunks"]:
            raise RuntimeError(f"n={n} rep={rep}: request {r['idx']} streamed no text")
        r["t_first"] = r["chunks"][0]
        r["ttft_s"] = r["t_first"] - r["t_send"]
        r["n_chunks"] = len(r["chunks"])
        r["stream_decode_tok_s"] = (r["n_out"] - 1) / (r["t_end"] - r["t_first"])

    n_prompt = sum(r["prompt_tokens"] for r in out)
    n_output = sum(r["n_out"] for r in out)
    t_all_first = max(r["t_first"] for r in out)
    t_end = max(r["t_end"] for r in out)
    tokens_after = 0.0
    for r in out:
        after = sum(1 for c in r["chunks"] if c > t_all_first)
        tokens_after += after * (r["n_out"] / r["n_chunks"])
    decode_window = t_end - t_all_first
    st = [r["server_timings"] for r in out if r.get("server_timings")]
    summary = {
        "n": n,
        "rep": rep,
        "tag": tag,
        "load_before": load0,
        "load_after": load1,
        "prompt_tokens": n_prompt,
        "output_tokens": n_output,
        "t_all_first_s": t_all_first,
        "t_end_s": t_end,
        "prefill_agg_tok_s": n_prompt / t_all_first,
        "tokens_after_all_first": tokens_after,
        "decode_window_s": decode_window,
        "decode_agg_tok_s": tokens_after / decode_window if decode_window > 0 else None,
        "streams_at_all_first": sum(1 for r in out if r["t_end"] > t_all_first),
        "output_tok_s": n_output / t_end,
        "total_tok_s": (n_prompt + n_output) / t_end,
        "ttft_mean_s": statistics.mean(r["ttft_s"] for r in out),
        "ttft_min_s": min(r["ttft_s"] for r in out),
        "ttft_max_s": max(r["ttft_s"] for r in out),
        "stream_decode_tok_s_mean": statistics.mean(r["stream_decode_tok_s"] for r in out),
        "stream_decode_tok_s_sum": sum(r["stream_decode_tok_s"] for r in out),
    }
    if st and all("prompt_ms" in s and "predicted_ms" in s for s in st):
        summary["server_prompt_ms_mean"] = statistics.mean(s["prompt_ms"] for s in st)
        summary["server_prompt_tok_s_mean"] = statistics.mean(s["prompt_n"] / s["prompt_ms"] * 1000.0 for s in st)
        summary["server_predicted_tok_s_sum"] = sum(s["predicted_n"] / s["predicted_ms"] * 1000.0 for s in st)
    requests = []
    for r in out:
        requests.append({k: v for k, v in r.items() if k != "chunks"})
    return summary, requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-10k", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    prompt = load_prompt(args.prompt_10k)
    levels = [int(x) for x in args.levels.split(",") if x]
    results = {"label": args.label, "model": args.model, "max_tokens": args.max_tokens, "runs": []}

    print(f"[{args.label}] warmup", flush=True)
    run_level(base, args.model, prompt, 1, 32, 0)
    for rep in range(1, args.reps + 1):
        for n in levels:
            time.sleep(2.0)
            summary, requests = run_level(base, args.model, prompt, n, args.max_tokens, rep)
            results["runs"].append({"summary": summary, "requests": requests})
            print(json.dumps(summary), flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    print(f"[{args.label}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
