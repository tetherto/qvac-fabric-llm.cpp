#!/usr/bin/env python3
"""Prefill/decode split bench for one OpenAI-compatible server, concurrency 1.

Splits single-request wall time into:
  prefill = client TTFT (first streamed token)
  decode  = remaining stream time / (n_out - 1)

Cache control does not rely on a flush endpoint. Every measured request gets a unique tag prepended, so a cold shape cannot hit any cache left by an earlier run (llama-server keeps a host-side prompt cache that no endpoint clears).
The warm shape prepends the same tag to both the prefix-fill request and the measured request, so only that rep's own prefix can be reused.

Prefill throughput divides by the tokens the server says it computed: prompt_tokens minus server-reported cache hits (usage.prompt_tokens_details.cached_tokens, or llama.cpp timings.cache_n).
A cold shape fails if the server reports a positive hit count. A server that reports no hit field at all (sglang without --enable-cache-report) is treated as zero on cold shapes and rejected on the warm shape.

Shapes: cold 10k, cold 110k, warm 100k prefix + 10k suffix.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid


def post(url, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def load_prompt(path):
    with open(path) as f:
        return json.loads(f.readline())["prompt"]


def flush(base):
    for ep in ("/flush_cache", "/slots/0?action=erase"):
        try:
            r = urllib.request.Request(
                base + ep, data=b"{}", headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(r, timeout=120).read()
            return ep
        except urllib.error.HTTPError as e:
            if e.code < 500:
                continue
        except Exception:
            continue
    return None


def read_cached(usage, timings):
    """Server-reported prompt tokens served from cache, or None if unreported."""
    if usage:
        details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            return details["cached_tokens"]
        if "cached_tokens" in usage:
            return usage["cached_tokens"]
    if timings and "cache_n" in timings:
        return timings["cache_n"]
    return None


def run_stream(base, model, prompt, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = None
    server_timings = None
    resp = post(base + "/v1/completions", payload)
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
        timings = chunk.get("timings")
        if timings:
            server_timings = timings
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
            completion_tokens = usage.get("completion_tokens") or completion_tokens
        hit = read_cached(usage, timings)
        if hit is not None:
            cached_tokens = hit
        choices = chunk.get("choices") or []
        if choices and choices[0].get("text"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_chunks += 1
    total = time.perf_counter() - t0
    if not completion_tokens:
        raise RuntimeError(
            "server reported no usage.completion_tokens in the stream; SSE chunk "
            "count is not a token count, refusing to guess decode rate"
        )
    if not prompt_tokens:
        raise RuntimeError("server reported no usage.prompt_tokens in the stream")
    if ttft is None:
        raise RuntimeError("stream produced no text chunk, cannot measure TTFT")
    decode_s = total - ttft
    return {
        "prompt_tokens": prompt_tokens,
        "reported_cached_tokens": cached_tokens,
        "n_out": completion_tokens,
        "n_chunks": n_chunks,
        "ttft_s": ttft,
        "total_s": total,
        "decode_s": decode_s,
        "decode_tok_s": (completion_tokens - 1) / decode_s,
        "itl_ms": decode_s * 1000.0 / (completion_tokens - 1),
        "server_timings": server_timings,
    }


def probe_prompt_tokens(base, model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    r = json.loads(post(base + "/v1/completions", payload).read())
    n = (r.get("usage") or {}).get("prompt_tokens")
    if not n:
        raise RuntimeError("prefix probe returned no usage.prompt_tokens")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-10k", required=True)
    ap.add_argument("--prompt-110k", required=True)
    ap.add_argument("--prefix-100k", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shapes", default="cold-10k,cold-110k,warm100k-plus-10k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    p10 = load_prompt(args.prompt_10k)
    p110 = load_prompt(args.prompt_110k)
    p100 = load_prompt(args.prefix_100k)
    shapes = [s for s in args.shapes.split(",") if s]

    results = {"label": args.label, "model": args.model, "shapes": []}

    print(f"[{args.label}] warmup", flush=True)
    run_stream(base, args.model, f"warmup {uuid.uuid4().hex}\n" + p10[:2000], 32)

    for rep in range(1, args.reps + 1):
        for shape in shapes:
            ep = flush(base)
            time.sleep(2.0)
            tag = f"run {uuid.uuid4().hex} rep {rep} {shape}\n"
            if shape == "warm100k-plus-10k":
                t_fill = time.perf_counter()
                n_prefix = probe_prompt_tokens(base, args.model, tag + p100)
                fill_s = time.perf_counter() - t_fill
                r = run_stream(base, args.model, tag + p110, args.max_tokens)
                r["prefix_fill_s"] = fill_s
                r["prefix_prompt_tokens"] = n_prefix
                if r["reported_cached_tokens"] is None:
                    raise RuntimeError(
                        "server did not report cached prompt tokens; cannot "
                        "attribute the cached-shape prefill rate. For sglang, "
                        "launch it with --enable-cache-report"
                    )
            else:
                prompt = p10 if shape == "cold-10k" else p110
                r = run_stream(base, args.model, tag + prompt, args.max_tokens)
                if r["reported_cached_tokens"]:
                    raise RuntimeError(
                        f"{shape} hit {r['reported_cached_tokens']} cached tokens, "
                        "not a cold measurement"
                    )
            r["cached_tokens"] = r["reported_cached_tokens"] or 0
            r["fresh_tokens"] = r["prompt_tokens"] - r["cached_tokens"]
            r["prefill_tok_s"] = r["fresh_tokens"] / r["ttft_s"]
            r["shape"] = shape
            r["rep"] = rep
            r["flush_endpoint"] = ep
            r["tag"] = tag.strip()
            results["shapes"].append(r)
            print(json.dumps(r), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.label}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
