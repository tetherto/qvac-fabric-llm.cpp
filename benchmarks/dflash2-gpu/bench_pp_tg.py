"""Prefill and decode speed of one OpenAI-compatible server for a long prompt, measured from the stream.

Prefill tok/s = prompt tokens / time to first streamed token.
Decode tok/s  = (completion tokens - 1) / (last token time - first token time).
Draft counts come from the engine's final stream chunk: Fabric `timings.draft_n{,_accepted}`,
TensorFold MLX `speculative.{drafted,accepted}`, TensorFold CUDA `tensorfold.rounds` (accepted derived).
Standard library only.

  python3 bench_pp_tg.py http://127.0.0.1:8093 --engine fabric --prompts prompt.json --reps 3 --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import sys
import urllib.request
from pathlib import Path

MAX_TOKENS = 1024
WARMUP_TOKENS = 64
TIMEOUT_S = 3600


def request_body(engine: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if engine == "fabric":
        body["cache_prompt"] = False
    return body


def stream(base: str, body: dict) -> dict:
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    extras = {}
    text = []
    chunks = []
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            chunks.append({"elapsed_s": time.perf_counter() - start, "data": chunk})
            for key in ("usage", "timings", "tensorfold", "speculative", "error"):
                if chunk.get(key):
                    extras[key] = chunk[key]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
    if first is None:
        raise RuntimeError(f"no tokens streamed; server error: {extras.get('error')}")
    return {"start": start, "first": first, "last": last, "extras": extras, "text": "".join(text),
            "chunks": chunks}


def draft_counts(extras: dict, completion: int) -> dict:
    if "timings" in extras and "draft_n" in extras["timings"]:
        t = extras["timings"]
        return {"accepted": t.get("draft_n_accepted"), "drafted": t.get("draft_n"), "source": "fabric timings"}
    if "speculative" in extras:
        s = extras["speculative"]
        return {"accepted": s.get("accepted"), "drafted": s.get("drafted"), "rounds": s.get("rounds"),
                "source": "tensorfold speculative"}
    rounds = (extras.get("tensorfold") or {}).get("rounds")
    if rounds is not None:
        return {"accepted": completion - 1 - rounds, "drafted": None, "rounds": rounds,
                "source": "tensorfold cuda: completion - 1 - rounds"}
    return {"accepted": None, "drafted": None, "source": "none reported"}


def cached_tokens(extras: dict) -> int | None:
    if "timings" in extras:
        return extras["timings"].get("cache_n")
    tf = extras.get("tensorfold") or {}
    if "cached" in tf:
        return tf["cached"]
    details = (extras.get("usage") or {}).get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def server_rates(extras: dict, prompt_n: int, completion: int) -> dict:
    t = extras.get("timings")
    if t:
        return {"prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second")}
    tf = extras.get("tensorfold") or {}
    prefill_s = tf.get("prefill_s", tf.get("prefill_seconds"))
    decode_s = tf.get("decode_s")
    return {"prefill_tps": prompt_n / prefill_s if prefill_s else None,
            "decode_tps": (completion - 1) / decode_s if decode_s else tf.get("tokens_per_second")}


def measure(base: str, engine: str, prompt: str, expected_prompt: int, max_tokens: int,
            cache_evidence: dict | None = None) -> dict:
    body = request_body(engine, prompt, max_tokens)
    r = stream(base, body)
    usage = r["extras"].get("usage") or {}
    timings = r["extras"].get("timings") or {}
    prompt_n = usage.get("prompt_tokens", timings.get("prompt_n"))
    completion = usage.get("completion_tokens", timings.get("predicted_n"))
    ttft = r["first"] - r["start"]
    decode_s = r["last"] - r["first"]
    cached = cached_tokens(r["extras"])
    proven_uncached = engine == "tf" and cached is None and bool(cache_evidence)
    counts_valid = isinstance(prompt_n, int) and isinstance(completion, int)
    checks = {"prompt_tokens": prompt_n == expected_prompt, "completion_tokens": completion == max_tokens,
              "cached_zero": cached == 0 or proven_uncached,
              "server_error_free": not r["extras"].get("error"),
              "positive_intervals": ttft > 0 and decode_s > 0}
    rates = server_rates(r["extras"], prompt_n, completion) if counts_valid else {}
    draft = draft_counts(r["extras"], completion) if isinstance(completion, int) else {}
    return {"prompt_tokens": prompt_n, "completion_tokens": completion, "cached_tokens": cached,
            "cache_status": "reported" if cached is not None else "evidenced" if proven_uncached else "unreported",
            "cache_evidence": cache_evidence, "request": body,
            "ttft_s": ttft, "decode_s": decode_s,
            "prefill_tps": prompt_n / ttft if isinstance(prompt_n, int) and ttft > 0 else None,
            "decode_tps": (completion - 1) / decode_s if isinstance(completion, int) and decode_s > 0 else None,
            "server": {"prefill_tps": rates.get("prefill_tps"), "decode_tps": rates.get("decode_tps")},
            "draft": {"accepted": None, "drafted": None, **draft},
            "checks": checks, "ok": all(checks.values()),
            "text_head": r["text"][:160], "text": r["text"],
            "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
            "chunks": r["chunks"], "stream_chunk_count": len(r["chunks"]), "extras": r["extras"]}


def median_of(runs: list, key) -> float | None:
    values = [key(run) for run in runs if key(run) is not None]
    return statistics.median(values) if values else None


def summarize(runs: list) -> dict:
    return {"prefill_tps": median_of(runs, lambda r: r["prefill_tps"]),
            "decode_tps": median_of(runs, lambda r: r["decode_tps"]),
            "server_prefill_tps": median_of(runs, lambda r: r["server"]["prefill_tps"]),
            "server_decode_tps": median_of(runs, lambda r: r["server"]["decode_tps"]),
            "accepted": median_of(runs, lambda r: r["draft"]["accepted"]),
            "drafted": median_of(runs, lambda r: r["draft"]["drafted"]),
            "all_ok": bool(runs) and all(run["ok"] for run in runs)}


def save_result(args, runs: list, provenance: dict, error: str | None = None) -> dict:
    result = {"label": args.label, "engine": args.engine, "summary": summarize(runs), "runs": runs,
              "provenance": provenance, "complete": len(runs) == args.reps and error is None, "error": error}
    result["summary"]["all_ok"] &= result["complete"]
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1) + "\n")
    temporary.replace(output)
    return result


def run_repetitions(args, fixture: dict, provenance: dict, cache_evidence: dict | None) -> dict:
    runs = []
    try:
        stream(args.base, request_body(args.engine, fixture["warmup"], WARMUP_TOKENS))
        for i in range(args.prompt_start, args.prompt_start + args.reps):
            run = measure(args.base, args.engine, fixture["prompts"][i], fixture["prompt_tokens"],
                          args.max_tokens, cache_evidence)
            run["prompt_index"] = i
            runs.append(run)
            save_result(args, runs, provenance)
            print(json.dumps({"label": args.label, "prompt_index": i, "prefill_tps": run["prefill_tps"],
                              "decode_tps": run["decode_tps"], "draft": run["draft"],
                              "checks": run["checks"]}), flush=True)
    except Exception as error:
        save_result(args, runs, provenance, str(error))
        raise
    return save_result(args, runs, provenance)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--engine", choices=("fabric", "tf"), required=True)
    p.add_argument("--prompts", required=True, help="JSON from make_prompt.py")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prompt-start", type=int, default=0)
    p.add_argument("--manifest", required=True, help="Run provenance JSON")
    p.add_argument("--cache-evidence", help="TensorFold source/log evidence JSON when no cache count is reported")
    p.add_argument("--label", default="")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    fixture_bytes = Path(args.prompts).read_bytes()
    fixture = json.loads(fixture_bytes)
    if args.reps < 1 or args.prompt_start < 0 or args.prompt_start + args.reps > len(fixture["prompts"]):
        p.error("requested prompt range is outside the fixture")
    if args.max_tokens < 2:
        p.error("--max-tokens must be at least 2 for decode timing")
    provenance = {"manifest": json.loads(Path(args.manifest).read_text()),
                  "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                  "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "argv": sys.argv, "started_unix_s": time.time()}
    cache_evidence = json.loads(Path(args.cache_evidence).read_text()) if args.cache_evidence else None
    if cache_evidence is not None and (args.engine != "tf" or cache_evidence.get("uncached") is not True
                                      or not cache_evidence.get("source_revision") or not cache_evidence.get("artifact")):
        p.error("cache evidence requires TensorFold, uncached=true, source_revision and artifact")
    result = run_repetitions(args, fixture, provenance, cache_evidence)
    print(json.dumps({"label": args.label, **result["summary"]}), flush=True)
    if not result["summary"]["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
