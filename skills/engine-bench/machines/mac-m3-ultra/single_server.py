#!/usr/bin/env python3
"""Prefill and decode of one sequence through a running llama-server, from the server's own timings.

usage: single_server.py --url URL --out bench-<engine>.jsonl --label TAG --errors errors.jsonl [--sizes 512,4096]
       [--gen 128] [--reps 3]

Replaces llama-bench in the weekly bench (since 2026-11-09): llama-bench builds its context without common's setup, so
qvac-fabric's Core ML FFN sidecar (--coreml-ffn) never runs there. Through llama-server every engine runs with the same
flags as in the other tests. Each request is a prompt of exactly N token ids (text from the server's /tokenize,
repeated) with no prompt cache, then --gen tokens with EOS ignored. prompt_per_second and predicted_per_second come
from the response's timings, so HTTP and tokenizing are not counted. One unmeasured request comes first.

Writes rows in llama-bench's jsonl shape, which summarize.py reads: {"n_prompt": N, "n_gen": 0, "avg_ts": prefill t/s}
per size and {"n_prompt": 0, "n_gen": G, "avg_ts": decode t/s} (decode after the smallest prompt), each with
stddev_ts, samples_ts and "tool": "llama-server". A failure appends a row to the errors file and exits 1.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

TEXT = ("The printing press changed how knowledge moved across Europe. Before it, books were copied by hand in "
        "monasteries and universities, and a single volume could take months. Gutenberg's movable type, oil-based ink "
        "and adapted wine press made it possible to print hundreds of identical pages a day. Within fifty years presses "
        "were running in more than two hundred cities, and millions of books were in circulation. Prices fell, literacy "
        "rose, and scholars could compare the same text in different places. Vernacular languages gained standard "
        "spellings, pamphlets spread religious and political arguments faster than authorities could answer them, and "
        "scientific results could be checked and built on by readers who never met the authors. ")


def post(url, body, timeout=1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--errors", required=True)
    ap.add_argument("--sizes", default="512,4096")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    step = "tokenize"
    try:
        toks = post(a.url + "/tokenize", {"content": TEXT})["tokens"]
        ids = []
        while len(ids) < max(sizes):
            ids += toks
        step = "warm-up"

        def body(n):
            return {"prompt": ids[:n], "n_predict": a.gen, "ignore_eos": True, "temperature": 0, "cache_prompt": False}

        post(a.url + "/completion", body(sizes[0]))
        pp: dict[int, list[float]] = {n: [] for n in sizes}
        tg: list[float] = []
        for rep in range(a.reps):
            for n in sizes:
                step = f"pp{n} round {rep + 1}"
                t = post(a.url + "/completion", body(n))["timings"]
                if t.get("prompt_n") != n or t.get("predicted_n") != a.gen:
                    raise RuntimeError(f"prompt_n {t.get('prompt_n')} (want {n}), predicted_n {t.get('predicted_n')} (want {a.gen})")
                pp[n].append(round(t["prompt_per_second"], 2))
                if n == sizes[0]:
                    tg.append(round(t["predicted_per_second"], 2))
                print(f"{a.label} pp{n} {t['prompt_per_second']:.1f} t/s, tg{a.gen} {t['predicted_per_second']:.2f} t/s", flush=True)  # noqa: NP100
    except Exception as e:
        with open(a.errors, "a") as f:
            f.write(json.dumps({"label": a.label, "step": step, "error": f"{type(e).__name__}: {e}"[:300],
                                "at": time.strftime("%H:%M:%S")}) + "\n")
        print(f"{a.label}: {step} failed: {e}", file=sys.stderr)  # noqa: NP100
        sys.exit(1)

    def row(n_prompt, n_gen, v):
        return {"n_prompt": n_prompt, "n_gen": n_gen, "avg_ts": round(statistics.mean(v), 2),
                "stddev_ts": round(statistics.stdev(v), 2) if len(v) > 1 else 0.0, "samples_ts": v, "tool": "llama-server"}

    with open(a.out, "w") as f:
        for n in sizes:
            f.write(json.dumps(row(n, 0, pp[n])) + "\n")
        f.write(json.dumps(row(0, a.gen, tg)) + "\n")


if __name__ == "__main__":
    main()
