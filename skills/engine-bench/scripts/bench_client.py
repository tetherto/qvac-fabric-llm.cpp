#!/usr/bin/env python3
"""OpenAI-compatible load generator for llama-server vs mlx_lm.server.

Uses /v1/completions with a hand-formatted Qwen chat prompt so both servers see
identical tokens (no chat-template differences).
"""
import zlib
import argparse
import glob
import http.client
import json
import os
import random
import statistics
import threading
import time
from urllib.parse import urlparse

EOS_IDS = [151643, 151645]  # Qwen <|endoftext|>, <|im_end|>


def fmt(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_corpus(path_glob):
    txt = []
    for f in sorted(glob.glob(path_glob, recursive=True)):
        try:
            txt.append(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            pass
    return "\n\n".join(txt)


ESSAY_TOPICS = [
    "the history of computer architecture", "how compilers optimize code", "the design of operating system schedulers",
    "distributed consensus algorithms", "the evolution of programming languages", "database storage engines",
    "network protocol design", "GPU programming models", "memory hierarchies and caching", "software testing practice",
    "the economics of open-source software", "numerical precision in machine learning", "file system design",
    "garbage collection strategies", "the history of the internet", "large-scale system reliability",
]


def make_doc_prompts(corpus, n, approx_tokens, nonce, long_output=False):
    chars = approx_tokens * 4
    # long prompts (depth sweeps up to 128k tokens) can outgrow the corpus: repeat it rather than fall short.
    # The "[req nonce-i]" tag at the start of every prompt still keeps prefixes unique.
    while corpus and len(corpus) < 2 * chars:
        corpus = corpus + "\n\n" + corpus
    rnd = random.Random(1234)
    out = []
    for i in range(n):
        off = rnd.randrange(0, max(1, len(corpus) - chars))
        chunk = corpus[off:off + chars]
        if long_output:
            # open-ended task that runs far past any max_tokens we use, so no engine needs EOS suppression
            topic = ESSAY_TOPICS[i % len(ESSAY_TOPICS)]
            out.append(fmt(f"[req {nonce}-{i}] Background document:\n\n{chunk}\n\nUsing the document above only as loose background, "
                           f"write an extremely long, exhaustive essay of at least 5000 words on {topic}. "
                           f"Cover every aspect in depth, with many sections and examples. Do not stop early."))
        else:
            out.append(fmt(f"[req {nonce}-{i}] Summarize the following document in detail, section by section:\n\n{chunk}"))
    return out


SPEC_PROMPTS = {
    "code": "Write a complete, well-commented Python implementation of an LRU cache class with get/put, plus a set of unittest test cases covering eviction order and capacity edge cases.",
    "rewrite": None,  # filled from corpus: rewrite a passage (high draft acceptance)
    "story": "Write a short story (about 400 words) about a lighthouse keeper who discovers a message in a bottle.",
    "explain": "Explain step by step how TCP congestion control works, covering slow start, congestion avoidance, fast retransmit and fast recovery.",
}


def request(url, body, timeout=1800) -> dict:
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]}")
    t_first = None
    n_chunks = 0
    usage = None
    text = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        ch = ev.get("choices") or []
        if ch and ch[0].get("text"):
            if t_first is None:
                t_first = time.perf_counter()
            n_chunks += 1
            text.append(ch[0]["text"])
        if ev.get("timings"):
            usage = usage or {}
            usage["timings"] = ev["timings"]
    t_end = time.perf_counter()
    conn.close()
    ntok = (usage or {}).get("completion_tokens") or n_chunks
    ptok = (usage or {}).get("prompt_tokens")
    return dict(t0=t0, t_first=t_first or t_end, t_end=t_end, n_tok=ntok, n_chunks=n_chunks,
                p_tok=ptok, text="".join(text), timings=(usage or {}).get("timings"))


def base_body(args, prompt, max_tokens, ban_eos):
    b: dict = dict(prompt=prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0, stream=True,
                   stream_options={"include_usage": True}, cache_prompt=False)
    if args.model:
        b["model"] = args.model
    if ban_eos and not args.long_output:
        b["logit_bias"] = {str(t): -100.0 for t in args.eos_ids}
        b["ignore_eos"] = True
    return b


def run_conc(args):
    corpus = load_corpus(args.corpus)
    results = []
    # warmup
    request(args.url, base_body(args, fmt("Hello, say hi."), 16, True))
    for C in [int(c) for c in args.concurrency.split(",")]:
        nonce = f"{args.label}-{C}-{time.time_ns()}"
        nreq = C * args.rounds
        prompts = make_doc_prompts(corpus, nreq, args.prompt_tokens, nonce, args.long_output)
        lock = threading.Lock()
        idx = [0]
        recs = []

        def worker():
            while True:
                with lock:
                    if idx[0] >= nreq:
                        return
                    i = idx[0]
                    idx[0] += 1
                r = request(args.url, base_body(args, prompts[i], args.gen, True))
                with lock:
                    recs.append(r)

        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker) for _ in range(C)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.perf_counter() - t0
        gen = sum(r["n_tok"] for r in recs)
        ptok = sum((r["p_tok"] or 0) for r in recs)
        ttft = [r["t_first"] - r["t0"] for r in recs]
        dec = [(r["n_tok"] - 1) / (r["t_end"] - r["t_first"]) for r in recs if r["t_end"] > r["t_first"] and r["n_tok"] > 1]
        row = dict(mode="conc", label=args.label, conc=C, n_req=nreq, wall_s=round(wall, 2),
                   gen_tok=gen, prompt_tok=ptok, agg_gen_tps=round(gen / wall, 1),
                   agg_total_tps=round((gen + ptok) / wall, 1),
                   ttft_mean_s=round(statistics.mean(ttft), 3), ttft_p90_s=round(sorted(ttft)[int(0.9 * (len(ttft) - 1))], 3),
                   per_req_tps_mean=round(statistics.mean(dec), 1) if dec else None,
                   short_outputs=sum(1 for r in recs if r["n_tok"] < args.gen))
        print(json.dumps(row), flush=True)  # noqa: NP100
        results.append(row)
    return results


def run_spec(args):
    corpus = load_corpus(args.corpus)
    rnd = random.Random(7)
    off = rnd.randrange(0, len(corpus) - 4000)
    prompts: dict[str, str] = {k: v or "" for k, v in SPEC_PROMPTS.items()}
    prompts["rewrite"] = "Rewrite the following text verbatim, but fix any typos and convert headings to bold text:\n\n" + corpus[off:off + 3000]
    request(args.url, base_body(args, fmt("Hello, say hi."), 16, False))
    results = []
    for name, p in prompts.items():
        for rep in range(args.reps):
            prefix = f"[spec-{rep}]\n" if args.fixed_prefix else f"[{args.label}-{rep}-{time.time_ns()}]\n"
            r = request(args.url, base_body(args, fmt(prefix + p), args.gen, False))
            dec_tps = (r["n_tok"] - 1) / (r["t_end"] - r["t_first"]) if r["n_tok"] > 1 else None
            row = dict(mode="spec", label=args.label, prompt=name, rep=rep, n_tok=r["n_tok"], p_tok=r["p_tok"],
                       ttft_s=round(r["t_first"] - r["t0"], 3), e2e_s=round(r["t_end"] - r["t0"], 2),
                       dec_tps=round(dec_tps, 1) if dec_tps else None,
                       text_hash=hash(r["text"]) & 0xFFFFFFFF, text_crc=zlib.crc32(r["text"].encode()), text_head=r["text"][:60])
            if r["timings"] and "draft_n" in r["timings"]:
                row["draft_n"] = r["timings"].get("draft_n")
                row["draft_acc"] = r["timings"].get("draft_n_accepted")
            print(json.dumps(row), flush=True)  # noqa: NP100
            results.append(row)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fixed-prefix", action="store_true", help="spec mode: the same prompt prefix for every engine and run")
    ap.add_argument("--mode", choices=["conc", "spec"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--corpus", default=os.path.join(os.environ.get("BENCH_DIR", os.path.expanduser("~/bench-weekly")), "corpus/docs/**/*.md"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="model id to send (required by some servers, e.g. oMLX)")
    ap.add_argument("--long-output", action="store_true", help="open-ended essay prompts, no EOS suppression")
    ap.add_argument("--eos-ids", type=lambda s: [int(x) for x in s.split(",")], default=EOS_IDS)
    a = ap.parse_args()
    rows = run_conc(a) if a.mode == "conc" else run_spec(a)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
