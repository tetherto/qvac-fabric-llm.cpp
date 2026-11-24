"""Runs the TurboVec bench matrix and writes a markdown table.

Setup: dim 1024, 100,000 base vectors, 1,000 queries, k 64, synthetic
unit-normalized vectors. Every sample is a fresh
process (cold add); the search time is the median of 5 calls in it, and the
table reports the median over --samples processes.

usage: python run_bench.py --out results.md
         [--fabric NAME=PATH ...]   bench_fabric builds; a name ending in
                                    "base" has no thread API
         [--upstream NAME=PYTHON ...]  python with turbovec installed
         [--threads 1,8] [--samples 5]
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys

import numpy as np

DIM, N_BASE, N_QUERY, K, ITERS = 1024, 100_000, 1_000, 64, 5
HERE = os.path.dirname(os.path.abspath(__file__))


def make_data(data_dir):
    base_path = os.path.join(data_dir, "base.f32")
    query_path = os.path.join(data_dir, "queries.f32")
    if not (os.path.exists(base_path) and os.path.exists(query_path)):
        rng = np.random.default_rng(0)
        for path, n in ((base_path, N_BASE), (query_path, N_QUERY)):
            v = rng.standard_normal((n, DIM), dtype=np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v.tofile(path)
    return base_path, query_path


def run(cmd, env=None):
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def host_name():
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except OSError:
        pass
    return f"{cpu}, {os.cpu_count()} cores, {platform.system()} {platform.machine()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabric", action="append", default=[])
    ap.add_argument("--upstream", action="append", default=[])
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()

    base_path, query_path = make_data(args.data_dir)
    threads = [int(t) for t in args.threads.split(",")]
    rows = []
    for bits in (2, 4):
        for nthreads in threads:
            impls = []
            for spec in args.fabric:
                name, path = spec.split("=", 1)
                if name.endswith("base") and nthreads != 1:
                    continue
                impls.append((name, [path, base_path, query_path, str(DIM), str(bits), str(K), str(nthreads), str(ITERS)], None))
            for spec in args.upstream:
                name, py = spec.split("=", 1)
                env = dict(os.environ, RAYON_NUM_THREADS=str(nthreads))
                impls.append((name, [py, os.path.join(HERE, "bench_upstream.py"), base_path, query_path, str(DIM), str(bits), str(K), str(ITERS)], env))
            for name, cmd, env in impls:
                # a "base" build (no thread API) is the old full-scan port, whose add takes minutes: 3 samples at most
                n = min(args.samples, 3) if name.endswith("base") else args.samples
                samples = [run(cmd, env) for _ in range(n)]
                add_ms = statistics.median(s["add_ms"] for s in samples)
                search_ms = statistics.median(s["search_ms"] for s in samples)
                top1 = {s["top1_q0"] for s in samples}
                row = (bits, nthreads, name, add_ms, search_ms, N_QUERY / (search_ms / 1e3), sorted(top1), n)
                rows.append(row)
                print(f"q{bits} t{nthreads} {name}: add {add_ms / 1e3:.3f} s, search {search_ms:.1f} ms, top1 {sorted(top1)}", flush=True)

    with open(args.out, "w") as f:
        f.write(f"Host: {host_name()}\n\n")
        f.write(f"dim {DIM}, {N_BASE:,} vectors, {N_QUERY:,} queries, k {K}; median of {args.samples} fresh processes, search median of {ITERS} calls each\n\n")
        f.write("| Bits | Threads | Implementation | Cold add | Search latency for 1,000 queries | Throughput | Top-1 id, query 0 |\n")
        f.write("| ---: | ---: | :--- | ---: | ---: | ---: | ---: |\n")
        for bits, nthreads, name, add_ms, search_ms, qps, top1, _ in rows:
            f.write(f"| {bits} | {nthreads} | {name} | {add_ms / 1e3:.3f} s | {search_ms:,.1f} ms | {qps:,.0f} q/s | {','.join(map(str, top1))} |\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"bits": bits, "threads": nthreads, "name": name, "add_ms": add_ms, "search_ms": search_ms,
                        "qps": round(qps, 1), "top1": top1, "samples": n}
                       for bits, nthreads, name, add_ms, search_ms, qps, top1, n in rows], f, indent=1)


if __name__ == "__main__":
    main()
