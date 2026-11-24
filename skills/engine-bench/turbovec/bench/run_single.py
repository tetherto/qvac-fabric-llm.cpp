"""Single-query bench: one query per search call, like RAG. Same data as
run_bench.py; each row is the median over --samples fresh processes of the
per-call p50 and p95 (BENCH_ONE=1 in bench_fabric.cpp / bench_upstream.py).

usage: python run_single.py --out results-single.md
         [--fabric NAME=PATH ...]     staged on and off are both run
         [--upstream NAME=PYTHON ...]
         [--threads 1,8] [--ks 10,64] [--samples 3]
"""
import argparse
import json
import os
import statistics

from run_bench import DIM, HERE, host_name, make_data, run

ITERS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabric", action="append", default=[])
    ap.add_argument("--upstream", action="append", default=[])
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--ks", default="10,64")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()

    base_path, query_path = make_data(args.data_dir)
    rows = []
    for bits in (2, 4):
        for k in (int(v) for v in args.ks.split(",")):
            for nthreads in (int(t) for t in args.threads.split(",")):
                impls = []
                for spec in args.fabric:
                    name, path = spec.split("=", 1)
                    cmd = [path, base_path, query_path, str(DIM), str(bits), str(k), str(nthreads), str(ITERS)]
                    if name.endswith("base"):
                        # no staged switch in that build
                        if nthreads == 1:
                            impls.append((name, cmd, dict(os.environ, BENCH_ONE="1")))
                        continue
                    for staged in ("1", "0"):
                        impls.append((f"{name} staged {'on' if staged == '1' else 'off'}", cmd,
                                      dict(os.environ, BENCH_ONE="1", BENCH_STAGED=staged)))
                for spec in args.upstream:
                    name, py = spec.split("=", 1)
                    cmd = [py, "-I", os.path.join(HERE, "bench_upstream.py"), base_path, query_path, str(DIM), str(bits), str(k), str(ITERS)]
                    impls.append((name, cmd, dict(os.environ, BENCH_ONE="1", RAYON_NUM_THREADS=str(nthreads))))
                for name, cmd, env in impls:
                    samples = [run(cmd, env) for _ in range(args.samples)]
                    p50 = statistics.median(s["p50_us"] for s in samples)
                    p95 = statistics.median(s["p95_us"] for s in samples)
                    top1 = sorted({s["top1_q0"] for s in samples})
                    rows.append((bits, k, nthreads, name, p50, p95, top1))
                    print(f"q{bits} k{k} t{nthreads} {name}: p50 {p50:.1f} us, p95 {p95:.1f} us, top1 {top1}", flush=True)

    with open(args.out, "w") as f:
        f.write(f"Host: {host_name()}\n\n")
        f.write(f"dim {DIM}, 100,000 vectors, 1,000 queries searched one per call, {ITERS} timed passes after a warm-up; "
                f"median of {args.samples} fresh processes. Upstream latency includes the Python call.\n\n")
        f.write("| Bits | k | Threads | Implementation | p50 per query | p95 per query | Top-1 id, query 0 |\n")
        f.write("| ---: | ---: | ---: | :--- | ---: | ---: | ---: |\n")
        for bits, k, nthreads, name, p50, p95, top1 in rows:
            f.write(f"| {bits} | {k} | {nthreads} | {name} | {p50:,.1f} us | {p95:,.1f} us | {','.join(map(str, top1))} |\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"bits": bits, "k": k, "threads": nthreads, "name": name, "p50_us": p50, "p95_us": p95, "top1": top1,
                        "samples": args.samples} for bits, k, nthreads, name, p50, p95, top1 in rows], f, indent=1)


if __name__ == "__main__":
    main()
