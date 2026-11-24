"""Recall of each implementation against an exact float32 search, plus how
often its top-k ids match another implementation's. Same data as
run_bench.py: dim 1024, 100,000 vectors, 1,000 queries, 1 thread, one batch
search per k (--ks, default 10 and 64; staged search's recall depends on k).

Each implementation runs once per bit width and k with BENCH_IDS set, so its
last search's ids land in <data-dir>/ids-<name>-q<bits>-k<k>.u64. The exact
top 64 per query is cached in <data-dir>/exact-top64.u64.

usage: python recall.py --out recall.md
         [--fabric NAME=PATH ...] [--upstream NAME=PYTHON ...]
         [--data-dir DIR] [--ks 10,64] [--json recall.json]
"""
import argparse
import json
import os
import sys

import numpy as np

from run_bench import DIM, K, HERE, host_name, make_data, run

RECALL_AT = (1, 10, 64)


def exact_top(base_path, query_path, data_dir):
    path = os.path.join(data_dir, f"exact-top{K}.u64")
    base = np.fromfile(base_path, dtype=np.float32).reshape(-1, DIM)
    queries = np.fromfile(query_path, dtype=np.float32).reshape(-1, DIM)
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.uint64).reshape(queries.shape[0], K)
    scores = queries @ base.T
    top = np.argpartition(-scores, K, axis=1)[:, :K]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1).astype(np.uint64)
    top.tofile(path)
    return top


def recall_at(found, exact, r):
    return float(np.mean([len(set(f[:r]) & set(e[:r])) / r for f, e in zip(found, exact)]))


def same_ids(a, b, r):
    return float(np.mean([len(set(x[:r]) & set(y[:r])) / r for x, y in zip(a, b)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabric", action="append", default=[])
    ap.add_argument("--upstream", action="append", default=[])
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--ks", default="10,64")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()

    ks = [int(v) for v in args.ks.split(",")]
    assert max(ks) <= K, f"k is at most {K}, the exact top cached here"
    base_path, query_path = make_data(args.data_dir)
    exact = exact_top(base_path, query_path, args.data_dir)
    n_q = exact.shape[0]
    names = []
    found = {}

    # A name containing "staged-off" turns staged search off; one
    # containing "calibrated" fits TQ+ on the first 4096 rows first.
    def variant(name):
        env = {}
        if "staged-off" in name:
            env["BENCH_STAGED"] = "0"
        if "calibrated" in name:
            env["BENCH_CALIBRATE"] = "4096"
        return env

    for bits in (2, 4):
        for k in ks:
            impls = []
            for spec in args.fabric:
                name, path = spec.split("=", 1)
                impls.append((name, [path, base_path, query_path, str(DIM), str(bits), str(k), "1", "1"], variant(name)))
            for spec in args.upstream:
                name, py = spec.split("=", 1)
                impls.append((name, [py, "-I", os.path.join(HERE, "bench_upstream.py"), base_path, query_path,
                                     str(DIM), str(bits), str(k), "1"], dict(variant(name), RAYON_NUM_THREADS="1")))
            for name, cmd, extra in impls:
                ids_path = os.path.join(args.data_dir, f"ids-{name}-q{bits}-k{k}.u64")
                run(cmd, dict(os.environ, BENCH_IDS=ids_path, **extra))
                found[(name, bits, k)] = np.fromfile(ids_path, dtype=np.uint64).reshape(n_q, k)
                if name not in names:
                    names.append(name)
                rs = ", ".join(f"recall@{r} {recall_at(found[(name, bits, k)], exact, r):.4f}" for r in RECALL_AT if r <= k)
                print(f"q{bits} k{k} {name}: {rs}", flush=True)

    rows, same = [], []
    for bits in (2, 4):
        for k in ks:
            for name in names:
                ids = found[(name, bits, k)]
                rows.append({"bits": bits, "k": k, "name": name,
                             **{f"r{r}": (round(recall_at(ids, exact, r), 4) if r <= k else None) for r in RECALL_AT}})
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    same.append({"bits": bits, "k": k, "a": a, "b": b,
                                 "top10": round(same_ids(found[(a, bits, k)], found[(b, bits, k)], min(10, k)), 4),
                                 "topk": round(same_ids(found[(a, bits, k)], found[(b, bits, k)], k), 4)})

    def dash(v):
        return "-" if v is None else f"{v:.4f}"

    with open(args.out, "w") as f:
        f.write(f"Host: {host_name()}\n\n")
        f.write(f"dim {DIM}, 100,000 vectors, {n_q:,} queries, 1 thread, one batch search per k; recall against an exact "
                f"float32 inner-product search over the same vectors\n\n")
        f.write("| Bits | k | Implementation | Recall@1 | Recall@10 | Recall@64 |\n| ---: | ---: | :--- | ---: | ---: | ---: |\n")
        for r in rows:
            f.write(f"| {r['bits']} | {r['k']} | {r['name']} | {dash(r['r1'])} | {dash(r['r10'])} | {dash(r['r64'])} |\n")
        if same:
            f.write("\nShare of top-k ids two implementations have in common, averaged over the queries\n\n")
            f.write("| Bits | k | Pair | Top 10 | Top k |\n| ---: | ---: | :--- | ---: | ---: |\n")
            for s in same:
                f.write(f"| {s['bits']} | {s['k']} | {s['a']} vs {s['b']} | {s['top10']:.4f} | {s['topk']:.4f} |\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"recall": rows, "same": same}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
