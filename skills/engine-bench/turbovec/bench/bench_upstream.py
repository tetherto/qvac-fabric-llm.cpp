"""Upstream turbovec bench, same protocol as bench_fabric.cpp: one cold add,
then the 1,000-query search timed `iters` times. Prints one JSON line.
Threads come from RAYON_NUM_THREADS, set by the caller.
BENCH_ONE=1 searches one query per call, as bench_fabric.cpp does; the
Python call overhead is included in each latency.
TURBOVEC_STAGE_TIMES=1 with a turbovec built from the stage-timers branch
prints its per-stage totals to stderr after the runs, in bench_fabric's
format (see print_stage_times). The PyPI wheel has no timers; it says so.

usage: python bench_upstream.py base.f32 queries.f32 dim bits k iters
"""
import json
import os
import sys
import time

import numpy as np
import turbovec

# Upstream's stages folded into ours (ggml-vector-index-turbovec-staged.cpp,
# StageClock). Ours charges the low planes build to seed and the exact
# tables build to rescore, so upstream's are folded the same way; a short
# query's unseeded rescan is part of our scan.
OUR_STAGES = [
    ("rotate", ["rotate"]),
    ("tables", ["sign_tables"]),
    ("seed", ["seed", "low_planes"]),
    ("scan", ["scan", "rescan"]),
    ("shortlist", ["shortlist"]),
    ("rank", ["rank"]),
    ("rescore", ["rescore", "exact_tables"]),
    ("batch_scan", ["batch_scan"]),
]


def print_stage_times(search_s):
    """Per-stage totals over every search this process ran, warm-up pass
    included, as bench_fabric's timers count. The denominator is the
    number of queries that went through upstream's planes path (the sum of
    nq over those search calls), the same as our "staged queries". Calls
    are the first listed piece's: per search call for rotate, tables,
    seed, scan, batch_scan, shortlist; per query for rank and rescore.
    search_s is the wall time of every idx.search call, Python included."""
    if os.environ.get("TURBOVEC_STAGE_TIMES", "") in ("", "0"):
        return
    fn = getattr(turbovec._turbovec, "_stage_times", None)
    if fn is None:
        print("stage times: this turbovec has no timers (PyPI build)", file=sys.stderr)
        return
    enabled, n_planes, n_other, rows = fn()
    sys.stdout.flush()  # the JSON line first when both streams share a file
    raw = {name: (ns, calls) for name, ns, calls in rows}
    q = max(1, n_planes)
    err = sys.stderr
    print(f"upstream stage times over {n_planes} planes-path queries "
          f"({n_other} took the full scan, enabled={enabled}):", file=err)
    total_ns = 0
    for ours, pieces in OUR_STAGES:
        ns = sum(raw[p][0] for p in pieces)
        total_ns += ns
        calls = raw[pieces[0]][1]
        print(f"  {ours:<10} {ns / 1e3 / q:10.1f} us/query  ({calls} calls, {ns / 1e6:.1f} ms total)", file=err)
    print(f"  {'sum':<10} {total_ns / 1e3 / q:10.1f} us/query", file=err)
    n_all = max(1, n_planes + n_other)
    print(f"  {'search()':<10} {search_s * 1e6 / n_all:10.1f} us/query  (wall, Python call included, "
          f"over {n_planes + n_other} queries)", file=err)
    print("  upstream pieces:", file=err)
    for name, ns, calls in rows:
        print(f"    {name:<12} {ns / 1e3 / q:10.1f} us/query  ({calls} calls, {ns / 1e6:.1f} ms total)", file=err)


def main():
    base_path, query_path = sys.argv[1], sys.argv[2]
    dim, bits, k, iters = (int(v) for v in sys.argv[3:7])
    base = np.fromfile(base_path, dtype=np.float32).reshape(-1, dim)
    queries = np.fromfile(query_path, dtype=np.float32).reshape(-1, dim)
    ids = np.arange(base.shape[0], dtype=np.uint64)

    t0 = time.perf_counter()
    idx = turbovec.IdMapIndex(dim, bits)
    calib = int(os.environ.get("BENCH_CALIBRATE", "0"))
    if calib > 0:
        idx.calibrate(base[:calib])
    idx.add_with_ids(base, ids)
    add_ms = (time.perf_counter() - t0) * 1e3

    search_s = 0.0
    if os.environ.get("BENCH_ONE", "0") != "0":
        call_us = []
        top1 = None
        for p in range(iters + 1):
            for q in range(queries.shape[0]):
                one = queries[q:q + 1]
                s0 = time.perf_counter()
                result = idx.search(one, k)
                us = (time.perf_counter() - s0) * 1e6
                if p > 0:
                    call_us.append(us)
                if q == 0:
                    top1 = int(np.asarray(result[1])[0][0])
                search_s += us / 1e6
        call_us.sort()
        print(json.dumps({"add_ms": round(add_ms, 3), "p50_us": round(call_us[len(call_us) // 2], 1),
                          "p95_us": round(call_us[len(call_us) * 95 // 100], 1), "top1_q0": top1}))
        print_stage_times(search_s)
        return

    search_ms = []
    result = None
    for _ in range(iters):
        s0 = time.perf_counter()
        result = idx.search(queries, k)
        search_ms.append((time.perf_counter() - s0) * 1e3)
    assert result is not None, "iters must be at least 1"
    search_s = sum(search_ms) / 1e3
    search_ms.sort()
    # BENCH_IDS=<path> writes the last search's ids, n_q x k uint64, for recall.py.
    ids_path = os.environ.get("BENCH_IDS", "")
    if ids_path:
        np.ascontiguousarray(np.asarray(result[1]), dtype=np.uint64).tofile(ids_path)
    top1 = int(np.asarray(result[1])[0][0])
    print(json.dumps({"add_ms": round(add_ms, 3), "search_ms": round(search_ms[len(search_ms) // 2], 3), "top1_q0": top1}))
    print_stage_times(search_s)


if __name__ == "__main__":
    main()
