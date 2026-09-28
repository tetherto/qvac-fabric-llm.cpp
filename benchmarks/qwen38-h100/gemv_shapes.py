#!/usr/bin/env python3
"""Per-shape GPU durations of the F8 GEMV kernels in an nsys sqlite export.

The shape is recovered from the grid (rows = gridX * rows per block); the durations are GPU timestamps, so a loaded
host does not inflate them. Usage: gemv_shapes.py <trace.sqlite> [--rows-per-block 4] [--skip N]
"""
import argparse
import sqlite3
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--rows-per-block", type=int, default=4)
    ap.add_argument("--skip", type=int, default=20, help="warmup launches to drop per (kernel, grid)")
    ap.add_argument("--pattern", default="mul_mat_vec_f8")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    q = (
        "SELECT k.gridX, k.end - k.start, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id WHERE s.value LIKE ? ORDER BY k.start"
    )
    groups = {}
    for grid_x, dur, name in db.execute(q, ("%" + args.pattern + "%",)):
        short = name.split("(")[0]
        groups.setdefault((short, grid_x), []).append(dur)

    print(f"{'kernel':<56} {'gridX':>6} {'rows':>6} {'n':>6} {'mean us':>9} {'median':>9} {'min':>9}")
    for (short, grid_x), durs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        d = durs[args.skip:] if len(durs) > args.skip else durs
        us = [x / 1000.0 for x in d]
        print(f"{short[:56]:<56} {grid_x:>6} {grid_x * args.rows_per_block:>6} {len(us):>6} "
              f"{statistics.fmean(us):>9.2f} {statistics.median(us):>9.2f} {min(us):>9.2f}")


if __name__ == "__main__":
    main()
