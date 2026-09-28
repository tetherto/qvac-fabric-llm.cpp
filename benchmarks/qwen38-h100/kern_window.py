#!/usr/bin/env python3
"""Aggregate kernels from an nsys sqlite export, restricted to the window after the last prefill kernel.

llama-bench `-p 0 -n N -d D` traces contain the depth fill (prefill) followed by the timed decode.
Decode at batch 1 never launches cuBLAS GEMM or the full-weight dequantize kernels, so the end of the
last such kernel marks the start of the decode window.

Usage: kern_window.py <trace.sqlite> [--n-tokens N] [--top K] [--marker REGEX]
Prints per-family time, per-token time, launch counts, and the idle gap between consecutive kernels.
"""

import argparse
import re
import sqlite3
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kern_families import family_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--marker", default=r"nvjet|dequantize_block_(q[0-9]_K|iq[0-9]_xs|q[0-9]_[01]|iq[0-9]_nl|q3_K)")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    q = (
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start"
    )
    rows = db.execute(q).fetchall()
    if not rows:
        print("no kernels", file=sys.stderr)
        return 1

    marker = re.compile(args.marker)
    t_cut = 0
    for start, end, name in rows:
        if marker.search(name):
            t_cut = max(t_cut, end)
    win = [r for r in rows if r[0] >= t_cut]
    if not win:
        print("empty window", file=sys.stderr)
        return 1

    span = win[-1][1] - win[0][0]
    busy = sum(e - s for s, e, _ in win)
    gaps = 0
    prev_end = win[0][1]
    for s, e, _ in win[1:]:
        if s > prev_end:
            gaps += s - prev_end
        prev_end = max(prev_end, e)

    fam_t = {}
    fam_n = {}
    per_name = {}
    for s, e, name in win:
        fam = family_of(name)
        fam_t[fam] = fam_t.get(fam, 0) + (e - s)
        fam_n[fam] = fam_n.get(fam, 0) + 1
        t, n = per_name.get(name, (0, 0))
        per_name[name] = (t + (e - s), n + 1)

    n = args.n_tokens
    print(f"decode window: {span/1e6:.1f} ms wall, {busy/1e6:.1f} ms kernel busy, {gaps/1e6:.1f} ms idle gaps, {len(win)} launches")
    print(f"per token: {span/1e6/n:.3f} ms wall, {busy/1e6/n:.3f} ms busy, {gaps/1e6/n:.3f} ms gaps, {len(win)/n:.0f} launches")
    print(f"{'family':18s} {'share':>7s} {'ms/token':>9s} {'launch/tok':>10s}")
    for fam, t in sorted(fam_t.items(), key=lambda kv: -kv[1]):
        print(f"{fam:18s} {100*t/busy:6.1f}% {t/1e6/n:9.3f} {fam_n[fam]/n:10.1f}")
    print()
    print(f"top {args.top} kernels:")
    for name, (t, cnt) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{100*t/busy:6.1f}% {t/1e6/n:8.3f} ms/tok {cnt/n:7.1f}x/tok  {name[:105]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
