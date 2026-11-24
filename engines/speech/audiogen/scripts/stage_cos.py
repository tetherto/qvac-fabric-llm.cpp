#!/usr/bin/env python3
"""Per-stage cosine similarity / rel_l2 between two --dump-stages directories.

Stage dumps are int32 {ndim=2, d0, d1} followed by d0*d1 float32 (StageDump::write).

Exits non-zero on mismatched stage rosters, malformed or nonfinite dumps, a
shape mismatch, an empty directory, or a worst-stage cosine under the bar.
A parity gate must actually compare valid stages before returning success.
"""
import pathlib
import struct
import sys

import numpy as np

PASS_COSINE = 0.999


def die(msg):
    print(f"stage_cos: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load(path):
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) != 12:
            die(f"{path}: truncated header, got {len(header)} of 12 bytes")
        ndim, d0, d1 = struct.unpack("iii", header)
        if ndim != 2 or d0 <= 0 or d1 <= 0:
            die(f"{path}: expected a nonempty 2-D stage, got {[ndim, d0, d1]}")
        payload = f.read()
    expected = d0 * d1
    if len(payload) != expected * 4:
        die(f"{path}: header declares {d0}x{d1} ({expected * 4} bytes) but payload holds {len(payload)} bytes")
    data = np.frombuffer(payload, dtype=np.float32)
    if not np.isfinite(data).all():
        die(f"{path}: stage contains nonfinite values")
    return data, (ndim, d0, d1)


def main():
    if len(sys.argv) != 3:
        die(f"usage: {pathlib.Path(sys.argv[0]).name} <dir-a> <dir-b>")
    a_dir, b_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for d in (a_dir, b_dir):
        if not d.is_dir():
            die(f"{d}: not a directory")

    stages = sorted(p.name for p in a_dir.glob("*.bin"))
    if not stages:
        die(f"{a_dir}: no *.bin stage dumps to compare")
    b_stages = {p.name for p in b_dir.glob("*.bin")}
    if set(stages) != b_stages:
        die(f"stage roster mismatch: only in {a_dir}: {sorted(set(stages) - b_stages)}; "
            f"only in {b_dir}: {sorted(b_stages - set(stages))}")

    print(f"{'stage':<20}{'shape':>18}{'cosine':>12}{'rel_l2':>12}{'max_abs':>12}")
    worst = 1.0
    for s in stages:
        b_path = b_dir / s
        if not b_path.exists():
            die(f"{s}: present in {a_dir} but missing from {b_dir}")
        a, a_dims = load(a_dir / s)
        b, b_dims = load(b_path)
        if a_dims != b_dims:
            die(f"{s}: shape mismatch, {list(a_dims[1:])} in {a_dir} vs {list(b_dims[1:])} in {b_dir}")

        a, b = a.astype(np.float64), b.astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 and nb == 0.0:
            cos = 1.0
        elif na == 0.0 or nb == 0.0:
            cos = 0.0  # one side collapsed to all-zero, the other did not
        else:
            cos = float(a @ b / (na * nb))
        rel = float(np.linalg.norm(a - b) / (na + 1e-30))
        max_abs = float(np.abs(a - b).max())
        if not np.isfinite([cos, rel, max_abs]).all():
            die(f"{s}: comparison produced nonfinite metrics")
        print(f"{s:<20}{str(list(a_dims[1:])):>18}{cos:12.6f}{rel:12.6f}{max_abs:12.4g}")
        worst = min(worst, cos)

    ok = worst >= PASS_COSINE
    print(f"\nworst-stage cosine over {len(stages)} stages: {worst:.6f}  "
          f"({'PASS' if ok else 'FAIL'} at >= {PASS_COSINE})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
