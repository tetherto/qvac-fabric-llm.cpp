#!/usr/bin/env python3
"""Compare per-AR-iteration dumps from two `mm3-replay --dump-iters` runs.

`--dump-iters N [--dump-dir DIR]` writes, for iteration i in 0..N-1, five raw
little-endian f32 files with no header: `ar-iter-<i>-last-hidden.f32` (2 x
hidden), `ar-iter-<i>-sem-logits.f32` (2 x semantic_vocab: conditional half
then unconditional half), `ar-iter-<i>-guided.f32` (CFG-guided logits,
semantic_vocab floats, EOS slot already dropped), `ar-iter-<i>-feedback.f32`
(2 x hidden), and `ar-iter-<i>-depth-hidden.f32` (depth-decoder hiddens for
the frame).

In `--mode replay`, the LM and depth decoder still run their full forward
pass every iteration and are only teacher-forced on the final token choice
(see mm3_ar_choose_semantic/mm3_ar_decode_depth in mm3-ar-loop.h), so two
dump directories captured from the same replay inputs at different quant
levels or on different backends have the same iteration count and are
directly comparable position by position. A mismatched iteration count
between the two directories means the inputs weren't the same replay run,
not that quantization changed anything, so it is treated as an error rather
than a truncated comparison.

usage:
  compare_ar_dumps.py REFERENCE_DIR CANDIDATE_DIR
      [--min-cosine X] [--min-argmax-agree PCT] [--json]

Exit 0 if the comparison ran clean and any given gates pass, 1 otherwise
(including on a directory that can't actually be compared -- a gate that
returns 0 without having compared anything is worse than no gate).

numpy carries the per-element work when it is importable; an equivalent
stdlib implementation takes over when it is not, so the gate runs anywhere.
"""
import argparse
import array
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

FIELDS = ["last-hidden", "sem-logits", "guided", "feedback", "depth-hidden"]
ARGMAX_FIELDS = ["guided", "sem-logits"]
TOP_K = 8
F32_BYTES = 4


def read_f32(path):
    if np is not None:
        return np.fromfile(path, dtype="<f4")
    raw = path.read_bytes()
    values = array.array("f")
    values.frombytes(raw[:len(raw) - len(raw) % F32_BYTES])
    if sys.byteorder != "little":
        values.byteswap()
    return values


def accumulate_finite(a, b):
    """Dot product, both squared norms, and the both-finite position count,
    accumulated in one pass over the vectors."""
    count = 0
    dot = square_a = square_b = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        count += 1
        dot += x * y
        square_a += x * x
        square_b += y * y
    return count, dot, math.sqrt(square_a), math.sqrt(square_b)


def finite_dot_norms(a, b):
    if np is None:
        return accumulate_finite(a, b)
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a) & np.isfinite(b)
    count = int(np.count_nonzero(finite))
    if count == 0:
        return 0, 0.0, 0.0, 0.0
    a_f, b_f = a[finite].astype(np.float64), b[finite].astype(np.float64)
    return count, float(a_f @ b_f), float(np.linalg.norm(a_f)), float(np.linalg.norm(b_f))


def argmax_index(vec):
    if np is None:
        return max(range(len(vec)), key=vec.__getitem__)
    return int(np.argmax(vec))


def top_k_indices(vec, k):
    """Indices of the k largest entries. Both backends sort ascending and
    stably, so a tie at the k-th position resolves the same way in each."""
    if k <= 0:
        return set()
    if np is None:
        return set(sorted(range(len(vec)), key=vec.__getitem__)[-k:])
    return set(np.argsort(vec, kind="stable")[-k:].tolist())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def min_index(values):
    return min(range(len(values)), key=values.__getitem__)


def die(message):
    print(f"compare_ar_dumps: {message}", file=sys.stderr)
    raise SystemExit(1)


def dump_path(dump_dir, iteration, field):
    return dump_dir / f"ar-iter-{iteration}-{field}.f32"


def load_field(dump_dir, iteration, field):
    path = dump_path(dump_dir, iteration, field)
    if not path.exists():
        die(f"{path}: missing")
    data = read_f32(path)
    if len(data) == 0:
        die(f"{path}: empty")
    return data


def count_iterations(dump_dir):
    count = 0
    while dump_path(dump_dir, count, "sem-logits").exists():
        count += 1
    if count == 0:
        die(f"{dump_dir}: no ar-iter-0-sem-logits.f32 (not a --dump-iters output directory)")
    return count


def cosine(a, b):
    """Cosine over the elementwise-both-finite subset, plus the fraction of
    positions that subset covers. `guided` carries -inf entries from top-k
    masking (and two runs need not mask the exact same indices, so the
    finite fraction is itself a signal: how much the two runs' top-k sets
    overlap, not just their relative order). Returns (cosine, finite_fraction);
    cosine is None only when no position is finite on both sides -- that is
    reported as missing, not silently coerced to a number."""
    count, dot, norm_a, norm_b = finite_dot_norms(a, b)
    finite_fraction = count / len(a) if len(a) else 0.0
    if count == 0:
        return None, finite_fraction

    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0, finite_fraction
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0, finite_fraction
    return dot / (norm_a * norm_b), finite_fraction


def conditional_half(sem_logits):
    if len(sem_logits) % 2 != 0:
        die("sem-logits length is odd; expected equal-sized [conditional, unconditional] halves")
    half = len(sem_logits) // 2
    return sem_logits[:half]


def top_k_overlap(ref_vec, cand_vec, k):
    k = min(k, len(ref_vec))
    return len(top_k_indices(ref_vec, k) & top_k_indices(cand_vec, k))


def argmax_stat(ref_vec, cand_vec):
    return {
        "agree": argmax_index(ref_vec) == argmax_index(cand_vec),
        "top_k_overlap": top_k_overlap(ref_vec, cand_vec, TOP_K),
    }


def compare_field_at_iteration(ref_dir, cand_dir, iteration, field):
    ref = load_field(ref_dir, iteration, field)
    cand = load_field(cand_dir, iteration, field)
    if len(ref) != len(cand):
        die(f"iteration {iteration} {field}: size mismatch ({len(ref)} vs {len(cand)})")
    return ref, cand


def compare_iteration(ref_dir, cand_dir, iteration, field_cosines, argmax_accum):
    per_field_cosine = {}
    for field in FIELDS:
        ref, cand = compare_field_at_iteration(ref_dir, cand_dir, iteration, field)
        cosine_value, finite_fraction = cosine(ref, cand)
        field_cosines[field].append({"cosine": cosine_value, "finite_fraction": finite_fraction})
        per_field_cosine[field] = cosine_value

        # argmax/top-k rank -inf lowest, which is what the sampler sees, so
        # they take the raw vectors; cosine needs the finite-both subset.
        if field == "guided":
            argmax_accum["guided"].append(argmax_stat(ref, cand))
        elif field == "sem-logits":
            argmax_accum["sem-logits"].append(argmax_stat(conditional_half(ref), conditional_half(cand)))

    comparable = [value for value in per_field_cosine.values() if value is not None]
    return min(comparable) if comparable else None


def compare(ref_dir, cand_dir, n_iterations):
    field_cosines = {field: [] for field in FIELDS}
    argmax_accum = {name: [] for name in ARGMAX_FIELDS}
    per_iteration_min = []

    for iteration in range(n_iterations):
        per_iteration_min.append(compare_iteration(ref_dir, cand_dir, iteration, field_cosines, argmax_accum))

    return field_cosines, argmax_accum, per_iteration_min


def summarize_field(entries):
    """entries[i] = {"cosine": float|None, "finite_fraction": float} for
    iteration i. Cosine stats are computed only over iterations that had a
    comparable (finite-on-both-sides) subset; worst_iteration still refers
    to the original iteration index, not a position in the filtered list."""
    n = len(entries)
    comparable = [(i, e["cosine"]) for i, e in enumerate(entries) if e["cosine"] is not None]
    mean_finite_fraction = mean([e["finite_fraction"] for e in entries])

    if not comparable:
        return {
            "n": n, "n_comparable": 0, "mean_cosine": None, "min_cosine": None,
            "worst_iteration": None, "mean_finite_fraction": mean_finite_fraction,
        }

    cosines_only = [c for _, c in comparable]
    worst_iteration, min_cosine = comparable[min_index(cosines_only)]
    return {
        "n": n,
        "n_comparable": len(comparable),
        "mean_cosine": mean(cosines_only),
        "min_cosine": float(min_cosine),
        "worst_iteration": int(worst_iteration),
        "mean_finite_fraction": mean_finite_fraction,
    }


def summarize_argmax(entries):
    n = len(entries)
    agree = sum(1 for e in entries if e["agree"])
    overlap_sum = sum(e["top_k_overlap"] for e in entries)
    return {
        "n": n,
        "argmax_agree_pct": 100.0 * agree / n if n else 0.0,
        "mean_top_k_overlap": overlap_sum / n if n else 0.0,
        "top_k": TOP_K,
    }


def build_summary(field_cosines, argmax_accum, per_iteration_min):
    comparable = [(i, v) for i, v in enumerate(per_iteration_min) if v is not None]
    if comparable:
        overall_worst_iteration, overall_worst_min_cosine = min(comparable, key=lambda pair: pair[1])
    else:
        overall_worst_iteration, overall_worst_min_cosine = None, None

    return {
        "fields": {field: summarize_field(entries) for field, entries in field_cosines.items()},
        "argmax": {name: summarize_argmax(entries) for name, entries in argmax_accum.items()},
        "overall_worst_iteration": overall_worst_iteration,
        "overall_worst_min_cosine": overall_worst_min_cosine,
    }


def comparability_violations(summary):
    if summary["overall_worst_min_cosine"] is not None:
        return []
    return ["no field had a comparable (finite-on-both-sides) position in any iteration"]


def apply_gates(summary, min_cosine, min_argmax_agree):
    violations = []
    if min_cosine is not None:
        for field, stats in summary["fields"].items():
            if stats["min_cosine"] is None:
                violations.append(f"{field}: no comparable (finite-on-both-sides) position in any iteration")
            elif stats["min_cosine"] < min_cosine:
                violations.append(
                    f"{field}: min_cosine {stats['min_cosine']:.6f} < {min_cosine} "
                    f"(worst at iteration {stats['worst_iteration']})"
                )
    if min_argmax_agree is not None:
        for name, stats in summary["argmax"].items():
            if stats["argmax_agree_pct"] < min_argmax_agree:
                violations.append(
                    f"{name}: argmax_agree_pct {stats['argmax_agree_pct']:.2f} < {min_argmax_agree}"
                )
    return violations


def fmt_float(value, spec=".6f"):
    return format(value, spec) if value is not None else "n/a"


def fmt_int(value):
    return str(value) if value is not None else "n/a"


def print_summary(summary, n_iterations):
    print(f"compared {n_iterations} iteration(s)\n")
    print(f"{'field':<14}{'n':>6}{'n_cmp':>7}{'mean_cosine':>14}{'min_cosine':>14}{'worst_iter':>12}{'finite%':>10}")
    for field, stats in summary["fields"].items():
        print(f"{field:<14}{stats['n']:>6}{stats['n_comparable']:>7}"
              f"{fmt_float(stats['mean_cosine']):>14}{fmt_float(stats['min_cosine']):>14}"
              f"{fmt_int(stats['worst_iteration']):>12}{stats['mean_finite_fraction']:>9.1%}")

    print(f"\n{'argmax':<26}{'n':>6}{'agree_pct':>12}{'mean_top8_overlap':>20}")
    for name, stats in summary["argmax"].items():
        print(f"{name:<26}{stats['n']:>6}{stats['argmax_agree_pct']:>11.2f}%{stats['mean_top_k_overlap']:>19.2f}")

    if summary["overall_worst_iteration"] is None:
        print("\noverall worst iteration: n/a (no field had a comparable position in any iteration)")
    else:
        print(f"\noverall worst iteration: {summary['overall_worst_iteration']} "
              f"(min cosine {summary['overall_worst_min_cosine']:.6f} across all fields at that iteration)")

    if summary["violations"]:
        print(f"\nGATE: FAIL ({len(summary['violations'])} violation(s))")
        for violation in summary["violations"]:
            print(f"  {violation}")
    elif summary.get("gated"):
        print("\nGATE: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--min-cosine", type=float, default=None,
                        help="every field's min_cosine across iterations must be >= this")
    parser.add_argument("--min-argmax-agree", type=float, default=None,
                        help="percent (0-100); both guided and sem-logits argmax_agree_pct must be >= this")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ref_dir, cand_dir = Path(args.reference_dir), Path(args.candidate_dir)
    for directory in (ref_dir, cand_dir):
        if not directory.is_dir():
            die(f"{directory}: not a directory")

    n_ref, n_cand = count_iterations(ref_dir), count_iterations(cand_dir)
    if n_ref != n_cand:
        die(f"iteration count mismatch: {ref_dir} has {n_ref}, {cand_dir} has {n_cand} "
            f"-- replay mode teacher-forces both runs to the same length, so a mismatch "
            f"means these two dump directories are not from the same replay inputs")

    field_cosines, argmax_accum, per_iteration_min = compare(ref_dir, cand_dir, n_ref)
    summary = build_summary(field_cosines, argmax_accum, per_iteration_min)
    summary["gated"] = args.min_cosine is not None or args.min_argmax_agree is not None
    summary["violations"] = (comparability_violations(summary) or
                             apply_gates(summary, args.min_cosine, args.min_argmax_agree))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, n_ref)

    return 1 if summary["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
