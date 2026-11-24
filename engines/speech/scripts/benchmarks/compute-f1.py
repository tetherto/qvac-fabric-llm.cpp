#!/usr/bin/env python3
# Compute frame-based Precision / Recall / F1 for a VAD hypothesis against a
# reference of speech-segment time ranges. Invoked by
# scripts/benchmarks/run-family.sh when a family's spec in families.json
# carries a `correctness` block with kind='f1'.
#
# Usage:
#   compute-f1.py --hypothesis-file HYP  --reference REF.json \
#                 [--collar-ms N]  [--json-out PATH]
#   compute-f1.py --self-test                       # run built-in checks
#
# Reference format: JSON array of {"start": <sec>, "end": <sec>} objects.
#     [{"start": 0.30, "end": 10.90}, ...]
#
# Hypothesis format is auto-detected:
#   (a) whisper-vad-speech-segments stdout, one line per segment:
#         "Speech segment N: start = 0.30, end = 10.20"
#       (header "Detected N speech segments:" and blank lines are ignored)
#   (b) JSON array with the same shape as the reference file — supported for
#       test parity and for local runs against arbitrary VAD backends.
#
# Output JSON:
#   {
#     "f1":              0.987,
#     "precision":       0.991,
#     "recall":          0.983,
#     "collar_ms":       100,
#     "n_ref_segments":  1,
#     "n_hyp_segments":  1,
#     "reference":       "path/to/ref.json"
#   }
#
# Implementation: frame-based (10 ms resolution) with a symmetric collar
# around reference-segment boundaries — frames inside the collar are not
# scored, matching the standard VAD evaluation convention (a VAD should not
# be penalized for a 40 ms boundary offset when the reference is only
# annotated to 100 ms precision). Non-matching stdout lines (banners,
# backend-init logs, unrecognized formats) are skipped with a stderr
# diagnostic — a hard raise would silently null the score on every dispatch
# the moment upstream adds a print, defeating the whole "correctness
# scoring catches regressions" purpose. Pure stdlib — no pyannote / Torch.

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


FRAME_MS = 10   # scoring resolution; matches compute-der.py's convention.


# ---- parsers ----------------------------------------------------------------


def parse_reference_json(text: str) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] from a JSON array of {start,end}."""
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"reference must be a JSON array, got {type(data).__name__}")
    out: list[tuple[float, float]] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict) or "start" not in obj or "end" not in obj:
            raise ValueError(f"reference entry {i}: expected object with start/end, got {obj!r}")
        s = float(obj["start"])
        e = float(obj["end"])
        if e > s:
            out.append((s, e))
    return out


# Format emitted by whisper-vad-speech-segments (third_party/whisper.cpp
# examples/vad-speech-segments/speech.cpp): "Speech segment N: start = X.XX, end = Y.YY".
# Pin the exact format via a regex rather than a positional split so a future
# upstream change (an added prefix, a different separator) fails loudly at
# parse time rather than silently mis-scoring.
#
# UNITS: the numbers printed by this format are CENTISECONDS (see
# whisper.cpp's `samples_to_cs()` at src/whisper.cpp:4758 and the debug print
# at 7115 which explicitly `/100.0`s the same field before displaying it as
# seconds). The example itself doesn't divide, so "start = 29.00" means
# 0.29 s. The parser below converts to seconds by dividing by 100.
_WHISPER_VAD_SEG_RE = re.compile(
    r"^Speech segment\s+\d+\s*:\s*start\s*=\s*([0-9.]+)\s*,\s*end\s*=\s*([0-9.]+)\s*$"
)


def parse_hypothesis(text: str) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] from a hypothesis file/string.

    Auto-detects between the two supported shapes:
      * A JSON array in the same shape as the reference.
      * whisper-vad-speech-segments text output, one "Speech segment N:"
        line per detected segment. Header/blank/other lines are skipped
        with a stderr diagnostic (mirrors compute-der.py's post-review
        defensive-skip behavior).

    Blank input (VAD emitted nothing) yields the empty list, which
    downstream scores as pure recall-miss.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # JSON-array shape: input starts with `[`. Fall through to text parse on
    # any JSON error rather than raising — some future variant might mix a
    # bracketed diagnostic with the segment listing.
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if not isinstance(data, list):
                raise ValueError("top-level not a list")
            out: list[tuple[float, float]] = []
            for i, obj in enumerate(data):
                if not isinstance(obj, dict) or "start" not in obj or "end" not in obj:
                    raise ValueError(f"entry {i}: missing start/end")
                s, e = float(obj["start"]), float(obj["end"])
                if e > s:
                    out.append((s, e))
            return out
        except (json.JSONDecodeError, ValueError) as e:
            print(f"compute-f1.py: JSON hypothesis parse failed ({e}); falling back to text parse",
                  file=sys.stderr)

    segments: list[tuple[float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = _WHISPER_VAD_SEG_RE.match(line)
        if m is None:
            # Header "Detected N speech segments:" is expected and silent;
            # everything else gets a diagnostic so upstream format drift
            # surfaces in the workflow log.
            if not line.startswith("Detected ") or "speech segments" not in line:
                print(f"compute-f1.py: skipping unrecognized hypothesis line {lineno}: {raw!r}",
                      file=sys.stderr)
            continue
        # Values are centiseconds — see _WHISPER_VAD_SEG_RE comment.
        s = float(m.group(1)) / 100.0
        e = float(m.group(2)) / 100.0
        if e > s:
            segments.append((s, e))
    return segments


# ---- framing / scoring ------------------------------------------------------


def _segments_to_mask(
    segments: list[tuple[float, float]],
    n_frames: int,
) -> list[bool]:
    """One bool per 10 ms frame: True iff any segment covers the frame."""
    mask = [False] * n_frames
    for start, end in segments:
        i0 = max(0, int(round(start * 1000 / FRAME_MS)))
        i1 = min(n_frames, int(round(end * 1000 / FRAME_MS)))
        for i in range(i0, i1):
            mask[i] = True
    return mask


def _collar_mask(
    ref_segments: list[tuple[float, float]],
    n_frames: int,
    collar_ms: int,
) -> list[bool]:
    """True for frames excluded from scoring by the collar around ref boundaries.

    Standard VAD evaluation: don't penalize the VAD for a boundary landing
    within +/- collar_ms of a reference boundary, because the reference
    itself is typically annotated to no better than that precision. Matches
    compute-der.py's collar semantics — collar_ms=0 disables exclusion
    entirely (no max(1,...) fudge).
    """
    if collar_ms <= 0:
        return [False] * n_frames
    excluded = [False] * n_frames
    half = int(round(collar_ms / FRAME_MS))
    for start, end in ref_segments:
        for edge_s in (start, end):
            center = int(round(edge_s * 1000 / FRAME_MS))
            lo = max(0, center - half)
            hi = min(n_frames, center + half)
            for i in range(lo, hi):
                excluded[i] = True
    return excluded


def compute_f1(
    hypothesis_segments: list[tuple[float, float]],
    reference_segments:  list[tuple[float, float]],
    collar_ms:           int = 100,
) -> dict:
    """Compute frame-based precision / recall / F1. See module docstring."""
    if collar_ms < 0:
        raise ValueError(f"collar_ms must be >= 0 (got {collar_ms})")

    max_end = 0.0
    for _, e in reference_segments:
        max_end = max(max_end, e)
    for _, e in hypothesis_segments:
        max_end = max(max_end, e)
    n_frames = int(round(max_end * 1000 / FRAME_MS))

    ref_mask = _segments_to_mask(reference_segments,  n_frames)
    hyp_mask = _segments_to_mask(hypothesis_segments, n_frames)
    collar   = _collar_mask(reference_segments, n_frames, collar_ms)

    # Precision/recall/F1 on the speech-positive class:
    #   TP = frames where ref speaks AND hyp speaks
    #   FP = frames where hyp speaks AND ref does not
    #   FN = frames where ref speaks AND hyp does not
    # Collar frames are excluded from all three so boundary-slop within the
    # tolerance doesn't inflate FP/FN.
    tp = fp = fn = 0
    for i in range(n_frames):
        if collar[i]:
            continue
        r, h = ref_mask[i], hyp_mask[i]
        if r and h:
            tp += 1
        elif h and not r:
            fp += 1
        elif r and not h:
            fn += 1

    # Undefined-boundary conventions mirror compute-wer.py / compute-der.py:
    # both-empty is a perfect (1.0) rather than degenerate (0.0/undefined),
    # so a family with no ref segments and a VAD that emitted nothing scores
    # as agreement instead of a null.
    if not reference_segments and not hypothesis_segments:
        precision = recall = f1 = 1.0
    elif not reference_segments:
        # Ref says silence throughout, hyp claims speech somewhere — all FP.
        precision = 0.0
        recall    = 1.0     # no ref speech to miss
        f1        = 0.0
    elif not hypothesis_segments:
        # Ref has speech, hyp emitted nothing — all FN.
        precision = 1.0     # no hyp claims to be wrong about
        recall    = 0.0
        f1        = 0.0
    elif (tp + fp + fn) == 0:
        # Both sides declared segments but the collar excluded every scorable
        # frame (e.g., a short reference segment entirely inside its own
        # boundary tolerance). Treat as agreement rather than 0.0 — the
        # both-empty branch above already establishes that "nothing to
        # disagree about" scores 1.0. Not exercisable at the shipped families
        # spec (10.6 s ref, 100 ms collar), but the alternative — silently
        # returning 0.0 for a perfect-but-uncounted hyp — would mislead any
        # future family that ships a shorter reference or a wider tolerance.
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) \
                    if (precision + recall) > 0 else 0.0

    return {
        "f1":              f1,
        "precision":       precision,
        "recall":          recall,
        "collar_ms":       collar_ms,
        "n_ref_segments":  len(reference_segments),
        "n_hyp_segments":  len(hypothesis_segments),
    }


# ---- self-test --------------------------------------------------------------


def _self_test() -> int:
    """Runs known F1 cases covering exact match, miss, FA, boundary slop
    with and without collar, empty ref/hyp, and hypothesis-parser dual-shape
    parity (whisper-vad text vs JSON array)."""

    # 4 s of speech from 2.0 to 6.0 out of an 8 s window. 400 speech frames
    # (frames 200-599), 400 non-speech frames.
    ref = [(2.0, 6.0)]

    cases: list[tuple[str, list, list, int, float, float, float]] = [
        # (label, ref, hyp, collar_ms, want_precision, want_recall, want_f1)
        ("identical",           ref, [(2.0, 6.0)],           0, 1.0, 1.0, 1.0),
        ("all-miss",            ref, [],                     0, 1.0, 0.0, 0.0),
        ("all-fa (empty ref)",  [],  [(2.0, 6.0)],           0, 0.0, 1.0, 0.0),
        ("both empty",          [],  [],                     0, 1.0, 1.0, 1.0),
        # Hyp covers half the ref: 200 TP, 0 FP, 200 FN.
        # precision = 200/200 = 1.0, recall = 200/400 = 0.5, f1 = 2*1*0.5/1.5 = 0.6666...
        ("half recall",         ref, [(2.0, 4.0)],           0, 1.0, 0.5, 2.0/3.0),
        # Hyp extends 2 s past the ref end: 400 TP, 200 FP, 0 FN.
        # precision = 400/600 = 0.6666..., recall = 400/400 = 1.0, f1 = 2*0.666*1/(1.666) = 0.8
        ("over-eager (200 FP)", ref, [(2.0, 8.0)],           0, 2.0/3.0, 1.0, 0.8),
    ]

    failed: list[str] = []
    for label, r, h, collar_ms, wp, wr, wf in cases:
        got = compute_f1(h, r, collar_ms=collar_ms)
        for name, want, actual in [("precision", wp, got["precision"]),
                                    ("recall",    wr, got["recall"]),
                                    ("f1",        wf, got["f1"])]:
            if abs(actual - want) > 1e-6:
                failed.append(f"{label}: {name} {actual} != {want}")

    # Boundary-slop case: hyp shifted 100 ms later than ref (2.1-6.1 vs
    # 2.0-6.0). Overlap = 390 frames (2.1-6.0), FN = 10 frames (2.0-2.1),
    # FP = 10 frames (6.0-6.1). Precision = recall = 390/400 = 0.975,
    # F1 = 0.975. With a 200 ms collar (±20 frame ring around 2.0 and 6.0),
    # both slop regions fall entirely inside the excluded frames, so
    # precision = recall = F1 = 1.0 — the collar-off vs collar-on delta
    # this pair pins down would fail loudly if _collar_mask regressed to
    # excluding a stray 1-frame ring when collar_ms=0 (compute-der.py bug
    # from PR #236's review that we don't want reintroduced here).
    slop = [(2.1, 6.1)]
    g0 = compute_f1(slop, ref, collar_ms=0)
    if not (abs(g0["precision"] - 0.975) < 1e-6 and abs(g0["recall"] - 0.975) < 1e-6):
        failed.append(f"boundary slop no-collar: {g0}")
    g200 = compute_f1(slop, ref, collar_ms=200)
    if not (abs(g200["f1"] - 1.0) < 1e-6):
        failed.append(f"boundary slop 200ms-collar: f1={g200['f1']} (expected 1.0)")

    # Parser dual-shape parity: whisper-vad text output and equivalent JSON
    # array must produce identical F1 scores against the same reference.
    # whisper-vad text values are CENTISECONDS (see _WHISPER_VAD_SEG_RE),
    # so "start = 200.00, end = 600.00" is 2.0-6.0 s — the same range the
    # JSON hypothesis names directly in seconds.
    hyp_text = (
        "\n"
        "Detected 1 speech segments:\n"
        "Speech segment 0: start = 200.00, end = 600.00\n"
        "\n"
    )
    hyp_json = '[{"start": 2.0, "end": 6.0}]'
    f_text = compute_f1(parse_hypothesis(hyp_text), ref, collar_ms=0)["f1"]
    f_json = compute_f1(parse_hypothesis(hyp_json), ref, collar_ms=0)["f1"]
    if f_text != f_json or f_text != 1.0:
        failed.append(f"parser parity: text={f_text}, json={f_json} (both should be 1.0)")

    # Robustness against non-recognized stdout chatter (banner/backend logs
    # from a future upstream change). Would fail loudly if the parser aborted
    # on the first stray line. Segment values in centiseconds as above.
    mixed = (
        "whisper_vad: loading model...\n"
        "Detected 1 speech segments:\n"
        "Speech segment 0: start = 200.00, end = 600.00\n"
        "info: total time = 42ms\n"
    )
    f_mixed = compute_f1(parse_hypothesis(mixed), ref, collar_ms=0)["f1"]
    if f_mixed != 1.0:
        failed.append(f"mixed-stdout robustness: f1={f_mixed} (expected 1.0)")

    # Regression guard for the centiseconds unit bug: the previous parser
    # treated "start = 29.00" as 29 s (past the end of an 11 s clip), giving
    # F1 = 0.0 on the real silero-on-jfk verification dispatch. Pin that the
    # whisper-vad text ranges from a realistic silero-on-jfk run land inside
    # the audio window and score a reasonable non-zero F1 against a
    # single-segment reference covering the same span. Would fail loudly if
    # the /100 conversion is ever accidentally removed.
    ref_jfk = [(0.30, 10.90)]
    silero_jfk = (
        "\nDetected 5 speech segments:\n"
        "Speech segment 0: start = 29.00, end = 221.00\n"
        "Speech segment 1: start = 330.00, end = 377.00\n"
        "Speech segment 2: start = 400.00, end = 435.00\n"
        "Speech segment 3: start = 538.00, end = 765.00\n"
        "Speech segment 4: start = 816.00, end = 1059.00\n\n"
    )
    got = compute_f1(parse_hypothesis(silero_jfk), ref_jfk, collar_ms=100)
    if not (0.7 < got["f1"] < 0.9):
        failed.append(f"centiseconds unit fix: silero-on-jfk F1={got['f1']} (expected ~0.75-0.85)")

    # No-scorable-frames edge case: a 50 ms ref segment with a 100 ms collar
    # excludes every ref frame. A hyp that matches exactly should score 1.0
    # (agreement, nothing to disagree about) — the pre-fix behavior collapsed
    # this to 0.0 via a div-by-zero fallback, misleading any future family
    # with short refs or wide tolerance. Verified: without the fix, tp=fp=fn=0
    # would land in the previous `else` branch and return 0.0.
    tiny = [(2.00, 2.05)]
    perfect = compute_f1([(2.00, 2.05)], tiny, collar_ms=100)
    if perfect["f1"] != 1.0:
        failed.append(
            f"no-scorable-frames edge case: matching hyp on collar-eaten ref "
            f"F1={perfect['f1']} (expected 1.0)"
        )

    if failed:
        for f in failed:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("compute-f1.py: self-test ok")
    return 0


# ---- CLI --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute VAD frame-F1 for one hyp/ref pair.")
    ap.add_argument("--hypothesis-file", type=pathlib.Path,
                    help="Hypothesis file: whisper-vad text output OR JSON array.")
    ap.add_argument("--reference",       type=pathlib.Path,
                    help="Reference JSON: [{start, end}, ...] array of speech segments.")
    ap.add_argument("--collar-ms",       type=int, default=100,
                    help="Symmetric collar around reference boundaries in ms (default: 100).")
    ap.add_argument("--json-out",        type=pathlib.Path,
                    help="Write result JSON to this path in addition to stdout.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run built-in F1 cases and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.reference:
        print("--reference is required", file=sys.stderr); return 2
    if not args.hypothesis_file:
        print("--hypothesis-file is required", file=sys.stderr); return 2

    ref_segments = parse_reference_json(args.reference.read_text(encoding="utf-8"))
    hyp_text = args.hypothesis_file.read_text(encoding="utf-8") if args.hypothesis_file.exists() else ""
    hyp_segments = parse_hypothesis(hyp_text)

    result = compute_f1(hyp_segments, ref_segments, collar_ms=args.collar_ms)
    result["reference"] = str(args.reference)

    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.json_out:
        args.json_out.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
