#!/usr/bin/env python3
# Compute Diarization Error Rate (DER) for one hypothesis against one
# reference RTTM. Invoked by scripts/benchmarks/run-family.sh when a family's
# spec in families.json carries a `correctness` block with kind='der'.
#
# Usage:
#   compute-der.py --hypothesis-jsonl HYP.jsonl \
#                  --reference        REF.rttm  \
#                  [--collar-ms N]              \
#                  [--json-out PATH]
#
#   compute-der.py --self-test                    # run built-in checks
#
# Hypothesis format: one JSON object per line, {"speaker": int, "start": s,
# "end": s}. Matches what parakeet-cli --emit jsonl writes on stdout for the
# diarize path.
#
# Reference format: standard RTTM. Only SPEAKER rows are parsed:
#     SPEAKER <file> <chan> <start> <duration> <NA> <NA> <spkr_id> <NA> <NA>
# Extra whitespace and comment lines (;) are tolerated.
#
# Output JSON:
#   {
#     "der":            0.082,
#     "miss":           0.010,
#     "false_alarm":    0.020,
#     "confusion":      0.052,
#     "collar_ms":      250,
#     "n_ref_speakers": 3,
#     "n_hyp_speakers": 3,
#     "reference":      "path/to/ref.rttm"
#   }
#
# Implementation: frame-based (10 ms resolution) with a collar around
# reference-segment boundaries, and optimal hypothesis→reference speaker
# mapping via brute-force permutation search. Sortformer's largest bench
# spec is 4 speakers => 24 permutations, so the O(n!) mapping is a
# non-issue in practice. Pure stdlib — no pyannote / Torch on CI.

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys


FRAME_MS = 10   # scoring resolution; 10 ms matches pyannote's default.


# ---- parsers ----------------------------------------------------------------


def parse_rttm(text: str) -> list[tuple[float, float, str]]:
    """Return [(start_s, duration_s, speaker_id), ...] from RTTM text.

    Only SPEAKER rows are parsed. Whitespace-tolerant. Bad rows raise a
    ValueError with the offending line so a malformed reference fails
    loudly rather than silently scoring against a subset.
    """
    segments: list[tuple[float, float, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            # RTTM allows SPKR-INFO / LEXEME / NOSCORE / etc. rows we don't
            # score against; skip rather than raise.
            continue
        if len(parts) < 8:
            raise ValueError(
                f"RTTM line {lineno}: SPEAKER row needs at least 8 fields, "
                f"got {len(parts)}: {raw!r}"
            )
        try:
            start = float(parts[3])
            dur   = float(parts[4])
        except ValueError as e:
            raise ValueError(f"RTTM line {lineno}: start/duration not numeric: {raw!r}") from e
        speaker = parts[7]
        if dur > 0:
            segments.append((start, dur, speaker))
    return segments


def parse_hypothesis_jsonl(text: str) -> list[tuple[float, float, str]]:
    """Return [(start_s, duration_s, speaker_id), ...] from JSONL hypothesis.

    Each non-blank line SHOULD be a JSON object with `speaker`, `start`, and
    either `end` or `duration`. Non-JSON lines (banners, backend-init logs,
    stray --verbose output from the CLI) are skipped with a stderr diagnostic
    rather than aborting the whole score — parakeet-cli --verbose currently
    only prints to stderr on the diarize path, but relying on that invariant
    would turn any future upstream leak into a silent null DER, defeating
    the whole "correctness scoring catches regressions" purpose. Blank
    stdout still yields the empty list (scored as pure miss downstream).
    """
    segments: list[tuple[float, float, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"compute-der.py: skipping non-JSON hypothesis line {lineno}: {raw!r}",
                  file=sys.stderr)
            continue
        if not isinstance(obj, dict):
            print(f"compute-der.py: skipping non-object hypothesis line {lineno}: "
                  f"expected object, got {type(obj).__name__}",
                  file=sys.stderr)
            continue
        if "speaker" not in obj or "start" not in obj:
            print(f"compute-der.py: skipping hypothesis line {lineno}, missing speaker/start: {raw!r}",
                  file=sys.stderr)
            continue
        start = float(obj["start"])
        if "duration" in obj:
            dur = float(obj["duration"])
        elif "end" in obj:
            dur = float(obj["end"]) - start
        else:
            print(f"compute-der.py: skipping hypothesis line {lineno}, missing end/duration: {raw!r}",
                  file=sys.stderr)
            continue
        speaker = str(obj["speaker"])
        if dur > 0:
            segments.append((start, dur, speaker))
    return segments


# ---- framing / scoring ------------------------------------------------------


def _segments_to_frames(
    segments: list[tuple[float, float, str]],
    n_frames: int,
    speaker_idx: dict[str, int],
) -> list[list[int]]:
    """One list of active-speaker indices per 10 ms frame.

    Overlapping segments (same-speaker or cross-speaker) collapse into a
    single frame carrying multiple speaker indices, matching standard DER
    handling of overlap.
    """
    frames: list[list[int]] = [[] for _ in range(n_frames)]
    for start, dur, spk in segments:
        i0 = max(0, int(round(start * 1000 / FRAME_MS)))
        i1 = min(n_frames, int(round((start + dur) * 1000 / FRAME_MS)))
        idx = speaker_idx.setdefault(spk, len(speaker_idx))
        for i in range(i0, i1):
            if idx not in frames[i]:
                frames[i].append(idx)
    return frames


def _collar_mask(
    ref_segments: list[tuple[float, float, str]],
    n_frames: int,
    collar_ms: int,
) -> list[bool]:
    """True for frames excluded by the collar around ref-segment boundaries.

    Standard DER practice: don't penalize the diarizer for boundary-timing
    slop within +/- collar_ms of any reference start or end. collar_ms=0
    disables exclusion entirely — no `max(1, ...)` here, otherwise a
    zero-collar request silently strips a 1-frame ring around every
    boundary, biasing DER by a symmetric-but-nonzero amount.
    """
    if collar_ms <= 0:
        return [False] * n_frames
    excluded = [False] * n_frames
    half = int(round(collar_ms / FRAME_MS))
    for start, dur, _ in ref_segments:
        for edge_s in (start, start + dur):
            center = int(round(edge_s * 1000 / FRAME_MS))
            lo = max(0, center - half)
            hi = min(n_frames, center + half)
            for i in range(lo, hi):
                excluded[i] = True
    return excluded


def _score_with_mapping(
    ref_frames: list[list[int]],
    hyp_frames: list[list[int]],
    collar: list[bool],
    hyp_to_ref: dict[int, int],
) -> tuple[int, int, int, int]:
    """Return (miss_frames, fa_frames, confusion_frames, ref_speech_frames)
    counted only on non-collar frames.

    Multi-speaker frames are scored per-speaker (each active ref speaker
    contributes one unit to ref_speech and one potential miss/confusion).
    """
    miss = fa = conf = ref_speech = 0
    for i, ex in enumerate(collar):
        if ex:
            continue
        r = ref_frames[i]
        h = hyp_frames[i]
        mapped_h = {hyp_to_ref[x] for x in h if x in hyp_to_ref}
        ref_speech += len(r)
        # md-eval per-frame split: correct = |r ∩ mapped_h|; unmatched ref
        # speakers split as miss = max(0, |r| - |mapped_h|) and the rest
        # are confusion. Total DER is invariant to this split (miss+conf =
        # unmatched-ref-count either way), but the miss/confusion fields
        # are consumed for regression diagnosis, so match the standard
        # rather than putting everything into confusion.
        correct = sum(1 for rs in r if rs in mapped_h)
        unmatched = len(r) - correct
        miss += max(0, len(r) - len(mapped_h))
        conf += unmatched - max(0, len(r) - len(mapped_h))
        fa   += max(0, len(mapped_h) - len(r))
    return miss, fa, conf, ref_speech


def _best_mapping(
    ref_frames: list[list[int]],
    hyp_frames: list[list[int]],
    collar: list[bool],
    n_ref_speakers: int,
    n_hyp_speakers: int,
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    """Search hyp->ref speaker mappings; return (mapping, best_counts).

    Brute-force over permutations. For n_hyp <= n_ref we permute a subset
    of ref indices to match each hyp speaker; if n_hyp > n_ref the extra
    hyp speakers map to unique out-of-range indices (score as pure FA).
    """
    best: tuple[int, int, int, int] | None = None
    best_map: dict[int, int] = {}

    ref_ids = list(range(n_ref_speakers))
    hyp_ids = list(range(n_hyp_speakers))

    # If either side has zero speakers there's nothing to map — score directly.
    if not hyp_ids:
        counts = _score_with_mapping(ref_frames, hyp_frames, collar, {})
        return {}, counts

    # Extend the ref-id pool with "sentinel" indices for extra hyp speakers
    # so every hyp id gets a mapping; sentinels aren't in ref_frames so
    # they'll score as FA under the "not in ref" branch of _score_with_mapping.
    pool = ref_ids + list(range(n_ref_speakers, n_ref_speakers + max(0, n_hyp_speakers - n_ref_speakers)))
    for perm in itertools.permutations(pool, r=len(hyp_ids)):
        mapping = dict(zip(hyp_ids, perm))
        counts = _score_with_mapping(ref_frames, hyp_frames, collar, mapping)
        miss, fa, conf, _ = counts
        score = miss + fa + conf
        if best is None or score < best[0] + best[1] + best[2]:
            best = counts
            best_map = mapping

    assert best is not None
    return best_map, best


def compute_der(
    hypothesis_segments: list[tuple[float, float, str]],
    reference_segments:  list[tuple[float, float, str]],
    collar_ms:           int = 250,
) -> dict:
    """Compute DER + breakdown. See module docstring for output schema."""
    if collar_ms < 0:
        raise ValueError(f"collar_ms must be >= 0 (got {collar_ms})")

    # Frame the timeline out to the max end of either side.
    max_end = 0.0
    for start, dur, _ in reference_segments:
        max_end = max(max_end, start + dur)
    for start, dur, _ in hypothesis_segments:
        max_end = max(max_end, start + dur)
    n_frames = int(round(max_end * 1000 / FRAME_MS))

    # Speaker id -> dense index. Ref and hyp keep separate index spaces so
    # the mapping search is unambiguous.
    ref_idx: dict[str, int] = {}
    hyp_idx: dict[str, int] = {}
    ref_frames = _segments_to_frames(reference_segments,  n_frames, ref_idx)
    hyp_frames = _segments_to_frames(hypothesis_segments, n_frames, hyp_idx)
    collar     = _collar_mask(reference_segments, n_frames, collar_ms)

    if not reference_segments:
        # Undefined against an empty ref — mirror compute-wer.py's convention:
        # 0.0 if hyp is also empty, else 1.0 (all hyp is FA).
        return {
            "der":            0.0 if not hypothesis_segments else 1.0,
            "miss":           0.0,
            "false_alarm":    0.0 if not hypothesis_segments else 1.0,
            "confusion":      0.0,
            "collar_ms":      collar_ms,
            "n_ref_speakers": 0,
            "n_hyp_speakers": len(hyp_idx),
        }

    _, (miss, fa, conf, ref_speech) = _best_mapping(
        ref_frames, hyp_frames, collar, len(ref_idx), len(hyp_idx),
    )
    if ref_speech == 0:
        # Entire ref was inside the collar or empty after framing — no
        # scorable frames. Report 0.0 rather than divide-by-zero.
        der = 0.0
        miss_r = fa_r = conf_r = 0.0
    else:
        der    = (miss + fa + conf) / ref_speech
        miss_r = miss / ref_speech
        fa_r   = fa   / ref_speech
        conf_r = conf / ref_speech

    return {
        "der":            der,
        "miss":           miss_r,
        "false_alarm":    fa_r,
        "confusion":      conf_r,
        "collar_ms":      collar_ms,
        "n_ref_speakers": len(ref_idx),
        "n_hyp_speakers": len(hyp_idx),
    }


# ---- self-test --------------------------------------------------------------


def _self_test() -> int:
    """Runs known DER cases covering perfect, miss, FA, confusion, and the
    permutation-swap that the mapping search must resolve."""

    def rttm(rows):
        return [(s, d, spk) for s, d, spk in rows]

    def jsonl(rows):
        return [(s, d, spk) for s, d, spk in rows]

    # 4 s of speaker A, then 4 s of speaker B, no collar to keep the math
    # obvious. FRAME_MS=10 => 800 frames total, 400 per speaker.
    ref_ab = rttm([(0.0, 4.0, "A"), (4.0, 4.0, "B")])

    cases: list[tuple[str, list, list, int, float]] = [
        # (label, ref, hyp, collar_ms, expected_der)
        ("identical",          ref_ab, jsonl([(0.0, 4.0, "0"), (4.0, 4.0, "1")]), 0, 0.0),
        ("swapped ids",        ref_ab, jsonl([(0.0, 4.0, "1"), (4.0, 4.0, "0")]), 0, 0.0),
        ("all-miss",           ref_ab, jsonl([]),                                  0, 1.0),
        ("half-miss",          ref_ab, jsonl([(0.0, 4.0, "0")]),                   0, 0.5),
        ("all-fa (empty ref)", [],     jsonl([(0.0, 4.0, "0")]),                   0, 1.0),
        ("both empty",         [],     jsonl([]),                                  0, 0.0),
        ("full confusion",     ref_ab, jsonl([(0.0, 8.0, "0")]),                   0, 0.5),
        # Boundary-slop case: hyp puts the speaker-change 100 ms before the
        # true 4.0 s boundary. With collar_ms=0, frames [390-400] score as
        # speaker confusion (hyp says B but ref is still A) = 10 / 800 =
        # 0.0125. A 200 ms collar excludes the ±20 frame ring around the
        # 4.0 s boundary — the confused frames all fall inside that ring,
        # so DER goes back to 0.0. Guards the collar-off vs collar-on delta,
        # and would fail loudly if _collar_mask reverted to the max(1, ...)
        # bug that always excluded a 1-frame ring around every boundary.
        ("boundary slop, no collar", ref_ab,
            jsonl([(0.0, 3.9, "0"), (3.9, 4.1, "1")]), 0,   0.0125),
        ("boundary slop, collar 200", ref_ab,
            jsonl([(0.0, 3.9, "0"), (3.9, 4.1, "1")]), 200, 0.0),
        # Extra-hyp-speaker path (n_hyp > n_ref). Sortformer is a 4-speaker
        # model; if it emits a spurious 4th speaker over a stretch that
        # already has a ref speaker, that stretch scores as an FA on top of
        # the correct match. Ref = 4 s A; hyp = 4 s of speaker 0 + 2 s of
        # speaker 1 overlapping = 200 confusion-free FA frames / 400 ref
        # frames = DER 0.5.
        ("extra hyp speaker (FA sentinel)",
            [(0.0, 4.0, "A")],
            jsonl([(0.0, 4.0, "0"), (0.0, 2.0, "1")]), 0, 0.5),
    ]

    failed: list[str] = []
    for label, ref, hyp, collar_ms, want in cases:
        got = compute_der(hyp, ref, collar_ms=collar_ms)
        if abs(got["der"] - want) > 1e-6:
            failed.append(f"{label}: der {got['der']} != {want}")

    # Duration-shape hypothesis parsing: parakeet-cli today emits `end`, but
    # the JSONL contract accepts either — pin that both shapes score identically
    # so a future CLI switch to `duration` doesn't silently break DER.
    hyp_end = '{"speaker":0,"start":0.000,"end":4.000}\n{"speaker":1,"start":4.000,"end":8.000}\n'
    hyp_dur = '{"speaker":0,"start":0.000,"duration":4.000}\n{"speaker":1,"start":4.000,"duration":4.000}\n'
    d_end = compute_der(parse_hypothesis_jsonl(hyp_end), ref_ab)["der"]
    d_dur = compute_der(parse_hypothesis_jsonl(hyp_dur), ref_ab)["der"]
    if d_end != d_dur or d_end != 0.0:
        failed.append(f"duration vs end shape: end={d_end}, duration={d_dur} (both should be 0.0)")

    # Robustness against non-JSON stdout chatter (e.g. --verbose banner lines
    # leaking to stdout instead of stderr from parakeet-cli). Mixed input must
    # score the JSONL portion correctly and ignore the noise — otherwise a
    # single upstream stray-print would silently null every DER dispatch.
    mixed = (
        "parakeet: using Metal backend\n"                        # non-JSON banner
        "load=42.1ms audio=8.00s samples=128000@16000Hz\n"       # non-JSON verbose line
        '{"speaker":0,"start":0.000,"end":4.000}\n'
        '{"speaker":1,"start":4.000,"end":8.000}\n'
        "\n"                                                     # blank
        "[diarize] total=123.4ms RTF=0.015 segments=2\n"         # non-JSON trailer
    )
    d_mixed = compute_der(parse_hypothesis_jsonl(mixed), ref_ab)["der"]
    if d_mixed != 0.0:
        failed.append(f"mixed stdout robustness: der={d_mixed} (expected 0.0 — non-JSON lines must be skipped)")

    # md-eval miss/confusion split: with ref speakers > hyp speakers on a
    # frame, unmatched ref speakers split as miss=max(0,N_ref-N_sys),
    # confusion=residual. Ref 4 s of {A,B} overlap, hyp 4 s of speaker 0
    # (maps to A). N_ref=2, N_sys=1, N_correct=1 => miss=1, conf=0 per frame.
    # 400 frames of that = miss=400, conf=0, ref_speech=800 (per-speaker
    # counting). DER = 400/800 = 0.5; miss = 0.5; confusion = 0.0. Under the
    # old code the split was miss=0, conf=0.5 — same DER, wrong breakdown.
    overlap_ref = [(0.0, 4.0, "A"), (0.0, 4.0, "B")]
    got = compute_der([(0.0, 4.0, "0")], overlap_ref, collar_ms=0)
    if abs(got["der"] - 0.5) > 1e-6 or abs(got["miss"] - 0.5) > 1e-6 or abs(got["confusion"]) > 1e-6:
        failed.append(
            f"md-eval split (ref>hyp): der={got['der']} miss={got['miss']} conf={got['confusion']} "
            f"(expected der=0.5 miss=0.5 conf=0.0)"
        )

    # RTTM parser sanity: the shipped abcba.rttm should round-trip to 5
    # segments across 3 speakers.
    here = pathlib.Path(__file__).resolve().parent
    abcba = here / ".." / ".." / "parakeet" / "test" / "samples" / "abcba.rttm"
    if abcba.is_file():
        segs = parse_rttm(abcba.read_text(encoding="utf-8"))
        if len(segs) != 5:
            failed.append(f"parse_rttm(abcba.rttm): expected 5 segments, got {len(segs)}")
        speakers = {s for _, _, s in segs}
        if speakers != {"A", "B", "C"}:
            failed.append(f"parse_rttm(abcba.rttm): expected speakers A/B/C, got {sorted(speakers)}")

    if failed:
        for f in failed:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("compute-der.py: self-test ok")
    return 0


# ---- CLI --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute DER for one hyp-JSONL / ref-RTTM pair.")
    ap.add_argument("--hypothesis-jsonl", type=pathlib.Path,
                    help="Hypothesis segments as JSONL (parakeet-cli --emit jsonl output).")
    ap.add_argument("--reference",        type=pathlib.Path,
                    help="Reference RTTM file.")
    ap.add_argument("--collar-ms",        type=int, default=250,
                    help="Symmetric collar around reference boundaries in ms (default: 250).")
    ap.add_argument("--json-out",         type=pathlib.Path,
                    help="Write result JSON to this path in addition to stdout.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run built-in DER cases and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.reference:
        print("--reference is required", file=sys.stderr); return 2
    if not args.hypothesis_jsonl:
        print("--hypothesis-jsonl is required", file=sys.stderr); return 2

    ref_segments = parse_rttm(args.reference.read_text(encoding="utf-8"))
    # An empty hypothesis file is a legitimate "diarizer emitted nothing"
    # case and must score as pure miss (DER 1.0 against a non-empty ref),
    # not as an error — mirrors compute-wer.py's --hypothesis-file "" path.
    hyp_text = args.hypothesis_jsonl.read_text(encoding="utf-8") if args.hypothesis_jsonl.exists() else ""
    hyp_segments = parse_hypothesis_jsonl(hyp_text)

    result = compute_der(hyp_segments, ref_segments, collar_ms=args.collar_ms)
    result["reference"] = str(args.reference)

    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.json_out:
        args.json_out.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
