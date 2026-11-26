#!/usr/bin/env python3
# Aggregate the per-(family, runner) result.json files produced by
# scripts/benchmarks/run-family.sh into a single markdown table.
#
# Called from .github/workflows/speech-benchmark-desktop.yml's summarize job:
#
#   summarize.py --results-dir all-results \
#                --out speech-benchmark-findings.md
#
# Also writes to $GITHUB_STEP_SUMMARY when the env var is set so the table
# is visible on the workflow run page without downloading the artifact.
#
# Column shape mirrors audiogen/benchmarks/comparison/reports/*.md
# (the closest in-repo precedent for this project's benchmark tables):
# The shared columns cover every family. Native engines can additionally report
# encoder backend/timing; the Parakeet Darwin comparison adds its forced-ggml
# baseline and baseline/Core ML speedups.

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Any


def load_results(root: pathlib.Path) -> list[dict[str, Any]]:
    """Walk `root` for every result.json; return a list sorted for a stable table."""
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            with path.open() as f:
                results.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            # A missing / malformed file is a bench-job crash — surface it as
            # a row rather than silently dropping data.
            results.append({
                "family": path.parent.name,
                "model": "?",
                "runner": "?",
                "os": "?",
                "backend": "?",
                "wall_ms_median": None,
                "rtf_median": None,
                "peak_rss_mib": None,
                "runs": 0,
                "status": "artifact-corrupt",
                "notes": f"could not parse {path}: {e}",
            })
    return sorted(
        results,
        key=lambda r: (r.get("family", ""), r.get("runner", ""), r.get("model", "")),
    )


def fmt_ms(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rtf(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rss(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "—"


def fmt_speedup(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}×"
    except (TypeError, ValueError):
        return "—"


def fmt_correctness(kind: Any, wer: Any, der: Any, f1: Any, tts: Any = None) -> str:
    """Kind-prefixed correctness cell.

    Only one score is populated per row (the family declared exactly one
    `correctness.kind`). Renders as "WER 0.00%" / "DER 8.2%" / "F1 98.70%"
    / "—" so a reader sees at a glance which metric the row is scored on.
    Blank when the family didn't declare a `correctness` block or the
    scoring step was skipped — distinct from a scored 0.00% run.
    """
    if kind == "tts_intelligibility":
        if not isinstance(tts, dict) or tts.get("status") != "ok":
            status = tts.get("status", "unavailable") if isinstance(tts, dict) else "unavailable"
            return f"TTS WER {status}"
        value = tts.get("wer")
        return f"TTS WER {float(value) * 100:.2f}%" if isinstance(value, (float, int)) else "TTS WER unavailable"
    if kind == "wer":
        v = wer
    elif kind == "der":
        v = der
    elif kind == "f1":
        v = f1
    else:
        return "—"
    if v is None:
        return "—"
    try:
        return f"{kind.upper()} {float(v) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_audio_quality(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "—"
    cells = []
    for kind in ("sisdr", "stoi"):
        metric = metrics.get(kind)
        if not isinstance(metric, dict):
            continue
        label = {"sisdr": "SI-SDR", "stoi": "STOI"}[kind]
        value = metric.get("value")
        if metric.get("status") != "ok" or value is None:
            cells.append(f"{label} — ({metric.get('status', 'unavailable')})")
            continue
        value = float(value)
        display = f"{value:.2f} dB" if kind == "sisdr" else f"{value * 100:.2f}%"
        cells.append(f"{label} {display}")
    return "; ".join(cells) or "—"


def fmt_music_alignment(metric: Any) -> str:
    if not isinstance(metric, dict):
        return "—"
    coverage = metric.get("coverage") or {}
    count = f"{coverage.get('scored', 0)}/{coverage.get('expected', '?')}"
    value = metric.get("score")
    status = str(metric.get("status", "unavailable")).replace("|", "\\|")
    if status in ("ok", "partial") and isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and -1 <= value <= 1:
        return f"CLAP {value:.4f} ({status}; {count})"
    return f"CLAP — ({status}; {count})"


def fmt_coreml_verified(v: Any) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "—"


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Single fixed-column table. Empty result set still produces a valid table."""
    header = (
        "| Family | Model | Runner | OS | Decoder | Encoder | Baseline encoder | Core ML verified "
        "| Encoder ms | Baseline encoder ms | Encoder speedup | End-to-end ms "
        "| Min | Max | Baseline end-to-end ms | E2E speedup | RTF | Correctness "
        "| Peak RSS MiB | Runs | Status | Notes |\n"
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            "| {family} | {model} | {runner} | {os} | {backend} | {encoder_backend} | {baseline_encoder_backend} "
            "| {coreml_verified} | {encoder_ms} | {baseline_encoder_ms} | {encoder_speedup} "
            "| {median} | {min} | {max} | {baseline_inference_ms} | {inference_speedup} | {rtf} | {correctness} | {rss} "
            "| {runs} | {status} | {notes} |".format(
                family=r.get("family", "?"),
                model=r.get("model", "?"),
                runner=r.get("runner", "?"),
                os=r.get("os", "?"),
                backend=r.get("backend", "?") or "unknown",
                encoder_backend=r.get("encoder_backend", "—") or "unknown",
                baseline_encoder_backend=r.get("baseline_encoder_backend") or "—",
                coreml_verified=fmt_coreml_verified(r.get("encoder_coreml_all_runs")),
                encoder_ms=fmt_ms(r.get("encoder_ms_median")),
                baseline_encoder_ms=fmt_ms(r.get("baseline_encoder_ms_median")),
                encoder_speedup=fmt_speedup(r.get("encoder_speedup")),
                median=fmt_ms(r.get("wall_ms_median")),
                min=fmt_ms(r.get("wall_ms_min")),
                max=fmt_ms(r.get("wall_ms_max")),
                baseline_inference_ms=fmt_ms(r.get("baseline_inference_ms_median")),
                inference_speedup=fmt_speedup(r.get("inference_speedup")),
                rtf=fmt_rtf(r.get("rtf_median")),
                correctness=(fmt_music_alignment(r.get("music_alignment")) if r.get("music_alignment") is not None else fmt_audio_quality(r.get("audio_quality")) if r.get("correctness_kind") == "audio" else fmt_correctness(r.get("correctness_kind"), r.get("wer_median"), r.get("der_median"), r.get("f1_median"), r.get("tts_intelligibility"))),
                rss=fmt_rss(r.get("peak_rss_mib")),
                runs=r.get("runs", 0),
                status=r.get("status", "?"),
                notes=(r.get("notes") or "").replace("|", "\\|"),
            )
        )
    if not rows:
        rows.append("| _no results collected_ | | | | | | | | | | | | | | | | | | | | | |")
    return header + "\n".join(rows) + "\n"


def render_legend() -> str:
    """Small key describing what each metric means, so a reader landing on
    the workflow run page without context can interpret the numbers."""
    return (
        "\n**Metric key**\n"
        "- **Median wall ms** — end-to-end wall-clock median across `--runs` timed "
        "iterations (warmup runs discarded). Includes model load + graph build for "
        "time-wrapped families; native `*-bench` binaries report their own totals.\n"
        "- **Median RTF** — real-time factor; wall / audio-seconds. `< 1.0` means "
        "faster than real-time. Blank when the family's output length isn't fixed "
        "(text-driven TTS, chatterbox, audio8).\n"
        "- **TTS WER** — WER of pinned Whisper Tiny CPU transcription against the synthesis prompt; lower is better. "
        "Scored once on the final measured output, not a median. Includes reference-ASR errors; no quality pass threshold. "
        "ASR time is excluded from synthesis timing. Missing/failed scoring is explicitly labeled.\n"
        "- **Correctness** — accuracy metric declared by the family's `correctness.kind`. "
        "Kind-prefixed. `WER 0.00%` is Word Error Rate of the bench transcript against a "
        "reference text, English-normalized (case-folded, punctuation stripped) — used by "
        "ASR families (parakeet, whisper). `DER 8.2%` is Diarization Error Rate — miss + "
        "false-alarm + speaker-confusion time over reference-speech time, with a 250 ms "
        "collar around segment boundaries — used by diarization families (sortformer). "
        "`F1 98.70%` is frame-based Precision/Recall/F1 of predicted speech vs a reference "
        "of speech-segment ranges, with a 100 ms boundary-tolerance collar — used by voice-"
        "activity detection (vad). Healthy build: `WER 0.00%` / low single-digit `DER` / "
        "`F1` close to 100%; worse numbers mean the model regressed or the reference "
        "drifted. Blank when the family didn't declare a `correctness` block or the scoring "
        "step was skipped — see the row's notes.\n"
        "- **Audio quality** — SI-SDR is in dB; STOI is a percentage. "
        "Higher is better. Scores use the final timed output, not a median. "
        "Unavailable metrics are labeled; details and reference/output WAVs are in artifacts.\n"
        "- **CLAP** — non-gating text/music cosine similarity, higher means stronger prompt alignment; not overall music quality. Equal-weight mean across timed runs with scored/expected coverage; failed scores remain null. Opt-in diagnostic generation writes WAVs within timing and uses a longer workload; compare only matching diagnostic configurations. CLAP preparation/scoring are excluded from generation timing.\n"
        "- **Peak RSS MiB** — maximum resident set size across all timed runs, via "
        "`/usr/bin/time` (GNU `-v` on Linux, BSD `-l` on macOS).\n"
        "- **Backend** — from the bench binary's JSON when it reports one, else the "
        "engine's `using <NAME> backend` / `backend: <NAME>` log lines, else ggml's "
        "own GPU compute-init logs; a green run with no GPU-engagement evidence is "
        "reported as `CPU`. `unknown` appears only on failed runs, whose logs may be "
        "truncated mid-init.\n"
        "- **Core ML comparison** — Core-ML-enabled Parakeet Darwin rows run twice on the "
        "same Apple Silicon runner. `Core ML verified=yes` means every measured "
        "encoder invocation used the sidecar; the baseline is forced through ggml "
        "with `PARAKEET_COREML_DISABLE=1`. Speedup is baseline divided by Core ML.\n"
        "- **Status** — `ok` / `not-in-registry` (whisper `small`) / `missing-model` "
        "(minimax — no S3 path yet) / `build-failed` / `run-failed` / `fetch-failed` / "
        "`infrastructure-failed` (a pre-bench workflow step — AWS OIDC, ggml build, "
        "cmake configure — did not complete).\n"
        "- **Linux vs macOS ggml build**: hosted Linux is a portable ggml build "
        "(no `-march=native`, no GPU backend) — its numbers are a **floor**, "
        "not a runner-optimal peak. Self-hosted macOS gets Metal + native "
        "codegen. Comparing the two columns as apples-to-apples is misleading; "
        "compare each family's Linux row across dispatches for CPU perf, "
        "the macOS row for Metal perf.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    # `expected_count` is emitted by the workflow's `plan` job. When set, we
    # also fail if fewer rows than planned made it to the summary — that's
    # the "cell died before writing its pre-flight seed" case, which the
    # `not ok_rows` gate below can't see because the missing row contributes
    # zero to `results`.
    #
    # The workflow calls this with `--expected-count "$EXPECTED_COUNT"`; if
    # the plan job failed before setting the output, EXPECTED_COUNT expands
    # to an empty string. Accept "" as "unset" so summarize can still render
    # whatever artifacts it does find (a plan-failed dispatch has already
    # turned the workflow red — summarize failing on top would just be
    # noise).
    def _opt_int(v: str) -> int | None:
        return int(v) if v else None
    ap.add_argument("--expected-count", type=_opt_int, default=None)
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1

    results = load_results(args.results_dir)

    title = "# Speech benchmark findings\n\n"
    subtitle = (
        f"Rows collected: **{len(results)}**. "
        "Wall-clock median across `--runs` timed iterations "
        "(warmup runs discarded).\n\n"
    )
    body = title + subtitle + render_markdown(results) + render_legend()

    args.out.write_text(body)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(body)
    print(f"wrote {args.out} ({len(results)} rows)")

    # Loudness gate. `bench` runs with `continue-on-error: true` so one
    # family failure doesn't skip the summary — but if EVERY cell failed
    # (misconfigured S3 role, wrong binary paths, ggml build broken) the
    # workflow would otherwise finish green with a table full of red
    # statuses that nobody looks at. Failing the summarize job here makes
    # the whole dispatch red so the reader has to look at the summary.
    #
    # Two independent trip conditions:
    #   (a) rows are collected but none succeeded — every cell wrote a
    #       failure result.
    #   (b) fewer rows than planned — a cell died before its pre-flight
    #       seed could write result.json (workflow env / harness bug).
    #       Needs --expected-count from the workflow's plan job.
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if results and not ok_rows:
        print(
            f"FAIL: {len(results)} rows collected but 0 have status=ok. "
            "Every cell failed — see the row-level notes.",
            file=sys.stderr,
        )
        return 1
    if args.expected_count is not None and len(results) < args.expected_count:
        print(
            f"FAIL: expected {args.expected_count} rows from the plan "
            f"but only {len(results)} arrived. "
            f"{args.expected_count - len(results)} cell(s) died before "
            "writing a result.json.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
