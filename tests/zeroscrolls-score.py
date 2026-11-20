#!/usr/bin/env python3
"""
Per-sample scorer for ZeroSCROLLS tasks.

The upstream ZeroSCROLLS repo (tau-nlp/zero_scrolls) does not ship a local
scoring script — the official protocol is "produce predictions JSON, submit
to the leaderboard server". For offline KV-cache quality evaluation we need
to compute metrics locally, so this script implements the per-task scoring
rules defined in the paper (Shaham et al., 2023, "ZeroSCROLLS").

Task → metric mapping (v1 subset; space_digest and book_sum_sort are deferred):

    gov_report, summ_screen_fd, qmsum, squality  →  ROUGE-L F1
    qasper, narrative_qa, musique                →  token-level F1 with
                                                    normalize_answer
                                                    (same as LongBench/QA)
    quality                                      →  exact-match on the
                                                    extracted A/B/C/D letter

For tasks with multiple references per input (NarrativeQA, Qasper),
the per-sample score is max(metric(pred, gt) for gt in references) — the
standard "best-of-references" convention.

Input JSON schema (stdin, one object per invocation):
    {
        "task": "<task name>",
        "prediction": "<model output>",
        "references": ["gt1", "gt2", ...],
        "all_classes": null   (unused, kept for parity with longbench-score)
    }

Output: float in [0, 1] on stdout. The orchestrator multiplies by 100 for
percent display.
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from typing import Optional


# ── F1 (token level), copied from upstream LongBench/metrics.py ────────────
# We don't import LongBench here because LongBench's metrics.py
# unconditionally imports jieba, and ZeroSCROLLS doesn't need Chinese
# tokenisation. Re-implementing the same logic keeps zeroscrolls a
# self-contained dependency on just `rouge`.


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _token_f1(pred_tokens, gt_tokens) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1(prediction: str, ground_truth: str, **_) -> float:
    p = _normalize_answer(prediction).split()
    g = _normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0
    return _token_f1(p, g)


# ── ROUGE-L F1, via the same `rouge` package upstream LongBench uses ───────
def rouge_l(prediction: str, ground_truth: str, **_) -> float:
    # Local import so the script still imports / --help works without `rouge`.
    from rouge import Rouge  # ty: ignore[unresolved-import]

    rouge = Rouge()
    # The `rouge` package crashes on empty strings; pad with a sentinel so
    # the metric returns 0 instead of throwing. Upstream LongBench does the
    # same try/except in metrics.py:rouge_score.
    try:
        scores = rouge.get_scores([prediction or "_"], [ground_truth or "_"], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])


# ── QuALITY exact-match on A/B/C/D ─────────────────────────────────────────
# QuALITY prompts the model with four choices labelled (A)/(B)/(C)/(D); the
# reference output is a single letter. Models often produce extra text
# ("The answer is (C). Because..."), so we extract the first standalone
# A/B/C/D letter from the prediction and compare. This mirrors the
# convention used by upstream evaluators that report QuALITY accuracy
# (e.g., the MMLU-style first-letter extractor).
_QUALITY_PATTERNS = [
    re.compile(r"\b([ABCD])\b"),
    re.compile(r"\(([ABCD])\)"),
    re.compile(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", re.IGNORECASE),
]


def quality_em(prediction: str, ground_truth: str, **_) -> float:
    pred_letter: Optional[str] = None
    # Try the most specific pattern first (handles "Answer: C") then fall
    # back to the looser standalone-letter match.
    for pat in reversed(_QUALITY_PATTERNS):
        m = pat.search(prediction)
        if m:
            pred_letter = m.group(1).upper()
            break
    if pred_letter is None:
        return 0.0
    gt_letter = ground_truth.strip().upper()
    if gt_letter not in {"A", "B", "C", "D"}:
        # ground truth itself might be "(C)" or "C." — strip down to bare letter
        m = re.search(r"[ABCD]", gt_letter)
        gt_letter = m.group(0) if m else gt_letter
    return 1.0 if pred_letter == gt_letter else 0.0


DATASET2METRIC = {
    "gov_report":     rouge_l,
    "summ_screen_fd": rouge_l,
    "qmsum":          rouge_l,
    "squality":       rouge_l,
    "qasper":         qa_f1,
    "narrative_qa":   qa_f1,
    "musique":        qa_f1,
    "quality":        quality_em,
}


def score_one(task: str, prediction: str, references, all_classes=None) -> float:
    if task not in DATASET2METRIC:
        raise ValueError(f"unknown ZeroSCROLLS task: {task!r}")

    # Same llama-completion / llama-server leading-whitespace quirk as
    # LongBench: predictions tend to start with " \n<answer>" rather than
    # "\n<answer>", so a naive lstrip("\n") leaves a space-only first line.
    # Whitespace-only lstrip restores the intended "first non-garbage" line.
    prediction = prediction.lstrip()

    # For QuALITY, the model often answers in the first line; for the
    # remaining tasks (free-form generation), we keep the whole prediction.
    if task == "quality":
        prediction = prediction.split("\n")[0]

    metric = DATASET2METRIC[task]
    best = 0.0
    for gt in references:
        if gt is None:
            continue
        s = metric(prediction, gt)
        if s > best:
            best = s
    return float(best)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin (default).")
    parser.add_argument("--task")
    parser.add_argument("--prediction")
    parser.add_argument("--references", help="JSON list of reference strings.")
    parser.add_argument("--all-classes", default=None, help="Ignored (kept for CLI parity).")
    args = parser.parse_args()

    if args.prediction is not None:
        if args.task is None or args.references is None:
            sys.stderr.write("ERROR: --task and --references required with --prediction\n")
            return 2
        payload = {
            "task": args.task,
            "prediction": args.prediction,
            "references": json.loads(args.references),
        }
    else:
        data = sys.stdin.read()
        if not data.strip():
            sys.stderr.write("ERROR: no JSON on stdin\n")
            return 2
        payload = json.loads(data)

    score = score_one(
        payload["task"],
        payload["prediction"],
        payload["references"],
        payload.get("all_classes"),
    )
    sys.stdout.write(f"{score:.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
