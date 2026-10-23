#!/usr/bin/env python3
"""
Per-sample scorer for LongBench-E tasks.

Reads a prediction + reference list from stdin (JSON) and prints the score as
a single float on stdout. Mirrors the rule used by upstream LongBench's
eval.py:scorer:

    score = max(metric(pred, gt, all_classes=...) for gt in answers)

The dispatch table comes directly from upstream's eval.py:dataset2metric and
the metric implementations are imported as-is from LongBench/metrics.py.

Input JSON schema (one object on a single line):
    {
        "task": "<task name>",
        "prediction": "<model output>",
        "references": ["gt1", "gt2", ...],
        "all_classes": ["cls1", ...] | null
    }

Output: float, e.g. "0.8765" — already in [0, 1], NOT scaled to percent.

Invoked once per sample by tests/longbench-bench.sh. The runner aggregates the
per-sample scores into mean/stdev and scales to percent in the report.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(SCRIPT_DIR, "longbench", "LongBench")
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)

from metrics import (  # noqa: E402  # ty: ignore[unresolved-import]
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)

# Subset of upstream's dataset2metric, restricted to LongBench-E English tasks.
# Chinese tasks (qa_f1_zh_score, rouge_zh_score, retrieval_zh_score) are
# excluded since the blog/paper benchmarks Llama-3.1-8B-Instruct, which is
# English-only, and we drop them at data-prep time.
DATASET2METRIC = {
    "qasper":               qa_f1_score,
    "multifieldqa_en":      qa_f1_score,
    "hotpotqa":             qa_f1_score,
    "2wikimqa":             qa_f1_score,
    "gov_report":           rouge_score,
    "multi_news":           rouge_score,
    "trec":                 classification_score,
    "triviaqa":             qa_f1_score,
    "samsum":               rouge_score,
    "passage_count":        count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc":                  code_sim_score,
    "repobench-p":          code_sim_score,
}

# Upstream applies this trim only for these tasks (eval.py:scorer line ~50).
# Replicated verbatim so per-task scores are byte-identical.
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum"}


def score_one(task: str, prediction: str, references, all_classes) -> float:
    if task not in DATASET2METRIC:
        raise ValueError(f"unknown LongBench-E task: {task!r}")

    # Upstream's eval.py does `prediction.lstrip("\n").split("\n")[0]` and
    # code_sim_score does the same internally. That works for HF-transformers
    # output which always starts with "\n". llama-server's /completion
    # output for these tasks tends to start with " \n<answer>" (leading
    # space token from BPE detokenisation of the prompt's trailing
    # "Answer:" or similar).
    # The narrow lstrip("\n") leaves the leading space in place, so
    # split("\n")[0] returns " " — and the actual answer on line 2 gets
    # silently dropped, scoring 0 even when the model answered correctly.
    # A whitespace-only lstrip restores upstream's intent ("strip leading
    # garbage, then take the first line / first non-comment line") without
    # changing scores on predictions that already start with "\n".
    prediction = prediction.lstrip()

    if task in FIRST_LINE_TASKS:
        prediction = prediction.split("\n")[0]

    metric = DATASET2METRIC[task]
    best = 0.0
    for gt in references:
        s = metric(prediction, gt, all_classes=all_classes)
        if s > best:
            best = s
    return float(best)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stdin", action="store_true",
        help="Read a single JSON object from stdin (default).",
    )
    parser.add_argument(
        "--task",
        help="Task name (overrides JSON's 'task' field; required if --prediction is given).",
    )
    parser.add_argument(
        "--prediction",
        help="Model prediction text (alternative to --stdin).",
    )
    parser.add_argument(
        "--references",
        help="JSON list of reference strings (alternative to --stdin).",
    )
    parser.add_argument(
        "--all-classes",
        help="JSON list of class names (or null) — only used by classification_score.",
    )
    args = parser.parse_args()

    if args.prediction is not None:
        if args.task is None or args.references is None:
            sys.stderr.write("ERROR: --task and --references required with --prediction\n")
            return 2
        payload = {
            "task": args.task,
            "prediction": args.prediction,
            "references": json.loads(args.references),
            "all_classes": json.loads(args.all_classes) if args.all_classes else None,
        }
    else:
        data = sys.stdin.read()
        if not data.strip():
            sys.stderr.write("ERROR: no JSON on stdin\n")
            return 2
        payload = json.loads(data)

    score = score_one(
        str(payload["task"]),
        str(payload["prediction"]),
        payload["references"],
        payload.get("all_classes"),
    )
    sys.stdout.write(f"{score:.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
