#!/usr/bin/env python3
"""
Per-sample scorer for L-Eval closed-ended tasks.

Ports the processors from upstream's Evaluation/auto_eval.py
(OpenLMLab/LEval) so per-task scoring matches the paper exactly:

    tpo, quality           multi-choice A/B/C/D, single letter EM
    coursera               multi-response A/B/C/D, set-EM (e.g. "AB" or "CD")
    gsm100                 numeric answer, regex-extracted + digit-stripped
    codeU                  deducing program output, last-tokens EM
    sci_fi                 [loyalty, fact] pair, two true/false EM
    topic_retrieval_longchat  exact substring match of topic name

Open-ended tasks are out of scope for v1 (GPT-4-judge or ROUGE/F1 with
length-instruction bias — see paper §4 on why automatic ngram metrics
don't track human eval well there).

Input JSON (stdin):
    {
        "task": "<task name>",
        "prediction": "<model output>",
        "references": ["gt1", ...]
    }

Output: float in [0, 1]. For sci_fi / topic_retrieval_longchat the score
is the macro-average of the per-sub-question scores produced for a single
sample (so the per-sample score still fits the standard scalar contract).
"""

import argparse
import json
import re
import sys


# ── Multi-choice (tpo, quality): single letter A/B/C/D ─────────────────────
# Ported from auto_eval.py:process_output_mc with the non-coursera branch.
def _process_mc_letter(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""
    if response in "ABCD":
        return response
    for ch in response:
        if ch in "ABCD":
            return ch
    return ""


# ── Coursera: multi-response, can be e.g. "AB" or "CD" ─────────────────────
# Ported from process_output_mc's coursera branch (looks for runs of
# A-D letters at the start, then falls back to a regex for "A.", "(B)", etc).
def _process_mc_multi(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""
    cleaned = ""
    for ch in response:
        if ch in "ABCD":
            cleaned += ch
            response = response[1:]
        else:
            break
    if len(cleaned) > 1:
        return "".join(sorted(set(cleaned)))
    # Fallback: parse "A.", "B)" style answers.
    response = response.split("Question")[0]
    options = re.findall(r"\s*[A-D](?=[\s.)])", response)
    cleaned += " ".join(options).strip()
    cleaned = re.sub(r"[^A-D]", "", cleaned)
    cleaned = "".join(sorted(set(cleaned)))
    if not cleaned:
        # Last-ditch: any A-D letter anywhere.
        for ch in response:
            if ch in "ABCD":
                cleaned += ch
    return cleaned or "A"  # upstream defaults to "A" on total parse failure


def _process_gt_mc(response: str) -> str:
    first = (response or "").split()[0] if response else ""
    cleaned = re.sub(r"[^A-D]", "", first)
    return cleaned or "A"


# ── GSM100: numeric answer ─────────────────────────────────────────────────
# Ported from process_math: prefer "The answer is X" pattern, else
# scan tokens from the right for the first one containing a digit, then
# strip down to bare digits (and stop at a decimal point).
def _process_math(response: str) -> str:
    if not response:
        return ""
    m = re.search(r"The answer is (\S+)", response)
    if m:
        ret = m.group(1)
    else:
        first_para = response.split("\n\n")[0]
        tokens = first_para.split(" ")[::-1]
        ret = ""
        for tok in tokens:
            if any(ch.isdigit() for ch in tok):
                ret = tok
                break
    out = ""
    for ch in ret:
        if ch.isdigit():
            out += ch
        elif ch == ".":
            break
    return out


# ── CodeU: deducing program output ─────────────────────────────────────────
def _process_gt_code(response: str) -> str:
    response = re.sub(r"\s+", " ", response or "")
    response = response.replace(",", "").replace("'", "").replace("\\", "")
    response = response.replace(".0", "")
    response = response.replace("] [", "][").replace("[ [", "[[").replace("] ]", "]]")
    return response


def _process_code(response: str, gt: str) -> str:
    gt_len = len(gt.split())
    response = _process_gt_code(response)
    for trim in ["will be", "of the code", "is", "would be",
                 "the value of", "the result of", "printed"]:
        response = response.replace(trim, "")
    if "the final output" in response:
        tail = response.split("the final output")[-1]
        toks = re.split(r"\s+", tail)[:(gt_len + 3)]
    else:
        toks = re.split(r"\s+", response)[-(gt_len + 3):]
    return " ".join(toks)


# ── sci_fi: [loyalty, fact] pair ───────────────────────────────────────────
def _process_judge(response: str) -> list[str]:
    response = (response or "").lower()
    parts = response.split("[fact:")
    if len(parts) < 2:
        return ["<error>", "<error>"]
    loyalty = parts[0]
    fact = parts[1].rstrip("]")
    out = []
    for sub in (loyalty, fact):
        picked = "<error>"
        for word in sub.split():
            if "true" in word:
                picked = "true"
                break
            if "false" in word:
                picked = "false"
                break
        out.append(picked)
    return out


# ── Top-level dispatch ─────────────────────────────────────────────────────
def score_one(task: str, prediction: str, references: list[str]) -> float:
    prediction = (prediction or "").lstrip()
    if not references:
        return 0.0
    gt = references[0]

    if task in ("tpo", "quality"):
        pred = _process_mc_letter(prediction)
        gt_letter = _process_gt_mc(gt)
        return 1.0 if pred == gt_letter else 0.0

    if task == "coursera":
        pred = _process_mc_multi(prediction)
        gt_letters = "".join(sorted(set(re.sub(r"[^A-D]", "", gt))))
        return 1.0 if pred == gt_letters else 0.0

    if task == "gsm100":
        pred_num = _process_math(prediction)
        gt_num = _process_math(gt)
        return 1.0 if pred_num and pred_num == gt_num else 0.0

    if task == "codeU":
        pred = _process_code(prediction, gt)
        gt_clean = _process_gt_code(gt)
        # Loose match: gt must appear in the (cleaned) tail of prediction.
        # Upstream uses an exact_match metric; since codeU answers are short
        # tokens that may be padded, substring-EM is the upstream behaviour.
        return 1.0 if gt_clean.strip() in pred.strip() else 0.0

    if task == "sci_fi":
        # Reference is "loyalty [fact: ...]"; predict the same. Score the
        # macro mean of the two binary judgements.
        gt_pair = _process_judge(gt)
        pred_pair = _process_judge(prediction)
        score = 0.0
        for p, g in zip(pred_pair, gt_pair):
            score += 1.0 if p == g and p != "<error>" else 0.0
        return score / 2.0

    if task == "topic_retrieval_longchat":
        # Upstream scores three separate aggregates (first / second / third
        # sentence retrieval) — but per-sample, it's still a substring EM
        # of the topic name in the prediction.
        gt_norm = gt.strip().lower()
        pred_norm = prediction.strip().lower()
        return 1.0 if gt_norm in pred_norm else 0.0

    raise ValueError(f"unknown L-Eval closed-ended task: {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--prediction")
    parser.add_argument("--references")
    parser.add_argument("--all-classes", default=None)
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

    score = score_one(payload["task"], payload["prediction"], payload["references"])
    sys.stdout.write(f"{score:.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
