#!/usr/bin/env python3
"""
Per-sample scorer for NIAH (Needle In A Haystack).

NIAH is a binary recall benchmark: a single sentence (the "needle") is
inserted at a known depth into a long context (the "haystack") of unrelated
text, and the model is asked a question whose answer is the needle. The
score is 1 if the answer is recovered in the model's response, 0 otherwise.

We use case-insensitive substring matching on an "answer key" string — the
canonical Greg-Kamradt-style scorer. Multiple acceptable keys per needle
(e.g., paraphrase tolerance) are supported via a list.

The model-as-judge approach (the upstream `needlehaystack` PyPI package's
default for OpenAI/Anthropic backends) is deliberately not used here: it
adds an OpenAI API dependency, costs money per call, and introduces
non-determinism that conflicts with our cache-comparison goal. Substring
match is what RULER's `niah_single_*` family also uses and is the standard
choice for offline / batch NIAH harnesses.

Input JSON (stdin):
    {
        "task":       "niah",
        "prediction": "<model output>",
        "references": ["dolores park", "sandwich and sit"]
    }

Output: 1.000000 if any reference appears as a case-insensitive substring
of the prediction, else 0.000000.
"""

import argparse
import json
import sys


def score_one(prediction: str, references) -> float:
    if not references:
        return 0.0
    p = (prediction or "").lower()
    for ref in references:
        if ref is None:
            continue
        if str(ref).lower() in p:
            return 1.0
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--task")           # unused but kept for CLI parity
    parser.add_argument("--prediction")
    parser.add_argument("--references")
    parser.add_argument("--all-classes")
    args = parser.parse_args()

    if args.prediction is not None:
        if args.references is None:
            sys.stderr.write("ERROR: --references required with --prediction\n")
            return 2
        payload = {
            "prediction": args.prediction,
            "references": json.loads(args.references),
        }
    else:
        data = sys.stdin.read()
        if not data.strip():
            sys.stderr.write("ERROR: no JSON on stdin\n")
            return 2
        payload = json.loads(data)

    score = score_one(payload["prediction"], payload["references"])
    sys.stdout.write(f"{score:.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
