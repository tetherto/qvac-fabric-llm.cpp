#!/usr/bin/env python3
"""Score retained timed music runs after performance capture; never gate generation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import statistics


def failure(reason, status="scorer-error"):
    return {"status": status, "score": None, "reason": reason}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--backend", default="unknown")
    ap.add_argument("--runs", required=True, type=int)
    ap.add_argument("--python", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    results = []
    for i in range(1, args.runs + 1):
        run_dir = args.artifacts / f"run-{i}"
        score_file = run_dir / "score.json"
        try:
            generation = json.loads((run_dir / "generation.json").read_text())
            generation.setdefault('observed_backend', args.backend)
            (run_dir / "generation.json").write_text(json.dumps(generation, allow_nan=False, indent=2) + "\n")
            if not (run_dir / "audio.wav").is_file():
                result = failure("generation did not produce audio.wav", "invalid-audio")
            elif not args.model_dir:
                result = failure("CLAP model preparation unavailable; see preparation.json", "model-missing")
            else:
                score_file.unlink(missing_ok=True)
                with (run_dir / "scorer.log").open("w") as log:
                    completed = subprocess.run([
                        args.python, str(Path(__file__).with_name("compute-music-alignment.py")),
                        "--wav", str(run_dir / "audio.wav"), "--caption", generation["caption"],
                        "--model-dir", args.model_dir, "--model-manifest", args.manifest,
                        "--json-out", str(score_file)], stdout=log, stderr=subprocess.STDOUT,
                        timeout=args.timeout, check=False)
                result = json.loads(score_file.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON value")))
                # Valid JSON exponents such as 1e999 also become infinity;
                # validate the whole nested result before retaining it.
                json.dumps(result, allow_nan=False)
                value = result.get("score") if isinstance(result, dict) else None
                if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                    result = failure("scorer returned malformed result")
                elif result["status"] == "ok" and (completed.returncode != 0 or isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not -1 <= value <= 1):
                    result = failure("scorer returned invalid score or failed exit status")
                elif result["status"] != "ok":
                    if result["status"] not in {"invalid-audio", "invalid-caption", "model-missing", "dependency-missing", "scorer-error", "timeout"}:
                        result = failure("scorer returned unknown status: " + result["status"])
                    result["score"] = None
        except FileNotFoundError as exc:
            result = failure(f"missing scorer dependency or artifact: {exc}", "dependency-missing" if not Path(args.python).is_file() else "scorer-error")
        except subprocess.TimeoutExpired:
            result = failure("CLAP scorer exceeded timeout", "timeout")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = failure(f"CLAP scorer unavailable or malformed result: {exc}")
        run_dir.mkdir(parents=True, exist_ok=True)
        score_file.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
        results.append({"run": i, "artifact_dir": str(run_dir), **result})
    values = [r["score"] for r in results if r["status"] == "ok"]
    report = {
        "status": "ok" if len(values) == args.runs and values else "partial" if values else "unavailable",
        "score": statistics.mean(values) if values else None,
        "aggregation": "equal-weight mean across timed runs",
        "reason": None if len(values) == args.runs and values else "one or more timed outputs could not be scored; see runs",
        "coverage": {"scored": len(values), "expected": args.runs}, "runs": results,
    }
    args.out.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
