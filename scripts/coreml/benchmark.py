#!/usr/bin/env python3
"""Run correctness and cold/warm comparisons in isolated processes; never count fallback as ANE."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def validate_run(proc, mode, repeats, require_ane):
    measurements = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    selected = proc.stderr.count("route=ANE-plan ")
    if proc.returncode or len(measurements) != 1:
        raise RuntimeError(f"{mode} failed or returned no single measurement: {proc.stderr}")
    for key in ("cold_ms", "warm_ms", "nmse", "max_abs"):
        value = measurements[0].get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"invalid measurement: {key}")
    if measurements[0]["nmse"] >= 5e-4:
        raise RuntimeError("correctness threshold exceeded")
    expected = 2*(repeats + 1)
    if mode != "coreml" and selected:
        raise RuntimeError(f"unexpected ANE selection in {mode}")
    if require_ane:
        if mode == "coreml" and selected != expected:
            raise RuntimeError(f"expected {expected} CoreML predictions, got {selected}; partial fallback is not a pass")
        reasons = {"injected-failure": "injected-failure", "prediction-failure": "injected-prediction-failure",
                   "missing-model": "model-load-or-layout"}
        if mode in reasons and proc.stderr.count(f"reason={reasons[mode]}\n") != expected:
            raise RuntimeError(f"{mode}: expected fallback was not exercised on every prediction")
    return measurements, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--shape", default="512x64x512")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("coreml-benchmark.json"))
    parser.add_argument("--require-ane", action="store_true")
    parser.add_argument("--timeout", type=int, default=300, help="maximum seconds per child process")
    args = parser.parse_args()
    m, n, k = map(int, args.shape.lower().split("x"))
    if not (0 < m <= 8192 and 0 < n <= 2048 and 0 < k <= 8192 and k % 32 == 0 and 0 < args.repeats <= 10000 and args.timeout > 0):
        parser.error("invalid shape, repeats or timeout")
    records = []
    successful_ane = 0
    for dtype in ("f32", "f16"):
        for mode in ("metal", "coreml", "injected-failure", "prediction-failure", "missing-model"):
            env = os.environ.copy()
            env.update(GGML_METAL_COREML="0" if mode == "metal" else "1",
                       GGML_METAL_COREML_DEBUG="1",
                       GGML_METAL_COREML_FORCE_FAILURE="1" if mode == "injected-failure" else "0",
                       GGML_METAL_COREML_MODEL_DIR=str(args.models.resolve()))
            if mode == "missing-model":
                env["GGML_METAL_COREML_MODEL_DIR"] = str(args.models.resolve() / "not-present")
            if mode == "prediction-failure":
                env["GGML_METAL_COREML_FORCE_FAILURE"] = "prediction"
            command = [str(args.binary.resolve()), "MTL", str(m), str(n), str(k), dtype, str(args.repeats)]
            try:
                proc = subprocess.run(command, env=env, text=True, capture_output=True, check=False, timeout=args.timeout)
            except subprocess.TimeoutExpired as error:
                records.append(dict(mode=mode, activation=dtype, error=f"timeout after {args.timeout}s"))
                args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
                raise RuntimeError(f"{mode}/{dtype} timed out") from error
            record = dict(mode=mode, activation=dtype, returncode=proc.returncode,
                          stdout=proc.stdout, diagnostics=proc.stderr)
            records.append(record)
            try:
                measurements, selected = validate_run(proc, mode, args.repeats, args.require_ane)
                record.update(ane_plan_predictions=selected, measurements=measurements)
            except (ValueError, RuntimeError) as error:
                record["error"] = str(error)
                raise
            finally:
                args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
            successful_ane += mode == "coreml" and selected > 0
            print(f"{dtype} {mode}: warm={measurements[-1]['warm_ms']:.3f} ms, ANE-plan predictions={selected}")
    if args.require_ane and successful_ane != 2:
        raise RuntimeError("both activation types must exercise CoreML; fallback is not an ANE pass")


if __name__ == "__main__":
    main()
