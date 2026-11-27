#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import math
import subprocess
import tempfile

STEM = "diar_streaming_sortformer_4spk-v2.1"
MODEL_NAME = STEM + ".q8_0.gguf"
ARMS = ("metal", "diarizer")
FIXTURE = "abcba"
CALLS = 2
CAPACITY = 2048
MAXIMUM_ERROR = 0.02
RELATIVE_L2 = 0.003
COREML_ENV_PREFIX = "PARAKEET_COREML_"
TIMEOUT_SECONDS = 600
COLLAR_MS = 250


def load_scorer():
    spec = importlib.util.spec_from_file_location("compute_der", Path(__file__).with_name("compute-der.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset_manifest(root):
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
             "sha256": hash_file(path)} for path in sorted(root.rglob("*")) if path.is_file()]


def stage_arm(assets, destination, arm):
    destination.mkdir(parents=True, exist_ok=False)
    model = assets / MODEL_NAME
    if not model.is_file() or model.stat().st_size == 0:
        raise ValueError(f"missing model: {model}")
    (destination / MODEL_NAME).symlink_to(model.resolve())
    names = {"metal": (), "diarizer": (STEM + f"-diarizer-{CAPACITY}",)}[arm]
    for name in names:
        bundle = assets / (name + ".mlmodelc")
        if not (bundle / "model.mil").is_file() or not (bundle / "weights/weight.bin").is_file():
            raise ValueError(f"missing or incomplete bundle: {bundle}")
        (destination / bundle.name).symlink_to(bundle.resolve(), target_is_directory=True)
    return destination / MODEL_NAME


def arm_environment(arm):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(COREML_ENV_PREFIX)}
    environment["PARAKEET_COREML_COMPUTE_UNITS"] = "cpu_and_ane"
    if arm == "metal":
        environment["PARAKEET_COREML_DISABLE"] = "1"
    else:
        environment["PARAKEET_COREML_STRICT"] = "1"
    return environment


def validate_result(result, arm):
    if result["coreml"] is not (arm == "diarizer"):
        raise ValueError(f"unexpected backend in {arm}")
    probabilities = result["probabilities"]
    if result["frames"] <= 0 or result["speakers"] <= 0 or not probabilities:
        raise ValueError("empty probability output")
    if len(probabilities) != result["frames"] * result["speakers"]:
        raise ValueError("invalid probability shape")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("invalid probability value")


def compare(actual, reference):
    if any(actual[key] != reference[key] for key in ("frames", "speakers", "stride")):
        raise ValueError("probability dimensions or time grid differ")
    deltas = [a - b for a, b in zip(actual["probabilities"], reference["probabilities"])]
    maximum = max(abs(delta) for delta in deltas)
    energy = sum(value * value for value in reference["probabilities"])
    relative = math.sqrt(sum(delta * delta for delta in deltas) / max(energy, 1e-30))
    return {"maximum_probability_error": maximum, "relative_l2": relative,
            "passed": maximum <= MAXIMUM_ERROR and relative <= RELATIVE_L2}


def run_arm(args, staging, arm):
    model = stage_arm(args.assets, staging / arm, arm)
    prefix = args.out / arm
    command = [str(args.binary.resolve()), str(model.absolute()),
               str(args.samples / (FIXTURE + ".wav")), str(prefix)]
    prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
    with prefix.with_suffix(".stdout").open("w") as stdout, prefix.with_suffix(".log").open("w") as stderr:
        subprocess.run(command, env=arm_environment(arm), stdout=stdout, stderr=stderr,
                       timeout=TIMEOUT_SECONDS, check=True)
    return load_calls(prefix, arm)


def load_calls(prefix, arm):
    results = []
    for call in range(1, CALLS + 1):
        result = json.loads(Path(f"{prefix}-{call}.json").read_text())
        validate_result(result, arm)
        results.append(result)
    return results


def score_calls(results, reference):
    scorer = load_scorer()
    scores = []
    for result in results:
        hypothesis = scorer.parse_hypothesis_jsonl(
            "\n".join(json.dumps(segment) for segment in result["segments"]))
        scores.append(scorer.compute_der(hypothesis, reference, collar_ms=COLLAR_MS))
    return scores


def run_comparison(args, staging):
    reference = load_scorer().parse_rttm((args.samples / (FIXTURE + ".rttm")).read_text())
    if len({segment[2] for segment in reference}) != 3:
        raise ValueError("expected a three-speaker reference")
    outputs = {arm: run_arm(args, staging, arm) for arm in ARMS}
    comparisons = [compare(actual, native) for actual, native in
                   zip(outputs["diarizer"], outputs["metal"])]
    return {"probability_comparisons": comparisons,
            "der": {arm: score_calls(outputs[arm], reference) for arm in ARMS},
            "passed": all(row["passed"] for row in comparisons),
            "thresholds": {"maximum_probability_error": MAXIMUM_ERROR, "relative_l2": RELATIVE_L2},
            "threshold_source": "engines/speech/parakeet/test/test_sortformer_diarizer_coreml_parity.cpp",
            "der_gate": None, "collar_ms": COLLAR_MS,
            "fixture": FIXTURE, "calls_per_backend": CALLS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"assets": asset_manifest(args.assets), "binary_sha256": hash_file(args.binary),
                "samples": asset_manifest(args.samples), "collar_ms": COLLAR_MS}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="sortformer-preflight-") as staging:
        result = run_comparison(args, Path(staging))
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
