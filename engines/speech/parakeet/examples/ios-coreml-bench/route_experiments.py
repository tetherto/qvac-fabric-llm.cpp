#!/usr/bin/env python3
"""Measure experimental EU route exports on macOS, separately from the runtime.

Use prepare to create masked English/French/silence fixtures and references,
compare to measure compiled predictions and placement, or load in a fresh process
to compare eager loading with a known route or gate-then-selected-route loading.
Load/first-prediction times reflect existing OS caches, not a fresh installation.
"""

import argparse
from collections import Counter
import gc
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import subprocess
import time
import wave

import coremltools as ct
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
ROUTES = ("en", "other", "gate")
INPUT_NAMES = ("mel", "mel_mask", "pad_mask", "key_bias")
MEL_BINS = 128
SAMPLE_HOP = 160
REPEATS = 5
MILLISECONDS = 1000
GATE_MEL_FRAMES = 2001


def exporter():
    spec = importlib.util.spec_from_file_location(
        "eu_experiment", SCRIPTS / "export-transcribepsy-eu-coreml-experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_wav_stems(wavs):
    paths_by_stem = {}
    for wav in wavs:
        previous = paths_by_stem.get(wav.stem)
        if previous is not None:
            raise ValueError(f"Duplicate WAV stem {wav.stem!r}: {previous} and {wav}; "
                             "use distinct filenames for every fixture")
        paths_by_stem[wav.stem] = wav


def prepare(args):
    validate_wav_stems(args.wav)
    import os
    import torch

    experiment = exporter()
    ane = experiment.ANE
    torch.set_num_threads(experiment.DEFAULT_THREADS)
    ref = ane.load_reference_encoder(SCRIPTS)
    weights, meta = ref.load_gguf(args.gguf)
    np.savez(args.output / "gate_weights.npz", threshold=meta["parakeet.eu.gate_threshold"],
             **{key: value.numpy() for key, value in weights.items() if key.startswith("eu.gate.")})
    frames = args.frames
    config = argparse.Namespace(n_mels=MEL_BINS, window_mel_frames=frames,
                                enc_frames=ane.encoder_frames_for_mel(frames, False))
    for wav in args.wav:
        mel_path = args.output / (wav.stem + ".mel")
        subprocess.run([str(args.binary), "--model", str(args.gguf), "--wav", str(wav),
                        "--language", "en", "--backend", "cpu", "--dump-mel", str(mel_path)],
                       env={**os.environ, "PARAKEET_COREML_DISABLE": "1",
                            "PARAKEET_ANE_BATCH_DECODE": "0"}, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mel = torch.from_numpy(np.fromfile(mel_path, np.float32).reshape(MEL_BINS, -1))[:, :frames]
        with wave.open(str(wav)) as audio:
            valid = min(mel.shape[1], (audio.getnframes() + SAMPLE_HOP - 1) // SAMPLE_HOP)
        values, count = ane.window_inputs(config, mel, valid)
        fixture = {name: value.numpy().astype(np.float16)
                   for name, value in zip(INPUT_NAMES, values)}
        fixture["valid_encoder_frames"] = count
        for route in ROUTES:
            fixture[route] = ane.reference_frames(ref, weights, meta, mel, valid, route).numpy()
        np.savez(args.output / (wav.stem + ".npz"), **fixture)
        print(f"Prepared {wav.name}, {valid} valid mel frames", flush=True)


def placement(path, function):
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()

    def walk(block):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            device = usage.preferred_compute_device if usage else None
            name = type(device).__name__ if device else "None"
            counts[name] += 1
            for nested in getattr(operation, "blocks", ()) or ():
                walk(nested)

    walk(plan.model_structure.program.functions[function or "main"].block)
    return dict(counts)


def load_model(path, function=None):
    started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE,
                                      function_name=function)
    return model, (time.perf_counter() - started) * MILLISECONDS


def distance(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), np.finfo(float).tiny))


def gate_probability(features, weights):
    import torch
    import torch.nn.functional as functional

    features = torch.from_numpy(features.astype(np.float32))

    def linear(values, name):
        prefix = "eu.gate." + name
        return functional.linear(values, torch.from_numpy(weights[prefix + ".weight"]),
                                  torch.from_numpy(weights[prefix + ".bias"]))

    attention = linear(torch.tanh(linear(features, "pool.proj.0")), "pool.proj.2").softmax(0)
    mean = (attention * features).sum(0)
    std = ((attention * features.square()).sum(0) - mean.square()).clamp(min=1e-8).sqrt()
    pooled = torch.cat((mean, std))
    normalized = functional.layer_norm(pooled, pooled.shape,
                                       torch.from_numpy(weights["eu.gate.norm.weight"]),
                                       torch.from_numpy(weights["eu.gate.norm.bias"]))
    return float(linear(functional.gelu(linear(normalized, "mlp.0")), "mlp.3").sigmoid())


def measure_prediction(model, fixture, scale=None):
    inputs = {name: fixture[name] for name in INPUT_NAMES}
    if scale is not None:
        inputs["adapter_scale"] = np.full((1, 1, 1, 1), scale, np.float16)
    started = time.perf_counter()
    output = model.predict(inputs)
    first_ms = (time.perf_counter() - started) * MILLISECONDS
    times = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        model.predict(inputs)
        times.append((time.perf_counter() - started) * MILLISECONDS)
    return output, {"first_prediction_ms": first_ms, "warm_median_ms": statistics.median(times)}


def compare_variant(path, function, route, fixtures, unified, gate_weights, baseline_dir):
    model, load_ms = load_model(path, function)
    rows = []
    for fixture_path in fixtures:
        with np.load(fixture_path) as fixture:
            count = int(fixture["valid_encoder_frames"])
            scale = int(route == "other") if unified else None
            output, timing = measure_prediction(model, fixture, scale)
            key = "gate_features" if unified and route == "gate" else "encoder_out"
            got = output[key][0, :, 0, :count].T
            if not np.isfinite(got).all():
                raise ValueError(f"Non-finite {path.name}/{route}/{fixture_path.stem}")
            row = {"fixture": fixture_path.stem, **timing,
                   "relative_l2_to_fp32": distance(got, fixture[route])}
            baseline = baseline_dir / f"{route}-{fixture_path.stem}.npy"
            if not unified and function is None:
                np.save(baseline, got)
            else:
                separate = np.load(baseline)
                row.update(identical_to_separate=bool(np.array_equal(got, separate)),
                           max_absolute_to_separate=float(np.max(np.abs(got.astype(float) - separate))))
            if route == "gate":
                threshold = float(gate_weights["threshold"])
                reference_probability = gate_probability(fixture[route], gate_weights)
                probability = gate_probability(got, gate_weights)
                row.update(reference_probability=reference_probability, probability=probability,
                           threshold=threshold,
                           decision_matches=(probability >= threshold) == (reference_probability >= threshold))
            rows.append(row)
    return {"path": str(path), "function": function, "route": route,
            "load_ms": load_ms, "placement": placement(path, function), "predictions": rows}


def directory_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def compare(args):
    fixtures = sorted(path for path in args.fixtures.glob("*.npz") if path.stem != "gate_weights")
    if not fixtures:
        raise ValueError("No prepared .npz fixtures")
    sizes = {str(path.name): directory_bytes(path) for path in args.models.iterdir()
             if path.suffix in (".mlpackage", ".mlmodelc")}
    records = []
    gate_weights = dict(np.load(args.fixtures / "gate_weights.npz"))
    baseline_dir = args.output / "baseline_predictions"
    baseline_dir.mkdir(exist_ok=True)
    for kind in ("separate", "shared", "unified"):
        for route in ROUTES:
            name = route if kind == "separate" else kind
            path = args.models / (name + ".mlmodelc")
            try:
                record = compare_variant(path, route if kind == "shared" else None,
                                         route, fixtures, kind == "unified", gate_weights, baseline_dir)
                record["kind"] = kind
            except Exception as exc:
                record = {"kind": kind, "route": route, "error": str(exc)}
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "comparison.json").write_text(json.dumps(
                {"sizes_bytes": sizes, "records": records}, indent=2))
            gc.collect()
    if any("error" in record for record in records):
        raise RuntimeError("Some variants failed; see comparison.json")


def write_phone_fixtures(args):
    fixtures = []
    for path in sorted(args.fixtures.glob("*.npz")):
        if path.stem == "gate_weights":
            continue
        directory = args.output / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        with np.load(path) as fixture:
            for name in (*INPUT_NAMES, *ROUTES):
                fixture[name].astype(np.float32).tofile(directory / (name + ".bin"))
            fixtures.append({"name": path.stem,
                             "valid_encoder_frames": int(fixture["valid_encoder_frames"]),
                             "shapes": {name: list(fixture[name].shape) for name in INPUT_NAMES}})
    if not fixtures:
        raise ValueError("No prepared .npz fixtures")
    (args.output / "manifest.json").write_text(json.dumps({"fixtures": fixtures}, indent=2))


def phone_prediction(result_path, route, name, fixtures):
    with np.load(fixtures / (name + ".npz")) as fixture:
        shape = fixture[route].shape
    return np.fromfile(result_path.parent / f"{route}-{name}.bin", np.float32).reshape(shape)


def summarize_native(args):
    sources = [(path, json.loads(path.read_text())) for path in args.native_results.rglob("results.json")]
    if not sources:
        raise ValueError("No native probe results.json files")
    weights = dict(np.load(args.fixtures / "gate_weights.npz"))
    baselines = {}
    for path, data in sources:
        for record in data["records"]:
            if record["kind"] == "separate":
                for row in record["predictions"]:
                    key = (data["model_directory"], record["route"], row["fixture"])
                    baselines[key] = phone_prediction(path, record["route"], row["fixture"], args.fixtures)
    for path, data in sources:
        data["source"] = str(path)
        for record in data["records"]:
            for row in record.get("predictions", []):
                route, name = record["route"], row["fixture"]
                got = phone_prediction(path, route, name, args.fixtures)
                baseline = baselines.get((data["model_directory"], route, name))
                if baseline is not None:
                    row["identical_to_separate"] = bool(np.array_equal(got, baseline))
                    row["max_absolute_to_separate"] = float(np.max(np.abs(got - baseline)))
                if route == "gate":
                    with np.load(args.fixtures / (name + ".npz")) as fixture:
                        reference = gate_probability(fixture[route], weights)
                    probability = gate_probability(got, weights)
                    threshold = float(weights["threshold"])
                    row.update(reference_probability=reference, probability=probability, threshold=threshold,
                               decision_matches=(probability >= threshold) == (reference >= threshold))
    (args.output / "native-summary.json").write_text(json.dumps([data for _, data in sources], indent=2))


def measure_loading(args):
    models = []
    records = []
    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for route in args.routes:
        suffix = "" if route == "en" else "-" + route
        name = route if args.model_layout == "experiment" else f"transcribepsy-250m-eu-encoder{suffix}"
        path = args.models / (name + ".mlmodelc")
        model, elapsed = load_model(path)
        models.append(model)
        records.append({"route": route, "load_ms": elapsed})
    result = {"routes": records, "total_load_ms": sum(row["load_ms"] for row in records),
              "initial_peak_rss_bytes": initial_peak,
              "final_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "note": "macOS process peak RSS; OS specialization caches retained"}
    (args.output / "loading.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare", "load", "phone-fixtures", "summarize-native"))
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--frames", type=int, default=GATE_MEL_FRAMES)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--model-layout", choices=("runtime", "experiment"), default="runtime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare" and (not args.gguf or not args.binary or not args.wav):
        parser.error("prepare requires --gguf, --binary and --wav")
    if args.frames < GATE_MEL_FRAMES:
        parser.error("--frames must cover the 2001-frame gate window")
    if args.action in ("compare", "load") and not args.models:
        parser.error("compare/load require --models")
    if args.action in ("compare", "phone-fixtures", "summarize-native") and not args.fixtures:
        parser.error("compare/phone-fixtures/summarize-native require --fixtures")
    if args.action == "summarize-native" and not args.native_results:
        parser.error("summarize-native requires --native-results")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "compare": compare, "load": measure_loading,
     "phone-fixtures": write_phone_fixtures, "summarize-native": summarize_native}[args.action](args)


if __name__ == "__main__":
    main()
