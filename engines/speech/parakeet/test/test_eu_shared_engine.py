#!/usr/bin/env python3
"""Full-checkpoint shared-sidecar regression through the public CLI/Engine.

Uses isolated deployment directories, including an encoder-less GGUF. The
separate-layout baseline must contain matching fixed routes and precise sidecars.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from test_transcribepsy_eu import sidecar_stem

ENGINE = Path(__file__).parents[1]


def invoke(args, model, wav, language, receipt, owned=False, units=None):
    environment = dict(os.environ)
    environment.pop("PARAKEET_COREML_ONLY", None)
    if owned:
        environment["PARAKEET_COREML_ONLY"] = "1"
    environment.pop("PARAKEET_COREML_DISABLE", None)
    environment.pop("PARAKEET_COREML_COMPUTE_UNITS", None)
    if units is not None:
        environment["PARAKEET_COREML_COMPUTE_UNITS"] = units
    command = [str(args.binary), "--model", str(model), "--wav", str(wav),
        "--language", language, "--backend", "cpu", "--threads", "4", "--bench",
        "--bench-runs", "1", "--bench-warmup", "0", "--bench-json", str(receipt)]
    command.append("--require-coreml")
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    data = json.loads(receipt.read_text())
    assert data["encoder_coreml_all_runs"], result.stderr
    assert data["encoder_owned_by_coreml"] == owned, data
    return result.stdout.strip(), result.stderr


def deploy(args, root, name, stripped=False, separate=False):
    directory = root / name
    directory.mkdir()
    model = directory / args.model.name
    if stripped:
        spec = importlib.util.spec_from_file_location("strip_encoder", ENGINE / "scripts/strip-encoder-gguf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.strip_encoder(args.model, model)
    else:
        model.symlink_to(args.model.resolve())
    stem = sidecar_stem(args.model)
    if separate:
        for path in args.model.parent.glob(stem + "-*.mlmodelc"):
            if "-shared" not in path.name:
                (directory / path.name).symlink_to(path.resolve(), target_is_directory=True)
    else:
        (directory / (stem + "-encoder-shared.mlmodelc")).symlink_to(args.shared.resolve(), target_is_directory=True)
        for path in args.model.parent.glob(stem + "-*.mlmodelc"):
            if "flexible" in path.name or "fp32" in path.name or "decoder" in path.name:
                (directory / path.name).symlink_to(path.resolve(), target_is_directory=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "shared", "english", "french"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not all(path.exists() for path in vars(args).values()):
        raise SystemExit(77)
    stem = sidecar_stem(args.model)
    suffixes = ("-encoder", "-encoder-other", "-encoder-gate", "-encoder-flexible",
                "-encoder-other-flexible", "-encoder-gate-flexible")
    if not all((args.model.parent / (stem + suffix + ".mlmodelc")).is_dir() for suffix in suffixes):
        raise SystemExit(77)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        separate = deploy(args, root, "separate", separate=True)
        shared = deploy(args, root, "shared")
        stripped = deploy(args, root, "stripped", stripped=True)
        for language, wav in (("en", args.english), ("other", args.french),
                              ("auto", args.english), ("auto", args.french)):
            expected, _ = invoke(args, separate, wav, language, root / "reference.json")
            assert expected
            for candidate, owned in ((shared, False), (shared, True), (stripped, True)):
                actual, log = invoke(args, candidate, wav, language, root / "actual.json", owned=owned)
                assert actual == expected, (language, expected, actual)
                assert "-encoder-shared.mlmodelc" in log, log
        for units in ("cpu_only", "cpu_and_gpu"):
            expected, _ = invoke(args, separate, args.french, "auto", root / "reference.json", units=units)
            actual, log = invoke(args, shared, args.french, "auto", root / "actual.json", units=units)
            assert actual == expected
            loaded = re.findall(r"full-EU Core ML (en|other|gate) encoder loaded from '[^']*-encoder-shared.mlmodelc'", log)
            assert loaded == ["en"], log
        # A corrupt shared bundle must recover using complete separate routes.
        fallback = deploy(args, root, "fallback", separate=True)
        stem = next(fallback.parent.glob("*-encoder.mlmodelc")).name.replace("-encoder.mlmodelc", "")
        (fallback.parent / (stem + "-encoder-shared.mlmodelc")).mkdir()
        invoke(args, fallback, args.french, "auto", root / "fallback.json", owned=True)
        (stripped.parent / (stem + "-encoder-shared.mlmodelc")).unlink()
        missing = subprocess.run([str(args.binary), "--model", str(stripped),
            "--wav", str(args.english), "--language", "en", "--backend", "cpu"],
            capture_output=True, text=True, timeout=60)
        assert missing.returncode != 0, "encoder-less GGUF loaded without its shared encoder"
        print("Shared/separate parity passed: en/other/auto, full/owned/encoder-less GGUF, corrupt-bundle fallback")


if __name__ == "__main__":
    main()
