#!/usr/bin/env python3
"""Integration check for the full TranscribePsy EU gate and CAFE routes.

Requires a converted full-EU GGUF. A missing optional model skips in CTest.
"""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


def transcribe(binary: Path, model: Path, wav: Path, language: str, coreml=False):
    environment = dict(os.environ)
    if not coreml:
        environment["PARAKEET_COREML_DISABLE"] = "1"
    else:
        environment.pop("PARAKEET_COREML_DISABLE", None)
    extra = (["--bench", "--bench-runs", "1", "--bench-warmup", "0", "--require-coreml"]
             if coreml else [])
    result = subprocess.run(
        [str(binary), "--model", str(model), "--wav", str(wav),
         "--language", language, "--backend", "cpu", "--threads", "4", "--verbose"] + extra,
        capture_output=True, text=True, check=True, env=environment,
    )
    return result.stdout.strip(), result.stderr


def gate_decision(log: str):
    match = re.search(
        r"TranscribePsy EU gate: p\(non-English\)=([0-9.]+), "
        r"threshold=([0-9.]+), route=(en|other), frames=(\d+)/(\d+)", log)
    assert match, "gate decision missing from log"
    return float(match.group(1)), float(match.group(2)), match.group(3), \
        int(match.group(4)), int(match.group(5))


def write_wav(path: Path, pcm: bytes, sample_rate: int):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def sidecar_stem(model: Path):
    return re.sub(r"[.-](f32|f16|bf16|q8_0|q5_0|q4_0|q6_k|q5_k|q4_k|mxfp4|nvfp4)$",
                  "", model.stem)


def check_sidecar_fallbacks(binary: Path, model: Path, french: Path):
    stem = sidecar_stem(model)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        candidate = directory / model.name
        candidate.symlink_to(model.resolve())
        encoder = model.parent / f"{stem}-encoder.mlmodelc"
        (directory / encoder.name).symlink_to(encoder.resolve(), target_is_directory=True)
        # Deliberately exchange the English and adapted encoder identities.
        (directory / f"{stem}-encoder-other.mlmodelc").symlink_to(
            encoder.resolve(), target_is_directory=True)
        command = [str(binary), "--model", str(candidate), "--wav", str(french),
                   "--backend", "cpu", "--threads", "4", "--bench", "--bench-runs", "1",
                   "--bench-warmup", "0", "--require-coreml"]
        environment = dict(os.environ)
        environment.pop("PARAKEET_COREML_DISABLE", None)
        environment.pop("PARAKEET_COREML_ONLY", None)
        swapped = subprocess.run(command + ["--language", "other"],
                                 env=environment, capture_output=True, text=True)
        assert swapped.returncode != 0 and "incompatible full-EU Core ML other" in swapped.stderr, swapped
        fallback_command = [part for part in command if part != "--require-coreml"]
        fallback = subprocess.run(fallback_command + ["--language", "other"],
                                  env=environment, capture_output=True, text=True)
        assert fallback.returncode == 0 and "analyses" in fallback.stdout, fallback
        other = directory / f"{stem}-encoder-other.mlmodelc"
        other.unlink()
        other.symlink_to((model.parent / other.name).resolve(), target_is_directory=True)
        # A missing gate must not pass --require-coreml just because the main
        # encoder loaded. Native gate fallback remains usable without the flag.
        missing = subprocess.run(command + ["--language", "auto"],
                                 env=environment, capture_output=True, text=True)
        assert missing.returncode != 0 and "gate encoder: ggml" in missing.stderr, missing
        missing_receipt = directory / "missing-owner.json"
        missing_owner = subprocess.run(fallback_command + ["--language", "auto", "--bench-json", str(missing_receipt)],
            env={**environment, "PARAKEET_COREML_ONLY": "1"}, capture_output=True, text=True)
        assert missing_owner.returncode == 0 and "analyses" in missing_owner.stdout, missing_owner
        missing_data = json.loads(missing_receipt.read_text())
        assert not missing_data["encoder_owned_by_coreml"] and not missing_data["encoder_coreml_all_runs"], missing_data
        gate = directory / f"{stem}-encoder-gate.mlmodelc"
        gate.symlink_to(encoder.resolve(), target_is_directory=True)
        wrong_gate = subprocess.run(command + ["--language", "auto"],
                                    env=environment, capture_output=True, text=True)
        assert wrong_gate.returncode != 0 and "incompatible full-EU Core ML gate" in wrong_gate.stderr, wrong_gate
        gate.unlink()
        gate.symlink_to((model.parent / gate.name).resolve(), target_is_directory=True)
        # Decoder availability must not masquerade as execution. Encoder-only
        # deployment stays valid when the optional decoder sidecar is absent.
        without_decoder = subprocess.run(command + ["--language", "other"],
            env={**environment, "PARAKEET_ANE_BATCH_DECODE": "1"}, capture_output=True, text=True)
        assert without_decoder.returncode == 0 and "analyses" in without_decoder.stdout, without_decoder
        assert "decoder_backend=CPU" in without_decoder.stderr, without_decoder
        with wave.open(str(french)) as source:
            pcm, rate = source.readframes(source.getnframes()), source.getframerate()
        oversized = directory / "oversized.wav"
        write_wav(oversized, pcm * 6, rate)
        owned_command = [str(binary), "--model", str(candidate), "--wav", str(oversized),
            "--backend", "cpu", "--threads", "4", "--language", "other", "--window-sec", "0",
            "--bench", "--bench-runs", "1", "--bench-warmup", "0", "--require-coreml"]
        missing_flexible = subprocess.run(owned_command,
            env={**environment, "PARAKEET_COREML_ONLY": "1"}, capture_output=True, text=True)
        assert missing_flexible.returncode != 0 and "weights were not loaded" in missing_flexible.stderr, missing_flexible


def check_flexible_only_cli(binary: Path, model: Path, french: Path):
    stem = sidecar_stem(model)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        candidate = directory / model.name
        candidate.symlink_to(model.resolve())
        for suffix in ("encoder-flexible", "encoder-other-flexible", "encoder-gate-flexible"):
            sidecar = model.parent / f"{stem}-{suffix}.mlmodelc"
            (directory / sidecar.name).symlink_to(sidecar.resolve(), target_is_directory=True)
        environment = dict(os.environ)
        for key in ("PARAKEET_COREML_DISABLE", "PARAKEET_COREML_ONLY", "PARAKEET_ANE_BATCH_DECODE"):
            environment.pop(key, None)
        for language in ("en", "other", "auto"):
            result = subprocess.run(
                [str(binary), "--model", str(candidate), "--wav", str(french),
                 "--language", language, "--backend", "cpu", "--threads", "4",
                 "--window-sec", "0", "--bench", "--bench-runs", "1", "--bench-warmup", "0",
                 "--require-coreml"], env=environment, capture_output=True, text=True)
            assert result.returncode == 0 and result.stdout.strip(), result
            assert "encoder_backend=coreml-cpu" in result.stderr, result
            if language == "auto":
                assert "gate encoder: coreml" in result.stderr, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--coreml", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        print(f"SKIP: model fixture missing: {args.model}")
        return 77
    required_sidecars = ("encoder", "encoder-other", "encoder-gate")
    if args.coreml and not all(
            (args.model.parent / f"{sidecar_stem(args.model)}-{suffix}.mlmodelc").is_dir()
            for suffix in required_sidecars):
        print("SKIP: full-EU Core ML sidecars missing")
        return 77

    samples = Path(__file__).parent / "samples"
    for filename, explicit, expected_route, words in (
        ("jfk.wav", "en", "en", ("fellow", "Americans", "country")),
        ("fr-multipiece-16k.wav", "other", "other", ("analyses", "politiques")),
    ):
        wav = samples / filename
        explicit_text, _ = transcribe(args.binary, args.model, wav, explicit, args.coreml)
        auto_text, log = transcribe(args.binary, args.model, wav, "auto", args.coreml)
        if args.coreml:
            assert "TranscribePsy EU gate encoder: coreml" in log, log
        probability, threshold, route, _, _ = gate_decision(log)
        assert route == expected_route, (filename, log)
        assert (probability >= threshold) == (expected_route == "other")
        assert auto_text == explicit_text, (filename, auto_text, explicit_text)
        assert all(word.lower() in auto_text.lower() for word in words), auto_text

    with wave.open(str(samples / "jfk.wav"), "rb") as source:
        rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        prefix = (pcm * 2)[:20 * rate * 2]
        loud_tail = b"".join(struct.pack("<h", int(28000 * math.sin(2 * math.pi * 440 * i / rate)))
                             for i in range(10 * rate))
        short_path = directory / "prefix.wav"
        long_path = directory / "long.wav"
        aligned_path = directory / "aligned.wav"
        silent_path = directory / "silent.wav"
        write_wav(short_path, prefix, rate)
        write_wav(long_path, prefix + loud_tail, rate)
        write_wav(aligned_path, pcm[:int(0.8 * rate) * 2], rate)
        write_wav(silent_path, bytes(int(0.8 * rate) * 2), rate)
        _, short_log = transcribe(args.binary, args.model, short_path, "auto", args.coreml)
        _, long_log = transcribe(args.binary, args.model, long_path, "auto", args.coreml)
        assert gate_decision(short_log)[:3] == gate_decision(long_log)[:3], \
            (short_log, long_log)
        _, aligned_log = transcribe(args.binary, args.model, aligned_path, "auto", args.coreml)
        expected = (10, 10 if args.coreml else 11)
        assert gate_decision(aligned_log)[3:] == expected, aligned_log
        _, silent_log = transcribe(args.binary, args.model, silent_path, "auto", args.coreml)
        assert gate_decision(silent_log)[3:] == expected, silent_log
    if args.coreml:
        check_sidecar_fallbacks(args.binary, args.model, samples / "fr-multipiece-16k.wav")
        flexible = ("encoder-flexible", "encoder-other-flexible", "encoder-gate-flexible")
        if all((args.model.parent / f"{sidecar_stem(args.model)}-{suffix}.mlmodelc").is_dir()
               for suffix in flexible):
            check_flexible_only_cli(args.binary, args.model, samples / "fr-multipiece-16k.wav")
    print("TranscribePsy EU: English/French gate and explicit routes agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
