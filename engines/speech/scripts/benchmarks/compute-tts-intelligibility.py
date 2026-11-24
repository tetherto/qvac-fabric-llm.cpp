#!/usr/bin/env python3
"""Transcribe one generated WAV with Whisper CPU and score English WER.

ASR runs outside synthesis timing. WER is an intelligibility proxy that includes
reference-ASR errors, not a general audio-quality score or a pass/fail gate.
Only a successful ASR invocation with a newly written transcript can be scored.
An empty successful transcript is valid and scores as all reference words deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave


def load_wer():
    spec = importlib.util.spec_from_file_location(
        "benchmark_wer", Path(__file__).with_name("compute-wer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and os.path.samefile(left, right))


def validate_paths(args) -> None:
    inputs = [args.audio, args.reference, args.asr_binary, args.asr_model]
    outputs = [args.json_out, args.transcript_out, args.log_out]
    for i, output in enumerate(outputs):
        for other in inputs + outputs[:i]:
            if same_file(output, other):
                raise ValueError(f"output path aliases another input/output: {output}")


def validate_wav(path: Path) -> dict:
    """Validate full PCM payload, retaining native rate/channels for the decoder.

    Silence is allowed: valid generated silence must be transcribed and counted
    as deletions rather than hidden as unavailable quality data.
    """
    with wave.open(str(path), "rb") as audio:
        channels, width, rate, frames, compression, _ = audio.getparams()
        if compression != "NONE" or width not in (1, 2, 3, 4):
            raise ValueError("audio must be uncompressed integer PCM WAV")
        if channels < 1 or rate < 1 or frames < 1:
            raise ValueError("audio WAV has no frames or invalid format")
        remaining = frames
        while remaining:
            chunk_frames = min(remaining, 65536)
            chunk = audio.readframes(chunk_frames)
            if len(chunk) != chunk_frames * channels * width:
                raise ValueError("audio WAV payload is truncated")
            remaining -= chunk_frames
    return {"sample_rate": rate, "channels": channels,
            "sample_width_bytes": width, "frames": frames,
            "duration_seconds": frames / rate}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(args, result: dict) -> None:
    # Reject aliases before truncating logs or removing stale transcripts.
    validate_paths(args)
    for output in (args.json_out, args.transcript_out, args.log_out):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.transcript_out.unlink(missing_ok=True)
    args.log_out.write_text("", encoding="utf-8")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    wer = load_wer()
    reference = args.reference.read_text(encoding="utf-8")
    if not wer.normalize_english(reference):
        raise ValueError("reference contains no words after English normalization")
    result["n_ref_words"] = len(wer.normalize_english(reference).split())
    result["audio_metadata"] = validate_wav(args.audio)
    if not args.asr_binary.is_file() or not os.access(args.asr_binary, os.X_OK):
        result["status"] = "unavailable"
        raise ValueError(f"ASR binary is missing or not executable: {args.asr_binary}")
    if not args.asr_model.is_file() or args.asr_model.stat().st_size == 0:
        result["status"] = "unavailable"
        raise ValueError(f"ASR model is missing or empty: {args.asr_model}")
    result["asr"]["model_sha256"] = sha256(args.asr_model)
    # Whisper appends .txt to --output-file. Use a private directory so existing
    # public artifacts cannot be mistaken for output of this invocation.
    with tempfile.TemporaryDirectory(prefix="tts-asr-") as temporary:
        prefix = Path(temporary) / "transcript"
        command = [str(args.asr_binary.resolve()), "-m", str(args.asr_model.resolve()),
                   "-f", str(args.audio.resolve()), "-l", "en", "-t", "4",
                   "-ng", "-nt", "-otxt", "-of", str(prefix),
                   "-bo", "1", "-bs", "1", "-tp", "0", "-nf"]
        result["asr"]["command"] = command
        with args.log_out.open("w", encoding="utf-8") as log:
            log.write("Command: " + json.dumps(command) + "\n")
            log.flush()
            try:
                completed = subprocess.run(command, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f"ASR timed out after {args.timeout:g} seconds") from exc
        result["asr"]["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise ValueError(f"ASR exited with code {completed.returncode}; see {args.log_out}")
        transcript = prefix.with_suffix(".txt")
        if not transcript.is_file():
            raise ValueError("ASR exited successfully but did not write a transcript")
        hypothesis = transcript.read_text(encoding="utf-8")
        args.transcript_out.write_text(hypothesis, encoding="utf-8")
    result.update(wer.compute_wer(hypothesis, reference, normalizer="english"))
    result["status"] = "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("audio", "reference", "asr-binary", "asr-model", "json-out",
                 "transcript-out", "log-out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300,
                        help="ASR timeout in seconds (default: 300)")
    args = parser.parse_args()
    result = {
        "kind": "tts_intelligibility", "status": "error", "wer": None,
        "n_ref_words": None, "n_hyp_words": None, "n_edits": None,
        "normalizer": "english", "reference": str(args.reference),
        "audio": str(args.audio), "transcript": str(args.transcript_out),
        "log": str(args.log_out),
        "asr": {"binary": str(args.asr_binary), "model": str(args.asr_model),
                "model_sha256": None, "backend": "cpu", "language": "en",
                "threads": 4, "temperature": 0, "temperature_fallback": False,
                "best_of": 1, "beam_size": 1, "timeout_seconds": args.timeout,
                "command": None, "returncode": None},
    }
    safe_outputs = False
    try:
        validate_paths(args)
        safe_outputs = True
        score(args, result)
    except (OSError, ValueError, EOFError, wave.Error) as exc:
        result["reason"] = str(exc)
    # Do not emit NaN/Infinity even for invalid timeout arguments.
    if not math.isfinite(args.timeout):
        result["asr"]["timeout_seconds"] = None
    if safe_outputs:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
                                     encoding="utf-8")
        except OSError as exc:
            result.update(status="error", wer=None, reason=f"cannot write score JSON: {exc}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
