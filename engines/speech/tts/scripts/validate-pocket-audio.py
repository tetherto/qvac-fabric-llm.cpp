#!/usr/bin/env python3
"""Transcribe a Pocket benchmark corpus and check PCM health.

Validation-only dependencies: faster-whisper, numpy, scipy. ASR word error
rates are a proxy for intelligibility, not a subjective naturalness score.
"""
import argparse
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel


def words(text):
    text = unicodedata.normalize("NFKD", text.lower()).replace("’", "'")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def distance(reference, actual):
    row = list(range(len(actual)+1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, observed in enumerate(actual, 1):
            next_row.append(min(row[j]+1, next_row[j-1]+1, row[j-1]+(expected != observed)))
        row = next_row
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", default="tiny.en")
    args = parser.parse_args()
    comparison = json.loads((args.directory/"comparison.json").read_text())
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=2)
    output = {"asr_model": args.model, "asr_compute_type": "int8", "cases": []}
    for case in comparison["cases"]:
        for implementation in ("upstream", "fabric"):
            path = args.directory/case["name"]/f"pocket-{implementation}.wav"
            rate, pcm = wavfile.read(path)
            if pcm.dtype == np.int16:
                pcm = pcm.astype(np.float32)/32768
            if rate != 24000 or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all() or not np.any(pcm):
                raise RuntimeError(f"Invalid PCM: {path}")
            segments, _ = model.transcribe(str(path), language="en", beam_size=5)
            transcript = " ".join(segment.text.strip() for segment in segments)
            expected, actual = words(case["text"]), words(transcript)
            record = dict(name=case["name"], implementation=implementation,
                          transcript=transcript, reference=case["text"],
                          word_error_rate=distance(expected, actual)/len(expected),
                          duration_seconds=len(pcm)/rate, peak=float(np.max(np.abs(pcm))),
                          rms=float(np.sqrt(np.mean(pcm**2))),
                          clipping_fraction=float(np.mean(np.abs(pcm) >= 1)))
            print(json.dumps(record), flush=True)
            output["cases"].append(record)
            (args.directory/"audio-validation.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
