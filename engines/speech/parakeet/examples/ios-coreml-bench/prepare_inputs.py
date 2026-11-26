#!/usr/bin/env python3
"""Prepare deterministic 16 kHz fixtures for the full-EU benchmark matrix."""
import argparse
import shutil
import wave
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_BYTES = 2
MEL_HOP_SAMPLES = 160


def read_pcm(path):
    with wave.open(str(path)) as source:
        if (source.getframerate(), source.getsampwidth(), source.getnchannels()) != (SAMPLE_RATE, SAMPLE_BYTES, 1):
            raise ValueError(f"Expected mono 16 kHz PCM16: {path}")
        return source.readframes(source.getnframes())


def write_pcm(path, pcm):
    with wave.open(str(path), "wb") as destination:
        destination.setparams((1, SAMPLE_BYTES, SAMPLE_RATE, 0, "NONE", "not compressed"))
        destination.writeframes(pcm)


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("jfk.wav", "fr-multipiece-16k.wav", "hi-16k.wav", "abcba.wav"):
        shutil.copyfile(source / name, output / name)
    english = read_pcm(source / "jfk.wav")
    french = read_pcm(source / "fr-multipiece-16k.wav")
    silence = bytes(SAMPLE_RATE * SAMPLE_BYTES)
    english_window = (english + silence * 15)[:15 * SAMPLE_RATE * SAMPLE_BYTES]
    for repeats in (1, 4, 8, 16, 20):
        write_pcm(output / f"en-{15 * repeats}s.wav", english_window * repeats)
    write_pcm(output / "fr-long.wav", (french + silence) * 12)
    write_pcm(output / "en-partial.wav", english[:int(0.8 * SAMPLE_RATE) * SAMPLE_BYTES])
    write_pcm(output / "silence800.wav", silence[:int(0.8 * SAMPLE_RATE) * SAMPLE_BYTES])
    write_pcm(output / "silence20.wav", silence * 20)
    write_pcm(output / "prefix-silence-fr.wav", silence * 20 + french)
    for frames in range(1, 9):
        samples = 1 + (frames - 1) * MEL_HOP_SAMPLES
        write_pcm(output / f"tiny-{frames}.wav", bytes(samples * SAMPLE_BYTES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.samples, args.output)


if __name__ == "__main__":
    main()
