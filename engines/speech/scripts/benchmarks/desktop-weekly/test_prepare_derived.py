#!/usr/bin/env python3
import array
import importlib.util
import math
import pathlib
import tempfile
import unittest
import wave

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_derived", "prepare-derived.py")
TONE_HZ = 440.0
AMPLITUDE = 10000


def tone(rate, seconds):
    return array.array("h", (int(AMPLITUDE * math.sin(2 * math.pi * TONE_HZ * index / rate)) for index in range(int(rate * seconds))))


class PrepareDerivedTest(unittest.TestCase):
    def test_resample_changes_rate_and_keeps_duration(self):
        samples = tone(16000, 0.5)
        resampled = prepare.resample_samples(samples, 16000, 24000)
        self.assertEqual(len(resampled), 12000)
        self.assertEqual(resampled[0], samples[0])
        self.assertAlmostEqual(max(resampled), AMPLITUDE, delta=AMPLITUDE * 0.05)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = str(pathlib.Path(tmp) / "in.wav")
            out = str(pathlib.Path(tmp) / "out.wav")
            prepare.write_pcm16_mono(source, tone(16000, 1.0), 16000)
            prepare.main(["resample", "--source", source, "--out", out, "--rate", "24000"])
            with wave.open(out, "rb") as audio:
                self.assertEqual(audio.getframerate(), 24000)
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getnframes(), 24000)

    def test_rejects_stereo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "stereo.wav")
            with wave.open(path, "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 4)
            with self.assertRaises(RuntimeError):
                prepare.read_pcm16_mono(path)


if __name__ == "__main__":
    unittest.main()
