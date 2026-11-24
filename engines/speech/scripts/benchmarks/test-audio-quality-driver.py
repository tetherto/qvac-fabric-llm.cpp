#!/usr/bin/env python3
"""Model-free integration tests for audio benchmark capture and scoring.

Run with python3 scripts/benchmarks/test-audio-quality-driver.py; requires jq.
The mock engine is intentionally unaware of scorer internals: it records the
requested paths and writes distinct warmup / timed outputs to expose stale WAVs.
"""
from __future__ import annotations

import json
import importlib.util
import math
import os
from pathlib import Path
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def write_wav(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def read_samples(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return struct.unpack("<" + "h" * (len(frames) // 2), frames)


def expected_sisdr(reference, hypothesis):
    ref_mean = sum(reference) / len(reference)
    hyp_mean = sum(hypothesis) / len(hypothesis)
    ref = [v - ref_mean for v in reference]
    hyp = [v - hyp_mean for v in hypothesis]
    scale = sum(a * b for a, b in zip(ref, hyp)) / sum(a * a for a in ref)
    target = [scale * a for a in ref]
    return 10 * math.log10(sum(a * a for a in target) /
                          sum((a - b) ** 2 for a, b in zip(hyp, target)))


@unittest.skipUnless(shutil.which("jq"), "run-family.sh requires jq")
class AudioDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="audio-driver-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = self.root / "build"
        (self.build / "bin").mkdir(parents=True)
        self.reference = self.root / "clean speech.wav"
        self.noisy = self.root / "hypothesis.wav"
        self.ref_samples = [int(10000 * math.sin(2 * math.pi * 223 * i / 16000))
                            for i in range(16000)]
        self.hyp_samples = [v + int(1000 * math.sin(2 * math.pi * 733 * i / 16000))
                            for i, v in enumerate(self.ref_samples)]
        write_wav(self.reference, self.ref_samples)
        write_wav(self.noisy, self.hyp_samples)
        self.log = self.root / "calls.jsonl"
        self.out = self.root / "result.json"
        stub = self.build / "bin" / "audio-stub"
        stub.write_text('''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
source, output = map(pathlib.Path, sys.argv[1:3])
log = pathlib.Path(os.environ['AUDIO_TEST_LOG'])
count = len(log.read_text().splitlines()) if log.exists() else 0
with log.open('a') as f:
    f.write(json.dumps({'input': str(source), 'output': str(output)}) + '\\n')
mode = os.environ.get('AUDIO_TEST_MODE', 'normal')
if mode == 'missing' or (mode == 'last-missing' and count == 2):
    sys.exit(0)
if mode == 'malformed':
    output.write_text('not a wave file')
else:
    shutil.copyfile(source if count < 2 else os.environ['AUDIO_TEST_HYP'], output)
''')
        stub.chmod(0o755)

    def run_driver(self, mode="normal", kind="audio", metrics=None, degradation=None,
                   without_optional_dependencies=False):
        correctness = {"kind": kind, "reference": str(self.reference)}
        if kind == "audio":
            correctness["metrics"] = metrics or ["sisdr"]
        if degradation is not None:
            correctness["degradation"] = degradation
        spec = {"audio-test": {
            "bench_kind": "time-wrapped", "binary": "bin/audio-stub",
            "cmake_target": "unused", "models": [], "notes": "test fixture",
            "args": ["${AUDIO_INPUT}", "${AUDIO_OUT}"],
            "audio_duration_seconds": 1, "correctness": correctness}}
        spec_path = self.root / "families.json"
        spec_path.write_text(json.dumps(spec))
        env = dict(os.environ, BENCH_FAMILIES_JSON=str(spec_path),
                   MODEL_S3_BUCKET="", AUDIO_TEST_LOG=str(self.log),
                   AUDIO_TEST_HYP=str(self.noisy), AUDIO_TEST_MODE=mode)
        if without_optional_dependencies:
            python_wrapper = self.root / "python-without-site"
            python_wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' -S "$@"\n')
            python_wrapper.chmod(0o755)
            env["AUDIO_QUALITY_PYTHON"] = str(python_wrapper)
            env.pop("PYTHONPATH", None)
        completed = subprocess.run([
            "bash", str(HERE / "run-family.sh"), "--family", "audio-test",
            "--runs", "2", "--warmup", "1", "--build-dir", str(self.build),
            "--models-root", str(self.root / "models"), "--out", str(self.out)
        ], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(self.out.exists(), completed.stderr)
        result = json.loads(self.out.read_text())
        self.assertEqual(result["status"], "ok", completed.stderr)
        self.assertGreater(result["wall_ms_median"], 0)
        return result

    def test_preparation_metadata_identifies_exact_input(self):
        self.run_driver(degradation={"snr_db": 10, "seed": 42})
        metadata = json.loads(self.out.with_suffix(".input-preparation.json").read_text())
        self.assertEqual(metadata["sample_rate"], 16000)
        self.assertEqual(metadata["frames"], len(self.ref_samples))
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["snr_db"], 10)
        self.assertGreater(metadata["scale"], 0)
        self.assertLessEqual(metadata["scale"], 1)

    def test_scores_final_timed_output_and_retains_artifacts(self):
        result = self.run_driver()
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({c["output"] for c in calls}), 3,
                         "warmup and timed runs must not share a WAV path")
        self.assertTrue(all(c["input"] == str(self.reference) for c in calls))
        self.assertEqual(result["correctness_kind"], "audio")
        self.assertEqual(result["correctness_reference"], str(self.reference))
        score = result["audio_quality"]["sisdr"]
        self.assertEqual(score["status"], "ok")
        self.assertAlmostEqual(score["value"], expected_sisdr(self.ref_samples, self.hyp_samples), places=4)
        for suffix, source in (("reference", self.reference), ("input", self.reference),
                               ("hypothesis", self.noisy)):
            self.assertEqual((self.root / f"result.{suffix}.wav").read_bytes(), source.read_bytes())
        report = json.loads((self.root / "result.audio-quality.json").read_text())
        self.assertEqual(report["metrics"], result["audio_quality"])

    def test_missing_last_output_cannot_reuse_warmup_or_earlier_run(self):
        # A previous dispatch's artifact must never become this dispatch's score.
        stale = self.root / "result.hypothesis.wav"
        shutil.copyfile(self.reference, stale)
        result = self.run_driver(mode="last-missing")
        score = result["audio_quality"]["sisdr"]
        self.assertIsNone(score["value"])
        self.assertNotEqual(score["status"], "ok")
        self.assertTrue(score.get("reason"))
        self.assertFalse(stale.exists(), "stale hypothesis artifact must be cleared")

    def test_invalid_or_absent_wav_is_diagnostic_without_losing_perf(self):
        for mode in ("missing", "malformed"):
            with self.subTest(mode=mode):
                result = self.run_driver(mode=mode)
                metric = result["audio_quality"]["sisdr"]
                self.assertIsNone(metric["value"])
                self.assertNotEqual(metric["status"], "ok")
                self.assertTrue(metric.get("reason"))
                self.assertTrue((self.root / "result.audio-quality.json").exists())

    def test_single_metric_kind_is_accepted(self):
        for kind in ("sisdr", "stoi"):
            with self.subTest(kind=kind):
                result = self.run_driver(kind=kind)
                self.assertEqual(result["correctness_kind"], "audio")
                self.assertEqual(set(result["audio_quality"]), {kind})
                if kind == "sisdr":
                    self.assertEqual(result["audio_quality"][kind]["status"], "ok")

    def test_optional_metric_failures_do_not_hide_sisdr(self):
        result = self.run_driver(metrics=["sisdr", "stoi"],
                                 without_optional_dependencies=True)
        self.assertEqual(set(result["audio_quality"]), {"sisdr", "stoi"})
        self.assertEqual(result["audio_quality"]["sisdr"]["status"], "ok")
        for name, status in (("stoi", "unavailable"),):
            metric = result["audio_quality"][name]
            self.assertEqual(metric["status"], status)
            self.assertIsNone(metric["value"])
            self.assertTrue(metric.get("reason"))

    def test_degradation_is_reproducible_and_keeps_clean_reference(self):
        config = {"snr_db": 10, "seed": 42}
        self.run_driver(degradation=config)
        first = (self.root / "result.input.wav").read_bytes()
        self.assertNotEqual(first, self.reference.read_bytes())
        self.log.unlink()
        self.run_driver(degradation=config)
        self.assertEqual(first, (self.root / "result.input.wav").read_bytes())
        self.assertEqual((self.root / "result.reference.wav").read_bytes(), self.reference.read_bytes())

    def test_degradation_attenuates_reference_and_input_together(self):
        loud = [3 * value for value in self.ref_samples]
        write_wav(self.reference, loud)
        self.run_driver(degradation={"snr_db": 0, "seed": 42})
        reference = read_samples(self.root / "result.reference.wav")
        noisy = read_samples(self.root / "result.input.wav")
        self.assertEqual(len(reference), len(loud))
        self.assertEqual(len(noisy), len(loud))
        self.assertLess(max(map(abs, reference)), max(map(abs, loud)))
        self.assertLessEqual(max(map(abs, noisy)), 32767)
        measured_snr = 10 * math.log10(sum(v * v for v in reference) /
            sum((n - r) ** 2 for n, r in zip(noisy, reference)))
        self.assertAlmostEqual(measured_snr, 0, places=2,
                               msg="common attenuation must preserve the requested SNR")


class AudioSummaryTests(unittest.TestCase):
    def test_summary_units_and_unavailable_metrics(self):
        spec = importlib.util.spec_from_file_location("benchmark_summary", HERE / "summarize.py")
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        rendered = summary.fmt_audio_quality({
            "sisdr": {"status": "ok", "value": -2.5, "unit": "dB"},
            "stoi": {"status": "ok", "value": 0.8125, "unit": "ratio"},
        })
        self.assertIn("SI-SDR -2.50 dB", rendered)
        self.assertIn("STOI 81.25%", rendered)
        unavailable = summary.fmt_audio_quality({
            "stoi": {"status": "unavailable", "value": None, "reason": "dependency missing"}})
        self.assertIn("unavailable", unavailable)
        self.assertNotIn("0.00%", unavailable)


if __name__ == "__main__":
    unittest.main()
