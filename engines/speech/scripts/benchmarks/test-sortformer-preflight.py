#!/usr/bin/env python3
import importlib.util
import json
import io
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("preflight", HERE / "sortformer-preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


PREPARE_SPEC = importlib.util.spec_from_file_location("prepare", HERE / "prepare-sortformer-preflight.py")
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


class PreparationTests(unittest.TestCase):
    def test_wrong_checkpoint_stops_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                PREPARE, "urlopen", return_value=io.BytesIO(b"incorrect")), patch.object(
                PREPARE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                PREPARE.prepare(Path(directory))
            run.assert_not_called()

    def test_verified_checkpoint_conversion_and_provenance(self):
        payload = b"verified checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def convert(command, **kwargs):
                self.assertTrue(kwargs["check"])
                checkpoint = Path(command[command.index("--ckpt") + 1])
                self.assertEqual(checkpoint.name, PREPARE.STEM + ".nemo")
                self.assertEqual(checkpoint.read_bytes(), payload)
                self.assertEqual(command[-4:], ["--quant", "q8_0", "--head", "sortformer"])
                Path(command[command.index("--out") + 1]).write_bytes(b"gguf")
            with patch.object(PREPARE, "CHECKPOINT_SHA256", digest), patch.object(
                    PREPARE, "urlopen", return_value=io.BytesIO(payload)) as download, patch.object(
                    PREPARE.subprocess, "run", side_effect=convert):
                PREPARE.prepare(root)
            self.assertIn(PREPARE.REVISION, download.call_args.args[0])
            report = json.loads((root / "results/model-provenance.json").read_text())
            self.assertEqual(report["checkpoint_sha256"], digest)
            self.assertEqual(report["gguf_sha256"], hashlib.sha256(b"gguf").hexdigest())

    def test_converter_failure_retains_provenance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                PREPARE, "download_checkpoint"), patch.object(
                PREPARE.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "converter")):
            root = Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                PREPARE.prepare(root)
            self.assertTrue((root / "results/model-provenance.json").exists())

    def test_missing_converter_output_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                PREPARE, "download_checkpoint"), patch.object(PREPARE.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "nonempty GGUF"):
                PREPARE.prepare(Path(directory))


class FullNetworkTests(unittest.TestCase):
    def result(self, probabilities=None, coreml=True):
        return {"coreml": coreml, "frames": 1, "speakers": 2, "stride": 0.08,
                "probabilities": [0.5, 0.25] if probabilities is None else probabilities,
                "segments": [{"speaker": 0, "start": 0, "end": 4}]}

    def test_thresholds_match_existing_gate(self):
        source = (HERE.parents[1] / "parakeet/test/test_sortformer_diarizer_coreml_parity.cpp").read_text()
        self.assertIn(f"kMaximumProbabilityError = {PREFLIGHT.MAXIMUM_ERROR};", source)
        self.assertIn(f"kMaximumRelativeError = {PREFLIGHT.RELATIVE_L2};", source)

    def test_probabilities_and_both_thresholds(self):
        native = self.result([1.0, 1.0], False)
        self.assertTrue(PREFLIGHT.compare(native, native)["passed"])
        self.assertFalse(PREFLIGHT.compare(self.result([0.99, 1.0]), native)["passed"])
        actual = self.result([1.0] * 1000)
        native = self.result([1.0] * 1000, False)
        actual["probabilities"][0] = 0.9781
        self.assertFalse(PREFLIGHT.compare(actual, native)["passed"])
        metric = PREFLIGHT.compare(self.result([0.5, 0.5]), self.result([1.0, 1.0], False))
        self.assertAlmostEqual(metric["maximum_probability_error"], 0.5)
        self.assertAlmostEqual(metric["relative_l2"], 0.5)

    def test_rejects_fallback_empty_invalid_and_mismatched_output(self):
        for result in (self.result(coreml=False), self.result([]), self.result([0.5]),
                       self.result([float("nan"), 0.5]), self.result([float("inf"), 0.5]),
                       self.result([-0.1, 0.5])):
            with self.subTest(result=result), self.assertRaises(ValueError):
                PREFLIGHT.validate_result(result, "diarizer")
        native = self.result(coreml=False)
        native["stride"] = 0.1
        with self.assertRaises(ValueError):
            PREFLIGHT.compare(self.result(), native)

    def test_environment_and_asset_isolation(self):
        with patch.dict(os.environ, {"PARAKEET_COREML_DISABLE": "1", "PARAKEET_COREML_ONLY": "1",
                                     "PARAKEET_COREML_STRICT": "0"}):
            self.assertEqual(PREFLIGHT.arm_environment("diarizer")["PARAKEET_COREML_STRICT"], "1")
            self.assertNotIn("PARAKEET_COREML_STRICT", PREFLIGHT.arm_environment("metal"))
            self.assertNotIn("PARAKEET_COREML_DISABLE", PREFLIGHT.arm_environment("diarizer"))
            self.assertNotIn("PARAKEET_COREML_ONLY", PREFLIGHT.arm_environment("metal"))
            self.assertEqual(PREFLIGHT.arm_environment("metal")["PARAKEET_COREML_DISABLE"], "1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing model"):
                PREFLIGHT.stage_arm(root, root / "missing", "diarizer")
            (root / PREFLIGHT.MODEL_NAME).write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "incomplete bundle"):
                PREFLIGHT.stage_arm(root, root / "incomplete", "diarizer")
            bundle = root / (PREFLIGHT.STEM + f"-diarizer-{PREFLIGHT.CAPACITY}.mlmodelc")
            (bundle / "weights").mkdir(parents=True)
            (bundle / "weights/weight.bin").write_bytes(b"weights")
            (bundle / "model.mil").write_text("model")
            PREFLIGHT.stage_arm(root, root / "metal", "metal")
            PREFLIGHT.stage_arm(root, root / "diarizer", "diarizer")
            self.assertEqual(len(list((root / "metal").iterdir())), 1)
            self.assertEqual(len(list((root / "diarizer").iterdir())), 2)

    def test_complete_calls_preserve_staged_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(binary=root / "binary", samples=root, assets=root, out=root)
            model = root / "staged" / PREFLIGHT.MODEL_NAME
            def invoke(command, **kwargs):
                self.assertEqual(command[1], str(model))
                self.assertEqual(command[2], str(root / "abcba.wav"))
                self.assertEqual(len(command), 4)
                for call in range(1, 3):
                    Path(f"{command[3]}-{call}.json").write_text(json.dumps(self.result()))
            with patch.object(PREFLIGHT, "stage_arm", return_value=model), patch.object(
                    PREFLIGHT.subprocess, "run", side_effect=invoke):
                self.assertEqual(len(PREFLIGHT.run_arm(args, root, "diarizer")), 2)
            (root / "diarizer-2.json").unlink()
            with self.assertRaises(FileNotFoundError):
                PREFLIGHT.load_calls(root / "diarizer", "diarizer")
            self.assertTrue((root / "diarizer-1.json").exists())

    def test_failed_gate_keeps_summary_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"binary")
            report = {"passed": False, "probability_comparisons": [
                {"maximum_probability_error": 0.0219, "relative_l2": 0.0135, "passed": False}]}
            arguments = ["preflight", "--binary", str(binary), "--assets", str(root),
                         "--samples", str(root), "--out", str(root / "out")]
            with patch("sys.argv", arguments), patch.object(PREFLIGHT, "run_comparison", return_value=report):
                self.assertEqual(PREFLIGHT.main(), 1)
            self.assertEqual(json.loads((root / "out/summary.json").read_text()), report)

    def test_both_calls_gate_and_der_collar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abcba.rttm").write_text("".join(
                f"SPEAKER sample 1 {index * 4} 4 <NA> <NA> {speaker} <NA> <NA>\n"
                for index, speaker in enumerate("ABC")))
            args = SimpleNamespace(samples=root)
            def run_arm(args, staging, arm):
                return [self.result(coreml=arm == "diarizer"),
                        self.result([0.4, 0.25] if arm == "diarizer" else None, arm == "diarizer")]
            with patch.object(PREFLIGHT, "run_arm", side_effect=run_arm):
                report = PREFLIGHT.run_comparison(args, root)
            self.assertFalse(report["passed"])
            self.assertTrue(report["probability_comparisons"][0]["passed"])
            self.assertEqual(report["der"]["metal"][0]["collar_ms"], 250)
            self.assertIsNone(report["der_gate"])


class DriverMedianTests(unittest.TestCase):
    def run_driver(self, count, malformed=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build/bin").mkdir(parents=True)
            reference = root / "reference.rttm"
            reference.write_text("SPEAKER sample 1 0 4 <NA> <NA> A <NA> <NA>\n")
            binary = root / "build/bin/probe"
            binary.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\n"
                              f"p=Path({str(root / 'count')!r})\n"
                              "n=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\n"
                              f"if n < {count}: print('{{\"speaker\":0,\"start\":0,\"end\":4}}')\n"
                              + ("else: print('{\"speaker\":0,\"start\":\"invalid\",\"end\":4}')\n" if malformed else ""))
            binary.chmod(0o755)
            spec = {"probe": {"bench_kind": "time-wrapped", "binary": "bin/probe", "cmake_target": "unused",
                              "models": [], "args": [], "audio_duration_seconds": 4,
                              "correctness": {"kind": "der", "reference": str(reference), "collar_ms": 0}}}
            (root / "spec.json").write_text(json.dumps(spec))
            environment = {**os.environ, "BENCH_FAMILIES_JSON": str(root / "spec.json")}
            environment.pop("MODEL_S3_BUCKET", None)
            subprocess.run(["bash", str(HERE / "run-family.sh"), "--family", "probe", "--runs", str(count),
                            "--warmup", "0", "--build-dir", str(root / "build"), "--models-root", str(root / "models"),
                            "--out", str(root / "result.json")], env=environment, check=True, capture_output=True)
            hypothesis = root / f"result.run-{count}.hypothesis.jsonl"
            self.assertTrue(hypothesis.is_file())
            if malformed:
                self.assertIn("invalid", hypothesis.read_text())
            return json.loads((root / "result.json").read_text()), json.loads((root / "result.der-runs.json").read_text())

    def test_odd_median_uses_every_run(self):
        result, report = self.run_driver(3)
        self.assertEqual(result["der_median"], 0)
        self.assertEqual([run["der"] for run in report["runs"]], [0, 0, 1])

    def test_even_median(self):
        result, report = self.run_driver(2)
        self.assertEqual(result["der_median"], 0.5)
        self.assertTrue(report["complete"])

    def test_scoring_failure_is_not_a_partial_median(self):
        result, report = self.run_driver(3, malformed=True)
        self.assertIsNone(result["der_median"])
        self.assertFalse(report["complete"])
        self.assertIsNone(report["runs"][-1]["der"])


if __name__ == "__main__":
    unittest.main()
