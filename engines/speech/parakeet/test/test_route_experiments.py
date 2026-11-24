#!/usr/bin/env python3
"""Fixture-based regression tests for route measurement and reporting."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    raise SystemExit(0)

try:
    import coremltools
except ModuleNotFoundError as error:
    if error.name != "coremltools":
        raise
    coremltools = ModuleType("coremltools")
    sys.modules["coremltools"] = coremltools

BENCH = Path(__file__).parents[1] / "examples/ios-coreml-bench"
SPEC = importlib.util.spec_from_file_location("route_experiments", BENCH / "route_experiments.py")
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
CHANNELS = 2
CAPACITY = 3
VALID_FRAMES = 2
FIXTURE_NAME = "short"
POSITIVE_GATE = np.array([[1, 0], [1, 0]], dtype=np.float32)
NEGATIVE_GATE = POSITIVE_GATE[:, ::-1].copy()
ROUTE_OFFSETS = {"en": 10, "other": 20, "gate": 30}


def write_gate_weights(directory):
    weights = {
        "pool.proj.0.weight": np.zeros((1, CHANNELS)), "pool.proj.0.bias": np.zeros(1),
        "pool.proj.2.weight": np.zeros((1, 1)), "pool.proj.2.bias": np.zeros(1),
        "norm.weight": np.ones(CHANNELS * 2), "norm.bias": np.zeros(CHANNELS * 2),
        "mlp.0.weight": np.array([[1, -1, 0, 0]]), "mlp.0.bias": np.zeros(1),
        "mlp.3.weight": np.array([[3]]), "mlp.3.bias": np.array([-1]),
    }
    np.savez(directory / "gate_weights.npz", threshold=np.float32(0.5),
             **{"eu.gate." + name: value.astype(np.float32) for name, value in weights.items()})


def write_fixture(directory):
    mel = np.array([[[[0.5, -1.5, 9]], [[2, 3.5, 8]]]], dtype=np.float16)
    mask = np.array([[[[1, 1, 0]]]], dtype=np.float16)
    fixture = {"mel": mel, "mel_mask": mask, "pad_mask": mask,
               "key_bias": np.array([0, 0, -10000], dtype=np.float16).reshape(1, CAPACITY, 1, 1),
               "valid_encoder_frames": VALID_FRAMES}
    for route, offset in ROUTE_OFFSETS.items():
        fixture[route] = mel[0, :, 0, :VALID_FRAMES].T.astype(np.float32) + offset
    np.savez(directory / (FIXTURE_NAME + ".npz"), **fixture)
    return fixture


def write_native_result(root, precision, kind, values, route="gate"):
    directory = root / f"{precision}-{kind}"
    directory.mkdir()
    values.astype(np.float32).tofile(directory / f"{route}-{FIXTURE_NAME}.bin")
    record = {"kind": kind, "route": route, "load_ms": 2.5,
              "predictions": [{"fixture": FIXTURE_NAME, "warm_median_ms": 1.25}]}
    data = {"model_directory": precision, "records": [record], "memory": {"thermal": 0}}
    (directory / "results.json").write_text(json.dumps(data))
    return directory


class RouteReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.fixture = write_fixture(self.fixtures)
        write_gate_weights(self.fixtures)

    def test_prepare_rejects_duplicate_stems_before_loading_weights_or_writing_fixtures(self):
        wavs = [self.root / "english/sample.wav", self.root / "french/sample.wav"]
        original = self.fixtures / "sample.npz"
        original.write_bytes(b"existing fixture")
        before = {path.name: path.read_bytes() for path in self.fixtures.iterdir()}
        with mock.patch.object(HARNESS, "exporter") as loader, \
                mock.patch.object(HARNESS.subprocess, "run") as runner:
            with self.assertRaisesRegex(ValueError, "Duplicate WAV stem 'sample'") as raised:
                HARNESS.prepare(SimpleNamespace(wav=wavs, output=self.fixtures))
            loader.assert_not_called()
            runner.assert_not_called()
        self.assertIn(str(wavs[0]), str(raised.exception))
        self.assertIn(str(wavs[1]), str(raised.exception))
        self.assertEqual({path.name: path.read_bytes() for path in self.fixtures.iterdir()}, before)

    def test_phone_serialization_preserves_shapes_valid_length_and_float32_values(self):
        output = self.root / "phone"
        HARNESS.main(["phone-fixtures", "--fixtures", str(self.fixtures), "--output", str(output)])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["fixtures"]), 1)
        entry = manifest["fixtures"][0]
        self.assertEqual(entry["name"], FIXTURE_NAME)
        self.assertEqual(entry["valid_encoder_frames"], VALID_FRAMES)
        self.assertEqual(entry["shapes"], {name: list(self.fixture[name].shape) for name in HARNESS.INPUT_NAMES})
        self.assertFalse((output / "gate_weights").exists())
        for name in (*HARNESS.INPUT_NAMES, *HARNESS.ROUTES):
            with self.subTest(name=name):
                path = output / FIXTURE_NAME / (name + ".bin")
                self.assertEqual(path.stat().st_size, self.fixture[name].size * np.dtype(np.float32).itemsize)
                np.testing.assert_array_equal(np.fromfile(path, np.float32), self.fixture[name].astype(np.float32).ravel())

    def test_native_report_matches_baselines_by_precision_and_reports_changed_gate_decisions(self):
        results = self.root / "results"
        results.mkdir()
        with np.load(self.fixtures / (FIXTURE_NAME + ".npz")) as fixture:
            values = dict(fixture)
        values["gate"] = POSITIVE_GATE
        np.savez(self.fixtures / (FIXTURE_NAME + ".npz"), **values)
        # Candidates precede baselines, and the two precisions have opposite gate decisions.
        for precision, baseline in (("sixbit", NEGATIVE_GATE), ("fp16", POSITIVE_GATE)):
            write_native_result(results, precision, "shared", POSITIVE_GATE)
            write_native_result(results, precision, "unified", baseline)
            write_native_result(results, precision, "separate", baseline)
        loading = results / "loading"
        loading.mkdir()
        load_record = {"kind": "load-en", "route": "en", "load_ms": 3.5}
        (loading / "results.json").write_text(json.dumps({"model_directory": "fp16", "records": [load_record]}))
        output = self.root / "summary"
        HARNESS.main(["summarize-native", "--native-results", str(results),
                      "--fixtures", str(self.fixtures), "--output", str(output)])
        report = json.loads((output / "native-summary.json").read_text())
        rows = {(data["model_directory"], record["kind"]): record
                for data in report for record in data["records"]}
        positive = rows["fp16", "unified"]["predictions"][0]
        negative = rows["sixbit", "unified"]["predictions"][0]
        changed = rows["sixbit", "shared"]["predictions"][0]
        self.assertTrue(positive["identical_to_separate"])
        self.assertTrue(negative["identical_to_separate"])
        self.assertFalse(changed["identical_to_separate"])
        self.assertEqual(changed["max_absolute_to_separate"], 1.0)
        self.assertEqual(negative["max_absolute_to_separate"], 0.0)
        self.assertGreater(positive["probability"], 0.9)
        self.assertLess(negative["probability"], 0.3)
        self.assertTrue(positive["decision_matches"])
        self.assertFalse(negative["decision_matches"])
        self.assertTrue(changed["decision_matches"])
        self.assertEqual(negative["threshold"], 0.5)
        self.assertGreater(negative["reference_probability"], 0.9)
        self.assertEqual(rows["fp16", "load-en"], load_record)
        original = json.loads((results / "sixbit-unified/results.json").read_text())
        self.assertNotIn("decision_matches", original["records"][0]["predictions"][0])

    def test_empty_reports_and_truncated_prediction_files_fail(self):
        with self.assertRaisesRegex(ValueError, "No native probe"):
            HARNESS.summarize_native(SimpleNamespace(native_results=self.root, fixtures=self.fixtures))
        path = write_native_result(self.root, "fp16", "separate", POSITIVE_GATE)
        (path / f"gate-{FIXTURE_NAME}.bin").write_bytes(b"\x00" * np.dtype(np.float32).itemsize)
        with self.assertRaises(ValueError):
            HARNESS.phone_prediction(path / "results.json", "gate", FIXTURE_NAME, self.fixtures)

    def test_compiled_comparison_uses_valid_rows_gate_output_and_runtime_route_scale(self):
        baseline_dir = self.root / "baselines"
        baseline_dir.mkdir()
        output = np.full((1, CHANNELS, 1, CAPACITY), 999, np.float16)
        gate = output.copy()
        gate[0, :, 0, :VALID_FRAMES] = POSITIVE_GATE.T
        predictor = mock.Mock()
        predictor.predict.return_value = {"encoder_out": output, "gate_features": gate}
        weights = dict(np.load(self.fixtures / "gate_weights.npz"))
        with mock.patch.object(HARNESS, "load_model", return_value=(predictor, 2.5)), \
                mock.patch.object(HARNESS, "placement", return_value={}):
            row = HARNESS.compare_variant(Path("en.mlmodelc"), None, "en",
                [self.fixtures / (FIXTURE_NAME + ".npz")], False, weights, baseline_dir)
            saved = np.load(baseline_dir / f"en-{FIXTURE_NAME}.npy")
            self.assertEqual(saved.shape, (VALID_FRAMES, CHANNELS))
            self.assertEqual(row["load_ms"], 2.5)
            np.save(baseline_dir / f"gate-{FIXTURE_NAME}.npy", POSITIVE_GATE)
            row = HARNESS.compare_variant(Path("unified.mlmodelc"), None, "gate",
                [self.fixtures / (FIXTURE_NAME + ".npz")], True, weights, baseline_dir)
            self.assertTrue(row["predictions"][0]["identical_to_separate"])
            np.testing.assert_array_equal(predictor.predict.call_args.args[0]["adapter_scale"],
                                          np.zeros((1, 1, 1, 1), np.float16))
            np.save(baseline_dir / f"other-{FIXTURE_NAME}.npy", saved)
            HARNESS.compare_variant(Path("unified.mlmodelc"), None, "other",
                [self.fixtures / (FIXTURE_NAME + ".npz")], True, weights, baseline_dir)
            scale = predictor.predict.call_args.args[0]["adapter_scale"]
            self.assertEqual(scale.dtype, np.float16)
            np.testing.assert_array_equal(scale, np.ones((1, 1, 1, 1), np.float16))
            output[0, 0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                HARNESS.compare_variant(Path("en.mlmodelc"), None, "en",
                    [self.fixtures / (FIXTURE_NAME + ".npz")], False, weights, baseline_dir)

    def test_loading_only_uses_requested_routes_and_serializes_total_for_both_layouts(self):
        for layout, names in (("experiment", ["gate", "other"]),
                              ("runtime", ["transcribepsy-250m-eu-encoder-gate", "transcribepsy-250m-eu-encoder-other"])):
            with self.subTest(layout=layout):
                output = self.root / layout
                output.mkdir()
                with mock.patch.object(HARNESS, "load_model", side_effect=[(object(), 1.25), (object(), 2.5)]) as loader:
                    HARNESS.measure_loading(SimpleNamespace(models=self.root, routes=["gate", "other"],
                                                             model_layout=layout, output=output))
                self.assertEqual([call.args[0].name for call in loader.call_args_list],
                                 [name + ".mlmodelc" for name in names])
                report = json.loads((output / "loading.json").read_text())
                self.assertEqual([row["route"] for row in report["routes"]], ["gate", "other"])
                self.assertEqual(report["total_load_ms"], 3.75)


if __name__ == "__main__":
    unittest.main()
