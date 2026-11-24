#!/usr/bin/env python3
"""Run the native probe with generated Core ML models, without ASR checkpoints."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

from test_route_experiments import (
    BENCH, CAPACITY, CHANNELS, FIXTURE_NAME, HARNESS, ROUTE_OFFSETS,
    VALID_FRAMES, coremltools as ct, np, torch, write_fixture,
)

SUPPORTED_MAC = platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15
HAVE_RUNTIME = SUPPORTED_MAC and hasattr(ct, "convert") and shutil.which("xcrun")
DTYPES = {"fp16": np.float16, "fp32": np.float32}


class ToyEncoder(torch.nn.Module):
    def __init__(self, route):
        super().__init__()
        self.route = route

    def forward(self, mel, mel_mask, pad_mask, key_bias):
        features = mel * mel_mask * pad_mask + key_bias.permute(0, 2, 3, 1)
        return features + ROUTE_OFFSETS[self.route]


class ToyRoutedEncoder(torch.nn.Module):
    def forward(self, mel, mel_mask, pad_mask, key_bias, adapter_scale):
        features = mel * mel_mask * pad_mask + key_bias.permute(0, 2, 3, 1)
        return (features + ROUTE_OFFSETS["en"] + adapter_scale *
                (ROUTE_OFFSETS["other"] - ROUTE_OFFSETS["en"]),
                features + ROUTE_OFFSETS["gate"])


def compile_model(model, path, fixture, dtype, unified=False):
    names = list(HARNESS.INPUT_NAMES)
    example = tuple(torch.from_numpy(fixture[name].astype(np.float32)) for name in names)
    outputs = [ct.TensorType(name="encoder_out", dtype=dtype)]
    if unified:
        names.append("adapter_scale")
        example += (torch.zeros(1, 1, 1, 1),)
        outputs.append(ct.TensorType(name="gate_features", dtype=dtype))
    converted = ct.convert(torch.jit.trace(model.eval(), example, check_trace=False),
        inputs=[ct.TensorType(name=name, shape=value.shape, dtype=dtype) for name, value in zip(names, example)],
        outputs=outputs, minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16 if dtype == np.float16 else ct.precision.FLOAT32,
        skip_model_load=True)
    converted.save(str(path))
    shutil.copytree(ct.utils.compile_model(str(path)), path.with_suffix(".mlmodelc"))


def build_models(directory, fixture, dtype):
    directory.mkdir()
    descriptor = ct.utils.MultiFunctionDescriptor()
    for route in HARNESS.ROUTES:
        path = directory / (route + ".mlpackage")
        compile_model(ToyEncoder(route), path, fixture, dtype)
        descriptor.add_function(str(path), "main", route)
    descriptor.default_function_name = "en"
    shared = directory / "shared.mlpackage"
    ct.utils.save_multifunction(descriptor, str(shared))
    shutil.copytree(ct.utils.compile_model(str(shared)), shared.with_suffix(".mlmodelc"))
    compile_model(ToyRoutedEncoder(), directory / "unified.mlpackage", fixture, dtype, unified=True)


@unittest.skipUnless(HAVE_RUNTIME, "Requires macOS 15+, coremltools and Xcode tools")
class RouteProbeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.fixtures = cls.root / "fixtures"
        cls.fixtures.mkdir()
        cls.fixture = write_fixture(cls.fixtures)
        cls.phone_fixtures = cls.root / "phone-fixtures"
        HARNESS.main(["phone-fixtures", "--fixtures", str(cls.fixtures), "--output", str(cls.phone_fixtures)])
        cls.binary = cls.root / "route-probe"
        subprocess.run(["xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-fobjc-arc",
                        "-framework", "CoreML", "-framework", "Foundation", str(BENCH / "route_probe.mm"),
                        "-o", str(cls.binary)], check=True, capture_output=True, text=True)
        for precision, dtype in DTYPES.items():
            build_models(cls.root / precision, cls.fixture, dtype)

    def run_probe(self, precision, kind, fixtures=None):
        label = precision + "-" + kind
        result = subprocess.run([str(self.binary), str(self.root / precision), kind, label,
                                 str(fixtures or self.phone_fixtures)],
                                env={**os.environ, "QVAC_ROUTE_OUTPUT_DIR": str(self.root / "outputs")},
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "outputs" / label
        return output, json.loads((output / "results.json").read_text())

    def test_prediction_modes_serialize_valid_rows_and_select_named_functions_and_route_inputs(self):
        for precision in DTYPES:
            for kind in ("separate", "shared", "unified"):
                with self.subTest(precision=precision, kind=kind):
                    output, data = self.run_probe(precision, kind)
                    self.assertEqual([record["route"] for record in data["records"]], list(HARNESS.ROUTES))
                    for record in data["records"]:
                        route = record["route"]
                        raw = output / f"{route}-{FIXTURE_NAME}.bin"
                        self.assertEqual(raw.stat().st_size, VALID_FRAMES * CHANNELS * np.dtype(np.float32).itemsize)
                        got = np.fromfile(raw, np.float32).reshape(VALID_FRAMES, CHANNELS)
                        np.testing.assert_array_equal(got, self.fixture[route])
                        self.assertEqual(record["predictions"][0]["relative_l2_to_fp32"], 0)
                        self.assertGreaterEqual(record["load_ms"], 0)
                        self.assertIn("footprint_bytes", record["memory"])

    def test_loading_modes_need_no_fixture_manifest_and_load_only_requested_routes(self):
        routes = {"load-eager": ["en", "other", "gate"], "load-en": ["en"],
                  "load-other": ["other"], "load-auto-en": ["gate", "en"],
                  "load-auto-other": ["gate", "other"]}
        for kind, expected in routes.items():
            with self.subTest(kind=kind):
                output, data = self.run_probe("fp16", kind, self.root / "missing-fixtures")
                self.assertEqual([record["route"] for record in data["records"]], expected)
                self.assertEqual(list(output.glob("*.bin")), [])
                self.assertTrue(all("predictions" not in record for record in data["records"]))
                self.assertTrue(all(record["load_ms"] >= 0 for record in data["records"]))

    def test_truncated_input_is_rejected_before_prediction(self):
        broken = self.root / "broken-fixtures"
        shutil.copytree(self.phone_fixtures, broken)
        (broken / FIXTURE_NAME / "mel.bin").write_bytes(b"\x00")
        result = subprocess.run([str(self.binary), str(self.root / "fp16"), "separate", "broken", str(broken)],
                                env={**os.environ, "QVAC_ROUTE_OUTPUT_DIR": str(self.root / "outputs")},
                                capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid input fixture", result.stderr)

    def test_invalid_manifests_fail_before_loading_models(self):
        cases = (
            (b'{"fixtures":[', "Invalid fixture manifest JSON"),
            (b'[]', "Fixture manifest root must be an object"),
            (b'{}', "nonempty fixtures array"),
            (b'{"fixtures":[]}', "nonempty fixtures array"),
            (b'{"fixtures":{}}', "nonempty fixtures array"),
            (b'{"fixtures":null}', "nonempty fixtures array"),
        )
        for index, (contents, message) in enumerate(cases):
            fixtures = self.root / f"invalid-manifest-{index}"
            fixtures.mkdir()
            (fixtures / "manifest.json").write_bytes(contents)
            for models in ("fp16", "missing-models"):
                with self.subTest(contents=contents, models=models):
                    label = f"invalid-manifest-{index}-{models}"
                    result = subprocess.run([str(self.binary), str(self.root / models), "separate", label, str(fixtures)],
                        env={**os.environ, "QVAC_ROUTE_OUTPUT_DIR": str(self.root / "outputs")},
                        capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse((self.root / "outputs" / label / "results.json").exists())


if __name__ == "__main__":
    if not HAVE_RUNTIME:
        print("SKIP: native route probe requires macOS 15+, coremltools and Xcode tools")
        raise SystemExit(77)
    unittest.main()
