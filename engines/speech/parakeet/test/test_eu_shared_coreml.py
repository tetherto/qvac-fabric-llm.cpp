#!/usr/bin/env python3
"""Exercise the production merger and native encoder wrapper with generated routes."""
import importlib.util
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

try:
    import coremltools as ct
    import numpy as np
    import torch
except ImportError:
    raise SystemExit(77)

ENGINE = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("merge_eu", ENGINE / "scripts/merge-transcribepsy-eu-coreml.py")
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)
CAPACITY = 2001
CHANNELS = 8
OFFSETS = {"en": 1, "other": 2, "gate": 3}


class ToyRoute(torch.nn.Module):
    def __init__(self, route):
        super().__init__()
        self.offset = OFFSETS[route]

    def forward(self, mel, mel_mask, pad_mask, key_bias):
        sampled = (mel * mel_mask)[:, :, :, ::8]
        return (sampled + self.offset) * pad_mask + key_bias.permute(0, 2, 3, 1)


def export_route(path, route):
    rows = (CAPACITY + 7) // 8
    example = (torch.zeros(1, CHANNELS, 1, CAPACITY), torch.ones(1, 1, 1, CAPACITY),
               torch.ones(1, 1, 1, rows), torch.zeros(1, rows, 1, 1))
    model = ct.convert(torch.jit.trace(ToyRoute(route), example),
        inputs=[ct.TensorType(name=name, shape=value.shape, dtype=np.float16)
                for name, value in zip(("mel", "mel_mask", "pad_mask", "key_bias"), example)],
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18, skip_model_load=True)
    model.user_defined_metadata.update({"parakeet.model_variant": MERGER.VARIANT,
        "parakeet.eu.route": route, "parakeet.encoder.precision": "fp16",
        "parakeet.encoder.n_layers": "1" if route == "gate" else "2"})
    model.save(str(path))


SUPPORTED = platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15


@unittest.skipUnless(SUPPORTED, "requires macOS 15+")
class SharedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.sources = {route: cls.root / (route + ".mlpackage") for route in MERGER.ROUTES}
        for route, path in cls.sources.items():
            export_route(path, route)
        cls.shared = cls.root / "shared.mlpackage"
        MERGER.merge_routes(cls.sources, cls.shared, cls.root)
        cls.binary = cls.root / "shared-io"
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-fobjc-arc",
            "-I" + str(ENGINE / "src"), str(ENGINE / "test/eu_shared_coreml_io.cpp"),
            str(ENGINE / "src/coreml/parakeet-encoder.mm"),
            str(ENGINE / "src/coreml/parakeet_coreml_shape.cpp"),
            "-framework", "CoreML", "-framework", "Foundation", "-o", str(cls.binary)],
            check=True, capture_output=True, text=True)

    def test_native_named_routes_masks_repeated_predictions_and_metadata_validation(self):
        for units in ("cpu", "ane"):
            result = subprocess.run([str(self.binary), str(self.shared.with_suffix(".mlmodelc")), units],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_merger_rejects_swapped_routes_and_overwrites(self):
        swapped = {**self.sources, "other": self.sources["en"]}
        with self.assertRaisesRegex(ValueError, "expected a full-EU"):
            MERGER.merge_routes(swapped, self.root / "bad.mlpackage")
        with self.assertRaisesRegex(ValueError, "already exists"):
            MERGER.merge_routes(self.sources, self.shared)

    def test_merger_rejects_mismatched_capacity_and_depth(self):
        original = ct.models.MLModel(str(self.sources["gate"]), skip_model_load=True)
        spec = original.get_spec()
        for feature in spec.description.input:
            if feature.name == "mel":
                feature.type.multiArrayType.shape[-1] += 8
        changed = ct.models.MLModel(spec, weights_dir=original.weights_dir, skip_model_load=True)
        path = self.root / "wrong-capacity.mlpackage"
        changed.save(str(path))
        with self.assertRaisesRegex(ValueError, "must match"):
            MERGER.merge_routes({**self.sources, "gate": path}, self.root / "bad.mlpackage")
        original.user_defined_metadata["parakeet.encoder.n_layers"] = "3"
        path = self.root / "wrong-depth.mlpackage"
        original.save(str(path))
        with self.assertRaisesRegex(ValueError, "Gate depth"):
            MERGER.merge_routes({**self.sources, "gate": path}, self.root / "bad.mlpackage")

    def test_merger_rejects_short_windows_and_incorrect_precision(self):
        _, inputs, outputs = MERGER.read_route(self.sources["en"], "en")
        short = {**inputs, "mel": ((1, CHANNELS, 1, CAPACITY - 1), inputs["mel"][1])}
        with self.assertRaisesRegex(ValueError, "at least"):
            MERGER.validate_shapes(self.sources["en"], short, outputs)
        wrong = {**outputs, "encoder_out": (outputs["encoder_out"][0],
                  ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32)}
        with self.assertRaisesRegex(ValueError, "must use FP16"):
            MERGER.validate_shapes(self.sources["en"], inputs, wrong)

    def test_experimental_bundle_does_not_pass_runtime_identity_checks(self):
        model = ct.models.MLModel(str(self.shared), skip_model_load=True)
        del model.user_defined_metadata["parakeet.eu.shared_version"]
        path = self.root / "unversioned.mlpackage"
        model.save(str(path))
        compiled = ct.utils.compile_model(str(path))
        result = subprocess.run([str(self.binary), compiled, "reject"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if not SUPPORTED:
        raise SystemExit(77)
    unittest.main()
