"""Sidecar preset contract checks that do not import ML dependencies or load weights."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import export_sidecar


CONFIGS = Path(__file__).resolve().parent / "configs"


def preset(**changes) -> dict:
    base = json.loads((CONFIGS / "qwen3.8-27b-64-dual.json").read_text())
    base.update(changes)
    return base


class TestPresetContract(unittest.TestCase):
    def check(self, value: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preset.json"
            path.write_text(json.dumps(value))
            return export_sidecar.read_preset(path)

    def test_committed_presets(self):
        single = export_sidecar.read_preset(CONFIGS / "qwen3.8-27b-24.json")
        self.assertEqual(export_sidecar.partitions(single), [("ffn-512", "manifest.json", 11264, 6144)])
        dual = export_sidecar.read_preset(CONFIGS / "qwen3.8-27b-64-dual.json")
        self.assertEqual(export_sidecar.partitions(dual), [("ffn-512", "manifest.json", 8192, 4608), ("ffn-512-1", "manifest-1.json", 12800, 4608)])
        self.assertEqual(dual["layers"], list(range(64)))

    def test_partitions_cover_suffix(self):
        for parts in (1, 2):
            value = self.check(preset(ane_partitions=parts))
            ranges = export_sidecar.partitions(value)
            self.assertEqual(ranges[0][2], value["channel_start"])
            self.assertEqual(sum(r[3] for r in ranges), value["channel_count"])
            for current, following in zip(ranges, ranges[1:]):
                self.assertEqual(current[2] + current[3], following[2])

    def test_rejects_invalid_presets(self):
        cases = {
            "channel_start": preset(channel_start=8000, channel_count=9408),
            "equal intermediate": preset(channel_count=9000),
            "ane_partitions": preset(ane_partitions=3),
            "multiple of 256": preset(channel_start=8448, channel_count=8960),
            "distinct": preset(layers=[1, 1, 2]),
            "within model_layers": preset(layers=[0, 64]),
            "fp16_layers": preset(layers=[40, 41], fp16_layers=[31]),
            "tokens": preset(tokens=256),
            "palette": preset(palette_bits=4),
        }
        for message, value in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.check(value)

    def test_existing_destination_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, "destination already exists"):
                export_sidecar.export(Path("missing.gguf"), CONFIGS / "qwen3.8-27b-64-dual.json", destination, 1, False)

    def test_preset_is_not_modified(self):
        value = preset()
        original = copy.deepcopy(value)
        export_sidecar.partitions(self.check(value))
        self.assertEqual(value, original)


if __name__ == "__main__":
    unittest.main()
