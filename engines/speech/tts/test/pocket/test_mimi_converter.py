#!/usr/bin/env python3
"""Mimi converter rejection checks; pass an upstream config YAML."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import yaml

CONFIG = yaml.safe_load(Path(sys.argv.pop(1)).read_text())
SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "convert-pocket-mimi-to-gguf.py"
spec = importlib.util.spec_from_file_location("mimi_converter", SOURCE)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class MimiConverterTests(unittest.TestCase):
    def test_unsupported_architecture_preserves_output(self):
        for fields in [dict(sample_rate=48000, frame_rate=25), dict(channels=2),
                       dict(frame_rate=25), dict(inner_dim=16), dict(outer_dim=256)]:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = copy.deepcopy(CONFIG)
                config["mimi"].update(fields)
                path = root / "config.yaml"
                path.write_text(yaml.safe_dump(config))
                output = root / "mimi.gguf"
                output.write_bytes(b"previous successful artifact")
                # Missing weights prove rejection happens before weight access.
                with self.assertRaisesRegex(ValueError, "Unsupported native codec"):
                    converter.convert(root / "absent.safetensors", path, output)
                self.assertEqual(output.read_bytes(), b"previous successful artifact")

    def test_unsupported_nested_settings(self):
        for group, field, value in [("seanet", "pad_mode", "replicate"),
                                     ("seanet", "ratios", [5, 6, 4]),
                                     ("transformer", "context", 249),
                                     ("transformer", "max_period", 20000),
                                     ("quantizer", "dimension", 16)]:
            with self.subTest(field=field):
                config = copy.deepcopy(CONFIG["mimi"])
                config[group][field] = value
                with self.assertRaisesRegex(ValueError, "Unsupported native codec"):
                    converter.validate_native_config(converter.MimiConfig.model_validate(config))

    def test_released_config(self):
        converter.validate_native_config(converter.MimiConfig.model_validate(CONFIG["mimi"]))


unittest.main()
