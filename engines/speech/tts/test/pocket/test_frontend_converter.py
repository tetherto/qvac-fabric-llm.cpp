#!/usr/bin/env python3
"""Frontend pair publication and prepared-voice validation regressions."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from safetensors.torch import save_file
import yaml

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "convert-pocket-frontend.py"
spec = importlib.util.spec_from_file_location("frontend_converter", SOURCE)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class FrontendConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump({"flow_lm": {"transformer": {"num_layers": 2, "num_heads": 2, "d_model": 8}}}))
        self.weights = self.root / "weights"; self.weights.write_bytes(b"source model")
        self.voice = self.root / "original-voice.safetensors"
        self.original_voice = {}
        for i in range(2):
            p = f"transformer.layers.{i}.self_attn/"
            self.original_voice[p+"cache"] = torch.ones(2, 1, 3, 2, 4)
            self.original_voice[p+"offset"] = torch.tensor([3])
        save_file(self.original_voice, self.voice)
        self.output = self.root / "bundle"; self.output.mkdir()
        self.frontend = self.output / "frontend.json"; self.frontend.write_bytes(b"old frontend")
        self.published_voice = self.output / "voice.gguf"; self.published_voice.write_bytes(b"old voice")

    def convert(self):
        # Tokenizer math is exercised against real upstream fixtures. This
        # isolates filesystem failures while retaining real voice validation.
        with patch.object(converter, "tokenizer_data", return_value={"text": "new frontend"}):
            converter.convert(self.weights, self.config, self.root / "tokenizer", self.voice, self.output)

    def unchanged(self):
        self.assertEqual(self.frontend.read_bytes(), b"old frontend")
        self.assertEqual(self.published_voice.read_bytes(), b"old voice")
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["frontend.json", "voice.gguf"])

    def test_staging_failure_preserves_pair(self):
        original = Path.write_text
        def fail(path, *args, **kwargs):
            if path.parent == self.output and path.name.startswith("frontend."):
                raise OSError("injected staging failure")
            return original(path, *args, **kwargs)
        with patch.object(Path, "write_text", fail), self.assertRaisesRegex(OSError, "staging failure"):
            self.convert()
        self.unchanged()

    def test_second_publish_failure_restores_pair(self):
        original = converter.os.replace
        def fail(source, dest):
            if Path(dest) == self.frontend:
                raise OSError("injected publication failure")
            return original(source, dest)
        with patch.object(converter.os, "replace", fail), self.assertRaisesRegex(OSError, "publication failure"):
            self.convert()
        self.unchanged()

    def test_inconsistent_offset_preserves_pair(self):
        self.original_voice["transformer.layers.1.self_attn/offset"] = torch.tensor([2])
        save_file(self.original_voice, self.voice)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self.convert()
        self.unchanged()

    def test_failed_rollback_keeps_recoverable_backup(self):
        original = converter.os.replace
        def fail(source, dest):
            if Path(dest) == self.frontend or Path(source).name.startswith("backup."):
                raise OSError("injected filesystem failure")
            return original(source, dest)
        with patch.object(converter.os, "replace", fail), self.assertRaisesRegex(RuntimeError, "backups retained"):
            self.convert()
        backups = list(self.output.glob("backup.*"))
        self.assertTrue(any(path.read_bytes() == b"old voice" for path in backups))
        self.assertEqual(self.frontend.read_bytes(), b"old frontend")

    def test_nonfinite_active_cache_preserves_pair(self):
        self.original_voice["transformer.layers.0.self_attn/cache"][0,0,0,0,0] = float("nan")
        save_file(self.original_voice, self.voice)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.convert()
        self.unchanged()

    def test_success_replaces_both(self):
        self.convert()
        self.assertNotEqual(self.frontend.read_bytes(), b"old frontend")
        self.assertEqual(self.published_voice.read_bytes()[:4], b"GGUF")


unittest.main()
