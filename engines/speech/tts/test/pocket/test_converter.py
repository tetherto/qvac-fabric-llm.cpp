#!/usr/bin/env python3
"""Converter regression tests. Usage: python test_converter.py fixture-dir."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import gguf
import numpy as np
from safetensors.torch import load_file, save_file
import torch
import yaml

FIXTURES = Path(sys.argv.pop(1))
SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "convert-pocket-flow-lm-to-gguf.py"
spec = importlib.util.spec_from_file_location("converter", SOURCE)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = load_file(str(FIXTURES / "model.safetensors"))
        self.config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
        self.output = self.root / "result.gguf"

    def convert(self, dtype="f32"):
        source = self.root / "source.safetensors"
        config = self.root / "config.yaml"
        save_file(self.weights, str(source))
        config.write_text(yaml.safe_dump(self.config))
        converter.convert(source, config, self.output, dtype)

    def fails(self):
        # A failed export must leave the previous valid output untouched.
        self.output.write_bytes(b"previous artifact")
        with self.assertRaises(ValueError):
            self.convert()
        self.assertEqual(self.output.read_bytes(), b"previous artifact")

    def test_bfloat_checkpoint(self):
        self.weights = {k: v.to(torch.bfloat16) for k, v in self.weights.items()}
        self.convert()
        reader = gguf.GGUFReader(self.output)
        tensor = next(t for t in reader.tensors if t.name == "input_linear.weight")
        np.testing.assert_array_equal(tensor.data, self.weights["flow_lm.input_linear.weight"].float().numpy())

    def test_half_keeps_vector_precision(self):
        self.convert("f16")
        types = {t.name: t.tensor_type for t in gguf.GGUFReader(self.output).tensors}
        self.assertEqual(types["input_linear.weight"], gguf.GGMLQuantizationType.F16)
        self.assertEqual(types["emb_std"], gguf.GGMLQuantizationType.F32)

    def test_missing_weight(self):
        del self.weights["flow_lm.input_linear.weight"]
        self.fails()

    def test_unknown_weight(self):
        self.weights["flow_lm.extra"] = torch.ones(1)
        self.fails()

    def test_wrong_shape(self):
        self.weights["flow_lm.input_linear.weight"] = torch.zeros(31, 8)
        self.fails()

    def test_non_float(self):
        self.weights["flow_lm.bos_emb"] = torch.ones(8, dtype=torch.int32)
        self.fails()

    def test_non_finite(self):
        self.weights["flow_lm.bos_emb"][0] = float("nan")
        self.fails()

    def test_nonpositive_std(self):
        self.weights["flow_lm.emb_std"][0] = 0
        self.fails()

    def test_wrong_flow_type(self):
        self.config["flow_lm"]["flow"]["type"] = "flow_matching"
        self.fails()

    def test_wrong_head_count(self):
        self.config["flow_lm"]["transformer"]["num_heads"] = 3
        self.fails()

    def test_wrong_bos_type(self):
        self.config["flow_lm"]["insert_bos_before_voice"] = "false"
        self.fails()

    def test_half_overflow_preserves_output(self):
        self.weights["flow_lm.input_linear.weight"][0, 0] = 1e10
        self.output.write_bytes(b"previous artifact")
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            self.convert("f16")
        self.assertEqual(self.output.read_bytes(), b"previous artifact")
        self.assertEqual(list(self.root.glob("result.gguf.*")), [])


if __name__ == "__main__":
    unittest.main()
