"""Read real GGUF output from a tiny safetensors conditioner checkpoint."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

REQUIRED_DEPENDENCIES = ("numpy", "gguf")
missing = [name for name in REQUIRED_DEPENDENCIES if importlib.util.find_spec(name) is None]
if missing:
    print("SKIP: missing " + ", ".join(missing))
    raise SystemExit(77)

import gguf
import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "convert-minimax-music3-to-gguf.py"
SPEC = importlib.util.spec_from_file_location("minimax_roundtrip_converter", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

@contextmanager
def read_gguf(path):
    reader = gguf.GGUFReader(str(path))
    try:
        yield reader
    finally:
        reader.data._mmap.close()


TINY_SHAPE = {"COND_LAYERS": 2, "COND_HIDDEN": 3, "COND_OUT": 2, "COND_KERNEL": 3}
QUANT_SHAPE = (2, 32)


def write_checkpoint(path, tensors):
    header, chunks, offset = {}, [], 0
    for name, tensor in tensors.items():
        raw = tensor.astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


class ConditionerRoundtripTests(unittest.TestCase):
    def test_checkpoint_maps_weights_shapes_and_metadata(self):
        tensors = {"cond_layer_logits": np.array([1, -1], dtype=np.float32),
                   "cond_layer_scale": np.array([0.5], dtype=np.float32),
                   "latent_conditioners.0.weight": np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 8,
                   "latent_conditioners.0.bias": np.array([0.25, -0.5], dtype=np.float32)}
        mapping = {"cond.layer_logits": "cond_layer_logits", "cond.layer_scale": "cond_layer_scale",
                   "cond.proj.weight": "latent_conditioners.0.weight", "cond.proj.bias": "latent_conditioners.0.bias"}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            checkpoint, output = root / "model.safetensors", root / "output"
            write_checkpoint(checkpoint, tensors)
            arguments = [str(SCRIPT), "--src", str(checkpoint), "--out", str(output), "--components", "cond"]
            with mock.patch.multiple(converter, **TINY_SHAPE), mock.patch.object(sys, "argv", arguments):
                converter.main()
            with read_gguf(output / "mm3-synth-f16.gguf") as reader:
                self.assertEqual(reader.fields["general.architecture"].contents(), "mm3")
                for name, value in TINY_SHAPE.items():
                    key = {"COND_LAYERS": "num_layers", "COND_HIDDEN": "hidden_dim",
                           "COND_OUT": "out_dim", "COND_KERNEL": "kernel_size"}[name]
                    self.assertEqual(reader.fields["mm3.cond." + key].contents(), value)
                actual = {tensor.name: tensor for tensor in reader.tensors}
                self.assertEqual(set(actual), set(mapping))
                for name, source in mapping.items():
                    tensor = actual[name]
                    self.assertEqual(list(tensor.shape), list(reversed(tensors[source].shape)))
                    expected_type = gguf.GGMLQuantizationType.F16 if name == "cond.proj.weight" else gguf.GGMLQuantizationType.F32
                    self.assertEqual(tensor.tensor_type, expected_type)
                    np.testing.assert_array_equal(tensor.data.astype(np.float32), tensors[source])


class TensorSerializationTests(unittest.TestCase):
    def test_quantization_preserves_protected_and_overflow_tensors(self):
        weights = np.linspace(-1, 1, np.prod(QUANT_SHAPE), dtype=np.float32).reshape(QUANT_SHAPE)
        overflow = np.full(QUANT_SHAPE, converter.F16_MAX * 2, dtype=np.float32)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            for quant in ("f16", "q8_0"):
                with self.subTest(quant=quant):
                    bundle = converter.Bundle("mm3")
                    bundle.put("dit.weight", weights, converter.QUANT)
                    bundle.put("protected.weight", weights, converter.F32)
                    bundle.put("overflow.weight", overflow, converter.QUANT)
                    path = Path(directory) / (quant + ".gguf")
                    bundle.write(str(path), quant)
                    with read_gguf(path) as reader:
                        actual = {tensor.name: tensor for tensor in reader.tensors}
                        quant_type = gguf.GGMLQuantizationType.Q8_0 if quant == "q8_0" else gguf.GGMLQuantizationType.F16
                        self.assertEqual(actual["dit.weight"].tensor_type, quant_type)
                        self.assertEqual(actual["protected.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
                        self.assertEqual(actual["overflow.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
                        decoded = gguf.quants.dequantize(actual["dit.weight"].data, quant_type)
                        np.testing.assert_allclose(decoded, weights, atol=0.01, rtol=0)
                        np.testing.assert_array_equal(actual["protected.weight"].data, weights)
                        np.testing.assert_array_equal(actual["overflow.weight"].data, overflow)

    def test_weight_norm_uses_each_output_channel_norm(self):
        directions = np.array([[[3, 4]], [[0, 2]]], dtype=np.float32)
        scales = np.array([10, 3], dtype=np.float32)
        actual = converter.fold_weight_norm(scales, directions, "fixture")
        np.testing.assert_array_equal(actual, [[[6, 8]], [[0, 3]]])
        with self.assertRaises(SystemExit):
            converter.fold_weight_norm(scales, np.zeros_like(directions), "fixture")


if __name__ == "__main__":
    unittest.main()
