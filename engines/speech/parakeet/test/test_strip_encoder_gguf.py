"""Encoder-less GGUF regressions; no model checkpoint or Core ML required."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import gguf
    import numpy as np
except ImportError as error:
    print(f"SKIP: encoder stripping requires numpy and gguf ({error})", file=sys.stderr)
    raise SystemExit(77)


SCRIPT = Path(__file__).parents[1] / "scripts" / "strip-encoder-gguf.py"
SPEC = importlib.util.spec_from_file_location("strip_encoder_gguf", SCRIPT)
STRIPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STRIPPER)
ALIGNMENT = 64


def field_bytes(field):
    return b"".join(part.tobytes() for part in field.parts)


def create_fixture(path, endianess, include_encoder=True):
    writer = gguf.GGUFWriter(path, "parakeet", endianess=endianess)
    writer.add_uint32("general.alignment", ALIGNMENT)
    writer.data_alignment = ALIGNMENT
    writer.add_string("parakeet.model_variant", "transcribepsy-250m-eu-sub")
    writer.add_string("custom.unicode", "café 日本語")
    writer.add_uint64("custom.uint64", 2**63 + 17)
    writer.add_bool("custom.bool", True)
    writer.add_array("tokenizer.ggml.tokens", ["", "▁hello", "é"])
    writer.add_array("custom.numbers", [1, -2, 3])
    if include_encoder:
        writer.add_tensor("encoder.weight", np.arange(32, dtype=np.float32))
    writer.add_tensor("decoder.weight", np.arange(15, dtype=np.float16).reshape(3, 5))
    packed = gguf.quants.quantize(np.arange(64, dtype=np.float32).reshape(2, 32),
                                 gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("decoder.quantized", packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    if include_encoder:
        writer.add_tensor("encoder.norm.bias", np.zeros(8, dtype=np.float32))
    writer.add_tensor("encoder_extra.weight", np.arange(4, dtype=np.int32))
    writer.add_tensor("decoder.encoder.weight", np.arange(6, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class StripEncoderGgufTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "full.gguf"
        self.output = Path(self.directory.name) / "coreml.gguf"

    def test_metadata_and_retained_payloads_survive_both_byte_orders(self):
        for endianess in (gguf.GGUFEndian.LITTLE, gguf.GGUFEndian.BIG):
            with self.subTest(endianess=endianess):
                self.source = self.source.with_name(f"full-{endianess.name}.gguf")
                self.output = self.output.with_name(f"coreml-{endianess.name}.gguf")
                create_fixture(self.source, endianess)
                before = self.source.read_bytes()
                self.assertEqual(STRIPPER.strip_encoder(self.source, self.output), (2, 4))
                source = gguf.GGUFReader(self.source)
                output = gguf.GGUFReader(self.output)
                retained = [tensor for tensor in source.tensors
                            if not tensor.name.startswith("encoder.")]
                self.assertEqual(list(source.fields), list(output.fields))
                for name, field in source.fields.items():
                    if name != "GGUF.tensor_count":
                        self.assertEqual(field_bytes(field), field_bytes(output.fields[name]), name)
                self.assertEqual([tensor.name for tensor in retained],
                                 [tensor.name for tensor in output.tensors])
                self.assertEqual(output.endianess, endianess)
                self.assertEqual(output.alignment, ALIGNMENT)
                for original, copied in zip(retained, output.tensors):
                    self.assertEqual(original.tensor_type, copied.tensor_type)
                    np.testing.assert_array_equal(original.shape, copied.shape)
                    self.assertEqual(original.data.tobytes(), copied.data.tobytes())
                    self.assertEqual(copied.data_offset % ALIGNMENT, 0)
                self.assertEqual(self.source.read_bytes(), before)
                expected = self.source.with_name("expected.gguf")
                create_fixture(expected, endianess, include_encoder=False)
                self.assertEqual(self.output.read_bytes(), expected.read_bytes())

    def test_existing_output_and_input_are_never_overwritten(self):
        create_fixture(self.source, gguf.GGUFEndian.LITTLE)
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "must differ"):
            STRIPPER.strip_encoder(self.source, self.source)
        self.output.write_bytes(b"existing output")
        with self.assertRaises(FileExistsError):
            STRIPPER.strip_encoder(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.assertEqual(self.source.read_bytes(), before)

    def test_missing_encoder_fails_without_creating_output(self):
        create_fixture(self.source, gguf.GGUFEndian.LITTLE, include_encoder=False)
        with self.assertRaisesRegex(ValueError, "no encoder"):
            STRIPPER.strip_encoder(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_encoder_only_input_keeps_metadata_with_zero_tensors(self):
        writer = gguf.GGUFWriter(self.source, "parakeet")
        writer.add_string("custom.note", "encoder only")
        writer.add_tensor("encoder.weight", np.arange(32, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.assertEqual(STRIPPER.strip_encoder(self.source, self.output), (1, 0))
        output = gguf.GGUFReader(self.output)
        self.assertEqual(output.tensors, [])
        self.assertEqual(output.fields["custom.note"].contents(), "encoder only")

    def test_failed_copy_removes_partial_output(self):
        create_fixture(self.source, gguf.GGUFEndian.LITTLE)
        with mock.patch.object(STRIPPER, "copy_tensor", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                STRIPPER.strip_encoder(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_strips_without_loading_coreml(self):
        create_fixture(self.source, gguf.GGUFEndian.LITTLE)
        result = subprocess.run([sys.executable, str(SCRIPT), "--gguf", str(self.source),
                                 "--out", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Removed 2 encoder tensors; retained 4 tensors", result.stdout)
        self.assertEqual(len(gguf.GGUFReader(self.output).tensors), 4)


if __name__ == "__main__":
    unittest.main()
