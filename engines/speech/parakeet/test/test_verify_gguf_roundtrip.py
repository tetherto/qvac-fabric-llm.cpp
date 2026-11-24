"""Tiny real-GGUF verifier regressions; requires numpy/gguf, no checkpoint/Torch."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import gguf
    import numpy as np
except ImportError as error:
    print(f'SKIP: GGUF roundtrip regressions require numpy and gguf ({error})', file=sys.stderr)
    raise SystemExit(77)


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'verify-gguf-roundtrip.py'
SPEC = importlib.util.spec_from_file_location('verify_gguf_roundtrip', SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
Q = gguf.GGMLQuantizationType


class RoundtripVerifierTests(unittest.TestCase):
    def verify(self, actual, expected, qtype, scale=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            writer = gguf.GGUFWriter(path, 'parakeet')
            writer.add_string('parakeet.model.type', 'ctc')
            writer.add_uint32('parakeet.encoder.n_layers', 0)
            if qtype in (Q.F32, Q.F16, Q.I32):
                dtype = {Q.F32: np.float32, Q.F16: np.float16, Q.I32: np.int32}[qtype]
                writer.add_tensor('test.weight', np.asarray(actual, dtype=dtype))
            else:
                packed = gguf.quants.quantize(np.asarray(actual, dtype=np.float32), qtype)
                if scale is not None:
                    # Q4_0/Q5_0/Q8_0 all begin each block with an F16 scale.
                    packed = packed.copy()
                    packed.view(np.uint8).reshape(-1)[:2] = np.asarray([scale], dtype='<f2').view(np.uint8)
                writer.add_tensor('test.weight', packed, raw_dtype=qtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            output = io.StringIO()
            with mock.patch.object(sys, 'argv', [str(SCRIPT), '--gguf', str(path), '--nemo', 'unused.nemo']), \
                    mock.patch.object(VERIFIER, 'load_sd', return_value={}), \
                    mock.patch.object(VERIFIER, 'build_expected_ctc', return_value={'test.weight': np.asarray(expected, dtype=np.float32)}), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = VERIFIER.main()
            return result, output.getvalue()

    def test_exact_f32(self):
        self.assertEqual(self.verify([1, -2, 0], [1, -2, 0], Q.F32)[0], 0)

    def test_changed_f32_rejected(self):
        self.assertEqual(self.verify([1, -2, .1], [1, -2, 0], Q.F32)[0], 2)

    def test_matching_f16_rounding(self):
        self.assertEqual(self.verify([1.0003, -2.0007], [1.0003, -2.0007], Q.F16)[0], 0)

    def test_nan_f16_output_rejected(self):
        result, output = self.verify([np.nan, 1], [0, 1], Q.F16)
        self.assertEqual(result, 2, output)

    def test_infinite_f16_output_rejected(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(self.verify([value, 1], [0, 1], Q.F16)[0], 2)

    def test_nonfinite_f16_reference_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(self.verify([1, 1], [value, 1], Q.F16)[0], 2)

    def test_nonfinite_bf16_output_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(self.verify([value, 1], [0, 1], Q.BF16)[0], 2)

    def test_bf16_round_to_nearest_even(self):
        # Midpoints around consecutive BF16 values round in opposite directions.
        actual = np.array([1 + 1 / 256, 1 + 3 / 256, -2.006], dtype=np.float32)
        packed = gguf.quants.quantize(actual, Q.BF16)
        rounded = gguf.quants.dequantize(packed, Q.BF16)
        np.testing.assert_array_equal(rounded[:2], [1, 1 + 1 / 64])
        self.assertEqual(self.verify(actual, actual, Q.BF16)[0], 0)

    def test_bf16_finite_error_rejected(self):
        result, output = self.verify([2, 1], [1, 1], Q.BF16)
        self.assertEqual(result, 2, output)

    def test_bf16_nonfinite_reference_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(self.verify([1, 1], [value, 1], Q.BF16)[0], 2)

    def test_matching_block_quantized_tensors(self):
        values = np.linspace(-2, 3, 64, dtype=np.float32).reshape(2, 32)
        for qtype in (Q.Q8_0, Q.Q5_0, Q.Q4_0):
            with self.subTest(qtype=qtype):
                result, output = self.verify(values, values, qtype)
                self.assertEqual(result, 0, output)

    def test_block_quantized_nonfinite_reference_rejected(self):
        values = np.linspace(-2, 3, 32, dtype=np.float32).reshape(1, 32)
        for qtype in (Q.Q8_0, Q.Q5_0, Q.Q4_0):
            for value in (np.nan, np.inf, -np.inf):
                expected = values.copy()
                expected[0, 0] = value
                with self.subTest(qtype=qtype, value=value):
                    result, output = self.verify(values, expected, qtype)
                    self.assertEqual(result, 2, output)

    def test_block_quantized_nonfinite_output_rejected(self):
        values = np.linspace(-2, 3, 32, dtype=np.float32).reshape(1, 32)
        for qtype in (Q.Q8_0, Q.Q5_0, Q.Q4_0):
            for scale in (np.nan, np.inf, -np.inf):
                with self.subTest(qtype=qtype, scale=scale), np.errstate(invalid='ignore'):
                    result, output = self.verify(values, values, qtype, scale=scale)
                    self.assertEqual(result, 2, output)

    def test_unsupported_storage_rejected(self):
        result, output = self.verify([1, 2], [1, 2], Q.I32)
        self.assertEqual(result, 2, output)

    def test_shape_mismatch_rejected(self):
        self.assertEqual(self.verify([[1, 2], [3, 4]], [1, 2, 3, 4], Q.F16)[0], 2)

    def test_singleton_squeeze_remains_supported(self):
        self.assertEqual(self.verify([[1, 2]], [1, 2], Q.F16)[0], 0)

    def test_empty_tensor_rejected(self):
        # GGUFWriter does not serialize zero extents; check the same production
        # comparison entry point with an empty decoded tensor.
        tensor = mock.Mock(tensor_type=Q.F16)
        with mock.patch.object(VERIFIER, 'ggml_to_numpy', return_value=np.array([], dtype=np.float32)):
            self.assertFalse(VERIFIER.compare_tensor(tensor, np.array([], dtype=np.float32))['pass'])


if __name__ == '__main__':
    unittest.main()
