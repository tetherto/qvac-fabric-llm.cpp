#!/usr/bin/env python3
"""Checkpoint-free regressions for Audio8 conversion verification.

NumPy exercises compare directly. Optional gguf coverage reads actual small
F32/F16/Q4_0/Q8_0 payloads before comparing them; no torch dependency.
"""
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import gguf
except ImportError:
    gguf = None


def load_verifier():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'verify-audio8-conversion.py'
    spec = importlib.util.spec_from_file_location('audio8_conversion_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, 'numpy not installed')
class CompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()

    def compare(self, name, actual, expected, storage='F32'):
        return self.verifier.compare(
            {name: (np.asarray(actual, np.float32), storage)},
            {name: np.asarray(expected, np.float32)}, 'lm',
        )[2]

    def test_nonfinite_actual_fails(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                failures = self.compare('lm/norm', [value], [1])
                self.assertTrue(failures, 'a nonfinite payload must fail')

    def test_nonfinite_reference_fails(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                failures = self.compare('lm/norm', [1], [value])
                self.assertTrue(failures, 'a nonfinite reference must fail')

    def test_exact_role_requires_f32(self):
        for name in ('lm/norm', 'lm/sem_head', 'fast/out', 'lm/rope_cos', 'lm/blk/0/wq_b'):
            for storage in ('F16', 'Q4_0', 'Q8_0'):
                with self.subTest(name=name, storage=storage):
                    self.assertTrue(self.compare(name, [[1]], [[1]], storage))

    def test_dense_role_disallows_block_quantization(self):
        values = np.ones((1, 32), np.float32)
        for storage in ('Q4_0', 'Q8_0'):
            with self.subTest(storage=storage):
                self.assertTrue(self.compare('lm/tok_emb', values, values, storage))

    def test_unaligned_body_cannot_claim_block_storage(self):
        values = np.ones((2, 33), np.float32)
        self.assertTrue(self.compare('lm/blk/0/wq', values, values, 'Q8_0'))

    def test_unsupported_type_fails(self):
        for storage in ('I32', 'BF16', 'Q4_K', 'unknown'):
            with self.subTest(storage=storage):
                self.assertTrue(self.compare('lm/norm', [1], [1], storage))

    def test_valid_controls_and_tolerances(self):
        self.assertEqual(self.compare('lm/norm', [1], [1]), [])
        self.assertEqual(self.compare('lm/tok_emb', [[1]], [[1]], 'F16'), [])
        values = np.ones((2, 32), np.float32)
        for storage in ('F32', 'F16', 'Q4_0', 'Q8_0'):
            with self.subTest(storage=storage):
                self.assertEqual(self.compare('lm/blk/0/wq', values, values, storage), [])
        self.assertTrue(self.compare('lm/norm', [1.01], [1]))
        self.assertEqual(self.verifier.RELATIVE_TOLERANCE,
                         {'F32': 1e-6, 'F16': 1e-3, 'Q8_0': 1/127, 'Q4_0': 1/8})

    def test_codec_role_policy(self):
        for name, shape, allowed in (
            ('q/post/blk/0/wq', (2, 32), ('F32', 'F16', 'Q4_0', 'Q8_0')),
            ('q/post/blk/0/wq', (2, 33), ('F32', 'F16')),
            ('dec/0/conv/weight', (3, 2, 32), ('F32', 'F16')),
            ('q/up/0/1/dwconv/conv/weight', (3, 32), ('F32',)),
            ('q/sem/0/codebook/weight', (3, 32), ('F32',)),
            ('q/post/blk/0/attn_scale', (32,), ('F32',)),
            ('rope/post_tf_cos', (3, 32), ('F32',)),
        ):
            values = np.ones(shape, np.float32)
            for storage in ('F32', 'F16', 'Q4_0', 'Q8_0'):
                with self.subTest(name=name, storage=storage, shape=shape):
                    failures = self.verifier.compare(
                        {name: (values, storage)}, {name: values}, 'codec/decoder',
                        self.verifier.codec_tensor_role,
                    )[2]
                    self.assertEqual(not failures, storage in allowed)

    @unittest.skipIf(gguf is None, 'gguf not installed')
    def test_real_storage_payloads(self):
        values = np.random.default_rng(51).uniform(-1, 1, (3, 32)).astype(np.float32)
        for storage in ('F32', 'F16', 'Q4_0', 'Q8_0'):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tiny.gguf'
                writer = gguf.GGUFWriter(path, 'audio8-lm')
                raw_type = None
                if storage.startswith('Q'):
                    raw_type = getattr(gguf.GGMLQuantizationType, storage)
                    payload = gguf.quants.quantize(values, raw_type)
                else:
                    payload = values.astype(np.float16 if storage == 'F16' else np.float32)
                writer.add_tensor('lm/blk/0/wq', payload, raw_dtype=raw_type)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                _, tensors = self.verifier.read_gguf(path)
                self.assertEqual(tensors['lm/blk/0/wq'][1], storage)
                self.assertEqual(self.verifier.compare(tensors, {'lm/blk/0/wq': values}, 'lm')[2], [])
                # A valid decoded payload still fails under a tensor role that
                # cannot use that storage; this checks beyond numeric closeness.
                if storage != 'F32':
                    value, kind = tensors['lm/blk/0/wq']
                    self.assertTrue(self.verifier.compare({'lm/sem_head': (value, kind)},
                                                         {'lm/sem_head': values}, 'lm')[2])

    @unittest.skipIf(gguf is None, 'gguf not installed')
    def test_real_nonfinite_payload_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nonfinite.gguf'
            writer = gguf.GGUFWriter(path, 'audio8-lm')
            writer.add_tensor('lm/norm', np.array([1, np.nan], np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            _, tensors = self.verifier.read_gguf(path)
            self.assertTrue(self.verifier.compare(tensors, {'lm/norm': np.ones(2, np.float32)}, 'lm')[2])

    def invalid_rope_planes(self):
        for value in (np.nan, np.inf, -np.inf):
            yield np.full((2, 2), value, np.float32), 'F32'
        for storage in ('F16', 'BF16', 'Q8_0'):
            yield np.ones((2, 2), np.float32), storage
        yield np.ones((2, 3), np.float32), 'F32'

    def codec_nonreference_fixture(self):
        fields = {
            'audio8.codec.part': 'decoder', 'audio8.codec.rope_precision': 'f32',
            'audio8.codec.max_frames': 2, 'audio8.codec.post_tf.head_dim': 4,
        }
        expected = {'dec/0/conv/weight': np.ones((1, 2, 2), np.float32)}
        planes = ('rope/post_tf_cos', 'rope/post_tf_sin')
        tensors = {name: (array, 'F32') for name, array in expected.items()}
        tensors.update({name: (np.full((2, 2), 0.25, np.float32), 'F32') for name in planes})
        return fields, expected, planes, tensors

    def check_codec_nonreference(self, fields, expected, tensors):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.verifier.verify_codec('unused', 'tiny.gguf', 'decoder', expected, fields, tensors)

    def test_codec_nonreference_rope_is_validated(self):
        fields, expected, planes, valid = self.codec_nonreference_fixture()
        for name in planes:
            for array, storage in self.invalid_rope_planes():
                with self.subTest(name=name, storage=storage, values=array):
                    tensors = dict(valid, **{name: (array, storage)})
                    failures = self.check_codec_nonreference(fields, expected, tensors)
                    self.assertTrue(any(name in failure for failure in failures), failures)

    def test_codec_nonreference_rope_skips_only_numeric_comparison(self):
        fields, expected, planes, tensors = self.codec_nonreference_fixture()
        self.assertEqual(self.check_codec_nonreference(fields, expected, tensors), [])
        for name in planes:
            missing = dict(tensors)
            del missing[name]
            self.assertTrue(self.check_codec_nonreference(fields, expected, missing))

    def lm_nonreference_fixture(self):
        fields = {'audio8.lm.rope_precision': 'f32'}
        config = {'max_seq_len': 2, 'head_dim': 4, 'num_codebooks': 2, 'fast_head_dim': 4}
        expected = {'lm/norm': np.ones(2, np.float32)}
        planes = self.verifier.LM_ROPE_TENSORS
        tensors = {'lm/norm': (expected['lm/norm'], 'F32')}
        tensors.update({name: (np.full((2, 2), 0.25, np.float32), 'F32') for name in planes})
        return fields, config, expected, planes, tensors

    def check_lm_nonreference(self, fields, config, expected, tensors):
        with mock.patch.object(self.verifier, 'read_gguf', return_value=(fields, tensors)), \
                mock.patch.object(self.verifier, 'load_config', return_value=config), \
                mock.patch.object(self.verifier, 'load_lm_state', return_value={}), \
                mock.patch.object(self.verifier, 'lm_expectations', return_value=dict(expected)), \
                mock.patch.object(self.verifier, 'verify_rope_layout', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            return self.verifier.verify_lm('unused', 'tiny.gguf')

    def test_lm_nonreference_rope_is_validated(self):
        fields, config, expected, planes, valid = self.lm_nonreference_fixture()
        for name in planes:
            for array, storage in self.invalid_rope_planes():
                with self.subTest(name=name, storage=storage, values=array):
                    tensors = dict(valid, **{name: (array, storage)})
                    failures = self.check_lm_nonreference(fields, config, expected, tensors)
                    self.assertTrue(any(name in failure for failure in failures), failures)

    def test_lm_nonreference_rope_skips_only_numeric_comparison(self):
        fields, config, expected, planes, tensors = self.lm_nonreference_fixture()
        self.assertEqual(self.check_lm_nonreference(fields, config, expected, tensors), [])
        for name in planes:
            missing = dict(tensors)
            del missing[name]
            self.assertTrue(self.check_lm_nonreference(fields, config, expected, missing))

    def test_excluded_values_are_not_reported_as_numeric_matches(self):
        values = np.full((2, 2), 0.25, np.float32)
        worst, counts, failures = self.verifier.compare(
            {'lm/norm': (np.ones(2, np.float32), 'F32'), 'lm/rope_cos': (values, 'F32')},
            {'lm/norm': np.ones(2, np.float32), 'lm/rope_cos': np.zeros_like(values)},
            'lm', value_exclusions={'lm/rope_cos'},
        )
        self.assertEqual(failures, [])
        self.assertEqual(dict(counts), {'lm/norm': 1})
        self.assertNotIn('lm/rope_cos', worst)


if __name__ == '__main__':
    unittest.main()
