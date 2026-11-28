#!/usr/bin/env python3
"""Model-free checks for native MiniMax directory resolution and provenance."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load('validate-music-models')
generation = load('music-generation-provenance')


class MusicModelValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def model(self, name, payload=b'GGUFfixture'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_all_native_tiers(self):
        for tier in validator.MINIMAX_TIERS:
            with self.subTest(tier=tier):
                directory = self.root / tier
                self.model(f'{tier}/mm3-lm-{tier}.gguf')
                self.model(f'{tier}/mm3-synth-{tier}.gguf')
                self.assertTrue(validator.validate(directory, 'minimax'))

    def test_unknown_or_empty_tier_rejected(self):
        for tier in ('q4_0', 'unknown', ''):
            with self.subTest(tier=tier):
                self.model(f'mm3-lm-{tier}.gguf')
                self.model(f'mm3-synth-{tier}.gguf')
                self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_split_directory_pair_and_case_insensitive_extension(self):
        lm = self.model('mm3/MM3-LM-Q4_K_M.GGUF')
        synth = self.model('Mm3-Synth-Q4_K_M.gguf')
        self.assertTrue(validator.validate(self.root, 'minimax'))
        self.assertEqual(validator.minimax_pair(self.root), (lm, synth))

    def test_mismatched_tiers_rejected(self):
        self.model('mm3-lm-q4_k_m.gguf')
        self.model('mm3-synth-f16.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_duplicates_across_directories_rejected_before_header_validation(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        self.model('mm3/MM3-LM-F16.gguf', b'broken')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_case_duplicate_rejected(self):
        self.model('mm3-lm-f16.gguf')
        second = self.model('MM3-LM-F16.gguf')
        self.model('mm3-synth-f16.gguf')
        if len(list(self.root.iterdir())) != 3:
            self.skipTest('filesystem is case insensitive')
        self.assertTrue(second.is_file())
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_unused_duplicate_tier_still_rejected(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        self.model('mm3-lm-unused.gguf')
        self.model('mm3/mm3-lm-unused.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_native_priority_does_not_skip_broken_preferred_pair(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        preferred = self.model('mm3-lm-q8_0.gguf', b'bad header')
        synth = self.model('mm3-synth-q8_0.gguf')
        self.assertEqual(validator.minimax_pair(self.root), (preferred, synth))
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_nested_non_mm3_directory_is_not_discovered(self):
        self.model('other/mm3-lm-f16.gguf')
        self.model('other/mm3-synth-f16.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_provenance_retains_preparation_and_allowlisted_build_options(self):
        model = self.model('mm3/MM3-LM-F16.GGUF')
        binary = self.model('build/bin/mm3-replay', b'binary')
        self.model('build/CMakeCache.txt', b'GGML_CUDA:BOOL=ON\nSECRET:STRING=private\nCMAKE_BUILD_TYPE:STRING=Release\n')
        metadata = {'source': {'revision': 'pinned'}, 'files': {'model': 'hash'}}
        self.model('provenance.json', json.dumps(metadata).encode())
        with patch.dict('os.environ', {'MM3_LM_FLASH': '1'}):
            result = generation.provenance(self.root, binary, 'gpu')
        self.assertEqual(result['preparation']['metadata'], metadata)
        self.assertEqual(result['model_files'], [generation.identity(model)])
        self.assertEqual(result['environment']['MM3_LM_FLASH'], '1')
        self.assertEqual(result['build_cache']['options'], {'GGML_CUDA': 'ON', 'CMAKE_BUILD_TYPE': 'Release'})
        self.assertEqual(result['requested_device'], 'gpu')

    def test_local_pair_does_not_require_preparation_or_build_metadata(self):
        binary = self.model('mm3-replay', b'binary')
        result = generation.provenance(self.root, binary, 'cpu')
        self.assertIsNone(result['preparation'])
        self.assertIsNone(result['build_cache'])

    def test_pilot_hashes_models_once_and_retains_current_binary_provenance(self):
        model = self.model('mm3-lm-f16.gguf')
        binary = self.model('mm3-replay', b'binary')
        cache = self.root / 'fingerprints.json'
        with patch.object(generation, 'identity', wraps=generation.identity) as identify:
            initial = generation.model_identities(self.root, cache)
            for _ in range(3):
                result = generation.provenance(self.root, binary, 'cpu', cache)
                self.assertEqual(result['model_files'], initial)
        self.assertEqual(sum(call.args[0] == model for call in identify.call_args_list), 1)
        self.assertEqual(sum(call.args[0] == binary for call in identify.call_args_list), 3)

    def test_standalone_provenance_still_hashes_manual_models(self):
        model = self.model('mm3-lm-f16.gguf')
        binary = self.model('mm3-replay', b'binary')
        with patch.object(generation, 'identity', wraps=generation.identity) as identify:
            generation.provenance(self.root, binary, 'cpu')
            generation.provenance(self.root, binary, 'cpu')
        self.assertEqual(sum(call.args[0] == model for call in identify.call_args_list), 2)

    def test_pilot_rejects_changed_replaced_added_or_removed_models(self):
        for change in ('contents', 'replacement', 'added', 'removed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / 'mm3-lm-f16.gguf'
                model.write_bytes(b'GGUFfixture')
                stat = model.stat()
                cache = root / 'fingerprints.json'
                generation.model_identities(root, cache)
                if change == 'contents':
                    model.write_bytes(b'GGUFchanged')
                    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                elif change == 'replacement':
                    replacement = root / 'replacement'
                    replacement.write_bytes(model.read_bytes())
                    replacement.replace(model)
                elif change == 'added':
                    (root / 'mm3-synth-f16.gguf').write_bytes(b'GGUFfixture')
                else:
                    model.unlink()
                with self.assertRaisesRegex(ValueError, 'model files changed'):
                    generation.model_identities(root, cache)

    def test_model_change_during_hashing_does_not_publish_fingerprints(self):
        model = self.model('mm3-lm-f16.gguf')
        cache = self.root / 'fingerprints.json'
        real_identity = generation.identity
        def change_after_hash(path):
            result = real_identity(path)
            path.write_bytes(b'GGUFchanged-size')
            return result
        with patch.object(generation, 'identity', side_effect=change_after_hash):
            with self.assertRaisesRegex(ValueError, 'while fingerprinting'):
                generation.model_identities(self.root, cache)
        self.assertFalse(cache.exists())


if __name__ == '__main__':
    unittest.main()
