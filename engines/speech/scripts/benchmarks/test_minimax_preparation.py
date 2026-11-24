#!/usr/bin/env python3
"""Model-free source/cache/transaction regressions; no network or large allocations."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('prepare_minimax', Path(__file__).with_name('prepare-minimax.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_preflight = module.preflight


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / 'libggml-base.so'
        self.library.write_bytes(b'quantization implementation')
        self.payloads = {f'{role}.safetensors' if role != 'license' else 'LICENSE': role.encode() * 8
                         for role in ('lm', 'dit', 'vocoder', 'license')}
        self.manifest = {'schema_version': 1, 'model_id': 'test/model', 'revision': 'a' * 40,
                         'license': {'name': 'test license'},
                         'resource_requirements': {'min_memory_bytes': 1, 'conversion_free_disk_bytes': 1},
                         'files': [{'path': path, 'role': 'license' if path == 'LICENSE' else path.split('.')[0],
                                    'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                   for path, data in self.payloads.items()]}
        self.dependencies = {'versions': module.locked_versions(), 'gguf_sources': {'writer.py': 'b' * 64},
                             'python': '3.12.0', 'platform': 'test', 'machine': 'test', 'byteorder': 'little'}
        for owner, name, replacement in [(module, 'dependency_identity', lambda: self.dependencies),
                                         (module, 'build_metadata', lambda path: {'source_checkout': {'commit': 'c' * 40, 'dirty': False}, 'quantizer_build': {'options': {'GGML_CUDA': 'OFF'}} if path else None}),
                                         (module, 'preflight', mock.Mock()),
                                         (module.subprocess, 'run', self.convert),
                                         (module.urllib.request, 'urlopen', self.open_url)]:
            patch = mock.patch.object(owner, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        self.calls = []

    def open_url(self, url, timeout):
        self.calls.append(url)
        return io.BytesIO(self.payloads[url.rsplit('/', 1)[-1]])

    def convert(self, command, **kwargs):
        self.assertEqual(kwargs['stdout'], module.sys.stderr)
        if '--out' in command:
            destination = Path(command[command.index('--out') + 1])
            quant = command[command.index('--quant') + 1]
            for role in ('lm', 'synth'):
                (destination / f'mm3-{role}-{quant}.gguf').write_bytes(b'GGUF' + role.encode() * 32)
        else:
            Path(command[2]).write_bytes(Path(command[1]).read_bytes() + b'quantized')
        return subprocess.CompletedProcess(command, 0)

    def prepare(self, **kwargs):
        if kwargs.get('quant') == 'q4_k_m' and 'quantizer_libraries' not in kwargs:
            kwargs['quantizer_libraries'] = [self.library]
        return module.prepare(self.manifest, self.root, **kwargs)

    def test_verified_warm_cache_needs_no_sources_or_conversion(self):
        result = self.prepare()
        self.assertFalse(result['cached'])
        self.assertEqual(len(self.calls), 4)
        self.assertIn('/resolve/' + 'a' * 40 + '/', self.calls[0])
        with (mock.patch.object(module, 'download', side_effect=AssertionError('download')),
              mock.patch.object(module.subprocess, 'run', side_effect=AssertionError('conversion'))):
            reused = self.prepare(offline=True)
        self.assertTrue(reused['cached'])
        record = json.loads(Path(result['provenance']).read_text())
        self.assertEqual(set(record['outputs']), {'mm3-lm-q8_0.gguf', 'mm3-synth-q8_0.gguf'})

    def test_corruption_and_missing_pair_are_never_verified(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                result = self.prepare()
                path = Path(result['model_dir']) / 'mm3-lm-q8_0.gguf'
                path.unlink() if missing else path.write_bytes(b'GGUF' + b'corrupted' * 8)
                with self.assertRaisesRegex(module.PreparationError, 'missing, incomplete, corrupt'):
                    self.prepare(verify_only=True)
                self.assertFalse(self.prepare(offline=True)['cached'])

    def test_missing_or_incompatible_provenance_is_not_reused(self):
        result = self.prepare()
        path = Path(result['provenance'])
        record = json.loads(path.read_text())
        record['identity']['converter_sha256'] = '0' * 64
        path.write_text(json.dumps(record))
        with self.assertRaises(module.PreparationError):
            self.prepare(verify_only=True)
        path.unlink()
        with self.assertRaises(module.PreparationError):
            self.prepare(verify_only=True)

    def test_dependency_and_quant_changes_invalidate_cache(self):
        result = self.prepare()
        self.dependencies['python'] = '3.12.1'
        with self.assertRaises(module.PreparationError):
            self.prepare(verify_only=True)
        self.assertNotEqual(result['cache_key'], self.prepare(offline=True)['cache_key'])
        with self.assertRaises(module.PreparationError):
            self.prepare(quant='f16', verify_only=True)

    def test_rebuild_hashes_cached_sources_once(self):
        result = self.prepare()
        record = json.loads(Path(result['provenance']).read_text())
        source = Path(record['source_dir'])
        source_paths = {source / item['path'] for item in self.manifest['files']}
        Path(result['provenance']).unlink()
        with mock.patch.object(module, 'digest', wraps=module.digest) as digest:
            rebuilt = self.prepare(offline=True)
        self.assertFalse(rebuilt['cached'])
        for path in source_paths:
            self.assertEqual(sum(call.args[0] == path for call in digest.call_args_list), 1, path)

    def test_download_uses_configurable_timeout(self):
        opener = mock.Mock(side_effect=self.open_url)
        with mock.patch.object(module, 'DOWNLOAD_TIMEOUT_SECONDS', 7):
            module.download(self.manifest['files'][0], self.manifest, self.root, False, opener=opener)
        self.assertEqual(opener.call_args.kwargs['timeout'], 7)

    def test_quantization_failure_keeps_stage_and_does_not_publish(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        def fail_quantization(command, **kwargs):
            if '--out' not in command:
                raise subprocess.CalledProcessError(2, command)
            return self.convert(command, **kwargs)
        with mock.patch.object(module.subprocess, 'run', side_effect=fail_quantization):
            with self.assertRaises(module.PreparationError) as error:
                self.prepare(quant='q4_k_m', quantizer=binary)
        self.assertEqual(error.exception.stage, 'quantization')
        self.assertEqual(list(self.root.rglob('provenance.json')), [])
        self.assertEqual(list(self.root.rglob('*.gguf')), [])

    def test_output_and_publication_errors_keep_stage(self):
        for action, stage in [('verify_outputs', 'output-verification'),
                              ('publish_pair_provenance_last', 'publication')]:
            with self.subTest(stage=stage), mock.patch.object(module, action, side_effect=OSError('test failure')):
                with self.assertRaises(module.PreparationError) as error:
                    self.prepare()
                self.assertEqual(error.exception.stage, stage)
                self.assertEqual(list(self.root.rglob('provenance.json')), [])

    def test_cache_key_only_has_no_model_preparation_side_effects(self):
        root = self.root / 'not-created'
        with (mock.patch.object(module, 'cache_lock', side_effect=AssertionError('cache access')),
              mock.patch.object(module, 'preflight', side_effect=AssertionError('resource check')),
              mock.patch.object(module, 'download', side_effect=AssertionError('download')),
              mock.patch.object(module.subprocess, 'run', side_effect=AssertionError('conversion'))):
            result = module.prepare(self.manifest, root, cache_key_only=True)
        self.assertEqual(result['stage'], 'identity')
        self.assertRegex(result['cache_key'], r'^[0-9a-f]{64}$')
        self.assertIsNone(result['model_dir'])
        self.assertFalse(root.exists())

    def test_exported_key_tracks_python_and_native_build_and_reuses_new_bundle(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        keys = set()
        for python, cache_hash in [('3.12.0', 'a' * 64), ('3.12.1', 'a' * 64), ('3.12.1', 'b' * 64)]:
            with self.subTest(python=python, cache_hash=cache_hash):
                self.dependencies['python'] = python
                build = {'source_checkout': None, 'quantizer_build': {'sha256': cache_hash}}
                with mock.patch.object(module, 'build_metadata', return_value=build):
                    exported = self.prepare(quant='q4_k_m', quantizer=binary, cache_key_only=True)
                    # Each environment change must miss the old *workflow* key,
                    # permitting actions/cache to save the newly prepared bundle.
                    self.assertNotIn(exported['cache_key'], keys)
                    keys.add(exported['cache_key'])
                    prepared = self.prepare(quant='q4_k_m', quantizer=binary)
                    self.assertFalse(prepared['cached'])
                    self.assertEqual(exported['cache_key'], prepared['cache_key'])
                    with (mock.patch.object(module, 'download', side_effect=AssertionError('download')),
                          mock.patch.object(module.subprocess, 'run', side_effect=AssertionError('conversion'))):
                        next_key = self.prepare(quant='q4_k_m', quantizer=binary, cache_key_only=True)
                        reused = self.prepare(quant='q4_k_m', quantizer=binary, offline=True)
                    self.assertEqual(next_key['cache_key'], exported['cache_key'])
                    self.assertTrue(reused['cached'])

    def test_cache_key_cli_emits_same_identity_json(self):
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(self.manifest))
        output = io.StringIO()
        with (mock.patch.object(module.sys, 'argv', ['prepare-minimax.py', '--models-root', str(self.root),
                                                    '--manifest', str(manifest), '--cache-key-only']),
              mock.patch.object(module.sys, 'stdout', output)):
            self.assertEqual(module.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['cache_key'], self.prepare()['cache_key'])

    def test_bad_download_cannot_publish_source_or_model(self):
        for content in (b'bad', b'x' * 200):
            with self.subTest(content=content), mock.patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(content)):
                with self.assertRaises(module.PreparationError) as error:
                    self.prepare()
                self.assertEqual(error.exception.stage, 'download')
                self.assertEqual(list(self.root.rglob('*.safetensors')), [])
                self.assertEqual(list(self.root.rglob('.download-*')), [])
                self.assertEqual(list(self.root.rglob('provenance.json')), [])

    def test_offline_missing_sources_and_verify_only_do_not_download(self):
        for kwargs in ({'offline': True}, {'verify_only': True}):
            with self.subTest(kwargs=kwargs), mock.patch.object(module.urllib.request, 'urlopen', side_effect=AssertionError('network')):
                with self.assertRaises(module.PreparationError):
                    self.prepare(**kwargs)

    def test_converter_failure_leaves_no_ready_pair(self):
        def fail(command, **kwargs):
            self.convert(command, **kwargs)
            raise subprocess.CalledProcessError(2, command)
        with mock.patch.object(module.subprocess, 'run', side_effect=fail):
            with self.assertRaises(module.PreparationError) as error:
                self.prepare()
        self.assertEqual(error.exception.stage, 'conversion')
        self.assertEqual(list(self.root.rglob('*.gguf')), [])
        self.assertEqual(list(self.root.rglob('provenance.json')), [])

    def test_unpinned_paths_and_symlinks_are_rejected(self):
        for value in ('../escape', '/absolute', 'bad\\name'):
            fixture = copy.deepcopy(self.manifest)
            fixture['files'][0]['path'] = value
            with self.assertRaises(ValueError):
                module.validate(fixture)
        fixture = copy.deepcopy(self.manifest)
        fixture['revision'] = 'main'
        with self.assertRaises(ValueError):
            module.validate(fixture)
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'minimax-sources').symlink_to(outside)
        with self.assertRaisesRegex(module.PreparationError, 'symlink'):
            self.prepare()

    def test_preflight_failure_precedes_download(self):
        with mock.patch.object(module, 'preflight', side_effect=module.PreparationError('preflight', 'insufficient memory')):
            with self.assertRaisesRegex(module.PreparationError, 'insufficient memory'):
                self.prepare()
        self.assertEqual(self.calls, [])

    def test_real_resource_guard_checks_disk_and_memory(self):
        usage = mock.Mock(free=0)
        with mock.patch.object(module.shutil, 'disk_usage', return_value=usage):
            with self.assertRaisesRegex(module.PreparationError, 'free disk'):
                real_preflight(self.manifest, self.root, 100)
        usage.free = 1000
        with (mock.patch.object(module.shutil, 'disk_usage', return_value=usage),
              mock.patch.object(module, 'memory_bytes', return_value=0)):
            with self.assertRaisesRegex(module.PreparationError, 'memory'):
                real_preflight(self.manifest, self.root, 100)

    def test_same_size_source_corruption_is_rejected_offline(self):
        result = self.prepare()
        record = json.loads(Path(result['provenance']).read_text())
        source = Path(record['source_dir']) / 'lm.safetensors'
        source.write_bytes(b'x' * source.stat().st_size)
        (Path(result['model_dir']) / 'mm3-lm-q8_0.gguf').unlink()
        with self.assertRaisesRegex(module.PreparationError, 'offline source cache missing or corrupt'):
            self.prepare(offline=True)

    def test_interrupted_download_removes_partial_file(self):
        class Interrupted(io.BytesIO):
            def read(self, size):
                if self.tell():
                    raise OSError('connection interrupted')
                return super().read(4)
        with mock.patch.object(module.urllib.request, 'urlopen', return_value=Interrupted(b'lm' * 8)):
            with self.assertRaisesRegex(module.PreparationError, 'connection interrupted'):
                self.prepare()
        self.assertEqual(list(self.root.rglob('.download-*')), [])
        self.assertEqual(list(self.root.rglob('*.safetensors')), [])

    def test_q4_quantizer_and_license_provenance(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        result = self.prepare(quant='q4_k_m', quantizer=binary)
        record = json.loads(Path(result['provenance']).read_text())
        self.assertEqual(record['identity']['quantizer_sha256'], module.digest(binary))
        self.assertEqual(record['identity']['quantizer_libraries'][self.library.name]['sha256'], module.digest(self.library))
        self.assertEqual(len(record['commands']), 3)
        self.assertEqual(record['commands'][1][-1], 'Q4_K_M')
        self.assertIn('mm3-synth-q4_k_m.gguf', record['commands'][2][-2])
        self.assertEqual(record['source_checkout']['commit'], 'c' * 40)
        self.assertEqual(list(Path(result['model_dir']).glob('*f16*')), [])
        (Path(result['model_dir']) / 'LICENSE').write_text('changed')
        with self.assertRaises(module.PreparationError):
            self.prepare(quant='q4_k_m', quantizer=binary, verify_only=True)

    def test_cross_host_prepared_pair_without_local_dependencies_or_quantizer(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        result = self.prepare(quant='q4_k_m', quantizer=binary)
        binary.unlink()
        self.library.unlink()
        with mock.patch.object(module, 'dependency_identity', side_effect=AssertionError('local dependencies')):
            reused = self.prepare(quant='q4_k_m', prepared_dir=Path(result['model_dir']))
        self.assertTrue(reused['cached'])
        (Path(result['model_dir']) / 'mm3-synth-q4_k_m.gguf').unlink()
        with self.assertRaises(module.PreparationError):
            self.prepare(quant='q4_k_m', prepared_dir=Path(result['model_dir']))

    def test_shared_library_changes_invalidate_cache_with_same_quantizer(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        result = self.prepare(quant='q4_k_m', quantizer=binary)
        self.library.write_bytes(b'new quantization implementation')
        with self.assertRaisesRegex(module.PreparationError, 'missing, incomplete, corrupt'):
            self.prepare(quant='q4_k_m', quantizer=binary, verify_only=True)
        rebuilt = self.prepare(quant='q4_k_m', quantizer=binary, offline=True)
        self.assertNotEqual(result['cache_key'], rebuilt['cache_key'])
        with self.assertRaisesRegex(module.PreparationError, 'quantizer-library'):
            self.prepare(quant='q4_k_m', quantizer=binary, quantizer_libraries=[])

    def test_namespaced_ggml_libraries_are_hashed_and_invalidate_cache(self):
        binary = self.root / 'quantizer'
        binary.write_bytes(b'fake quantizer')
        binary.chmod(0o755)
        for filename in ('libqvac-speech-ggml-base.0.10.2.dylib',
                         'libqvac-speech-ggml-base.so.0.10.2'):
            with self.subTest(filename=filename):
                library = self.root / filename
                library.write_bytes(b'quantization implementation')
                arguments = dict(quant='q4_k_m', quantizer=binary,
                                 quantizer_libraries=[library])
                identity = self.prepare(**arguments, cache_key_only=True)
                result = self.prepare(**arguments)
                self.assertEqual(identity['cache_key'], result['cache_key'])
                record = json.loads(Path(result['provenance']).read_text())
                self.assertEqual(record['identity']['quantizer_libraries'][filename]['sha256'],
                                 hashlib.sha256(library.read_bytes()).hexdigest())
                self.assertTrue(self.prepare(**arguments, offline=True)['cached'])
                library.write_bytes(b'updated quantization implementation')
                changed = self.prepare(**arguments, cache_key_only=True)
                self.assertNotEqual(identity['cache_key'], changed['cache_key'])

    def test_unverified_native_candidates_are_rejected_in_root_and_mm3(self):
        result = self.prepare(quant='f16')
        target = Path(result['model_dir'])
        for relative in ('mm3-lm-q8_0.gguf', 'mm3/MM3-SYNTH-F16.GGUF', 'MM3-LM-F16.GGUF'):
            with self.subTest(relative=relative):
                extra = target / relative
                extra.parent.mkdir(exist_ok=True)
                extra.write_bytes(b'GGUF' + b'extra' * 20)
                with self.assertRaisesRegex(module.PreparationError, 'unverified MiniMax candidates'):
                    self.prepare(quant='f16', offline=True)
                with self.assertRaises(module.PreparationError):
                    self.prepare(quant='f16', prepared_dir=target)
                extra.unlink()


if __name__ == '__main__':
    unittest.main()
