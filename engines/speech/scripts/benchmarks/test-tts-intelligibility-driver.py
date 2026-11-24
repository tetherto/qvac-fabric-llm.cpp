#!/usr/bin/env python3
"""Model-free integration checks for TTS synthesis -> reference ASR -> WER."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import wave

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_module('tts_prepare', 'prepare-tts-asr.py')
SUMMARY = load_module('tts_summary', 'summarize.py')
MODEL = b'synthetic pinned reference ASR model'
MODEL_SPEC = {'repo': 'test/reference', 'revision': 'a' * 40,
              'name': 'tiny.bin', 'sha256': hashlib.sha256(MODEL).hexdigest()}

SYNTH = r'''#!/usr/bin/env python3
import json, os, pathlib, struct, sys, wave
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
events = pathlib.Path(os.environ['STUB_EVENTS'])
previous = [json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
index = 1 + sum(x['kind'] == 'synth' for x in previous)
output = pathlib.Path(value('--wav-out'))
with events.open('a') as log:
    log.write(json.dumps({'kind': 'synth', 'index': index, 'text': value('--text'), 'output': str(output)}) + '\n')
mode = os.environ.get('STUB_AUDIO', 'all')
if mode == 'all' or (mode == 'warmup-only' and index == 1):
    with wave.open(str(output), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(struct.pack('<h', index) * 1600)
if '--json-out' in args:
    pathlib.Path(value('--json-out')).write_text(json.dumps({
        'inference_ms': {'median': 12.5, 'min': 11.0, 'max': 14.0},
        'backend': 'CPU'}))
print('using CPU backend', file=sys.stderr)
'''

ASR = r'''#!/usr/bin/env python3
import json, os, pathlib, struct, sys, wave
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
with wave.open(value('-f'), 'rb') as wav:
    marker = struct.unpack('<h', wav.readframes(1))[0]
with pathlib.Path(os.environ['STUB_EVENTS']).open('a') as log:
    log.write(json.dumps({'kind': 'asr', 'marker': marker, 'args': args}) + '\n')
print('reference ASR stub invoked')
if os.environ.get('STUB_ASR') == 'fail':
    print('synthetic decoder failure', file=sys.stderr)
    sys.exit(7)
pathlib.Path(value('-of') + '.txt').write_text(os.environ['STUB_TRANSCRIPT'])
'''


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tts driver spaces ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / 'build'
        (self.build / 'bin').mkdir(parents=True)
        self.models = self.root / 'models'
        cached = self.models / 'reference-asr' / MODEL_SPEC['sha256'] / MODEL_SPEC['name']
        cached.parent.mkdir(parents=True)
        cached.write_bytes(MODEL)
        self.reference = self.root / 'prompt.txt'
        self.reference.write_text('The quick brown fox jumps.')
        self.events = self.root / 'events.jsonl'
        self.output = self.root / 'result.json'
        self.spec = self.root / 'families.json'
        self.family = {'bench_kind': 'time-wrapped', 'binary': 'bin/synth',
                       'cmake_target': 'stub', 'models': [],
                       'args': ['--text', '${TTS_TEXT}', '--wav-out', '${TTS_AUDIO_OUT}'],
                       'correctness': {'kind': 'tts_intelligibility',
                                       'reference': str(self.reference),
                                       'asr_model': MODEL_SPEC},
                       'audio_duration_seconds': None}
        for filename, source in [('synth', SYNTH), ('whisper-cli', ASR)]:
            binary = self.build / 'bin' / filename
            binary.write_text(source)
            binary.chmod(0o755)
        self.env = dict(os.environ, STUB_EVENTS=str(self.events),
                        STUB_TRANSCRIPT=self.reference.read_text(),
                        BENCH_FAMILIES_JSON=str(self.spec))
        self.env.pop('MODEL_S3_BUCKET', None)
        self.env.pop('STUB_AUDIO', None)
        self.env.pop('STUB_ASR', None)

    def run_driver(self):
        self.spec.write_text(json.dumps({'stub-tts': self.family}))
        completed = subprocess.run(
            ['bash', str(HERE / 'run-family.sh'), '--family', 'stub-tts',
             '--runs', '2', '--warmup', '1', '--build-dir', str(self.build),
             '--models-root', str(self.models), '--out', str(self.output)],
            env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(self.output.read_text())

    def artifact(self, suffix):
        return self.root / ('result.tts' + suffix)

    def assert_perf(self, result):
        self.assertEqual(result['status'], 'ok')
        self.assertGreater(result['wall_ms_median'], 0)
        self.assertEqual(result['correctness_kind'], 'tts_intelligibility')

    def test_final_timed_audio_scored_once_and_artifacts(self):
        result = self.run_driver()
        self.assert_perf(result)
        score = result['tts_intelligibility']
        self.assertEqual(score['status'], 'ok')
        self.assertEqual(score['wer'], 0)
        self.assertEqual(score['n_ref_words'], 5)
        events = [json.loads(x) for x in self.events.read_text().splitlines()]
        self.assertEqual([x['kind'] for x in events], ['synth'] * 3 + ['asr'])
        self.assertEqual(events[-1]['marker'], 3)
        self.assertEqual(len({x['output'] for x in events[:3]}), 3)
        self.assertTrue(all(x['text'] == self.reference.read_text() for x in events[:3]))
        args = events[-1]['args']
        self.assertIn('-ng', args)
        self.assertEqual(args[args.index('-l') + 1], 'en')
        self.assertNotIn('--prompt', args)
        self.assertEqual(self.artifact('-reference.txt').read_bytes(), self.reference.read_bytes())
        self.assertEqual(self.artifact('-transcript.txt').read_text(), self.env['STUB_TRANSCRIPT'])
        self.assertIn('reference ASR stub invoked', self.artifact('-asr.log').read_text())
        self.assertEqual(json.loads(self.artifact('-intelligibility.json').read_text()), score)
        with wave.open(str(self.artifact('.wav'))) as wav:
            self.assertEqual(struct.unpack('<h', wav.readframes(1))[0], 3)
        self.assertIn('TTS WER 0.00%', SUMMARY.render_markdown([result]))

    def test_prompt_exact_bytes_with_multiline_and_shell_characters(self):
        self.reference.write_text('One & two $HOME `literal`.\nSecond line.\n\n')
        self.env['STUB_TRANSCRIPT'] = self.reference.read_text()
        result = self.run_driver()
        self.assert_perf(result)
        events = [json.loads(x) for x in self.events.read_text().splitlines()]
        self.assertTrue(all(x['text'] == self.reference.read_text()
                            for x in events if x['kind'] == 'synth'), events)

    def test_nonzero_wer_is_reported_without_quality_gate(self):
        self.env['STUB_TRANSCRIPT'] = 'The quick red fox jumps.'
        result = self.run_driver()
        self.assert_perf(result)
        self.assertAlmostEqual(result['tts_intelligibility']['wer'], 0.2)
        self.assertIn('TTS WER 20.00%', SUMMARY.render_markdown([result]))

    def test_missing_final_audio_cannot_reuse_warmup_or_stale_artifacts(self):
        self.run_driver()
        self.events.unlink()
        self.env['STUB_AUDIO'] = 'warmup-only'
        result = self.run_driver()
        self.assert_perf(result)
        self.assertEqual(result['tts_intelligibility']['status'], 'error')
        self.assertIsNone(result['tts_intelligibility']['wer'])
        self.assertFalse(self.artifact('.wav').exists())
        self.assertFalse(self.artifact('-transcript.txt').exists())
        self.assertNotIn('"kind": "asr"', self.events.read_text())

    def test_asr_failure_retains_performance_and_log(self):
        self.env['STUB_ASR'] = 'fail'
        result = self.run_driver()
        self.assert_perf(result)
        self.assertEqual(result['tts_intelligibility']['status'], 'error')
        self.assertIsNone(result['tts_intelligibility']['wer'])
        self.assertIn('synthetic decoder failure', self.artifact('-asr.log').read_text())
        self.assertIn('TTS WER error', SUMMARY.render_markdown([result]))

    def test_artifact_copy_failure_retains_performance(self):
        stubbin = self.root / 'stubbin'
        stubbin.mkdir()
        copy = stubbin / 'cp'
        copy.write_text('#!/bin/sh\necho "synthetic copy failure" >&2\nexit 1\n')
        copy.chmod(0o755)
        self.env['PATH'] = str(stubbin) + os.pathsep + self.env['PATH']
        result = self.run_driver()
        self.assert_perf(result)
        score = result['tts_intelligibility']
        self.assertEqual(score['status'], 'error')
        self.assertIsNone(score['wer'])
        self.assertIn('could not preserve', score['reason'])
        self.assertEqual(json.loads(self.artifact('-intelligibility.json').read_text()), score)
        self.assertNotIn('"kind": "asr"', self.events.read_text())

    def test_missing_asr_binary_retains_performance(self):
        (self.build / 'bin/whisper-cli').unlink()
        result = self.run_driver()
        self.assert_perf(result)
        self.assertEqual(result['tts_intelligibility']['status'], 'unavailable')
        self.assertIsNone(result['tts_intelligibility']['wer'])
        self.assertIn('TTS WER unavailable', SUMMARY.render_markdown([result]))

    def test_model_preparation_failure_retains_performance_and_wav(self):
        self.family['correctness']['asr_model'] = dict(MODEL_SPEC, revision='unpinned')
        result = self.run_driver()
        self.assert_perf(result)
        score = result['tts_intelligibility']
        self.assertEqual(score['status'], 'unavailable')
        self.assertIsNone(score['wer'])
        self.assertIn('pinned commit', score['reason'])
        self.assertTrue(self.artifact('.wav').is_file())
        self.assertEqual(json.loads(self.artifact('-intelligibility.json').read_text()), score)
        self.assertNotIn('"kind": "asr"', self.events.read_text())

    def test_native_mode_emits_and_scores_wav(self):
        self.family['bench_kind'] = 'native'
        self.family['args'] += ['--json-out', '${JSON_OUT}', '--runs', '${RUNS}',
                                '--warmup', '${WARMUP}']
        result = self.run_driver()
        self.assert_perf(result)
        self.assertEqual(result['wall_ms_median'], 12.5)
        self.assertEqual(result['tts_intelligibility']['wer'], 0)
        events = [json.loads(x) for x in self.events.read_text().splitlines()]
        self.assertEqual([x['kind'] for x in events], ['synth', 'asr'])
        self.assertEqual(events[-1]['marker'], 1)


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / 'reference-asr' / MODEL_SPEC['sha256'] / MODEL_SPEC['name']

    def test_verified_cache_does_not_download(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(MODEL)
        with mock.patch.object(PREPARE.urllib.request, 'urlopen') as download:
            self.assertEqual(PREPARE.prepare(MODEL_SPEC, self.root), self.target)
        download.assert_not_called()

    def test_corrupt_cache_is_replaced_only_by_verified_download(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'corrupt cached model')
        with mock.patch.object(PREPARE.urllib.request, 'urlopen', return_value=io.BytesIO(MODEL)):
            self.assertEqual(PREPARE.prepare(MODEL_SPEC, self.root), self.target)
        self.assertEqual(self.target.read_bytes(), MODEL)
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_wrong_hash_download_never_published(self):
        with mock.patch.object(PREPARE.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad response')):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                PREPARE.prepare(MODEL_SPEC, self.root)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_invalid_identity_fails_before_network(self):
        with mock.patch.object(PREPARE.urllib.request, 'urlopen') as download:
            for invalid in ({'revision': 'main'}, {'name': '../escape.bin'}, {'sha256': 'bad'}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    PREPARE.prepare(dict(MODEL_SPEC, **invalid), self.root)
        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
