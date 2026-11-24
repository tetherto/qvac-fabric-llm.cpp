#!/usr/bin/env python3
import copy
import importlib.util
import unittest
import json
import tempfile
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('pilot_summary', Path(__file__).with_name('summarize-music-pilot.py'))
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


RUN_SPEC = importlib.util.spec_from_file_location('pilot_runner', Path(__file__).with_name('run-music-pilot.py'))
runner = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(runner)


def fixture():
    return {'family': 'acestep', 'manifest_sha256': 'fixed',
            'configuration': {'checkpoint_identity': 'source-v1', 'generation_settings': {'duration': 20}},
            'manifest': {'prompts': [{'id': 'p1', 'subset': 'primary'}, {'id': 'p2', 'subset': 'primary'},
                                     {'id': 's1', 'subset': 'stress'}],
                         'seeds': [1, 2, 3], 'bootstrap': {'seed': 24846, 'replicates': 1000}},
            'records': [{'prompt_id': p, 'seed': s, 'status': 'ok', 'score': score, 'scorer_provenance': {'revision': 'v1'}}
                        for p, s, score in [('p1', 1, .2), ('p1', 2, .4), ('p1', 3, .6), ('p2', 1, .8), ('s1', 1, -1)]]}


class PilotTests(unittest.TestCase):
    def test_equal_prompt_weights_and_partial_coverage(self):
        result = summary.summarize(fixture())['subsets']
        self.assertAlmostEqual(result['primary']['equal_prompt_mean'], .6)
        self.assertEqual(result['primary']['successful'], 4)
        self.assertEqual(result['primary']['expected'], 6)
        self.assertTrue(result['primary']['partial'])
        self.assertEqual(result['stress']['equal_prompt_mean'], -1)
        self.assertEqual(len(result['primary']['prompts'][1]['missing']), 2)

    def test_pair_first_then_average_and_bootstrap(self):
        baseline = fixture()
        candidate = copy.deepcopy(baseline)
        candidate['records'][0]['score'] += .3
        candidate['records'][3]['score'] -= .2
        result = summary.summarize(candidate, baseline)['subsets']['primary']
        self.assertAlmostEqual(result['equal_prompt_paired_delta'], -.05)
        self.assertEqual(result['paired_count'], 4)
        self.assertEqual(result['paired_prompt_bootstrap_95'], summary.summarize(candidate, baseline)['subsets']['primary']['paired_prompt_bootstrap_95'])
        self.assertTrue(result['paired_partial'])

    def test_null_invalid_never_zero(self):
        pilot = fixture()
        for row in pilot['records']:
            row['score'] = float('nan')
        result = summary.summarize(pilot)['subsets']['primary']
        self.assertEqual(result['successful'], 0)
        self.assertIsNone(result['equal_prompt_mean'])

    def test_incompatible_rejected(self):
        for change in ['family', 'manifest_sha256', 'checkpoint_identity', 'scorer_provenance']:
            candidate = fixture()
            if change in candidate:
                candidate[change] = 'different'
            elif change == 'checkpoint_identity':
                candidate['configuration'][change] = 'different'
            else:
                candidate['records'][0][change] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.summarize(candidate, fixture())

    def test_mixed_provenance_rejected_without_baseline(self):
        pilot = fixture()
        pilot['records'][1]['scorer_provenance'] = {'revision': 'v2'}
        with self.assertRaises(ValueError):
            summary.summarize(pilot)

    def test_duplicate_and_unexpected_rejected(self):
        candidate = fixture()
        candidate['records'].append(candidate['records'][0])
        with self.assertRaises(ValueError):
            summary.summarize(candidate)
        candidate = fixture()
        candidate['records'][0]['seed'] = 999
        with self.assertRaises(ValueError):
            summary.summarize(candidate)

    def test_runner_applies_settings_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            config.write_text(json.dumps({'name': 'candidate', 'checkpoint_identity': 'source-v1',
                                          'backend': 'GPU', 'build_options': 'test'}))
            models = root / 'models'
            models.mkdir()
            for name in ('text-embedding.gguf', 'vae.gguf', 'acestep-lm.gguf', 'dit.gguf'):
                (models / name).write_bytes(b'GGUFfixture')
            calls = []
            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                if '--family' in command:
                    env = kwargs['env']
                    self.assertEqual(env['MUSIC_DEVICE'], 'gpu')
                    self.assertEqual(env['MUSIC_DURATION'], '20')
                    self.assertEqual(env['MUSIC_SEED'], '17')
                    self.assertIn('acoustic guitar', env['MUSIC_CAPTION'])
                    fingerprints = Path(env['MUSIC_MODEL_FINGERPRINT_CACHE'])
                    self.assertEqual(fingerprints, root / 'out/model-fingerprints.json')
                    self.assertEqual(len(json.loads(fingerprints.read_text())['model_files']), 4)
                    destination = Path(command[-1])
                    artifacts = destination.with_suffix('.music.test') / 'run-1'
                    artifacts.mkdir(parents=True)
                    (artifacts / 'audio.wav').write_bytes(b'test WAV retained')
                    score = {'status': 'ok', 'score': .5, 'provenance': {'revision': 'v1'},
                             'policy_version': 'v1', 'artifact_dir': str(artifacts)}
                    destination.write_text(json.dumps({'status': 'ok', 'music_alignment': {'runs': [score]}}))
                else:
                    destination = Path(command[command.index('--json-out') + 1])
                    destination.write_text(json.dumps({'status': 'ok', 'score': .4}))
                return 0
            argv = ['run-music-pilot.py', '--family', 'acestep', '--configuration', str(config),
                    '--build-dir', str(root / 'build'), '--scorer-model-dir', str(root / 'clap'),
                    '--model-dir', str(root / 'models'), '--out-dir', str(root / 'out'),
                    '--limit', '1', '--controls', '--repeat-generation']
            with patch('sys.argv', argv), patch.object(runner, 'run_bounded', side_effect=fake_run):
                runner.main()
            pilot = json.loads((root / 'out/pilot.json').read_text())
            self.assertEqual(len(pilot['records']), 81)
            self.assertEqual(pilot['records'][0]['score'], .5)
            self.assertEqual(pilot['records'][1]['status'], 'not-run')
            self.assertEqual(len(pilot['controls']), 2)
            self.assertEqual(len(calls), 4)
            self.assertIn('benchmark_source_hashes', pilot['configuration'])
            self.assertIsNotNone(pilot['controls'][0]['mismatch_prompt_id'])
            repeated = pilot['controls'][1]
            self.assertEqual(repeated['status'], 'ok')
            self.assertTrue(repeated['wav_bytes_equal'])
            self.assertEqual(repeated['original_wav_sha256'], repeated['repeat_wav_sha256'])
            self.assertEqual(repeated['score_delta'], 0)
            self.assertEqual(repeated['score_comparison_status'], 'ok')
            self.assertEqual(repeated['original_result'], pilot['records'][0]['performance_result'])
            self.assertEqual(repeated['result'], json.loads((root / 'out/p01-17/repeat-result.json').read_text()))
            reported = summary.summarize(pilot)['controls'][1]
            self.assertEqual(reported['score_delta'], 0)
            self.assertTrue(reported['wav_bytes_equal'])

    def test_startup_fingerprints_use_preparation_report_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / 'prepared'
            models.mkdir()
            for role in ('lm', 'synth'):
                (models / f'mm3-{role}-f16.gguf').write_bytes(b'GGUFfixture')
            report = root / 'preparation.json'
            report.write_text(json.dumps({'status': 'ok', 'model_dir': str(models)}))
            cache = root / 'fingerprints.json'
            args = SimpleNamespace(family='minimax', models_root=root / 'models')
            env = {'MINIMAX_PREPARATION_REPORT': str(report),
                   'MUSIC_ALIGNMENT_MODEL_DIR': str(root / 'ignored'),
                   'MUSIC_MODEL_FINGERPRINT_CACHE': str(cache)}
            runner.initialize_model_fingerprints(args, env)
            self.assertEqual(json.loads(cache.read_text())['snapshot']['root'], str(models))

    def test_invalid_driver_score_rejected(self):
        for value in (True, float('nan'), float('inf'), 1.1, None, '0.2'):
            self.assertEqual(runner.normalize_score({'status': 'ok', 'score': value})['status'], 'scorer-error')
        self.assertIsNone(runner.normalize_score({'status': 'timeout', 'score': .2})['score'])

    def test_record_failure_retains_null_score_and_skips_controls(self):
        for failure in ('missing-result', 'timeout'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = SimpleNamespace(family='acestep', build_dir=root / 'build', models_root=root / 'models',
                                       generation_timeout=1, controls=True, repeat_generation=True)
                row = {'prompt_id': 'p1', 'seed': 1, 'status': 'not-run', 'score': None}
                effect = subprocess.TimeoutExpired('driver', 1) if failure == 'timeout' else None
                with patch.object(runner, 'run_bounded', side_effect=effect):
                    runner.run_record(row, root / 'p1-1', args, {})
                self.assertEqual(row['status'], 'timeout' if failure == 'timeout' else 'run-failed')
                self.assertIsNone(row['score'])
                self.assertGreaterEqual(row['elapsed_seconds'], 0)
                pilot = {'controls': []}
                state = {'categories': set(), 'repeated': False}
                runner.run_controls({}, row, root / 'p1-1', args, {}, pilot, state)
                self.assertEqual(pilot['controls'], [])
                self.assertEqual(state, {'categories': set(), 'repeated': False})

    def test_cohort_controls_once_per_primary_category_and_repeat_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(family='minimax', build_dir=root / 'build', models_root=root / 'models',
                                   out_dir=root, generation_timeout=1, limit=7, controls=True, repeat_generation=True)
            prompts = [{'id': identifier, 'category': category, 'subset': subset,
                        'caption': identifier, 'lyrics': ''}
                       for identifier, category, subset in [('p1', 'a', 'primary'), ('p2', 'a', 'primary'),
                                                             ('p3', 'b', 'primary'), ('s1', 'c', 'stress')]]
            pilot = {'manifest': {'prompts': prompts, 'seeds': [1, 2]},
                     'configuration': {'generation_settings': {'max_frames': 123}}, 'records': [], 'controls': []}
            env = {'MUSIC_DEVICE': 'cpu'}
            generated = []

            def fake_run(command, **kwargs):
                destination = Path(command[-1])
                generated.append((kwargs['env'].copy(), destination.name))
                self.assertEqual(kwargs['env']['MUSIC_MAX_FRAMES'], '123')
                self.assertEqual(kwargs['env']['MUSIC_CAPTION'], destination.parent.name.rsplit('-', 1)[0])
                self.assertEqual(kwargs['env']['MUSIC_SEED'], destination.parent.name.rsplit('-', 1)[1])
                if len(generated) > 2:
                    saved = json.loads((root / 'pilot.json').read_text())
                    self.assertTrue(all(row['status'] == 'ok' for row in saved['records']))
                artifacts = destination.with_suffix('.music.test') / 'run-1'
                artifacts.mkdir(parents=True)
                (artifacts / 'audio.wav').write_bytes(b'retained audio')
                runner.write(destination, {'status': 'ok', 'music_alignment': {
                    'runs': [{'status': 'ok', 'score': .5}]}})
                return 0

            with patch.object(runner, 'run_bounded', side_effect=fake_run), \
                    patch.object(runner, 'score_control', return_value={'status': 'ok', 'score': .4}) as score:
                runner.run_cohort(args, pilot, env)
            self.assertEqual(score.call_count, 4)
            self.assertEqual(len(generated), 8)
            self.assertEqual(generated[0][0], generated[1][0])
            self.assertEqual(generated[1][1], 'repeat-result.json')
            self.assertEqual(env, {'MUSIC_DEVICE': 'cpu'})
            self.assertEqual([control.get('category') for control in pilot['controls']], ['a', None, 'b'])
            saved = json.loads((root / 'pilot.json').read_text())
            self.assertEqual(saved, pilot)
            self.assertEqual(len(saved['records']), 8)
            self.assertEqual(saved['records'][-1]['status'], 'not-run')

    def test_repeat_generation_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(family='acestep', build_dir=root / 'build', models_root=root / 'models',
                                   generation_timeout=1)
            with patch.object(runner, 'run_bounded', side_effect=subprocess.TimeoutExpired('driver', 1)):
                control = runner.repeat_generation({'prompt_id': 'p1', 'seed': 1}, root, args, {})
            self.assertEqual(control['status'], 'timeout')
            self.assertEqual(control['prompt_id'], 'p1')
            self.assertIsNone(control['wav_bytes_equal'])
            self.assertIsNone(control['score_delta'])
            self.assertEqual(control['score_comparison_status'], 'unavailable')
            self.assertEqual(control['result']['status'], 'scorer-error')

    def test_repeat_generation_evidence_and_failed_controls(self):
        cases = ('different-wav', 'different-policy', 'different-provenance', 'missing-policy',
                 'invalid-score', 'missing-wav', 'missing-result', 'failed-driver', 'nonzero-exit')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = SimpleNamespace(family='minimax', build_dir=root / 'build', models_root=root / 'models',
                                       generation_timeout=1)
                original = root / 'result.music.test/run-1/audio.wav'
                original.parent.mkdir(parents=True)
                original.write_bytes(b'original WAV')
                score = {'status': 'ok', 'score': .25, 'policy_version': 'v1', 'provenance': {'revision': 'v1'}}
                row = {'prompt_id': 'p1', 'seed': 1, 'score_artifact': score,
                       'performance_result': {'status': 'ok', 'music_alignment': {'runs': [score]}}}

                def fake_run(command, **kwargs):
                    destination = Path(command[-1])
                    repeated = copy.deepcopy(score)
                    repeated['score'] = .5
                    if case == 'different-policy':
                        repeated['policy_version'] = 'v2'
                    elif case == 'different-provenance':
                        repeated['provenance']['revision'] = 'v2'
                    elif case == 'missing-policy':
                        repeated.pop('policy_version')
                    elif case == 'invalid-score':
                        repeated['score'] = 'invalid'
                    if case != 'missing-wav':
                        wav = destination.with_suffix('.music.test') / 'run-1/audio.wav'
                        wav.parent.mkdir(parents=True)
                        wav.write_bytes(b'changed WAV' if case == 'different-wav' else original.read_bytes())
                    if case != 'missing-result':
                        result = {'status': 'ok', 'music_alignment': {'runs': [repeated]}}
                        if case == 'failed-driver':
                            result = {'status': 'missing-model', 'notes': 'model unavailable'}
                        runner.write(destination, result)
                    return 1 if case == 'nonzero-exit' else 0

                with patch.object(runner, 'run_bounded', side_effect=fake_run):
                    control = runner.repeat_generation(row, root, args, {})
                self.assertEqual(control['original_wav_sha256'], runner.digest(original))
                self.assertEqual(control['original_result'], row['performance_result'])
                self.assertEqual(control['original_score'], score)
                if case == 'different-wav':
                    self.assertFalse(control['wav_bytes_equal'])
                    self.assertEqual(control['score_delta'], .25)
                    self.assertEqual(control['score_comparison_status'], 'ok')
                else:
                    self.assertIsNone(control['score_delta'])
                if case in ('different-policy', 'different-provenance', 'missing-policy'):
                    self.assertEqual(control['score_comparison_status'], 'incompatible-policy')
                expected_status = {'invalid-score': 'scorer-error', 'missing-wav': 'missing-audio',
                                   'missing-result': 'missing-result', 'failed-driver': 'missing-model',
                                   'nonzero-exit': 'run-failed'}.get(case, 'ok')
                self.assertEqual(control['status'], expected_status)
                if case == 'missing-wav':
                    self.assertIsNone(control['repeat_wav_sha256'])
                    self.assertIsNone(control['wav_bytes_equal'])

    def test_manifest_size(self):
        import json
        manifest = json.loads(Path(__file__).with_name('fixtures').joinpath('music-pilot-v1.json').read_text())
        self.assertEqual(sum(p['subset'] == 'primary' for p in manifest['prompts']), 24)
        self.assertEqual(len(manifest['seeds']), 3)
        self.assertEqual(len({p['id'] for p in manifest['prompts']}), len(manifest['prompts']))


if __name__ == '__main__':
    unittest.main()
