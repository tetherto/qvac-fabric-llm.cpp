#!/usr/bin/env python3
"""Model-free coverage of opt-in CLI selection, output isolation and scoring errors."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


@unittest.skipUnless(shutil.which('jq'), 'driver requires jq')
class MusicDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'result.json'
        self.models = self.root / 'models'
        self.models.mkdir()
        for name in ('Qwen3-Embedding.gguf', 'acestep-lm.gguf', 'vae.gguf', 'acestep-turbo.gguf', 'mm3-lm-f16.gguf', 'mm3-synth-f16.gguf'):
            (self.models/name).write_bytes(b'GGUFfixture')
        self.stub = self.root / 'engine'
        self.stub.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
log=pathlib.Path(os.environ['MUSIC_TEST_LOG'])
count=len(log.read_text().splitlines()) if log.exists() else 0
with log.open('a') as f: f.write(json.dumps(args)+'\\n')
print('generation stdout '+str(count))
print('generation stderr '+str(count), file=sys.stderr)
if os.environ.get('MUSIC_TEST_BACKEND'):
 print('[MM3] Using GPU backend '+os.environ['MUSIC_TEST_BACKEND']+' (device0); CPU handles unsupported ops', file=sys.stderr)
if os.environ.get('MUSIC_TEST_FAIL')=='1' and count==1: sys.exit(7)
out=pathlib.Path(args[args.index('--out')+1])
if '--mode' in args:
 assert args[args.index('--mode')+1]=='full'
 out=out/'audio.wav'
if os.environ.get('MUSIC_TEST_MISSING')=='1' and count==2: sys.exit(0)
out.write_bytes(str(count).encode())
''')
        self.stub.chmod(0o755)
        self.scorer = self.root / 'scorer-python'
        self.scorer.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
if os.environ.get('MUSIC_TEST_SLEEP')=='1':
 import time; time.sleep(3)
# Assert every generation invocation finished before the first score.
assert len(pathlib.Path(os.environ['MUSIC_TEST_LOG']).read_text().splitlines()) == 3
out=pathlib.Path(args[args.index('--json-out')+1])
if os.environ.get('MUSIC_TEST_BAD')=='1': out.write_text('{"status":"ok","score":2}')
else: out.write_text('{"status":"ok","score":0.25,"provenance":{"test":true}}')
''')
        self.scorer.chmod(0o755)
        self.log = self.root / 'calls.jsonl'

    def run_driver(self, family='acestep', enabled=True, **extra):
        spec = json.loads((HERE/'families.json').read_text())
        spec[family]['binary']='engine'
        spec_path = self.root/'families.json'
        spec_path.write_text(json.dumps(spec))
        env=dict(os.environ, BENCH_FAMILIES_JSON=str(spec_path),
                 MUSIC_ALIGNMENT='1' if enabled else '0', MUSIC_ALIGNMENT_MODEL_DIR=str(self.models),
                 MUSIC_CLAP_MODEL_DIR=str(self.models), MUSIC_ALIGNMENT_PYTHON=str(self.scorer),
                 MUSIC_TEST_LOG=str(self.log), MUSIC_CAPTION='Piano & drums ${literal}', **extra)
        cp=subprocess.run(['bash',str(HERE/'run-family.sh'),'--family',family,'--build-dir',str(self.root),
                           '--models-root',str(self.models),'--runs','2','--warmup','1','--out',str(self.out)],
                          cwd=ROOT,env=env,text=True,capture_output=True)
        self.assertEqual(cp.returncode,0,cp.stderr)
        return json.loads(self.out.read_text())

    def test_acestep_artifacts_and_postpass(self):
        r=self.run_driver()
        self.assertEqual(r['status'],'ok')
        self.assertEqual(r['music_alignment']['coverage'],{'scored':2,'expected':2})
        self.assertEqual(r['music_alignment']['score'],0.25)
        calls=[json.loads(s) for s in self.log.read_text().splitlines()]
        self.assertEqual(len({c[c.index('--out')+1] for c in calls}),3)
        self.assertEqual(calls[1][calls[1].index('--caption')+1],'Piano & drums ${literal}')
        for item in r['music_alignment']['runs']:
            d=Path(item['artifact_dir'])
            self.assertTrue((d/'audio.wav').exists())
            self.assertIn('generation stdout '+str(item['run']), (d/'generation.stdout.log').read_text())
            self.assertIn('generation stderr '+str(item['run']), (d/'generation.stderr.log').read_text())
            self.assertEqual(json.loads((d/'generation.json').read_text())['seed'],'42')
        self.assertIn('WAV writing included',r['notes'])

    def test_failed_generation_logs_survive_cleanup(self):
        r=self.run_driver(MUSIC_TEST_FAIL='1')
        self.assertEqual(r['status'],'run-failed')
        artifacts=next(self.root.glob('result.music.*'))
        for name, count in [('warmup-1', 0), ('run-1', 1)]:
            d=artifacts/name
            self.assertIn('generation stdout '+str(count), (d/'generation.stdout.log').read_text())
            self.assertIn('generation stderr '+str(count), (d/'generation.stderr.log').read_text())
        self.assertFalse((artifacts/'run-2').exists())

    def test_minimax_full_mode(self):
        r=self.run_driver('minimax')
        self.assertEqual(r['status'],'ok')
        self.assertEqual(r['music_alignment']['score'],0.25)
        self.assertIsNone(r['rtf_median'])
        generation=Path(r['music_alignment']['runs'][0]['artifact_dir'])/'generation.json'
        self.assertIsNone(json.loads(generation.read_text())['duration_seconds'])

    def test_prepared_minimax_bundle(self):
        report=self.root/'preparation.json'
        report.write_text(json.dumps({'status':'ok','stage':'ready',
                                     'model_dir':str(self.models),'quant':'f16'}))
        r=self.run_driver('minimax',MINIMAX_PREPARATION_REPORT=str(report))
        self.assertEqual(r['status'],'ok')
        self.assertEqual(r['model'],'minimax: f16')
        self.assertEqual(r['model_preparation']['model_dir'],str(self.models))

    def test_preparation_failure_preserves_stage_and_reason(self):
        report=self.root/'preparation.json'
        report.write_text(json.dumps({'status':'preparation-failed','stage':'resources',
                                     'reason':'Insufficient memory for conversion','model_dir':None}))
        r=self.run_driver('minimax',MINIMAX_PREPARATION_REPORT=str(report))
        self.assertEqual(r['status'],'preparation-failed')
        self.assertEqual(r['model_preparation']['stage'],'resources')
        self.assertEqual(r['music_alignment']['stage'],'preparation')
        self.assertIn('Insufficient memory',r['music_alignment']['reason'])
        self.assertIsNone(r['music_alignment']['score'])
        self.assertFalse(self.log.exists())

    def test_invalid_preparation_report_does_not_use_stale_local_pair(self):
        report=self.root/'preparation.json'
        report.write_text('{truncated')
        r=self.run_driver('minimax',MINIMAX_PREPARATION_REPORT=str(report))
        self.assertEqual(r['status'],'preparation-failed')
        self.assertFalse(self.log.exists())

    def test_successful_preparation_report_requires_existing_model_directory(self):
        missing = self.root / 'disappeared-models'
        missing.mkdir()
        missing.rmdir()
        for fields in ({}, {'model_dir': None}, {'model_dir': str(missing)}):
            with self.subTest(fields=fields):
                report = self.root / 'preparation.json'
                report.write_text(json.dumps({'status': 'ok', 'quant': 'f16', **fields}))
                result = self.run_driver('minimax', MINIMAX_PREPARATION_REPORT=str(report))
                self.assertEqual(result['status'], 'preparation-failed')
                self.assertEqual(result['model_preparation']['stage'], 'report')
                self.assertEqual(result['model_preparation']['reason'], 'Prepared MiniMax model directory is missing')
                self.assertEqual(result['music_alignment']['reason'], result['model_preparation']['reason'])
                self.assertIsNone(result['music_alignment']['score'])
                self.assertFalse(self.log.exists())

    def test_pilot_fingerprints_reject_model_changes_before_generation(self):
        cache = self.root / 'model-fingerprints.json'
        first = self.run_driver('minimax', MUSIC_MODEL_FINGERPRINT_CACHE=str(cache))
        self.assertEqual(first['status'], 'ok')
        self.assertTrue(cache.is_file())
        original_cache = cache.read_bytes()
        self.log.unlink()
        second = self.run_driver('minimax', MUSIC_MODEL_FINGERPRINT_CACHE=str(cache))
        self.assertEqual(second['status'], 'ok')
        self.assertEqual(cache.read_bytes(), original_cache)
        self.log.unlink()
        (self.models / 'mm3-lm-f16.gguf').write_bytes(b'GGUFchanged-model-payload')
        changed = self.run_driver('minimax', MUSIC_MODEL_FINGERPRINT_CACHE=str(cache))
        self.assertEqual(changed['status'], 'run-failed')
        self.assertFalse(self.log.exists())

    def test_minimax_native_gpu_log_attribution(self):
        for backend in ('CUDA','MTL0','Vulkan0'):
            with self.subTest(backend=backend):
                self.log.unlink(missing_ok=True)
                r=self.run_driver('minimax',MUSIC_DEVICE='gpu',MUSIC_TEST_BACKEND=backend)
                self.assertEqual(r['backend'],backend)
                d=Path(r['music_alignment']['runs'][0]['artifact_dir'])
                self.assertEqual(json.loads((d/'generation.json').read_text())['observed_backend'],backend)

    def test_missing_last_wav_is_partial(self):
        r=self.run_driver(MUSIC_TEST_MISSING='1')
        self.assertEqual(r['status'],'ok')
        self.assertEqual(r['music_alignment']['status'],'partial')
        self.assertEqual(r['music_alignment']['coverage']['scored'],1)
        self.assertEqual(r['music_alignment']['runs'][1]['status'],'invalid-audio')

    def test_bad_score_preserves_performance(self):
        r=self.run_driver(MUSIC_TEST_BAD='1')
        self.assertEqual(r['status'],'ok')
        self.assertIsNotNone(r['wall_ms_median'])
        self.assertIsNone(r['music_alignment']['score'])
        self.assertEqual(r['music_alignment']['coverage']['scored'],0)

    def test_missing_local_stages_is_explicit(self):
        for p in self.models.glob('*.gguf'): p.unlink()
        r=self.run_driver('minimax')
        self.assertEqual(r['status'],'missing-model')
        self.assertEqual(r['music_alignment']['status'],'model-missing')
        self.assertEqual(r['music_alignment']['coverage']['expected'],2)
        self.assertFalse(self.log.exists())

    def test_gpu_request_and_model_fingerprints(self):
        r=self.run_driver('minimax', MUSIC_DEVICE='gpu')
        calls=[json.loads(s) for s in self.log.read_text().splitlines()]
        self.assertEqual(calls[0][calls[0].index('--device')+1],'gpu')
        d=Path(r['music_alignment']['runs'][0]['artifact_dir'])
        generation=json.loads((d/'generation.json').read_text())
        self.assertEqual(generation['observed_backend'],'CPU')
        self.assertEqual(generation['provenance']['requested_device'],'gpu')
        self.assertEqual(len(generation['provenance']['binary']['sha256']),64)
        self.assertTrue(generation['provenance']['model_files'])

    def test_timeout_preserves_performance(self):
        r=self.run_driver(MUSIC_TEST_SLEEP='1', MUSIC_ALIGNMENT_TIMEOUT='0.05')
        self.assertEqual(r['status'],'ok')
        self.assertEqual(r['music_alignment']['runs'][0]['status'],'timeout')
        self.assertIsNone(r['music_alignment']['score'])

    def test_rerun_cannot_reuse_old_wavs(self):
        first=self.run_driver()
        self.log.unlink()
        second=self.run_driver(MUSIC_TEST_MISSING='1')
        self.assertNotEqual(first['music_alignment']['runs'][0]['artifact_dir'],second['music_alignment']['runs'][0]['artifact_dir'])
        self.assertEqual(second['music_alignment']['coverage']['scored'],1)

    def test_default_minimax_stays_missing(self):
        r=self.run_driver('minimax',enabled=False,MODEL_S3_BUCKET='test')
        self.assertEqual(r['status'],'missing-model')
        self.assertIsNone(r['music_alignment'])
        self.assertFalse(self.log.exists())


class SummaryTests(unittest.TestCase):
    def test_partial_and_missing_scores(self):
        spec=importlib.util.spec_from_file_location('summary',HERE/'summarize.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.fmt_music_alignment(None),'—')
        self.assertIn('0.2500 (partial; 1/2)',m.fmt_music_alignment({'status':'partial','score':.25,'coverage':{'scored':1,'expected':2}}))
        self.assertIn('CLAP —',m.fmt_music_alignment({'status':'ok','score':float('nan')}))


if __name__=='__main__': unittest.main()
