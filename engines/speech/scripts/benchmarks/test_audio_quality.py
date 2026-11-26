#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import warnings
import wave
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

quality = load('compute-audio-quality')
prepare = load('prepare-audio-quality')


class AudioQualityTests(unittest.TestCase):
    def test_analytic_sisdr_and_invariance(self):
        reference = [1., -1., 1., -1.]
        estimate = [1.5, -.5, .5, -1.5]
        expected = 10 * math.log10(4)
        self.assertAlmostEqual(quality.sisdr(reference, estimate), expected)
        self.assertAlmostEqual(quality.sisdr(reference, [-3*x+7 for x in estimate]), expected)

    def test_undefined_sisdr(self):
        for ref, hyp in [([], []), ([0, 0], [1, -1]), ([1, -1], [0, 0]),
                         ([1, -1], [1, -1]), ([1, -1], [math.nan, 1]),
                         ([1, -1], [1]), ([1, -1, 1, -1], [1, 1, -1, -1])]:
            with self.subTest(ref=ref, hyp=hyp), self.assertRaises(ValueError):
                quality.sisdr(ref, hyp)

    def test_noise_determinism_and_snr(self):
        reference = [0.1*math.sin(i/13) for i in range(16000)]
        clean, noisy, scale = prepare.noisy_pair(reference, 10, 42)
        self.assertEqual((clean, noisy, scale), prepare.noisy_pair(reference, 10, 42))
        measured = 10 * math.log10(sum(x*x for x in clean) / sum((x-y)**2 for x,y in zip(clean,noisy)))
        self.assertAlmostEqual(measured, 10)
        self.assertEqual(len(noisy), len(reference))
        self.assertNotEqual(noisy, prepare.noisy_pair(reference, 10, 43)[1])

    def test_clipping_avoided_for_both_members(self):
        clean, noisy, scale = prepare.noisy_pair([.99, -.99]*200, 0, 42)
        self.assertLess(scale, 1)
        self.assertLess(max(map(abs, noisy)), 1)
        self.assertLess(max(map(abs, clean)), 1)

    def test_cli_and_failure_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, hyp, out = [Path(directory)/name for name in ['ref.wav','hyp.wav','result.json']]
            samples = [.1*math.sin(i/13) for i in range(8000)]
            prepare.write_wav(ref, samples, 16000)
            noisy = prepare.noisy_pair(samples, 10, 42)[1]
            prepare.write_wav(hyp, noisy, 16000)
            subprocess.run([sys.executable, str(ROOT/'compute-audio-quality.py'), '--reference', str(ref),
                            '--hypothesis-file', str(hyp), '--metrics', 'sisdr', '--json-out', str(out)],
                           check=True, capture_output=True)
            result = json.loads(out.read_text())
            self.assertEqual(result['metrics']['sisdr']['status'], 'ok')
            with patch.object(quality, 'compute_stoi', side_effect=ImportError('pystoi')):
                missing = quality.score(ref, hyp, ['sisdr','stoi'])
            self.assertEqual(missing['metrics']['sisdr']['status'], 'ok')
            self.assertEqual(missing['metrics']['stoi']['status'], 'unavailable')
            prepare.write_wav(hyp, [0]*8000, 16000)
            silent = quality.score(ref,hyp,['sisdr'])['metrics']['sisdr']
            self.assertEqual(silent['status'], 'error')
            self.assertIn('silent', silent['reason'])
            prepare.write_wav(hyp, noisy[:-1], 16000)
            self.assertIn('sample counts differ', quality.score(ref,hyp,['sisdr'])['metrics']['sisdr']['reason'])
            absent = quality.score(ref, Path(directory)/'absent.wav', ['sisdr'])
            self.assertEqual(absent['metrics']['sisdr']['status'], 'error')

    def test_prepare_cli_preserves_frames_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, clean, noisy = [Path(directory)/name for name in ['ref.wav','clean.wav','noisy.wav']]
            prepare.write_wav(ref, [.9, -.9]*4000, 22050)
            subprocess.run([sys.executable, str(ROOT/'prepare-audio-quality.py'), '--reference', str(ref),
                            '--output', str(noisy), '--reference-output', str(clean)], check=True, capture_output=True)
            for path in [clean, noisy]:
                samples, rate = quality.read_wav(path)
                self.assertEqual(len(samples), 8000)
                self.assertEqual(rate, 22050)

    def test_pcm_widths_and_stereo_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pcm.wav'
            for width in (1,2,3,4):
                limit = 1 << (8*width-1)
                values = [-limit, 0, limit-1]
                raw = b''.join((x + (128 if width == 1 else 0)).to_bytes(width, 'little', signed=width != 1) for x in values)
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(width)
                    wav.setframerate(8000)
                    wav.writeframes(raw)
                samples, rate = quality.read_wav(path)
                self.assertEqual(samples, [x/limit for x in values])
                self.assertEqual(rate, 8000)
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(bytes(8))
            with self.assertRaisesRegex(ValueError, 'mono'):
                quality.read_wav(path)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy unavailable')
    def test_stoi_warning_is_not_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, hyp = Path(directory)/'ref.wav', Path(directory)/'hyp.wav'
            samples = [.1*math.sin(i/13) for i in range(8000)]
            prepare.write_wav(ref, samples, 8000)
            prepare.write_wav(hyp, prepare.noisy_pair(samples, 10, 42)[1], 8000)
            def short_stoi(*args, **kwargs):
                warnings.warn('not enough frames', RuntimeWarning)
                return 1e-5
            with patch.dict(sys.modules, {'pystoi': types.SimpleNamespace(stoi=short_stoi)}):
                metric = quality.score(ref,hyp,['stoi'])['metrics']['stoi']
            self.assertEqual(metric['status'], 'error')
            self.assertIsNone(metric['value'])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy unavailable')
    def test_cross_rate_known_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, hyp = Path(directory)/'ref.wav', Path(directory)/'hyp.wav'
            prepare.write_wav(ref, [.1*math.sin(2*math.pi*300*i/8000) for i in range(8000)], 8000)
            prepare.write_wav(hyp, [.1*math.sin(2*math.pi*300*i/16000) for i in range(16000)], 16000)
            result = quality.score(ref,hyp,['sisdr'])
            self.assertEqual(result['scoring_sample_rate'], 16000)
            self.assertEqual(result['metrics']['sisdr']['status'], 'ok')
            self.assertGreater(result['metrics']['sisdr']['value'], 40)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy unavailable')
    def test_resample_duration(self):
        self.assertEqual(len(quality.resample([.1]*8000, 8000, 16000)), 16000)

    @unittest.skipUnless(importlib.util.find_spec('pystoi'), 'pystoi unavailable')
    def test_real_stoi(self):
        # A varying envelope produces nonconstant band envelopes used by STOI.
        ref = [.2*(.6+.4*math.sin(i/800))*math.sin(i/8) + .05*math.sin(i/3.1) for i in range(32000)]
        noisy = prepare.noisy_pair(ref, -5, 42)[1]
        perfect = quality.compute_stoi(ref, ref, 16000)
        degraded = quality.compute_stoi(ref, noisy, 16000)
        self.assertAlmostEqual(perfect, 1, places=5)
        self.assertLess(degraded, perfect)
        with self.assertRaises(RuntimeWarning):
            quality.compute_stoi(ref[:1000], ref[:1000], 16000)


if __name__ == '__main__':
    unittest.main()
