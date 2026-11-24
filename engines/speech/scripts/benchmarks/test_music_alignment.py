#!/usr/bin/env python3
"""Model-free tests for the frozen music diagnostic policy."""
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import music_alignment as music


class MusicAlignmentTests(unittest.TestCase):
    def test_wav_integer_normalization_and_stereo_shape(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.wav'
            for dtype, values, expected in (
                (np.uint8, [0, 128, 255], [-1., 0., 127 / 128]),
                (np.int16, [-32768, 0, 32767], [-1., 0., 32767 / 32768]),
                (np.int32, [-2147483648, 0, 2147483647], [-1., 0., 2147483647 / 2147483648]),
            ):
                with self.subTest(dtype=dtype):
                    pcm = np.array([values, values], dtype=dtype).T
                    wavfile.write(path, 44100, pcm)
                    audio, rate = music.read_wav(path)
                    self.assertEqual(rate, 44100)
                    self.assertEqual(audio.shape, (3, 2))
                    self.assertEqual(audio.dtype, np.float64)
                    np.testing.assert_array_equal(audio[:, 0], expected)
                    np.testing.assert_array_equal(audio[:, 1], expected)

    def test_wav_pcm24_uses_left_aligned_scale(self):
        # scipy writes int32 as 32-bit PCM; use the standard WAV writer to
        # exercise an actual packed three-byte PCM payload.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pcm24.wav'
            values = [-8388608, -1, 0, 1, 8388607]
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(3)
                stream.setframerate(48000)
                stream.writeframes(b''.join(x.to_bytes(3, 'little', signed=True) for x in values))
            audio, rate = music.read_wav(path)
            self.assertEqual(audio.shape, (5, 1))
            np.testing.assert_array_equal(audio[:, 0], np.array(values) / 8388608)

    def test_float_wav_preserves_amplitude_and_invalid_values(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'float.wav'
            for dtype in (np.float32, np.float64):
                source = np.array([-.25, .5, 1.5], dtype=dtype)
                wavfile.write(path, 48000, source)
                audio, rate = music.read_wav(path)
                np.testing.assert_array_equal(audio[:, 0], source)
                self.assertEqual(audio.shape, (3, 1))
                for invalid in (float('nan'), float('inf')):
                    wavfile.write(path, 48000, np.array([invalid], dtype=dtype))
                    audio, rate = music.read_wav(path)
                    with self.assertRaises(music.AlignmentError):
                        music.validate_audio(audio, rate)

    def test_wav_rejects_other_containers_corrupt_and_compressed(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audio.wav'
            for payload in (b'', b'not a wav', b'fLaC' + b'\0' * 32):
                path.write_bytes(payload)
                with self.assertRaises(music.AlignmentError) as error:
                    music.read_wav(path)
                self.assertEqual(error.exception.status, 'invalid-audio')
            wavfile.write(path, 48000, np.ones(8, dtype=np.int16))
            original = path.read_bytes()
            path.write_bytes(original[:-4])
            with self.assertRaises(music.AlignmentError):
                music.read_wav(path)
            payload = bytearray(original)
            payload[20:22] = (7).to_bytes(2, 'little')  # mu-law is unsupported
            path.write_bytes(payload)
            with self.assertRaises(music.AlignmentError):
                music.read_wav(path)

    def test_windows_preserve_tail_and_do_not_repeat_short_audio(self):
        windows = list(music.fixed_windows(np.arange(7, dtype=np.float32), 3))
        self.assertEqual([(x[0], x[1]) for x in windows], [(0, 3), (3, 3), (6, 1)])
        np.testing.assert_array_equal(windows[-1][2], [6, 0, 0])
        self.assertEqual(len(list(music.fixed_windows(np.ones(6), 3))), 2)

    def test_duration_weighted_aggregation(self):
        self.assertAlmostEqual(music.weighted_score([
            {'score': .3, 'valid_samples': 10}, {'score': .9, 'valid_samples': 2}]), .4)
        for value in (float('nan'), float('inf'), 1.01):
            with self.assertRaises(music.AlignmentError):
                music.bounded_cosine(value)
        self.assertEqual(music.bounded_cosine(1.0000001), 1)

    def test_audio_rejects_empty_zero_nonfinite_and_stereo_cancellation(self):
        for audio in (np.array([]), np.zeros(8), np.array([float('nan')]),
                      np.array([float('inf')]), np.array([[.5, -.5], [.1, -.1]])):
            with self.subTest(audio=audio), self.assertRaises(music.AlignmentError) as error:
                music.validate_audio(audio, 48000)
            self.assertEqual(error.exception.status, 'invalid-audio')

    def test_audio_diagnostics_without_normalizing_quiet_material(self):
        audio = np.array([[.002, .004], [.004, .006]])
        mono, info = music.validate_audio(audio, 2)
        np.testing.assert_allclose(mono, [.003, .005])
        self.assertEqual(info['duration_seconds'], 1)
        self.assertEqual(info['channels'], 2)
        self.assertEqual(info['clipping_fraction'], 0)
        self.assertAlmostEqual(info['peak'], .006)

    def test_resampler_preserves_duration_and_rejects_out_of_band_aliases(self):
        rate = 96000
        t = np.arange(rate) / rate
        low = music.resample_to(np.sin(2 * np.pi * 1000 * t), rate)
        high = music.resample_to(np.sin(2 * np.pi * 35000 * t), rate)
        self.assertEqual(len(low), 48000)
        self.assertGreater(float(np.sqrt(np.mean(low ** 2))), .7)
        self.assertLess(float(np.sqrt(np.mean(high[100:-100] ** 2))), .005)

    def test_caption_limit_counts_special_tokens_and_never_truncates(self):
        class Tokenizer:
            model_max_length = 5
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {'input_ids': [0] + text.split() + [2]}
        class Processor:
            tokenizer = Tokenizer()
        processor = Processor()
        self.assertEqual(music.validate_caption(processor, 'one two three'), 5)
        self.assertFalse(processor.tokenizer.kwargs['truncation'])
        for text in ('', ' ', 'one two three four'):
            with self.assertRaises(music.AlignmentError):
                music.validate_caption(processor, text)

    def test_scoring_uses_all_windows_and_caption_once(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch unavailable')
        class Tokenizer:
            model_max_length = 512
            def __call__(self, text, **kwargs):
                return {'input_ids': [0, 1, 2]}
        class Processor:
            tokenizer = Tokenizer()
            def __init__(self):
                self.audio = []
            def __call__(self, **kwargs):
                if 'text' in kwargs:
                    return {'input_ids': torch.tensor([[1]])}
                self.audio.append(kwargs['audio'][0])
                return {'input_features': torch.tensor([[len(self.audio)]])}
        class Model:
            calls = 0
            def get_text_features(self, **kwargs):
                self.calls += 1
                return torch.tensor([[1., 0.]])
            def get_audio_features(self, **kwargs):
                if kwargs['input_features'].item() == 1:
                    return torch.tensor([[1., 0.]])
                return torch.tensor([[0., 1.]])
        model, processor = Model(), Processor()
        result = music.score_loaded_audio(model, processor, np.ones(720000), 48000, 'piano')
        self.assertAlmostEqual(result['score'], 2 / 3)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(processor.audio), 2)
        np.testing.assert_array_equal(processor.audio[1][240000:], 0)
        with self.assertRaises(music.AlignmentError):
            music.embedding(torch.zeros(1, 2))
        with self.assertRaises(music.AlignmentError):
            music.embedding(torch.tensor([[float('nan'), 1.]]))

    def make_model(self, root):
        files = {}
        for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                     'tokenizer.json', 'model.safetensors'):
            (root / name).write_bytes(b'{}')
            files[name] = hashlib.sha256(b'{}').hexdigest()
        manifest = {'model_id': 'example/model', 'revision': 'a' * 40, 'files': files,
                    'sampling_rate': 48000, 'window_samples': 480000}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_local_model_integrity_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, manifest = self.make_model(root)
            self.assertEqual(music.verify_model(root, manifest_path), manifest)
            (root / 'config.json').write_text('changed')
            with self.assertRaisesRegex(music.AlignmentError, 'checksum'):
                music.verify_model(root, manifest_path)
            manifest['revision'] = 'main'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(music.AlignmentError, 'immutable'):
                music.verify_model(root, manifest_path)

    def test_reject_unverified_loader_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _ = self.make_model(root)
            (root / 'added_tokens.json').write_text('{}')
            with self.assertRaisesRegex(music.AlignmentError, 'unverified'):
                music.verify_model(root, manifest_path)

    def test_failure_is_null_and_does_not_download(self):
        result = music.score_file('missing.wav', 'piano', '/nonexistent/model', '/nonexistent/manifest')
        self.assertEqual(result['status'], 'model-missing')
        self.assertIsNone(result['score'])
        with patch.object(music, 'load_clap_local', side_effect=ImportError('missing torch')):
            result = music.score_file('missing.wav', 'piano', '', '')
        self.assertEqual(result['status'], 'dependency-missing')
        self.assertIsNone(result['score'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
