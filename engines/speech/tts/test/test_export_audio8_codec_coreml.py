#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-audio8-codec-coreml.py.

A tiny synthesis stack is built from random tensors in the GGUF's own naming
and layout, so the checks need no checkpoint: the phase-form transposed
convolution must equal torch's ConvTranspose1d cropped to the causal window
(codec_ops.cpp's definition); the whole stack must be causal, which is what
lets the engine drop each window's leading context and zero-pad the last
window on the right; and the sidecar stem rule must match coreml_path.cpp.
Requires torch; skips without it, mirroring the converter tests.
"""
import importlib.util
import pathlib
import unittest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'export-audio8-codec-coreml.py'


def load_exporter():
    spec = importlib.util.spec_from_file_location('export_audio8_codec_coreml', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LATENT = 8
STRIDES = [2, 4, 2, 2]
DILATIONS = [1, 3, 9]
CHANNELS = [12, 10, 6, 4]  # after each decoder stage
FRAME = 4 * 2 * 4 * 2 * 2  # upsample 2 * 2, then the decoder strides


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def tiny_tensors(rng):
    """Random tensors in the decoder GGUF's names and ggml layouts: full
    kernels [taps, out, in], depthwise kernels [taps, channels], pointwise
    projections as Linear [out, in], snake alphas [channels]."""
    t = {}
    for i in range(2):
        t[f'q/up/{i}/0/conv/weight'] = rand(rng, 4, LATENT, LATENT)
        t[f'q/up/{i}/0/conv/bias'] = rand(rng, LATENT)
        t[f'q/up/{i}/1/dwconv/conv/weight'] = rand(rng, 7, LATENT)
        t[f'q/up/{i}/1/dwconv/conv/bias'] = rand(rng, LATENT)
        t[f'q/up/{i}/1/norm/weight'] = 1.0 + rand(rng, LATENT)
        t[f'q/up/{i}/1/norm/bias'] = rand(rng, LATENT)
        t[f'q/up/{i}/1/pwconv1/weight'] = rand(rng, 3 * LATENT, LATENT)
        t[f'q/up/{i}/1/pwconv1/bias'] = rand(rng, 3 * LATENT)
        t[f'q/up/{i}/1/pwconv2/weight'] = rand(rng, LATENT, 3 * LATENT)
        t[f'q/up/{i}/1/pwconv2/bias'] = rand(rng, LATENT)
        t[f'q/up/{i}/1/gamma'] = rand(rng, LATENT)
    width = 16
    t['dec/0/conv/weight'] = rand(rng, 7, width, LATENT)
    t['dec/0/conv/bias'] = rand(rng, width)
    for n, (stride, out_ch) in enumerate(zip(STRIDES, CHANNELS)):
        prefix = f'dec/{n + 1}/block'
        t[prefix + '/0/alpha'] = np.abs(rand(rng, width)) + 0.5
        t[prefix + '/1/conv/weight'] = rand(rng, 2 * stride, out_ch, width)
        t[prefix + '/1/conv/bias'] = rand(rng, out_ch)
        for r in range(len(DILATIONS)):
            unit = f'{prefix}/{2 + r}/block'
            t[unit + '/0/alpha'] = np.abs(rand(rng, out_ch)) + 0.5
            t[unit + '/1/conv/weight'] = rand(rng, 7, out_ch, out_ch)
            t[unit + '/1/conv/bias'] = rand(rng, out_ch)
            t[unit + '/2/alpha'] = np.abs(rand(rng, out_ch)) + 0.5
            t[unit + '/3/conv/weight'] = rand(rng, 1, out_ch, out_ch)
            t[unit + '/3/conv/bias'] = rand(rng, out_ch)
        width = out_ch
    tail = len(STRIDES) + 1
    t[f'dec/{tail}/alpha'] = np.abs(rand(rng, width)) + 0.5
    t[f'dec/{tail + 1}/conv/weight'] = rand(rng, 7, 1, width)
    t[f'dec/{tail + 1}/conv/bias'] = rand(rng, 1)
    return t


def tiny_hparams():
    return {
        'latent_dim': LATENT, 'frame_size': FRAME, 'sample_rate': 44100,
        'snake_epsilon': 1e-9, 'convnext_norm_eps': 1e-6,
        'decoder_strides': STRIDES, 'residual_dilations': DILATIONS,
        'upsample_stages': 2, 'upsample_stride': 2,
    }


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class SynthesisStackTest(unittest.TestCase):
    def setUp(self):
        self.exporter = load_exporter()
        self.rng = np.random.default_rng(1234)
        torch.manual_seed(1234)  # the causality tolerances assume a fixed draw
        self.model = self.exporter.SynthesisStack(tiny_tensors(self.rng), tiny_hparams()).eval()

    def test_phase_transpose_matches_conv_transpose(self):
        # codec_ops.cpp: output column l * stride + p sums W[g * stride + p] x[l - g],
        # which is torch's conv_transpose1d(padding=0) cropped to L * stride columns.
        for stride, taps in ((2, 4), (4, 8), (8, 16), (2, 6)):
            kernel = rand(self.rng, taps, 5, 3)  # [taps, out, in]
            bias = rand(self.rng, 5)
            module = self.exporter.CausalConvTranspose(kernel, bias, stride)
            x = torch.randn(1, 3, 11)
            with torch.no_grad():
                got = module(x)
                weight = torch.from_numpy(kernel.transpose(2, 1, 0).copy())  # [in, out, taps]
                want = F.conv_transpose1d(x, weight, torch.from_numpy(bias), stride=stride)[..., :11 * stride]
            self.assertEqual(tuple(got.shape), (1, 5, 11 * stride), (stride, taps))
            self.assertTrue(torch.allclose(got, want, atol=1e-5), (stride, taps))

    def test_output_shape(self):
        x = torch.randn(1, LATENT, 6)
        with torch.no_grad():
            self.assertEqual(tuple(self.model(x).shape), (1, 1, 6 * FRAME))

    def test_stack_is_causal(self):
        # A change at frame f must leave every sample before f * FRAME alone,
        # and zeros appended on the right must not change what came before:
        # both are what coreml_windows.h relies on.
        x = torch.randn(1, LATENT, 12)
        changed = x.clone()
        changed[..., 7:] += 1.0
        padded = F.pad(x, (0, 5))
        with torch.no_grad():
            base, moved, longer = self.model(x), self.model(changed), self.model(padded)
        self.assertTrue(torch.allclose(base[..., :7 * FRAME], moved[..., :7 * FRAME], atol=1e-4))
        self.assertFalse(torch.allclose(base[..., 7 * FRAME:], moved[..., 7 * FRAME:], atol=1e-3))
        # torch picks different conv algorithms for different lengths, and the
        # random-weight stack amplifies their rounding to a few 1e-4 by the
        # tanh output; a genuine non-causal leak is orders of magnitude larger.
        self.assertTrue(torch.allclose(base, longer[..., :12 * FRAME], atol=2e-3))

    def test_context_drop_matches_full_pass(self):
        # A window starting mid-utterance reproduces the full pass once its
        # leading frames are dropped: the receptive field of the tiny stack is
        # small, so a generous drop must agree to rounding while the very
        # first kept frame of a zero-dropped window must not.
        x = torch.randn(1, LATENT, 40)
        with torch.no_grad():
            full = self.model(x)
            window = self.model(x[..., 10:])
        drop = 24
        self.assertTrue(torch.allclose(full[..., (10 + drop) * FRAME:], window[..., drop * FRAME:], atol=1e-5))
        self.assertFalse(torch.allclose(full[..., 10 * FRAME:11 * FRAME], window[..., :FRAME], atol=1e-3))

    def test_fold_matches_flat(self):
        # Folding long stages into rows is exact: every causal convolution takes
        # its left context from the previous row, so the folded stack equals
        # the flat one to rounding, for a stack that folds once (at the stage
        # whose output first exceeds the row) and for one that folds on entry.
        x = torch.randn(1, LATENT, 12)
        with torch.no_grad():
            flat = self.model(x)
            for rows in (64, 128, 256):
                self.exporter.FOLD_ROW = rows
                try:
                    folded = self.model(x)
                finally:
                    self.exporter.FOLD_ROW = 0
                self.assertEqual(tuple(folded.shape), tuple(flat.shape), rows)
                self.assertTrue(torch.allclose(folded, flat, atol=1e-4), rows)

    def test_fold_helpers(self):
        exporter = self.exporter
        x = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
        exporter.FOLD_ROW = 4
        try:
            f = exporter.fold(x)
            self.assertEqual(tuple(f.shape), (2, 3, 4, 4))
            self.assertTrue(torch.equal(exporter.unfold(f), x))
            # left context of 2: row r starts with the last 2 samples of row r - 1, row 0 with zeros
            c = exporter.causal_context(f, 2)
            self.assertEqual(tuple(c.shape), (2, 3, 4, 6))
            self.assertTrue(torch.equal(c[0, 0, 0, :2], torch.zeros(2)))
            self.assertTrue(torch.equal(c[0, 0, 1, :2], x[0, 0, 2:4]))
            self.assertTrue(torch.equal(c[0, 0, 1, 2:], x[0, 0, 4:8]))
            # a sequence that does not divide, or is not longer than a row, stays flat
            self.assertEqual(exporter.fold(torch.zeros(1, 3, 18)).dim(), 3)
            self.assertEqual(exporter.fold(torch.zeros(1, 3, 4)).dim(), 3)
        finally:
            exporter.FOLD_ROW = 0

    def test_polynomial_snake_matches_sin(self):
        # The range-reduced polynomial sine agrees with torch.sin over the
        # activation range the stack sees (|alpha x| stays below ~10 on the
        # reference decode) and well beyond it.
        exporter = self.exporter
        alpha = np.abs(rand(self.rng, 5)) + 0.5
        snake = exporter.Snake(alpha, 1e-9)
        x = torch.linspace(-60.0, 60.0, 20001).repeat(5, 1)[None]
        with torch.no_grad():
            exporter.Snake.mode = 'sin'
            want = snake(x)
            exporter.Snake.mode = 'poly'
            try:
                got = snake(x)
            finally:
                exporter.Snake.mode = 'sin'
        self.assertTrue(torch.allclose(got, want, atol=2e-5), (got - want).abs().max())


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class SidecarStemTest(unittest.TestCase):
    def test_matches_coreml_path_rule(self):
        stem = load_exporter().sidecar_stem
        self.assertEqual(stem('models/audio8-codec-decoder-q8_0.gguf'), 'audio8-codec-decoder')
        self.assertEqual(stem('models/audio8-codec-decoder-f32.gguf'), 'audio8-codec-decoder')
        self.assertEqual(stem('audio8-codec-decoder.Q8_0.gguf'), 'audio8-codec-decoder')
        self.assertEqual(stem('audio8-codec-decoder.gguf'), 'audio8-codec-decoder')
        self.assertEqual(stem('/opt/m.odels/decoder.gguf'), 'decoder')


if __name__ == '__main__':
    unittest.main()
