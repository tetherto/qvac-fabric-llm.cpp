#!/usr/bin/env python3
"""Model-free encoder tests: padding invariance, dilation, relative attention,
style conditioning, and traced runtime lengths. Needs numpy and torch.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    print('SKIP: numpy and torch required')
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('encoder_export', SCRIPTS / 'export-supertonic-encoder-coreml.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def random_tensors(seed=7):
    rng = np.random.default_rng(seed)
    tensors = {}
    def value(name, shape, scale=0.03):
        tensors[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    def linear(prefix, hidden=256, inputs=256, matmul=False):
        value(prefix + '.weight', (inputs, hidden) if matmul else (hidden, inputs, 1))
        value(prefix + '.bias', (hidden,))
    def norm(prefix):
        tensors[prefix + '.weight'] = np.ones(256, np.float32)
        value(prefix + '.bias', (256,))
    value(exporter.TEXT + 'text_embedder.char_embedder.weight', (32, 256), 0.5)
    for i in range(6):
        prefix = exporter.TEXT + f'convnext.convnext.{i}'
        value(prefix + '.dwconv.weight', (256, 1, 7), 0.1)
        value(prefix + '.dwconv.bias', (256,))
        norm(prefix + '.norm.norm')
        linear(prefix + '.pwconv1', hidden=64)
        linear(prefix + '.pwconv2', inputs=64)
        value(prefix + '.gamma', (256,), 0.2)
    for i in range(4):
        prefix = exporter.TEXT + f'attn_encoder.attn_layers.{i}'
        for name in ('q', 'k', 'v', 'o'):
            linear(prefix + '.conv_' + name)
        for name in ('k', 'v'):
            value(prefix + '.emb_rel_' + name, (1, 9, 64), 0.2)
        for n in (1, 2):
            norm(exporter.TEXT + f'attn_encoder.norm_layers_{n}.{i}.norm')
        prefix = exporter.TEXT + f'attn_encoder.ffn_layers.{i}'
        linear(prefix + '.conv_1', hidden=64)
        linear(prefix + '.conv_2', inputs=64)
    for i in (1, 2):
        prefix = exporter.SPEECH + f'attention{i}'
        for name in ('W_query', 'W_value', 'out_fc'):
            linear(prefix + '.' + name + '.linear', matmul=True)
        value(f'text_encoder:/speech_prompted_text_encoder/attention{i}/tanh/Tanh_output_0', (2, 1, 128, 50), 0.2)
    norm(exporter.SPEECH + 'norm.norm')
    return tensors


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tensors = random_tensors()

    def test_padding_matches_exact_length_for_all_dilations(self):
        rng = np.random.default_rng(12)
        for dilations in ([1] * 6, [1, 1, 2, 2, 4, 4]):
            padded = exporter.TextEncoder(self.tensors, dilations, 32).eval()
            for length in (1, 2, 9, 17, 31, 32):
                with self.subTest(dilations=dilations, length=length), torch.no_grad():
                    exact = exporter.TextEncoder(self.tensors, dilations, length).eval()
                    ids = torch.from_numpy(rng.integers(0, 32, (1, 32), dtype=np.int32))
                    style = torch.from_numpy(rng.standard_normal((1, 50, 256)).astype(np.float32))
                    size = torch.tensor([length], dtype=torch.int32)
                    want = exact(ids[:, :length], size, style)
                    got = padded(ids, size, style)[..., :length]
                    torch.testing.assert_close(got, want, atol=3e-6, rtol=3e-5)

    def test_relative_attention_matches_independent_numpy(self):
        rng = np.random.default_rng(4)
        t = self.tensors
        layer = exporter.AttentionLayer(t, 0).eval()
        x = rng.standard_normal((1, 5, 256)).astype(np.float32)
        prefix = exporter.TEXT + 'attn_encoder.attn_layers.0'
        def linear(x, p):
            return x @ t[p + '.weight'].reshape(t[p + '.weight'].shape[0], -1).T + t[p + '.bias']
        q, k, v = [linear(x, prefix + '.conv_' + n).reshape(5, 4, 64).transpose(1, 0, 2) for n in ('q', 'k', 'v')]
        scores = q @ k.transpose(0, 2, 1) / 8
        rk = t[prefix + '.emb_rel_k'].reshape(9, 64)
        rv = t[prefix + '.emb_rel_v'].reshape(9, 64)
        for i in range(5):
            for j in range(5):
                scores[:, i, j] += (q[:, i] * rk[j-i+4]).sum(-1) / 8
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        y = probs @ v
        for i in range(5):
            for j in range(5):
                y[:, i] += probs[:, i, j, None] * rv[j-i+4]
        y = y.transpose(1, 0, 2).reshape(1, 5, 256)
        def norm(x, p):
            return (x-x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True)+1e-6) * t[p+'.weight'] + t[p+'.bias']
        x = norm(x + linear(y, prefix+'.conv_o'), exporter.TEXT+'attn_encoder.norm_layers_1.0.norm')
        ff = exporter.TEXT + 'attn_encoder.ffn_layers.0'
        want = norm(x + linear(np.maximum(linear(x, ff+'.conv_1'), 0), ff+'.conv_2'), exporter.TEXT+'attn_encoder.norm_layers_2.0.norm')
        positions = torch.arange(5)
        bands = torch.stack([(positions[None, :]-positions[:, None] == i-4).float() for i in range(9)])
        with torch.no_grad():
            # Recreate the original input, which the NumPy reference reassigned.
            original = np.random.default_rng(4).standard_normal((1, 5, 256)).astype(np.float32)
            got = layer(torch.from_numpy(original), torch.zeros(1, 1, 1, 5), bands).numpy()
        np.testing.assert_allclose(got, want, atol=2e-6, rtol=3e-5)

    def test_trace_keeps_length_and_style_dynamic(self):
        model = exporter.TextEncoder(self.tensors, [1, 1, 2, 2, 4, 4], 32).eval()
        ids = torch.arange(32, dtype=torch.int32)[None, :]
        style = torch.zeros(1, 50, 256)
        with torch.no_grad():
            traced = torch.jit.trace(model, (ids, torch.tensor([16], dtype=torch.int32), style))
            for length in (1, 9, 32):
                size = torch.tensor([length], dtype=torch.int32)
                torch.testing.assert_close(traced(ids, size, style), model(ids, size, style))
            changed = torch.ones_like(style)
            self.assertGreater((traced(ids, size, style)-traced(ids, size, changed)).abs().max().item(), 0.01)


if __name__ == '__main__':
    unittest.main()
