#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-moss-speech-tokenizer-coreml.py:
the PyTorch rebuild against an independent NumPy Whisper-VQ encoder on tiny
random tensors in the GGUF's names and layouts, the causality that makes
right-padding a short segment exact, the sidecar stem rule against
speech_tokenizer_sidecar_path, and (with coremltools) the converted program's
interface. Skips without numpy or torch.
"""
import importlib.util
import math
import pathlib
import sys
import unittest

try:
    import numpy as np
    import torch
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(0)

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "export-moss-speech-tokenizer-coreml.py"

N_MELS = 4
WIDTH = 8
HEADS = 2
FF = 16
POOL = 2
TOKENS = 4
FRAMES = TOKENS * POOL
MEL_FRAMES = FRAMES * 2
CTX = FRAMES + 3
LAYERS = 2
EPS = 1e-5
TOLERANCE = 1e-4
SEED = 26333
PREFIX = "whispervq."


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_moss_speech_tokenizer_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def tiny_hparams():
    return {"n_mels": N_MELS, "block_count": LAYERS, "embedding_length": WIDTH, "context_length": CTX,
            "pooling_kernel": POOL, "n_heads": HEADS, "tokens": TOKENS, "frames": FRAMES, "mel_frames": MEL_FRAMES}


def layer_tensors(rng, layer):
    prefix = f"{PREFIX}blk.{layer}."
    t = {prefix + "attn_norm.weight": 1.0 + rand(rng, WIDTH), prefix + "attn_norm.bias": rand(rng, WIDTH),
         prefix + "attn_k.weight": rand(rng, WIDTH, WIDTH),
         prefix + "ffn_norm.weight": 1.0 + rand(rng, WIDTH), prefix + "ffn_norm.bias": rand(rng, WIDTH),
         prefix + "ffn_up.weight": rand(rng, FF, WIDTH), prefix + "ffn_up.bias": rand(rng, FF),
         prefix + "ffn_down.weight": rand(rng, WIDTH, FF), prefix + "ffn_down.bias": rand(rng, WIDTH)}
    for name in ("attn_q", "attn_v", "attn_output"):
        t[prefix + name + ".weight"] = rand(rng, WIDTH, WIDTH)
        t[prefix + name + ".bias"] = rand(rng, WIDTH)
    return t


def tiny_tensors(rng):
    t = {PREFIX + "conv1.weight": rand(rng, WIDTH, N_MELS, 3), PREFIX + "conv1.bias": rand(rng, WIDTH, 1),
         PREFIX + "conv2.weight": rand(rng, WIDTH, WIDTH, 3), PREFIX + "conv2.bias": rand(rng, WIDTH, 1),
         PREFIX + "pos_embd": rand(rng, CTX, WIDTH)}
    for layer in range(LAYERS):
        t.update(layer_tensors(rng, layer))
    return t


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight.reshape(-1) + bias.reshape(-1)


def causal_conv(x, weight, bias, stride):
    padded = np.pad(x, ((0, 0), (2, 0)))
    out_len = (x.shape[1] - 1) // stride + 1
    taps = np.stack([padded[:, k:k + stride * out_len:stride] for k in range(3)], axis=-1)
    return gelu(np.einsum("ctk,ock->ot", taps, weight) + bias.reshape(-1, 1))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def causal_attention(x, t, prefix):
    q = x @ t[prefix + "attn_q.weight"].T + t[prefix + "attn_q.bias"]
    k = x @ t[prefix + "attn_k.weight"].T
    v = x @ t[prefix + "attn_v.weight"].T + t[prefix + "attn_v.bias"]
    head = WIDTH // HEADS
    future = np.triu(np.ones((x.shape[0], x.shape[0]), dtype=bool), k=1)
    heads = []
    for h in range(HEADS):
        cols = slice(h * head, (h + 1) * head)
        scores = np.where(future, -np.inf, q[:, cols] @ k[:, cols].T / math.sqrt(head))
        heads.append(softmax(scores) @ v[:, cols])
    return np.concatenate(heads, axis=-1) @ t[prefix + "attn_output.weight"].T + t[prefix + "attn_output.bias"]


def reference_block(x, t, layer):
    prefix = f"{PREFIX}blk.{layer}."
    x = x + causal_attention(layer_norm(x, t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"]), t, prefix)
    hidden = gelu(layer_norm(x, t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"])
                  @ t[prefix + "ffn_up.weight"].T + t[prefix + "ffn_up.bias"])
    return x + hidden @ t[prefix + "ffn_down.weight"].T + t[prefix + "ffn_down.bias"]


def reference_encoder(mel, t):
    x = causal_conv(mel, t[PREFIX + "conv1.weight"], t[PREFIX + "conv1.bias"], 1)
    x = causal_conv(x, t[PREFIX + "conv2.weight"], t[PREFIX + "conv2.bias"], 2).T + t[PREFIX + "pos_embd"][:FRAMES]
    for layer in range(LAYERS):
        x = reference_block(x, t, layer)
    return x.reshape(TOKENS, POOL, WIDTH).mean(axis=1)


class TokenizerEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        cls.tensors = tiny_tensors(rng)
        cls.mel = rand(rng, N_MELS, MEL_FRAMES)
        cls.model = cls.exporter.TokenizerEncoder(cls.tensors, tiny_hparams()).eval()

    def encode(self, mel):
        with torch.no_grad():
            return self.model(torch.from_numpy(mel)[None])[0].numpy()

    def test_matches_numpy_reference(self):
        got = self.encode(self.mel)
        self.assertEqual(got.shape, (TOKENS, WIDTH))
        np.testing.assert_allclose(got, reference_encoder(self.mel.astype(np.float64), self.tensors), atol=TOLERANCE)

    def test_right_padding_keeps_leading_tokens(self):
        kept = TOKENS // 2
        padded = self.mel.copy()
        padded[:, kept * POOL * 2:] = 0.0
        np.testing.assert_allclose(self.encode(padded)[:kept], self.encode(self.mel)[:kept], atol=TOLERANCE)

    def test_rejects_a_segment_past_the_context(self):
        hp = tiny_hparams()
        hp["context_length"] = FRAMES - 1
        with self.assertRaises(ValueError):
            self.exporter.validate_geometry(hp)


class SidecarStemTest(unittest.TestCase):
    def test_matches_speech_tokenizer_sidecar_path(self):
        exporter = load_exporter()
        cases = {"models/moss-speech-codec-f16.gguf": "moss-speech-codec-tokenizer",
                 "moss-speech-codec-Q8_0.gguf": "moss-speech-codec-tokenizer",
                 "moss-speech-codec.gguf": "moss-speech-codec-tokenizer"}
        for path, stem in cases.items():
            self.assertEqual(exporter.sidecar_stem(path), stem, path)


class CoremlContractTest(unittest.TestCase):
    def test_converted_interface(self):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            self.skipTest("coremltools is not installed")
        exporter = load_exporter()
        model = exporter.TokenizerEncoder(tiny_tensors(np.random.default_rng(SEED)), tiny_hparams()).eval()
        spec = exporter.convert_coreml(model, tiny_hparams(), "float16").get_spec()
        inputs = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
        self.assertEqual(inputs, {"mel": [1, N_MELS, MEL_FRAMES]})
        self.assertEqual([o.name for o in spec.description.output], ["states"])


if __name__ == "__main__":
    unittest.main()
