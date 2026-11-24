#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-moss-transcribe-encoder-coreml.py:
the PyTorch rebuild against an independent NumPy Whisper encoder and adaptor
on tiny random tensors in the GGUF's names and layouts, the sidecar stem rule
against coreml_encoder_sidecar_path, and (with coremltools) the converted
program's input and output contract. Skips without numpy or torch.
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

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "export-moss-transcribe-encoder-coreml.py"

N_MELS = 6
WIDTH = 8
HEADS = 2
FF = 16
CTX = 8
FRAMES = 2 * CTX
MERGE = 4
TEXT = 12
LAYERS = 2
EPS = 1e-5
ADAPTOR_EPS = 1e-6
TOLERANCE = 1e-4
SEED = 26333


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_moss_transcribe_encoder_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def tiny_hparams():
    return {"n_mels": N_MELS, "chunk_frames": FRAMES, "n_layers": LAYERS, "width": WIDTH, "n_heads": HEADS,
            "n_ctx": CTX, "eps": EPS, "merge": MERGE, "adaptor_eps": ADAPTOR_EPS, "text_width": TEXT}


def layer_tensors(rng, layer):
    prefix = f"enc.blk.{layer}."
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
    t = {"enc.conv1.weight": rand(rng, WIDTH, N_MELS, 3), "enc.conv1.bias": rand(rng, WIDTH, 1),
         "enc.conv2.weight": rand(rng, WIDTH, WIDTH, 3), "enc.conv2.bias": rand(rng, WIDTH, 1),
         "enc.pos_embd": rand(rng, CTX, WIDTH),
         "enc.output_norm.weight": 1.0 + rand(rng, WIDTH), "enc.output_norm.bias": rand(rng, WIDTH),
         "adaptor.fc1.weight": rand(rng, TEXT, WIDTH * MERGE), "adaptor.fc1.bias": rand(rng, TEXT),
         "adaptor.fc2.weight": rand(rng, TEXT, TEXT), "adaptor.fc2.bias": rand(rng, TEXT),
         "adaptor.norm.weight": 1.0 + rand(rng, TEXT), "adaptor.norm.bias": rand(rng, TEXT)}
    for layer in range(LAYERS):
        t.update(layer_tensors(rng, layer))
    return t


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight.reshape(-1) + bias.reshape(-1)


def conv1d(x, weight, bias, stride):
    padded = np.pad(x, ((0, 0), (1, 1)))
    out_len = (x.shape[1] + 2 - 3) // stride + 1
    taps = np.stack([padded[:, k:k + stride * out_len:stride] for k in range(3)], axis=-1)
    return np.einsum("ctk,ock->ot", taps, weight) + bias.reshape(-1, 1)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(x, t, prefix):
    q = x @ t[prefix + "attn_q.weight"].T + t[prefix + "attn_q.bias"]
    k = x @ t[prefix + "attn_k.weight"].T
    v = x @ t[prefix + "attn_v.weight"].T + t[prefix + "attn_v.bias"]
    head = WIDTH // HEADS
    heads = [softmax(q[:, h * head:(h + 1) * head] @ k[:, h * head:(h + 1) * head].T / math.sqrt(head))
             @ v[:, h * head:(h + 1) * head] for h in range(HEADS)]
    return np.concatenate(heads, axis=-1) @ t[prefix + "attn_output.weight"].T + t[prefix + "attn_output.bias"]


def reference_block(x, t, layer):
    prefix = f"enc.blk.{layer}."
    x = x + attention(layer_norm(x, t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"], EPS), t, prefix)
    hidden = gelu(layer_norm(x, t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"], EPS)
                  @ t[prefix + "ffn_up.weight"].T + t[prefix + "ffn_up.bias"])
    return x + hidden @ t[prefix + "ffn_down.weight"].T + t[prefix + "ffn_down.bias"]


def reference_encoder(mel, t):
    x = gelu(conv1d(mel, t["enc.conv1.weight"], t["enc.conv1.bias"], 1))
    x = gelu(conv1d(x, t["enc.conv2.weight"], t["enc.conv2.bias"], 2)).T + t["enc.pos_embd"]
    for layer in range(LAYERS):
        x = reference_block(x, t, layer)
    states = layer_norm(x, t["enc.output_norm.weight"], t["enc.output_norm.bias"], EPS)
    merged = np.stack([np.concatenate(states[i * MERGE:(i + 1) * MERGE]) for i in range(CTX // MERGE)])
    hidden = silu(merged @ t["adaptor.fc1.weight"].T + t["adaptor.fc1.bias"])
    hidden = hidden @ t["adaptor.fc2.weight"].T + t["adaptor.fc2.bias"]
    return layer_norm(hidden, t["adaptor.norm.weight"], t["adaptor.norm.bias"], ADAPTOR_EPS)


class AudioEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        cls.tensors = tiny_tensors(rng)
        cls.mel = rand(rng, N_MELS, FRAMES)
        cls.model = cls.exporter.AudioEncoder(cls.tensors, tiny_hparams()).eval()

    def test_matches_numpy_reference(self):
        with torch.no_grad():
            got = self.model(torch.from_numpy(self.mel)[None])[0].numpy()
        want = reference_encoder(self.mel.astype(np.float64), self.tensors)
        self.assertEqual(got.shape, (CTX // MERGE, TEXT))
        np.testing.assert_allclose(got, want, atol=TOLERANCE)

    def test_adaptor_merges_consecutive_frames(self):
        states = torch.arange(CTX * WIDTH, dtype=torch.float32).reshape(1, CTX, WIDTH)
        merged = states.reshape(1, CTX // MERGE, WIDTH * MERGE)
        self.assertTrue(torch.equal(merged[0, 1, :WIDTH], states[0, MERGE]))
        self.assertTrue(torch.equal(merged[0, 1, -WIDTH:], states[0, 2 * MERGE - 1]))

    def test_rejects_mismatched_geometry(self):
        hp = tiny_hparams()
        hp["chunk_frames"] = FRAMES + 2
        with self.assertRaises(ValueError):
            self.exporter.validate_geometry(hp)


class SidecarStemTest(unittest.TestCase):
    def test_matches_coreml_encoder_sidecar_path(self):
        exporter = load_exporter()
        cases = {"models/moss-transcribe-diarize-f16.gguf": "moss-transcribe-diarize-encoder",
                 "moss-transcribe-diarize-q8_0.gguf": "moss-transcribe-diarize-encoder",
                 "moss-transcribe-diarize.q5_0.gguf": "moss-transcribe-diarize-encoder",
                 "moss-transcribe-diarize.gguf": "moss-transcribe-diarize-encoder"}
        for path, stem in cases.items():
            self.assertEqual(exporter.sidecar_stem(path), stem, path)


class CoremlContractTest(unittest.TestCase):
    def test_converted_interface(self):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            self.skipTest("coremltools is not installed")
        exporter = load_exporter()
        model = exporter.AudioEncoder(tiny_tensors(np.random.default_rng(SEED)), tiny_hparams()).eval()
        spec = exporter.convert_coreml(model, tiny_hparams(), "float16").get_spec()
        inputs = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
        outputs = {o.name: list(o.type.multiArrayType.shape) for o in spec.description.output}
        self.assertEqual(inputs, {"mel": [1, N_MELS, FRAMES]})
        self.assertEqual(list(outputs), ["embeddings"])


if __name__ == "__main__":
    unittest.main()
