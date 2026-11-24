#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-moss-codec-coreml.py: a tiny
decoder in the GGUF's names and layouts, decoded chunk by chunk through the
cached keys and values with the commit protocol the engine drives, must equal
one pass over the whole sequence and an independent NumPy decoder that rotates
interleaved pairs and masks the attention band the way codec.cpp does; the
rotary and mask tables must match the engine's examples; the sidecar stem rule
must match codec_decoder_sidecar_path; and (with coremltools) the converted
program must declare the inputs, output and states the engine binds.
Skips without numpy or torch.
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

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "export-moss-codec-coreml.py"

CODE_DIM = 6
D_MODEL = 8
HEADS = 2
FF = 12
CONTEXT = 5
PATCH = 2
FINAL_PATCH = 4
FRAMES = 11
CHUNK = 3
TOLERANCE = 1e-4
MASKED = -1e4
SEED = 26333


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_moss_codec_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def transformer_module(context):
    return {"kind": "Transformer", "d_model": D_MODEL, "heads": HEADS, "layers": 1, "context": context,
            "max_period": 10000.0}


def tiny_hparams():
    modules = [transformer_module(CONTEXT), {"kind": "PatchedPretransform", "patch": PATCH},
               transformer_module(2 * CONTEXT), {"kind": "PatchedPretransform", "patch": FINAL_PATCH}]
    return {"code_dim": CODE_DIM, "modules": modules}


def layer_tensors(rng, prefix):
    return {prefix + "attn_qkv.weight": rand(rng, 3 * D_MODEL, D_MODEL),
            prefix + "attn_output.weight": rand(rng, D_MODEL, D_MODEL),
            prefix + "ffn_up.weight": rand(rng, FF, D_MODEL), prefix + "ffn_down.weight": rand(rng, D_MODEL, FF),
            prefix + "attn_norm.weight": 1.0 + rand(rng, D_MODEL), prefix + "attn_norm.bias": rand(rng, D_MODEL),
            prefix + "ffn_norm.weight": 1.0 + rand(rng, D_MODEL), prefix + "ffn_norm.bias": rand(rng, D_MODEL),
            prefix + "attn_scale.scale": rand(rng, D_MODEL), prefix + "ffn_scale.scale": rand(rng, D_MODEL)}


def tiny_tensors(rng):
    t = {"blk.0.input_proj.weight": rand(rng, D_MODEL, CODE_DIM, 1),
         "blk.1.input_proj.weight": rand(rng, D_MODEL, D_MODEL // PATCH),
         "blk.1.output_proj.weight": rand(rng, FINAL_PATCH, D_MODEL)}
    t.update(layer_tensors(rng, "blk.0.layer.0."))
    t.update(layer_tensors(rng, "blk.1.layer.0."))
    return t


def layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5) * weight + bias


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def rope_interleaved(x):
    head_dim = x.shape[-1]
    out = x.copy()
    for pos in range(x.shape[0]):
        for i in range(head_dim // 2):
            theta = pos * 10000.0 ** (-2.0 * i / head_dim)
            a, b = x[pos, 2 * i], x[pos, 2 * i + 1]
            out[pos, 2 * i] = a * math.cos(theta) - b * math.sin(theta)
            out[pos, 2 * i + 1] = a * math.sin(theta) + b * math.cos(theta)
    return out


def band_attention(q, k, v, context):
    positions = np.arange(q.shape[0])
    delta = positions[:, None] - positions[None, :]
    scores = np.where((delta >= 0) & (delta < context), q @ k.T / math.sqrt(q.shape[-1]), -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True) @ v


def reference_layer(x, t, prefix, context):
    h = layer_norm(x, t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"])
    qkv = h @ t[prefix + "attn_qkv.weight"].T
    head = D_MODEL // HEADS
    heads = []
    for index in range(HEADS):
        cols = slice(index * head, (index + 1) * head)
        q = rope_interleaved(qkv[:, :D_MODEL][:, cols])
        k = rope_interleaved(qkv[:, D_MODEL:2 * D_MODEL][:, cols])
        heads.append(band_attention(q, k, qkv[:, 2 * D_MODEL:][:, cols], context))
    x = x + np.concatenate(heads, axis=-1) @ t[prefix + "attn_output.weight"].T * t[prefix + "attn_scale.scale"]
    f = layer_norm(x, t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"])
    return x + gelu_tanh(f @ t[prefix + "ffn_up.weight"].T) @ t[prefix + "ffn_down.weight"].T * t[prefix + "ffn_scale.scale"]


def patch_decode(x, patch):
    frames, channels = x.shape
    return x.reshape(frames, channels // patch, patch).transpose(0, 2, 1).reshape(frames * patch, channels // patch)


def reference_decode(latents, t):
    x = latents @ t["blk.0.input_proj.weight"].reshape(D_MODEL, CODE_DIM).T
    x = patch_decode(reference_layer(x, t, "blk.0.layer.0.", CONTEXT), PATCH)
    x = x @ t["blk.1.input_proj.weight"].T
    x = reference_layer(x, t, "blk.1.layer.0.", 2 * CONTEXT) @ t["blk.1.output_proj.weight"].T
    return patch_decode(x, FINAL_PATCH).reshape(-1)


def run(exporter, model, hp, latents, committed, commit, chunk):
    tables = [torch.from_numpy(table) for table in exporter.stage_tables(hp, chunk, committed)]
    with torch.no_grad():
        return model(torch.from_numpy(latents)[None], torch.tensor([commit]), *tables)[0].numpy()


def stream(exporter, tensors, hp, latents):
    model = exporter.StreamingDecoder(tensors, hp, CHUNK, cache_dtype=torch.float32).eval()
    hop = PATCH * FINAL_PATCH
    pieces, committed = [], 0
    while committed + CHUNK <= FRAMES:
        pieces.append(run(exporter, model, hp, latents[committed:committed + CHUNK], committed, 1.0, CHUNK))
        committed += CHUNK
    rest = FRAMES - committed
    padded = np.zeros((CHUNK, CODE_DIM), dtype=np.float32)
    padded[:rest] = latents[committed:]
    preview = run(exporter, model, hp, padded, committed, 0.0, CHUNK)
    again = run(exporter, model, hp, padded, committed, 0.0, CHUNK)
    return np.concatenate(pieces + [preview[:rest * hop]]), preview, again


class StreamingDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        cls.tensors = tiny_tensors(rng)
        cls.latents = rand(rng, FRAMES, CODE_DIM)
        cls.hp = tiny_hparams()

    def test_one_pass_matches_numpy_reference(self):
        model = self.exporter.StreamingDecoder(self.tensors, self.hp, FRAMES, cache_dtype=torch.float32).eval()
        got = run(self.exporter, model, self.hp, self.latents, 0, 1.0, FRAMES)
        np.testing.assert_allclose(got, reference_decode(self.latents.astype(np.float64), self.tensors), atol=TOLERANCE)

    def test_chunks_match_one_pass_and_previews_do_not_commit(self):
        model = self.exporter.StreamingDecoder(self.tensors, self.hp, FRAMES, cache_dtype=torch.float32).eval()
        whole = run(self.exporter, model, self.hp, self.latents, 0, 1.0, FRAMES)
        streamed, preview, again = stream(self.exporter, self.tensors, self.hp, self.latents)
        np.testing.assert_allclose(streamed, whole, atol=TOLERANCE)
        np.testing.assert_array_equal(preview, again)

    def test_tables_match_the_engine(self):
        np.testing.assert_array_equal(self.exporter.band_mask(0, 2, 3, 4),
                                      np.array([[MASKED, MASKED, MASKED, 0, MASKED],
                                                [MASKED, MASKED, MASKED, 0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(self.exporter.band_mask(5, 2, 3, 4),
                                      np.array([[0, 0, 0, 0, MASKED], [MASKED, 0, 0, 0, 0]], dtype=np.float32))
        cos, sin = self.exporter.rope_table(3, 2, 4, 10000.0)
        self.assertAlmostEqual(float(cos[0, 0]), math.cos(3.0), places=6)
        self.assertAlmostEqual(float(sin[0, 1]), math.sin(0.03), places=6)
        self.assertEqual(float(cos[0, 0]), float(cos[0, 2]))


class SidecarStemTest(unittest.TestCase):
    def test_matches_codec_decoder_sidecar_path(self):
        exporter = load_exporter()
        self.assertEqual(exporter.sidecar_stem("models/moss-codec-decoder-q8_0.gguf"), "moss-codec-decoder")
        self.assertEqual(exporter.sidecar_stem("moss-codec-decoder.gguf"), "moss-codec-decoder")


class CoremlContractTest(unittest.TestCase):
    def test_converted_interface(self):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            self.skipTest("coremltools is not installed")
        exporter = load_exporter()
        hp = tiny_hparams()
        model = exporter.StreamingDecoder(tiny_tensors(np.random.default_rng(SEED)), hp, CHUNK).eval()
        description = exporter.convert_coreml(model, hp, CHUNK).get_spec().description
        inputs = {i.name: list(i.type.multiArrayType.shape) for i in description.input}
        self.assertEqual(inputs["latents"], [1, CHUNK, CODE_DIM])
        self.assertEqual(inputs["mask_1"], [CHUNK * PATCH, 2 * CONTEXT - 1 + CHUNK * PATCH])
        self.assertEqual(len(inputs), 2 + 3 * 2)
        self.assertEqual([o.name for o in description.output], ["pcm"])
        self.assertEqual(sorted(s.name for s in description.state), ["k_0_0", "k_1_0", "v_0_0", "v_1_0"])


if __name__ == "__main__":
    unittest.main()
