#!/usr/bin/env python3
"""Tests for convert-cosyvoice3-llm-to-gguf.py's fused-qkv layout and dtype
policy.

Pins the rules a wrong conversion breaks at runtime: each layer's q/k/v
projections must come out as one fused qkv_proj tensor whose rows are exactly
q ++ k ++ v (the engine slices its matvec output by feature offset), the
separate q/k/v_proj tensors must be gone, and q8_0 mode must quantize only
the transformer-body matmul weights while biases, norms and the embeddings
stay f32.

Skips (passing) when torch or gguf is unavailable, like the fixture-gated
C++ tests.

Run directly or via ctest (test-convert-cosyvoice3-llm):
    python3 scripts/test_convert_cosyvoice3_llm.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "convert-cosyvoice3-llm-to-gguf.py")

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "gguf", "numpy"))

# The converter hard-codes n_head=14 and quantizes only when the in-dim is a
# multiple of 32, so the synthetic model uses hidden = 224 (14 * 16, and
# 7 * 32).
DEPTH = 2
HIDDEN = 224
HEAD_DIM = 16
N_KV = 2
KV_DIM = N_KV * HEAD_DIM
INTER = 64
VOCAB = 96
SPEECH_VOCAB = 90


def synthetic_llm_checkpoint(path):
    import torch
    sd = {}

    def add(name, *shape):
        sd[name] = torch.randn(*shape) * 0.1

    add("llm.model.model.embed_tokens.weight", VOCAB, HIDDEN)
    add("llm.model.model.norm.weight", HIDDEN)
    add("speech_embedding.weight", SPEECH_VOCAB, HIDDEN)
    add("llm_decoder.weight", SPEECH_VOCAB, HIDDEN)
    for i in range(DEPTH):
        blk = f"llm.model.model.layers.{i}"
        add(f"{blk}.input_layernorm.weight", HIDDEN)
        add(f"{blk}.post_attention_layernorm.weight", HIDDEN)
        add(f"{blk}.self_attn.q_proj.weight", HIDDEN, HIDDEN)
        add(f"{blk}.self_attn.q_proj.bias", HIDDEN)
        add(f"{blk}.self_attn.k_proj.weight", KV_DIM, HIDDEN)
        add(f"{blk}.self_attn.k_proj.bias", KV_DIM)
        add(f"{blk}.self_attn.v_proj.weight", KV_DIM, HIDDEN)
        add(f"{blk}.self_attn.v_proj.bias", KV_DIM)
        add(f"{blk}.self_attn.o_proj.weight", HIDDEN, HIDDEN)
        add(f"{blk}.mlp.gate_proj.weight", INTER, HIDDEN)
        add(f"{blk}.mlp.up_proj.weight", INTER, HIDDEN)
        add(f"{blk}.mlp.down_proj.weight", HIDDEN, INTER)
    torch.save(sd, path)
    return sd


def read_tensors(path):
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(path)
    return {t.name: (GGMLQuantizationType(int(t.tensor_type)).name, t.data)
            for t in reader.tensors}


@unittest.skipUnless(HAVE_DEPS, "torch/gguf/numpy not installed")
class ConvertLlmTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self._tmp.name, "llm.pt")
        self.sd = synthetic_llm_checkpoint(self.ckpt)

    def tearDown(self):
        self._tmp.cleanup()

    def convert(self, dtype):
        out = os.path.join(self._tmp.name, f"llm-{dtype}.gguf")
        subprocess.run([sys.executable, SCRIPT, "--llm", self.ckpt,
                        "--outfile", out, "--dtype", dtype],
                       check=True, capture_output=True)
        return read_tensors(out)

    def type_name(self, tensors, name):
        self.assertIn(name, tensors)
        return tensors[name][0]

    def check_fused_layout(self, tensors):
        for i in range(DEPTH):
            self.assertIn(f"lm/blk/{i}/qkv_proj/weight", tensors)
            self.assertIn(f"lm/blk/{i}/qkv_proj/bias", tensors)
            for proj in ("q_proj", "k_proj", "v_proj"):
                self.assertNotIn(f"lm/blk/{i}/{proj}/weight", tensors)
                self.assertNotIn(f"lm/blk/{i}/{proj}/bias", tensors)

    def test_f32_fused_rows(self):
        import numpy as np
        import torch
        tensors = self.convert("f32")
        self.check_fused_layout(tensors)
        blk = "llm.model.model.layers.0.self_attn"
        expect = torch.cat([self.sd[f"{blk}.{p}_proj.weight"] for p in "qkv"], dim=0).numpy()
        got = tensors["lm/blk/0/qkv_proj/weight"][1]
        self.assertEqual(got.shape, expect.shape)
        self.assertTrue(np.array_equal(got, expect))
        expect_b = torch.cat([self.sd[f"{blk}.{p}_proj.bias"] for p in "qkv"], dim=0).numpy()
        self.assertTrue(np.array_equal(tensors["lm/blk/0/qkv_proj/bias"][1], expect_b))

    def test_q8(self):
        tensors = self.convert("q8_0")
        self.check_fused_layout(tensors)
        self.assertIn("Q8_0", self.type_name(tensors, "lm/blk/0/qkv_proj/weight"))
        self.assertIn("Q8_0", self.type_name(tensors, "lm/blk/0/gate/weight"))
        self.assertIn("Q8_0", self.type_name(tensors, "lm/blk/1/down/weight"))
        self.assertIn("F32", self.type_name(tensors, "lm/blk/0/qkv_proj/bias"))
        self.assertIn("F32", self.type_name(tensors, "lm/blk/0/in_ln/weight"))
        self.assertIn("F32", self.type_name(tensors, "lm/embed_tokens/weight"))
        self.assertIn("F32", self.type_name(tensors, "lm/speech_embedding/weight"))
        self.assertIn("F32", self.type_name(tensors, "lm/llm_decoder/weight"))


if __name__ == "__main__":
    unittest.main()
