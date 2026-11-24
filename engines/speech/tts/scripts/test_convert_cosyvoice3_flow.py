#!/usr/bin/env python3
"""Tests for convert-cosyvoice3-flow-to-gguf.py's dtype policy.

Pins the three rules a wrong conversion breaks at runtime: rand_noise must
stay f32 for every --dtype (the C++ reads it raw as float), the per-block
attention projections must come out as one fused to_qkv tensor, and q8_0 mode
must quantize only the 2D matmul weights while conv kernels, norms, biases
and the token embedding stay float.

Skips (passing) when torch or gguf is unavailable, like the fixture-gated
C++ tests.  Method names kept short (TruffleHog Lob detector).

Run directly or via ctest (test-convert-cosyvoice3-flow):
    python3 scripts/test_convert_cosyvoice3_flow.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "convert-cosyvoice3-flow-to-gguf.py")

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "gguf", "numpy"))

DEPTH = 2
DIM = 64
DIM_HEAD = 32
FF_INNER = 128
PROJ_IN = 320
VOCAB = 100
IN_SIZE = 80
SPK = 192
MEL = 80
CONV_K = 31


def synthetic_flow_checkpoint(path):
    import torch
    sd = {}

    def add(name, *shape):
        sd[name] = torch.randn(*shape) * 0.1

    add("input_embedding.weight", VOCAB, IN_SIZE)
    add("spk_embed_affine_layer.weight", MEL, SPK)
    add("spk_embed_affine_layer.bias", MEL)
    add("pre_lookahead_layer.conv1.weight", DIM, IN_SIZE, 4)
    add("pre_lookahead_layer.conv1.bias", DIM)
    add("pre_lookahead_layer.conv2.weight", IN_SIZE, DIM, 3)
    add("pre_lookahead_layer.conv2.bias", IN_SIZE)
    add("decoder.estimator.input_embed.proj.weight", DIM, PROJ_IN)
    add("decoder.estimator.input_embed.proj.bias", DIM)
    add("decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight", DIM, DIM // 16, CONV_K)
    add("decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias", DIM)
    add("decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight", DIM, DIM // 16, CONV_K)
    add("decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias", DIM)
    add("decoder.estimator.rotary_embed.inv_freq", DIM_HEAD // 2)
    add("decoder.estimator.time_embed.time_mlp.0.weight", DIM, 256)
    add("decoder.estimator.time_embed.time_mlp.0.bias", DIM)
    add("decoder.estimator.time_embed.time_mlp.2.weight", DIM, DIM)
    add("decoder.estimator.time_embed.time_mlp.2.bias", DIM)
    for i in range(DEPTH):
        blk = f"decoder.estimator.transformer_blocks.{i}"
        add(f"{blk}.attn_norm.linear.weight", 6 * DIM, DIM)
        add(f"{blk}.attn_norm.linear.bias", 6 * DIM)
        for proj in ("to_q", "to_k", "to_v"):
            add(f"{blk}.attn.{proj}.weight", DIM, DIM)
            add(f"{blk}.attn.{proj}.bias", DIM)
        add(f"{blk}.attn.to_out.0.weight", DIM, DIM)
        add(f"{blk}.attn.to_out.0.bias", DIM)
        add(f"{blk}.ff.ff.0.0.weight", FF_INNER, DIM)
        add(f"{blk}.ff.ff.0.0.bias", FF_INNER)
        add(f"{blk}.ff.ff.2.weight", DIM, FF_INNER)
        add(f"{blk}.ff.ff.2.bias", DIM)
    add("decoder.estimator.norm_out.linear.weight", 2 * DIM, DIM)
    add("decoder.estimator.norm_out.linear.bias", 2 * DIM)
    add("decoder.estimator.proj_out.weight", MEL, DIM)
    add("decoder.estimator.proj_out.bias", MEL)
    torch.save(sd, path)


def read_tensor_types(path):
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(path)
    return {t.name: GGMLQuantizationType(int(t.tensor_type)).name for t in reader.tensors}


@unittest.skipUnless(HAVE_DEPS, "torch/gguf/numpy not installed")
class ConvertFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self._tmp.name, "flow.pt")
        synthetic_flow_checkpoint(self.ckpt)

    def tearDown(self):
        self._tmp.cleanup()

    def convert(self, dtype):
        out = os.path.join(self._tmp.name, f"flow-{dtype}.gguf")
        subprocess.run([sys.executable, SCRIPT, "--flow", self.ckpt,
                        "--outfile", out, "--dtype", dtype],
                       check=True, capture_output=True)
        return read_tensor_types(out)

    def type_name(self, types, name):
        self.assertIn(name, types)
        return types[name]

    def check_rand_noise_f32(self, types):
        self.assertIn("F32", self.type_name(types, "flow/rand_noise"))

    def test_q8(self):
        types = self.convert("q8_0")
        self.check_rand_noise_f32(types)
        self.assertIn("Q8_0", self.type_name(types, "flow/blk/0/attn/to_qkv/weight"))
        self.assertIn("Q8_0", self.type_name(types, "flow/blk/0/ff/ff/0/0/weight"))
        self.assertNotIn("flow/blk/0/attn/to_q/weight", types)
        self.assertIn("F32", self.type_name(types, "flow/blk/0/attn/to_qkv/bias"))
        self.assertIn("F32", self.type_name(types, "flow/input_embedding/weight"))
        self.assertIn("F32", self.type_name(types, "flow/input_embed/conv_pos_embed/conv1/0/weight"))

    def test_bf16(self):
        types = self.convert("bf16")
        self.check_rand_noise_f32(types)
        self.assertIn("BF16", self.type_name(types, "flow/blk/0/attn/to_qkv/weight"))
        self.assertIn("BF16", self.type_name(types, "flow/blk/0/ff/ff/0/0/weight"))
        self.assertIn("F32", self.type_name(types, "flow/blk/0/attn/to_qkv/bias"))
        self.assertIn("F16", self.type_name(types, "flow/input_embed/conv_pos_embed/conv1/0/weight"))
        self.assertIn("F16", self.type_name(types, "flow/input_embedding/weight"))

    def test_f16(self):
        types = self.convert("f16")
        self.check_rand_noise_f32(types)
        self.assertIn("F16", self.type_name(types, "flow/blk/0/attn/to_qkv/weight"))
        self.assertIn("F32", self.type_name(types, "flow/blk/0/attn/to_qkv/bias"))
        self.assertIn("F32", self.type_name(types, "flow/rotary_embed/inv_freq"))

    def test_f32(self):
        types = self.convert("f32")
        self.check_rand_noise_f32(types)
        self.assertIn("F32", self.type_name(types, "flow/blk/0/attn/to_qkv/weight"))


if __name__ == "__main__":
    unittest.main()
