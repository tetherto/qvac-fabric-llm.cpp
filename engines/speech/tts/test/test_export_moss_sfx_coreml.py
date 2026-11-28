#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-moss-sfx-coreml.py: the DiT
rebuild (with its half-split rotary layout) against an independent NumPy DiT
that rotates interleaved pairs the way ggml does, the phase-form transposed
convolution against torch's ConvTranspose1d, the Snake spellings and the
half-precision clamp, the timestep embedding, the sidecar stem rule, and (with
coremltools) both converted interfaces. Skips without numpy or torch.
"""
import importlib.util
import math
import pathlib
import sys
import unittest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(0)

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "export-moss-sfx-coreml.py"

WIDTH = 16
HEADS = 2
FF = 24
CHANNELS = 4
TEXT_DIM = 12
TEXT_TOKENS = 5
FREQ_DIM = 8
FRAMES = 7
LAYERS = 2
EPS = 1e-6
RATES = [2, 3]
DECODER_DIM = 8
SAMPLE_RATE = 60
TOLERANCE = 2e-4
SEED = 26333


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_moss_sfx_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def dit_hparams():
    return {"dit_layers": LAYERS, "dit_width": WIDTH, "dit_heads": HEADS, "channels": CHANNELS,
            "text_dim": TEXT_DIM, "freq_dim": FREQ_DIM, "eps": EPS, "text_tokens": TEXT_TOKENS, "frames": FRAMES}


def vae_hparams():
    return {"latent_dim": CHANNELS, "decoder_dim": DECODER_DIM, "rates": RATES, "hop": int(np.prod(RATES)),
            "sample_rate": SAMPLE_RATE}


def add_linear(t, rng, name, inputs, outputs):
    t[name + ".weight"] = rand(rng, outputs, inputs)
    t[name + ".bias"] = rand(rng, outputs)


def dit_layer(t, rng, layer):
    prefix = f"dit.blk.{layer}."
    for name in ("self_q", "self_k", "self_v", "self_o", "cross_q", "cross_k", "cross_v", "cross_o"):
        add_linear(t, rng, prefix + name, WIDTH, WIDTH)
    for name in ("self_norm_q", "self_norm_k", "cross_norm_q", "cross_norm_k", "cross_norm"):
        t[prefix + name + ".weight"] = 1.0 + rand(rng, WIDTH)
    t[prefix + "cross_norm.bias"] = rand(rng, WIDTH)
    add_linear(t, rng, prefix + "ffn_up", WIDTH, FF)
    add_linear(t, rng, prefix + "ffn_down", FF, WIDTH)
    t[prefix + "modulation"] = rand(rng, 6, WIDTH)


def dit_tensors(rng):
    t = {}
    add_linear(t, rng, "dit.patch_embd", CHANNELS, WIDTH)
    add_linear(t, rng, "dit.time_embd_1", FREQ_DIM, WIDTH)
    add_linear(t, rng, "dit.time_embd_2", WIDTH, WIDTH)
    add_linear(t, rng, "dit.time_proj", WIDTH, 6 * WIDTH)
    add_linear(t, rng, "dit.text_embd_1", TEXT_DIM, WIDTH)
    add_linear(t, rng, "dit.text_embd_2", WIDTH, WIDTH)
    add_linear(t, rng, "dit.head", WIDTH, CHANNELS)
    t["dit.head.modulation"] = rand(rng, 2, WIDTH)
    for layer in range(LAYERS):
        dit_layer(t, rng, layer)
    return t


def lin(x, t, name):
    return x @ t[name + ".weight"].T + t[name + ".bias"]


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True) + EPS)


def rms(x, weight):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS) * weight


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


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(q, k, v, rotate):
    head = WIDTH // HEADS
    outs = []
    for h in range(HEADS):
        cols = slice(h * head, (h + 1) * head)
        qh, kh = q[:, cols], k[:, cols]
        if rotate:
            qh, kh = rope_interleaved(qh), rope_interleaved(kh)
        outs.append(softmax(qh @ kh.T / math.sqrt(head)) @ v[:, cols])
    return np.concatenate(outs, axis=-1)


def reference_block(x, ctx, t_mod, t, layer):
    p = f"dit.blk.{layer}."
    mod = t[p + "modulation"] + t_mod
    a = norm(x) * (1 + mod[1]) + mod[0]
    q, k = rms(lin(a, t, p + "self_q"), t[p + "self_norm_q.weight"]), rms(lin(a, t, p + "self_k"), t[p + "self_norm_k.weight"])
    x = x + lin(attention(q, k, lin(a, t, p + "self_v"), True), t, p + "self_o") * mod[2]
    c = norm(x) * t[p + "cross_norm.weight"] + t[p + "cross_norm.bias"]
    q = rms(lin(c, t, p + "cross_q"), t[p + "cross_norm_q.weight"])
    k = rms(lin(ctx, t, p + "cross_k"), t[p + "cross_norm_k.weight"])
    x = x + lin(attention(q, k, lin(ctx, t, p + "cross_v"), False), t, p + "cross_o")
    f = norm(x) * (1 + mod[4]) + mod[3]
    return x + lin(gelu_tanh(lin(f, t, p + "ffn_up")), t, p + "ffn_down") * mod[5]


def reference_dit(latents, timestep, context, t):
    x = lin(latents.T, t, "dit.patch_embd")
    temb = lin(silu(lin(timestep, t, "dit.time_embd_1")), t, "dit.time_embd_2")
    t_mod = lin(silu(temb), t, "dit.time_proj").reshape(6, WIDTH)
    ctx = lin(gelu_tanh(lin(context, t, "dit.text_embd_1")), t, "dit.text_embd_2")
    for layer in range(LAYERS):
        x = reference_block(x, ctx, t_mod, t, layer)
    mod = t["dit.head.modulation"] + temb
    return lin(norm(x) * (1 + mod[1]) + mod[0], t, "dit.head").T


class DitTest(unittest.TestCase):
    def test_matches_interleaved_rope_reference(self):
        exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        tensors = dit_tensors(rng)
        latents, context = rand(rng, CHANNELS, FRAMES), rand(rng, TEXT_TOKENS, TEXT_DIM)
        timestep = exporter.timestep_sinusoid(321.0, FREQ_DIM)
        model = exporter.Dit(tensors, dit_hparams()).eval()
        with torch.no_grad():
            got = model(torch.from_numpy(latents)[None], torch.from_numpy(timestep)[None],
                        torch.from_numpy(context)[None])[0].numpy()
        want = reference_dit(latents.astype(np.float64), timestep.astype(np.float64), context.astype(np.float64),
                             tensors)
        np.testing.assert_allclose(got, want, atol=TOLERANCE)

    def test_timestep_embedding_puts_cosines_first(self):
        exporter = load_exporter()
        emb = exporter.timestep_sinusoid(5.0, FREQ_DIM)
        self.assertAlmostEqual(float(emb[0]), math.cos(5.0), places=6)
        self.assertAlmostEqual(float(emb[FREQ_DIM // 2]), math.sin(5.0), places=6)


class VaeTest(unittest.TestCase):
    def test_phase_upsample_matches_conv_transpose(self):
        exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        for stride in (2, 3, 4, 5, 8):
            in_ch, out_ch, length = 3, 2, 6
            weight = rand(rng, in_ch, out_ch, 2 * stride)
            bias = rand(rng, out_ch)
            columns = weight.reshape(in_ch, out_ch * 2 * stride).T
            module = exporter.Upsample({"u.weight": columns, "u.bias": bias.reshape(-1, 1)}, "u", stride, length)
            x = torch.from_numpy(rand(rng, 1, in_ch, length))
            want = F.conv_transpose1d(x, torch.from_numpy(weight), torch.from_numpy(bias), stride=stride,
                                      padding=(stride + 1) // 2, output_padding=stride % 2)
            with torch.no_grad():
                np.testing.assert_allclose(module(x).numpy(), want.numpy(), atol=1e-5, err_msg=f"stride {stride}")

    def test_snake_spellings_and_clamp(self):
        exporter = load_exporter()
        alpha = np.array([0.5, 2.0, 1e-6], dtype=np.float32)
        inv = 1.0 / (alpha + 1e-9)
        x = torch.linspace(-3.0, 3.0, 9).reshape(1, 1, 9).repeat(1, 3, 1)
        snake = exporter.Snake(alpha, inv)
        self.assertLessEqual(float(snake.inv.max()), exporter.HALF_SAFE_INVERSE)
        exporter.Snake.mode = "sin"
        exact = snake(x)
        exporter.Snake.mode = "poly"
        np.testing.assert_allclose(snake(x).detach().numpy(), exact.detach().numpy(), atol=1e-4)
        self.assertTrue(torch.allclose(exact[0, 2], x[0, 2], atol=1e-4))

    def test_block_lengths_follow_the_rates(self):
        exporter = load_exporter()
        self.assertEqual(exporter.block_lengths([8, 5, 4], 10), [10, 80, 400])


class SidecarStemTest(unittest.TestCase):
    def test_matches_sfx_sidecar_paths(self):
        exporter = load_exporter()
        self.assertEqual(exporter.sidecar_stem("models/moss-sfx-v2-q8_0.gguf", "dit"), "moss-sfx-v2-dit")
        self.assertEqual(exporter.sidecar_stem("moss-sfx-v2-f16.gguf", "vae"), "moss-sfx-v2-vae")
        self.assertEqual(exporter.sidecar_stem("moss-sfx-v2.gguf", "vae"), "moss-sfx-v2-vae")


class CoremlContractTest(unittest.TestCase):
    def test_converted_interfaces(self):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            self.skipTest("coremltools is not installed")
        exporter = load_exporter()
        rng = np.random.default_rng(SEED)
        dit = exporter.Dit(dit_tensors(rng), dit_hparams()).eval()
        spec = exporter.convert_coreml(dit, "dit", dit_hparams(), 0, "float16").get_spec()
        inputs = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
        self.assertEqual(inputs, {"latents": [1, CHANNELS, FRAMES], "timestep": [1, FREQ_DIM],
                                  "context": [1, TEXT_TOKENS, TEXT_DIM]})
        self.assertEqual([o.name for o in spec.description.output], ["velocity"])


if __name__ == "__main__":
    unittest.main()
