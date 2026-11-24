"""Pocket TTS Mimi streaming decoder as a Core ML model, rebuilt from mimi.gguf.

One prediction decodes FRAMES latent frames exactly as one Mimi::decode call
of src/pocket/mimi.cpp does: quantizer.output_proj, the depthwise upsample
(stride 16), the 2-layer transformer over a 250-position sliding window, and
the SEANet decoder (conv, three transposed-conv + residual stages, conv).

The streaming state is explicit so the model can run on the Neural Engine
(Core ML state cannot): every causal conv takes its history (the last inputs
of the previous call) and returns the new one, and the transformer takes the
last KEEP keys/values of each layer (after RoPE, oldest first, unfilled rows
masked) and returns the window after this call. The host feeds the outputs
back, and supplies RoPE tables for the call's positions and the attention
mask:

  latents [FRAMES, 32], cos/sin [n, 32], mask [n, KEEP + n],
  keys/values [2, 8, KEEP, 64], one input per HISTORIES entry
  -> pcm [FRAMES * 1920], new_keys/new_values, new_<history> each,
  with n = FRAMES * 16.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ARCH = "pocket-tts-mimi"
DIM, LATENT, HEADS, HEAD_DIM, LAYERS, FF = 512, 32, 8, 64, 2, 2048
WINDOW = 250
KEEP = WINDOW - 1
UP = 16
RATIOS = (6, 5, 4)
FRAME_SAMPLES = 1920
FRAMES = 1
ROPE_BASE = 10000.0
MASKED = -1e4
GELU_C = 0.7978845608028654


def histories():
    """(name, channels, length) of each causal-conv history, in input order."""
    specs = [("h_upsample", DIM, 1), ("h_conv0", DIM, 6)]
    width = DIM
    for i in range(len(RATIOS)):
        specs.append((f"h_convtr{i}", width, 1))
        width //= 2
        specs.append((f"h_residual{i}", width, 2))
    specs.append(("h_final", width, 2))
    return specs


HISTORIES = histories()
INPUTS = ("latents", "cos", "sin", "mask", "keys", "values") + tuple(name for name, _, _ in HISTORIES)
OUTPUTS = ("pcm", "new_keys", "new_values") + tuple("new_" + name for name, _, _ in HISTORIES)


def read_gguf(path):
    """(decoder tensors, metadata) of a Mimi GGUF."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()
    if arch != ARCH:
        raise ValueError(f"{path} is a {arch} GGUF, not a Pocket Mimi GGUF")
    tensors = {t.name: np.array(t.data, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))
               for t in reader.tensors if not t.name.startswith(("encoder.", "enc.", "downsample."))}
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith("pocket.") and name != "pocket.mimi_config"}
    return tensors, metadata


def rope_tables(position, n):
    """cos, sin [n, HEAD_DIM / 2] of ggml_rope_ext mode 0 at positions position.."""
    theta = np.arange(position, position + n, dtype=np.float64)[:, None] * \
        ROPE_BASE ** (-2.0 * np.arange(HEAD_DIM // 2) / HEAD_DIM)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def attention_mask(position, n):
    """[n, KEEP + n]: query i sees the WINDOW positions up to its own, none before 0."""
    mask = np.full((n, KEEP + n), MASKED, np.float32)
    first = KEEP - min(position, KEEP)
    for i in range(n):
        mask[i, max(i, first):KEEP + i + 1] = 0
    return mask


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def constant(array):
    return nn.Parameter(torch.from_numpy(np.array(array, dtype=np.float32)), requires_grad=False)


class Decoder(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        t = tensors
        self.proj = constant(t["quantizer.output_proj.weight"])  # [DIM, LATENT]
        # Upsample: out[c, 16 t + j] = x[c, t] w[j, c] + x[c, t - 1] w[16 + j, c], as a grouped
        # two-tap conv over [history, x] to channels c * 16 + j, then interleaved.
        w = t["upsample.convtr.convtr.weight"]  # [2 * UP, DIM]
        upsample = np.zeros((DIM * UP, 1, 2), np.float32)
        for j in range(UP):
            upsample[j::UP, 0, 0], upsample[j::UP, 0, 1] = w[UP + j], w[j]
        self.upsample = constant(upsample)
        self.p = nn.ParameterDict()
        for i in range(LAYERS):
            for name in ("self_attn.in_proj.weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight",
                         "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "layer_scale_1.scale",
                         "layer_scale_2.scale"):
                self.p[f"l{i}_" + name.replace(".", "_")] = constant(t[f"dec.layers.{i}.{name}"])
        self.out_channels = []

        def conv(name, key):
            self.p[name + "_w"] = constant(t[key + ".weight"].transpose(1, 2, 0))  # [K, out, in] -> [out, in, K]
            self.p[name + "_b"] = constant(t[key + ".bias"])

        def convtr(name, key, ratio):
            # out[o, r t + k] = W[k] x[t] + W[r + k] x[t - 1]: a two-tap conv to r * out channels.
            k = t[key + ".weight"]  # [2r, out, in]
            out = k.shape[1]
            weight = np.zeros((ratio * out, k.shape[2], 2), np.float32)
            for phase in range(ratio):
                weight[phase * out:(phase + 1) * out, :, 0] = k[ratio + phase]
                weight[phase * out:(phase + 1) * out, :, 1] = k[phase]
            self.p[name + "_w"] = constant(weight)
            self.p[name + "_b"] = constant(t[key + ".bias"])
            self.out_channels.append(out)

        conv("conv0", "decoder.model.0.conv")
        for i, ratio in enumerate(RATIOS):
            convtr(f"convtr{i}", f"decoder.model.{2 + 3 * i}.convtr", ratio)
            conv(f"residual{i}a", f"decoder.model.{3 + 3 * i}.block.1.conv")
            conv(f"residual{i}b", f"decoder.model.{3 + 3 * i}.block.3.conv")
        conv("final", "decoder.model.11.conv")

    def rope(self, x, cos, sin):
        """x [HEADS, n, HEAD_DIM]; rotates adjacent pairs. Shapes are constants: traced sizes do not convert."""
        pairs = x.reshape(HEADS, -1, HEAD_DIM // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).reshape(HEADS, -1, HEAD_DIM)

    def transformer(self, x, cos, sin, mask, keys, values):
        """x [n, DIM] -> x, and each layer's KEEP newest keys/values."""
        p, new_keys, new_values = self.p, [], []
        for i in range(LAYERS):
            h = F.layer_norm(x, (DIM,), p[f"l{i}_norm1_weight"], p[f"l{i}_norm1_bias"], 1e-5)
            qkv = h @ p[f"l{i}_self_attn_in_proj_weight"].t()
            q, k, v = (qkv[:, j * DIM:(j + 1) * DIM].reshape(-1, HEADS, HEAD_DIM).transpose(0, 1) for j in range(3))
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            k, v = torch.cat((keys[i], k), 1), torch.cat((values[i], v), 1)
            new_keys.append(k[:, -KEEP:])
            new_values.append(v[:, -KEEP:])
            scores = (q @ k.transpose(-1, -2)) * (1.0 / HEAD_DIM ** 0.5) + mask
            attended = (scores.softmax(-1) @ v).transpose(0, 1).reshape(-1, DIM)
            x = x + (attended @ p[f"l{i}_self_attn_out_proj_weight"].t()) * p[f"l{i}_layer_scale_1_scale"]
            h = F.layer_norm(x, (DIM,), p[f"l{i}_norm2_weight"], p[f"l{i}_norm2_bias"], 1e-5)
            h = gelu(h @ p[f"l{i}_linear1_weight"].t()) @ p[f"l{i}_linear2_weight"].t()
            x = x + h * p[f"l{i}_layer_scale_2_scale"]
        return x, torch.stack(new_keys), torch.stack(new_values)

    @staticmethod
    def streamed(x, history, length, op):
        """op over [history, x] (x [1, C, T]); also returns the last `length` inputs."""
        combined = torch.cat((history[None], x), 2)
        return op(combined), combined[0, :, -length:]

    def forward(self, latents, cos, sin, mask, keys, values, *history):
        p, lengths, new_history = self.p, [length for _, _, length in HISTORIES], []

        def upsample(z):
            y = F.conv1d(z, self.upsample, groups=DIM)  # [1, DIM * UP, T]
            return y.reshape(DIM, UP, -1).permute(0, 2, 1).reshape(1, DIM, -1)

        def conv(name):
            return lambda z: F.conv1d(z, p[name + "_w"], p[name + "_b"])

        def convtr(i, ratio):
            def op(z):
                y = F.conv1d(z, p[f"convtr{i}_w"])  # [1, ratio * out, T]
                out = self.out_channels[i]
                return y.reshape(ratio, out, -1).permute(1, 2, 0).reshape(1, out, -1) + p[f"convtr{i}_b"][None, :, None]
            return op

        x = (latents @ self.proj.t()).t()[None]
        x, h = self.streamed(x, history[0], lengths[0], upsample)
        new_history.append(h)
        x, new_keys, new_values = self.transformer(x[0].t(), cos, sin, mask, keys, values)
        x, h = self.streamed(x.t()[None], history[1], lengths[1], conv("conv0"))
        new_history.append(h)
        for i, ratio in enumerate(RATIOS):
            x, h = self.streamed(F.elu(x), history[2 + 2 * i], lengths[2 + 2 * i], convtr(i, ratio))
            new_history.append(h)
            y, h = self.streamed(F.elu(x), history[3 + 2 * i], lengths[3 + 2 * i], conv(f"residual{i}a"))
            new_history.append(h)
            x = x + conv(f"residual{i}b")(F.elu(y))
        x, h = self.streamed(F.elu(x), history[-1], lengths[-1], conv("final"))
        new_history.append(h)
        return (x.reshape(-1), new_keys, new_values, *new_history)


class Stream:
    """Host-side streaming state, as the runtime keeps it."""

    def __init__(self):
        self.position = 0
        self.keys = np.zeros((LAYERS, HEADS, KEEP, HEAD_DIM), np.float32)
        self.values = np.zeros_like(self.keys)
        self.history = [np.zeros((channels, length), np.float32) for _, channels, length in HISTORIES]

    def inputs(self, latents):
        n = latents.shape[0] * UP
        cos, sin = rope_tables(self.position, n)
        return dict(zip(INPUTS, [np.asarray(latents, np.float32), cos, sin, attention_mask(self.position, n),
                                 self.keys, self.values, *self.history]))

    def update(self, outputs, frames):
        self.keys, self.values = outputs["new_keys"], outputs["new_values"]
        self.history = [outputs["new_" + name] for name, _, _ in HISTORIES]
        self.position += frames * UP
        return outputs["pcm"].reshape(-1)


def run(model, latents, calls):
    """Eager decode of latents [T, 32] in calls of the given frame counts."""
    stream, pcm, at = Stream(), [], 0
    with torch.no_grad():
        for count in calls:
            inputs = stream.inputs(latents[at:at + count])
            outputs = model(*(torch.from_numpy(inputs[name]) for name in INPUTS))
            pcm.append(stream.update({name: o.numpy() for name, o in zip(OUTPUTS, outputs)}, count))
            at += count
    return np.concatenate(pcm)


def random_checkpoint(seed, source="random"):
    """(decoder tensors, metadata) of a random Mimi with the checkpoint's names and shapes."""
    rng = np.random.default_rng(seed)

    def t(*shape, scale=None, offset=0.0):
        fan_in = shape[-1] if scale is None else None
        sigma = scale if scale is not None else 1.0 / np.sqrt(fan_in)
        return (offset + rng.normal(0, sigma, shape)).astype(np.float32)

    tensors = {"quantizer.output_proj.weight": t(DIM, LATENT), "upsample.convtr.convtr.weight": t(2 * UP, DIM, scale=0.3)}
    for i in range(LAYERS):
        p = f"dec.layers.{i}"
        tensors[p + ".self_attn.in_proj.weight"] = t(3 * DIM, DIM)
        tensors[p + ".self_attn.out_proj.weight"] = t(DIM, DIM)
        tensors[p + ".linear1.weight"], tensors[p + ".linear2.weight"] = t(FF, DIM), t(DIM, FF)
        for n in ("1", "2"):
            tensors[f"{p}.norm{n}.weight"], tensors[f"{p}.norm{n}.bias"] = t(DIM, scale=0.1, offset=1.0), t(DIM, scale=0.1)
            tensors[f"{p}.layer_scale_{n}.scale"] = t(DIM, scale=0.1, offset=1.0)

    def conv(key, out, inp, kernel):
        tensors[key + ".weight"] = (rng.normal(0, 1.0 / np.sqrt(inp * kernel), (kernel, out, inp))).astype(np.float32)
        tensors[key + ".bias"] = t(out, scale=0.05)

    conv("decoder.model.0.conv", DIM, DIM, 7)
    width = DIM
    for i, ratio in enumerate(RATIOS):
        conv(f"decoder.model.{2 + 3 * i}.convtr", width // 2, width, 2 * ratio)
        width //= 2
        conv(f"decoder.model.{3 + 3 * i}.block.1.conv", width // 2, width, 3)
        conv(f"decoder.model.{3 + 3 * i}.block.3.conv", width, width // 2, 1)
    conv("decoder.model.11.conv", 1, width, 3)
    return tensors, {"pocket.source_sha256": source}
