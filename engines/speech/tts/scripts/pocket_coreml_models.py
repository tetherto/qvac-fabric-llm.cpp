"""Pocket TTS FlowLM Core ML step rebuilt from the native GGUF tensors.

One prediction is one generated frame of src/pocket/engine.cpp's AR loop:
FlowLM::advance (input_linear, the causal transformer, out_norm) followed by
one LSD step of FlowLM::sample (s = 0, t = 1), mirroring the ggml graphs in
src/pocket/flow_lm.cpp:

  latent [1, L], noise [1, L], cos/sin [1, head_dim / 2] (RoPE at the
  frame's position), mask [1, 1, 1, C] (0 at positions <= the frame's,
  MASKED after) and write_mask [1, 1, C, 1] (1 at the frame's position)
  -> hidden [1, D] (out_norm output) and next_latent [1, L] (noise + the
  one-step flow velocity).

The KV cache lives in Core ML state ("keys" and "values", float16
[layers, heads, C, head_dim], keys after RoPE); each prediction blends the
frame's row into it. The host writes the prefix rows (voice and text,
computed by ggml) into the state before the first frame of a chunk. The
time embeddings of s = 0 and t = 1 are folded into a constant at export.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ARCH = "pocket-tts-flow-lm"
MASKED = -1e4
TIME_FREQS = 128
NORM_EPS = 1e-5
FLOW_NORM_EPS = 1e-6
GELU_C = math.sqrt(2.0 / math.pi)
INPUTS = ("latent", "noise", "cos", "sin", "mask", "write_mask")
OUTPUTS = ("hidden", "next_latent")
STATES = ("keys", "values")


def read_gguf(path):
    """(tensors, metadata) of a FlowLM GGUF, matrices widened to float32."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()
    if arch != ARCH:
        raise ValueError(f"{path} is a {arch} GGUF, not a Pocket FlowLM GGUF")
    tensors = {}
    for t in reader.tensors:
        if t.tensor_type.name not in ("F32", "F16"):
            raise ValueError(f"{t.name} is {t.tensor_type.name}; FlowLM GGUFs store F32 or F16 tensors")
        shape = tuple(int(d) for d in reversed(t.shape))
        tensors[t.name] = np.asarray(t.data, dtype=np.float32).reshape(shape)
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith("pocket.") and name != "pocket.config_json"}
    return tensors, metadata


def dimensions(metadata):
    dims = {k: int(metadata["pocket." + k]) for k in
            ("dim", "heads", "layers", "ff_dim", "latent_dim", "flow_dim", "flow_depth")}
    dims["head_dim"] = dims["dim"] // dims["heads"]
    dims["rope_base"] = float(metadata["pocket.rope_base"])
    return dims


def rope_tables(position, head_dim, base):
    """cos, sin [head_dim / 2] of ggml_rope_ext mode 0 at `position`."""
    theta = position * base ** (-2.0 * np.arange(head_dim // 2, dtype=np.float64) / head_dim)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(GELU_C * (x + 0.044715 * x * x * x)))


class Weights(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.params = nn.ParameterDict({name.replace(".", "__"): nn.Parameter(torch.from_numpy(
            np.array(value, dtype=np.float32)), requires_grad=False) for name, value in tensors.items()})

    def __getitem__(self, name):
        return self.params[name.replace(".", "__")]


class FlowLM(nn.Module):
    """The FlowLM pieces shared by the step and the test references."""

    def __init__(self, tensors, metadata):
        super().__init__()
        d = dimensions(metadata)
        self.dim, self.heads, self.layers = d["dim"], d["heads"], d["layers"]
        self.head_dim, self.latent_dim = d["head_dim"], d["latent_dim"]
        self.flow_dim, self.flow_depth = d["flow_dim"], d["flow_depth"]
        self.rope_base = d["rope_base"]
        self.w = Weights(tensors)

    def linear(self, x, name, bias=True):
        y = x @ self.w[name + ".weight"].t()
        return y + self.w[name + ".bias"] if bias else y

    def norm(self, x, name, width, eps):
        return F.layer_norm(x, (width,), self.w[name + ".weight"], self.w[name + ".bias"], eps)

    def time_embedding(self, index, time):
        """Kyutai's "RMSNorm" time embedding: divides by the unbiased variance."""
        p = f"flow_net.time_embed.{index}"
        freqs = self.w[p + ".freqs"]
        t = torch.cat((torch.cos(time * freqs), torch.sin(time * freqs)))[None]
        h = self.linear(F.silu(self.linear(t, p + ".mlp.0")), p + ".mlp.2")
        c = h - h.mean(-1, keepdim=True)
        var = (c * c).mean(-1, keepdim=True) * (self.flow_dim / (self.flow_dim - 1)) + NORM_EPS
        return h / torch.sqrt(var) * self.w[p + ".mlp.3.alpha"]

    def time_condition(self, s, t):
        return 0.5 * (self.time_embedding(0, s) + self.time_embedding(1, t))

    def velocity(self, hidden, latent, time_condition):
        f = self.flow_dim
        y = F.silu(self.linear(hidden, "flow_net.cond_embed") + time_condition)
        x = self.linear(latent, "flow_net.input_proj")
        for i in range(self.flow_depth):
            p = f"flow_net.res_blocks.{i}"
            m = self.linear(y, p + ".adaLN_modulation.1")
            shift, scale, gate = m[..., :f], m[..., f:2 * f], m[..., 2 * f:]
            h = self.norm(x, p + ".in_ln", f, FLOW_NORM_EPS) * (1 + scale) + shift
            x = x + gate * self.linear(F.silu(self.linear(h, p + ".mlp.0")), p + ".mlp.2")
        m = self.linear(y, "flow_net.final_layer.adaLN_modulation.1")
        shift, scale = m[..., :f], m[..., f:]
        return self.linear(F.layer_norm(x, (f,), eps=FLOW_NORM_EPS) * (1 + scale) + shift,
                           "flow_net.final_layer.linear")

    def rope(self, x, cos, sin):
        """x [heads, n, head_dim]; cos/sin [n, head_dim / 2]. Rotates adjacent pairs.

        Shapes are spelled as constants: traced tensor sizes do not convert."""
        pairs = x.reshape(self.heads, -1, self.head_dim // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
        return rotated.reshape(self.heads, -1, self.head_dim)

    def qkv(self, x, layer, cos, sin):
        """x [n, D] -> q, k (after RoPE), v as [heads, n, head_dim]."""
        p = f"transformer.layers.{layer}"
        qkv = self.linear(self.norm(x, p + ".norm1", self.dim, NORM_EPS), p + ".self_attn.in_proj", False)
        q, k, v = (qkv[:, j * self.dim:(j + 1) * self.dim].reshape(-1, self.heads, self.head_dim).transpose(0, 1)
                   for j in range(3))
        return self.rope(q, cos, sin), self.rope(k, cos, sin), v

    def finish(self, x, layer, q, keys, values, mask):
        """Attention over keys/values [heads, C, head_dim] with an additive mask, then the MLP."""
        p = f"transformer.layers.{layer}"
        scores = (q @ keys.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim)) + mask
        attended = (scores.softmax(-1) @ values).transpose(0, 1).reshape(-1, self.dim)
        x = x + self.linear(attended, p + ".self_attn.out_proj", False)
        h = self.linear(self.norm(x, p + ".norm2", self.dim, NORM_EPS), p + ".linear1", False)
        return x + self.linear(gelu(h), p + ".linear2", False)

    def causal(self, embeddings, start=0, keys=None, values=None):
        """Reference forward of rows at positions start.. over an optional prefix cache.

        embeddings [n, D]; keys/values: per-layer [heads, start, head_dim] or None.
        Returns (out_norm hidden [n, D], keys, values) with the cache extended."""
        n = embeddings.shape[0]
        tables = [rope_tables(start + i, self.head_dim, self.rope_base) for i in range(n)]
        cos = torch.from_numpy(np.stack([c for c, _ in tables]))
        sin = torch.from_numpy(np.stack([s for _, s in tables]))
        total = start + n
        mask = torch.full((n, total), MASKED)
        for i in range(n):
            mask[i, :start + i + 1] = 0
        x = embeddings
        out_keys, out_values = [], []
        for i in range(self.layers):
            q, k, v = self.qkv(x, i, cos, sin)
            if keys is not None:
                k, v = torch.cat((keys[i], k), 1), torch.cat((values[i], v), 1)
            out_keys.append(k)
            out_values.append(v)
            x = self.finish(x, i, q, k, v, mask)
        return self.norm(x, "out_norm", self.dim, NORM_EPS), out_keys, out_values


class FlowStep(FlowLM):
    """One generated frame with the KV cache in Core ML state (see the module docstring)."""

    def __init__(self, tensors, metadata, context, state_dtype=torch.float16):
        super().__init__(tensors, metadata)
        self.context = context
        shape = (self.layers, self.heads, context, self.head_dim)
        # Core ML state is float16; tests run the eager module with float32 state.
        self.register_buffer("keys", torch.zeros(shape, dtype=state_dtype))
        self.register_buffer("values", torch.zeros(shape, dtype=state_dtype))
        with torch.no_grad():
            self.register_buffer("time_bias", self.time_condition(0.0, 1.0).detach().clone())

    def forward(self, latent, noise, cos, sin, mask, write_mask):
        x = self.linear(latent, "input_linear", False)
        for i in range(self.layers):
            q, k, v = self.qkv(x, i, cos, sin)
            keys = self.keys[i:i + 1] * (1 - write_mask) + k * write_mask
            values = self.values[i:i + 1] * (1 - write_mask) + v * write_mask
            self.keys[i:i + 1] = keys
            self.values[i:i + 1] = values
            x = self.finish(x, i, q, keys[0], values[0], mask[0, 0])
        hidden = self.norm(x, "out_norm", self.dim, NORM_EPS)
        return hidden, noise + self.velocity(hidden, noise, self.time_bias)


def example_inputs(metadata, context, position=0):
    d = dimensions(metadata)
    cos, sin = rope_tables(position, d["head_dim"], d["rope_base"])
    mask = torch.full((1, 1, 1, context), MASKED)
    mask[..., :position + 1] = 0
    write_mask = torch.zeros(1, 1, context, 1)
    write_mask[:, :, position] = 1
    return {"latent": torch.zeros(1, d["latent_dim"]), "noise": torch.zeros(1, d["latent_dim"]),
            "cos": torch.from_numpy(cos)[None], "sin": torch.from_numpy(sin)[None],
            "mask": mask, "write_mask": write_mask}


def random_checkpoint(seed, dim=16, heads=2, layers=2, ff_dim=32, latent_dim=4, flow_dim=8, flow_depth=1,
                      vocab_size=8, source="random"):
    """(tensors, metadata) of a random FlowLM with the checkpoint's tensor names, for tests and fixtures."""
    rng = np.random.default_rng(seed)

    def t(*shape, scale=0.2, offset=0.0):
        return (offset + rng.normal(0, scale, shape)).astype(np.float32)

    d, f, l = dim, flow_dim, latent_dim
    tensors = {"conditioner.embed.weight": t(vocab_size + 1, d, scale=1.0), "bos_emb": t(l, scale=1.0),
               "emb_mean": t(l), "emb_std": t(l, scale=0.1, offset=1.0), "input_linear.weight": t(d, l),
               "out_norm.weight": t(d, offset=1.0), "out_norm.bias": t(d), "out_eos.weight": t(1, d),
               "out_eos.bias": t(1), "flow_net.input_proj.weight": t(f, l), "flow_net.input_proj.bias": t(f),
               "flow_net.cond_embed.weight": t(f, d), "flow_net.cond_embed.bias": t(f),
               "flow_net.final_layer.linear.weight": t(l, f), "flow_net.final_layer.linear.bias": t(l),
               "flow_net.final_layer.adaLN_modulation.1.weight": t(2 * f, f),
               "flow_net.final_layer.adaLN_modulation.1.bias": t(2 * f)}
    for i in range(layers):
        p = f"transformer.layers.{i}"
        for norm in ("norm1", "norm2"):
            tensors[f"{p}.{norm}.weight"], tensors[f"{p}.{norm}.bias"] = t(d, offset=1.0), t(d)
        tensors[p + ".self_attn.in_proj.weight"] = t(3 * d, d)
        tensors[p + ".self_attn.out_proj.weight"] = t(d, d)
        tensors[p + ".linear1.weight"], tensors[p + ".linear2.weight"] = t(ff_dim, d), t(d, ff_dim)
    for i in range(2):
        p = f"flow_net.time_embed.{i}"
        tensors[p + ".freqs"] = np.exp(-np.log(10000.0) * np.arange(TIME_FREQS) / TIME_FREQS).astype(np.float32)
        tensors[p + ".mlp.0.weight"], tensors[p + ".mlp.0.bias"] = t(f, 2 * TIME_FREQS), t(f)
        tensors[p + ".mlp.2.weight"], tensors[p + ".mlp.2.bias"] = t(f, f), t(f)
        tensors[p + ".mlp.3.alpha"] = t(f, offset=1.0)
    for i in range(flow_depth):
        p = f"flow_net.res_blocks.{i}"
        tensors[p + ".in_ln.weight"], tensors[p + ".in_ln.bias"] = t(f, offset=1.0), t(f)
        for mlp in ("mlp.0", "mlp.2"):
            tensors[f"{p}.{mlp}.weight"], tensors[f"{p}.{mlp}.bias"] = t(f, f), t(f)
        tensors[p + ".adaLN_modulation.1.weight"], tensors[p + ".adaLN_modulation.1.bias"] = t(3 * f, f), t(3 * f)
    metadata = {"pocket.dim": dim, "pocket.heads": heads, "pocket.layers": layers, "pocket.ff_dim": ff_dim,
                "pocket.latent_dim": latent_dim, "pocket.flow_dim": flow_dim, "pocket.flow_depth": flow_depth,
                "pocket.vocab_size": vocab_size, "pocket.rope_base": 10000.0, "pocket.source_sha256": source}
    return tensors, metadata
