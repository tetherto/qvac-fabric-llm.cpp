#!/usr/bin/env python3
"""Export the MOSS-TTS codec decoder's transformer stack from the decoder GGUF
to a stateful Core ML package for the engine's codec sidecar.

The stack -- every causal sliding-window transformer and patch shuffle from the
quantizer output to the waveform -- is rebuilt in PyTorch from the GGUF
tensors the ggml engine loads, so the export needs neither the checkpoint nor
its remote code. The residual quantizer stays on ggml. One prediction decodes
a fixed chunk of frames, [1, chunk, code_dim] latents -> [1, chunk * hop]
samples, and keeps each attention layer's last context - 1 rotated keys and
values in Core ML state, exactly what codec.cpp keeps between streamed chunks;
a `commit` input of 0 leaves the state untouched, so a partial chunk can be
decoded zero-padded and decoded again once it fills. The engine supplies, per
transformer stage, the rotary tables of the chunk's absolute positions and the
banded causal mask over the cached and fresh keys. The q/k projections are
stored with each head's rotary pairs split into halves, which leaves every
attention score unchanged and lets the rotation run as two contiguous slices.
Core ML state needs macOS 15 / iOS 18.

Export and compile the sidecar next to the decoder GGUF (the engine finds
`moss-codec-decoder.mlmodelc` beside `moss-codec-decoder-<quant>.gguf`):

  python tts/scripts/export-moss-codec-coreml.py \\
      --gguf models/moss-codec-decoder-f16.gguf --compile-dir models
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "moss-tts-audio-decoder"
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
TRANSFORMER = "Transformer"
PATCH = "PatchedPretransform"
DEFAULT_CHUNK_FRAMES = 25
LAYER_NORM_EPS = 1e-5
MASKED_SCORE = -1e4
DEFAULT_MAX_PERIOD = 10000.0


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
                        requires_grad=False)


def bf16_to_f32(raw):
    return (np.asarray(raw).view(np.uint16).reshape(-1).astype(np.uint32) << 16).view(np.float32)


def read_tensor(tensor):
    from gguf.quants import dequantize

    shape = [int(d) for d in reversed(tensor.shape)]
    kind = tensor.tensor_type.name
    if kind in ("F32", "F16"):
        return np.asarray(tensor.data).astype(np.float32).reshape(shape)
    if kind == "BF16":
        return bf16_to_f32(tensor.data).reshape(shape)
    return dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32).reshape(shape)


def as_matrix(array):
    return array.reshape(array.shape[0], -1)


def read_module(field, index):
    prefix = f"decoder.{index}."
    kind = str(field(prefix + "module_type"))
    if kind == PATCH:
        return {"kind": PATCH, "patch": int(field(prefix + "patch_size"))}
    if kind != TRANSFORMER:
        raise ValueError(f"unsupported decoder module {kind}")
    return {"kind": TRANSFORMER, "d_model": int(field(prefix + "d_model")), "heads": int(field(prefix + "num_heads")),
            "layers": int(field(prefix + "num_layers")), "context": int(field(prefix + "context")),
            "max_period": float(field(prefix + "max_period", DEFAULT_MAX_PERIOD))}


def read_modules(field):
    return [read_module(field, index) for index in range(int(field("decoder.block_count")))]


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")

    def field(name, fallback=None):
        key = f"{ARCH}.{name}"
        if key not in reader.fields and fallback is not None:
            return fallback
        return reader.fields[key].contents()

    hp = {"code_dim": int(field("quantizer.output_dim")), "modules": read_modules(field)}
    tensors = {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith("blk.")}
    return tensors, hp


def stage_rates(modules):
    rates, rate = [], 1
    for module in modules:
        if module["kind"] == TRANSFORMER:
            rates.append(rate)
        else:
            rate *= module["patch"]
    return rates, rate


def transformer_modules(modules):
    return [module for module in modules if module["kind"] == TRANSFORMER]


def rotary_half_order(width, head_dim):
    half = head_dim // 2
    within = np.concatenate([np.arange(half) * 2, np.arange(half) * 2 + 1])
    return np.concatenate([head * head_dim + within for head in range(width // head_dim)])


def qkv_order(d_model, head_dim):
    rotary = rotary_half_order(d_model, head_dim)
    return np.concatenate([rotary, d_model + rotary, 2 * d_model + np.arange(d_model)])


def gelu(x, mode):
    return F.gelu(x, approximate="tanh" if mode == "tanh" else "none")


class Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = constant(as_matrix(weight))

    def forward(self, x):
        return F.linear(x, self.weight)


class LayerNorm(nn.Module):
    def __init__(self, weight, bias, width):
        super().__init__()
        self.weight = constant(weight.reshape(-1))
        self.bias = constant(bias.reshape(-1))
        self.width = width

    def forward(self, x):
        return F.layer_norm(x, (self.width,), self.weight, self.bias, LAYER_NORM_EPS)


def optional_scale(t, name, width):
    return constant(t[name].reshape(-1) if name in t else np.ones(width, dtype=np.float32))


class StreamingLayer(nn.Module):
    def __init__(self, t, prefix, module, positions, gelu_mode):
        super().__init__()
        d_model, heads = module["d_model"], module["heads"]
        head_dim = d_model // heads
        self.attn_norm = LayerNorm(t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"], d_model)
        self.ffn_norm = LayerNorm(t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"], d_model)
        self.qkv = Linear(as_matrix(t[prefix + "attn_qkv.weight"])[qkv_order(d_model, head_dim)])
        self.out = Linear(t[prefix + "attn_output.weight"])
        self.up = Linear(t[prefix + "ffn_up.weight"])
        self.down = Linear(t[prefix + "ffn_down.weight"])
        self.attn_scale = optional_scale(t, prefix + "attn_scale.scale", d_model)
        self.ffn_scale = optional_scale(t, prefix + "ffn_scale.scale", d_model)
        self.d_model, self.heads, self.head_dim = d_model, heads, head_dim
        self.positions = positions
        self.gelu_mode = gelu_mode

    def split_heads(self, x):
        return x.reshape(1, self.positions, self.heads, self.head_dim).transpose(1, 2)

    def rotate(self, x, cos, sin):
        half = self.head_dim // 2
        turned = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + turned * sin

    def attention(self, x, cos, sin, mask, old_k, old_v):
        qkv = self.qkv(x)
        d = self.d_model
        q = self.rotate(self.split_heads(qkv[:, :, :d]), cos, sin)
        k = self.rotate(self.split_heads(qkv[:, :, d:2 * d]), cos, sin)
        v = self.split_heads(qkv[:, :, 2 * d:])
        keys, values = torch.cat([old_k, k], dim=2), torch.cat([old_v, v], dim=2)
        scores = torch.matmul(q, keys.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim)) + mask
        attended = torch.matmul(torch.softmax(scores, dim=-1), values)
        out = self.out(attended.transpose(1, 2).reshape(1, self.positions, d))
        return out, keys[:, :, self.positions:, :], values[:, :, self.positions:, :]

    def forward(self, x, cos, sin, mask, old_k, old_v):
        attended, new_k, new_v = self.attention(self.attn_norm(x), cos, sin, mask, old_k, old_v)
        x = x + attended * self.attn_scale
        x = x + self.down(gelu(self.up(self.ffn_norm(x)), self.gelu_mode)) * self.ffn_scale
        return x, new_k, new_v


class StreamingTransformer(nn.Module):
    def __init__(self, t, block, module, positions, gelu_mode):
        super().__init__()
        prefix = f"blk.{block}."
        self.input_proj = Linear(t[prefix + "input_proj.weight"]) if prefix + "input_proj.weight" in t else None
        self.output_proj = Linear(t[prefix + "output_proj.weight"]) if prefix + "output_proj.weight" in t else None
        self.layers = nn.ModuleList(
            StreamingLayer(t, f"{prefix}layer.{layer}.", module, positions, gelu_mode)
            for layer in range(module["layers"]))
        self.capacity = module["context"] - 1
        self.heads = module["heads"]
        self.head_dim = module["d_model"] // module["heads"]


class PatchShuffle(nn.Module):
    def __init__(self, patch, positions, channels):
        super().__init__()
        self.patch, self.positions, self.channels = patch, positions, channels

    def forward(self, x):
        out_channels = self.channels // self.patch
        x = x.reshape(1, self.positions, out_channels, self.patch).permute(0, 1, 3, 2)
        return x.reshape(1, self.positions * self.patch, out_channels)


def build_stages(t, modules, chunk_frames, gelu_mode):
    stages, rate, block, channels = [], 1, 0, None
    for module in modules:
        positions = chunk_frames * rate
        if module["kind"] == TRANSFORMER:
            stages.append(StreamingTransformer(t, block, module, positions, gelu_mode))
            channels = as_matrix(t[f"blk.{block}.output_proj.weight"]).shape[0] \
                if f"blk.{block}.output_proj.weight" in t else module["d_model"]
            block += 1
        else:
            stages.append(PatchShuffle(module["patch"], positions, channels))
            channels //= module["patch"]
            rate *= module["patch"]
    return nn.ModuleList(stages)


def cache_name(kind, stage, layer):
    return f"{kind}_{stage}_{layer}"


class StreamingDecoder(nn.Module):
    def __init__(self, t, hp, chunk_frames, cache_dtype=torch.float16, gelu_mode="tanh"):
        super().__init__()
        self.stages = build_stages(t, hp["modules"], chunk_frames, gelu_mode)
        _, self.hop = stage_rates(hp["modules"])
        self.chunk_frames = chunk_frames
        self.register_caches(cache_dtype)

    def register_caches(self, dtype):
        for index, stage in enumerate(self.transformers()):
            for layer in range(len(stage.layers)):
                shape = (1, stage.heads, stage.capacity, stage.head_dim)
                self.register_buffer(cache_name("k", index, layer), torch.zeros(shape, dtype=dtype))
                self.register_buffer(cache_name("v", index, layer), torch.zeros(shape, dtype=dtype))

    def transformers(self):
        return [stage for stage in self.stages if isinstance(stage, StreamingTransformer)]

    def update(self, cache, fresh, commit):
        old = cache.float()
        cache[:, :, :, :] = (old + commit * (fresh - old)).to(cache.dtype)

    def run_transformer(self, stage, index, x, commit, cos, sin, mask):
        if stage.input_proj is not None:
            x = stage.input_proj(x)
        for layer_index, layer in enumerate(stage.layers):
            k_cache = getattr(self, cache_name("k", index, layer_index))
            v_cache = getattr(self, cache_name("v", index, layer_index))
            x, new_k, new_v = layer(x, cos, sin, mask, k_cache.float(), v_cache.float())
            self.update(k_cache, new_k, commit)
            self.update(v_cache, new_v, commit)
        return stage.output_proj(x) if stage.output_proj is not None else x

    def forward(self, latents, commit, *tables):
        x, transformer = latents, 0
        for stage in self.stages:
            if isinstance(stage, StreamingTransformer):
                cos, sin, mask = tables[3 * transformer:3 * transformer + 3]
                x = self.run_transformer(stage, transformer, x, commit, cos, sin, mask)
                transformer += 1
            else:
                x = stage(x)
        return x.reshape(1, self.chunk_frames * self.hop)


def rope_table(first, count, head_dim, max_period):
    half = head_dim // 2
    inv = max_period ** (-np.arange(half, dtype=np.float64) * 2.0 / head_dim)
    angles = (first + np.arange(count, dtype=np.float64))[:, None] * inv[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    return np.concatenate([cos, cos], axis=1).astype(np.float32), np.concatenate([sin, sin], axis=1).astype(np.float32)


def band_mask(first, count, capacity, context):
    queries = first + np.arange(count)[:, None]
    keys = np.concatenate([first - capacity + np.arange(capacity), first + np.arange(count)])[None, :]
    delta = queries - keys
    visible = (keys >= 0) & (delta >= 0) & (delta < context)
    return np.where(visible, 0.0, MASKED_SCORE).astype(np.float32)


def stage_tables(hp, chunk_frames, committed_frames):
    rates, _ = stage_rates(hp["modules"])
    tables = []
    for module, rate in zip(transformer_modules(hp["modules"]), rates):
        first, count = committed_frames * rate, chunk_frames * rate
        cos, sin = rope_table(first, count, module["d_model"] // module["heads"], module["max_period"])
        tables += [cos, sin, band_mask(first, count, module["context"] - 1, module["context"])]
    return tables


def table_names(hp):
    names = []
    for index, _ in enumerate(transformer_modules(hp["modules"])):
        names += [f"cos_{index}", f"sin_{index}", f"mask_{index}"]
    return names


def state_specs(model):
    import coremltools as ct

    specs = []
    for name, buffer in model.named_buffers():
        specs.append(ct.StateType(wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16), name=name))
    return specs


def load_decoder(gguf_path, chunk_frames, gelu_mode):
    tensors, hp = read_gguf(gguf_path)
    return StreamingDecoder(tensors, hp, chunk_frames, gelu_mode=gelu_mode).eval(), hp, tensors


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path):
    return strip_quant_tag(Path(gguf_path).stem)


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def convert_coreml(model, hp, chunk_frames):
    import coremltools as ct

    latents = torch.zeros(1, chunk_frames, hp["code_dim"])
    commit = torch.ones(1)
    tables = [torch.from_numpy(table) for table in stage_tables(hp, chunk_frames, 0)]
    with torch.no_grad():
        traced = torch.jit.trace(model, (latents, commit, *tables))
    names = ["latents", "commit"] + table_names(hp)
    shapes = [tuple(latents.shape), tuple(commit.shape)] + [tuple(table.shape) for table in tables]
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float16) for name, shape in zip(names, shapes)],
        outputs=[ct.TensorType(name="pcm", dtype=np.float16)],
        states=state_specs(model),
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )


def compile_model(package_path, compile_dir):
    Path(compile_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package_path), str(compile_dir)],
                   capture_output=True, text=True, check=True)
    return Path(compile_dir) / (Path(package_path).stem + ".mlmodelc")


def count_placements(plan):
    counts = {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            device = type(usage.preferred_compute_device).__name__
            counts[device] = counts.get(device, 0) + 1
    return counts


def report_placement(compiled):
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.ALL)
    print(f"[placement] {count_placements(plan)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, help="the MOSS codec decoder GGUF (any quantization)")
    p.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES,
                   help="codec frames per sidecar call (default %(default)s, the engine's stream chunk)")
    p.add_argument("--gelu", default="tanh", choices=["tanh", "erf"],
                   help="feed-forward GELU: tanh matches codec.cpp, erf matches the reference model")
    p.add_argument("--out", help=".mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc, where the engine looks")
    p.add_argument("--palettize", type=int, default=0, choices=[0, 4, 6, 8],
                   help="k-means weight LUT bits (0 = keep float16 weights)")
    return p.parse_args()


def export(model, hp, args):
    out = Path(args.out) if args.out else Path(args.gguf).with_name(sidecar_stem(args.gguf) + ".mlpackage")
    mlmodel = convert_coreml(model, hp, args.chunk_frames)
    if args.palettize:
        mlmodel = palettize(mlmodel, args.palettize)
        print(f"[export] palettized weights to {args.palettize}-bit LUT")
    mlmodel.save(str(out))
    print(f"[export] saved {out}")
    if args.compile_dir:
        compiled = compile_model(out, args.compile_dir)
        print(f"[export] compiled {compiled}")
        report_placement(compiled)


def main():
    args = parse_args()
    model, hp, _ = load_decoder(args.gguf, args.chunk_frames, args.gelu)
    _, hop = stage_rates(hp["modules"])
    print(f"[load] codec decoder ready from {args.gguf}: {len(transformer_modules(hp['modules']))} transformer "
          f"stages, chunk {args.chunk_frames} frames -> {args.chunk_frames * hop} samples")
    export(model, hp, args)


if __name__ == "__main__":
    main()
