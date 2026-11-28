#!/usr/bin/env python3
"""Export the MOSS-SoundEffect diffusion transformer or VAE decoder from the
model GGUF to a Core ML package for the engine's sidecars.

Both stages are rebuilt in PyTorch from the GGUF tensors the ggml engine
loads, so the export needs neither the checkpoint nor its remote code; the
text encoder and the sampler stay on ggml.

--stage dit: one velocity prediction over the whole latent,
  latents [1, channels, frames] + timestep [1, freq_dim] (the sinusoidal
  embedding the engine computes) + context [1, text_tokens, text_dim]
  -> velocity [1, channels, frames]. The q/k projections are stored with each
  head's rotary pairs split into halves, which leaves every attention score
  unchanged and lets the rotation run as two contiguous slices.
--stage vae: one fixed window of latent frames, [1, latent_dim, window]
  -> [1, 1, window * hop] samples. The default window is the ggml decoder's
  256-frame window plus its 32 frames of context on each side. Every transposed convolution is emitted in
  its exact phase form, a two-tap convolution followed by a depth-to-space
  shuffle, instead of ConvTranspose1d, and each Snake's 1 / alpha is clamped
  to the half-precision range (a channel whose alpha is that small adds about
  alpha * x^2, which the clamp scales down further).

Export and compile both next to the GGUF (the engine finds
`moss-sfx-v2-dit.mlmodelc` and `moss-sfx-v2-vae.mlmodelc` beside
`moss-sfx-v2-<quant>.gguf`):

  python tts/scripts/export-moss-sfx-coreml.py --stage dit \\
      --gguf models/moss-sfx-v2-f16.gguf --compile-dir models
  python tts/scripts/export-moss-sfx-coreml.py --stage vae \\
      --gguf models/moss-sfx-v2-f16.gguf --compile-dir models

Check the rebuild against the dumps of dump-moss-sfx-reference.py first with
--parity-dir <dir> --no-export.
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "moss-sfx"
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
STAGE_SUFFIX = {"dit": "-dit", "vae": "-vae"}
ROPE_BASE = 10000.0
TIMESTEP_MAX_PERIOD = 10000.0
MODULATION_ROWS = 6
RESIDUAL_DILATIONS = (1, 3, 9)
RESIDUAL_KERNEL = 7
EDGE_KERNEL = 7
CHANNEL_REDUCTION = 2
DEFAULT_VAE_WINDOW = 320
PARITY_CONTEXT_FRAMES = 32
PARITY_COSINE = 0.9999
COSINE_EPS = 1e-12
TWO_PI = 2.0 * math.pi
HALF_SAFE_INVERSE = 6.0e4
SIN_POLY = (0.9999791158, -0.1666240169, 0.008308850563, -0.0001926317971, 2.147054556e-06)


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


def read_stage_tensors(reader, stage):
    return {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith(stage + ".")}


def read_geometry(reader):
    def field(name):
        return reader.fields[f"{ARCH}.{name}"].contents()

    hp = {
        "dit_layers": int(field("dit.block_count")),
        "dit_width": int(field("dit.embedding_length")),
        "dit_heads": int(field("dit.attention.head_count")),
        "channels": int(field("dit.in_channels")),
        "text_dim": int(field("dit.text_dim")),
        "freq_dim": int(field("dit.freq_dim")),
        "eps": float(field("dit.epsilon")),
        "text_tokens": int(field("text.context_length")),
        "latent_dim": int(field("vae.latent_dim")),
        "decoder_dim": int(field("vae.decoder_dim")),
        "sample_rate": int(field("vae.sample_rate")),
        "rates": [int(r) for r in field("vae.decoder_rates")],
        "max_seconds": float(field("max_seconds")),
        "train_timesteps": int(field("num_train_timesteps")),
    }
    hp["hop"] = int(np.prod(hp["rates"]))
    hp["frames"] = int(round(hp["sample_rate"] * hp["max_seconds"])) // hp["hop"]
    return hp


def read_gguf(path, stage):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")
    return read_stage_tensors(reader, stage), read_geometry(reader)


def timestep_sinusoid(timestep, dim):
    half = dim // 2
    freqs = np.exp(-math.log(TIMESTEP_MAX_PERIOD) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
    args = np.float32(timestep) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


def rope_tables(frames, head_dim):
    half = head_dim // 2
    inv = ROPE_BASE ** (-np.arange(half, dtype=np.float64) * 2.0 / head_dim)
    angles = np.arange(frames, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    return np.concatenate([cos, cos], axis=1), np.concatenate([sin, sin], axis=1)


def rotary_half_order(width, head_dim):
    half = head_dim // 2
    within = np.concatenate([np.arange(half) * 2, np.arange(half) * 2 + 1])
    return np.concatenate([head * head_dim + within for head in range(width // head_dim)])


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def rms_norm(x, weight, eps):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


def plain_norm(x, width, eps):
    return F.layer_norm(x, (width,), None, None, eps)


def modulate(x, shift, scale, width, eps):
    normed = plain_norm(x, width, eps)
    return normed + normed * scale + shift


class Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = constant(weight)
        self.bias = constant(bias.reshape(-1)) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def linear(t, name, order=None):
    weight, bias = t[name + ".weight"], t[name + ".bias"].reshape(-1)
    if order is not None:
        weight, bias = weight[order], bias[order]
    return Linear(weight, bias)


class DitBlock(nn.Module):
    def __init__(self, t, layer, hp):
        super().__init__()
        prefix = f"dit.blk.{layer}."
        order = rotary_half_order(hp["dit_width"], hp["dit_width"] // hp["dit_heads"])
        self.self_q = linear(t, prefix + "self_q", order)
        self.self_k = linear(t, prefix + "self_k", order)
        self.self_v = linear(t, prefix + "self_v")
        self.self_o = linear(t, prefix + "self_o")
        self.self_norm_q = constant(t[prefix + "self_norm_q.weight"].reshape(-1)[order])
        self.self_norm_k = constant(t[prefix + "self_norm_k.weight"].reshape(-1)[order])
        self.cross_q = linear(t, prefix + "cross_q")
        self.cross_k = linear(t, prefix + "cross_k")
        self.cross_v = linear(t, prefix + "cross_v")
        self.cross_o = linear(t, prefix + "cross_o")
        self.cross_norm_q = constant(t[prefix + "cross_norm_q.weight"].reshape(-1))
        self.cross_norm_k = constant(t[prefix + "cross_norm_k.weight"].reshape(-1))
        self.cross_norm_weight = constant(t[prefix + "cross_norm.weight"].reshape(-1))
        self.cross_norm_bias = constant(t[prefix + "cross_norm.bias"].reshape(-1))
        self.up = linear(t, prefix + "ffn_up")
        self.down = linear(t, prefix + "ffn_down")
        self.modulation = constant(t[prefix + "modulation"])
        self.heads = hp["dit_heads"]
        self.head_dim = hp["dit_width"] // hp["dit_heads"]
        self.frames = hp["frames"]
        self.text_tokens = hp["text_tokens"]
        self.width = hp["dit_width"]
        self.eps = hp["eps"]

    def split_heads(self, x, length):
        return x.reshape(1, length, self.heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(1, self.frames, self.heads * self.head_dim)

    def rotate(self, x, cos, sin):
        half = self.head_dim // 2
        turned = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + turned * sin

    def attend(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        return self.merge_heads(torch.matmul(torch.softmax(scores, dim=-1), v))

    def self_attention(self, x, cos, sin):
        q = rms_norm(self.self_q(x), self.self_norm_q, self.eps)
        k = rms_norm(self.self_k(x), self.self_norm_k, self.eps)
        q = self.rotate(self.split_heads(q, self.frames), cos, sin)
        k = self.rotate(self.split_heads(k, self.frames), cos, sin)
        return self.self_o(self.attend(q, k, self.split_heads(self.self_v(x), self.frames)))

    def cross_attention(self, x, context):
        normed = plain_norm(x, self.width, self.eps) * self.cross_norm_weight + self.cross_norm_bias
        q = self.split_heads(rms_norm(self.cross_q(normed), self.cross_norm_q, self.eps), self.frames)
        k = self.split_heads(rms_norm(self.cross_k(context), self.cross_norm_k, self.eps), self.text_tokens)
        v = self.split_heads(self.cross_v(context), self.text_tokens)
        return self.cross_o(self.attend(q, k, v))

    def forward(self, x, context, t_mod, cos, sin):
        mod = self.modulation + t_mod
        x = x + self.self_attention(modulate(x, mod[0], mod[1], self.width, self.eps), cos, sin) * mod[2]
        x = x + self.cross_attention(x, context)
        return x + self.down(gelu_tanh(self.up(modulate(x, mod[3], mod[4], self.width, self.eps)))) * mod[5]


def dit_blocks(t, hp):
    return nn.ModuleList(DitBlock(t, layer, hp) for layer in range(hp["dit_layers"]))


def run_dit_blocks(blocks, x, context, t_mod, cos, sin):
    for block in blocks:
        x = block(x, context, t_mod, cos, sin)
    return x


class Dit(nn.Module):
    def __init__(self, t, hp):
        super().__init__()
        self.patch_embd = linear(t, "dit.patch_embd")
        self.time_embd_1 = linear(t, "dit.time_embd_1")
        self.time_embd_2 = linear(t, "dit.time_embd_2")
        self.time_proj = linear(t, "dit.time_proj")
        self.text_embd_1 = linear(t, "dit.text_embd_1")
        self.text_embd_2 = linear(t, "dit.text_embd_2")
        self.head = linear(t, "dit.head")
        self.head_modulation = constant(t["dit.head.modulation"])
        self.blocks = dit_blocks(t, hp)
        cos, sin = rope_tables(hp["frames"], hp["dit_width"] // hp["dit_heads"])
        self.register_buffer("cos", torch.from_numpy(cos.astype(np.float32)), persistent=False)
        self.register_buffer("sin", torch.from_numpy(sin.astype(np.float32)), persistent=False)
        self.width = hp["dit_width"]
        self.eps = hp["eps"]

    def forward(self, latents, timestep, context):
        x = self.patch_embd(latents.transpose(1, 2))
        t = self.time_embd_2(F.silu(self.time_embd_1(timestep)))[0]
        t_mod = self.time_proj(F.silu(t)).reshape(MODULATION_ROWS, self.width)
        text = self.text_embd_2(gelu_tanh(self.text_embd_1(context)))
        x = run_dit_blocks(self.blocks, x, text, t_mod, self.cos, self.sin)
        mod = self.head_modulation + t
        return self.head(modulate(x, mod[0], mod[1], self.width, self.eps)).transpose(1, 2)


class Snake(nn.Module):
    mode = "poly"

    def __init__(self, alpha, inv):
        super().__init__()
        self.alpha = constant(alpha.reshape(1, -1, 1))
        self.inv = constant(np.clip(inv, -HALF_SAFE_INVERSE, HALF_SAFE_INVERSE).reshape(1, -1, 1))

    def wave(self, t):
        if Snake.mode == "sin":
            return torch.sin(t)
        u = t * (1.0 / TWO_PI)
        r = (u - torch.round(u)) * TWO_PI
        r2 = r * r
        a1, a3, a5, a7, a9 = SIN_POLY
        return r * (a1 + r2 * (a3 + r2 * (a5 + r2 * (a7 + r2 * a9))))

    def forward(self, x):
        s = self.wave(self.alpha * x)
        return x + self.inv * (s * s)


def snake(t, name):
    return Snake(t[name + ".alpha"], t[name + ".inv"])


class Conv(nn.Module):
    def __init__(self, t, name, dilation=1):
        super().__init__()
        self.weight = constant(t[name + ".weight"])
        self.bias = constant(t[name + ".bias"].reshape(-1))
        self.dilation = dilation
        self.padding = (self.weight.shape[-1] - 1) * dilation // 2

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, padding=self.padding, dilation=self.dilation)


def phase_kernel(columns, stride, out_channels):
    in_channels = columns.shape[1]
    taps = columns.shape[0] // out_channels
    if taps != 2 * stride:
        raise ValueError(f"transposed kernel of {taps} taps is not twice the stride {stride}")
    weight = columns.T.reshape(in_channels, out_channels, taps)
    phase = np.zeros((stride * out_channels, in_channels, 2), dtype=np.float32)
    for r in range(stride):
        phase[r * out_channels:(r + 1) * out_channels, :, 0] = weight[:, :, r + stride].T
        phase[r * out_channels:(r + 1) * out_channels, :, 1] = weight[:, :, r].T
    return phase


class Upsample(nn.Module):
    def __init__(self, t, name, stride, length):
        super().__init__()
        bias = t[name + ".bias"].reshape(-1)
        self.out_channels = bias.shape[0]
        self.weight = constant(phase_kernel(t[name + ".weight"], stride, self.out_channels))
        self.bias = constant(bias.reshape(1, -1, 1))
        self.stride = stride
        self.length = length
        self.crop = (stride + 1) // 2

    def forward(self, x):
        z = F.conv1d(F.pad(x, (1, 1)), self.weight)
        z = z.reshape(1, self.stride, self.out_channels, self.length + 1).permute(0, 2, 3, 1)
        full = z.reshape(1, self.out_channels, (self.length + 1) * self.stride)
        return full[:, :, self.crop:self.crop + self.length * self.stride] + self.bias


class ResidualUnit(nn.Module):
    def __init__(self, t, name, dilation):
        super().__init__()
        self.snake1 = snake(t, name + ".snake1")
        self.conv1 = Conv(t, name + ".conv1", dilation)
        self.snake2 = snake(t, name + ".snake2")
        self.conv2 = Conv(t, name + ".conv2")

    def forward(self, x):
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, t, block, stride, length):
        super().__init__()
        name = f"vae.blk.{block}"
        self.snake = snake(t, name + ".snake")
        self.up = Upsample(t, name + ".up", stride, length)
        self.units = nn.ModuleList(ResidualUnit(t, f"{name}.res.{i}", d) for i, d in enumerate(RESIDUAL_DILATIONS))

    def forward(self, x):
        x = self.up(self.snake(x))
        for unit in self.units:
            x = unit(x)
        return x


def block_lengths(rates, frames):
    return [frames * int(np.prod(rates[:block])) for block in range(len(rates))]


def decoder_blocks(t, rates, frames):
    lengths = block_lengths(rates, frames)
    return nn.ModuleList(DecoderBlock(t, block, stride, lengths[block]) for block, stride in enumerate(rates))


def run_decoder_blocks(blocks, x):
    for block in blocks:
        x = block(x)
    return x


class VaeDecoder(nn.Module):
    def __init__(self, t, hp, frames):
        super().__init__()
        self.post_quant = linear(t, "vae.post_quant")
        self.conv_in = Conv(t, "vae.conv_in")
        self.blocks = decoder_blocks(t, hp["rates"], frames)
        self.snake_out = snake(t, "vae.snake_out")
        self.conv_out = Conv(t, "vae.conv_out")

    def forward(self, latents):
        x = self.post_quant(latents.transpose(1, 2)).transpose(1, 2)
        x = run_decoder_blocks(self.blocks, self.conv_in(x))
        return torch.tanh(self.conv_out(self.snake_out(x)))


def load_stage(gguf_path, stage, window):
    tensors, hp = read_gguf(gguf_path, stage)
    module = Dit(tensors, hp) if stage == "dit" else VaeDecoder(tensors, hp, window)
    return module.eval(), hp, tensors


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path, stage):
    return strip_quant_tag(Path(gguf_path).stem) + STAGE_SUFFIX[stage]


def compare(tag, got, want):
    n = min(got.size, want.size)
    a = got.reshape(-1)[:n].astype(np.float64)
    b = want.reshape(-1)[:n].astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + COSINE_EPS))
    print(f"[parity] {tag}: n={n} cosine={cos:.8f} max_abs_err={float(np.abs(a - b).max()):.3e}")
    return cos


def read_bin(ref, name, shape):
    return np.fromfile(Path(ref) / f"{name}.bin", dtype=np.float32).reshape(shape)


def parity_dit(model, hp, ref):
    sigmas = np.fromfile(Path(ref) / "sigmas.bin", dtype=np.float32)
    latents = torch.from_numpy(read_bin(ref, "noise", (hp["channels"], hp["frames"])))[None]
    timestep = torch.from_numpy(timestep_sinusoid(sigmas[0] * hp["train_timesteps"], hp["freq_dim"]))[None]
    worst = 1.0
    for name, context in (("v_pos0", "context_pos"), ("v_neg0", "context_neg")):
        ctx = torch.from_numpy(read_bin(ref, context, (hp["text_tokens"], hp["text_dim"])))[None]
        with torch.no_grad():
            velocity = model(latents, timestep, ctx)[0].numpy()
        worst = min(worst, compare(name, velocity, read_bin(ref, name, (hp["channels"], hp["frames"]))))
    return worst


def parity_vae(tensors, hp, ref):
    latents = torch.from_numpy(read_bin(ref, "latents", (hp["latent_dim"], hp["frames"])))[None]
    audio = np.fromfile(Path(ref) / "audio.bin", dtype=np.float32)
    frames = min(hp["frames"], -(-audio.size // hp["hop"]) + PARITY_CONTEXT_FRAMES)
    with torch.no_grad():
        pcm = VaeDecoder(tensors, hp, frames).eval()(latents[:, :, :frames])[0, 0].numpy()
    return compare("vae", pcm[:audio.size], audio)


def run_parity(model, tensors, hp, stage, ref):
    worst = parity_dit(model, hp, ref) if stage == "dit" else parity_vae(tensors, hp, ref)
    if worst < PARITY_COSINE:
        raise SystemExit("[parity] the PyTorch rebuild does not match the reference pipeline")


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def stage_io(stage, hp, window):
    if stage == "dit":
        inputs = [("latents", (1, hp["channels"], hp["frames"])), ("timestep", (1, hp["freq_dim"])),
                  ("context", (1, hp["text_tokens"], hp["text_dim"]))]
        return inputs, "velocity"
    return [("latents", (1, hp["latent_dim"], window))], "pcm"


def convert_coreml(model, stage, hp, window, precision):
    import coremltools as ct

    inputs, output = stage_io(stage, hp, window)
    examples = tuple(torch.zeros(shape) for _, shape in inputs)
    with torch.no_grad():
        traced = torch.jit.trace(model, examples)
    half = precision == "float16"
    io_dtype = np.float16 if half else np.float32
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=shape, dtype=io_dtype) for name, shape in inputs],
        outputs=[ct.TensorType(name=output, dtype=io_dtype)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16 if half else ct.precision.FLOAT32,
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
    p.add_argument("--gguf", required=True, help="the MOSS-SoundEffect GGUF (any quantization)")
    p.add_argument("--stage", required=True, choices=sorted(STAGE_SUFFIX), help="the stage to export")
    p.add_argument("--window", type=int, default=DEFAULT_VAE_WINDOW,
                   help="latent frames per VAE sidecar call (default %(default)s)")
    p.add_argument("--snake", default="poly", choices=["poly", "sin"],
                   help="VAE sine spelling: a range-reduced polynomial or torch.sin")
    p.add_argument("--out", help=".mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc, where the engine looks")
    p.add_argument("--precision", default="float16", choices=["float16", "float32"],
                   help="compute and I/O precision of the exported program (default %(default)s)")
    p.add_argument("--palettize", type=int, default=0, choices=[0, 4, 6, 8],
                   help="k-means weight LUT bits (0 = keep the weights at the export precision)")
    p.add_argument("--parity-dir", help="dump-moss-sfx-reference.py output to check the rebuild against")
    p.add_argument("--no-export", action="store_true", help="only run the parity check")
    return p.parse_args()


def export(model, hp, args):
    stem = sidecar_stem(args.gguf, args.stage)
    out = Path(args.out) if args.out else Path(args.gguf).with_name(stem + ".mlpackage")
    mlmodel = convert_coreml(model, args.stage, hp, args.window, args.precision)
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
    Snake.mode = args.snake
    model, hp, tensors = load_stage(args.gguf, args.stage, args.window)
    print(f"[load] {args.stage} ready from {args.gguf}: {hp['frames']} latent frames, hop {hp['hop']}")
    if args.parity_dir:
        run_parity(model, tensors, hp, args.stage, args.parity_dir)
    if not args.no_export:
        export(model, hp, args)


if __name__ == "__main__":
    main()
