#!/usr/bin/env python3
"""Export the MOSS-Speech Whisper-VQ encoder from the codec GGUF to a Core ML
package for the engine's speech tokenizer sidecar.

The causal convolutions, the causal transformer blocks and the average pool
in front of the vector quantizer are rebuilt in PyTorch from the GGUF tensors
the ggml engine loads, so the export needs neither the checkpoint nor its
remote code. The sidecar maps one 30 s segment of log-mel features,
[1, n_mels, segment_frames], to the pooled states of every speech token the
segment holds, [1, segment_tokens, width]; the nearest-code search stays on
ggml in f32, so the sidecar never decides a code on its own. Shorter segments
are zero-padded on the right, which cannot reach the earlier tokens of a
causal encoder.

Export and compile the sidecar next to the codec GGUF (the engine finds
`moss-speech-codec-tokenizer.mlmodelc` beside `moss-speech-codec-<quant>.gguf`):

  python tts/scripts/export-moss-speech-tokenizer-coreml.py \\
      --gguf models/moss-speech-codec-f16.gguf --compile-dir models
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "moss-speech-codec"
PREFIX = "whispervq."
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SIDECAR_SUFFIX = "-tokenizer"
CONV_KERNEL = 3
CONV_DOWNSAMPLE = 2
LAYER_NORM_EPS = 1e-5
MASKED_SCORE = -1e4
COSINE_EPS = 1e-12


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


def read_owned_tensors(reader):
    return {t.name: read_tensor(t) for t in reader.tensors
            if t.name.startswith(PREFIX) and not t.name.endswith(("codebook", "mel_filters"))}


def ceil_div(value, divisor):
    return -(-value // divisor)


def read_geometry(reader):
    def field(name):
        return int(reader.fields[f"{ARCH}.vq.{name}"].contents())

    hp = {name: field(name) for name in ("sample_rate", "hop_length", "n_mels", "chunk_samples", "block_count",
                                         "embedding_length", "context_length", "pooling_kernel")}
    hp["n_heads"] = field("attention.head_count")
    samples_per_token = hp["hop_length"] * CONV_DOWNSAMPLE * hp["pooling_kernel"]
    hp["tokens"] = ceil_div(hp["chunk_samples"], samples_per_token)
    hp["frames"] = hp["tokens"] * hp["pooling_kernel"]
    hp["mel_frames"] = hp["frames"] * CONV_DOWNSAMPLE
    return hp


def validate_geometry(hp):
    if hp["frames"] > hp["context_length"]:
        raise ValueError(f"a segment of {hp['frames']} frames exceeds the {hp['context_length']}-frame context")
    if hp["embedding_length"] % hp["n_heads"] != 0:
        raise ValueError(f"width {hp['embedding_length']} is not a multiple of {hp['n_heads']} heads")


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")
    hp = read_geometry(reader)
    validate_geometry(hp)
    return read_owned_tensors(reader), hp


class Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = constant(weight)
        self.bias = constant(bias.reshape(-1)) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = constant(weight.reshape(-1))
        self.bias = constant(bias.reshape(-1))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, LAYER_NORM_EPS)


def causal_mask(frames):
    allowed = torch.tril(torch.ones(frames, frames, dtype=torch.bool))
    return torch.where(allowed, torch.zeros(frames, frames), torch.full((frames, frames), MASKED_SCORE))


class CausalBlock(nn.Module):
    def __init__(self, t, prefix, hp):
        super().__init__()
        self.attn_norm = LayerNorm(t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"])
        self.q = Linear(t[prefix + "attn_q.weight"], t[prefix + "attn_q.bias"])
        self.k = Linear(t[prefix + "attn_k.weight"])
        self.v = Linear(t[prefix + "attn_v.weight"], t[prefix + "attn_v.bias"])
        self.out = Linear(t[prefix + "attn_output.weight"], t[prefix + "attn_output.bias"])
        self.ffn_norm = LayerNorm(t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"])
        self.up = Linear(t[prefix + "ffn_up.weight"], t[prefix + "ffn_up.bias"])
        self.down = Linear(t[prefix + "ffn_down.weight"], t[prefix + "ffn_down.bias"])
        self.n_heads = hp["n_heads"]
        self.head_dim = hp["embedding_length"] // hp["n_heads"]
        self.frames = hp["frames"]

    def split_heads(self, x):
        return x.reshape(1, self.frames, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(1, self.frames, self.n_heads * self.head_dim)

    def attention(self, x, mask):
        q, k, v = self.split_heads(self.q(x)), self.split_heads(self.k(x)), self.split_heads(self.v(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / float(self.head_dim) ** 0.5) + mask
        return self.out(self.merge_heads(torch.matmul(torch.softmax(scores, dim=-1), v)))

    def forward(self, x, mask):
        x = x + self.attention(self.attn_norm(x), mask)
        return x + self.down(F.gelu(self.up(self.ffn_norm(x))))


def causal_blocks(t, hp):
    return nn.ModuleList(CausalBlock(t, f"{PREFIX}blk.{i}.", hp) for i in range(hp["block_count"]))


def run_blocks(blocks, x, mask):
    for block in blocks:
        x = block(x, mask)
    return x


class TokenizerEncoder(nn.Module):
    def __init__(self, t, hp):
        super().__init__()
        self.conv1_weight = constant(t[PREFIX + "conv1.weight"])
        self.conv1_bias = constant(t[PREFIX + "conv1.bias"].reshape(-1))
        self.conv2_weight = constant(t[PREFIX + "conv2.weight"])
        self.conv2_bias = constant(t[PREFIX + "conv2.bias"].reshape(-1))
        self.positions = constant(t[PREFIX + "pos_embd"][:hp["frames"]])
        self.blocks = causal_blocks(t, hp)
        self.register_buffer("mask", causal_mask(hp["frames"]), persistent=False)
        self.tokens = hp["tokens"]
        self.pooling = hp["pooling_kernel"]
        self.width = hp["embedding_length"]

    def causal_conv(self, x, weight, bias, stride):
        x = F.conv1d(F.pad(x, (CONV_KERNEL - 1, 0)), weight, bias, stride=stride)
        return F.gelu(x)

    def forward(self, mel):
        x = self.causal_conv(mel, self.conv1_weight, self.conv1_bias, 1)
        x = self.causal_conv(x, self.conv2_weight, self.conv2_bias, CONV_DOWNSAMPLE)
        x = run_blocks(self.blocks, x.transpose(1, 2) + self.positions, self.mask)
        return x.reshape(1, self.tokens, self.pooling, self.width).mean(dim=2)


def load_encoder(gguf_path):
    tensors, hp = read_gguf(gguf_path)
    return TokenizerEncoder(tensors, hp).eval(), hp


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path):
    return strip_quant_tag(Path(gguf_path).stem) + SIDECAR_SUFFIX


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def convert_coreml(model, hp, precision):
    import coremltools as ct

    example = torch.zeros(1, hp["n_mels"], hp["mel_frames"])
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    half = precision == "float16"
    io_dtype = np.float16 if half else np.float32
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=example.shape, dtype=io_dtype)],
        outputs=[ct.TensorType(name="states", dtype=io_dtype)],
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
    p.add_argument("--gguf", required=True, help="the MOSS-Speech codec GGUF (any quantization)")
    p.add_argument("--out", help=".mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc, where the engine looks")
    p.add_argument("--precision", default="float16", choices=["float16", "float32"],
                   help="compute and I/O precision of the exported program (default %(default)s)")
    p.add_argument("--palettize", type=int, default=0, choices=[0, 4, 6, 8],
                   help="k-means weight LUT bits (0 = keep the weights at the export precision)")
    return p.parse_args()


def export(model, hp, args):
    out = Path(args.out) if args.out else Path(args.gguf).with_name(sidecar_stem(args.gguf) + ".mlpackage")
    mlmodel = convert_coreml(model, hp, args.precision)
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
    model, hp = load_encoder(args.gguf)
    print(f"[load] Whisper-VQ encoder ready from {args.gguf}: {hp['block_count']} layers, width "
          f"{hp['embedding_length']}, segment {hp['mel_frames']} mel frames -> {hp['tokens']} tokens")
    export(model, hp, args)


if __name__ == "__main__":
    main()
