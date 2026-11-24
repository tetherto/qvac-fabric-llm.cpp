#!/usr/bin/env python3
"""Export the MOSS-Transcribe-Diarize audio encoder and adaptor from the model
GGUF to a Core ML package for the engine's encoder sidecar.

The Whisper-shaped encoder and the temporal-merge adaptor are rebuilt in
PyTorch from the GGUF tensors the ggml engine loads, so the export needs
neither the checkpoint nor its remote code. The sidecar maps one window of
log-mel features, [1, n_mels, chunk_frames], to the adapted embeddings of
every token the window holds, [1, chunk_tokens, text_width]; the decoder
stays on ggml. The feed-forward down projection runs unscaled: the power-of-two
pre-scaling transcribe_encoder.cpp needs for half-precision CPU accumulation
pushes small activations below the Neural Engine's half-precision normal range
and costs it most of its accuracy.

Export and compile the sidecar next to the GGUF (the engine finds
`moss-transcribe-diarize-encoder.mlmodelc` beside
`moss-transcribe-diarize-<quant>.gguf`):

  python parakeet/scripts/export-moss-transcribe-encoder-coreml.py \\
      --gguf models/moss-transcribe-diarize-f16.gguf --compile-dir models

Check the PyTorch rebuild against the dumps of
dump-moss-transcribe-reference.py first:

  python parakeet/scripts/export-moss-transcribe-encoder-coreml.py \\
      --gguf models/moss-transcribe-diarize-f16.gguf --parity-dir artifacts/moss-ref --no-export
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "moss-transcribe"
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SIDECAR_SUFFIX = "-encoder"
OWNED_PREFIXES = ("enc.", "adaptor.")
CONV_KERNEL = 3
CONV_DOWNSAMPLE = 2
PARITY_COSINE = 0.9999
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
    return {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith(OWNED_PREFIXES)}


def read_geometry(reader):
    def field(name):
        return reader.fields[f"{ARCH}.{name}"].contents()

    return {
        "n_mels": int(field("audio.n_mels")),
        "chunk_frames": int(field("audio.chunk_frames")),
        "n_layers": int(field("encoder.block_count")),
        "width": int(field("encoder.embedding_length")),
        "n_heads": int(field("encoder.attention.head_count")),
        "n_ctx": int(field("encoder.context_length")),
        "eps": float(field("encoder.attention.layer_norm_epsilon")),
        "merge": int(field("adaptor.merge_size")),
        "adaptor_eps": float(field("adaptor.layer_norm_epsilon")),
        "text_width": int(field("text.embedding_length")),
    }


def validate_geometry(hp):
    frames = hp["chunk_frames"]
    if frames % CONV_DOWNSAMPLE != 0 or frames // CONV_DOWNSAMPLE != hp["n_ctx"]:
        raise ValueError(f"chunk_frames {frames} does not downsample to the encoder context {hp['n_ctx']}")
    if hp["n_ctx"] % hp["merge"] != 0:
        raise ValueError(f"encoder context {hp['n_ctx']} is not a multiple of merge size {hp['merge']}")
    if hp["width"] % hp["n_heads"] != 0:
        raise ValueError(f"encoder width {hp['width']} is not a multiple of {hp['n_heads']} heads")


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")
    hp = read_geometry(reader)
    validate_geometry(hp)
    return read_owned_tensors(reader), hp


def chunk_tokens(hp):
    return hp["n_ctx"] // hp["merge"]


class Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = constant(weight)
        self.bias = constant(bias.reshape(-1)) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(self, weight, bias, eps):
        super().__init__()
        self.weight = constant(weight.reshape(-1))
        self.bias = constant(bias.reshape(-1))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class EncoderBlock(nn.Module):
    def __init__(self, t, prefix, hp):
        super().__init__()
        self.attn_norm = LayerNorm(t[prefix + "attn_norm.weight"], t[prefix + "attn_norm.bias"], hp["eps"])
        self.q = Linear(t[prefix + "attn_q.weight"], t[prefix + "attn_q.bias"])
        self.k = Linear(t[prefix + "attn_k.weight"])
        self.v = Linear(t[prefix + "attn_v.weight"], t[prefix + "attn_v.bias"])
        self.out = Linear(t[prefix + "attn_output.weight"], t[prefix + "attn_output.bias"])
        self.ffn_norm = LayerNorm(t[prefix + "ffn_norm.weight"], t[prefix + "ffn_norm.bias"], hp["eps"])
        self.up = Linear(t[prefix + "ffn_up.weight"], t[prefix + "ffn_up.bias"])
        self.down = Linear(t[prefix + "ffn_down.weight"], t[prefix + "ffn_down.bias"])
        self.n_heads = hp["n_heads"]
        self.head_dim = hp["width"] // hp["n_heads"]
        self.frames = hp["n_ctx"]

    def split_heads(self, x):
        return x.reshape(1, self.frames, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(1, self.frames, self.n_heads * self.head_dim)

    def attention(self, x):
        q, k, v = self.split_heads(self.q(x)), self.split_heads(self.k(x)), self.split_heads(self.v(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / float(self.head_dim) ** 0.5)
        return self.out(self.merge_heads(torch.matmul(torch.softmax(scores, dim=-1), v)))

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        return x + self.down(F.gelu(self.up(self.ffn_norm(x))))


def encoder_blocks(t, hp):
    return nn.ModuleList(EncoderBlock(t, f"enc.blk.{i}.", hp) for i in range(hp["n_layers"]))


def run_blocks(blocks, x):
    for block in blocks:
        x = block(x)
    return x


class AudioEncoder(nn.Module):
    def __init__(self, t, hp):
        super().__init__()
        self.conv1_weight = constant(t["enc.conv1.weight"])
        self.conv1_bias = constant(t["enc.conv1.bias"].reshape(-1))
        self.conv2_weight = constant(t["enc.conv2.weight"])
        self.conv2_bias = constant(t["enc.conv2.bias"].reshape(-1))
        self.positions = constant(t["enc.pos_embd"])
        self.blocks = encoder_blocks(t, hp)
        self.output_norm = LayerNorm(t["enc.output_norm.weight"], t["enc.output_norm.bias"], hp["eps"])
        self.fc1 = Linear(t["adaptor.fc1.weight"], t["adaptor.fc1.bias"])
        self.fc2 = Linear(t["adaptor.fc2.weight"], t["adaptor.fc2.bias"])
        self.adaptor_norm = LayerNorm(t["adaptor.norm.weight"], t["adaptor.norm.bias"], hp["adaptor_eps"])
        self.merged_width = hp["width"] * hp["merge"]
        self.tokens = chunk_tokens(hp)

    def states(self, mel):
        padding = CONV_KERNEL // 2
        x = F.gelu(F.conv1d(mel, self.conv1_weight, self.conv1_bias, padding=padding))
        x = F.gelu(F.conv1d(x, self.conv2_weight, self.conv2_bias, stride=CONV_DOWNSAMPLE, padding=padding))
        x = run_blocks(self.blocks, x.transpose(1, 2) + self.positions)
        return self.output_norm(x)

    def adapt(self, states):
        merged = states.reshape(1, self.tokens, self.merged_width)
        return self.adaptor_norm(self.fc2(F.silu(self.fc1(merged))))

    def forward(self, mel):
        return self.adapt(self.states(mel))


def load_encoder(gguf_path):
    tensors, hp = read_gguf(gguf_path)
    return AudioEncoder(tensors, hp).eval(), hp


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path):
    return strip_quant_tag(Path(gguf_path).stem) + SIDECAR_SUFFIX


def compare(tag, got, want):
    n = min(got.size, want.size)
    a = got.reshape(-1)[:n].astype(np.float64)
    b = want.reshape(-1)[:n].astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + COSINE_EPS))
    print(f"[parity] {tag}: n={n} cosine={cos:.8f} max_abs_err={float(np.abs(a - b).max()):.3e}")
    return cos


def run_parity(model, hp, ref_dir):
    ref = Path(ref_dir)
    mel = np.fromfile(ref / "mel_chunk0.bin", dtype=np.float32).reshape(hp["n_mels"], hp["chunk_frames"])
    with torch.no_grad():
        states = model.states(torch.from_numpy(mel)[None])
        embeddings = model.adapt(states)[0].numpy()
    encoder = compare("encoder", states[0].numpy(), np.fromfile(ref / "encoder_chunk0.bin", dtype=np.float32))
    adaptor = compare("adaptor", embeddings, np.fromfile(ref / "audio_embeddings.bin", dtype=np.float32))
    if min(encoder, adaptor) < PARITY_COSINE:
        raise SystemExit("[parity] the PyTorch rebuild does not match the reference model")


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def convert_coreml(model, hp, precision):
    import coremltools as ct

    example = torch.zeros(1, hp["n_mels"], hp["chunk_frames"])
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    half = precision == "float16"
    io_dtype = np.float16 if half else np.float32
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=example.shape, dtype=io_dtype)],
        outputs=[ct.TensorType(name="embeddings", dtype=io_dtype)],
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
    p.add_argument("--gguf", required=True, help="the MOSS-Transcribe-Diarize GGUF (any quantization)")
    p.add_argument("--out", help=".mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc, where the engine looks")
    p.add_argument("--precision", default="float16", choices=["float16", "float32"],
                   help="compute and I/O precision of the exported program (default %(default)s)")
    p.add_argument("--palettize", type=int, default=0, choices=[0, 4, 6, 8],
                   help="k-means weight LUT bits (0 = keep the weights at the export precision)")
    p.add_argument("--parity-dir", help="reference dumps to check the PyTorch rebuild against")
    p.add_argument("--no-export", action="store_true", help="only run the parity check")
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
    print(f"[load] encoder ready from {args.gguf}: {hp['n_layers']} layers, width {hp['width']}, "
          f"window {hp['chunk_frames']} mel frames -> {chunk_tokens(hp)} tokens of {hp['text_width']}")
    if args.parity_dir:
        run_parity(model, hp, args.parity_dir)
    if not args.no_export:
        export(model, hp, args)


if __name__ == "__main__":
    main()
