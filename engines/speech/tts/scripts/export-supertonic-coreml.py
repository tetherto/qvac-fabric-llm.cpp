#!/usr/bin/env python3
"""Export the Supertonic vocoder from the model GGUF to a Core ML package for
the Apple sidecar.

The vocoder -- latent unpack + denormalize, embed conv, ten causal ConvNeXt
blocks (depthwise dilations 1,2,4,1,2,4,1,1,1,1), the pre-baked BatchNorm
affine, and the head (conv, PReLU, pointwise conv) -- is rebuilt in pure
PyTorch from the same GGUF tensors the ggml engine loads, so it matches
supertonic_vocoder.cpp numerically and needs neither the ONNX bundle nor its
runtime. Every convolution is causal with REPLICATE (edge) left padding,
matching causal_replicate_pad_1d in the ggml graph.

The engine walks an utterance in fixed windows of the exported width, keeping
each window's output only past the stack's receptive field
(supertonic_coreml_vocoder_context_frames), so one fixed-shape graph covers
every utterance length. 64 latent frames (4.46 s at 44.1 kHz) is the default;
--window trades compute-unit placement and per-window overhead.

Export and compile the sidecar next to the GGUF (the engine finds
`supertonic2-vocoder.mlmodelc` beside `supertonic2[-<quant>].gguf`):

  python scripts/export-supertonic-coreml.py \
      --gguf models/supertonic2.gguf --compile-dir models

Check the PyTorch rebuild against the reference dumps first (the final_latent
/ wav_full fixtures dump-supertonic-reference.py writes):

  python scripts/export-supertonic-coreml.py \
      --gguf models/supertonic2.gguf --parity-dir artifacts/supertonic-ref
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCODER_DILATIONS = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)
CONVNEXT_NORM_EPS = 1e-6
BATCHNORM_EPS = 1e-5
DEFAULT_WINDOW = 64


def load_gguf_helpers():
    import importlib.util

    module_path = Path(__file__).resolve().parent / 'supertonic_coreml_gguf.py'
    spec = importlib.util.spec_from_file_location('supertonic_coreml_gguf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gguf_helpers = load_gguf_helpers()
EMBED_W = gguf_helpers.EMBED_W
EMBED_B = gguf_helpers.EMBED_B
HEAD_PRELU = gguf_helpers.HEAD_PRELU
read_gguf = gguf_helpers.read_gguf
sidecar_stem = gguf_helpers.sidecar_stem


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
                        requires_grad=False)


class CausalConv(nn.Module):
    """An ONNX row-major kernel [OC, IC, K] as a replicate-left-padded Conv1d,
    matching conv1d_causal_ggml + causal_replicate_pad_1d.  A 2-D [OC, IC]
    kernel is a K=1 pointwise conv that requantize-gguf.py stored squeezed
    (`supertonic.pwconv_squeezed`), re-expanded here like the C++ loader."""

    def __init__(self, kernel, bias, dilation=1):
        super().__init__()
        if kernel.ndim == 2:
            kernel = kernel[:, :, None]
        out_ch, in_ch, taps = kernel.shape
        self.conv = nn.Conv1d(in_ch, out_ch, taps, dilation=dilation, bias=bias is not None)
        self.conv.weight = constant(kernel)
        if bias is not None:
            self.conv.bias = constant(bias.reshape(-1))
        self.left = (taps - 1) * dilation

    def forward(self, x):
        if self.left == 0:
            return self.conv(x)
        return self.conv(F.pad(x, (self.left, 0), mode='replicate'))


class CausalDepthwiseConv(nn.Module):
    """An ONNX depthwise kernel [C, 1, K] as a replicate-left-padded grouped
    Conv1d, matching depthwise_conv1d_causal_ggml."""

    def __init__(self, kernel, bias, dilation):
        super().__init__()
        channels, _, taps = kernel.shape
        self.conv = nn.Conv1d(channels, channels, taps, dilation=dilation, groups=channels,
                              bias=bias is not None)
        self.conv.weight = constant(kernel)
        if bias is not None:
            self.conv.bias = constant(bias.reshape(-1))
        self.left = (taps - 1) * dilation

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0), mode='replicate'))


class ChannelLayerNorm(nn.Module):
    """Per-position layer norm over the channel dim of [1, C, T], matching
    layer_norm_channel_ggml (biased variance, eps inside the sqrt)."""

    def __init__(self, gamma, beta, eps):
        super().__init__()
        self.gamma = constant(gamma.reshape(1, -1, 1))
        self.beta = constant(beta.reshape(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        centred = x - mean
        var = (centred * centred).mean(dim=1, keepdim=True)
        return centred * torch.rsqrt(var + self.eps) * self.gamma + self.beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, t, prefix, dilation):
        super().__init__()
        self.dwconv = CausalDepthwiseConv(t[prefix + '.dwconv.net.weight'],
                                          t.get(prefix + '.dwconv.net.bias'), dilation)
        self.norm = ChannelLayerNorm(t[prefix + '.norm.norm.weight'],
                                     t[prefix + '.norm.norm.bias'], CONVNEXT_NORM_EPS)
        self.pwconv1 = CausalConv(t[prefix + '.pwconv1.weight'], t.get(prefix + '.pwconv1.bias'))
        self.pwconv2 = CausalConv(t[prefix + '.pwconv2.weight'], t.get(prefix + '.pwconv2.bias'))
        self.gamma = constant(t[prefix + '.gamma'].reshape(1, -1, 1))

    def forward(self, x):
        y = self.norm(self.dwconv(x))
        y = self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + y * self.gamma


class Vocoder(nn.Module):
    """From the engine's latent [1, latent_channels, W] to the waveform
    [1, 1, W * base_chunk_size * compress_factor]. Shapes come from the
    tensors, so a tiny synthetic tensor set builds a tiny stack for the unit
    test."""

    def __init__(self, t, hp):
        super().__init__()
        self.latent_dim = int(t['vocoder:tts.ae.latent_mean'].size)
        self.compress_factor = hp['ttl_chunk_compress_factor']
        scale = float(t['vocoder:tts.ttl.normalizer.scale'].reshape(-1)[0])
        self.denorm_scale = constant(
            t['vocoder:tts.ae.latent_std'].reshape(1, -1, 1) / scale)
        self.denorm_shift = constant(t['vocoder:tts.ae.latent_mean'].reshape(1, -1, 1))
        self.embed = CausalConv(t[EMBED_W], t.get(EMBED_B))
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(t, f'vocoder:tts.ae.decoder.convnext.{i}', dilation)
            for i, dilation in enumerate(VOCODER_DILATIONS))
        gamma = t['vocoder:tts.ae.decoder.final_norm.norm.weight'].reshape(-1)
        beta = t['vocoder:tts.ae.decoder.final_norm.norm.bias'].reshape(-1)
        mean = t['vocoder:tts.ae.decoder.final_norm.norm.running_mean'].reshape(-1)
        var = t['vocoder:tts.ae.decoder.final_norm.norm.running_var'].reshape(-1)
        bn_scale = gamma / np.sqrt(var + BATCHNORM_EPS)
        self.bn_scale = constant(bn_scale.reshape(1, -1, 1))
        self.bn_shift = constant((beta - mean * bn_scale).reshape(1, -1, 1))
        self.head1 = CausalConv(t['vocoder:tts.ae.decoder.head.layer1.net.weight'],
                                t.get('vocoder:tts.ae.decoder.head.layer1.net.bias'))
        self.prelu_slope = float(t[HEAD_PRELU].reshape(-1)[0])
        self.head2 = CausalConv(t['vocoder:tts.ae.decoder.head.layer2.weight'], None)

    def unpack(self, latent):
        batch, channels, frames = latent.shape
        x = latent.view(batch, self.latent_dim, self.compress_factor, frames)
        x = x.permute(0, 1, 3, 2)
        return x.reshape(batch, self.latent_dim, frames * self.compress_factor)

    def forward(self, latent):
        x = self.unpack(latent)
        x = x * self.denorm_scale + self.denorm_shift
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = x * self.bn_scale + self.bn_shift
        x = self.head1(x)
        x = F.leaky_relu(x, negative_slope=self.prelu_slope)
        x = self.head2(x)
        batch, samples_per_frame, frames = x.shape
        wav = x.permute(0, 2, 1).reshape(batch, 1, frames * samples_per_frame)
        return wav


def load_vocoder(gguf_path):
    tensors, hp = read_gguf(gguf_path)
    return Vocoder(tensors, hp).eval(), hp


def receptive_field_frames(model, hp):
    """Latent frames of causal left context; must match
    supertonic_coreml_vocoder_context_frames in supertonic_vocoder.cpp."""
    receptive = model.embed.left
    for block in model.blocks:
        receptive += block.dwconv.left
    receptive += model.head1.left + model.head2.left
    factor = hp['ttl_chunk_compress_factor']
    return (receptive + factor - 1) // factor


def compare(tag, got, want):
    n = min(got.size, want.size)
    a, b = got.reshape(-1)[:n].astype(np.float64), want.reshape(-1)[:n].astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_err = float(np.abs(a - b).max())
    print(f'[parity] {tag}: n={n} cosine={cos:.8f} max_abs_err={max_err:.3e}')
    return cos, max_err


def run_parity(model, ref_dir):
    """The PyTorch rebuild against the ONNX reference's own dumps:
    final_latent.npy [1, latent_channels, L] in, wav_full.npy out."""
    ref = Path(ref_dir)
    latent = np.load(ref / 'final_latent.npy').astype(np.float32)
    with torch.no_grad():
        wav = model(torch.from_numpy(latent))[0, 0].numpy()
    cos, max_err = compare('waveform', wav, np.load(ref / 'wav_full.npy'))
    if cos < 0.9999 or max_err > 1e-2:
        raise SystemExit('[parity] the PyTorch rebuild does not match the reference decode')


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode='kmeans', nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def convert_coreml(model, hp, window, out_path, compile_dir, compute_units, palettize_nbits):
    import coremltools as ct

    context = receptive_field_frames(model, hp)
    if window <= context:
        raise SystemExit(f'[export] --window {window} does not clear the receptive field '
                         f'({context} latent frames of causal context)')
    example = torch.zeros(1, hp['latent_channels'], window)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='latent', shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name='wav', dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=getattr(ct.ComputeUnit, compute_units),
        convert_to='mlprogram',
    )
    if palettize_nbits:
        mlmodel = palettize(mlmodel, palettize_nbits)
        print(f'[export] palettized weights to {palettize_nbits}-bit LUT')
    mlmodel.save(str(out_path))
    samples = window * hp['base_chunk_size'] * hp['ttl_chunk_compress_factor']
    print(f'[export] saved {out_path} (window={window} latent frames, '
          f'{samples} samples, {samples / hp["sample_rate"]:.2f} s, '
          f'context={context} frames)')
    if compile_dir:
        compiled = compile_model(out_path, compile_dir)
        print(f'[export] compiled {compiled}')
        report_placement(compiled)


def compile_model(package_path, compile_dir):
    Path(compile_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(package_path), str(compile_dir)],
                   capture_output=True, text=True, check=True)
    return Path(compile_dir) / (Path(package_path).stem + '.mlmodelc')


def report_placement(package_path, compute_units=None):
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(package_path), compute_units=compute_units if compute_units is not None else ct.ComputeUnit.ALL)
    program = plan.model_structure.program
    counts = {}
    for op in program.functions['main'].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        device = type(usage.preferred_compute_device).__name__
        counts[device] = counts.get(device, 0) + 1
    print(f'[placement] {counts}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gguf', required=True, help='the Supertonic model GGUF')
    p.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                   help='latent frames per sidecar call (default %(default)s)')
    p.add_argument('--out', help='.mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)')
    p.add_argument('--compile-dir', help='also compile to <dir>/<stem>.mlmodelc, where the engine looks')
    p.add_argument('--compute-units', default='ALL', choices=['ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'])
    p.add_argument('--palettize', type=int, default=0, choices=[0, 4, 6, 8],
                   help='k-means weight LUT bits (0 = keep float16 weights)')
    p.add_argument('--parity-dir', help='reference dumps to check the PyTorch rebuild against')
    p.add_argument('--no-export', action='store_true', help='only run the parity check')
    return p.parse_args()


def main():
    args = parse_args()
    model, hp = load_vocoder(args.gguf)
    print(f'[load] vocoder ready from {args.gguf}: latent_channels={hp["latent_channels"]} '
          f'chunk={hp["base_chunk_size"]}x{hp["ttl_chunk_compress_factor"]} '
          f'context={receptive_field_frames(model, hp)} latent frames')
    if args.parity_dir:
        run_parity(model, args.parity_dir)
    if args.no_export:
        return
    out = Path(args.out) if args.out else Path(args.gguf).with_name(sidecar_stem(args.gguf) + '.mlpackage')
    convert_coreml(model, hp, args.window, out, args.compile_dir, args.compute_units, args.palettize)


if __name__ == '__main__':
    main()
