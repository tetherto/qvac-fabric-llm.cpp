#!/usr/bin/env python3
"""Export the Audio8 codec's synthesis stack from the decoder GGUF to a Core ML
package for the Apple Neural Engine sidecar.

The stack -- the two stride-2 upsampling stages (transposed conv + ConvNeXt
block) and the DAC decoder (input conv, four Snake / transposed-conv /
residual-unit stages, Snake, output conv, tanh) -- is rebuilt in pure PyTorch
from the same GGUF tensors the ggml engine loads, so it matches
codec_decode.cpp numerically and needs neither the checkpoint nor its remote
code. The quantizer banks and the windowed post transformer stay on ggml: the
sidecar takes the post transformer's output, [1, latent_dim, window] frames,
and returns [1, 1, window * frame_size] samples.

Every convolution is causal, and every transposed convolution is exported in
its exact phase form -- a causal Conv1d over the input followed by a
depth-to-space shuffle, which is how codec_ops.cpp computes it too -- rather
than as ConvTranspose1d, whose native Neural Engine kernel has been measured
to miscompute at stride 4 (see audiogen/scripts/export-vae-coreml.py).

The engine walks an utterance in fixed windows of the exported width, dropping
each window's causal context (coreml_windows.h), so one fixed-shape graph
covers every utterance length. 64 frames (2.97 s, 131072 samples) is the
default; --window trades Neural Engine placement and per-window overhead.

Export and compile the sidecar next to the GGUF (the engine finds
`audio8-codec-decoder.mlmodelc` beside `audio8-codec-decoder-<quant>.gguf`):

  python scripts/export-audio8-codec-coreml.py \
      --gguf models/audio8-codec-decoder-f32.gguf --compile-dir models

Check the PyTorch rebuild against the reference dumps first (the codec_post /
codec_latent / codec_wav fixtures dump-audio8-codec-reference.py writes):

  python scripts/export-audio8-codec-coreml.py \
      --gguf models/audio8-codec-decoder-f32.gguf --parity-dir artifacts/audio8-ref
"""
import argparse
import subprocess
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KV = 'audio8.codec.'
DEFAULT_WINDOW = 64
QUANT_TAGS = ('f32', 'f16', 'bf16', 'q8_0', 'q5_0', 'q4_0')


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
                        requires_grad=False)


# The Neural Engine takes this stack's causal convolutions only up to 16384
# samples per row (measured on an M2, macOS 15.7: 16384 stays, 32768 and up
# go to the GPU, and the plan then alternates devices op by op through the
# whole tail). Folding a long sequence into rows of FOLD_ROW samples, (1, C, L)
# -> (1, C, L / FOLD_ROW, FOLD_ROW), keeps every op on it: each causal
# convolution takes its left context from the end of the previous row (zeros
# for the first row, which is the causal padding), so the fold is exact.
# 0 disables folding (`--fold-rows 0`).
FOLD_ROW = 8192


def fold(x):
    """(1, C, L) -> (1, C, L / FOLD_ROW, FOLD_ROW) when the sequence is long
    enough to leave the Neural Engine and divides evenly; otherwise unchanged."""
    if FOLD_ROW and x.dim() == 3 and x.shape[-1] > FOLD_ROW and x.shape[-1] % FOLD_ROW == 0:
        return x.reshape(x.shape[0], x.shape[1], x.shape[-1] // FOLD_ROW, FOLD_ROW)
    return x


def unfold(x):
    if x.dim() == 4:
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    return x


def causal_context(x, left):
    """Left-pad a folded (1, C, H, W) tensor with the last `left` samples of
    the previous row (zeros before row 0), so a valid conv along W equals the
    causal conv along the unfolded sequence."""
    if left == 0:
        return x
    prev = torch.cat([torch.zeros_like(x[:, :, :1, :]), x[:, :, :-1, :]], dim=2)
    return torch.cat([prev[:, :, :, -left:], x], dim=3)


def per_channel(param, x):
    """A (1, C, 1) parameter broadcast against a 3-D or folded 4-D activation."""
    return param if x.dim() == 3 else param.unsqueeze(-1)


class CausalConv(nn.Module):
    """A ggml tap-major kernel [taps, out, in] as a left-padded Conv1d."""

    def __init__(self, kernel, bias, dilation=1):
        super().__init__()
        taps, out_ch, in_ch = kernel.shape
        self.conv = nn.Conv1d(in_ch, out_ch, taps, dilation=dilation, bias=bias is not None)
        self.conv.weight = constant(kernel.transpose(1, 2, 0))
        if bias is not None:
            self.conv.bias = constant(bias.reshape(-1))
        self.left = (taps - 1) * dilation

    def forward(self, x):
        if x.dim() == 4:
            return F.conv2d(causal_context(x, self.left), self.conv.weight.unsqueeze(2),
                            self.conv.bias, dilation=(1, self.conv.dilation[0]))
        return self.conv(F.pad(x, (self.left, 0)))


class CausalDepthwiseConv(nn.Module):
    """A ggml depthwise kernel [taps, channels] as a grouped Conv1d."""

    def __init__(self, kernel, bias):
        super().__init__()
        taps, channels = kernel.shape
        self.conv = nn.Conv1d(channels, channels, taps, groups=channels, bias=bias is not None)
        self.conv.weight = constant(kernel.T[:, None, :])
        if bias is not None:
            self.conv.bias = constant(bias.reshape(-1))
        self.left = taps - 1

    def forward(self, x):
        if x.dim() == 4:
            return F.conv2d(causal_context(x, self.left), self.conv.weight.unsqueeze(2),
                            self.conv.bias, groups=self.conv.groups)
        return self.conv(F.pad(x, (self.left, 0)))


def phase_kernel(kernel, stride):
    """The causal Conv1d kernel [stride * out, in, taps / stride] whose output,
    depth-to-space shuffled, is codec_ops.cpp's causal transposed convolution:
    output column l * stride + p sums W[:, :, g * stride + p] x[l - g] over the
    whole strides g the kernel spans. `kernel` is ggml's [taps, out, in]."""
    taps, out_ch, in_ch = kernel.shape
    if taps % stride != 0:
        raise ValueError(f'transposed kernel of {taps} taps is not a multiple of stride {stride}')
    groups = taps // stride
    phase = np.zeros((stride * out_ch, in_ch, groups), dtype=np.float32)
    for p in range(stride):
        for position in range(groups):
            g = groups - 1 - position
            phase[p * out_ch:(p + 1) * out_ch, :, position] = kernel[g * stride + p]
    return phase


class CausalConvTranspose(nn.Module):
    """codec_ops.cpp's causal transposed convolution in its exact phase form."""

    def __init__(self, kernel, bias, stride):
        super().__init__()
        taps, out_ch, in_ch = kernel.shape
        groups = taps // stride
        self.conv = nn.Conv1d(in_ch, stride * out_ch, groups, bias=bias is not None)
        self.conv.weight = constant(phase_kernel(kernel, stride))
        if bias is not None:
            self.conv.bias = constant(np.tile(bias.reshape(-1), stride))
        self.left = groups - 1
        self.stride = stride
        self.out_ch = out_ch

    def forward(self, x):
        if x.dim() == 4:
            y = F.conv2d(causal_context(x, self.left), self.conv.weight.unsqueeze(2), self.conv.bias)
            batch, _, rows, width = y.shape
            y = y.view(batch, self.stride, self.out_ch, rows, width)
            # Depth-to-space along the row, then split each row of width *
            # stride samples into stride rows of the original width: a
            # row-major reshape, so time order is kept and the row width stays
            # what the Neural Engine accepts.
            y = y.permute(0, 2, 3, 4, 1).reshape(batch, self.out_ch, rows, width * self.stride)
            return y.reshape(batch, self.out_ch, rows * self.stride, width)
        y = self.conv(F.pad(x, (self.left, 0)))
        batch, _, length = y.shape
        y = y.view(batch, self.stride, self.out_ch, length)
        return y.permute(0, 2, 3, 1).reshape(batch, self.out_ch, length * self.stride)


class Snake(nn.Module):
    """x + sin(alpha x)^2 / (alpha + eps), the division folded into a multiply.

    Two spellings of the sine, chosen by `Snake.mode`:

    `sin`: `torch.sin`, squared as a product on purpose -- `** 2` lowers to a
    `pow` the Neural Engine miscomputes when its result feeds a convolution
    (measured on an M2, macOS 15.7: the first DAC stage at cosine 0.39 against
    the CPU, every other unit exact), while `sin * sin` is exact everywhere.
    The Neural Engine has no `sin` kernel on macOS 15, though, so every Snake
    leaves it and the plan pays a device handoff per activation.

    `poly`: an odd degree-9 polynomial for sin on [-pi, pi] (a least-squares
    fit on Chebyshev nodes, max error 6e-6) after reducing the argument in
    *turns*: u = t / 2pi, f = u - round(u), r = 2pi f. Every op has
    a Neural Engine kernel, so the whole stack stays on it. The reduction must
    be done in turns: reducing in radians (t - 2pi round(t / 2pi)) is
    miscomputed on the Neural Engine (NaN / inf on an M2, macOS 15.7) while the
    turns form is exact there (probe: cosine 0.999996 against fp32 torch,
    the same as `sin`). fp16 keeps `u` exact to the integer for |u| < 2048,
    i.e. |alpha x| < ~12800; the stack's activations stay far below that."""

    mode = 'poly'

    def __init__(self, alpha, eps):
        super().__init__()
        alpha = alpha.reshape(1, -1, 1).astype(np.float32)
        self.alpha = constant(alpha)
        self.scale = constant(1.0 / (alpha + eps))

    def forward(self, x):
        t = per_channel(self.alpha, x) * x
        if Snake.mode == 'poly':
            u = t * (1.0 / (2.0 * math.pi))
            r = (u - torch.round(u)) * (2.0 * math.pi)
            r2 = r * r
            # Degree-9 odd least-squares fit on Chebyshev nodes over [-pi, pi]
            # (max error 6e-6, float32 Horner), not the Taylor coefficients
            # (2e-3 at the ends of the range).
            wave = r * (0.9999791158 + r2 * (-0.1666240169 + r2 * (0.008308850563
                        + r2 * (-0.0001926317971 + r2 * 2.147054556e-06))))
        else:
            wave = torch.sin(t)
        return x + per_channel(self.scale, x) * (wave * wave)


def pointwise(weight, bias):
    """The ConvNeXt block's pointwise projections, stored as Linear [out, in]
    or as a one-tap kernel [1, out, in]."""
    if weight.ndim == 3:
        return CausalConv(weight, bias)
    return CausalConv(weight[None, :, :], bias)


class ConvNeXtBlock(nn.Module):
    def __init__(self, t, prefix, eps):
        super().__init__()
        self.dwconv = CausalDepthwiseConv(t[prefix + '/dwconv/conv/weight'],
                                          t.get(prefix + '/dwconv/conv/bias'))
        self.norm_w = constant(t[prefix + '/norm/weight'].reshape(1, -1, 1))
        self.norm_b = constant(t[prefix + '/norm/bias'].reshape(1, -1, 1))
        self.pwconv1 = pointwise(t[prefix + '/pwconv1/weight'], t.get(prefix + '/pwconv1/bias'))
        self.pwconv2 = pointwise(t[prefix + '/pwconv2/weight'], t.get(prefix + '/pwconv2/bias'))
        self.gamma = constant(t[prefix + '/gamma'].reshape(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        y = self.dwconv(x)
        mean = y.mean(dim=1, keepdim=True)
        centred = y - mean
        var = (centred * centred).mean(dim=1, keepdim=True)
        y = (centred * torch.rsqrt(var + self.eps) * per_channel(self.norm_w, y)
             + per_channel(self.norm_b, y))
        y = self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + y * per_channel(self.gamma, y)


class ResidualUnit(nn.Module):
    def __init__(self, t, prefix, dilation, snake_eps):
        super().__init__()
        self.snake1 = Snake(t[prefix + '/0/alpha'], snake_eps)
        self.conv1 = CausalConv(t[prefix + '/1/conv/weight'], t.get(prefix + '/1/conv/bias'), dilation)
        self.snake2 = Snake(t[prefix + '/2/alpha'], snake_eps)
        self.conv2 = CausalConv(t[prefix + '/3/conv/weight'], t.get(prefix + '/3/conv/bias'))

    def forward(self, x):
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class DecoderStage(nn.Module):
    def __init__(self, t, prefix, stride, dilations, snake_eps):
        super().__init__()
        self.snake = Snake(t[prefix + '/0/alpha'], snake_eps)
        self.conv_t = CausalConvTranspose(t[prefix + '/1/conv/weight'], t.get(prefix + '/1/conv/bias'), stride)
        self.units = nn.ModuleList(
            ResidualUnit(t, f'{prefix}/{2 + r}/block', d, snake_eps) for r, d in enumerate(dilations))

    def forward(self, x):
        # Fold on entry, and again after the upsampling: a stage whose input is
        # still short can produce a row the Neural Engine no longer takes.
        x = fold(self.conv_t(self.snake(fold(x))))
        for unit in self.units:
            x = unit(x)
        return x


class SynthesisStack(nn.Module):
    """From the post transformer's output to the waveform: the upsampling
    stages, then the DAC decoder. Shapes come from the tensors, so a tiny
    synthetic tensor set builds a tiny stack for the unit test."""

    def __init__(self, t, hp):
        super().__init__()
        snake_eps = hp['snake_epsilon']
        self.upsample = nn.ModuleList()
        for i in range(hp['upsample_stages']):
            self.upsample.append(CausalConvTranspose(t[f'q/up/{i}/0/conv/weight'],
                                                     t.get(f'q/up/{i}/0/conv/bias'),
                                                     hp['upsample_stride']))
            self.upsample.append(ConvNeXtBlock(t, f'q/up/{i}/1', hp['convnext_norm_eps']))
        self.dec_in = CausalConv(t['dec/0/conv/weight'], t.get('dec/0/conv/bias'))
        strides = hp['decoder_strides']
        self.stages = nn.ModuleList(
            DecoderStage(t, f'dec/{n + 1}/block', stride, hp['residual_dilations'], snake_eps)
            for n, stride in enumerate(strides))
        tail = len(strides) + 1
        self.snake_out = Snake(t[f'dec/{tail}/alpha'], snake_eps)
        self.dec_out = CausalConv(t[f'dec/{tail + 1}/conv/weight'], t.get(f'dec/{tail + 1}/conv/bias'))

    def latent(self, post):
        x = post
        for stage in self.upsample:
            x = stage(x)
        return x

    def forward(self, post):
        x = self.dec_in(self.latent(post))
        for stage in self.stages:
            x = stage(x)
        return unfold(torch.tanh(self.dec_out(self.snake_out(fold(x)))))


# The quantizer's resamplers are a fixed pair of stride-2 stages
# (read_resamplers in gguf.cpp).
UPSAMPLE_STAGES = 2
UPSAMPLE_STRIDE = 2


def read_gguf(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(path))
    tensors = {}
    for t in reader.tensors:
        shape = [int(d) for d in reversed(t.shape)]
        if t.tensor_type.name in ('F32', 'F16'):
            data = np.asarray(t.data).astype(np.float32)
        elif t.tensor_type.name == 'BF16':
            raw = np.asarray(t.data).view(np.uint16).reshape(-1)
            data = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            data = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        tensors[t.name] = data.reshape(shape)

    def field(name):
        return reader.fields[KV + name].contents()

    hp = {
        'latent_dim': int(field('latent_dim')),
        'frame_size': int(field('frame_size')),
        'sample_rate': int(field('sample_rate')),
        'snake_epsilon': float(field('snake_epsilon')),
        'convnext_norm_eps': float(field('convnext_norm_eps')),
        'decoder_strides': [int(s) for s in field('decoder_strides')],
        'residual_dilations': [int(d) for d in field('residual_dilations')],
        'upsample_stages': UPSAMPLE_STAGES,
        'upsample_stride': UPSAMPLE_STRIDE,
        'part': str(field('part')),
    }
    if hp['part'] != 'decoder':
        raise ValueError(f'{path} holds the codec {hp["part"]}, not the decoder')
    return tensors, hp


def load_stack(gguf_path):
    tensors, hp = read_gguf(gguf_path)
    return SynthesisStack(tensors, hp).eval(), hp


def sidecar_stem(gguf_path):
    """coreml_codec_sidecar_path's rule: drop the extension and a trailing
    quantisation tag, so every tier of the decoder shares one sidecar."""
    stem = Path(gguf_path).stem
    for sep in ('-', '.'):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def compare(tag, got, want):
    n = min(got.size, want.size)
    a, b = got.reshape(-1)[:n].astype(np.float64), want.reshape(-1)[:n].astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_err = float(np.abs(a - b).max())
    print(f'[parity] {tag}: n={n} cosine={cos:.8f} max_abs_err={max_err:.3e}')
    return cos, max_err


def run_parity(model, ref_dir):
    """The PyTorch rebuild against the reference codec's own stage dumps:
    codec_post.npy [latent_dim, T] in, codec_latent.npy [latent_dim, 4T] and
    codec_wav.npy [N] out."""
    ref = Path(ref_dir)
    post = np.load(ref / 'codec_post.npy').astype(np.float32)
    x = torch.from_numpy(post).unsqueeze(0)
    with torch.no_grad():
        latent = model.latent(x)[0].numpy()
        pcm = model(x)[0, 0].numpy()
    compare('latent', latent, np.load(ref / 'codec_latent.npy'))
    cos, max_err = compare('waveform', pcm, np.load(ref / 'codec_wav.npy'))
    if cos < 0.9999 or max_err > 1e-3:
        raise SystemExit('[parity] the PyTorch rebuild does not match the reference decode')


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto

    op_config = cto.OpPalettizerConfig(mode='kmeans', nbits=nbits)
    return cto.palettize_weights(mlmodel, cto.OptimizationConfig(global_config=op_config))


def convert_coreml(model, hp, window, out_path, compile_dir, compute_units, palettize_nbits):
    import coremltools as ct

    example = torch.zeros(1, hp['latent_dim'], window)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='post', shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name='pcm', dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=getattr(ct.ComputeUnit, compute_units),
        convert_to='mlprogram',
    )
    if palettize_nbits:
        mlmodel = palettize(mlmodel, palettize_nbits)
        print(f'[export] palettized weights to {palettize_nbits}-bit LUT')
    mlmodel.save(str(out_path))
    print(f'[export] saved {out_path} (window={window} frames, '
          f'{window * hp["frame_size"]} samples, {window * hp["frame_size"] / hp["sample_rate"]:.2f} s)')
    if compile_dir:
        compiled = compile_model(out_path, compile_dir)
        print(f'[export] compiled {compiled}')
        report_placement(compiled)


def compile_model(package_path, compile_dir):
    Path(compile_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(package_path), str(compile_dir)],
                   capture_output=True, text=True, check=True)
    return Path(compile_dir) / (Path(package_path).stem + '.mlmodelc')


def report_placement(package_path):
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(package_path), compute_units=ct.ComputeUnit.ALL)
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
    p.add_argument('--gguf', required=True, help='the codec decoder GGUF')
    p.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                   help='post frames per sidecar call (default %(default)s)')
    p.add_argument('--out', help='.mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)')
    p.add_argument('--compile-dir', help='also compile to <dir>/<stem>.mlmodelc, where the engine looks')
    p.add_argument('--fold-rows', type=int, default=FOLD_ROW,
                   help='fold stages longer than this many samples into rows of that width so the '
                        'Neural Engine keeps them (16384 is its limit, measured on an M2); 0 leaves sequences flat')
    p.add_argument('--snake', default='poly', choices=['poly', 'sin'],
                   help='sine spelling: a range-reduced polynomial (stays on the Neural Engine) or torch.sin (leaves it)')
    p.add_argument('--compute-units', default='ALL', choices=['ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'])
    p.add_argument('--palettize', type=int, default=0, choices=[0, 4, 6, 8],
                   help='k-means weight LUT bits (0 = keep float16 weights)')
    p.add_argument('--parity-dir', help='reference dumps to check the PyTorch rebuild against')
    p.add_argument('--no-export', action='store_true', help='only run the parity check')
    return p.parse_args()


def main():
    args = parse_args()
    Snake.mode = args.snake
    global FOLD_ROW
    FOLD_ROW = args.fold_rows
    model, hp = load_stack(args.gguf)
    print(f'[load] synthesis stack ready from {args.gguf}: latent_dim={hp["latent_dim"]} '
          f'frame_size={hp["frame_size"]} strides={hp["decoder_strides"]}')
    if args.parity_dir:
        run_parity(model, args.parity_dir)
    if args.no_export:
        return
    out = Path(args.out) if args.out else Path(args.gguf).with_name(sidecar_stem(args.gguf) + '.mlpackage')
    convert_coreml(model, hp, args.window, out, args.compile_dir, args.compute_units, args.palettize)


if __name__ == '__main__':
    main()
