#!/usr/bin/env python3
"""Export the ACE-Step AutoencoderOobleck VAE decoder from vae-BF16.gguf to a
Core ML package for the Apple Neural Engine sidecar.

The decoder is rebuilt in pure PyTorch from the same GGUF tensors the ggml
engine loads (weight-norm fused, snake alpha/beta exponentiated), so it matches
vae_ggml.cpp numerically. The engine decodes long latents in fixed overlapping
windows of the exported length, so a single fixed-shape Core ML graph covers
every window. 64 latent frames is the Neural Engine sweet spot: the ANE caps
the output width near 131072 samples (68 * 1920 fits, 72 * 1920 splits the
graph onto the GPU), non-64-aligned windows run over 2x slower per frame, and
a window of 128 or more fails ANE compilation outright and falls to a far
slower Core ML GPU path.

Parity check against an engine dump (music-cli --dump-stages <dir>):

  python scripts/export-vae-coreml.py --gguf models/vae-BF16.gguf \
      --parity-latent /tmp/acestep-dump/08_dit_latent.bin \
      --parity-pcm    /tmp/acestep-dump/09_vae_pcm.bin

Export and compile the fixed-shape sidecar:

  python scripts/export-vae-coreml.py --gguf models/vae-BF16.gguf \
      --out models/vae-decoder.mlpackage --compile-dir models
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LATENT_CHANNELS = 64
UPSAMPLE = 1920
STRIDES = [10, 6, 4, 4, 2]
IN_CH = [2048, 1024, 512, 256, 128]
OUT_CH = [1024, 512, 256, 128, 128]
DILATIONS = [1, 3, 9]
WN_EPS = 1e-12
DEFAULT_T_LATENT = 64


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.inv_beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + self.inv_beta * torch.sin(self.alpha * x) ** 2


class ResUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.snake1 = Snake(channels)
        self.conv1 = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation)
        self.snake2 = Snake(channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        y = self.conv1(self.snake1(x))
        y = self.conv2(self.snake2(y))
        return x + y


ANE_BROKEN_DECONV_STRIDES = (4,)


class PhaseUpsample(nn.Module):
    """Exact ConvTranspose1d(k=2s, stride=s, pad=s/2) replacement as a k=3 conv
    plus depth-to-space. The native op miscomputes on the Apple Neural Engine
    for stride 4 (M5, macOS 26: output cosine 0.71 vs CPU); the phase form
    matches the CPU reference at fp16 rounding error."""

    def __init__(self, conv_t):
        super().__init__()
        in_ch, out_ch = conv_t.in_channels, conv_t.out_channels
        stride = conv_t.stride[0]
        self.stride, self.out_ch = stride, out_ch
        self.conv = nn.Conv1d(in_ch, stride * out_ch, 3, padding=1)
        self.conv.weight.data = phase_conv_weight(conv_t.weight.data, out_ch, stride)
        self.conv.bias.data = conv_t.bias.data.repeat(stride)

    def forward(self, x):
        y = self.conv(x)
        batch, _, t = y.shape
        y = y.view(batch, self.stride, self.out_ch, t)
        return y.permute(0, 2, 3, 1).reshape(batch, self.out_ch, self.stride * t)


def phase_conv_weight(w, out_ch, stride):
    half = stride // 2
    kw = torch.zeros(stride * out_ch, w.shape[0], 3)
    for r in range(stride):
        rows = slice(r * out_ch, (r + 1) * out_ch)
        kw[rows, :, 1] = w[:, :, r + half].T
        if r < half:
            kw[rows, :, 0] = w[:, :, r + stride + half].T
        else:
            kw[rows, :, 2] = w[:, :, r - half].T
    return kw


def make_ane_safe(model):
    for block in model.blocks:
        if isinstance(block.conv_t1, nn.ConvTranspose1d) \
                and block.conv_t1.stride[0] in ANE_BROKEN_DECONV_STRIDES:
            block.conv_t1 = PhaseUpsample(block.conv_t1)
    return model


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        kernel = stride * 2
        self.snake1 = Snake(in_ch)
        self.conv_t1 = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride,
                                          padding=(kernel - stride) // 2)
        self.res_units = nn.ModuleList(ResUnit(out_ch, d) for d in DILATIONS)

    def forward(self, x):
        x = self.conv_t1(self.snake1(x))
        for unit in self.res_units:
            x = unit(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(LATENT_CHANNELS, IN_CH[0], 7, padding=3)
        self.blocks = nn.ModuleList(
            DecoderBlock(i, o, s) for i, o, s in zip(IN_CH, OUT_CH, STRIDES))
        self.snake1 = Snake(OUT_CH[-1])
        self.conv2 = nn.Conv1d(OUT_CH[-1], 2, 7, padding=3, bias=False)

    def forward(self, latent):
        x = self.conv1(latent)
        for block in self.blocks:
            x = block(x)
        return self.conv2(self.snake1(x))


def read_gguf_tensors(path):
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    tensors = {}
    for t in reader.tensors:
        data = np.asarray(t.data)
        if t.tensor_type.name == 'BF16':
            raw = data.view(np.uint16).reshape(-1)
            data = (raw.astype(np.uint32) << 16).view(np.float32)
        shape = [int(d) for d in reversed(t.shape)]
        tensors[t.name] = data.reshape(shape).astype(np.float32)
    return tensors


def fuse_weight_norm(tensors, prefix):
    v = tensors[prefix + '.weight_v']
    g = tensors[prefix + '.weight_g'].reshape(-1)
    flat = v.reshape(v.shape[0], -1)
    norm = np.sqrt((flat * flat).sum(axis=1)) + WN_EPS
    return (flat * (g / norm)[:, None]).reshape(v.shape)


def load_snake(module, tensors, prefix):
    alpha = np.exp(tensors[prefix + '.alpha'].reshape(-1))
    beta = np.exp(tensors[prefix + '.beta'].reshape(-1))
    module.alpha.data = torch.from_numpy(alpha).reshape(1, -1, 1)
    module.inv_beta.data = torch.from_numpy((1.0 / beta).astype(np.float32)).reshape(1, -1, 1)


def load_conv(module, tensors, prefix, bias=True):
    module.weight.data = torch.from_numpy(fuse_weight_norm(tensors, prefix).copy())
    if bias:
        module.bias.data = torch.from_numpy(tensors[prefix + '.bias'].reshape(-1).copy())


def load_res_unit(unit, tensors, prefix):
    load_snake(unit.snake1, tensors, prefix + '.snake1')
    load_conv(unit.conv1, tensors, prefix + '.conv1')
    load_snake(unit.snake2, tensors, prefix + '.snake2')
    load_conv(unit.conv2, tensors, prefix + '.conv2')


def load_decoder(tensors):
    model = OobleckDecoder()
    load_conv(model.conv1, tensors, 'decoder.conv1')
    for i, block in enumerate(model.blocks):
        prefix = f'decoder.block.{i}'
        load_snake(block.snake1, tensors, prefix + '.snake1')
        load_conv(block.conv_t1, tensors, prefix + '.conv_t1')
        for r, unit in enumerate(block.res_units):
            load_res_unit(unit, tensors, f'{prefix}.res_unit{r + 1}')
    load_snake(model.snake1, tensors, 'decoder.snake1')
    load_conv(model.conv2, tensors, 'decoder.conv2', bias=False)
    model.eval()
    return model


def read_stage_dump(path):
    header = np.fromfile(path, dtype=np.int32, count=3)
    d0, d1 = int(header[1]), int(header[2])
    data = np.fromfile(path, dtype=np.float32, offset=12)
    return data.reshape(d0, d1)


def run_parity(model, latent_path, pcm_path):
    latent = read_stage_dump(latent_path)
    ref = read_stage_dump(pcm_path)
    x = torch.from_numpy(latent.T.copy()).unsqueeze(0)
    with torch.no_grad():
        out = model(x)[0].numpy()
    got = out.T
    n = min(len(got), len(ref))
    a, b = got[:n].reshape(-1), ref[:n].reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    max_err = float(np.abs(a - b).max())
    print(f'[parity] T_latent={latent.shape[0]} frames={n} cosine={cos:.8f} max_abs_err={max_err:.6f}')
    return cos


def palettize(mlmodel, nbits):
    import coremltools.optimize.coreml as cto
    op_config = cto.OpPalettizerConfig(mode='kmeans', nbits=nbits)
    config = cto.OptimizationConfig(global_config=op_config)
    return cto.palettize_weights(mlmodel, config)


def convert_coreml(model, t_latent, out_path, compile_dir, compute_units, palettize_nbits):
    import coremltools as ct
    example = torch.zeros(1, LATENT_CHANNELS, t_latent)
    traced = torch.jit.trace(make_ane_safe(model), example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='latent', shape=example.shape, dtype=np.float16)],
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
    print(f'[export] saved {out_path} (T_latent={t_latent}, T_audio={t_latent * UPSAMPLE})')
    if compile_dir:
        compiled = compile_model(out_path, compile_dir)
        print(f'[export] compiled {compiled}')
        report_placement(compiled)


def compile_model(package_path, compile_dir):
    result = subprocess.run(
        ['xcrun', 'coremlcompiler', 'compile', str(package_path), str(compile_dir)],
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
    p = argparse.ArgumentParser()
    p.add_argument('--gguf', required=True)
    p.add_argument('--t-latent', type=int, default=DEFAULT_T_LATENT)
    p.add_argument('--out')
    p.add_argument('--compile-dir')
    p.add_argument('--compute-units', default='ALL',
                   choices=['ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'])
    p.add_argument('--palettize', type=int, default=0, choices=[0, 4, 6, 8],
                   help='k-means weight LUT bits (0 = keep float16 weights)')
    p.add_argument('--parity-latent')
    p.add_argument('--parity-pcm')
    return p.parse_args()


def main():
    args = parse_args()
    model = load_decoder(read_gguf_tensors(args.gguf))
    print(f'[load] decoder ready from {args.gguf}')
    if args.parity_latent and args.parity_pcm:
        run_parity(model, args.parity_latent, args.parity_pcm)
    if args.out:
        convert_coreml(model, args.t_latent, args.out, args.compile_dir, args.compute_units,
                       args.palettize)


if __name__ == '__main__':
    main()
