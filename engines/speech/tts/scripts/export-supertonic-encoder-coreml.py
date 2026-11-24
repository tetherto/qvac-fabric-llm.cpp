#!/usr/bin/env python3
"""Export a Supertonic 1/2/3 GGUF text encoder to <model>-encoder.mlmodelc.

Fixed token capacity (128 by default), with a runtime valid length. Replicate
padding is restored before EVERY dilated ConvNeXt convolution; padded keys
are masked in self attention. Valid outputs therefore match an unpadded run.
Longer inputs use the runtime's ggml fallback, never independent windows.
Requires torch, numpy, gguf and coremltools; export from F32/F16 weights.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from supertonic_coreml_gguf import dequantized, resolve_tensor_names, sidecar_stem

TEXT = 'text_encoder:tts.ttl.text_encoder.'
SPEECH = 'text_encoder:tts.ttl.speech_prompted_text_encoder.'


def parameter(value):
    return nn.Parameter(torch.from_numpy(np.array(value, dtype=np.float32, copy=True)), requires_grad=False)


class Linear(nn.Module):
    def __init__(self, tensors, prefix, matmul=False):
        super().__init__()
        w = tensors[prefix + '.weight']
        # ONNX MatMul constants are [IC, OC]; Conv1d weights are [OC, IC, 1].
        if matmul:
            w = w.T
        else:
            w = w.reshape(w.shape[0], -1)
        self.weight = parameter(w)
        self.bias = parameter(tensors[prefix + '.bias'].reshape(-1))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Norm(nn.Module):
    def __init__(self, tensors, prefix):
        super().__init__()
        self.weight = parameter(tensors[prefix + '.weight'].reshape(-1))
        self.bias = parameter(tensors[prefix + '.bias'].reshape(-1))

    def forward(self, x):
        return F.layer_norm(x, (self.weight.numel(),), self.weight, self.bias, 1e-6)


class ConvNeXt(nn.Module):
    def __init__(self, tensors, prefix, dilation):
        super().__init__()
        self.kernel = parameter(tensors[prefix + '.dwconv.weight'])
        self.bias = parameter(tensors[prefix + '.dwconv.bias'].reshape(-1))
        self.dilation = dilation
        self.pad = (self.kernel.shape[-1] - 1) * dilation // 2
        self.norm = Norm(tensors, prefix + '.norm.norm')
        self.pw1 = Linear(tensors, prefix + '.pwconv1')
        self.pw2 = Linear(tensors, prefix + '.pwconv2')
        self.gamma = parameter(tensors[prefix + '.gamma'].reshape(-1))

    def forward(self, x, valid, last):
        # Filling the padded tail once at the input is insufficient: each block
        # changes the last valid frame. Repeat it again before each convolution.
        edge = (x * last).sum(1, keepdim=True)
        y = (x * valid + edge * (1.0 - valid)).transpose(1, 2)
        y = F.conv1d(F.pad(y, (self.pad, self.pad), mode='replicate'),
                     self.kernel, self.bias, dilation=self.dilation, groups=self.kernel.shape[0])
        y = self.norm(y.transpose(1, 2))
        return x + self.gamma * self.pw2(F.gelu(self.pw1(y), approximate='none'))


class AttentionLayer(nn.Module):
    def __init__(self, tensors, index):
        super().__init__()
        prefix = TEXT + f'attn_encoder.attn_layers.{index}'
        self.q = Linear(tensors, prefix + '.conv_q')
        self.k = Linear(tensors, prefix + '.conv_k')
        self.v = Linear(tensors, prefix + '.conv_v')
        self.out = Linear(tensors, prefix + '.conv_o')
        self.rel_k = parameter(tensors[prefix + '.emb_rel_k'].reshape(9, 64))
        self.rel_v = parameter(tensors[prefix + '.emb_rel_v'].reshape(9, 64))
        self.norm1 = Norm(tensors, TEXT + f'attn_encoder.norm_layers_1.{index}.norm')
        self.norm2 = Norm(tensors, TEXT + f'attn_encoder.norm_layers_2.{index}.norm')
        self.ff1 = Linear(tensors, TEXT + f'attn_encoder.ffn_layers.{index}.conv_1')
        self.ff2 = Linear(tensors, TEXT + f'attn_encoder.ffn_layers.{index}.conv_2')

    def forward(self, x, key_mask, bands):
        length = x.shape[1]
        def heads(y):
            return y.reshape(1, length, 4, 64).permute(0, 2, 1, 3)
        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) * 0.125
        rel_scores = torch.matmul(q, self.rel_k.T) * 0.125
        for i in range(9):
            scores = scores + rel_scores[..., i:i+1] * bands[i]
        probs = torch.softmax(scores + key_mask, dim=-1)
        y = torch.matmul(probs, v)
        for i in range(9):
            y = y + (probs * bands[i]).sum(-1, keepdim=True) * self.rel_v[i]
        y = y.permute(0, 2, 1, 3).reshape(1, length, 256)
        x = self.norm1(x + self.out(y))
        return self.norm2(x + self.ff2(F.relu(self.ff1(x))))


class SpeechAttention(nn.Module):
    def __init__(self, tensors, index):
        super().__init__()
        prefix = SPEECH + f'attention{index}'
        self.q = Linear(tensors, prefix + '.W_query.linear', matmul=True)
        self.value = Linear(tensors, prefix + '.W_value.linear', matmul=True)
        self.out = Linear(tensors, prefix + '.out_fc.linear', matmul=True)
        self.key = parameter(tensors[f'text_encoder:/speech_prompted_text_encoder/attention{index}/tanh/Tanh_output_0'].reshape(1, 2, 128, 50))

    def forward(self, x, style):
        length = x.shape[1]
        q = self.q(x).reshape(1, length, 2, 128).permute(0, 2, 1, 3)
        v = self.value(style).reshape(1, 50, 2, 128).permute(0, 2, 1, 3)
        probs = torch.softmax(torch.matmul(q, self.key) * (1.0 / 16.0), dim=-1)
        y = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(1, length, 256)
        return self.out(y)


class TextEncoder(nn.Module):
    def __init__(self, tensors, dilations, capacity=128):
        super().__init__()
        if capacity < 1 or len(dilations) != 6 or any(d < 1 for d in dilations):
            raise ValueError('positive capacity and six positive ConvNeXt dilations required')
        self.embedding = parameter(tensors[TEXT + 'text_embedder.char_embedder.weight'])
        self.blocks = nn.ModuleList([ConvNeXt(tensors, TEXT + f'convnext.convnext.{i}', d)
                                     for i, d in enumerate(dilations)])
        self.attention = nn.ModuleList([AttentionLayer(tensors, i) for i in range(4)])
        self.speech1 = SpeechAttention(tensors, 1)
        self.speech2 = SpeechAttention(tensors, 2)
        self.norm = Norm(tensors, SPEECH + 'norm.norm')
        positions = torch.arange(capacity, dtype=torch.float32)
        self.register_buffer('positions', positions)
        delta = positions[None, :] - positions[:, None]
        self.register_buffer('bands', torch.stack([(delta == i - 4).float() for i in range(9)]))

    def forward(self, text_ids, text_length, style_ttl):
        length = text_length[0].to(torch.float32)
        valid = (self.positions < length).to(torch.float32)[None, :, None]
        last = (self.positions == length - 1).to(torch.float32)[None, :, None]
        mask = (1.0 - valid.transpose(1, 2)[:, None]) * -10000.0
        x = F.embedding(text_ids.to(torch.int64), self.embedding)
        for block in self.blocks:
            x = block(x, valid, last)
        residual = x
        for layer in self.attention:
            x = layer(x, mask, self.bands)
        shared = x + residual
        x = shared + self.speech1(shared, style_ttl)
        x = shared + self.speech2(x, style_ttl)
        return self.norm(x).transpose(1, 2)


def load_encoder(path, capacity):
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    def strings(name):
        field = reader.fields.get('supertonic.' + name)
        return [str(s) for s in field.contents()] if field else []
    names = resolve_tensor_names(strings('source_names'), strings('tensor_names'),
                                 strings('source_aliases'), strings('source_alias_targets'))
    storage = {t.name: t for t in reader.tensors}
    tensors = {}
    for source, name in names.items():
        if not source.startswith('text_encoder:'):
            continue
        tensor = storage[name]
        if tensor.tensor_type.name not in ('F32', 'F16', 'BF16'):
            if tensor.tensor_type.name.startswith('I'):
                continue
            raise ValueError('export the encoder from an F32/F16 GGUF, not block-quantized weights')
        tensors[source] = dequantized(tensor)
    field = reader.fields.get('supertonic.text_convnext_dilations')
    dilations = [int(d) for d in field.contents()] if field else [1] * 6
    return TextEncoder(tensors, dilations, capacity).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', required=True)
    p.add_argument('--capacity', type=int, default=128)
    p.add_argument('--out', type=Path)
    p.add_argument('--compile-dir', type=Path)
    p.add_argument('--compute-units', default='CPU_AND_NE', choices=['ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'],
                   help='export-time placement report; runtime placement is controlled separately')
    p.add_argument('--parity-dir', type=Path, help='dump-supertonic-reference.py output directory')
    p.add_argument('--no-export', action='store_true')
    args = p.parse_args()
    model = load_encoder(args.gguf, args.capacity)
    if args.parity_dir:
        ids = np.load(args.parity_dir / 'text_ids.npy').astype(np.int32).reshape(1, -1)
        length = ids.shape[1]
        if length < 1 or length > args.capacity:
            raise ValueError('reference token length exceeds capacity')
        padded = np.pad(ids, ((0, 0), (0, args.capacity - length)))
        style = np.load(args.parity_dir / 'style_ttl.npy').astype(np.float32)
        with torch.no_grad():
            got = model(torch.from_numpy(padded), torch.tensor([length], dtype=torch.int32), torch.from_numpy(style)).numpy()[..., :length]
        want = np.load(args.parity_dir / 'text_emb.npy')
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)
        print('[parity] PyTorch encoder matches ONNX reference')
    if args.no_export:
        return
    import coremltools as ct
    spec = importlib.util.spec_from_file_location('vocoder_export', Path(__file__).with_name('export-supertonic-coreml.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    inputs = (torch.zeros(1, args.capacity, dtype=torch.int32),
              torch.tensor([min(16, args.capacity)], dtype=torch.int32), torch.zeros(1, 50, 256))
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs)
    compute_units = getattr(ct.ComputeUnit, args.compute_units)
    exported = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name='text_ids', shape=inputs[0].shape, dtype=np.int32),
                ct.TensorType(name='text_length', shape=(1,), dtype=np.int32),
                ct.TensorType(name='style_ttl', shape=(1, 50, 256), dtype=np.float32)],
        outputs=[ct.TensorType(name='text_emb', dtype=np.float32)])
    out = args.out or Path(args.gguf).with_name(sidecar_stem(args.gguf).removesuffix('-vocoder') + '-encoder.mlpackage')
    exported.save(str(out))
    print(f'[export] saved {out} (capacity={args.capacity})')
    if args.compile_dir:
        compiled = helper.compile_model(out, args.compile_dir)
        print(f'[export] compiled {compiled}')
        helper.report_placement(compiled, compute_units)


if __name__ == '__main__':
    main()
