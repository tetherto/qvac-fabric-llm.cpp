#!/usr/bin/env python3
"""Check compiled FP32 native-padding programs against the F32 model reference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import wave

import coremltools as ct
import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_prediction(model, mel, valid, route, weights, meta, reference, gpu_capacity=0):
    route_meta = dict(meta)
    if route == 'gate':
        route_meta['parakeet.encoder.n_layers'] = int(meta['parakeet.eu.gate_layer']) + 1
    with torch.inference_mode():
        expected = reference.encoder_forward(mel, weights, route_meta,
            mel_valid_len=valid, eu_use_adapters=route == 'other')[0]
    if gpu_capacity:
        capacity = gpu_capacity
        frames = expected.shape[0]
        slab = np.zeros((1, 128, 1, capacity), np.float32)
        slab[0, :, 0, :mel.shape[1]] = mel.numpy()
        mask = np.zeros((1, 1, 1, capacity), np.float32)
        mask[..., :valid] = 1
        encoder_capacity = (capacity + 7) // 8
        padding = np.zeros((1, 1, 1, encoder_capacity), np.float32)
        padding[..., :frames] = 1
        bias = np.full((1, encoder_capacity, 1, 1), -1e4, np.float32)
        bias[:, :frames] = 0
        values = {'mel': slab, 'mel_mask': mask, 'pad_mask': padding, 'key_bias': bias}
        prediction = model.predict(values)['encoder_out'][0, :, 0, :frames].T
    else:
        mask = (np.arange(mel.shape[1]) < valid).astype(np.float32)
        prediction = model.predict({'mel': mel.numpy(), 'mel_valid_mask': mask})['encoder_out']
    actual = torch.from_numpy(np.ascontiguousarray(prediction))
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    delta = (actual - expected).double()
    relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.double()))
    assert relative < 1e-3, f'{route} compiled precision failed: {relative}'
    return dict(route=route, input_frames=mel.shape[1], valid_frames=valid,
                encoder_frames=actual.shape[0], placement='coreml-cpu-gpu' if gpu_capacity else 'coreml-cpu',
                relative_l2=relative, max_absolute=float(delta.abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    reference = load_module('precise_reference', scripts / 'ref-encoder-from-gguf.py')
    weights, meta = reference.load_gguf(args.assets / 'models/transcribepsy-250m-eu.f16.gguf')
    torch.set_num_threads(4)
    rows = []
    for route, wav in [('en', 'en-60s.wav'), ('other', 'fr-long.wav'), ('gate', 'en-15s.wav')]:
        dump = args.output.with_suffix('.mel')
        command = [str(args.binary), '--model', str(args.assets / 'models/transcribepsy-250m-eu.f16.gguf'),
                   '--wav', str(args.assets / 'inputs' / wav), '--language', 'en', '--backend', 'cpu',
                   '--window-sec', '0', '--dump-mel', str(dump)]
        subprocess.run(command, env={**os.environ, 'PARAKEET_COREML_DISABLE': '1'},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        mel = torch.from_numpy(np.fromfile(dump, np.float32).reshape(128, -1).copy())
        with wave.open(str(args.assets / 'inputs' / wav)) as audio:
            valid = (audio.getnframes() + 159) // 160
        suffix = '' if route == 'en' else '-' + route
        stem = 'transcribepsy-250m-eu-encoder' + suffix
        path = args.assets / 'models' / (stem + '-flexible.mlmodelc')
        model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
        cases = [(mel, valid)]
        for frames in range(1, 9):
            cases.append((torch.zeros(128, frames), frames))
        for frames in (399, 400, 401, 402, 407, 408, 409):
            value = mel[:, :frames].clone()
            value[:, -1] = 0
            cases.append((value, frames - 1))
        for value, length in cases:
            row = record_prediction(model, value, length, route, weights, meta, reference, False)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model
        if route != 'gate':
            for capacity in (1500, 6008):
                path = args.assets / 'models' / (stem + f'-fp32-{capacity}.mlmodelc')
                model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
                selected = cases if capacity == 6008 else cases[1:]
                for value, length in selected:
                    row = record_prediction(model, value, length, route, weights, meta,
                                            reference, capacity)
                    row['capacity'] = capacity
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                del model
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
