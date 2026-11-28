#!/usr/bin/env python3
"""Compare actual compiled predictions; requires Torch, coremltools, and full-EU fixtures."""
import importlib.util, json, os, subprocess, wave
from pathlib import Path
import numpy as np, torch, coremltools as ct
import argparse
def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Compare compiled FP16/6-bit encoders against the F32 reference')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
def compare_models(configuration):
    ROOT = configuration.assets.resolve()
    S = Path(__file__).resolve().parents[2] / 'scripts'
    BIN = str(configuration.binary.resolve())
    ref = module('ref', S / 'ref-encoder-from-gguf.py')
    ane = module('ane', S / 'export-encoder-coreml-ane.py')
    torch.set_num_threads(4)
    (W, meta) = ref.load_gguf(ROOT / 'models/transcribepsy-250m-eu.f16.gguf')
    rows = []

    def gate(x):
        import torch.nn.functional as F

        def linear(x, key):
            return F.linear(x, W['eu.gate.' + key + '.weight'], W['eu.gate.' + key + '.bias'])
        a = linear(torch.tanh(linear(x, 'pool.proj.0')), 'pool.proj.2').softmax(0)
        mean = (a * x).sum(0)
        std = ((a * x * x).sum(0) - mean * mean).clamp(min=1e-08).sqrt()
        p = F.layer_norm(torch.cat((mean, std)), (1280,), W['eu.gate.norm.weight'], W['eu.gate.norm.bias'])
        return float(linear(F.gelu(linear(p, 'mlp.0')), 'mlp.3').sigmoid())
    for wav in ['en-15s.wav', 'en-partial.wav', 'fr-multipiece-16k.wav']:
        melpath = ROOT / (wav + '.mel')
        command = [BIN, '--model', str(ROOT / 'models/transcribepsy-250m-eu.f16.gguf'), '--wav', str(ROOT / 'inputs' / wav), '--language', 'en', '--backend', 'cpu', '--dump-mel', str(melpath)]
        subprocess.run(command, env={**os.environ, 'PARAKEET_COREML_DISABLE': '1', 'PARAKEET_ANE_BATCH_DECODE': '0'}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        mel = torch.from_numpy(np.fromfile(melpath, np.float32).reshape(128, -1))
        valid = min(mel.shape[1], (wave.open(str(ROOT / 'inputs' / wav)).getnframes() + 159) // 160)
        for route in ['en', 'other', 'gate']:
            want = ane.reference_frames(ref, W, meta, mel, valid, route)
            capacity = 2001 if route == 'gate' else 1500
            config = type('Config', (), dict(n_mels=128, window_mel_frames=capacity, enc_frames=(capacity + 7) // 8))()
            (values, T) = ane.window_inputs(config, mel[:, :capacity], valid)
            inputs = {name: v.numpy() for (name, v) in zip(['mel', 'mel_mask', 'pad_mask', 'key_bias'], values)}
            for folder in ['models', 'models6']:
                name = 'transcribepsy-250m-eu-encoder' + ('' if route == 'en' else '-' + route) + '.mlmodelc'
                model = ct.models.CompiledMLModel(str(ROOT / folder / name), compute_units=ct.ComputeUnit.CPU_AND_NE)
                got = torch.from_numpy(model.predict(inputs)['encoder_out'][0, :, 0, :T].T.copy())
                assert torch.isfinite(got).all()
                record = dict(wav=wav, route=route, precision='FP16' if folder == 'models' else '6-bit', frames=T, relative_l2=ane.relative_error(got, want), max_absolute=float((got - want).abs().max()))
                if route == 'gate':
                    record.update(reference_gate=gate(want), compiled_gate=gate(got))
                    record['gate_absolute_error'] = abs(record['reference_gate'] - record['compiled_gate'])
                rows.append(record)
                print(json.dumps(record), flush=True)
                del model
    return rows


def main(argv=None):
    configuration = parse_arguments(argv)
    records = compare_models(configuration)
    configuration.output.write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
