#!/usr/bin/env python3
"""Intrusive audio metrics. Scores are informational; failures remain explicit in JSON."""
import argparse
import json
import math
from pathlib import Path
import sys
import wave


def read_wav(path):
    """Read uncompressed mono PCM without optional numerical dependencies."""
    with wave.open(str(path), 'rb') as wav:
        if wav.getnchannels() != 1:
            raise ValueError('expected mono WAV')
        width, rate, frames = wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if width not in (1, 2, 3, 4) or rate <= 0:
            raise ValueError('unsupported PCM width or sample rate')
        raw = wav.readframes(frames)
    if len(raw) != frames * width:
        raise ValueError('truncated WAV data')
    if not frames:
        raise ValueError('empty WAV')
    scale = float(1 << (8 * width - 1))
    samples = [(int.from_bytes(raw[i:i + width], 'little', signed=width != 1)
                - (128 if width == 1 else 0)) / scale
               for i in range(0, len(raw), width)]
    return samples, rate


def validate(samples):
    if not samples or not all(math.isfinite(x) for x in samples):
        raise ValueError('audio must contain nonempty finite samples')


def sisdr(reference, hypothesis):
    """Zero-mean SI-SDR in dB; mathematical infinities are explicitly unavailable."""
    validate(reference)
    validate(hypothesis)
    if len(reference) != len(hypothesis):
        raise ValueError('sample counts differ; no automatic truncation or alignment')
    ref_mean = math.fsum(reference) / len(reference)
    hyp_mean = math.fsum(hypothesis) / len(hypothesis)
    ref = [x - ref_mean for x in reference]
    hyp = [x - hyp_mean for x in hypothesis]
    ref_energy = math.fsum(x * x for x in ref)
    hyp_energy = math.fsum(x * x for x in hyp)
    if ref_energy == 0:
        raise ValueError('reference is silent or constant after mean removal')
    if hyp_energy == 0:
        raise ValueError('hypothesis is silent or constant after mean removal')
    alpha = math.fsum(x * y for x, y in zip(ref, hyp)) / ref_energy
    target_energy = alpha * alpha * ref_energy
    residual_energy = math.fsum((y - alpha * x) ** 2 for x, y in zip(ref, hyp))
    if target_energy == 0:
        raise ValueError('SI-SDR is negative infinity (orthogonal signals)')
    if residual_energy == 0:
        raise ValueError('SI-SDR is positive infinity (exact scaled reconstruction)')
    return 10 * (math.log10(target_energy) - math.log10(residual_energy))


def resample(samples, rate, target_rate):
    if rate == target_rate:
        return samples
    from scipy.signal import resample_poly
    factor = math.gcd(rate, target_rate)
    return resample_poly(samples, target_rate // factor, rate // factor).tolist()


def compute_stoi(reference, hypothesis, rate):
    import warnings
    import numpy as np
    from pystoi import stoi
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        return float(stoi(np.asarray(reference), np.asarray(hypothesis), rate, extended=False))


def score(reference_path, hypothesis_path, metrics):
    result = {'reference': str(reference_path), 'hypothesis': str(hypothesis_path),
              'preprocessing': {'zero_mean_sisdr': True, 'alignment': 'none'}, 'metrics': {}}
    try:
        reference, ref_rate = read_wav(reference_path)
        hypothesis, hyp_rate = read_wav(hypothesis_path)
        result['reference_frames'] = len(reference)
        result['hypothesis_frames'] = len(hypothesis)
        result['reference_sample_rate'] = ref_rate
        result['hypothesis_sample_rate'] = hyp_rate
        rate = ref_rate if ref_rate == hyp_rate else 16000
        reference = resample(reference, ref_rate, rate)
        hypothesis = resample(hypothesis, hyp_rate, rate)
        result['scoring_sample_rate'] = rate
        if len(reference) != len(hypothesis):
            raise ValueError('sample counts differ after resampling; no automatic truncation or alignment')
        if not any(reference):
            raise ValueError('reference is silent')
        if not any(hypothesis):
            raise ValueError('hypothesis is silent (no audio output)')
        input_error = None
    except Exception as exc:
        input_error = exc
    for name in metrics:
        metric = {'value': None, 'unit': 'dB' if name == 'sisdr' else 'score'}
        result['metrics'][name] = metric
        try:
            if input_error is not None:
                raise input_error
            value = sisdr(reference, hypothesis) if name == 'sisdr' else compute_stoi(reference, hypothesis, rate)
            if not math.isfinite(value):
                raise ValueError('metric returned a nonfinite value')
            metric.update(value=value, status='ok')
        except (ImportError, ModuleNotFoundError) as exc:
            metric.update(status='unavailable', reason=f'optional dependency unavailable: {exc}')
        except Exception as exc:
            metric.update(status='error', reason=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--hypothesis-file', required=True)
    parser.add_argument('--metrics', default='sisdr,stoi')
    parser.add_argument('--json-out')
    args = parser.parse_args()
    metrics = list(dict.fromkeys(args.metrics.split(',')))
    if not metrics or any(x not in ('sisdr', 'stoi') for x in metrics):
        parser.error('--metrics must be a comma-separated subset of sisdr,stoi')
    result = score(args.reference, args.hypothesis_file, metrics)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.json_out:
        Path(args.json_out).write_text(encoded, encoding='utf-8')
    sys.stdout.write(encoded)


if __name__ == '__main__':
    main()
