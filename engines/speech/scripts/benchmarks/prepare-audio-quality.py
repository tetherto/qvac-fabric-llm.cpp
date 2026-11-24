#!/usr/bin/env python3
"""Prepare deterministic noisy mono PCM16 audio; never changes duration or rate."""
import argparse
import importlib.util
import math
import json
from pathlib import Path
import random
import struct
import wave

_SPEC = importlib.util.spec_from_file_location('audio_quality', Path(__file__).with_name('compute-audio-quality.py'))
_quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_quality)


def noisy_pair(reference, snr_db, seed):
    _quality.validate(reference)
    if not math.isfinite(snr_db):
        raise ValueError('SNR must be finite')
    energy = math.fsum(x*x for x in reference)
    if energy == 0:
        raise ValueError('cannot add noise at a defined SNR to silence')
    rng = random.Random(seed)
    noise = [rng.gauss(0, 1) for _ in reference]
    gain = math.sqrt(energy / math.fsum(x*x for x in noise)) * 10 ** (-snr_db / 20)
    noisy = [x + gain*n for x, n in zip(reference, noise)]
    # Scale both members together to avoid clipping and preserve the requested SNR.
    peak = max(max(abs(x) for x in reference), max(abs(x) for x in noisy))
    scale = min(1.0, (32767 / 32768) / peak)
    return [x*scale for x in reference], [x*scale for x in noisy], scale


def write_wav(path, samples, rate):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b''.join(struct.pack('<h', max(-32768, min(32767, round(x*32768)))) for x in samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-output', help='write the paired clean reference if common attenuation is needed')
    parser.add_argument('--snr-db', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        source, rate = _quality.read_wav(args.reference)
        clean, noisy, scale = noisy_pair(source, args.snr_db, args.seed)
        if scale < 1 and not args.reference_output:
            raise ValueError('noise needs attenuation to avoid clipping; supply --reference-output for a matching clean reference')
        write_wav(args.output, noisy, rate)
        if args.reference_output:
            write_wav(args.reference_output, clean, rate)
        print(json.dumps({'reference': args.reference, 'output': args.output,
                          'reference_output': args.reference_output, 'sample_rate': rate,
                          'frames': len(source), 'snr_db': args.snr_db,
                          'seed': args.seed, 'scale': scale}, allow_nan=False))
    except (ValueError, OSError, wave.Error, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
