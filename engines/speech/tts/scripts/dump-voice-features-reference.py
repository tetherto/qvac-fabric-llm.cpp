#!/usr/bin/env python3
"""Generate a matched WAV, filterbank GGUF and prompt_feat.npy without a model.

The saved float32 WAV is the exact input to the independent torch STFT
reference. Resampling happens before saving it, so the C++ mel check does
not compare different resampler implementations.
"""

import argparse
from pathlib import Path

import gguf
import librosa
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn.functional as functional

SAMPLE_RATE = 24000
FFT_SIZE = 1920
HOP_SIZE = 480
MEL_CHANNELS = 80
MAX_FREQUENCY = 8000
LOG_FLOOR = 1e-5
PCM16_SCALE = 32768.0


def load_waveform(path):
    sample_rate, samples = wavfile.read(path)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / PCM16_SCALE
    elif samples.dtype != np.float32:
        raise ValueError("Input WAV must contain PCM16 or float32 samples")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if samples.size <= FFT_SIZE or not np.isfinite(samples).all():
        raise ValueError("Input must contain more than one FFT window of finite audio")
    return np.ascontiguousarray(samples, dtype=np.float32)


def compute_features(samples, filterbank):
    padding = (FFT_SIZE - HOP_SIZE) // 2
    signal = torch.from_numpy(samples).reshape(1, 1, -1)
    padded = functional.pad(signal, (padding, padding), mode="reflect").squeeze(1)
    spectrum = torch.stft(
        padded, FFT_SIZE, hop_length=HOP_SIZE, win_length=FFT_SIZE,
        window=torch.hann_window(FFT_SIZE), center=False, return_complex=True,
    )
    # Chatterbox's established C++ path uses magnitude without an epsilon.
    mel = torch.from_numpy(filterbank) @ spectrum.abs()
    features = torch.log(mel.clamp_min(LOG_FLOOR)).squeeze(0).transpose(0, 1)
    return np.ascontiguousarray(features.numpy(), dtype=np.float32)


def write_filterbank(path, filterbank):
    writer = gguf.GGUFWriter(str(path), "chatterbox-s3gen")
    writer.add_tensor("s3gen/mel_fb/24k_80", filterbank)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def dump_reference(wav_path, output):
    samples = load_waveform(wav_path)
    filterbank = np.ascontiguousarray(librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=MEL_CHANNELS,
        fmin=0, fmax=MAX_FREQUENCY,
    ), dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    wavfile.write(output / "reference.wav", SAMPLE_RATE, samples)
    write_filterbank(output / "filterbank.gguf", filterbank)
    np.save(output / "prompt_feat.npy", compute_features(samples, filterbank))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dump_reference(args.wav, args.out)


if __name__ == "__main__":
    main()
