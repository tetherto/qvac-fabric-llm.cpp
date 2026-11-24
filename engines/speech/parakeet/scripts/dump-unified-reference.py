#!/usr/bin/env python3
"""Dump Unified RNN-T reference transcripts and tensors from NVIDIA NeMo.

Three references are written for one WAV and one (left, chunk, right) context:
  offline/        full-context encoder, the model's offline mode
  chunked/        full audio through the encoder with the chunked-with-right-context mask
  buffered/       NeMo's buffered streaming: per step the encoder sees [left + chunk + right]
                  audio, only the chunk frames are kept, RNN-T decoding is stateful across steps

Per-step tensors (mel, encoder output, tokens) are written for the buffered path so the C++
cache-aware session can be checked step by step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
HOP = 160
SUBSAMPLING = 8
SAMPLES_PER_FRAME = HOP * SUBSAMPLING


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--nemo-model", type=Path, default=Path("models/parakeet-unified-en-0.6b.nemo"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/unified-ref"))
    parser.add_argument("--left", type=int, default=70, help="left context in encoder frames (70 = 5.6 s)")
    parser.add_argument("--chunk", type=int, default=7, help="chunk size in encoder frames (7 = 560 ms)")
    parser.add_argument("--right", type=int, default=7, help="right context in encoder frames")
    parser.add_argument("--capture-steps", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_model(path, device):
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    model = EncDecRNNTBPEModel.restore_from(str(path), map_location=device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def load_audio(path):
    audio, rate = sf.read(str(path), dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio)


def save_tensor(path, tensor):
    np.save(str(path), tensor.detach().cpu().numpy())


def hypothesis_text(hypothesis):
    return hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)


def hypothesis_tokens(hypothesis):
    sequence = getattr(hypothesis, "y_sequence", None)
    if sequence is None:
        return []
    if isinstance(sequence, torch.Tensor):
        return sequence.detach().cpu().tolist()
    return list(sequence)


def encode(model, audio):
    with torch.inference_mode():
        features, feature_lengths = model.preprocessor(
            input_signal=audio.unsqueeze(0), length=torch.tensor([audio.shape[0]])
        )
        encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
    return features, encoded, encoded_lengths


def decode(model, encoded, encoded_lengths, partial=None):
    with torch.inference_mode():
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_lengths,
            return_hypotheses=True,
            partial_hypotheses=partial,
        )
    return hypotheses


def set_context(model, context):
    model.encoder.set_default_att_context_size(list(context))


def dump_full_audio(model, audio, out_dir):
    features, encoded, encoded_lengths = encode(model, audio)
    hypotheses = decode(model, encoded, encoded_lengths)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(out_dir / "mel.npy", features[0])
    save_tensor(out_dir / "encoder.npy", encoded[0])
    text = hypothesis_text(hypotheses[0])
    (out_dir / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    json.dump(
        {"tokens": hypothesis_tokens(hypotheses[0]), "encoder_frames": int(encoded_lengths[0])},
        open(out_dir / "meta.json", "w"),
        indent=2,
    )
    return text


def step_windows(n_samples, left, chunk, right):
    chunk_samples = chunk * SAMPLES_PER_FRAME
    left_samples = left * SAMPLES_PER_FRAME
    right_samples = right * SAMPLES_PER_FRAME
    starts = range(0, n_samples, chunk_samples)
    return [window_for(start, n_samples, chunk_samples, left_samples, right_samples) for start in starts]


def window_for(start, n_samples, chunk_samples, left_samples, right_samples):
    begin = max(0, start - left_samples)
    end = min(n_samples, start + chunk_samples + right_samples)
    left_frames = (start - begin) // SAMPLES_PER_FRAME
    chunk_end = min(n_samples, start + chunk_samples)
    chunk_frames = -(-(chunk_end - start) // SAMPLES_PER_FRAME)
    return {"begin": begin, "end": end, "left_frames": left_frames, "chunk_frames": chunk_frames}


def dump_buffered(model, audio, out_dir, context, capture_steps):
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = step_windows(audio.shape[0], *context)
    if not windows:
        raise ValueError("audio is shorter than one encoder frame; nothing to stream")
    partial = None
    steps = []
    for index, window in enumerate(windows):
        partial, record = run_buffered_step(model, audio, window, partial, out_dir, index, index < capture_steps)
        steps.append(record)
    text = hypothesis_text(partial[0])
    (out_dir / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    json.dump({"context": list(context), "steps": steps}, open(out_dir / "steps.json", "w"), indent=2)
    return text


def run_buffered_step(model, audio, window, partial, out_dir, index, capture):
    segment = audio[window["begin"]:window["end"]]
    features, encoded, encoded_lengths = encode(model, segment)
    first = window["left_frames"]
    last = min(int(encoded_lengths[0]), first + window["chunk_frames"])
    chunk_encoded = encoded[:, :, first:last].contiguous()
    chunk_lengths = torch.tensor([last - first])
    hypotheses = decode(model, chunk_encoded, chunk_lengths, partial)
    record = {
        "index": index,
        "begin_sample": window["begin"],
        "end_sample": window["end"],
        "left_frames": first,
        "chunk_frames": last - first,
        "encoder_frames": int(encoded_lengths[0]),
        "text_so_far": hypothesis_text(hypotheses[0]),
    }
    if capture:
        step_dir = out_dir / f"step-{index:03d}"
        step_dir.mkdir(exist_ok=True)
        save_tensor(step_dir / "mel.npy", features[0])
        save_tensor(step_dir / "encoder_window.npy", encoded[0])
        save_tensor(step_dir / "encoder_chunk.npy", chunk_encoded[0])
        json.dump({"tokens": hypothesis_tokens(hypotheses[0])}, open(step_dir / "tokens.json", "w"))
    return hypotheses, record


def main():
    args = parse_args()
    model = load_model(args.nemo_model, args.device)
    audio = load_audio(args.wav)
    context = (args.left, args.chunk, args.right)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    offline = dump_full_audio(model, audio, out / "offline")
    set_context(model, context)
    chunked = dump_full_audio(model, audio, out / "chunked")
    buffered = dump_buffered(model, audio, out / "buffered", context, args.capture_steps)
    json.dump(
        {"wav": str(args.wav), "context": list(context), "offline": offline, "chunked": chunked, "buffered": buffered},
        open(out / "summary.json", "w"),
        indent=2,
    )
    print("offline :", offline)
    print("chunked :", chunked)
    print("buffered:", buffered)


if __name__ == "__main__":
    main()
