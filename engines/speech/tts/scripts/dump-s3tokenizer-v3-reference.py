#!/usr/bin/env python3
"""Dump speech_tokenizer_v3 reference tensors for test-s3tokenizer-v3.

Runs the CosyVoice3 speech tokenizer on a reference wav twice — through
onnxruntime (the exact engine the upstream frontend uses) and through the
S3Tokenizer project's torch reimplementation initialized from the same ONNX —
verifies the two token streams agree, and writes the fixtures the C++ parity
harness consumes:

    wav_16k.npy     float32 (1, N)      16 kHz mono waveform (frontend chain)
    log_mel.npy     float32 (1, 128, T) whisper.log_mel_spectrogram(n_mels=128)
    tokens.npy      int32  (T_tok,)     ONNX token stream (ground truth)

With --dump-layers it additionally hooks the torch encoder and dumps
conv1/conv2 activations, every residual block output, and the pre-FSQ hidden
state — the debugging ladder for walking the C++ graph layer by layer.

Run where torch + whisper + onnxruntime are available:
    python3 dump-s3tokenizer-v3-reference.py \
        --onnx models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx \
        --wav CosyVoice/asset/zero_shot_prompt.wav \
        --out-dir artifacts/s3tok-v3-ref \
        --s3tokenizer-repo /path/to/S3Tokenizer
"""

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dump speech_tokenizer_v3 reference tensors.")
    ap.add_argument("--onnx", required=True, help="speech_tokenizer_v3.onnx path.")
    ap.add_argument("--wav", required=True, help="Reference wav (any sample rate).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--s3tokenizer-repo", default=None,
                    help="Checkout of github.com/xingchensong/S3Tokenizer "
                         "(omit if the s3tokenizer package is installed).")
    ap.add_argument("--dump-layers", action="store_true",
                    help="Also dump per-layer torch encoder activations.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.s3tokenizer_repo is not None:
        sys.path.insert(0, args.s3tokenizer_repo)

    import torch
    import torchaudio
    import whisper
    import onnxruntime
    from s3tokenizer.utils import onnx2torch_v3
    from s3tokenizer.model_v3 import S3TokenizerV3

    os.makedirs(args.out_dir, exist_ok=True)

    # Mirror the upstream frontend chain: load_wav(wav, 16000) -> mono ->
    # resample; no loudness normalization, no trimming.
    speech, sr = torchaudio.load(args.wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    assert speech.shape[1] / 16000 <= 30, "tokenizer input must be <= 30 s"

    mel = whisper.log_mel_spectrogram(speech, n_mels=128)
    mel_len = np.array([mel.shape[2]], dtype=np.int32)
    print(f"wav: {speech.shape[1]} samples @16k, mel T={mel.shape[2]}")

    sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    tok_onnx = sess.run(None, {ins[0].name: mel.numpy(),
                               ins[1].name: mel_len})[0].flatten().astype(np.int64)

    model = S3TokenizerV3("speech_tokenizer_v3")
    missing, unexpected = model.load_state_dict(
        onnx2torch_v3(args.onnx, None, False), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    with torch.no_grad():
        tok_t, tok_lens = model.quantize(mel, torch.tensor(mel_len))
    tok_torch = tok_t[0, : tok_lens[0].item()].numpy().astype(np.int64)

    n = min(len(tok_onnx), len(tok_torch))
    n_match = int((tok_onnx[:n] == tok_torch[:n]).sum())
    print(f"onnx tokens n={len(tok_onnx)}, torch n={len(tok_torch)}, "
          f"match {n_match}/{n}")
    if len(tok_onnx) != len(tok_torch) or n_match != n:
        print("error: onnx and torch token streams disagree; fixtures would be "
              "ambiguous", file=sys.stderr)
        sys.exit(1)

    np.save(os.path.join(args.out_dir, "wav_16k.npy"),
            speech.numpy().astype(np.float32))
    np.save(os.path.join(args.out_dir, "log_mel.npy"),
            mel.numpy().astype(np.float32))
    np.save(os.path.join(args.out_dir, "tokens.npy"),
            tok_onnx.astype(np.int32))

    if args.dump_layers:
        acts = {}

        def hook(name):
            def f(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                acts[name] = t.detach().numpy().astype(np.float32)
            return f

        enc = model.encoder
        enc.conv1.register_forward_hook(hook("conv1"))
        enc.conv2.register_forward_hook(hook("conv2"))
        for i, blk in enumerate(enc.blocks):
            blk.register_forward_hook(hook(f"block{i}"))
        with torch.no_grad():
            hidden, _ = enc(mel, torch.tensor(mel_len))
        np.save(os.path.join(args.out_dir, "pre_fsq_hidden.npy"),
                hidden.numpy().astype(np.float32))
        for k, v in acts.items():
            np.save(os.path.join(args.out_dir, f"{k}.npy"), v)
        print(f"dumped {len(acts) + 1} layer activations")

    print(f"wrote fixtures to {args.out_dir}")


if __name__ == "__main__":
    main()
