#!/usr/bin/env python3
"""
Convert the CosyVoice3 supervised speech tokenizer (speech_tokenizer_v3.onnx)
to a standalone GGUF for the native voice-cloning front-end.

speech_tokenizer_v3 shares the S3TokenizerV2 architecture the ggml port in
src/s3tokenizer.cpp already implements (2-conv stem, FSMN residual attention
blocks with NEOX RoPE, FSQ 3^8 codebook); v3 differs only in depth (12 blocks
vs 6). The GGUF therefore reuses the existing `s3tokv2.*` KV / `s3tokv2/*`
tensor schema — that prefix names the storage schema, not the checkpoint —
with `s3tokv2.n_audio_layer = 12` and `s3tokv2.tokenizer_version = 3` marking
the v3 weights.

Weights are extracted through the S3Tokenizer project's own ONNX loader
(`s3tokenizer.utils.onnx2torch_v3`, https://github.com/xingchensong/S3Tokenizer)
so the ONNX-initializer -> module-name mapping stays upstream's problem; the
resulting state_dict keys map 1:1 onto the GGUF tensor names
(`encoder.blocks.N.attn.query.weight` -> `s3tokv2/encoder/blocks/N/attn/query/weight`).

Two auxiliary filterbanks are baked in:
 - `s3tokv2/mel_fb` (128, 201): the 16 kHz/n_fft-400 Whisper mel filterbank the
   upstream frontend feeds the tokenizer through (whisper.log_mel_spectrogram).
 - `cosyvoice3/mel_fb_24k_80` (80, 961): the 24 kHz flow-mel filterbank used
   for the cloning prompt_feat, so the cloning front-end is self-contained in
   this one file.  CosyVoice3's feat_extractor (matcha mel_spectrogram per
   cosyvoice3.yaml) uses fmin 0, fmax null -> sr/2 = 12000; NOT the fmax-8000
   bank Chatterbox's s3gen uses.  Verified against the frontend's prompt_feat
   dump: fmax=None reproduces it to ~1e-6, fmax=8000 is off by ~5.8.

--outtype:
  f32   parity anchor (~970 MB)
  f16   default distribution (~490 MB); 2-D block weights stored F16
  q8_0  size-sensitive targets (~260 MB); 2-D block weights quantized, all
        stems/norms/biases/filterbanks/FSQ kept F32

Requires: torch, gguf, whisper (openai-whisper), librosa, and either the
`s3tokenizer` package or --s3tokenizer-repo pointing at its checkout.
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert speech_tokenizer_v3.onnx to GGUF.")
    ap.add_argument("--onnx", type=Path, required=True,
                    help="Path to speech_tokenizer_v3.onnx (Fun-CosyVoice3-0.5B).")
    ap.add_argument("--out", type=Path, required=True, help="Output GGUF path.")
    ap.add_argument("--outtype", choices=("f32", "f16", "q8_0"), default="f16")
    ap.add_argument("--s3tokenizer-repo", type=Path, default=None,
                    help="Checkout of github.com/xingchensong/S3Tokenizer "
                         "(omit if the s3tokenizer package is installed).")
    return ap.parse_args()


N_LAYER = 12
N_STATE = 1280

HPARAMS_U32 = {
    "s3tokv2.n_mels":            128,
    "s3tokv2.n_audio_state":     N_STATE,
    "s3tokv2.n_audio_head":      20,
    "s3tokv2.n_audio_layer":     N_LAYER,
    "s3tokv2.head_dim":          64,
    "s3tokv2.mlp_ratio":         4,
    "s3tokv2.fsmn_kernel":       31,
    "s3tokv2.fsq_levels":        3,
    "s3tokv2.fsq_dim":           8,
    "s3tokv2.codebook_size":     3 ** 8,
    "s3tokv2.conv_stride":       2,
    "s3tokv2.n_fft":             400,
    "s3tokv2.hop":               160,
    "s3tokv2.sample_rate":       16000,
    "s3tokv2.rope_max_pos":      2048,
    "s3tokv2.tokenizer_version": 3,
}


def is_block_matmul_weight(name: str, shape: tuple) -> bool:
    """The large 2-D projection weights inside the attention blocks are the
    only tensors worth compressing; everything else (stems, FSMN depthwise
    convs, norms, biases, filterbanks, the 8-d FSQ projection) stays F32 so
    the numerically sensitive paths keep full precision."""
    if len(shape) != 2:
        return False
    if not name.startswith("s3tokv2/encoder/blocks/"):
        return False
    return min(shape) >= N_STATE


def add_tensor(writer: gguf.GGUFWriter, name: str, arr: np.ndarray,
               outtype: str, stats: dict) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    if outtype == "f32" or not is_block_matmul_weight(name, arr.shape):
        writer.add_tensor(name, arr)
        return
    if outtype == "f16":
        writer.add_tensor(name, arr.astype(np.float16))
        stats["n_f16"] = stats.get("n_f16", 0) + 1
        return
    qdata = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape,
                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    stats["n_q8"] = stats.get("n_q8", 0) + 1


def main() -> None:
    args = parse_args()
    if args.s3tokenizer_repo is not None:
        sys.path.insert(0, str(args.s3tokenizer_repo))
    from s3tokenizer.utils import onnx2torch_v3

    print(f"loading {args.onnx} via onnx2torch_v3 ...")
    state = onnx2torch_v3(str(args.onnx), None, False)

    expected = 4 + 16 * N_LAYER + 2
    if len(state) != expected:
        print(f"error: expected {expected} tensors, got {len(state)}",
              file=sys.stderr)
        sys.exit(1)

    import whisper
    mel_fb = whisper.audio.mel_filters(torch.device("cpu"), 128).numpy()
    if mel_fb.shape != (128, 201):
        print(f"error: whisper mel filterbank shape {mel_fb.shape}", file=sys.stderr)
        sys.exit(1)

    import librosa
    mel_fb_24k = librosa.filters.mel(
        sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=None).astype(np.float32)

    writer = gguf.GGUFWriter(str(args.out), "s3tokenizer")
    writer.add_name("CosyVoice3 speech tokenizer v3")
    writer.add_description(
        "Supervised speech tokenizer (speech_tokenizer_v3) for CosyVoice3 "
        "zero-shot voice cloning; s3tokv2 storage schema, 12 layers.")
    for k, v in HPARAMS_U32.items():
        writer.add_uint32(k, v)
    writer.add_float32("s3tokv2.rope_theta", 10000.0)

    stats: dict = {}
    for k in sorted(state.keys()):
        name = "s3tokv2/" + k.replace(".", "/")
        add_tensor(writer, name, state[k].to(torch.float32).numpy(), args.outtype, stats)
    add_tensor(writer, "s3tokv2/mel_fb", mel_fb, args.outtype, stats)
    add_tensor(writer, "cosyvoice3/mel_fb_24k_80", mel_fb_24k, args.outtype, stats)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.0f} MB, outtype={args.outtype}, "
          f"{len(state) + 2} tensors, {stats})")


if __name__ == "__main__":
    main()
