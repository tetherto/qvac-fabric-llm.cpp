#!/usr/bin/env python3
"""
Convert a 3D-Speaker CAM++ speaker-verification checkpoint to a standalone
GGUF for the CosyVoice3 voice-cloning front-end.

CosyVoice3 ships the stock 3D-Speaker `campplus_cn_common` model (the same
checkpoint Chatterbox embeds in its S3Gen GGUF as `speaker_encoder.*`); its
campplus.onnx export BN-folds and anonymizes the head convolutions, so this
converter takes the original torch checkpoint instead:

    https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin
    (mirror of modelscope iic/speech_campplus_sv_zh-cn_16k-common)

The emitted tensor names (`campplus/head/...`, `campplus/xvector/...`),
per-BN fused scale/shift pairs, hyperparameter KVs and the Kaldi mel
filterbank replicate what convert-s3gen-to-gguf.py embeds for Chatterbox, so
src/campplus.cpp's `campplus_load` reads the file unchanged.  Parity of the
resulting embedding against the CosyVoice frontend's campplus.onnx output is
asserted by test-cosyvoice-frontend.

All tensors stay F32 (~28 MB; nothing here is worth compressing).
"""

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch

BN_EPS = 1e-5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert a CAM++ torch checkpoint to GGUF.")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="campplus_cn_common.bin (torch state_dict).")
    ap.add_argument("--out", type=Path, required=True, help="Output GGUF path.")
    return ap.parse_args()


def kaldi_mel_filterbank() -> np.ndarray:
    sr, nfft, n_mels, low, high = 16000, 512, 80, 20.0, 8000.0
    mel_low = 1127.0 * np.log(1.0 + low / 700.0)
    mel_high = 1127.0 * np.log(1.0 + high / 700.0)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)
    bin_freq = np.arange(nfft // 2 + 1, dtype=np.float64) * sr / nfft
    bin_mel = 1127.0 * np.log(1.0 + bin_freq / 700.0)
    fb = np.zeros((n_mels, nfft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        mel_center = mel_low + (m + 1) * mel_delta
        mel_lo, mel_hi = mel_center - mel_delta, mel_center + mel_delta
        for k, mb in enumerate(bin_mel):
            if mb < mel_lo or mb > mel_hi:
                continue
            if mb <= mel_center:
                fb[m, k] = (mb - mel_lo) / (mel_center - mel_lo)
            else:
                fb[m, k] = (mel_hi - mb) / (mel_hi - mel_center)
    return fb


def main() -> None:
    args = parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    # Group BN tensors by module prefix; a prefix is BN-owned iff it has both
    # running stats.  Each BN is emitted as a fused per-channel scale/shift
    # pair (y = x * scale + shift), matching what campplus_load expects.
    bn_groups: dict = {}
    for k in state:
        parts = k.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in ("weight", "bias", "running_mean",
                                            "running_var", "num_batches_tracked"):
            bn_groups.setdefault(parts[0], {})[parts[1]] = state[k]
    bn_prefixes = {p for p, t in bn_groups.items()
                   if "running_mean" in t and "running_var" in t}

    writer = gguf.GGUFWriter(str(args.out), "campplus")
    writer.add_name("CAM++ speaker encoder (campplus_cn_common)")
    writer.add_description(
        "3D-Speaker CAM++ 192-d speaker embedding for CosyVoice3 zero-shot "
        "voice cloning; same weights CosyVoice ships as campplus.onnx.")

    n_bn = 0
    n_conv = 0
    for k in sorted(state.keys()):
        parts = k.rsplit(".", 1)
        prefix, last = (parts[0], parts[1]) if len(parts) == 2 else (k, "")
        if last == "num_batches_tracked":
            continue
        gguf_base = "campplus/" + prefix.replace(".", "/")
        if prefix in bn_prefixes:
            if last in ("weight", "bias", "running_var"):
                continue
            grp = bn_groups[prefix]
            mean = grp["running_mean"].float()
            var = grp["running_var"].float()
            denom = torch.sqrt(var + BN_EPS)
            if "weight" in grp and "bias" in grp:
                scale = grp["weight"].float() / denom
                shift = grp["bias"].float() - mean * scale
            else:
                scale = 1.0 / denom
                shift = -mean * scale
            writer.add_tensor(gguf_base + "/s",
                              np.ascontiguousarray(scale.numpy().astype(np.float32)))
            writer.add_tensor(gguf_base + "/b",
                              np.ascontiguousarray(shift.numpy().astype(np.float32)))
            n_bn += 1
            continue
        writer.add_tensor("campplus/" + k.replace(".", "/"),
                          np.ascontiguousarray(state[k].float().numpy()))
        n_conv += 1

    writer.add_uint32("campplus.feat_dim",        80)
    writer.add_uint32("campplus.embedding_size",  192)
    writer.add_uint32("campplus.growth_rate",     32)
    writer.add_uint32("campplus.bn_size",         4)
    writer.add_uint32("campplus.init_channels",   128)
    writer.add_uint32("campplus.block1_layers",   12)
    writer.add_uint32("campplus.block2_layers",   24)
    writer.add_uint32("campplus.block3_layers",   16)
    writer.add_uint32("campplus.block1_dilation", 1)
    writer.add_uint32("campplus.block2_dilation", 2)
    writer.add_uint32("campplus.block3_dilation", 2)
    writer.add_uint32("campplus.kernel_size",     3)
    writer.add_uint32("campplus.seg_pool_len",    100)
    writer.add_uint32("campplus.sample_rate",     16000)

    fb = kaldi_mel_filterbank()
    writer.add_tensor("campplus/mel_fb_kaldi_80", np.ascontiguousarray(fb))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{n_conv} conv/linear tensors + {n_bn} fused BNs + kaldi fb {fb.shape})")


if __name__ == "__main__":
    main()
