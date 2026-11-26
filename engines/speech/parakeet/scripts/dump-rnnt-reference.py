#!/usr/bin/env python3
"""Dump a NeMo greedy RNN-T reference for token-level decoder parity.

This is intended for hybrid ``EncDecHybridRNNTCTCBPEModel`` checkpoints:
it explicitly selects the RNN-T branch before writing ``token_ids.npy`` and
``transcript.txt``. By default it also writes the post-preprocessor mel and
encoder output used to diagnose parity failures.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav", type=Path, required=True,
                        help="Input audio file")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/rnnt-ref"),
                        help="Output directory")
    parser.add_argument("--nemo-model", type=Path, required=True,
                        help="Hybrid or plain RNN-T .nemo checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-encoder-dump", action="store_true",
                        help="Write only token IDs and transcript")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import nemo.collections.asr as nemo_asr

    print(f"[rnnt-ref] restoring {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.ASRModel.restore_from(
        str(args.nemo_model), map_location=args.device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    if hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoder_type="rnnt")
        print("[rnnt-ref] selected hybrid RNN-T branch", file=sys.stderr)

    if not args.no_encoder_dump:
        import soundfile as sf

        wav, sample_rate = sf.read(
            str(args.wav), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        expected_rate = int(model.cfg.preprocessor.sample_rate)
        if sample_rate != expected_rate:
            raise ValueError(
                f"audio is {sample_rate} Hz; model expects {expected_rate} Hz")

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(args.device)
        length = torch.tensor(
            [len(wav)], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            mel, mel_length = model.preprocessor(
                input_signal=wav_tensor, length=length)
            np.save(
                args.out / "mel.npy",
                mel[0].detach().cpu().numpy().astype(np.float32),
            )
            encoder_out, _ = model.encoder(
                audio_signal=mel, length=mel_length)
            encoder_array = (
                encoder_out[0].permute(1, 0).detach().cpu().numpy()
                .astype(np.float32)
            )
            np.save(args.out / "encoder_out.npy", encoder_array)

    hypotheses = model.transcribe([str(args.wav)], batch_size=1)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    hypothesis = hypotheses[0] if isinstance(hypotheses, list) else hypotheses

    text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
    (args.out / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    if not hasattr(hypothesis, "y_sequence"):
        raise RuntimeError("NeMo hypothesis has no y_sequence token IDs")

    sequence = hypothesis.y_sequence
    token_ids = (
        sequence.detach().cpu().numpy()
        if hasattr(sequence, "detach") else np.asarray(sequence)
    ).astype(np.int32)
    np.save(args.out / "token_ids.npy", token_ids)
    print(
        f"[rnnt-ref] wrote {len(token_ids)} tokens and transcript to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
