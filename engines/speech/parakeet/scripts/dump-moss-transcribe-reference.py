#!/usr/bin/env python3
"""Dump MOSS-Transcribe-Diarize stage references from the PyTorch pipeline
(fp32, CPU) for test-moss-transcribe-parity: the first chunk's log-mel and
encoder output, the adapted audio embeddings, the prompt ids with time
markers, the prefill logits of the last prompt position, and the greedy
generated ids and text, as raw little-endian .bin files plus meta.json.

usage: dump-moss-transcribe-reference.py <hf_checkpoint_dir> <audio_16k.wav> <out_dir>
           --upstream <MOSS-Transcribe-Diarize checkout> [--max-new-tokens N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MAX_NEW_TOKENS = 1024
SAMPLE_RATE = 16000


def save(out: Path, name: str, tensor) -> None:
    array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") and tensor.is_floating_point() \
        else (tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor))
    array.astype(np.int32 if array.dtype.kind in "iu" else np.float32).tofile(out / f"{name}.bin")


def read_mono_16k(path: Path) -> np.ndarray:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz; resample it to {SAMPLE_RATE} Hz first")
    return audio.mean(axis=1)


def load(model_dir: str, upstream: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    sys.path.insert(0, str(upstream))
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor


def prepare(processor, audio: np.ndarray):
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    messages = [{"role": "user", "content": [{"type": "audio", "audio": "in-memory"},
                                             {"type": "text", "text": DEFAULT_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=text, audio=[audio], return_tensors="pt")


def dump_audio_stages(model, inputs, out: Path):
    features = inputs["input_features"]
    save(out, "mel_chunk0", features[0])
    encoded = model.model.whisper_encoder(features[:1], return_dict=True).last_hidden_state
    save(out, "encoder_chunk0", encoded[0])
    adapted = model.model.get_audio_features(features, inputs["audio_feature_lengths"],
                                             inputs["audio_chunk_mapping"])[0]
    save(out, "audio_embeddings", adapted[0])
    return adapted


def dump_prefill_logits(model, inputs, out: Path) -> None:
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    input_features=inputs["input_features"],
                    audio_feature_lengths=inputs["audio_feature_lengths"],
                    audio_chunk_mapping=inputs["audio_chunk_mapping"])
    save(out, "prefill_logits", outputs.logits[0, -1])


def dump_generation(model, processor, inputs, out: Path, max_new_tokens: int) -> str:
    generated = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               input_features=inputs["input_features"],
                               audio_feature_lengths=inputs["audio_feature_lengths"],
                               audio_chunk_mapping=inputs["audio_chunk_mapping"],
                               max_new_tokens=max_new_tokens, do_sample=False)
    ids = generated[0][inputs["input_ids"].shape[1]:]
    save(out, "generated_ids", ids)
    text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
    (out / "text.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("audio", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    import torch

    torch.set_grad_enabled(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load(args.model_dir, args.upstream)
    audio = read_mono_16k(args.audio)
    save(args.out_dir, "audio", audio)
    inputs = prepare(processor, audio)
    save(args.out_dir, "prompt_ids", inputs["input_ids"][0])
    adapted = dump_audio_stages(model, inputs, args.out_dir)
    dump_prefill_logits(model, inputs, args.out_dir)
    text = dump_generation(model, processor, inputs, args.out_dir, args.max_new_tokens)
    (args.out_dir / "meta.json").write_text(json.dumps({
        "audio": str(args.audio), "samples": int(audio.shape[0]), "chunks": int(inputs["input_features"].shape[0]),
        "audio_tokens": int(adapted.shape[1]), "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "max_new_tokens": args.max_new_tokens}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
