#!/usr/bin/env python3
"""Dump offline and cache-aware Nemotron reference tensors from NVIDIA NeMo.

The dump is intended for C++ numerical-parity work. It records the offline
preprocessor, encoder, prompt projection, tokens and transcript, plus the
initial encoder caches and leading cache-aware stream steps.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


RIGHT_CONTEXT_TO_CHUNK_MS = {
    0: 80,
    1: 160,
    3: 320,
    6: 560,
    13: 1120,
}

LEFT_CONTEXT_FRAMES = 56
ENCODER_WIDTH = 1024
PROMPT_WIDTH = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--nemo-model",
        type=Path,
        default=Path("models/nemotron-3.5-asr-streaming-0.6b.nemo"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/nemotron-ref"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-lang", default="en-US")
    parser.add_argument(
        "--right-context",
        type=int,
        choices=tuple(RIGHT_CONTEXT_TO_CHUNK_MS),
        default=13,
    )
    parser.add_argument(
        "--capture-steps",
        type=positive_int,
        default=2,
        help="Number of leading stream steps whose tensors are written.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Write offline parity tensors without rerunning the stream dump.",
    )
    return parser.parse_args()


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value):
    if not isinstance(value, torch.Tensor):
        return None
    return list(value.shape)


def tensor_to_numpy(value):
    return value.detach().cpu().numpy()


def save_tensor(path, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor for {path.name}, got {type(value)}")
    np.save(path, np.ascontiguousarray(tensor_to_numpy(value)))


def save_optional_tensor(path, value):
    if isinstance(value, torch.Tensor):
        save_tensor(path, value)


def extract_transcriptions(hypotheses):
    if hypotheses is None:
        return []
    return [
        str(hypothesis.text)
        if hasattr(hypothesis, "text")
        else str(hypothesis)
        for hypothesis in hypotheses
    ]


def hypothesis_token_ids(hypotheses):
    if not hypotheses:
        return None
    sequence = getattr(hypotheses[0], "y_sequence", None)
    if sequence is None:
        return None
    if isinstance(sequence, torch.Tensor):
        return sequence.detach().cpu().to(torch.int32)
    return torch.as_tensor(sequence, dtype=torch.int32)


def resolve_prompt_index(model, target_lang):
    prompt_dictionary = model.cfg.model_defaults.get("prompt_dictionary", {})
    if target_lang not in prompt_dictionary:
        available = ", ".join(sorted(prompt_dictionary))
        raise ValueError(
            f"Unknown target language {target_lang!r}. Available: {available}",
        )
    return int(prompt_dictionary[target_lang])


def configure_model(model, target_lang, right_context):
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    model.set_inference_prompt(target_lang)
    model.encoder.set_default_att_context_size(
        att_context_size=[LEFT_CONTEXT_FRAMES, right_context],
    )


def register_prompt_capture(model, capture):
    def capture_input(_module, inputs):
        capture["input"] = inputs[0].detach().cpu()

    def capture_output(_module, _inputs, output):
        capture["output"] = output.detach().cpu()

    return (
        model.prompt_kernel.register_forward_pre_hook(capture_input),
        model.prompt_kernel.register_forward_hook(capture_output),
    )


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def prompt_tensors(capture):
    prompt_input = capture.get("input")
    prompt_output = capture.get("output")
    if prompt_input is None or prompt_output is None:
        raise RuntimeError("NeMo did not execute prompt_kernel for this step")
    if prompt_input.shape[-1] != ENCODER_WIDTH + PROMPT_WIDTH:
        raise RuntimeError(
            "Unexpected prompt input width "
            f"{prompt_input.shape[-1]} "
            f"(expected {ENCODER_WIDTH + PROMPT_WIDTH})",
        )
    return (
        prompt_input[..., :ENCODER_WIDTH],
        prompt_input[..., ENCODER_WIDTH:],
        prompt_output,
    )


def drop_extra_pre_encoded(model, step_index):
    if step_index == 0:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


def clone_cache(cache):
    return cache.detach().cpu().clone()


def save_initial_state(output_dir, cache_channel, cache_time, cache_length):
    initial_dir = output_dir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(initial_dir / "cache_channel.npy", cache_channel)
    save_tensor(initial_dir / "cache_time.npy", cache_time)
    save_tensor(initial_dir / "cache_length.npy", cache_length)


def save_captured_step(
    output_dir,
    step_index,
    chunk_audio,
    chunk_lengths,
    cache_before,
    cache_after,
    pred_out_stream,
    prompt_capture,
    transcribed_texts,
    hypotheses,
):
    step_dir = output_dir / f"step-{step_index:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    cache_channel_before, cache_time_before, cache_length_before = cache_before
    cache_channel_after, cache_time_after, cache_length_after = cache_after
    encoder_raw, prompt_one_hot, prompt_output = prompt_tensors(prompt_capture)

    save_tensor(step_dir / "processed_signal.npy", chunk_audio)
    save_tensor(step_dir / "processed_signal_length.npy", chunk_lengths)
    save_tensor(step_dir / "cache_channel_before.npy", cache_channel_before)
    save_tensor(step_dir / "cache_time_before.npy", cache_time_before)
    save_tensor(step_dir / "cache_length_before.npy", cache_length_before)
    save_tensor(step_dir / "encoder_raw.npy", encoder_raw)
    save_tensor(step_dir / "prompt_one_hot.npy", prompt_one_hot)
    save_tensor(step_dir / "prompt_output.npy", prompt_output)
    save_tensor(step_dir / "cache_channel_after.npy", cache_channel_after)
    save_tensor(step_dir / "cache_time_after.npy", cache_time_after)
    save_tensor(step_dir / "cache_length_after.npy", cache_length_after)
    save_optional_tensor(step_dir / "pred_out_stream.npy", pred_out_stream)
    save_optional_tensor(
        step_dir / "token_ids.npy",
        hypothesis_token_ids(hypotheses),
    )

    return {
        "index": step_index,
        "processed_signal_shape": tensor_shape(chunk_audio),
        "processed_signal_length": tensor_to_numpy(chunk_lengths).tolist(),
        "cache_channel_before_shape": tensor_shape(cache_channel_before),
        "cache_time_before_shape": tensor_shape(cache_time_before),
        "cache_length_before": tensor_to_numpy(cache_length_before).tolist(),
        "encoder_raw_shape": tensor_shape(encoder_raw),
        "prompt_one_hot_shape": tensor_shape(prompt_one_hot),
        "prompt_output_shape": tensor_shape(prompt_output),
        "cache_channel_after_shape": tensor_shape(cache_channel_after),
        "cache_time_after_shape": tensor_shape(cache_time_after),
        "cache_length_after": tensor_to_numpy(cache_length_after).tolist(),
        "transcriptions": extract_transcriptions(transcribed_texts),
    }


def stream_reference(model, streaming_buffer, output_dir, capture_steps):
    batch_size = len(streaming_buffer.streams_length)
    cache_channel, cache_time, cache_length = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )
    save_initial_state(
        output_dir,
        cache_channel,
        cache_time,
        cache_length,
    )

    prompt_capture = {}
    hook_handles = register_prompt_capture(model, prompt_capture)
    previous_hypotheses = None
    pred_out_stream = None
    transcribed_texts = None
    captured_steps = []
    step_count = 0

    try:
        for step_index, (chunk_audio, chunk_lengths) in enumerate(
            streaming_buffer,
        ):
            prompt_capture.clear()
            cache_before = (
                clone_cache(cache_channel),
                clone_cache(cache_time),
                clone_cache(cache_length),
            )
            chunk_audio = chunk_audio.to(
                device=model.device,
                dtype=model.dtype,
            )
            chunk_lengths = chunk_lengths.to(device=model.device)

            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_channel,
                    cache_time,
                    cache_length,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_channel,
                    cache_last_time=cache_time,
                    cache_last_channel_len=cache_length,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=drop_extra_pre_encoded(
                        model,
                        step_index,
                    ),
                    return_transcription=True,
                )

            if step_index < capture_steps:
                cache_after = (
                    clone_cache(cache_channel),
                    clone_cache(cache_time),
                    clone_cache(cache_length),
                )
                captured_steps.append(
                    save_captured_step(
                        output_dir,
                        step_index,
                        chunk_audio.detach().cpu(),
                        chunk_lengths.detach().cpu(),
                        cache_before,
                        cache_after,
                        pred_out_stream,
                        prompt_capture,
                        transcribed_texts,
                        previous_hypotheses,
                    ),
                )
            step_count = step_index + 1
    finally:
        remove_hooks(hook_handles)

    return transcribed_texts, captured_steps, step_count


def build_streaming_buffer(model, wav_path):
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    streaming_buffer.append_audio_file(str(wav_path), stream_id=-1)
    return streaming_buffer


def load_audio(wav_path, expected_sample_rate, device):
    import soundfile as sf

    audio, sample_rate = sf.read(
        str(wav_path),
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Audio sample rate is {sample_rate}; "
            f"expected {expected_sample_rate}",
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    signal = torch.from_numpy(audio).unsqueeze(0).to(device)
    length = torch.tensor([signal.shape[1]], dtype=torch.long, device=device)
    return signal, length


def offline_reference(model, wav_path, prompt_index, output_dir):
    sample_rate = int(model.cfg.preprocessor.sample_rate)
    signal, signal_length = load_audio(
        wav_path,
        sample_rate,
        model.device,
    )
    prompt_indices = torch.tensor(
        [prompt_index],
        dtype=torch.long,
        device=model.device,
    )
    prompt_capture = {}
    hook_handles = register_prompt_capture(model, prompt_capture)
    try:
        with torch.inference_mode():
            processed_signal, processed_length = model.preprocessor(
                input_signal=signal,
                length=signal_length,
            )
            encoded, encoded_length = model(
                processed_signal=processed_signal,
                processed_signal_length=processed_length,
                prompt_indices=prompt_indices,
            )
            hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
    finally:
        remove_hooks(hook_handles)

    encoder_raw, prompt_one_hot, prompt_output = prompt_tensors(
        prompt_capture,
    )
    offline_dir = output_dir / "offline"
    offline_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(
        offline_dir / "mel.npy",
        processed_signal[0].transpose(0, 1),
    )
    save_tensor(offline_dir / "encoder_raw.npy", encoder_raw[0])
    save_tensor(offline_dir / "prompt_one_hot.npy", prompt_one_hot[0])
    save_tensor(offline_dir / "prompt_output.npy", prompt_output[0])
    save_optional_tensor(
        offline_dir / "token_ids.npy",
        hypothesis_token_ids(hypotheses),
    )

    transcriptions = extract_transcriptions(hypotheses)
    (offline_dir / "transcript.txt").write_text(
        "\n".join(transcriptions) + "\n",
        encoding="utf-8",
    )
    return {
        "mel_shape": list(processed_signal[0].transpose(0, 1).shape),
        "encoder_raw_shape": list(encoder_raw[0].shape),
        "prompt_output_shape": list(prompt_output[0].shape),
        "encoded_length": tensor_to_numpy(encoded_length).tolist(),
        "transcriptions": transcriptions,
    }


def write_manifest(
    args,
    model,
    prompt_index,
    offline,
    captured_steps,
    step_count,
    transcriptions,
    nemo_version,
):
    manifest = {
        "checkpoint": str(args.nemo_model),
        "checkpoint_sha256": sha256_file(args.nemo_model),
        "wav": str(args.wav),
        "nemo_version": nemo_version,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "target_lang": args.target_lang,
        "prompt_index": prompt_index,
        "left_context_frames": LEFT_CONTEXT_FRAMES,
        "right_context_frames": args.right_context,
        "chunk_ms": RIGHT_CONTEXT_TO_CHUNK_MS[args.right_context],
        "offline": offline,
        "total_stream_steps": step_count,
        "captured_stream_steps": captured_steps,
        "final_transcriptions": transcriptions,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_inputs(args):
    if not args.nemo_model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.nemo_model}")
    if not args.wav.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.wav}")


def main():
    args = parse_args()
    validate_inputs(args)
    args.out.mkdir(parents=True, exist_ok=True)

    import nemo
    import nemo.collections.asr as nemo_asr

    print(f"[nemotron-ref] restoring {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(args.nemo_model),
        map_location=args.device,
    )
    model = model.to(args.device)
    configure_model(model, args.target_lang, args.right_context)
    prompt_index = resolve_prompt_index(model, args.target_lang)

    print(
        "[nemotron-ref] "
        f"language={args.target_lang} prompt={prompt_index} "
        f"context=[{LEFT_CONTEXT_FRAMES},{args.right_context}] "
        f"chunk={RIGHT_CONTEXT_TO_CHUNK_MS[args.right_context]}ms",
        file=sys.stderr,
    )

    offline = offline_reference(
        model,
        args.wav,
        prompt_index,
        args.out,
    )
    if args.offline_only:
        print(
            f"[nemotron-ref] offline output -> {args.out / 'offline'}",
            file=sys.stderr,
        )
        return

    streaming_buffer = build_streaming_buffer(model, args.wav)
    transcribed_texts, captured_steps, step_count = stream_reference(
        model,
        streaming_buffer,
        args.out,
        args.capture_steps,
    )
    transcriptions = extract_transcriptions(transcribed_texts)
    (args.out / "transcript.txt").write_text(
        "\n".join(transcriptions) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        args,
        model,
        prompt_index,
        offline,
        captured_steps,
        step_count,
        transcriptions,
        nemo.__version__,
    )
    print(
        f"[nemotron-ref] {step_count} steps; output -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
