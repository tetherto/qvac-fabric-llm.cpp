#!/usr/bin/env python3
"""Convert the MOSS-Speech codec (fnlp/MOSS-Speech-Codec) into one GGUF:

- the Whisper-VQ speech tokenizer (a causal Whisper encoder truncated at the
  quantizer, average pooling x4 and a 16384-entry codebook, 12.5 tokens/s);
- the CosyVoice2 token-to-wave decoder (flow encoder, conditional flow-matching
  estimator and HiFT vocoder), stored under the tensor names of
  convert-s3gen-to-gguf.py so the tts-cpp S3Gen code loads it;
- the CAM++ speaker encoder, copied from a CAM++ GGUF produced by
  convert-campplus-to-gguf.py (same weights the codec ships as campplus.onnx).

The default reply voice (a WAV such as the upstream assets/prompt_en.wav) is
stored as raw mono samples so the engine can speak without a voice prompt.

usage: convert-moss-speech-codec-to-gguf.py <codec_dir> --campplus-gguf campplus.gguf
           --voice-wav prompt_en.wav [--outtype f16|q8_0|f32] [--outfile out.gguf]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from safetensors.torch import load_file

ARCH = "moss-speech-codec"
SCRIPTS = Path(__file__).resolve().parent
ENCODER_BLOCKS = 6
UP_ENCODER_BLOCKS = 4
CFM_TIMESTEPS = 10
CFM_CFG_RATE = 0.7
FLOW_SAMPLE_RATE = 24000
FLOW_MEL_N_FFT = 1920
FLOW_MEL_BANDS = 80
FLOW_MEL_FMAX = 8000

VQ_LAYER_RENAMES = {
    "self_attn_layer_norm.weight": "attn_norm.weight",
    "self_attn_layer_norm.bias": "attn_norm.bias",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.out_proj.weight": "attn_output.weight",
    "self_attn.out_proj.bias": "attn_output.bias",
    "final_layer_norm.weight": "ffn_norm.weight",
    "final_layer_norm.bias": "ffn_norm.bias",
    "fc1.weight": "ffn_up.weight",
    "fc1.bias": "ffn_up.bias",
    "fc2.weight": "ffn_down.weight",
    "fc2.bias": "ffn_down.bias",
}


def load_s3gen_helpers():
    spec = importlib.util.spec_from_file_location("_s3gen_converter", SCRIPTS / "convert-s3gen-to-gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S3GEN = load_s3gen_helpers()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def prefixed(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {prefix + name: tensor for name, tensor in state.items() if torch.is_tensor(tensor)}


def load_decoder_state(flow_dir: Path) -> dict[str, torch.Tensor]:
    flow = torch.load(flow_dir / "flow.pt", map_location="cpu", weights_only=True)
    hift = torch.load(flow_dir / "hift.pt", map_location="cpu", weights_only=True)
    return S3GEN.expand_weight_norm({**prefixed(flow, "flow."), **prefixed(hift, "mel2wav.")})


def add(writer: gguf.GGUFWriter, name: str, tensor, quant: str) -> None:
    array = tensor if isinstance(tensor, np.ndarray) else S3GEN.as_numpy(tensor, dtype=torch.float32)
    S3GEN.add_tensor_maybe_q(writer, name, np.ascontiguousarray(array), quant)


def emit_flow_front(writer: gguf.GGUFWriter, state: dict, quant: str) -> None:
    pairs = {
        "flow/input_embedding": "flow.input_embedding.weight",
        "flow/spk_embed_affine/w": "flow.spk_embed_affine_layer.weight",
        "flow/spk_embed_affine/b": "flow.spk_embed_affine_layer.bias",
        "flow/encoder_proj/w": "flow.encoder_proj.weight",
        "flow/encoder_proj/b": "flow.encoder_proj.bias",
        "flow/encoder/embed/linear/w": "flow.encoder.embed.out.0.weight",
        "flow/encoder/embed/linear/b": "flow.encoder.embed.out.0.bias",
        "flow/encoder/embed/norm/w": "flow.encoder.embed.out.1.weight",
        "flow/encoder/embed/norm/b": "flow.encoder.embed.out.1.bias",
        "flow/encoder/pre_lookahead/conv1/w": "flow.encoder.pre_lookahead_layer.conv1.weight",
        "flow/encoder/pre_lookahead/conv1/b": "flow.encoder.pre_lookahead_layer.conv1.bias",
        "flow/encoder/pre_lookahead/conv2/w": "flow.encoder.pre_lookahead_layer.conv2.weight",
        "flow/encoder/pre_lookahead/conv2/b": "flow.encoder.pre_lookahead_layer.conv2.bias",
        "flow/encoder/up_layer/conv/w": "flow.encoder.up_layer.conv.weight",
        "flow/encoder/up_layer/conv/b": "flow.encoder.up_layer.conv.bias",
        "flow/encoder/up_embed/linear/w": "flow.encoder.up_embed.out.0.weight",
        "flow/encoder/up_embed/linear/b": "flow.encoder.up_embed.out.0.bias",
        "flow/encoder/up_embed/norm/w": "flow.encoder.up_embed.out.1.weight",
        "flow/encoder/up_embed/norm/b": "flow.encoder.up_embed.out.1.bias",
        "flow/encoder/after_norm/w": "flow.encoder.after_norm.weight",
        "flow/encoder/after_norm/b": "flow.encoder.after_norm.bias",
    }
    for name, key in pairs.items():
        add(writer, name, state[key], quant)


def emit_conformer_blocks(writer: gguf.GGUFWriter, state: dict, quant: str) -> None:
    for index in range(ENCODER_BLOCKS):
        S3GEN.export_conformer_block(writer, state, f"flow.encoder.encoders.{index}",
                                     f"flow/encoder/block{index}", quant)
    for index in range(UP_ENCODER_BLOCKS):
        S3GEN.export_conformer_block(writer, state, f"flow.encoder.up_encoders.{index}",
                                     f"flow/encoder/up_block{index}", quant)


def emit_prefix(writer: gguf.GGUFWriter, state: dict, source: str, target: str, quant: str) -> int:
    keys = sorted(key for key in state if key.startswith(source))
    for key in keys:
        add(writer, key.replace(source, target).replace(".", "/"), state[key], quant)
    return len(keys)


def emit_decoder(writer: gguf.GGUFWriter, state: dict, quant: str) -> None:
    import librosa

    emit_flow_front(writer, state, quant)
    emit_conformer_blocks(writer, state, quant)
    if emit_prefix(writer, state, "flow.decoder.estimator.", "cfm/", quant) == 0:
        raise RuntimeError("flow.pt has no decoder estimator tensors")
    if emit_prefix(writer, state, "mel2wav.", "hift/", quant) == 0:
        raise RuntimeError("hift.pt has no vocoder tensors")
    mel = librosa.filters.mel(sr=FLOW_SAMPLE_RATE, n_fft=FLOW_MEL_N_FFT, n_mels=FLOW_MEL_BANDS, fmin=0,
                              fmax=FLOW_MEL_FMAX).astype(np.float32)
    add(writer, "s3gen/mel_fb/24k_80", mel, quant)


def add_decoder_metadata(writer: gguf.GGUFWriter, state: dict) -> None:
    up_kernel = int(state["flow.encoder.up_layer.conv.weight"].shape[-1])
    writer.add_bool("s3gen.meanflow", False)
    writer.add_uint32("s3gen.n_timesteps", CFM_TIMESTEPS)
    writer.add_float32("s3gen.cfg_rate", CFM_CFG_RATE)
    writer.add_uint32("s3gen.speech_vocab_size", int(state["flow.input_embedding.weight"].shape[0]))
    writer.add_uint32("s3gen.input_size", int(state["flow.input_embedding.weight"].shape[1]))
    writer.add_uint32("s3gen.output_size", FLOW_MEL_BANDS)
    writer.add_uint32("s3gen.encoder.n_blocks", ENCODER_BLOCKS)
    writer.add_uint32("s3gen.encoder.up_n_blocks", UP_ENCODER_BLOCKS)
    writer.add_uint32("s3gen.encoder.attention_heads", 8)
    writer.add_uint32("s3gen.encoder.head_dim", 64)
    writer.add_uint32("s3gen.encoder.ff_size", int(state["flow.encoder.encoders.0.feed_forward.w_1.weight"].shape[0]))
    writer.add_uint32("s3gen.encoder.token_mel_ratio", (up_kernel - 1) // 2)
    writer.add_uint32("s3gen.encoder.pre_lookahead_len", int(state["flow.encoder.pre_lookahead_layer.conv1.weight"].shape[-1]) - 1)
    writer.add_float32("s3gen.layer_norm_eps", 1e-12)
    writer.add_uint32("s3gen.spk_embed_dim", int(state["flow.spk_embed_affine_layer.weight"].shape[1]))
    writer.add_uint32("s3gen.sample_rate", FLOW_SAMPLE_RATE)


def vq_name(key: str) -> str | None:
    if key in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
        return "whispervq." + key
    if key == "embed_positions.weight":
        return "whispervq.pos_embd"
    if key == "codebook.weight":
        return "whispervq.codebook"
    if key.startswith("layers."):
        index, rest = key[len("layers."):].split(".", 1)
        mapped = VQ_LAYER_RENAMES.get(rest)
        return f"whispervq.blk.{index}.{mapped}" if mapped else None
    return None


def vq_tensor(name: str, tensor: torch.Tensor) -> np.ndarray:
    array = S3GEN.as_numpy(tensor, dtype=torch.float32)
    if name.endswith(("conv1.weight", "conv2.weight")):
        return array.astype(np.float16)
    if name.endswith(("conv1.bias", "conv2.bias")):
        return array.reshape(-1, 1)
    return array


def is_vq_matrix(name: str, array: np.ndarray) -> bool:
    return array.ndim == 2 and name.startswith("whispervq.blk.") and name.endswith(".weight") and "norm" not in name


def vq_storage(name: str, array: np.ndarray, quant: str) -> np.ndarray:
    if quant != "f32" and is_vq_matrix(name, array):
        return array.astype(np.float16)
    return array


def emit_vq_encoder(writer: gguf.GGUFWriter, codec_dir: Path, config: dict[str, Any], quant: str) -> None:
    state = {key[len("encoder."):]: tensor for key, tensor in load_file(codec_dir / "model.safetensors").items()
             if key.startswith("encoder.")}
    layers = int(config["quantize_position"])
    for key, tensor in state.items():
        name = vq_name(key)
        if name is None or (name.startswith("whispervq.blk.") and int(name.split(".")[2]) >= layers):
            continue
        writer.add_tensor(name, np.ascontiguousarray(vq_storage(name, vq_tensor(name, tensor), quant)))


def emit_vq_mel_filters(writer: gguf.GGUFWriter, codec_dir: Path) -> None:
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(str(codec_dir))
    writer.add_tensor("whispervq.mel_filters", np.ascontiguousarray(extractor.mel_filters.T.astype(np.float32)))


def add_vq_metadata(writer: gguf.GGUFWriter, config: dict[str, Any], preprocessor: dict[str, Any]) -> None:
    if not (config.get("encoder_causal_attention") and config.get("encoder_causal_convolution")):
        raise ValueError("only the causal Whisper-VQ encoder is supported")
    if config.get("pooling_position") != config.get("quantize_position") or config.get("pooling_type") != "avg":
        raise ValueError("the pooling must be an average pool right before the quantizer")
    key = f"{ARCH}.vq"
    writer.add_uint32(f"{key}.sample_rate", int(preprocessor["sampling_rate"]))
    writer.add_uint32(f"{key}.n_fft", int(preprocessor["n_fft"]))
    writer.add_uint32(f"{key}.hop_length", int(preprocessor["hop_length"]))
    writer.add_uint32(f"{key}.n_mels", int(config["num_mel_bins"]))
    writer.add_uint32(f"{key}.chunk_samples", int(preprocessor["n_samples"]))
    writer.add_uint32(f"{key}.block_count", int(config["quantize_position"]))
    writer.add_uint32(f"{key}.embedding_length", int(config["d_model"]))
    writer.add_uint32(f"{key}.feed_forward_length", int(config["encoder_ffn_dim"]))
    writer.add_uint32(f"{key}.attention.head_count", int(config["encoder_attention_heads"]))
    writer.add_uint32(f"{key}.context_length", int(config["max_source_positions"]))
    writer.add_uint32(f"{key}.pooling_kernel", int(config["pooling_kernel_size"]))
    writer.add_uint32(f"{key}.codebook_size", int(config["quantize_vocab_size"]))


def copy_campplus(writer: gguf.GGUFWriter, path: Path) -> None:
    reader = gguf.GGUFReader(str(path))
    for tensor in reader.tensors:
        if tensor.name.startswith("campplus/"):
            data = np.asarray(tensor.data)
            writer.add_tensor(tensor.name, data.reshape([int(d) for d in reversed(tensor.shape)]))
    for field in reader.fields.values():
        if field.name.startswith("campplus.") and field.types == [gguf.GGUFValueType.UINT32]:
            writer.add_uint32(field.name, int(field.parts[field.data[0]][0]))


def emit_default_voice(writer: gguf.GGUFWriter, voice_wav: Path) -> None:
    import soundfile

    samples, rate = soundfile.read(str(voice_wav), dtype="float32", always_2d=True)
    writer.add_uint32(f"{ARCH}.voice.sample_rate", int(rate))
    writer.add_tensor("voice.default_audio", np.ascontiguousarray(samples.mean(axis=1).astype(np.float32)))


def convert(codec_dir: Path, campplus: Path, voice_wav: Path, outfile: Path, quant: str) -> None:
    config = read_json(codec_dir / "config.json")
    preprocessor = read_json(codec_dir / "preprocessor_config.json")
    state = load_decoder_state(codec_dir / "flow")
    writer = gguf.GGUFWriter(str(outfile), ARCH)
    try:
        writer.add_name(codec_dir.name)
        writer.add_string("s3gen.quantization", quant)
        add_vq_metadata(writer, config, preprocessor)
        emit_vq_encoder(writer, codec_dir, config, quant)
        emit_vq_mel_filters(writer, codec_dir)
        add_decoder_metadata(writer, state)
        emit_decoder(writer, state, quant)
        copy_campplus(writer, campplus)
        emit_default_voice(writer, voice_wav)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(f"{ARCH}: wrote {outfile}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codec_dir", type=Path)
    parser.add_argument("--campplus-gguf", type=Path, required=True)
    parser.add_argument("--voice-wav", type=Path, required=True)
    parser.add_argument("--outtype", choices=("f32", "f16", "q8_0"), default="f16")
    parser.add_argument("--outfile", type=Path, default=None)
    args = parser.parse_args()
    codec_dir = args.codec_dir.resolve()
    outfile = args.outfile or codec_dir / f"moss-speech-codec-{args.outtype}.gguf"
    convert(codec_dir, args.campplus_gguf, args.voice_wav, outfile, args.outtype)


if __name__ == "__main__":
    main()
