#!/usr/bin/env python3
"""Convert a Hugging Face MOSS-Transcribe-Diarize checkpoint (Whisper-shaped
audio encoder, 4x temporal-merge adaptor, Qwen3 decoder) into the single
moss-transcribe GGUF that the tts-cpp MOSS transcription engine loads.

usage: convert-moss-transcribe-to-gguf.py <hf_checkpoint_dir>
           [--outtype f16|bf16|f32|q8_0|q5_0] [--outfile out.gguf]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import gguf
import numpy as np

ARCH = "moss-transcribe"
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
SYSTEM_PROMPT = "You are a helpful assistant."
HOTWORD_PREFIX = "热词提示："
HOTWORD_SEPARATOR = ", "
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_PAD_TOKEN = "<|audio_pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
ENCODER_LAYER_NORM_EPS = 1e-5
MEL_MAX_FREQUENCY = 8000.0
MEL_MIN_FREQUENCY = 0.0
DEFAULT_MAX_NEW_TOKENS = 5120

TEXT_LAYER_TENSORS = 11
TEXT_GLOBAL_TENSORS = 2
ENCODER_LAYER_TENSORS = 15
ENCODER_GLOBAL_TENSORS = 7
ADAPTOR_TENSORS = 6
MEL_TENSORS = 1

SAFETENSORS_DTYPES = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
}

OUTTYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
}

TEXT_LAYER_RENAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

ENCODER_LAYER_RENAMES = {
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

ENCODER_GLOBAL_RENAMES = {
    "conv1.weight": "enc.conv1.weight",
    "conv1.bias": "enc.conv1.bias",
    "conv2.weight": "enc.conv2.weight",
    "conv2.bias": "enc.conv2.bias",
    "embed_positions.weight": "enc.pos_embd",
    "layer_norm.weight": "enc.output_norm.weight",
    "layer_norm.bias": "enc.output_norm.bias",
}

ADAPTOR_RENAMES = {
    "layers.0.weight": "adaptor.fc1.weight",
    "layers.0.bias": "adaptor.fc1.bias",
    "layers.2.weight": "adaptor.fc2.weight",
    "layers.2.bias": "adaptor.fc2.bias",
    "layers.3.weight": "adaptor.norm.weight",
    "layers.3.bias": "adaptor.norm.bias",
}

TEXT_PREFIX = "model.language_model."
ENCODER_PREFIX = "model.whisper_encoder."
ADAPTOR_PREFIX = "model.vq_adaptor."


@dataclass(frozen=True)
class TensorLocation:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    data_start: int


class SafeTensorsIndex:
    def __init__(self, model_dir: Path):
        self.locations: OrderedDict[str, TensorLocation] = OrderedDict()
        self._index_shards(model_dir)
        if not self.locations:
            raise FileNotFoundError(f"no safetensors files under {model_dir}")

    def _index_shards(self, model_dir: Path) -> None:
        for shard_path in sorted(model_dir.glob("*.safetensors")):
            self._index_shard(shard_path)

    def _index_shard(self, shard_path: Path) -> None:
        with shard_path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for tensor_name, meta in header.items():
            if tensor_name == "__metadata__":
                continue
            self.locations[tensor_name] = TensorLocation(
                shard=shard_path,
                dtype=meta["dtype"],
                shape=tuple(int(v) for v in meta["shape"]),
                data_offsets=(int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
                data_start=data_start,
            )

    def __iter__(self) -> Iterator[str]:
        return iter(self.locations.keys())

    def load(self, name: str) -> np.ndarray:
        loc = self.locations[name]
        offset = loc.data_start + loc.data_offsets[0]
        if loc.dtype == "BF16":
            raw = np.memmap(loc.shard, mode="r", dtype=np.uint16, offset=offset, shape=loc.shape)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        dtype = SAFETENSORS_DTYPES.get(loc.dtype)
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {loc.dtype!r} for {name!r}")
        return np.asarray(np.memmap(loc.shard, mode="r", dtype=dtype, offset=offset, shape=loc.shape))


def quantizable(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> bool:
    if qtype == gguf.GGMLQuantizationType.BF16:
        return True
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]
    return data.shape[-1] % block_size == 0


class Emitter:
    def __init__(self, writer: gguf.GGUFWriter, outtype: str):
        self.writer = writer
        self.outtype = outtype
        self.names: set[str] = set()

    def matrix(self, name: str, tensor: np.ndarray) -> None:
        data = np.ascontiguousarray(tensor, dtype=np.float32)
        qtype = OUTTYPES[self.outtype]
        if qtype == gguf.GGMLQuantizationType.F32:
            self._add(name, data)
            return
        if qtype == gguf.GGMLQuantizationType.F16 or not quantizable(data, qtype):
            self._add(name, data.astype(np.float16))
            return
        self._add(name, gguf.quants.quantize(data, qtype), raw_dtype=qtype)

    def conv(self, name: str, tensor: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(tensor, dtype=np.float16))

    def exact(self, name: str, tensor: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(tensor, dtype=np.float32))

    def _add(self, name: str, data: np.ndarray, raw_dtype: Any = None) -> None:
        if name in self.names:
            raise RuntimeError(f"duplicate tensor {name}")
        self.writer.add_tensor(name, data, raw_dtype=raw_dtype)
        self.names.add(name)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def map_text_tensor_name(name: str) -> str | None:
    name = name[len(TEXT_PREFIX):]
    if name == "embed_tokens.weight":
        return "text.token_embd.weight"
    if name == "norm.weight":
        return "text.output_norm.weight"
    if (match := re.fullmatch(r"layers\.(\d+)\.(.+)", name)) is not None:
        mapped = TEXT_LAYER_RENAMES.get(match.group(2))
        return f"text.blk.{match.group(1)}.{mapped}" if mapped else None
    return None


def map_encoder_tensor_name(name: str) -> str | None:
    name = name[len(ENCODER_PREFIX):]
    if name in ENCODER_GLOBAL_RENAMES:
        return ENCODER_GLOBAL_RENAMES[name]
    if (match := re.fullmatch(r"layers\.(\d+)\.(.+)", name)) is not None:
        mapped = ENCODER_LAYER_RENAMES.get(match.group(2))
        return f"enc.blk.{match.group(1)}.{mapped}" if mapped else None
    return None


def map_adaptor_tensor_name(name: str) -> str | None:
    return ADAPTOR_RENAMES.get(name[len(ADAPTOR_PREFIX):])


def column(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1, 1)


def is_matrix(mapped: str, tensor: np.ndarray) -> bool:
    return mapped.endswith(".weight") and tensor.ndim == 2 and "norm" not in mapped


def emit_text_tensor(emitter: Emitter, mapped: str, tensor: np.ndarray) -> None:
    if is_matrix(mapped, tensor):
        emitter.matrix(mapped, tensor)
    else:
        emitter.exact(mapped, tensor)


def emit_encoder_tensor(emitter: Emitter, mapped: str, tensor: np.ndarray) -> None:
    if mapped in ("enc.conv1.weight", "enc.conv2.weight"):
        emitter.conv(mapped, tensor)
    elif mapped in ("enc.conv1.bias", "enc.conv2.bias"):
        emitter.exact(mapped, column(tensor))
    elif mapped == "enc.pos_embd":
        emitter.exact(mapped, tensor)
    elif is_matrix(mapped, tensor):
        emitter.matrix(mapped, tensor)
    else:
        emitter.exact(mapped, tensor)


def emit_tensor(emitter: Emitter, name: str, tensor: np.ndarray) -> None:
    if name.startswith(TEXT_PREFIX):
        mapped = map_text_tensor_name(name)
        if mapped:
            emit_text_tensor(emitter, mapped, tensor)
            return
    if name.startswith(ENCODER_PREFIX):
        mapped = map_encoder_tensor_name(name)
        if mapped:
            emit_encoder_tensor(emitter, mapped, tensor)
            return
    if name.startswith(ADAPTOR_PREFIX):
        mapped = map_adaptor_tensor_name(name)
        if mapped:
            emit_text_tensor(emitter, mapped, tensor)
            return
    raise RuntimeError(f"unmapped tensor {name}")


def emit_checkpoint(emitter: Emitter, index: SafeTensorsIndex) -> None:
    for name in index:
        emit_tensor(emitter, name, index.load(name))


def hertz_to_mel(freq: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def triangular_filters(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def slaney_mel_filters(n_fft: int, n_mels: int, sample_rate: int) -> np.ndarray:
    num_frequency_bins = 1 + n_fft // 2
    mel_min = hertz_to_mel(np.array([MEL_MIN_FREQUENCY]))[0]
    mel_max = hertz_to_mel(np.array([MEL_MAX_FREQUENCY]))[0]
    mel_freqs = np.linspace(mel_min, mel_max, n_mels + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, sample_rate // 2, num_frequency_bins)
    filters = triangular_filters(fft_freqs, filter_freqs)
    enorm = 2.0 / (filter_freqs[2:n_mels + 2] - filter_freqs[:n_mels])
    return (filters * np.expand_dims(enorm, 0)).astype(np.float32)


def emit_mel_filters(emitter: Emitter, preprocessor: dict[str, Any]) -> None:
    filters = slaney_mel_filters(int(preprocessor["n_fft"]), int(preprocessor["feature_size"]),
                                 int(preprocessor["sampling_rate"]))
    emitter.exact("audio.mel_filters", filters.T)


def vocabulary_size(vocab: dict[str, int], added_tokens: list[dict]) -> int:
    return max([max(vocab.values())] + [int(added["id"]) for added in added_tokens]) + 1


def place_vocabulary(tokens: list[str], vocab: dict[str, int]) -> None:
    for token, token_id in vocab.items():
        tokens[token_id] = token


def place_added_tokens(tokens: list[str], added_tokens: list[dict]) -> None:
    for added in added_tokens:
        tokens[int(added["id"])] = added["content"]


def added_token_type(added: dict) -> int:
    return int(gguf.TokenType.CONTROL if added.get("special", False) else gguf.TokenType.USER_DEFINED)


def place_added_token_types(types: list[int], added_tokens: list[dict]) -> None:
    for added in added_tokens:
        types[int(added["id"])] = added_token_type(added)


def read_tokenizer(model_dir: Path) -> tuple[list[str], list[str], list[int]]:
    tokenizer = read_json(model_dir / "tokenizer.json")
    model = tokenizer["model"]
    added_tokens = tokenizer.get("added_tokens", [])
    tokens = [""] * vocabulary_size(model["vocab"], added_tokens)
    place_vocabulary(tokens, model["vocab"])
    place_added_tokens(tokens, added_tokens)
    types = [int(gguf.TokenType.NORMAL)] * len(tokens)
    place_added_token_types(types, added_tokens)
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model["merges"]]
    return tokens, merges, types


def add_tokenizer(writer: gguf.GGUFWriter, model_dir: Path) -> list[str]:
    tokens, merges, types = read_tokenizer(model_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_array("tokenizer.ggml.token_type", types)
    return tokens


def add_special_tokens(writer: gguf.GGUFWriter, tokens: list[str], config: dict[str, Any]) -> None:
    key = f"{ARCH}.token"
    writer.add_uint32(f"{key}.audio_start", tokens.index(AUDIO_START_TOKEN))
    writer.add_uint32(f"{key}.audio_end", tokens.index(AUDIO_END_TOKEN))
    writer.add_uint32(f"{key}.audio_pad", int(config.get("audio_token_id", tokens.index(AUDIO_PAD_TOKEN))))
    writer.add_uint32(f"{key}.im_start", tokens.index(IM_START_TOKEN))
    writer.add_uint32(f"{key}.im_end", tokens.index(IM_END_TOKEN))
    writer.add_uint32(f"{key}.pad", tokens.index(PAD_TOKEN))


def add_audio_metadata(writer: gguf.GGUFWriter, preprocessor: dict[str, Any]) -> None:
    key = f"{ARCH}.audio"
    writer.add_uint32(f"{key}.sample_rate", int(preprocessor["sampling_rate"]))
    writer.add_uint32(f"{key}.n_fft", int(preprocessor["n_fft"]))
    writer.add_uint32(f"{key}.hop_length", int(preprocessor["hop_length"]))
    writer.add_uint32(f"{key}.n_mels", int(preprocessor["feature_size"]))
    writer.add_uint32(f"{key}.chunk_samples", int(preprocessor["n_samples"]))
    writer.add_uint32(f"{key}.chunk_frames", int(preprocessor["nb_max_frames"]))


def add_encoder_metadata(writer: gguf.GGUFWriter, audio: dict[str, Any]) -> None:
    key = f"{ARCH}.encoder"
    writer.add_uint32(f"{key}.block_count", int(audio["encoder_layers"]))
    writer.add_uint32(f"{key}.embedding_length", int(audio["d_model"]))
    writer.add_uint32(f"{key}.feed_forward_length", int(audio["encoder_ffn_dim"]))
    writer.add_uint32(f"{key}.attention.head_count", int(audio["encoder_attention_heads"]))
    writer.add_uint32(f"{key}.context_length", int(audio["max_source_positions"]))
    writer.add_float32(f"{key}.attention.layer_norm_epsilon", ENCODER_LAYER_NORM_EPS)


def add_adaptor_metadata(writer: gguf.GGUFWriter, config: dict[str, Any], text: dict[str, Any]) -> None:
    key = f"{ARCH}.adaptor"
    writer.add_uint32(f"{key}.merge_size", int(config["audio_merge_size"]))
    writer.add_float32(f"{key}.layer_norm_epsilon", float(text.get("rms_norm_eps", 1e-6)))


def add_text_metadata(writer: gguf.GGUFWriter, text: dict[str, Any]) -> None:
    n_heads = int(text["num_attention_heads"])
    key = f"{ARCH}.text"
    writer.add_uint32(f"{key}.block_count", int(text["num_hidden_layers"]))
    writer.add_uint32(f"{key}.embedding_length", int(text["hidden_size"]))
    writer.add_uint32(f"{key}.feed_forward_length", int(text["intermediate_size"]))
    writer.add_uint32(f"{key}.attention.head_count", n_heads)
    writer.add_uint32(f"{key}.attention.head_count_kv", int(text["num_key_value_heads"]))
    writer.add_uint32(f"{key}.attention.key_length", int(text.get("head_dim", int(text["hidden_size"]) // n_heads)))
    writer.add_uint32(f"{key}.context_length", int(text["max_position_embeddings"]))
    writer.add_float32(f"{key}.rope.freq_base", float(text.get("rope_theta", 1000000.0)))
    writer.add_float32(f"{key}.attention.layer_norm_rms_epsilon", float(text.get("rms_norm_eps", 1e-6)))


def default_prompt_ids(model_dir: Path) -> list[int]:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json")).encode(DEFAULT_PROMPT, add_special_tokens=False).ids


def add_prompt_metadata(writer: gguf.GGUFWriter, processor: dict[str, Any], generation: dict[str, Any]) -> None:
    writer.add_float32(f"{ARCH}.audio_tokens_per_second", float(processor["audio_tokens_per_second"]))
    writer.add_uint32(f"{ARCH}.time_marker_every_seconds", int(processor["time_marker_every_seconds"]))
    writer.add_bool(f"{ARCH}.time_markers", bool(processor.get("enable_time_marker", True)))
    writer.add_string(f"{ARCH}.system_prompt", SYSTEM_PROMPT)
    writer.add_string(f"{ARCH}.default_prompt", DEFAULT_PROMPT)
    writer.add_string(f"{ARCH}.hotword_prefix", HOTWORD_PREFIX)
    writer.add_string(f"{ARCH}.hotword_separator", HOTWORD_SEPARATOR)
    writer.add_uint32(f"{ARCH}.default_max_new_tokens", int(generation.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)))


def expected_tensor_count(text: dict[str, Any], audio: dict[str, Any]) -> int:
    return (TEXT_GLOBAL_TENSORS + TEXT_LAYER_TENSORS * int(text["num_hidden_layers"]) +
            ENCODER_GLOBAL_TENSORS + ENCODER_LAYER_TENSORS * int(audio["encoder_layers"]) +
            ADAPTOR_TENSORS + MEL_TENSORS)


def validate_config(config: dict[str, Any]) -> None:
    if config["audio_config"].get("scale_embedding", False):
        raise ValueError("scaled encoder embeddings are not supported")
    if int(config["adaptor_input_dim"]) != int(config["audio_merge_size"]) * int(config["audio_config"]["d_model"]):
        raise ValueError("adaptor input width must be merge_size * encoder width")


def add_metadata(writer: gguf.GGUFWriter, model_dir: Path, config: dict[str, Any]) -> None:
    writer.add_type("model")
    writer.add_name(model_dir.name)
    add_audio_metadata(writer, read_json(model_dir / "preprocessor_config.json"))
    add_encoder_metadata(writer, config["audio_config"])
    add_adaptor_metadata(writer, config, config["text_config"])
    add_text_metadata(writer, config["text_config"])
    add_prompt_metadata(writer, read_json(model_dir / "processor_config.json"),
                        read_json(model_dir / "generation_config.json"))
    add_special_tokens(writer, add_tokenizer(writer, model_dir), config)
    writer.add_array(f"{ARCH}.default_prompt_ids", default_prompt_ids(model_dir))


def emit_tensors(writer: gguf.GGUFWriter, model_dir: Path, config: dict[str, Any], outtype: str) -> None:
    emitter = Emitter(writer, outtype)
    emit_checkpoint(emitter, SafeTensorsIndex(model_dir))
    emit_mel_filters(emitter, read_json(model_dir / "preprocessor_config.json"))
    expected = expected_tensor_count(config["text_config"], config["audio_config"])
    if len(emitter.names) != expected:
        raise RuntimeError(
            f"emitted {len(emitter.names)} tensors but the architecture needs {expected}; "
            "the checkpoint uses tensor names this converter does not map")


def write_gguf(writer: gguf.GGUFWriter) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)


def convert(model_dir: Path, outfile: Path, outtype: str) -> None:
    config = read_json(model_dir / "config.json")
    validate_config(config)
    writer = gguf.GGUFWriter(path=outfile, arch=ARCH)
    try:
        add_metadata(writer, model_dir, config)
        emit_tensors(writer, model_dir, config, outtype)
        write_gguf(writer)
        print(f"{ARCH}: wrote {outfile}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outtype", choices=tuple(OUTTYPES), default="f16")
    parser.add_argument("--outfile", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    outfile = args.outfile or model_dir / f"{model_dir.name}-{args.outtype}.gguf"
    convert(model_dir, outfile, args.outtype)


if __name__ == "__main__":
    main()
