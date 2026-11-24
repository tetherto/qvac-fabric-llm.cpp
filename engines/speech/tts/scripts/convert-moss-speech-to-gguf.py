#!/usr/bin/env python3
"""Convert the Hugging Face MOSS-Speech checkpoint (MossSpeechForCausalLM: a
Qwen3 trunk shared by a text and an audio branch, each with its own layers,
final norm and output head) into the moss-speech GGUF that the tts-cpp MOSS
speech-to-speech engine loads. The codec is converted separately by
convert-moss-speech-codec-to-gguf.py.

usage: convert-moss-speech-to-gguf.py <hf_checkpoint_dir>
           [--outtype bf16|f16|q8_0|f32] [--outfile out.gguf]
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

ARCH = "moss-speech"
LAYER_TENSORS = 11
GLOBAL_TENSORS = 6
AUDIO_OUTPUT_SYSTEM_PROMPT = "You are a helpful assistant. Respond with spoken outputs."
TEXT_OUTPUT_SYSTEM_PROMPT = "You are a helpful assistant. Respond with text outputs."
TEXT_PLACEHOLDER_TOKEN_ID = 151667
AUDIO_PAD_TOKEN_ID = 512
SOSP_TOKEN_ID = 151646
EOSP_TOKEN_ID = 16384
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"

SAFETENSORS_DTYPES = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
}

OUTTYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}

LAYER_RENAMES = {
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

BLOCK_PREFIXES = {
    "model.shared_block.layers.": "blk.",
    "model.text_block.layers.": "text.blk.",
    "model.audio_block.layers.": "audio.blk.",
}

GLOBAL_RENAMES = {
    "model.embed_tokens.weight": "text.token_embd.weight",
    "model.audio_embed.weight": "audio.token_embd.weight",
    "model.text_norm.weight": "text.output_norm.weight",
    "model.audio_norm.weight": "audio.output_norm.weight",
    "text_lm_head.weight": "text.output.weight",
    "audio_lm_head.weight": "audio.output.weight",
}


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


def map_block_tensor(name: str) -> str | None:
    for prefix, target in BLOCK_PREFIXES.items():
        if not name.startswith(prefix):
            continue
        match = re.fullmatch(r"(\d+)\.(.+)", name[len(prefix):])
        mapped = LAYER_RENAMES.get(match.group(2)) if match else None
        return f"{target}{match.group(1)}.{mapped}" if mapped else None
    return None


def map_tensor_name(name: str) -> str | None:
    return GLOBAL_RENAMES.get(name) or map_block_tensor(name)


def is_matrix(mapped: str, tensor: np.ndarray) -> bool:
    return tensor.ndim == 2 and "norm" not in mapped


def quantizable(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> bool:
    if qtype in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return True
    return data.shape[-1] % gguf.GGML_QUANT_SIZES[qtype][0] == 0


def encode_tensor(writer: gguf.GGUFWriter, mapped: str, tensor: np.ndarray, outtype: str) -> None:
    data = np.ascontiguousarray(tensor, dtype=np.float32)
    qtype = OUTTYPES[outtype]
    if not is_matrix(mapped, data) or qtype == gguf.GGMLQuantizationType.F32:
        writer.add_tensor(mapped, data)
    elif qtype == gguf.GGMLQuantizationType.F16 or not quantizable(data, qtype):
        writer.add_tensor(mapped, data.astype(np.float16))
    else:
        writer.add_tensor(mapped, gguf.quants.quantize(data, qtype), raw_dtype=qtype)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def vocabulary_size(vocab: dict[str, int], added_tokens: list[dict]) -> int:
    return max([max(vocab.values())] + [int(added["id"]) for added in added_tokens]) + 1


def place_vocabulary(tokens: list[str], vocab: dict[str, int]) -> None:
    for token, token_id in vocab.items():
        tokens[token_id] = token


def place_added_tokens(tokens: list[str], types: list[int], added_tokens: list[dict]) -> None:
    for added in added_tokens:
        tokens[int(added["id"])] = added["content"]
        types[int(added["id"])] = int(gguf.TokenType.CONTROL if added.get("special", False)
                                      else gguf.TokenType.USER_DEFINED)


def read_tokenizer(model_dir: Path) -> tuple[list[str], list[str], list[int]]:
    tokenizer = read_json(model_dir / "tokenizer.json")
    model = tokenizer["model"]
    added_tokens = tokenizer.get("added_tokens", [])
    tokens = [""] * vocabulary_size(model["vocab"], added_tokens)
    types = [int(gguf.TokenType.NORMAL)] * len(tokens)
    place_vocabulary(tokens, model["vocab"])
    place_added_tokens(tokens, types, added_tokens)
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model["merges"]]
    return tokens, merges, types


def add_tokenizer(writer: gguf.GGUFWriter, model_dir: Path) -> list[str]:
    tokens, merges, types = read_tokenizer(model_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_array("tokenizer.ggml.token_type", types)
    return tokens


def add_model_metadata(writer: gguf.GGUFWriter, config: dict[str, Any]) -> None:
    n_heads = int(config["num_attention_heads"])
    writer.add_uint32(f"{ARCH}.block_count", int(config["num_shared_layers"]))
    writer.add_uint32(f"{ARCH}.modality_block_count", int(config["num_modality_layers"]))
    writer.add_uint32(f"{ARCH}.context_length", int(config["max_position_embeddings"]))
    writer.add_uint32(f"{ARCH}.embedding_length", int(config["hidden_size"]))
    writer.add_uint32(f"{ARCH}.feed_forward_length", int(config["intermediate_size"]))
    writer.add_uint32(f"{ARCH}.attention.head_count", n_heads)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", int(config["num_key_value_heads"]))
    writer.add_uint32(f"{ARCH}.attention.key_length", int(config.get("head_dim", int(config["hidden_size"]) // n_heads)))
    writer.add_float32(f"{ARCH}.rope.freq_base", float(config.get("rope_theta", 1000000.0)))
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))
    writer.add_uint32(f"{ARCH}.text_vocab_size", int(config["vocab_size"]))
    writer.add_uint32(f"{ARCH}.audio_vocab_size", int(config["audio_vocab_size"]))


def add_token_metadata(writer: gguf.GGUFWriter, tokens: list[str], config: dict[str, Any]) -> None:
    key = f"{ARCH}.token"
    writer.add_uint32(f"{key}.text_placeholder", int(config.get("modality_pad_token_id", TEXT_PLACEHOLDER_TOKEN_ID)))
    writer.add_uint32(f"{key}.audio_pad", int(config.get("audio_pad_token_id", AUDIO_PAD_TOKEN_ID)))
    writer.add_uint32(f"{key}.speech_start", int(config.get("sosp_token_id", SOSP_TOKEN_ID)))
    writer.add_uint32(f"{key}.speech_end", int(config.get("eosp_token_id", EOSP_TOKEN_ID)))
    writer.add_uint32(f"{key}.im_start", tokens.index(IM_START_TOKEN))
    writer.add_uint32(f"{key}.im_end", tokens.index(IM_END_TOKEN))
    writer.add_uint32(f"{key}.pad", tokens.index(PAD_TOKEN))


def add_prompt_metadata(writer: gguf.GGUFWriter) -> None:
    writer.add_string(f"{ARCH}.audio_output_system_prompt", AUDIO_OUTPUT_SYSTEM_PROMPT)
    writer.add_string(f"{ARCH}.text_output_system_prompt", TEXT_OUTPUT_SYSTEM_PROMPT)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("channels", 2) != 2:
        raise ValueError("MOSS-Speech checkpoints carry exactly two channels (text and audio)")
    if int(config["num_shared_layers"]) + int(config["num_modality_layers"]) != int(config["num_hidden_layers"]):
        raise ValueError("shared plus modality layers must add up to num_hidden_layers")
    if config.get("rope_scaling") not in (None, {}):
        raise ValueError("RoPE scaling is not supported")


def expected_tensor_count(config: dict[str, Any]) -> int:
    layers = int(config["num_shared_layers"]) + 2 * int(config["num_modality_layers"])
    return GLOBAL_TENSORS + LAYER_TENSORS * layers


def emit_tensors(writer: gguf.GGUFWriter, index: SafeTensorsIndex, outtype: str) -> int:
    emitted: set[str] = set()
    for name in index:
        mapped = map_tensor_name(name)
        if mapped is None:
            raise RuntimeError(f"unmapped tensor {name}")
        encode_tensor(writer, mapped, index.load(name), outtype)
        emitted.add(mapped)
    return len(emitted)


def convert(model_dir: Path, outfile: Path, outtype: str) -> None:
    config = read_json(model_dir / "config.json")
    validate_config(config)
    writer = gguf.GGUFWriter(path=outfile, arch=ARCH)
    try:
        writer.add_type("model")
        writer.add_name(model_dir.name)
        add_model_metadata(writer, config)
        add_token_metadata(writer, add_tokenizer(writer, model_dir), config)
        add_prompt_metadata(writer)
        emitted = emit_tensors(writer, SafeTensorsIndex(model_dir), outtype)
        if emitted != expected_tensor_count(config):
            raise RuntimeError(f"emitted {emitted} tensors but the architecture needs {expected_tensor_count(config)}")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(f"{ARCH}: wrote {outfile}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outtype", choices=tuple(OUTTYPES), default="bf16")
    parser.add_argument("--outfile", type=Path, default=None)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    convert(model_dir, args.outfile or model_dir / f"moss-speech-{args.outtype}.gguf", args.outtype)


if __name__ == "__main__":
    main()
