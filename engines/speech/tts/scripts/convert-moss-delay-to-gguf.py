#!/usr/bin/env python3
"""Convert a Hugging Face MOSS Delay TTS checkpoint (MOSS-TTS-v1.5 or
MOSS-TTSD, MossTTSDelay architecture over a Qwen3 backbone) into the
moss-tts-delay GGUF that the tts-cpp MOSS engine loads.

usage: convert-moss-delay-to-gguf.py <hf_checkpoint_dir> [--outtype f16|f32]
           [--outfile out.gguf]
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

ARCH = "moss-tts-delay"

SAFETENSORS_DTYPES = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
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
        for shard_path in sorted(model_dir.glob("*.safetensors")):
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
        if not self.locations:
            raise FileNotFoundError(f"no safetensors files under {model_dir}")

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


def count_audio_channels(index: "SafeTensorsIndex") -> int:
    channels = set()
    for name in index:
        if (match := re.fullmatch(r"emb_ext\.(\d+)\.weight", name)) is not None:
            channels.add(int(match.group(1)))
    return len(channels)


def reconcile_n_vq(hparams: dict[str, Any], index: "SafeTensorsIndex") -> None:
    tensor_n_vq = count_audio_channels(index)
    config_n_vq = int(hparams.get("n_vq", 0))
    if tensor_n_vq > 0 and tensor_n_vq != config_n_vq:
        print(f"note: config declares n_vq={config_n_vq} but the checkpoint carries "
              f"{tensor_n_vq} audio channels; using {tensor_n_vq}")
        hparams["n_vq"] = tensor_n_vq


def load_hparams(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    language = config.get("language_config")
    if isinstance(language, dict):
        merged = {key: value for key, value in language.items()
                  if key not in ("architectures", "model_type")}
        config = {**config, **merged}
    return config


def map_backbone_tensor_name(name: str) -> str | None:
    if name.startswith("language_model."):
        name = name[len("language_model."):]
    if name.startswith("model."):
        name = name[len("model."):]
    if (match := re.fullmatch(r"emb_ext\.(\d+)\.weight", name)) is not None:
        return f"token_embd_audio.{match.group(1)}.weight"
    if (match := re.fullmatch(r"lm_heads\.(\d+)\.weight", name)) is not None:
        head = int(match.group(1))
        return "output.weight" if head == 0 else f"output_audio.{head - 1}.weight"
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if (match := re.fullmatch(r"layers\.(\d+)\.(.+)", name)) is not None:
        mapped = LAYER_RENAMES.get(match.group(2))
        return f"blk.{match.group(1)}.{mapped}" if mapped else None
    if name == "lm_head.weight":
        return "output.weight"
    return None


def convert_tensor_dtype(tensor: np.ndarray, outtype: str) -> np.ndarray:
    if outtype == "f16" and tensor.ndim > 1:
        return np.ascontiguousarray(tensor.astype(np.float16, copy=False))
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False))


def read_tokenizer(model_dir: Path) -> tuple[list[str], list[str]]:
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    model = tokenizer["model"]
    vocab: dict[str, int] = model["vocab"]
    size = max(vocab.values()) + 1
    for added in tokenizer.get("added_tokens", []):
        size = max(size, int(added["id"]) + 1)
    tokens = [""] * size
    for token, token_id in vocab.items():
        tokens[token_id] = token
    for added in tokenizer.get("added_tokens", []):
        tokens[int(added["id"])] = added["content"]
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model["merges"]]
    return tokens, merges


def add_metadata(writer: gguf.GGUFWriter, hparams: dict[str, Any], model_name: str) -> None:
    writer.add_type("model")
    writer.add_name(model_name)
    n_heads = int(hparams["num_attention_heads"])
    head_dim = int(hparams.get("head_dim", hparams["hidden_size"] // n_heads))
    writer.add_uint32(f"{ARCH}.block_count", int(hparams["num_hidden_layers"]))
    writer.add_uint32(f"{ARCH}.context_length", int(hparams["max_position_embeddings"]))
    writer.add_uint32(f"{ARCH}.embedding_length", int(hparams["hidden_size"]))
    writer.add_uint32(f"{ARCH}.feed_forward_length", int(hparams["intermediate_size"]))
    writer.add_uint32(f"{ARCH}.attention.head_count", n_heads)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", int(hparams["num_key_value_heads"]))
    writer.add_uint32(f"{ARCH}.attention.key_length", head_dim)
    writer.add_uint32(f"{ARCH}.attention.value_length", head_dim)
    writer.add_float32(f"{ARCH}.rope.freq_base", float(hparams.get("rope_theta", 10000.0)))
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon",
                       float(hparams.get("rms_norm_eps", 1e-6)))
    writer.add_uint32(f"{ARCH}.n_vq", int(hparams["n_vq"]))
    writer.add_uint32(f"{ARCH}.audio_vocab_size", int(hparams["audio_vocab_size"]))
    writer.add_uint32(f"{ARCH}.audio_pad_code", int(hparams["audio_pad_code"]))
    writer.add_uint32(f"{ARCH}.audio_start_token_id", int(hparams["audio_start_token_id"]))
    writer.add_uint32(f"{ARCH}.audio_end_token_id", int(hparams["audio_end_token_id"]))
    writer.add_uint32(f"{ARCH}.audio_user_slot_token_id", int(hparams["audio_user_slot_token_id"]))
    writer.add_uint32(f"{ARCH}.audio_assistant_gen_slot_token_id",
                      int(hparams["audio_assistant_gen_slot_token_id"]))
    writer.add_uint32(f"{ARCH}.audio_assistant_delay_slot_token_id",
                      int(hparams["audio_assistant_delay_slot_token_id"]))
    if "sampling_rate" in hparams:
        writer.add_uint32(f"{ARCH}.sampling_rate", int(hparams["sampling_rate"]))


def add_tokenizer(writer: gguf.GGUFWriter, model_dir: Path) -> None:
    tokens, merges = read_tokenizer(model_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--outfile", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    outfile = args.outfile or model_dir / f"{model_dir.name}-{args.outtype}.gguf"
    hparams = load_hparams(model_dir)
    index = SafeTensorsIndex(model_dir)
    reconcile_n_vq(hparams, index)

    writer = gguf.GGUFWriter(path=outfile, arch=ARCH)
    try:
        add_metadata(writer, hparams, model_dir.name)
        add_tokenizer(writer, model_dir)
        emitted: set[str] = set()
        for name in index:
            mapped = map_backbone_tensor_name(name)
            if mapped is None or mapped in emitted:
                continue
            writer.add_tensor(mapped, convert_tensor_dtype(index.load(name), args.outtype))
            emitted.add(mapped)
        expected = 3 + 2 * int(hparams["n_vq"]) + 11 * int(hparams["num_hidden_layers"])
        if len(emitted) != expected:
            raise RuntimeError(
                f"emitted {len(emitted)} tensors but the architecture needs {expected}; "
                "the checkpoint uses tensor names this converter does not map")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(f"{ARCH}: wrote {outfile}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
