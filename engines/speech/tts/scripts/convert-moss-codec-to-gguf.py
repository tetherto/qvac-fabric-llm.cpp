#!/usr/bin/env python3
"""Split a Hugging Face MOSS Audio Tokenizer checkpoint into the
moss-tts-audio-encoder and moss-tts-audio-decoder GGUF files that the
tts-cpp MOSS engine loads.

usage: convert-moss-codec-to-gguf.py <hf_checkpoint_dir> [--outtype f16|f32]
           [--encoder-outfile out.gguf] [--decoder-outfile out.gguf]
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import gguf
import numpy as np

ARCH_ENCODER = "moss-tts-audio-encoder"
ARCH_DECODER = "moss-tts-audio-decoder"

DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_DOWNSAMPLE_RATE = 1_920
DEFAULT_CONTEXT_DURATION = 10.0

SUPPORTED_MODULE_TYPES = {"PatchedPretransform", "Transformer"}
SUPPORTED_QUANTIZER_TYPES = {"rlfq"}

SAFETENSORS_DTYPES = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
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
            header, data_start = self._load_header(shard_path)
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

    @staticmethod
    def _load_header(shard_path: Path) -> tuple[dict[str, Any], int]:
        with shard_path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        return header, 8 + header_len

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


def merge_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, v.ndim))
    norm = np.linalg.norm(v.astype(np.float32), axis=axes, keepdims=True)
    norm = np.maximum(norm, np.finfo(np.float32).eps)
    return g.astype(np.float32) * v.astype(np.float32) / norm


def map_tensor_name(name: str) -> str | None:
    if ".parametrizations.weight.original0" in name:
        return name.replace(".parametrizations.weight.original0", ".weight")
    if ".parametrizations.weight.original1" in name:
        return None
    return name


def convert_tensor_dtype(tensor: np.ndarray, outtype: str) -> np.ndarray:
    if not np.issubdtype(tensor.dtype, np.floating):
        return np.ascontiguousarray(tensor)
    if outtype == "f16" and tensor.ndim > 1:
        return np.ascontiguousarray(tensor.astype(np.float16, copy=False))
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False))


def validate_config(config: dict[str, Any]) -> None:
    quantizer_type = config.get("quantizer_type") or config.get("quantizer_kwargs", {}).get("quantizer_type")
    if quantizer_type not in SUPPORTED_QUANTIZER_TYPES:
        raise ValueError(f"unsupported quantizer_type {quantizer_type!r}")
    for section in ("encoder_kwargs", "decoder_kwargs"):
        for idx, module_cfg in enumerate(config.get(section, [])):
            module_type = module_cfg.get("module_type")
            if module_type not in SUPPORTED_MODULE_TYPES:
                raise ValueError(f"unsupported {section}[{idx}].module_type={module_type!r}")
            if module_type != "Transformer":
                continue
            if module_cfg.get("gating", "none") != "none":
                raise ValueError(f"unsupported {section}[{idx}].gating")
            if module_cfg.get("positional_embedding", "rope") != "rope":
                raise ValueError(f"unsupported {section}[{idx}].positional_embedding")
            if module_cfg.get("weights_per_step") or module_cfg.get("weights_per_step_schedule"):
                raise ValueError(f"unsupported {section}[{idx}] weights_per_step")


def add_config_value(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        writer.add_uint32(key, value) if value >= 0 else writer.add_int32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        raise TypeError(f"unsupported config value for {key!r}: {type(value)!r}")


def write_module_list_metadata(writer: gguf.GGUFWriter, arch: str, section_name: str,
                               module_cfgs: list[dict[str, Any]], initial_frame_rate: float,
                               context_duration: float, is_encoder: bool) -> None:
    writer.add_uint32(f"{arch}.{section_name}.block_count", len(module_cfgs))
    frame_rate = initial_frame_rate
    for idx, module_cfg in enumerate(module_cfgs):
        prefix = f"{arch}.{section_name}.{idx}"
        module_type = module_cfg["module_type"]
        writer.add_string(f"{prefix}.module_type", module_type)
        if module_type == "PatchedPretransform":
            patch_size = int(module_cfg["patch_size"])
            writer.add_uint32(f"{prefix}.patch_size", patch_size)
            frame_rate = frame_rate / patch_size if is_encoder else frame_rate * patch_size
            continue
        add_config_value(writer, f"{prefix}.context", int(frame_rate * context_duration))
        for key in ("input_dimension", "output_dimension", "d_model", "num_heads", "num_layers",
                    "dim_feedforward", "causal", "norm", "positional_embedding", "max_period",
                    "layer_scale", "conv_layout", "gating"):
            if key in module_cfg:
                add_config_value(writer, f"{prefix}.{key}", module_cfg[key])


def add_common_metadata(writer: gguf.GGUFWriter, arch: str, config: dict[str, Any], model_name: str) -> None:
    writer.add_type("model")
    writer.add_name(model_name)
    quantizer_cfg = dict(config.get("quantizer_kwargs", {}))
    writer.add_uint32(f"{arch}.sampling_rate", int(config.get("sampling_rate", DEFAULT_SAMPLING_RATE)))
    writer.add_uint32(f"{arch}.downsample_rate", int(config.get("downsample_rate", DEFAULT_DOWNSAMPLE_RATE)))
    writer.add_float32(f"{arch}.causal_transformer_context_duration",
                       float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION)))
    writer.add_uint32(f"{arch}.code_dim", int(config.get("code_dim", quantizer_cfg.get("output_dim", 0))))
    writer.add_string(f"{arch}.quantizer_type",
                      config.get("quantizer_type") or quantizer_cfg.get("quantizer_type", "rlfq"))
    writer.add_uint32(f"{arch}.quantizer.input_dim", int(quantizer_cfg["input_dim"]))
    writer.add_uint32(f"{arch}.quantizer.rvq_dim", int(quantizer_cfg.get("rvq_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{arch}.quantizer.output_dim", int(quantizer_cfg.get("output_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{arch}.quantizer.num_quantizers", int(quantizer_cfg["num_quantizers"]))
    writer.add_uint32(f"{arch}.quantizer.codebook_size", int(quantizer_cfg["codebook_size"]))
    writer.add_uint32(f"{arch}.quantizer.codebook_dim", int(quantizer_cfg["codebook_dim"]))


def build_transformer_block_index_map(module_cfgs: list[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    tensor_block = 0
    for module_idx, module_cfg in enumerate(module_cfgs):
        if module_cfg.get("module_type") != "Transformer":
            continue
        result[module_idx] = tensor_block
        tensor_block += 1
    return result


def map_transformer_tensor_name(tensor_block: int, tail: str) -> str | None:
    if tail == "input_proj.weight":
        return f"blk.{tensor_block}.input_proj.weight"
    if tail == "output_proj.weight":
        return f"blk.{tensor_block}.output_proj.weight"
    parts = tail.split(".")
    if len(parts) < 5 or parts[0] != "transformer" or parts[1] != "layers":
        return None
    layer_prefix = f"blk.{tensor_block}.layer.{int(parts[2])}"
    layer_tail = ".".join(parts[3:])
    renames = {
        "layer_scale_1.scale": "attn_scale.scale",
        "layer_scale_2.scale": "ffn_scale.scale",
        "linear1.weight": "ffn_up.weight",
        "linear2.weight": "ffn_down.weight",
        "norm1.weight": "attn_norm.weight",
        "norm1.bias": "attn_norm.bias",
        "norm2.weight": "ffn_norm.weight",
        "norm2.bias": "ffn_norm.bias",
        "self_attn.in_projs.0.weight": "attn_qkv.weight",
        "self_attn.out_projs.0.weight": "attn_output.weight",
    }
    mapped = renames.get(layer_tail)
    return f"{layer_prefix}.{mapped}" if mapped else None


def map_split_tensor_name(name: str, encoder_block_map: dict[int, int],
                          decoder_block_map: dict[int, int]) -> str | None:
    mapped = map_tensor_name(name)
    if mapped is None:
        return None
    for section, block_map in (("encoder.", encoder_block_map), ("decoder.", decoder_block_map)):
        if mapped.startswith(section):
            module_idx_str, tail = mapped[len(section):].split(".", 1)
            return map_transformer_tensor_name(block_map[int(module_idx_str)], tail)
    return mapped


def is_encoder_tensor(name: str) -> bool:
    if name.startswith("encoder.") or name.startswith("quantizer.input_proj."):
        return True
    return name.startswith("quantizer.quantizers.") and (
        ".in_proj." in name or ".out_proj." in name or ".codebook." in name)


def is_decoder_tensor(name: str) -> bool:
    if name.startswith("decoder.") or name.startswith("quantizer.output_proj."):
        return True
    return name.startswith("quantizer.quantizers.") and (".out_proj." in name or ".codebook." in name)


def iter_filtered_tensors(index: SafeTensorsIndex, outtype: str,
                          include_fn: Callable[[str], bool],
                          rename_fn: Callable[[str], str | None]) -> Iterator[tuple[str, np.ndarray]]:
    emitted: set[str] = set()
    for name in index:
        mapped_name = map_tensor_name(name)
        renamed_name = rename_fn(name)
        if mapped_name is None or renamed_name is None or renamed_name in emitted:
            continue
        if not include_fn(mapped_name):
            continue
        if ".parametrizations.weight.original0" in name:
            prefix = name.replace(".parametrizations.weight.original0", "")
            weight = merge_weight_norm(index.load(f"{prefix}.parametrizations.weight.original0"),
                                       index.load(f"{prefix}.parametrizations.weight.original1"))
            yield renamed_name, convert_tensor_dtype(weight, outtype)
        else:
            yield renamed_name, convert_tensor_dtype(index.load(name), outtype)
        emitted.add(renamed_name)


def encoder_final_frame_rate(config: dict[str, Any]) -> float:
    frame_rate = float(config.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    for module_cfg in config.get("encoder_kwargs", []):
        if module_cfg.get("module_type") == "PatchedPretransform":
            frame_rate /= int(module_cfg["patch_size"])
    return frame_rate


def convert_one(model_dir: Path, outfile: Path, outtype: str, model_name: str,
                include_fn: Callable[[str], bool], arch: str) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    validate_config(config)
    index = SafeTensorsIndex(model_dir)
    encoder_block_map = build_transformer_block_index_map(list(config.get("encoder_kwargs", [])))
    decoder_block_map = build_transformer_block_index_map(list(config.get("decoder_kwargs", [])))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path=outfile, arch=arch)
    try:
        add_common_metadata(writer, arch, config, model_name)
        context_duration = float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION))
        if arch == ARCH_ENCODER:
            write_module_list_metadata(writer, arch, "encoder", list(config.get("encoder_kwargs", [])),
                                       float(config.get("sampling_rate", DEFAULT_SAMPLING_RATE)),
                                       context_duration, True)
        else:
            write_module_list_metadata(writer, arch, "decoder", list(config.get("decoder_kwargs", [])),
                                       encoder_final_frame_rate(config), context_duration, False)
        for name, tensor in iter_filtered_tensors(
                index, outtype, include_fn,
                lambda raw: map_split_tensor_name(raw, encoder_block_map, decoder_block_map)):
            writer.add_tensor(name, tensor)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(f"{arch}: wrote {outfile}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--encoder-outfile", type=Path, default=None)
    parser.add_argument("--decoder-outfile", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    encoder_outfile = args.encoder_outfile or model_dir / f"{model_dir.name}-encoder-{args.outtype}.gguf"
    decoder_outfile = args.decoder_outfile or model_dir / f"{model_dir.name}-decoder-{args.outtype}.gguf"
    convert_one(model_dir, encoder_outfile, args.outtype, f"{model_dir.name} Encoder",
                is_encoder_tensor, ARCH_ENCODER)
    convert_one(model_dir, decoder_outfile, args.outtype, f"{model_dir.name} Decoder",
                is_decoder_tensor, ARCH_DECODER)


if __name__ == "__main__":
    main()
