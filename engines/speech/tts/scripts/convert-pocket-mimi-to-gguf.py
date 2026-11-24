#!/usr/bin/env python3
"""Convert Pocket TTS's Mimi codec to Fabric GGUF (local files, no downloads)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import gguf
import numpy as np
from safetensors import safe_open
import yaml
from pocket_tts.models.mimi import build_mimi
from pocket_tts.utils.config import MimiConfig


def validate_native_config(cfg: MimiConfig):
    expected = {
        "sample_rate": 24000, "frame_rate": 12.5, "channels": 1, "inner_dim": 32, "outer_dim": 512,
        "seanet": dict(dimension=512, channels=1, n_filters=64, n_residual_layers=1,
                       ratios=[6, 5, 4], kernel_size=7, residual_kernel_size=3,
                       last_kernel_size=3, dilation_base=2, pad_mode="constant", compress=2),
        "transformer": dict(d_model=512, num_heads=8, num_layers=2, context=250,
                            dim_feedforward=2048, input_dimension=512,
                            output_dimensions=[512], max_period=10000),
        "quantizer": dict(dimension=32, output_dimension=512),
    }
    def check(actual, wanted, prefix="mimi"):
        for key, value in wanted.items():
            if isinstance(value, dict):
                check(actual[key], value, prefix + "." + key)
            elif actual[key] != value:
                raise ValueError(f"Unsupported native codec configuration: {prefix}.{key}={actual[key]!r}")
    check(cfg.model_dump(mode="json"), expected)
    if not np.isfinite(cfg.transformer.layer_scale):
        raise ValueError("Non-finite layer scale")


def convert(weights: Path, config: Path, output: Path):
    if output.resolve() in (weights.resolve(), config.resolve()):
        raise ValueError("Output must differ from source files")
    cfg = MimiConfig.model_validate(yaml.safe_load(config.read_text())["mimi"])
    validate_native_config(cfg)
    # Validate against the actual upstream module shapes before rewriting any
    # layout. Conversion-only PyTorch is not a native runtime dependency.
    model = build_mimi(cfg)
    expected = model.state_dict()
    data = {}
    with safe_open(str(weights), framework="pt") as source:
        names = {k.removeprefix("mimi.") for k in source.keys() if k.startswith("mimi.")}
        if names != expected.keys():
            raise ValueError(f"Mimi tensor mismatch: missing={expected.keys() - names}, extra={names - expected.keys()}")
        for name, tensor in expected.items():
            x = source.get_tensor("mimi." + name)
            if x.shape != tensor.shape or not x.is_floating_point():
                raise ValueError(f"Invalid Mimi tensor {name}: {x.shape}, expected {tensor.shape}")
            x = x.float().numpy()
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite Mimi tensor {name}")
            if x.ndim == 3:
                if "upsample.convtr" in name:
                    # Depthwise transposed convolution: [C,1,K] -> [K,C].
                    x = x[:, 0, :].T
                elif ".convtr.weight" in name:
                    # [IC,OC,K] -> [K,OC,IC], phase grouped at runtime.
                    x = x.transpose(2, 1, 0)
                elif name == "quantizer.output_proj.weight":
                    x = x[:, :, 0]
                else:
                    # Conv1d: [OC,IC,K] -> [K,OC,IC], one matmul per tap.
                    x = x.transpose(2, 0, 1)
            name = name.replace("decoder_transformer.transformer.layers.", "dec.layers.")
            name = name.replace("encoder_transformer.transformer.layers.", "enc.layers.")
            if len(name.encode()) >= 64:
                raise ValueError(f"GGML tensor name is too long: {name}")
            data[name] = np.ascontiguousarray(x)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(fd)
    writer = None
    try:
        writer = gguf.GGUFWriter(tmp, "pocket-tts-mimi")
        writer.add_uint32("pocket.schema_version", 1)
        writer.add_string("pocket.mimi_config", cfg.model_dump_json())
        with weights.open("rb") as f:
            writer.add_string("pocket.source_sha256", hashlib.file_digest(f, "sha256").hexdigest())
        for name, x in data.items():
            writer.add_tensor(name, x)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        os.replace(tmp, output)
    finally:
        if writer:
            writer.close()
        Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    convert(a.weights, a.config, a.output)
