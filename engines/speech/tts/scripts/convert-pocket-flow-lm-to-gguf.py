#!/usr/bin/env python3
"""Export the Pocket TTS FlowLM stage (not a complete speech engine).

Consumes local Kyutai model.safetensors + YAML config. No model downloads or
pickle deserialization. F32 is the reference tier; F16 stores only matrices
in half precision, retaining normalization, frequency and bias tensors as F32.
Mimi is deliberately excluded: this artifact produces continuous latents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import gguf
import numpy as np
import yaml
from safetensors import safe_open

ARCH = "pocket-tts-flow-lm"
VERSION = 1


def dimensions(config: dict) -> dict:
    lm = config["flow_lm"]
    transformer, flow = lm["transformer"], lm["flow"]
    if transformer.get("context") is not None or transformer.get("layer_scale") is not None:
        raise ValueError("Only full-causal FlowLM without layer scale is supported")
    if flow.get("type", "lsd") != "lsd":
        raise ValueError("Only the released two-time-condition LSD flow is supported")
    dims = {
        "dim": transformer["d_model"],
        "heads": transformer["num_heads"],
        "layers": transformer["num_layers"],
        "ff_dim": int(transformer["d_model"] * transformer["hidden_scale"]),
        "latent_dim": config["mimi"]["quantizer"]["dimension"],
        "flow_dim": flow["dim"],
        "flow_depth": flow["depth"],
        "vocab_size": lm["lookup_table"]["n_bins"],
    }
    limits = {"dim": 4096, "heads": 128, "layers": 48, "ff_dim": 16384,
              "latent_dim": 512, "flow_dim": 4096, "flow_depth": 32, "vocab_size": 65536}
    for key, value in dims.items():
        if type(value) is not int or not 0 < value <= limits[key]:
            raise ValueError(f"Invalid {key}: {value!r}")
    if dims["dim"] % dims["heads"] or (dims["dim"] // dims["heads"]) % 2:
        raise ValueError("Attention requires an even head dimension")
    if dims["flow_dim"] < 2:
        raise ValueError("Flow time normalization needs at least two channels")
    return dims


def expected_shapes(d: dict, bos_before_voice: bool) -> dict:
    # Shapes are PyTorch order; GGUF reverses dimensions, never tensor bytes.
    result = {}

    def linear(name, inp, out, bias=True):
        result[name + ".weight"] = (out, inp)
        if bias:
            result[name + ".bias"] = (out,)

    def norm(name, width):
        result[name + ".weight"] = (width,)
        result[name + ".bias"] = (width,)

    width, latent, flow = d["dim"], d["latent_dim"], d["flow_dim"]
    result["conditioner.embed.weight"] = (d["vocab_size"] + 1, width)
    result["bos_emb"] = result["emb_mean"] = result["emb_std"] = (latent,)
    if bos_before_voice:
        result["bos_before_voice"] = (1, 1, width)
    linear("input_linear", latent, width, False)
    norm("out_norm", width)
    linear("out_eos", width, 1)
    for i in range(d["layers"]):
        p = f"transformer.layers.{i}"
        linear(p + ".self_attn.in_proj", width, 3 * width, False)
        linear(p + ".self_attn.out_proj", width, width, False)
        norm(p + ".norm1", width)
        norm(p + ".norm2", width)
        linear(p + ".linear1", width, d["ff_dim"], False)
        linear(p + ".linear2", d["ff_dim"], width, False)
    p = "flow_net."
    linear(p + "input_proj", latent, flow)
    linear(p + "cond_embed", width, flow)
    for i in range(2):
        t = p + f"time_embed.{i}."
        result[t + "freqs"] = (128,)
        linear(t + "mlp.0", 256, flow)
        linear(t + "mlp.2", flow, flow)
        result[t + "mlp.3.alpha"] = (flow,)
    for i in range(d["flow_depth"]):
        r = p + f"res_blocks.{i}."
        norm(r + "in_ln", flow)
        linear(r + "mlp.0", flow, flow)
        linear(r + "mlp.2", flow, flow)
        linear(r + "adaLN_modulation.1", flow, 3 * flow)
    linear(p + "final_layer.linear", flow, latent)
    linear(p + "final_layer.adaLN_modulation.1", flow, 2 * flow)
    return result


def convert(weights: Path, config_path: Path, output: Path, dtype="f32"):
    if output.resolve() in (weights.resolve(), config_path.resolve()):
        raise ValueError("Output must not replace the source weights or configuration")
    if dtype not in ("f32", "f16"):
        raise ValueError("Supported storage types are f32 and f16")
    config = yaml.safe_load(config_path.read_text())
    d = dimensions(config)
    bos = config["flow_lm"].get("insert_bos_before_voice", False)
    if type(bos) is not bool:
        raise ValueError("insert_bos_before_voice must be boolean")
    rope = float(config["flow_lm"]["transformer"].get("max_period", 10000))
    if not np.isfinite(rope) or rope <= 0:
        raise ValueError("max_period must be finite and positive")
    # Released bundles mix BF16/F32; NumPy's safetensors reader cannot decode
    # BF16. Materialize only FlowLM tensors through torch, never the Mimi half.
    tensors = {}
    with safe_open(str(weights), framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("flow_lm."):
                value = source.get_tensor(name)
                if not value.is_floating_point():
                    raise ValueError(f"Non-floating tensor: {name}")
                tensors[name.removeprefix("flow_lm.")] = value.float().numpy()
    expected = expected_shapes(d, bos)
    # Projection is present in cloning bundles, absent in some prepared-voice
    # bundles. Preserve it for a future voice frontend, with explicit shape.
    if "speaker_proj_weight" in tensors:
        speaker_dim = config["mimi"].get("inner_dim") or config["mimi"]["seanet"]["dimension"]
        if speaker_dim != d["latent_dim"]:
            raise ValueError("Unsupported speaker projection input dimension")
        expected["speaker_proj_weight"] = (d["dim"], speaker_dim)
    missing, extra = expected.keys() - tensors.keys(), tensors.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"FlowLM tensor mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    for name, shape in expected.items():
        a = tensors[name]
        if a.shape != shape or a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(f"Invalid tensor {name}: expected finite float {shape}, got {a.shape}/{a.dtype}")
    if np.any(tensors["emb_std"] <= 0):
        raise ValueError("emb_std must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after all validation and writes succeed; preserve an existing
    # artifact if a conversion fails or is interrupted before os.replace().
    fd, tmp = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(fd)
    writer = None
    try:
        writer = gguf.GGUFWriter(tmp, ARCH)
        writer.add_uint32("pocket.schema_version", VERSION)
        writer.add_string("pocket.config_json", json.dumps(config, sort_keys=True))
        with weights.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        writer.add_string("pocket.source_sha256", digest)
        writer.add_string("pocket.flow_type", "lsd")
        writer.add_bool("pocket.bos_before_voice", bos)
        writer.add_float32("pocket.rope_base", rope)
        for name, value in d.items():
            writer.add_uint32("pocket." + name, value)
        for name, array in tensors.items():
            # BOS is saved as [1,1,D] upstream; present a vector to native code.
            if name == "bos_before_voice":
                array = array.reshape(-1)
            target = np.float16 if dtype == "f16" and array.ndim == 2 else np.float32
            converted = np.ascontiguousarray(array, dtype=target)
            if not np.isfinite(converted).all():
                raise ValueError(f"{name} overflows {dtype}")
            writer.add_tensor(name, converted)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        os.replace(tmp, output)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = p.parse_args()
    convert(args.weights, args.config, args.output, args.dtype)
