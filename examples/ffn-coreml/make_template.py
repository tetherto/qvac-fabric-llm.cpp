#!/usr/bin/env python3
"""Build a weight-free int8 Core ML template for the FFN sidecar.

The runtime (ffn-coreml-otf.h) writes weights/weight.bin for each layer from the GGUF at load time, so users do not
run a conversion. The program is the one export_ffn.py builds (1x1 conv layout, tanh SiLU, channel chunks) with
per-channel symmetric int8 weights, as linear_quantize_weights makes them. layout.txt lists every weight blob of the
program in file order: offset of its metadata, MILBlob data type, role, kind, chunk and shape.
"""

import argparse
from pathlib import Path
import shutil
import struct

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np


def build(path, channels, hidden, tokens, chunk, weights):
    def silu(gate):
        half = mb.mul(x=gate, y=0.5)
        return mb.mul(x=half, y=mb.add(x=mb.tanh(x=half), y=1.0))

    @mb.program(input_specs=[mb.TensorSpec(shape=(tokens, hidden))], opset_version=ct.target.macOS15)
    def program(x):
        x = mb.expand_dims(x=mb.transpose(x=x, perm=[1, 0]), axes=[0, 2])
        result = None
        for start in range(0, channels, chunk):
            end = min(start + chunk, channels)
            up = mb.conv(x=x, weight=weights["up"][start:end, :, None, None])
            gate = mb.conv(x=x, weight=weights["gate"][start:end, :, None, None])
            part = mb.conv(x=mb.mul(x=silu(gate), y=up), weight=weights["down"][:, start:end, None, None].copy())
            result = part if result is None else mb.add(x=result, y=part)
        return mb.transpose(x=mb.squeeze(x=result, axes=[0, 2]), perm=[1, 0], name="y")

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=4096)
    model = ct.optimize.coreml.linear_quantize_weights(model, ct.optimize.coreml.OptimizationConfig(global_config=config))
    model.save(str(path))
    return model


def quantize(w):
    # coremltools linear_symmetric per channel (axis 0), computed in FP16 like quantize_weight_by_dtype
    w = w.astype(np.float16)
    amax = np.abs(w.astype(np.float32)).max(axis=1, keepdims=True)
    scale = (amax * 2 / 254).astype(np.float16)
    q = np.rint((w.astype(np.float32) / scale.astype(np.float32)).astype(np.float16).astype(np.float32))
    return np.clip(q, -127, 127).astype(np.int8), scale.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    C, H = args.channels, args.hidden
    rng = np.random.default_rng(11259)
    weights = {k: (rng.standard_normal(s) * 0.02).astype(np.float16) for k, s in (("gate", (C, H)), ("up", (C, H)), ("down", (H, C)))}
    work = args.output.with_name(args.output.name + ".build")
    shutil.rmtree(work, ignore_errors=True)
    package = work / "ffn.mlpackage"
    model = build(package, C, H, args.tokens, args.chunk, weights)

    # identify every blob by its contents: int8 data and FP16 scales of a weight chunk, or the zero conv bias
    blob = (package / "Data/com.apple.CoreML/weights/weight.bin").read_bytes()
    count, version = struct.unpack_from("<II", blob, 0)
    if version != 2:
        raise ValueError(f"unexpected MILBlob version {version}")
    candidates = []
    for start in range(0, C, args.chunk):
        end = min(start + args.chunk, C)
        for role, w in (("up", weights["up"][start:end]), ("gate", (weights["gate"][start:end] * np.float16(0.5)).astype(np.float16)),
                        ("down", weights["down"][:, start:end])):
            q, s = quantize(w)
            candidates.append((role, "data", start, end, q))
            candidates.append((role, "scale", start, end, s))
    lines = []
    offset = 64
    for _ in range(count):
        sentinel, dtype, size, data = struct.unpack_from("<IIQQ", blob, offset)
        if sentinel != 0xDEADBEEF:
            raise ValueError(f"bad blob metadata at {offset}")
        raw = blob[data:data + size]
        match = [c for c in candidates if c[4].tobytes() == raw]
        if match:
            role, kind, start, end, arr = match[0]
            rows, cols = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (arr.shape[0], 1)
        elif dtype == 1 and not any(raw):
            role, kind, start, end, rows, cols = "bias", "zero", 0, 0, size // 2, 1
        else:
            raise ValueError(f"unidentified blob at {offset} (dtype {dtype}, {size} bytes)")
        lines.append(f"{offset} {dtype} {size} {role} {kind} {start} {end} {rows} {cols}")
        offset = data + size
        offset += (-offset) % 64
    if offset != len(blob):
        raise ValueError("weight.bin has trailing data")

    args.output.mkdir(parents=True)
    shutil.copy(package / "Manifest.json", args.output / "Manifest.json")
    shutil.copy(package / "Data/com.apple.CoreML/model.mlmodel", args.output / "model.mlmodel")
    (args.output / "layout.txt").write_text("\n".join(lines) + "\n")
    shutil.rmtree(work)
    print(f"template {args.output}: {count} blobs, weight.bin {len(blob)} bytes", flush=True)


if __name__ == "__main__":
    main()
