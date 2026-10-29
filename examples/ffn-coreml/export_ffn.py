#!/usr/bin/env python3
"""Export one dense SwiGLU block from GGUF for a Core ML offload experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 512])
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--layout", choices=["conv", "linear"], default="conv")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default="sigmoid")
    parser.add_argument("--output-scale", type=int, choices=[1, 16, 256], default=1)
    parser.add_argument("--no-reference-files", action="store_true")
    parser.add_argument("--channel-start", type=int, default=0)
    parser.add_argument("--channel-count", type=int)
    args = parser.parse_args()
    if args.layer < 0 or not 1 <= args.chunk <= 8192 or any(t < 1 or t > 512 for t in args.tokens):
        parser.error("require layer >= 0, chunk in 1..8192 and tokens in 1..512")
    args.output.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(str(args.gguf), "r")
    tensors = {t.name: t for t in reader.tensors}
    weights = {}
    metadata = {"source": str(args.gguf), "layer": args.layer, "chunk": args.chunk, "layout": args.layout,
                "activation": args.activation, "output_scale": args.output_scale,
                "reference_files": not args.no_reference_files, "weights": {}}
    for kind in ("gate", "up", "down"):
        name = f"blk.{args.layer}.ffn_{kind}.weight"
        tensor = tensors[name]
        weight = dequantize(tensor.data, tensor.tensor_type).astype(np.float16)
        if weight.ndim != 2 or not np.isfinite(weight).all():
            raise ValueError(f"Invalid weight: {name}")
        weights[kind] = weight
        metadata["weights"][kind] = {"name": name, "shape": list(weight.shape), "ggml_type": tensor.tensor_type.name,
                                     "sha256": hashlib.sha256(weight).hexdigest()}
    hidden = weights["up"].shape[1]
    intermediate = weights["up"].shape[0]
    if weights["gate"].shape != (intermediate, hidden) or weights["down"].shape != (hidden, intermediate):
        raise ValueError("Expected a dense SwiGLU block")
    count = args.channel_count if args.channel_count is not None else intermediate - args.channel_start
    if args.channel_start < 0 or count <= 0 or args.channel_start + count > intermediate:
        parser.error("invalid intermediate channel range")
    metadata["source_weights"] = metadata["weights"]
    metadata["weights"] = {}
    end = args.channel_start + count
    for kind in ("gate", "up", "down"):
        full = weights[kind]
        weight = (full[:, args.channel_start:end] if kind == "down" else full[args.channel_start:end]).copy()
        weights[kind] = weight
        if not args.no_reference_files:
            weight.tofile(args.output / f"{kind}.f16")
        metadata["weights"][kind] = dict(metadata["source_weights"][kind], shape=list(weight.shape),
                                          sha256=hashlib.sha256(weight).hexdigest())
    metadata.update(source_intermediate=intermediate, channel_start=args.channel_start, channel_count=count)
    intermediate = count
    metadata.update(hidden=hidden, intermediate=intermediate, tokens=args.tokens)
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for tokens in args.tokens:
        def silu(gate):
            if args.activation == "tanh":
                half = mb.mul(x=gate, y=0.5)
                return mb.mul(x=half, y=mb.add(x=mb.tanh(x=half), y=1.0))
            return mb.mul(x=gate, y=mb.sigmoid(x=gate))

        @mb.program(input_specs=[mb.TensorSpec(shape=(tokens, hidden))], opset_version=ct.target.macOS15)
        def program(x):
            if args.layout == "linear":
                up = mb.matmul(x=x, y=weights["up"], transpose_y=True)
                gate = mb.matmul(x=x, y=weights["gate"], transpose_y=True)
                activated = mb.mul(x=silu(gate), y=up)
                if args.output_scale != 1:
                    activated = mb.mul(x=activated, y=float(args.output_scale))
                return mb.matmul(x=activated, y=weights["down"], transpose_y=True, name="y")
            x = mb.expand_dims(x=mb.transpose(x=x, perm=[1, 0]), axes=[0, 2])
            result = None
            # Split intermediate channels for ANE compilation.
            for start in range(0, intermediate, args.chunk):
                end = min(start + args.chunk, intermediate)
                up = mb.conv(x=x, weight=weights["up"][start:end, :, None, None])
                gate = mb.conv(x=x, weight=weights["gate"][start:end, :, None, None])
                activated = mb.mul(x=silu(gate), y=up)
                if args.output_scale != 1:
                    activated = mb.mul(x=activated, y=float(args.output_scale))
                part = mb.conv(x=activated, weight=weights["down"][:, start:end, None, None].copy())
                result = part if result is None else mb.add(x=result, y=part)
            return mb.transpose(x=mb.squeeze(x=result, axes=[0, 2]), perm=[1, 0], name="y")

        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        model.user_defined_metadata["source_layer"] = str(args.layer)
        model.user_defined_metadata["output_scale"] = str(args.output_scale)
        model.user_defined_metadata["channel_start"] = str(args.channel_start)
        model.user_defined_metadata["channel_count"] = str(count)
        model.save(str(args.output / f"ffn-{tokens}.mlpackage"))
        print(f"Exported layer={args.layer} tokens={tokens}", flush=True)


if __name__ == "__main__":
    main()
