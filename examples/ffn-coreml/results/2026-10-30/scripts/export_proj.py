#!/usr/bin/env python3
"""Export a row slice of one DeltaNet input projection ([attn_qkv; attn_gate]) as a Core ML package."""
import argparse
import json
from pathlib import Path
import sys

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

sys.path.insert(0, "/Users/ishan/qwen38-accuracy-20261029/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    tensors = {t.name: t for t in GGUFReader(str(args.gguf), "r").tensors}
    parts = []
    for kind in ("attn_qkv", "attn_gate"):
        t = tensors[f"blk.{args.layer}.{kind}.weight"]
        parts.append(dequantize(t.data, t.tensor_type).astype(np.float16))
    weight = np.concatenate(parts)[args.row_start:args.row_start + args.row_count].copy()
    rows, hidden = weight.shape
    assert rows == args.row_count and hidden == 5120 and np.isfinite(weight).all()
    args.output.mkdir(parents=True, exist_ok=True)

    @mb.program(input_specs=[mb.TensorSpec(shape=(512, hidden))], opset_version=ct.target.macOS15)
    def program(x):
        x = mb.expand_dims(x=mb.transpose(x=x, perm=[1, 0]), axes=[0, 2])
        outputs = [mb.conv(x=x, weight=weight[s:s + args.chunk, :, None, None]) for s in range(0, rows, args.chunk)]
        y = outputs[0] if len(outputs) == 1 else mb.concat(values=outputs, axis=1)
        return mb.transpose(x=mb.squeeze(x=y, axes=[0, 2]), perm=[1, 0], name="y")

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    model.user_defined_metadata.update(source_layer=str(args.layer), row_start=str(args.row_start), row_count=str(rows))
    model.save(str(args.output / "ffn-512.mlpackage"))
    (args.output / "manifest.json").write_text(json.dumps(dict(layer=args.layer, row_start=args.row_start, row_count=rows,
                                                               source_types=[tensors[f"blk.{args.layer}.{k}.weight"].tensor_type.name for k in ("attn_qkv", "attn_gate")]), indent=2) + "\n")
    print(f"Exported layer={args.layer} rows={args.row_start}+{rows}", flush=True)


if __name__ == "__main__":
    main()
