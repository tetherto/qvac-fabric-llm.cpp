#!/usr/bin/env python3
"""Palettize one exported FFN model and retain the original accuracy reference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import coremltools as ct


def operation_counts(model):
    counts = Counter()

    def visit(block):
        for op in block.operations:
            counts[op.type] += 1
            for nested in op.blocks:
                visit(nested)

    for function in model.get_spec().mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit(block)
    return counts


def weight_bytes(package):
    return sum(p.stat().st_size for p in (package / "Data/com.apple.CoreML/weights").rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=0, help="0 selects one palette per tensor")
    parser.add_argument("--no-reference-files", action="store_true")
    parser.add_argument("--skip-ops", nargs="*", default=[], help="Keep the named weight operations in FP16")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 512 or not 1 <= args.workers <= 20 or args.group_size not in (0, 16, 32):
        parser.error("require tokens in 1..512, workers in 1..20 and group-size in 0,16,32")
    if args.destination.exists():
        parser.error("destination already exists")
    package = args.source / f"ffn-{args.tokens}.mlpackage"
    if not package.is_dir():
        parser.error(f"missing source package: {package}")
    manifest = json.loads((args.source / "manifest.json").read_text())
    for kind in (() if args.no_reference_files else ("gate", "up", "down")):
        if not (args.source / f"{kind}.f16").is_file():
            parser.error(f"missing reference weights: {kind}")
    args.destination.mkdir(parents=True)
    for kind in (() if args.no_reference_files else ("gate", "up", "down")):
        (args.destination / f"{kind}.f16").symlink_to((args.source / f"{kind}.f16").resolve())
    model = ct.models.MLModel(str(package), skip_model_load=True)
    before = operation_counts(model)
    weight_ops = []
    for function in model.get_spec().mlProgram.functions.values():
        for block in function.block_specializations.values():
            for op in block.operations:
                if op.type in ("conv", "linear", "matmul"):
                    weight_ops.append(op.attributes["name"].immediateValue.tensor.strings.values[0])
    if len(set(args.skip_ops)) != len(args.skip_ops) or not set(args.skip_ops).issubset(weight_ops):
        parser.error("skip-ops must name distinct weight operations in the source model")
    config = ct.optimize.coreml.OpPalettizerConfig(
        mode="kmeans", nbits=args.bits, num_kmeans_workers=args.workers, weight_threshold=4096,
        granularity="per_grouped_channel" if args.group_size else "per_tensor", group_size=args.group_size or 32)
    print(f"Palettizing bits={args.bits} tokens={args.tokens} group_size={args.group_size}", flush=True)
    start = time.monotonic()
    compressed = ct.optimize.coreml.palettize_weights(model, ct.optimize.coreml.OptimizationConfig(global_config=config, op_name_configs={name: None for name in args.skip_ops}))
    compressed.user_defined_metadata.update(model.user_defined_metadata)
    after = operation_counts(compressed)
    expected = sum(before[k] for k in ("conv", "linear", "matmul")) - len(args.skip_ops)
    if expected == 0 or after["constexpr_lut_to_dense"] != expected:
        raise ValueError(f"Expected {expected} palette operations; found {after}")
    output = args.destination / package.name
    compressed.save(str(output))
    record = {
        "bits": args.bits, "mode": "kmeans", "group_size": args.group_size, "tokens": args.tokens,
        "fp16_operations": args.skip_ops,
        "coremltools": ct.__version__, "source_package": str(package),
        "source_model_sha256": hashlib.sha256((package / "Data/com.apple.CoreML/model.mlmodel").read_bytes()).hexdigest(),
        "source_weight_bytes": weight_bytes(package), "compressed_weight_bytes": weight_bytes(output),
        "palette_operations": after["constexpr_lut_to_dense"], "compression_seconds": time.monotonic() - start,
        "operations": dict(after),
    }
    manifest["compression"] = record
    manifest["reference_files"] = not args.no_reference_files
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
