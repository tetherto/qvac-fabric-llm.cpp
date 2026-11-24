#!/usr/bin/env python3
"""Merge validated full-EU route exports into a runtime shared-weight sidecar.

Export en/other/gate with export-encoder-coreml-ane.py first, using identical
window capacities and precision. Palettize each source before merging.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import coremltools as ct

ROUTES = ("en", "other", "gate")
VARIANT = "transcribepsy-250m-eu"
INPUTS = {"mel", "mel_mask", "pad_mask", "key_bias"}
MIN_GATE_MEL_FRAMES = 2001
SHARED_VERSION = "1"
SUBSAMPLING_FACTOR = 8


def feature_signature(features):
    return {feature.name: (tuple(feature.type.multiArrayType.shape),
                           feature.type.multiArrayType.dataType)
            for feature in features}


def validate_shapes(path, inputs, outputs):
    mel = inputs["mel"][0]
    out = outputs["encoder_out"][0]
    if (len(mel) != 4 or mel[0] != 1 or mel[1] <= 0 or mel[2] != 1 or
            mel[3] < MIN_GATE_MEL_FRAMES or len(out) != 4 or out[1] <= 0):
        raise ValueError(f"{path}: shared routes need fixed NCHW shapes and at least {MIN_GATE_MEL_FRAMES} mel frames")
    rows = (mel[3] + SUBSAMPLING_FACTOR - 1) // SUBSAMPLING_FACTOR
    expected = {"mel_mask": (1, 1, 1, mel[3]), "pad_mask": (1, 1, 1, rows),
                "key_bias": (1, rows, 1, 1)}
    if any(inputs[name][0] != shape for name, shape in expected.items()) or out != (1, out[1], 1, rows):
        raise ValueError(f"{path}: mask and output shapes must match the encoder capacity")
    fp16 = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
    if any(dtype != fp16 for _, dtype in (*inputs.values(), *outputs.values())):
        raise ValueError(f"{path}: shared route inputs and outputs must use FP16")


def read_route(path, route):
    model = ct.models.MLModel(str(path), skip_model_load=True)
    spec = model.get_spec()
    metadata = dict(model.user_defined_metadata)
    if (metadata.get("parakeet.model_variant") != VARIANT or
            metadata.get("parakeet.eu.route") != route or
            metadata.get("parakeet.encoder.precision") != "fp16"):
        raise ValueError(f"{path}: expected a full-EU FP16 {route} export")
    layers = int(metadata.get("parakeet.encoder.n_layers", "0"))
    inputs = feature_signature(spec.description.input)
    outputs = feature_signature(spec.description.output)
    if layers <= 0 or set(inputs) != INPUTS or set(outputs) != {"encoder_out"}:
        raise ValueError(f"{path}: invalid route interface or encoder depth")
    validate_shapes(path, inputs, outputs)
    return layers, inputs, outputs


def merge_routes(sources, output, compile_dir=None):
    routes = {route: read_route(sources[route], route) for route in ROUTES}
    layers, inputs, outputs = routes["en"]
    if routes["other"] != routes["en"] or routes["gate"][1:] != (inputs, outputs):
        raise ValueError("Route capacities, precision and output dimensions must match")
    if routes["gate"][0] > layers:
        raise ValueError("Gate depth must not exceed encoder depth")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if compile_dir is not None and (compile_dir / (output.stem + ".mlmodelc")).exists():
        raise ValueError("Compiled output already exists")
    descriptor = ct.utils.MultiFunctionDescriptor()
    for route in ROUTES:
        descriptor.add_function(str(sources[route]), "main", route)
    descriptor.default_function_name = "en"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        merged = Path(temporary) / "merged.mlpackage"
        ct.utils.save_multifunction(descriptor, str(merged))
        model = ct.models.MLModel(str(merged), skip_model_load=True)
        metadata = model.user_defined_metadata
        metadata["parakeet.model_variant"] = VARIANT
        metadata["parakeet.eu.shared_version"] = SHARED_VERSION
        metadata["parakeet.encoder.precision"] = "fp16"
        for route in ROUTES:
            metadata[f"parakeet.eu.{route}.n_layers"] = str(routes[route][0])
        model.save(str(output))
    if compile_dir is not None:
        compile_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["xcrun", "coremlc", "compile", str(output), str(compile_dir)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for route in ROUTES:
        parser.add_argument(f"--{route}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-dir", type=Path)
    args = parser.parse_args()
    if args.out.suffix != ".mlpackage":
        parser.error("--out must end in .mlpackage")
    merge_routes({route: getattr(args, route) for route in ROUTES}, args.out, args.compile_dir)


if __name__ == "__main__":
    main()
