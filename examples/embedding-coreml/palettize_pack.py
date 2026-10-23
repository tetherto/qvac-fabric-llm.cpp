"""Compress packed-function matrix weights; preserve the six FP16 single-text functions."""

import argparse
import hashlib
import json
from pathlib import Path

import coremltools as ct


def palettize(source: Path, destination: Path, bits: int) -> None:
    if destination.exists():
        raise ValueError("Destination already exists; refusing to overwrite a previous experiment")
    package = source / "EmbeddingGemma2Text.mlpackage"
    spec = ct.utils.load_spec(str(package))
    description = next(item for item in spec.description.functions if item.name == "pack_256")
    pack = type(spec.mlProgram.functions["pack_256"])()
    pack.CopyFrom(spec.mlProgram.functions["pack_256"])
    del spec.description.input[:]
    del spec.description.output[:]
    spec.description.input.extend(description.input)
    spec.description.output.extend(description.output)
    del spec.description.functions[:]
    spec.description.defaultFunctionName = ""
    for name in list(spec.mlProgram.functions):
        del spec.mlProgram.functions[name]
    spec.mlProgram.functions["main"].CopyFrom(pack)
    destination.mkdir(parents=True)
    work = destination / "conversion"
    work.mkdir()
    model = ct.models.MLModel(spec, weights_dir=str(package / "Data/com.apple.CoreML/weights"),
                             compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    model.save(str(work / "packed-fp16.mlpackage"))
    metadata = ct.optimize.coreml.get_weights_metadata(model, weight_threshold=4096)
    config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=bits,
                                                  num_kmeans_workers=8, weight_threshold=4096)
    selected = {name: config for name, value in metadata.items() if "weight" in name and value.val.ndim == 2}
    if len(selected) < 100:
        raise ValueError("Unexpected matrix-weight metadata")
    print(f"Compressing {len(selected)} matrix tensors to {bits}-bit palettes", flush=True)
    optimization = ct.optimize.coreml.OptimizationConfig(op_name_configs=selected)
    compressed = ct.optimize.coreml.palettize_weights(model, optimization)
    compressed_path = work / f"packed-e{bits}.mlpackage"
    compressed.save(str(compressed_path))
    functions = ct.utils.MultiFunctionDescriptor()
    for name in ct.utils.load_spec(str(package)).mlProgram.functions:
        if name != "pack_256":
            functions.add_function(str(package), name, name)
    functions.add_function(str(compressed_path), "main", "pack_256")
    functions.default_function_name = "embed_32"
    output = destination / package.name
    ct.utils.save_multifunction(functions, str(output))
    (destination / "embeddings.bf16").symlink_to((source / "embeddings.bf16").resolve())
    record = {
        "source_model_sha256": hashlib.sha256((package / "Data/com.apple.CoreML/model.mlmodel").read_bytes()).hexdigest(),
        "variant_model_sha256": hashlib.sha256((output / "Data/com.apple.CoreML/model.mlmodel").read_bytes()).hexdigest(),
        "palette_bits": bits,
        "matrix_tensors_selected": len(selected),
        "compressed_function": "pack_256",
        "single_functions": "original FP16",
        "coremltools": ct.__version__,
    }
    (destination / "variant.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bits", type=int, choices=(6, 8), default=8)
    arguments = parser.parse_args()
    palettize(arguments.source, arguments.destination, arguments.bits)
