"""Expand the pinned Core ML package's row-independent pooling head to 16 texts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import coremltools as ct


SOURCE_SHA256 = "0df9aca02807cf490db6d4f1c68607a8c7f486ad23b376d1968fd3be5eb14677"
MODEL_PATH = Path("Data/com.apple.CoreML/model.mlmodel")


def expand(source: Path, destination: Path) -> None:
    package = source / "EmbeddingGemma2Text.mlpackage"
    model_path = package / MODEL_PATH
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Expected the pinned FluidInference model revision; refusing an unknown graph")
    if destination.exists():
        raise ValueError("Destination already exists; the original and previous variants must be preserved")
    table = source / "embeddings.bf16"
    if table.stat().st_size != 262144 * 512 * 2:
        raise ValueError("Invalid token table")
    spec = ct.utils.load_spec(str(model_path))
    unchanged = {name: function.SerializeToString() for name, function in spec.mlProgram.functions.items() if name != "pack_256"}
    function = spec.mlProgram.functions["pack_256"]
    pool = next(value for value in function.inputs if value.name == "pool")
    shape = pool.type.tensorType.dimensions
    if len(shape) != 2 or tuple(d.constant.size for d in shape) != (8, 256):
        raise ValueError("Unexpected packed input shape")
    shape[0].constant.size = 16
    changed = 0
    # These operations preserve the first (text) axis in this pinned graph.
    allowed = {"matmul", "linear", "abs", "reduce_max", "maximum", "real_div", "mul", "reduce_sum", "add", "rsqrt", "cast"}
    for block in function.block_specializations.values():
        dependent = {"pool"}
        for operation in block.operations:
            inputs = {binding.name for argument in operation.inputs.values() for binding in argument.arguments if binding.HasField("name")}
            if not inputs.intersection(dependent):
                continue
            if operation.type not in allowed or operation.blocks:
                raise ValueError("Unexpected operation downstream of pooling: " + operation.type)
            for output in operation.outputs:
                dimensions = output.type.tensorType.dimensions
                if len(dimensions) != 2 or not dimensions[0].HasField("constant") or dimensions[0].constant.size != 8:
                    raise ValueError("Pooling output does not preserve the text axis: " + output.name)
                dimensions[0].constant.size = 16
                dependent.add(output.name)
                changed += 1
        if not set(block.outputs).issubset(dependent):
            raise ValueError("Packed output is not derived from the pooling head")
    if changed != 12:
        raise ValueError(f"Unexpected number of pooling outputs: {changed}")
    description = next(item for item in spec.description.functions if item.name == "pack_256")
    pool_description = next(item for item in description.input if item.name == "pool")
    embedding_description = next(item for item in description.output if item.name == "embedding")
    if tuple(pool_description.type.multiArrayType.shape) != (8, 256) or tuple(embedding_description.type.multiArrayType.shape) != (8, 768):
        raise ValueError("Unexpected function feature descriptions")
    pool_description.type.multiArrayType.shape[0] = 16
    embedding_description.type.multiArrayType.shape[0] = 16
    for name, data in unchanged.items():
        if spec.mlProgram.functions[name].SerializeToString() != data:
            raise ValueError("An unrelated function changed")
    destination.mkdir(parents=True)
    variant = destination / package.name
    shutil.copytree(package, variant)
    ct.utils.save_spec(spec, str(variant / MODEL_PATH))
    (destination / "embeddings.bf16").symlink_to(table.resolve())
    record = {
        "source_model_sha256": digest,
        "variant_model_sha256": hashlib.sha256((variant / MODEL_PATH).read_bytes()).hexdigest(),
        "pack_slots": 16,
        "pooling_outputs_resized": changed,
        "coremltools": ct.__version__,
        "weights_modified": False,
    }
    (destination / "variant.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    expand(arguments.source, arguments.destination)
