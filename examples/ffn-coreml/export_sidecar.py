#!/usr/bin/env python3
"""Export and compress the Core ML FFN sidecar for every layer of a preset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


TOOLS = Path(__file__).resolve().parent
TOKENS = 512
PACKAGE_NAMES = ("ffn-512", "ffn-512-1")
MANIFEST_NAMES = ("manifest.json", "manifest-1.json")


def read_preset(path: Path) -> dict:
    preset = json.loads(path.read_text())
    layers = preset.get("layers")
    if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers) or layers != sorted(layers):
        raise ValueError("layers must be a sorted list of distinct layer ids")
    if any(not isinstance(layer, int) or not 0 <= layer < preset.get("model_layers", 0) for layer in layers):
        raise ValueError("layer ids must be within model_layers")
    if preset.get("tokens") != TOKENS:
        raise ValueError(f"tokens must be {TOKENS}")
    intermediate, start, count = preset.get("intermediate"), preset.get("channel_start"), preset.get("channel_count")
    if not all(isinstance(v, int) for v in (intermediate, start, count)) or start <= 0 or count <= 0 or start + count != intermediate:
        raise ValueError("channel_start + channel_count must equal intermediate")
    if start % 256:
        raise ValueError("channel_start must be a multiple of 256")
    parts = preset.get("ane_partitions", 1)
    if parts not in (1, 2):
        raise ValueError("ane_partitions must be 1 or 2")
    if count % (parts * 256):
        raise ValueError("each Core ML partition must be a multiple of 256 channels")
    if not set(preset.get("fp16_layers", [])) <= set(layers):
        raise ValueError("fp16_layers must be selected layers")
    if preset.get("palette_bits") not in (6, 8) or preset.get("palette_group_size", 0) not in (0, 16, 32):
        raise ValueError("unsupported palette settings")
    if preset.get("activation") not in ("sigmoid", "tanh") or preset.get("output_scale") not in (1, 16, 256):
        raise ValueError("unsupported activation or output scale")
    return preset


def partitions(preset: dict) -> list:
    parts = preset.get("ane_partitions", 1)
    size = preset["channel_count"] // parts
    return [(PACKAGE_NAMES[i], MANIFEST_NAMES[i], preset["channel_start"] + i * size, size) for i in range(parts)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def run(script: str, arguments: list) -> None:
    # coremltools keeps a k-means worker pool per process, so each step gets a fresh one
    subprocess.run([sys.executable, str(TOOLS / script), *arguments], check=True)


def export(gguf: Path, preset_path: Path, destination: Path, workers: int, verify: bool) -> None:
    preset = read_preset(preset_path)
    if destination.exists():
        raise ValueError("destination already exists")
    model_hash = sha256(gguf) if verify else None
    if model_hash and preset.get("model_sha256") and model_hash != preset["model_sha256"]:
        raise ValueError("GGUF SHA256 does not match the preset")
    import coremltools as ct
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        packages = []
        for layer in preset["layers"]:
            output = staging / f"layer{layer}"
            output.mkdir()
            for name, manifest, start, count in partitions(preset):
                fp16 = staging / "fp16"
                run("export_ffn.py", [str(gguf), str(fp16), "--layer", str(layer), "--tokens", str(TOKENS), "--chunk", str(preset["chunk"]),
                                      "--activation", preset["activation"], "--output-scale", str(preset["output_scale"]),
                                      "--channel-start", str(start), "--channel-count", str(count), "--no-reference-files"])
                source = fp16
                if layer not in preset.get("fp16_layers", []):
                    source = staging / "palette"
                    run("palettize_ffn.py", [str(fp16), str(source), "--bits", str(preset["palette_bits"]), "--tokens", str(TOKENS),
                                             "--workers", str(workers), "--group-size", str(preset.get("palette_group_size", 0)), "--no-reference-files"])
                (source / f"ffn-{TOKENS}.mlpackage").rename(output / f"{name}.mlpackage")
                (source / "manifest.json").rename(output / manifest)
                packages.append(dict(layer=layer, package=f"layer{layer}/{name}.mlpackage", channel_start=start, channel_count=count,
                                     manifest_sha256=sha256(output / manifest)))
                shutil.rmtree(fp16)
                shutil.rmtree(staging / "palette", ignore_errors=True)
        (staging / "layers.txt").write_text("".join(f"{layer}\n" for layer in preset["layers"]))
        record = dict(source=gguf.name, source_sha256=model_hash, preset=preset, coremltools=ct.__version__,
                      tools={name: sha256(TOOLS / name) for name in ("export_sidecar.py", "export_ffn.py", "palettize_ffn.py")},
                      packages=packages)
        (staging / "export.json").write_text(json.dumps(record, indent=2) + "\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="source GGUF model")
    parser.add_argument("preset", type=Path, help="preset JSON, for example configs/qwen3.8-27b-64-dual.json")
    parser.add_argument("destination", type=Path, help="new output directory")
    parser.add_argument("--workers", type=int, default=8, help="k-means workers for palettization")
    parser.add_argument("--no-verify", action="store_true", help="skip the GGUF SHA256 check")
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error("workers must be in 1..20")
    export(args.gguf, args.preset, args.destination, args.workers, not args.no_verify)


if __name__ == "__main__":
    main()
