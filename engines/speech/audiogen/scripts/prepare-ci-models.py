"""Stage pinned, checksum-verified real models for audiogen CI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ENGINE = Path(__file__).resolve().parents[1]
MANIFEST = ENGINE / "test" / "ci-models.json"
CHUNK_BYTES = 8 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 120
COMMON_ACE_FILES = ("Qwen3-Embedding-0.6B-Q8_0.gguf", "acestep-5Hz-lm-0.6B-Q8_0.gguf", "vae-BF16.gguf")
MINIMAX_ROLES = ("lm", "synth")
CONVERTED_PREFIX = "converted-"
SHA256_HEX_LENGTH = 64
SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination, expected_hash):
    if destination.is_file() and sha256(destination) == expected_hash:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        print("Downloading " + destination.name, flush=True)
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT_SECONDS) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, CHUNK_BYTES)
        if sha256(temporary) != expected_hash:
            raise ValueError("Checksum mismatch: " + destination.name)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_group(group, directory):
    for filename, checksum in group["files"].items():
        url = "https://huggingface.co/{repository}/resolve/{revision}/{filename}".format(**group, filename=filename)
        download_file(url, directory / filename, checksum)


def link_base_models(turbo, base):
    for name in COMMON_ACE_FILES:
        source = turbo / name
        destination = base / name
        if destination.exists():
            if os.path.samefile(source, destination) or sha256(source) == sha256(destination):
                continue
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)


def conversion_key():
    inputs = (ENGINE / "scripts" / "convert-minimax-music3-to-gguf.py",
              ENGINE / "src" / "acestep" / "quantize_gguf.h",
              ENGINE / "src" / "acestep" / "quantize_policy.h", MANIFEST)
    return hashlib.sha256("".join(sha256(path) for path in inputs).encode()).hexdigest()


def prune_converted_models(cache, current):
    root = cache.resolve()
    if current.resolve().parent != root:
        raise ValueError("Converted models must be a direct child of the cache directory")
    for directory in root.glob(CONVERTED_PREFIX + "*"):
        key = directory.name[len(CONVERTED_PREFIX):]
        if directory == current or directory.is_symlink() or not directory.is_dir():
            continue
        if len(key) != SHA256_HEX_LENGTH or not set(key).issubset(SHA256_HEX_DIGITS):
            continue
        target = directory.resolve()
        if target.parent != root:
            raise ValueError("Converted models resolved outside the cache directory")
        shutil.rmtree(target)


def convert_minimax(cache, build):
    output = cache / (CONVERTED_PREFIX + conversion_key())
    prune_converted_models(cache, output)
    native, quantized = output / "f16", output / "q4"
    converter = ENGINE / "scripts" / "convert-minimax-music3-to-gguf.py"
    tokenizer = cache / "minimax-tokenizer" / "qwen_7B" / "qwen3-8B-tokenizer-music" / "tokenizer.json"
    subprocess.run([sys.executable, str(converter), "--src", str(cache / "minimax-source"),
                    "--tokenizer", str(tokenizer), "--out", str(native), "--quant", "f16"], check=True)
    quantized.mkdir(parents=True, exist_ok=True)
    executable = build / ("acestep-quantize.exe" if os.name == "nt" else "acestep-quantize")
    for role in MINIMAX_ROLES:
        destination = quantized / ("mm3-" + role + "-q4_k_m.gguf")
        if destination.is_file():
            continue
        temporary = destination.with_suffix(".partial")
        try:
            subprocess.run([str(executable), str(native / ("mm3-" + role + "-f16.gguf")),
                            str(temporary), "Q4_K_M"], check=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return native, quantized


def prepare_models(cache, build):
    manifest = json.loads(MANIFEST.read_text())
    for name, group in manifest.items():
        download_group(group, cache / name)
    link_base_models(cache / "acestep", cache / "acestep-base")
    native, quantized = convert_minimax(cache, build)
    return {"AUDIOGEN_TEST_MODELS_DIR": cache / "acestep",
            "AUDIOGEN_TEST_BASE_MODELS_DIR": cache / "acestep-base",
            "AUDIOGEN_TEST_MINIMAX_MODELS_DIR": native,
            "AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR": quantized}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-env", type=Path, required=True)
    args = parser.parse_args()
    environment = prepare_models(args.cache_dir.resolve(), args.build_dir.resolve())
    with args.output_env.open("a", encoding="utf-8") as output:
        for name, path in environment.items():
            output.write(f"{name}={path}\n")


if __name__ == "__main__":
    main()
