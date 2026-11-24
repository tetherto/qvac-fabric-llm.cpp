#!/usr/bin/env python3
"""Create a complete native Pocket bundle in a new directory from local assets."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


def module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "config", "tokenizer", "voice", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    a = p.parse_args()
    if a.output.exists():
        p.error("output must be a new directory; create a new bundle for updates")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=a.output.name + ".staging-", dir=a.output.parent))
    try:
        module("convert-pocket-flow-lm-to-gguf").convert(a.weights, a.config, staging / "flow-lm.gguf", a.dtype)
        module("convert-pocket-mimi-to-gguf").convert(a.weights, a.config, staging / "mimi.gguf")
        module("convert-pocket-frontend").convert(a.weights, a.config, a.tokenizer, a.voice, staging)
        assets = {}
        for path in sorted(staging.iterdir()):
            with path.open("rb") as f:
                assets[path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, "sha256").hexdigest())
        (staging / "manifest.json").write_text(json.dumps(dict(engine="pocket", schema_version=1, assets=assets), indent=2) + "\n")
        # No existing files are replaced. The completed directory becomes
        # visible only after all conversion and validation succeeds.
        os.rename(staging, a.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
