"""Record a run's immutable lane metadata, binary and exact launch configuration."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

HASH_CHUNK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_environment() -> dict:
    return {key: value for key, value in os.environ.items()
            if key.startswith(("GGML_", "CUDA_", "VK_", "TENSORFOLD_"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--draft-width", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    command = [str(binary), *args.server_args[1:]] if args.server_args[:1] == ["--"] else [str(binary), *args.server_args]
    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True)
    cache = binary.parent.parent / "CMakeCache.txt"
    manifest = {"label": args.label, "device": args.device, "backend": args.backend,
                "draft_width": args.draft_width, "lane": json.loads(Path(args.lane).read_text()),
                "binary": str(binary), "binary_sha256": sha256(binary),
                "binary_version": version.stdout + version.stderr,
                "cmake_cache_sha256": sha256(cache), "command": command,
                "environment": relevant_environment(), "platform": platform.platform(),
                "hostname": platform.node(), "created_unix_s": time.time()}
    Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
