#!/usr/bin/env python3
"""Exercise the CLI benchmark's Parakeet Core ML reporting contract."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def check_required_without_sidecar(binary: Path, model: Path, wav: Path) -> None:
    """use = require with no adjacent sidecar: a clean error exit, not a crash."""
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / model.name
        staged.symlink_to(model.resolve())
        completed = subprocess.run(
            [str(binary), "--model", str(staged), "--wav", str(wav),
             "--coreml-use", "require"],
            text=True,
            capture_output=True,
            check=False,
        )
    # A negative return code means the process died from a signal (abort).
    if completed.returncode <= 0 or "error: coreml:" not in completed.stderr:
        raise AssertionError(
            "--coreml-use require without a sidecar must exit non-zero with "
            f"an error message; got {completed.returncode}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--language")
    parser.add_argument("--check-require-without-sidecar", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "bench.json"
        command = [
            str(args.binary),
            "--model", str(args.model),
            "--wav", str(args.wav),
            "--bench",
            "--bench-warmup", "0",
            "--bench-runs", "1",
            "--bench-json", str(result_path),
            "--require-coreml",
            "--n-gpu-layers", "999",
        ]
        if args.language:
            command.extend(["--language", args.language])
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            raise AssertionError(
                f"CLI failed with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        payload = json.loads(result_path.read_text())
        assert payload["encoder_coreml_all_runs"] is True, payload
        assert payload["encoder_backend"].startswith("coreml-"), payload
        assert payload["backend"].startswith("ggml-"), payload

    if args.check_require_without_sidecar:
        check_required_without_sidecar(args.binary, args.model, args.wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
