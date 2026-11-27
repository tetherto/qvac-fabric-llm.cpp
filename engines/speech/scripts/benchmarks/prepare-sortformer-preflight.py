#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

STEM = "diar_streaming_sortformer_4spk-v2.1"
REVISION = "cd03eee90fbec18297ac31b8c21546e596b7f71c"
CHECKPOINT_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
URL = f"https://huggingface.co/nvidia/{STEM}/resolve/{REVISION}/{STEM}.nemo"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "parakeet/scripts/convert-nemo-to-gguf.py"
DOWNLOAD_TIMEOUT_SECONDS = 120


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_checkpoint(path):
    with urlopen(URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Sortformer checkpoint SHA-256 mismatch")


def prepare(root):
    assets = root / "assets"
    results = root / "results"
    assets.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    checkpoint = root / (STEM + ".nemo")
    download_checkpoint(checkpoint)
    model = assets / (STEM + ".q8_0.gguf")
    command = [sys.executable, str(CONVERTER), "--ckpt", str(checkpoint),
               "--out", str(model), "--quant", "q8_0", "--head", "sortformer"]
    provenance = {"checkpoint_url": URL, "checkpoint_revision": REVISION,
                  "checkpoint_sha256": CHECKPOINT_SHA256, "converter_sha256": sha256(CONVERTER),
                  "command": command}
    report = results / "model-provenance.json"
    report.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run(command, check=True)
    if not model.is_file() or model.stat().st_size == 0:
        raise ValueError("converter did not produce a nonempty GGUF")
    provenance["gguf_sha256"] = sha256(model)
    report.write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == "__main__":
    main()
