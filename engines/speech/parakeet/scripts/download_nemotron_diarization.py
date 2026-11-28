"""Download NVIDIA's Nemotron 3 Diarization GGUF and optional NeMo checkpoint."""

import argparse
from pathlib import Path


MODEL_REPOSITORY = "nvidia/Nemotron-3-Diarization"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
GGUF_FILENAME = "Nemotron-3-Diarization.q8_0.gguf"
NEMO_FILENAME = "Nemotron-3-Diarization.nemo"
DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models"


def download_artifact(filename, model_directory, download):
    model_directory.mkdir(parents=True, exist_ok=True)
    return Path(download(
        repo_id=MODEL_REPOSITORY,
        filename=filename,
        revision=MODEL_REVISION,
        local_dir=str(model_directory),
    ))


def download_model(model_directory, include_source=False, download=None):
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    artifacts = [download_artifact(GGUF_FILENAME, model_directory, download)]
    if include_source:
        artifacts.append(download_artifact(NEMO_FILENAME, model_directory, download))
    return artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--include-source", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in download_model(args.model_dir, args.include_source):
        print(path)


if __name__ == "__main__":
    main()
