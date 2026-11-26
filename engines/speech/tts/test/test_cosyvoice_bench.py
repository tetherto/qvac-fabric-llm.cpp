import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SUPPORT_FILES = ("voice.gguf", "vocab.json", "merges.txt")
MODEL_STAGES = ("llm", "flow", "hift")
THREADS = 4
SKIP = 77


def copy_support_files(source, destination):
    for name in SUPPORT_FILES:
        shutil.copyfile(source / name, destination / name)


def append_overrides(command, overrides):
    for stage, path in overrides.items():
        command.extend([f"--{stage}-gguf", str(path)])


def check_result(result, overrides):
    expected = {stage: str(path) for stage, path in overrides.items()}
    if result["model_overrides"] != expected:
        raise AssertionError("benchmark must record each explicit model path")
    if result["tokens_pinned"] or result["work"]["speech_tokens"] <= 0:
        raise AssertionError("benchmark must run the overridden LM")
    if result["work"]["n_samples"] <= 0 or result["stages"]["dit_euler"]["median_ms"] <= 0:
        raise AssertionError("benchmark must synthesize through the overridden flow and HiFT")
    decode_ms = result["stages"]["lm_decode"]["median_ms"]
    per_token_ms = result["stages"]["lm_decode_per_token"]["median_ms"]
    if not math.isclose(per_token_ms * result["work"]["decode_steps"], decode_ms, rel_tol=1e-4):
        raise AssertionError("per-token decode time must use the executed decode-step count")


def run_benchmark(executable, model_dir, overrides, temporary):
    copy_support_files(model_dir, temporary)
    output = temporary / "result.json"
    command = [
        str(executable), "--model-dir", str(temporary), "--backend", "cpu",
        "--text", "Hello.", "--greedy", "--threads", str(THREADS),
        "--runs", "1", "--warmup", "0", "--json-out", str(output),
    ]
    append_overrides(command, overrides)
    subprocess.run(command, check=True)
    check_result(json.loads(output.read_text()), overrides)


def main():
    executable, model_dir, *models = map(Path, sys.argv[1:])
    required = models + [model_dir / name for name in SUPPORT_FILES]
    if not all(path.is_file() for path in required):
        print("CosyVoice benchmark fixtures are not available")
        sys.exit(SKIP)
    overrides = dict(zip(MODEL_STAGES, models))
    with tempfile.TemporaryDirectory(prefix="cosyvoice-bench-") as directory:
        run_benchmark(executable, model_dir, overrides, Path(directory))


if __name__ == "__main__":
    main()
