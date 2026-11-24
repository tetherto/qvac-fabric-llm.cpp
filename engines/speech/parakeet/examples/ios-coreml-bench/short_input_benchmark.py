#!/usr/bin/env python3
"""Exercise single-encoder-frame CPU Core ML inputs against native inference."""
import subprocess
import benchmark as batch


def check_missing_owned_flexible(device):
    model = "models6/transcribepsy-250m-eu.f16.gguf"
    wav = "en-60s.wav"
    args = ["--model", str(batch.ROOT / model) if device == "mac" else model,
            "--wav", str(batch.ROOT / "inputs" / wav) if device == "mac" else wav,
            "--backend", "cpu", "--threads", "4", "--language", "en", "--window-sec", "0",
            "--bench", "--bench-runs", "1", "--bench-warmup", "0", "--require-coreml"]
    if device == "iphone":
        batch.wait_for_phone_cooldown()
    command, environment = batch.build_command(device, args, {"PARAKEET_COREML_ONLY": "1"})
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)
    content = result.stdout + result.stderr
    (batch.OUT / f"{device}-owned-missing-flexible.log").write_text(content)
    if result.returncode == 0 or "weights were not loaded" not in content:
        raise RuntimeError(f"Owned {device} model silently used a missing flexible encoder")
    print(device, "owned missing flexible sidecar rejected safely", flush=True)


def main():
    batch.configure()
    for device in batch.DEVICES:
        for frames in range(1, 9):
            wav = f"tiny-{frames}.wav"
            batch.REFERENCES[wav] = ""
            for language in ("en", "other", "auto"):
                name = f"tiny-{frames}-{language}"
                native = batch.run(device, name + "-native", wav, "cpu", language, runs=1, warmup=0)
                coreml = batch.run(device, name + "-coreml", wav, "coreml", language,
                    settings={"PARAKEET_COREML_COMPUTE_UNITS": "cpu_only"}, runs=1, warmup=0)
                if not coreml["encoder_coreml_all_runs"] or coreml["transcript"] != native["transcript"]:
                    raise RuntimeError(f"Tiny-input parity or dispatch mismatch: {device} {name}")
        batch.run(device, "tiny-owner", "tiny-1.wav", "coreml", "auto",
            settings={"PARAKEET_COREML_COMPUTE_UNITS": "cpu_only", "PARAKEET_COREML_ONLY": "1"},
            runs=1, warmup=0)
        check_missing_owned_flexible(device)


if __name__ == "__main__":
    main()
