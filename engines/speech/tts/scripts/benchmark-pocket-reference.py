#!/usr/bin/env python3
"""Record upstream Pocket TTS CPU audio and latency using explicit local assets."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import scipy.io.wavfile
import torch
import yaml
from pocket_tts import TTSModel

PIN = "0c2db3bdea7c991c568989cc11b503f14483fabc"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "config", "tokenizer", "voice", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--text", default="Hello! This is Pocket TTS. We are bringing natural, local speech generation to Fabric.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=1)
    a = p.parse_args()
    if a.runs < 1:
        p.error("runs must be positive")
    if not math.isfinite(a.temperature) or not 0 <= a.temperature <= 10 or not 1 <= a.steps <= 64:
        p.error("temperature must be finite and 0..10; steps must be 1..64")
    direct = json.loads(importlib.metadata.distribution("pocket-tts").read_text("direct_url.json"))
    if direct.get("vcs_info", {}).get("commit_id") != PIN:
        raise RuntimeError("Install the pinned reference in pocket-flow-requirements.txt")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "upstream-benchmark.json").unlink(missing_ok=True)
    cfg = yaml.safe_load(a.config.read_text())
    cfg["weights_path"] = str(a.weights.resolve())
    cfg["weights_path_without_voice_cloning"] = None
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(a.tokenizer.resolve())
    local_config = a.output / "reference-local.yaml"
    local_config.write_text(yaml.safe_dump(cfg))
    torch.set_num_threads(1)  # upstream default; LM and Mimi run on separate threads
    start = time.perf_counter()
    model = TTSModel.load_model(config=local_config, sampler_decode_steps=a.steps, temp=a.temperature)
    voice = model.get_state_for_audio_prompt(a.voice)
    load_seconds = time.perf_counter() - start
    runs = []
    # Untimed warmup includes graph/kernel initialization.
    torch.manual_seed(a.seed)
    model.generate_audio(copy.deepcopy(voice), "A short warm up.")
    for run in range(a.runs):
        torch.manual_seed(a.seed)
        chunks = []
        first = None
        start = time.perf_counter()
        for chunk in model.generate_audio_stream(copy.deepcopy(voice), a.text):
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk.cpu().numpy().reshape(-1))
        elapsed = time.perf_counter() - start
        pcm = np.concatenate(chunks)
        duration = len(pcm) / model.sample_rate
        if not np.isfinite(pcm).all() or not np.any(pcm):
            raise RuntimeError("Invalid reference audio")
        runs.append(dict(seconds=elapsed, first_audio_seconds=first,
                         audio_seconds=duration, real_time_factor=elapsed / duration,
                         peak=float(np.abs(pcm).max()), rms=float(np.sqrt(np.mean(pcm ** 2)))))
        if run == 0:
            scipy.io.wavfile.write(a.output / "pocket-upstream.wav", model.sample_rate, pcm)
        print(json.dumps(runs[-1]), flush=True)
    # Separate trace run: capture Mimi inputs without distorting timed runs.
    latent_chunks = []
    original_decode = model.mimi.decode_from_latent
    def trace(latent, state):
        latent_chunks.append(latent.detach().cpu().numpy())
        return original_decode(latent, state)
    model.mimi.decode_from_latent = trace
    torch.manual_seed(a.seed)
    traced = model.generate_audio(copy.deepcopy(voice), a.text).numpy().reshape(-1)
    np.save(a.output / "latents.npy", np.concatenate(latent_chunks, axis=1)[0].astype(np.float32))
    np.save(a.output / "pcm.npy", traced.astype(np.float32))
    with torch.no_grad():
        encoded = model.mimi.encode_to_latent(torch.from_numpy(traced)[None, None])
    np.save(a.output / "encoded.npy", np.ascontiguousarray(encoded[0].numpy(), dtype=np.float32))
    def sha(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    report = dict(implementation="Kyutai upstream PyTorch", commit=PIN,
                  machine=platform.platform(), processor=platform.machine(),
                  torch=torch.__version__, threads_per_worker=1, worker_threads=2,
                  quantized=False, sample_rate=model.sample_rate, steps=a.steps, temperature=a.temperature,
                  seed=a.seed, text=a.text, warmups=1, load_seconds=load_seconds,
                  weights_sha256=sha(a.weights), voice_sha256=sha(a.voice),
                  tokenizer_sha256=sha(a.tokenizer), runs=runs,
                  max_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (a.output / "upstream-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
