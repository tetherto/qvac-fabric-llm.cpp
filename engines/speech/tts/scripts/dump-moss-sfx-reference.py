#!/usr/bin/env python3
"""Dump MOSS-SoundEffect stage references from the PyTorch pipeline (fp32, CPU)
for test-moss-sfx-parity: token ids, prompt and negative contexts, the initial
noise, the first-step DiT velocities, the schedule, the latents after a short
guided sampling loop, and their VAE decode, as raw little-endian .bin files.

usage: dump-moss-sfx-reference.py <hf_checkpoint_dir> <out_dir> --upstream <MOSS-TTS checkout>
The defaults reproduce the fixture test-moss-sfx-parity expects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HOP_LENGTH = 960
LATENT_CHANNELS = 128


def save(out: Path, name: str, tensor) -> None:
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array.astype(np.int32 if array.dtype.kind in "iu" else np.float32).tofile(out / f"{name}.bin")


def load_pipeline(model_dir: str, upstream: Path):
    import torch

    sys.path.insert(0, str(upstream))
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    return MossSoundEffectPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, device="cpu")


def dump_text(engine, out: Path, prompt: str) -> tuple:
    ids, mask = engine.prompter.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    save(out, "ids", ids[0, : int(mask.sum())])
    positive = engine.prompter.encode_prompt(prompt, device="cpu")
    negative = engine.prompter.encode_prompt("", device="cpu")
    save(out, "context_pos", positive[0])
    save(out, "context_neg", negative[0])
    return positive, negative


def initial_noise(frames: int, seed: int):
    import torch

    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((1, LATENT_CHANNELS, frames)).astype(np.float32))


def sample(engine, out: Path, latents, positive, negative, args):
    engine.scheduler.set_timesteps(args.steps, shift=args.shift)
    save(out, "sigmas", engine.scheduler.sigmas)
    for index, timestep in enumerate(engine.scheduler.timesteps):
        t = timestep.unsqueeze(0)
        v_pos = engine.dit(latents, t, positive)
        v_neg = engine.dit(latents, t, negative)
        if index == 0:
            save(out, "v_pos0", v_pos[0])
            save(out, "v_neg0", v_neg[0])
        velocity = v_neg + args.cfg * (v_pos - v_neg)
        latents = engine.scheduler.step(velocity, engine.scheduler.timesteps[index], latents)
    return latents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--prompt", default="A glass falls and shatters on a tile floor.")
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.set_grad_enabled(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.model_dir, args.upstream)
    prompt = f"{args.prompt.strip()} duration: {args.seconds:.1f}s"
    positive, negative = dump_text(pipe.engine, args.out_dir, prompt)
    frames = pipe.sample_rate * pipe.max_inference_seconds // HOP_LENGTH
    noise = initial_noise(frames, args.noise_seed)
    save(args.out_dir, "noise", noise[0])
    latents = sample(pipe.engine, args.out_dir, noise.clone(), positive, negative, args)
    save(args.out_dir, "latents", latents[0])
    audio = pipe.engine.vae.decode(latents)[0, 0, : int(pipe.sample_rate * args.seconds)]
    save(args.out_dir, "audio", audio)
    (args.out_dir / "meta.json").write_text(json.dumps({
        "prompt": prompt, "steps": args.steps, "seconds": args.seconds, "cfg": args.cfg,
        "shift": args.shift, "frames": frames, "sample_rate": pipe.sample_rate}, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
