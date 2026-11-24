#!/usr/bin/env python3
"""Inspect Nemotron architecture, prompt aliases, and tensor shapes."""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml


RELEVANT_TENSOR_TERMS = (
    "prompt",
    "language",
    "decoder",
    "joint",
    "encoder",
)

ARCHITECTURE_FIELDS = (
    ("target",),
    ("nemo_version",),
    ("sample_rate",),
    ("num_prompts",),
    ("encoder", "_target_"),
    ("encoder", "feat_in"),
    ("encoder", "n_layers"),
    ("encoder", "d_model"),
    ("encoder", "n_heads"),
    ("encoder", "subsampling"),
    ("encoder", "subsampling_factor"),
    ("encoder", "subsampling_conv_channels"),
    ("encoder", "att_context_size"),
    ("encoder", "att_context_style"),
    ("encoder", "conv_kernel_size"),
    ("encoder", "conv_context_size"),
    ("encoder", "causal_downsampling"),
    ("encoder", "conv_norm_type"),
    ("decoder", "_target_"),
    ("decoder", "vocab_size"),
    ("decoder", "prednet", "pred_hidden"),
    ("decoder", "prednet", "pred_rnn_layers"),
    ("joint", "_target_"),
    ("joint", "num_classes"),
    ("joint", "jointnet", "encoder_hidden"),
    ("joint", "jointnet", "pred_hidden"),
    ("joint", "jointnet", "joint_hidden"),
    ("model_defaults", "initialize_prompt_feature"),
    ("model_defaults", "num_prompts"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a .nemo archive")
    return parser.parse_args()


def get_archive_member(archive, name):
    for candidate in (name, f"./{name}"):
        try:
            return archive.getmember(candidate)
        except KeyError:
            continue
    raise KeyError(f"{name} not found in checkpoint")


def read_config(archive):
    member = get_archive_member(archive, "model_config.yaml")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError("model_config.yaml could not be read")
    return yaml.safe_load(source.read().decode("utf-8"))


def read_state_dict(archive):
    member = get_archive_member(archive, "model_weights.ckpt")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError("model_weights.ckpt could not be read")

    with tempfile.TemporaryFile() as weights_file:
        shutil.copyfileobj(source, weights_file)
        weights_file.seek(0)
        checkpoint = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=True,
        )

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


def nested_value(config, path):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def print_architecture(config):
    print("ARCHITECTURE")
    for path in ARCHITECTURE_FIELDS:
        value = nested_value(config, path)
        print(f"{'.'.join(path)} = {value!r}")


def prompt_alias_groups(config):
    prompt_dictionary = nested_value(
        config,
        ("model_defaults", "prompt_dictionary"),
    )
    if not isinstance(prompt_dictionary, dict):
        return {}

    groups = {}
    for alias, prompt_index in prompt_dictionary.items():
        groups.setdefault(int(prompt_index), []).append(str(alias))
    return groups


def print_prompt_aliases(config):
    print("\nPROMPT ALIASES")
    groups = prompt_alias_groups(config)
    if not groups:
        print("<none>")
        return

    for prompt_index in sorted(groups):
        aliases = ", ".join(sorted(groups[prompt_index]))
        print(f"{prompt_index}: {aliases}")


def is_relevant_tensor(name):
    lowered = name.lower()
    return any(term in lowered for term in RELEVANT_TENSOR_TERMS)


def relevant_tensor_rows(state_dict):
    rows = []
    for name, tensor in state_dict.items():
        if not is_relevant_tensor(name):
            continue
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else "<no shape>"
        dtype = str(tensor.dtype) if hasattr(tensor, "dtype") else type(tensor).__name__
        rows.append((str(name), shape, dtype))
    return sorted(rows)


def print_relevant_tensors(state_dict):
    print("\nRELEVANT TENSORS")
    rows = relevant_tensor_rows(state_dict)
    for name, shape, dtype in rows:
        print(f"{name}: shape={shape} dtype={dtype}")
    print(f"\nRelevant tensor count: {len(rows)}")
    print(f"Total tensor count: {len(state_dict)}")


def inspect_checkpoint(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with tarfile.open(checkpoint_path, "r") as archive:
        config = read_config(archive)
        state_dict = read_state_dict(archive)

    print_architecture(config)
    print_prompt_aliases(config)
    print_relevant_tensors(state_dict)


def main():
    args = parse_args()
    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
