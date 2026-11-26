#!/usr/bin/env python3
"""Dump upstream PyTorch FlowLM fixtures for Fabric's native implementation.

Default: small, deterministic weights (offline, no model download). Pass
--weights and --config to replay an actual checkpoint with the same harness.
The oracle recomputes full prefixes, independently testing native KV updates.
"""
from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import distribution
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
import yaml

from pocket_tts.models.flow_lm import FlowLMModel, lsd_decode
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.transformer import StreamingTransformer

REFERENCE_COMMIT = "0c2db3bdea7c991c568989cc11b503f14483fabc"
spec = importlib.util.spec_from_file_location("converter", Path(__file__).with_name("convert-pocket-flow-lm-to-gguf.py"))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def check_reference():
    direct = distribution("pocket-tts").read_text("direct_url.json")
    if not direct or json.loads(direct).get("vcs_info", {}).get("commit_id") != REFERENCE_COMMIT:
        raise RuntimeError(f"Install pocket-tts from git commit {REFERENCE_COMMIT}; see pocket-flow-requirements.txt")


def small_config():
    return {"flow_lm": {"insert_bos_before_voice": True,
            "transformer": {"d_model": 32, "num_heads": 4, "num_layers": 2,
                            "hidden_scale": 3, "max_period": 10000},
            "flow": {"dim": 16, "depth": 2, "type": "lsd"},
            "lookup_table": {"n_bins": 64}},
            "mimi": {"quantizer": {"dimension": 8}, "inner_dim": 8}}


class Conditioner(nn.Module):
    # Only the embedding participates in this stage: the tokenizer is a
    # separate frontend and the fixtures supply integer IDs explicitly.
    def __init__(self, bins, dim):
        super().__init__()
        self.embed = nn.Embedding(bins + 1, dim)


def build(config):
    d = converter.dimensions(config)
    net = SimpleMLPAdaLN(d["latent_dim"], d["flow_dim"], d["latent_dim"],
                        d["dim"], d["flow_depth"], num_time_conds=2)
    transformer = StreamingTransformer(d["dim"], d["heads"], d["layers"],
                    dim_feedforward=d["ff_dim"], max_period=config["flow_lm"]["transformer"]["max_period"])
    model = FlowLMModel(Conditioner(d["vocab_size"], d["dim"]), net, transformer,
                       dim=d["dim"], ldim=d["latent_dim"],
                       insert_bos_before_voice=config["flow_lm"].get("insert_bos_before_voice", False))
    return model.eval()


@torch.no_grad()
def generate(output: Path, weights=None, config_path=None):
    check_reference()
    torch.set_num_threads(1)
    torch.manual_seed(24306)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").unlink(missing_ok=True)
    actual_checkpoint = weights is not None
    config = yaml.safe_load(config_path.read_text()) if config_path else small_config()
    model = build(config)
    if weights:
        state = {k.removeprefix("flow_lm."): v for k, v in load_file(str(weights)).items()
                 if k.startswith("flow_lm.")}
        if "speaker_proj_weight" in state:
            model.register_parameter("speaker_proj_weight", nn.Parameter(torch.empty_like(state["speaker_proj_weight"])))
        model.load_state_dict(state, strict=True)
    else:
        # Nontrivial normalization affine parameters catch omitted biases/scales.
        for name, p in model.named_parameters():
            if ".norm" in name or "in_ln" in name or name.startswith("out_norm") or name.endswith("alpha"):
                p.copy_(torch.randn_like(p) * 0.1 + (0 if name.endswith("bias") else 1))
        model.emb_mean.copy_(torch.randn_like(model.emb_mean) * 0.2)
        model.emb_std.copy_(torch.rand_like(model.emb_std) + 0.5)
        weights = output / "model.safetensors"
        save_file({"flow_lm." + k: v.contiguous() for k, v in model.state_dict().items()}, str(weights))
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    converter.convert(weights, config_path, output / "flow-lm-f32.gguf")
    converter.convert(weights, config_path, output / "flow-lm-f16.gguf", "f16")
    d = converter.dimensions(config)

    def dump(name, value):
        np.save(output / (name + ".npy"), value.detach().cpu().numpy().astype(np.float32))

    ids = torch.tensor([1, 7, 3, 12, 5])
    np.save(output / "text_ids.npy", ids.numpy().astype(np.int32))
    text = model.conditioner.embed(ids)[None]
    dump("text_embeddings", text)
    prefix = torch.randn(1, 3, d["dim"]) * 0.2
    if model.insert_bos_before_voice:
        prefix = torch.cat([model.bos_before_voice, prefix], dim=1)
    prefix = torch.cat([prefix, text], dim=1)
    dump("embeddings", prefix)
    dump("prefill", model.out_norm(model.transformer(prefix, None)))
    previous = model.bos_emb[None, None]
    for step in range(4):
        prefix = torch.cat([prefix, model.input_linear(previous)], dim=1)
        hidden = model.out_norm(model.transformer(prefix, None))[:, -1]
        dump(f"step{step}_hidden", hidden)
        dump(f"step{step}_eos", model.out_eos(hidden))
        noise = torch.randn(1, d["latent_dim"]) * 0.6
        dump(f"step{step}_noise", noise)
        for count in [1, 2, 4, 16]:
            latent = lsd_decode(lambda s, t, x: model.flow_net(hidden, s, t, x), noise, count)
            dump(f"step{step}_sample{count}", latent)
            if count == 4:
                previous = latent[:, None]
                dump(f"step{step}_denormalized", latent * model.emb_std + model.emb_mean)
    (output / "provenance.json").write_text(json.dumps({"reference_commit": REFERENCE_COMMIT,
        "torch": torch.__version__, "seed": 24306, "actual_checkpoint": actual_checkpoint}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if bool(args.weights) != bool(args.config):
        parser.error("--weights and --config must be supplied together")
    generate(args.output, args.weights, args.config)
