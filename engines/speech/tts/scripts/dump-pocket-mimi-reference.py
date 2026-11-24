#!/usr/bin/env python3
"""Non-zero upstream Mimi encoder/decoder fixture; no model downloads."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import safetensors.torch
import torch
import yaml
from pocket_tts.models.mimi import build_mimi
from pocket_tts.modules.stateful_module import StatefulModule, init_states, increment_steps
from pocket_tts.utils.config import MimiConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
pin = "0c2db3bdea7c991c568989cc11b503f14483fabc"
direct = json.loads(importlib.metadata.distribution("pocket-tts").read_text("direct_url.json"))
if direct.get("vcs_info", {}).get("commit_id") != pin:
    raise RuntimeError("Install the pinned upstream requirements")
torch.manual_seed(1742)
torch.set_num_threads(1)
cfg = MimiConfig.model_validate(yaml.safe_load(a.config.read_text())["mimi"])
model = build_mimi(cfg).eval()
for name, module in model.named_modules():
    if isinstance(module, StatefulModule):
        module._module_absolute_name = name
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "provenance.json").unlink(missing_ok=True)
weights = a.output / "random-mimi.safetensors"
safetensors.torch.save_file({"mimi."+k: v for k, v in model.state_dict().items()}, weights)
subprocess.run([sys.executable, str(Path(__file__).with_name("convert-pocket-mimi-to-gguf.py")),
                "--weights", str(weights), "--config", str(a.config), "--output", str(a.output / "mimi.gguf")], check=True)
with torch.no_grad():
    latents = torch.randn(1, 37, 32)
    state = init_states(model, batch_size=1, sequence_length=37*16)
    chunks = []
    for chunk in latents.split(1, dim=1):
        chunks.append(model.decode_from_latent(chunk, state))
        increment_steps(model, state, increment=16)
    pcm = torch.cat(chunks, dim=-1)
    encoded = model.encode_to_latent(pcm)
    for name, value in dict(latents=latents[0], pcm=pcm.reshape(-1), encoded=encoded[0]).items():
        np.save(a.output / (name+".npy"), np.ascontiguousarray(value.numpy(), dtype=np.float32))
(a.output / "provenance.json").write_text(json.dumps(dict(upstream_commit=pin, seed=1742, config=cfg.model_dump(mode="json")), indent=2))
