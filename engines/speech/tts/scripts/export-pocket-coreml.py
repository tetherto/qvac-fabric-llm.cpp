#!/usr/bin/env python3
"""Export Pocket TTS Core ML sidecars: the FlowLM generation step and the Mimi decoder.

Each sidecar is rebuilt in PyTorch from the same GGUF tensors the ggml engine
loads and written next to its GGUF under the engine's naming rule. The GGUF's
architecture selects which sidecar is exported:

  flow-lm.gguf (or flow-lm-f16.gguf) -> flow-lm-step.mlmodelc
  mimi.gguf                          -> mimi-decoder.mlmodelc

Step: one prediction runs one generated frame (FlowLM advance plus a one-step
LSD flow) with the KV cache in Core ML state; see
scripts/pocket_coreml_models.py. `--context` fixes how many positions (voice,
text and generated frames) the state holds; the engine keeps a chunk that
needs more on ggml. Core ML state cannot be placed on the Neural Engine, so
the engine loads the step on CPU + GPU by default.

Mimi decoder: one prediction decodes one latent frame into 1920 samples, with
the conv histories and the transformer's attention window as explicit inputs
and outputs so it runs on the Neural Engine (the engine's default for it); see
scripts/pocket_coreml_mimi.py.

Both compute in float16 with float32 inputs and outputs, target macOS 15 /
iOS 18, and record the checkpoint's source hash; the engine refuses a sidecar
of another checkpoint.

  python3 scripts/export-pocket-coreml.py --gguf models/pocket/flow-lm.gguf models/pocket/mimi.gguf --compile
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import pocket_coreml_mimi as mimi
from pocket_coreml_models import ARCH, INPUTS, OUTPUTS, STATES, FlowStep, example_inputs, read_gguf

QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SUFFIX = "-step"
MIMI_SUFFIX = "-decoder"
DEFAULT_CONTEXT = 1024
SCHEMA = "1"


def model_stem(path):
    return re.sub(r"[.-](?:" + "|".join(QUANT_TAGS) + r")$", "", Path(path).stem, flags=re.IGNORECASE)


def architecture(path):
    from gguf import GGUFReader

    return GGUFReader(str(path)).fields["general.architecture"].contents()


def convert(model, inputs):
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(inputs[name] for name in INPUTS), check_trace=False)
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(getattr(model, name).shape), dtype=np.float16),
                           name=name) for name in STATES]
    return ct.convert(traced, convert_to="mlprogram",
                      inputs=[ct.TensorType(name=name, shape=tuple(inputs[name].shape), dtype=np.float32)
                              for name in INPUTS],
                      outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUTS],
                      states=states, minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16)


def convert_decoder(model, inputs):
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.from_numpy(inputs[name]) for name in mimi.INPUTS),
                                 check_trace=False)
    return ct.convert(traced, convert_to="mlprogram",
                      inputs=[ct.TensorType(name=name, shape=inputs[name].shape, dtype=np.float32)
                              for name in mimi.INPUTS],
                      outputs=[ct.TensorType(name=name, dtype=np.float32) for name in mimi.OUTPUTS],
                      minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)


def prepare(args, stem):
    package, compiled = args.output / (stem + ".mlpackage"), args.output / (stem + ".mlmodelc")
    for path in (package, compiled) if args.compile else (package,):
        if path.exists():
            if not args.force:
                raise FileExistsError(f"{path} exists (pass --force to replace it)")
            shutil.rmtree(path)
    return package, compiled


def finish(args, converted, package, compiled, stem, source, manifest):
    import coremltools as ct

    converted.user_defined_metadata["pocket.coreml.schema"] = SCHEMA
    converted.user_defined_metadata["pocket.source_sha256"] = source
    converted.user_defined_metadata["pocket.coreml.source"] = manifest["source"]
    converted.save(str(package))
    manifest.update(source_sha256=source, schema=SCHEMA, coremltools=ct.__version__, torch=torch.__version__)
    (args.output / (stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output)],
                       check=True, capture_output=True, text=True)
        print(compiled)
    print(package)


def export_step(path, args):
    tensors, metadata = read_gguf(path)
    stem = model_stem(path) + SUFFIX
    package, compiled = prepare(args, stem)
    model = FlowStep(tensors, metadata, args.context).eval()
    inputs = example_inputs(metadata, args.context)
    converted = convert(model, inputs)
    converted.user_defined_metadata["pocket.coreml.context"] = str(args.context)
    finish(args, converted, package, compiled, stem, str(metadata["pocket.source_sha256"]),
           {"source": path.name, "context": args.context,
            "inputs": {k: list(v.shape) for k, v in inputs.items()}, "outputs": list(OUTPUTS),
            "states": {name: list(getattr(model, name).shape) for name in STATES}})


def export_decoder(path, args):
    tensors, metadata = mimi.read_gguf(path)
    stem = model_stem(path) + MIMI_SUFFIX
    package, compiled = prepare(args, stem)
    inputs = mimi.Stream().inputs(np.zeros((mimi.FRAMES, mimi.LATENT), np.float32))
    converted = convert_decoder(mimi.Decoder(tensors).eval(), inputs)
    converted.user_defined_metadata["pocket.coreml.frames"] = str(mimi.FRAMES)
    finish(args, converted, package, compiled, stem, str(metadata["pocket.source_sha256"]),
           {"source": path.name, "frames": mimi.FRAMES,
            "inputs": {k: list(v.shape) for k, v in inputs.items()}, "outputs": list(mimi.OUTPUTS)})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, type=Path, nargs="+",
                        help="Pocket FlowLM GGUF (F32 or F16) and/or Mimi GGUF")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help="step positions the state holds: voice + text + generated frames (default %(default)s)")
    parser.add_argument("--output", type=Path, help="output directory (default: next to each GGUF)")
    parser.add_argument("--compile", action="store_true", help="also compile each package to <stem>.mlmodelc")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args()
    if not 2 <= args.context <= 8192:
        parser.error("--context must be 2..8192 (the engine's context limit)")
    return args


def main():
    args = parse_args()
    output = args.output
    for path in args.gguf:
        args.output = output or path.parent
        args.output.mkdir(parents=True, exist_ok=True)
        arch = architecture(path)
        if arch == ARCH:
            export_step(path, args)
        elif arch == mimi.ARCH:
            export_decoder(path, args)
        else:
            raise ValueError(f"{path} is a {arch} GGUF, not a Pocket FlowLM or Mimi GGUF")


if __name__ == "__main__":
    main()
