#!/usr/bin/env python3
"""Write tiny Chatterbox S3Gen sidecars for the runtime's feature-shape checks.

`good-s3gen-{encoder,cfm,hift}.mlmodelc` declare the shapes the runtime packs
(encoder tokens_8, CFM frames_16, HiFT window 32). Every
`<stage>-<feature>-s3gen-<stage>.mlmodelc` declares one feature of that stage
with a doubled leading dimension, so the shared runner would copy past the
buffer the runtime packs for it. test-chatterbox-coreml --shape-fixture
(CHATTERBOX_COREML_TEST_SHAPES=<out>) must serve the good sidecars and reject
every other one before predicting. The models only sum their inputs.

`routing-s3gen-{encoder,cfm,hift}.mlmodelc` are well-formed multi-width
sidecars (encoder tokens_{8,12,16,20}, CFM frames_{16,24,32,40}, HiFT window
32) for the routing checks of test-chatterbox-coreml --pending-fixture: which
width a call acquires, what loads first, the resident cap and failure
handling.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chatterbox_coreml_models import EMBED, HIFT_HOP, MEL, STFT_BINS, TIME_EMBED  # noqa: E402

TOKENS, FRAMES, WINDOW, RATIO = 8, 16, 32, 2
ROUTING_TOKENS = (8, 12, 16, 20)
ROUTING_FRAMES = tuple(t * RATIO for t in ROUTING_TOKENS)


def encoder(tokens):
    return (f"tokens_{tokens}",
            {"embedding": (1, tokens, EMBED), "valid": (1, 1, tokens)},
            {"mu": (1, tokens * RATIO, MEL)})


def cfm(frames):
    plane = (1, MEL, frames)
    return (f"frames_{frames}",
            {"x": plane, "mu": plane, "cond": plane, "spks": (1, MEL), "t_emb": (1, TIME_EMBED),
             "valid": (1, 1, frames)},
            {"velocity": plane})


def hift():
    stft = WINDOW * HIFT_HOP + 1
    return ("main",
            {"mel": (1, MEL, WINDOW), "source": (1, STFT_BINS, stft), "valid": (1, 1, WINDOW)},
            {"spectrum": (1, STFT_BINS, stft)})


def stages():
    return {"encoder": encoder(TOKENS), "cfm": cfm(FRAMES), "hift": hift()}


class Probe(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("base", torch.zeros(shape))

    def forward(self, *inputs):
        return self.base + sum(x.mean() for x in inputs)


def convert(inputs, outputs):
    import coremltools as ct

    (name, shape), = outputs.items()
    examples = tuple(torch.zeros(s) for s in inputs.values())
    traced = torch.jit.trace(Probe(shape).eval(), examples, check_trace=False)
    return ct.convert(traced, convert_to="mlprogram",
                      inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in inputs.items()],
                      outputs=[ct.TensorType(name=name, dtype=np.float32)],
                      minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT32)


def write_sidecar(out, stem, stage, function, inputs, outputs, scratch):
    import coremltools as ct

    package = Path(scratch) / f"{stem}-{stage}.mlpackage"
    model = convert(inputs, outputs)
    if function == "main":
        model.save(str(package))
    else:
        part = Path(scratch) / f"{stem}-{stage}-part.mlpackage"
        model.save(str(part))
        descriptor = ct.utils.MultiFunctionDescriptor()
        descriptor.add_function(str(part), src_function_name="main", target_function_name=function)
        descriptor.default_function_name = function
        ct.utils.save_multifunction(descriptor, str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)],
                   check=True, stdout=subprocess.DEVNULL)


def write_multi(out, stem, stage, functions, scratch):
    """One multi-function sidecar with every (function, inputs, outputs) in `functions`."""
    import coremltools as ct

    descriptor = ct.utils.MultiFunctionDescriptor()
    for function, inputs, outputs in functions:
        part = Path(scratch) / f"{stem}-{stage}-{function}.mlpackage"
        convert(inputs, outputs).save(str(part))
        descriptor.add_function(str(part), src_function_name="main", target_function_name=function)
    descriptor.default_function_name = functions[0][0]
    package = Path(scratch) / f"{stem}-{stage}.mlpackage"
    ct.utils.save_multifunction(descriptor, str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)],
                   check=True, stdout=subprocess.DEVNULL)


def doubled(shape):
    return (shape[0] * 2,) + tuple(shape[1:])


def write_fixture(out):
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for stage, (function, inputs, outputs) in stages().items():
            write_sidecar(out, "good-s3gen", stage, function, inputs, outputs, scratch)
            for feature in list(inputs) + list(outputs):
                bad_inputs = {n: doubled(s) if n == feature else s for n, s in inputs.items()}
                bad_outputs = {n: doubled(s) if n == feature else s for n, s in outputs.items()}
                write_sidecar(out, f"{stage}-{feature}-s3gen", stage, function, bad_inputs, bad_outputs, scratch)
                print(f"[{stage}] {feature} declared {doubled({**inputs, **outputs}[feature])}")
        write_multi(out, "routing-s3gen", "encoder", [encoder(t) for t in ROUTING_TOKENS], scratch)
        write_multi(out, "routing-s3gen", "cfm", [cfm(f) for f in ROUTING_FRAMES], scratch)
        write_sidecar(out, "routing-s3gen", "hift", *hift(), scratch)
        print(f"[routing] encoder {ROUTING_TOKENS}, cfm {ROUTING_FRAMES}, hift window {WINDOW}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    write_fixture(parser.parse_args().out)
