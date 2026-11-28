#!/usr/bin/env python3
"""Experimental full-EU exports; these packages are not runtime sidecars.

Export en/other/gate with identical capacities, merge them with shared weights,
or export a single graph with a binary adapter_scale input and gate_features.
Automatic routing must evaluate the gate with adapter_scale=0 first.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


def load_encoder_exporter():
    path = Path(__file__).with_name("export-encoder-coreml-ane.py")
    spec = importlib.util.spec_from_file_location("eu_experiment_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANE = load_encoder_exporter()
ROUTES = ("en", "other", "gate")
INPUT_NAMES = ("mel", "mel_mask", "pad_mask", "key_bias")
DEFAULT_THREADS = 4


class RoutedEncoder(ANE.AneEncoder):
    """One trunk with scaled adapter residuals and an intermediate gate output."""

    def __init__(self, ref, weights, meta, window_mel_frames):
        super().__init__(ref, weights, meta, window_mel_frames, eu_route="other")
        self.gate_layer = int(meta["parakeet.eu.gate_layer"])
        if not 0 <= self.gate_layer < self.n_layers:
            raise ValueError("Gate layer must be within the encoder trunk")

    def forward(self, mel, mel_mask, pad_mask, key_bias, adapter_scale):
        x = self.subsample(mel, mel_mask) * pad_mask
        gate_features = x
        for index in range(self.n_layers):
            x = self.block(x, pad_mask, key_bias, index)
            if index in self.adapter_layers:
                adapter = F.gelu(self._conv(x, f"eu{index}_down"), approximate="none")
                x = x + adapter_scale * self._conv(adapter, f"eu{index}_up")
            if index == self.gate_layer:
                gate_features = x
        return x, gate_features


def check_routed_parity(model, ref, weights, meta):
    torch.manual_seed(0)
    full = torch.randn(model.n_mels, model.window_mel_frames)
    lengths = (model.window_mel_frames, model.window_mel_frames // 3 + 7)
    for length in lengths:
        mel = full[:, :length]
        inputs, valid = ANE.window_inputs(model, mel)
        for scale, route in ((0, "en"), (1, "other")):
            with torch.inference_mode():
                output, gate = model(*inputs, torch.full((1, 1, 1, 1), float(scale)))
            want = ANE.reference_frames(ref, weights, meta, mel, eu_route=route)
            got = output[0, :, 0, :valid].T
            error = ANE.relative_error(got, want)
            print(f"[check] {route} frames={length} relative_l2={error:.6g}", flush=True)
            if not torch.isfinite(got).all() or error >= ANE.PARITY_REL_TOLERANCE:
                raise ValueError("Routed encoder differs from the independent reference")
            if scale == 0:
                want_gate = ANE.reference_frames(ref, weights, meta, mel, eu_route="gate")
                gate_error = ANE.relative_error(gate[0, :, 0, :valid].T, want_gate)
                if not torch.isfinite(gate).all() or gate_error >= ANE.PARITY_REL_TOLERANCE:
                    raise ValueError("Unadapted gate features differ from the reference")


def convert_encoder(model, unified):
    mel = torch.zeros(model.n_mels, model.window_mel_frames)
    example, _ = ANE.window_inputs(model, mel)
    names = list(INPUT_NAMES)
    outputs = [ct.TensorType(name="encoder_out", dtype=np.float16)]
    if unified:
        example += (torch.zeros(1, 1, 1, 1),)
        names.append("adapter_scale")
        outputs.append(ct.TensorType(name="gate_features", dtype=np.float16))
    traced = torch.jit.trace(model, example, check_trace=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=value.shape, dtype=np.float16)
                for name, value in zip(names, example)],
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )


def export_encoder(args):
    ref = ANE.load_reference_encoder(Path(__file__).parent)
    weights, meta = ref.load_gguf(args.gguf)
    ANE.validate_eu_route(meta, "other")
    unified = args.route == "unified"
    if unified:
        model = RoutedEncoder(ref, weights, meta, args.window_mel_frames).eval()
        check_routed_parity(model, ref, weights, meta)
    else:
        model = ANE.AneEncoder(ref, weights, meta, args.window_mel_frames,
                               eu_route=args.route).eval()
        if not ANE.check_parity(model, ref, weights, meta):
            raise ValueError("Fixed route failed reference parity")
    converted = convert_encoder(model, unified)
    if args.nbits:
        converted = ANE.palettize(converted, args.nbits, args.palettize_workers)
    converted.user_defined_metadata["parakeet.experimental"] = "shared-eu-routes"
    converted.save(str(args.out))
    compile_package(args.out)


def merge_encoders(args):
    if args.nbits:
        raise ValueError("Palettize each source route before merging; multifunction PTQ is unsupported")
    descriptor = ct.utils.MultiFunctionDescriptor()
    for route in ROUTES:
        descriptor.add_function(str(args.source_dir / f"{route}.mlpackage"), "main", route)
    descriptor.default_function_name = "en"
    ct.utils.save_multifunction(descriptor, str(args.out))
    compile_package(args.out)


def compile_package(path):
    subprocess.run(["xcrun", "coremlc", "compile", str(path), str(path.parent)], check=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "merge"))
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--route", choices=(*ROUTES, "unified"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nbits", type=int, choices=(0, 6), default=0)
    parser.add_argument("--palettize-workers", type=int, default=1)
    parser.add_argument("--window-mel-frames", type=int, default=ANE.EU_GATE_WINDOW_MEL_FRAMES)
    args = parser.parse_args(argv)
    if args.palettize_workers < 1:
        parser.error("--palettize-workers must be positive")
    if args.out.suffix != ".mlpackage":
        parser.error("--out must end in .mlpackage")
    if args.out.exists() or args.out.with_suffix(".mlmodelc").exists():
        parser.error("Use a new output path; existing experiment artifacts are preserved")
    if args.action == "export":
        if not args.gguf or not args.route:
            parser.error("export requires --gguf and --route")
        if args.window_mel_frames < ANE.EU_GATE_WINDOW_MEL_FRAMES:
            parser.error("Experiments require at least 2001 mel frames for the 20-second gate")
    elif not args.source_dir:
        parser.error("merge requires --source-dir")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.set_num_threads(DEFAULT_THREADS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "export":
        export_encoder(args)
    else:
        merge_encoders(args)


if __name__ == "__main__":
    main()
