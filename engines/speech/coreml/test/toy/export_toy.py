#!/usr/bin/env python3
"""Toy Core ML exporter for the shared runtime's tests (test builds only).

Builds one of three tiny ML Programs with coremltools' MIL builder from the
weights of a toy GGUF, the way a real exporter (*/scripts/export-*-coreml.py)
builds its model from a production GGUF:

  stateless  y = linear(relu(linear(x, w1, b1)), transpose(w2), 0.5 * b2)
  multi      two functions sharing w1:  main: y = linear(x, w1, b1)
                                        alt:  y = linear(x, w1, b1) * s
  state      an MLState accumulator:    acc += linear(x, w1, b1); y = acc

`--init <out.gguf>` writes a seed GGUF with the toy's tensors and metadata
(`qvac-coreml-tool dump` turns it into the template's reference-gguf.json).

The toys are routing and lifetime fixtures for test-coreml-toy and a smoke
test for coreml_template.py. They are never a gate on a model.

Usage:
  export_toy.py --kind stateless --init toy.gguf
  export_toy.py --kind stateless --gguf toy.gguf --output out/   (writes out/toy-stateless.mlpackage)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ARCH = "qvac-toy"
KINDS = ("stateless", "multi", "state")
D_IN, D_HID, D_OUT = 8, 32, 48


def tensors_for(kind):
    """name -> numpy shape (numpy order = GGUF ne reversed)."""
    if kind == "stateless":
        return {"toy.w1": (D_HID, D_IN), "toy.b1": (D_HID,), "toy.w2": (D_HID, D_OUT), "toy.b2": (D_OUT,)}
    if kind == "multi":
        return {"toy.w1": (D_HID, D_IN), "toy.b1": (D_HID,), "toy.s": (D_HID,)}
    return {"toy.w1": (D_HID, D_IN), "toy.b1": (D_HID,)}


def write_seed(kind, path):
    import gguf

    rng = np.random.default_rng(1234 + KINDS.index(kind))
    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_string("qvac_toy.kind", kind)
    w.add_uint32("qvac_toy.d_in", D_IN)
    for name, shape in tensors_for(kind).items():
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.25)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_weights(path, kind):
    import gguf

    r = gguf.GGUFReader(str(path))
    by_name = {t.name: t for t in r.tensors}
    out = {}
    for name, shape in tensors_for(kind).items():
        t = by_name[name]
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            sys.exit(f"export_toy: {name} must be f32 (run qvac-coreml-tool f32 on a quantised GGUF first)")
        out[name] = np.array(t.data, dtype=np.float32).reshape(shape)
    return out


def build(kind, w):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    target = ct.target.macOS15
    meta = {"qvac.toy.kind": kind, "qvac.toy.version": "1"}

    def finish(prog):
        m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=target,
                       compute_precision=ct.precision.FLOAT16)
        for k, v in meta.items():
            m.user_defined_metadata[k] = v
        return m

    if kind == "stateless":
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_IN), dtype=ct.converters.mil.mil.types.fp32)],
                    opset_version=target)
        def prog(x):
            h = mb.linear(x=x, weight=w["toy.w1"], bias=w["toy.b1"])
            h = mb.relu(x=h)
            y = mb.linear(x=h, weight=np.ascontiguousarray(w["toy.w2"].T), bias=w["toy.b2"] * np.float32(0.5))
            return mb.identity(x=y, name="y")

        return [("main", finish(prog))]

    if kind == "multi":
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_IN), dtype=ct.converters.mil.mil.types.fp32)],
                    opset_version=target)
        def main(x):
            return mb.identity(x=mb.linear(x=x, weight=w["toy.w1"], bias=w["toy.b1"]), name="y")

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_IN), dtype=ct.converters.mil.mil.types.fp32)],
                    opset_version=target)
        def alt(x):
            h = mb.linear(x=x, weight=w["toy.w1"], bias=w["toy.b1"])
            return mb.identity(x=mb.mul(x=h, y=w["toy.s"].reshape(1, D_HID)), name="y")

        return [("main", finish(main)), ("alt", finish(alt))]

    fp16 = ct.converters.mil.mil.types.fp16

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_IN), dtype=fp16),
                             mb.StateTensorSpec(shape=(1, D_HID), dtype=fp16)],
                opset_version=target)
    def state(x, acc):
        h = mb.linear(x=x, weight=w["toy.w1"].astype(np.float16), bias=w["toy.b1"].astype(np.float16))
        cur = mb.read_state(input=acc)
        y = mb.add(x=cur, y=h)
        mb.coreml_update_state(state=acc, value=y)
        return mb.identity(x=y, name="y")

    return [("main", finish(state))]


def export(kind, gguf_path, out_dir):
    import tempfile

    import coremltools as ct

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg = out_dir / f"toy-{kind}.mlpackage"
    models = build(kind, read_weights(gguf_path, kind))
    if len(models) == 1:
        models[0][1].save(str(pkg))
        return pkg
    with tempfile.TemporaryDirectory() as tmp:
        desc = ct.utils.MultiFunctionDescriptor()
        for name, m in models:
            p = Path(tmp) / f"{name}.mlpackage"
            m.save(str(p))
            desc.add_function(str(p), src_function_name="main", target_function_name=name)
        desc.default_function_name = "main"
        ct.utils.save_multifunction(desc, str(pkg))
    # save_multifunction writes a fresh spec without the functions' user
    # metadata: set it on the combined spec (a fixed-size change; the
    # weights and the manifest are untouched).
    from coremltools.proto import Model_pb2

    spec_path = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    for k, v in models[0][1].user_defined_metadata.items():
        spec.description.metadata.userDefined[k] = v
    spec_path.write_bytes(spec.SerializeToString(deterministic=True))
    return pkg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--init", help="write the seed GGUF here and exit")
    ap.add_argument("--gguf", help="toy GGUF (f32 tensors)")
    ap.add_argument("--output", help="output directory")
    a = ap.parse_args(argv)
    if a.init:
        write_seed(a.kind, a.init)
        return 0
    if not a.gguf or not a.output:
        ap.error("--gguf and --output are required")
    print(export(a.kind, a.gguf, a.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
