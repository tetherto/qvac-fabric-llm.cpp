#!/usr/bin/env python3
"""Prepare shape-specific public Core ML programs (m = outputs, n = tokens, k = inputs)."""
import argparse
from pathlib import Path


def parse_shape(value):
    try:
        m, n, k = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MxNxK") from exc
    if not (256 <= m <= 8192 and 64 <= n <= 2048 and 256 <= k <= 8192):
        raise argparse.ArgumentTypeError("shape outside prototype limits")
    if any(v % 32 for v in (m, n, k)) or m*n*k < 16*1024*1024 or 2*(m*k+n*k+m*n) > 128*1024*1024:
        raise argparse.ArgumentTypeError("shape rejected by alignment, work or scratch limit")
    return m, n, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=parse_shape, action="append", required=True)
    args = parser.parse_args()
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    args.output.mkdir(parents=True, exist_ok=True)
    for m, n, k in args.shape:
        @mb.program(input_specs=[mb.TensorSpec((1, 1, m, k), dtype=types.fp16),
                                 mb.TensorSpec((1, 1, n, k), dtype=types.fp16)],
                    opset_version=ct.target.iOS18)
        def program(w, x):
            return mb.matmul(x=x, y=w, transpose_y=True, name="y")

        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        model.short_description = "GGML Q8_0 unpacked by Metal; dynamic FP16 matmul; schema v1"
        path = args.output / f"q8-fp16-v1-{m}-{n}-{k}.mlpackage"
        if path.exists():
            raise FileExistsError(f"refusing to replace {path}")
        model.save(str(path))
        print(path)


if __name__ == "__main__":
    main()
