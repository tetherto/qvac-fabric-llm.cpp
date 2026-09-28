#!/usr/bin/env python3
# Rewrite a GGUF with its BF16/F16 output head quantized to Q8_0 (32-wide blocks, gguf-py's numpy quantizer, bit-exact
# with ggml-quants.c). Every other field and tensor is copied unchanged. No sidecar: Q8_0 carries its own scales.
#
# usage: gguf_q8_output_head.py <in.gguf> <out.gguf>
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO / "gguf-py"))
import gguf  # noqa: E402

logger = logging.getLogger("gguf-q8-output-head")

ROWS_PER_CHUNK = 8192


def bf16_rows_to_f32(raw: np.ndarray) -> np.ndarray:
    # bf16 is the upper half of the f32 bit pattern
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = gguf.GGUFReader(src, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    head = next((t for t in reader.tensors if t.name == "output.weight"), None)
    if head is None:
        raise SystemExit("no output.weight tensor (tied embeddings)")
    if head.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F16):
        raise SystemExit(f"output.weight is {head.tensor_type.name}, expected BF16 or F16")

    n_rows = int(head.data.shape[0])
    # the reader exposes BF16 data as raw bytes and F16 as float16: count the elements per row after the view
    raw = np.ascontiguousarray(head.data).reshape(n_rows, -1)
    n_cols = raw.shape[1] // (2 if head.tensor_type == gguf.GGMLQuantizationType.BF16 and raw.dtype == np.uint8 else 1)
    if n_cols % 32 != 0:
        raise SystemExit(f"output.weight row width {n_cols} is not a multiple of the Q8_0 block (32)")
    chunks = []
    for r0 in range(0, n_rows, ROWS_PER_CHUNK):
        rows = raw[r0:r0 + ROWS_PER_CHUNK]
        if head.tensor_type == gguf.GGMLQuantizationType.BF16:
            f32 = bf16_rows_to_f32(rows)
        else:
            f32 = rows.view(np.float16).astype(np.float32)
        f32 = np.ascontiguousarray(f32.reshape(rows.shape[0], n_cols))
        chunks.append(gguf.quants.quantize(f32, gguf.GGMLQuantizationType.Q8_0))
    q_u8 = np.ascontiguousarray(np.concatenate(chunks, axis=0))
    logger.info(f"output.weight [{n_rows}, {n_cols}] {head.tensor_type.name} -> Q8_0 {list(q_u8.shape)} bytes {q_u8.nbytes}")
    total = 0
    for t in reader.tensors:
        if t.name == "output.weight":
            writer.add_tensor_info("output.weight", q_u8.shape, q_u8.dtype, q_u8.nbytes, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            total += q_u8.nbytes
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            total += t.n_bytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done = 0
    for t in reader.tensors:
        if t.name == "output.weight":
            writer.write_tensor_data(q_u8)
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        done += t.n_bytes
        if done % (1 << 32) < t.n_bytes:
            logger.info(f"{done / 2**30:.1f} of {total / 2**30:.1f} GiB")
    writer.close()
    logger.info(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
