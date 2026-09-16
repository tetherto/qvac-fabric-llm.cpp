#!/usr/bin/env python3
# Rewrite an F8_E4M3 GGUF with its BF16/F16 output head quantized to F8_E4M3 + 128x128 block scales, using the
# converter's quantizer (the same result as convert_hf_to_gguf.py --outtype fp8 --fp8-output-head without the
# safetensors checkpoint). Every other field and tensor is copied unchanged.
#
# usage: gguf_fp8_output_head.py <in.gguf> <out.gguf>
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO / "gguf-py"))
sys.path.insert(1, str(REPO))
import gguf  # noqa: E402
from conversion.base import fp8_block_quantize  # noqa: E402

logger = logging.getLogger("gguf-fp8-output-head")


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
    if any(t.name == "output.scale" for t in reader.tensors):
        raise SystemExit("output.scale exists already")

    raw = np.ascontiguousarray(head.data)
    if head.tensor_type == gguf.GGMLQuantizationType.BF16:
        w = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).float()
    else:
        w = torch.from_numpy(raw.view(np.float16)).float()
    w = w.reshape(head.data.shape[0], -1)
    block = gguf.GGML_F8_E4M3_SCALE_BLOCK
    q_u8, s_f32 = fp8_block_quantize(w, block)
    logger.info(f"output.weight {list(w.shape)} {head.tensor_type.name} -> F8_E4M3 + scale {list(s_f32.shape)}")

    total = 0
    for t in reader.tensors:
        if t.name == "output.weight":
            writer.add_tensor_info("output.weight", q_u8.shape, q_u8.dtype, q_u8.nbytes, gguf.GGMLQuantizationType.F8_E4M3)
            writer.add_tensor_info("output.scale", s_f32.shape, s_f32.dtype, s_f32.nbytes, gguf.GGMLQuantizationType.F32)
            total += q_u8.nbytes + s_f32.nbytes
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
            writer.write_tensor_data(s_f32)
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        done += t.n_bytes
        if done % (1 << 32) < t.n_bytes:
            logger.info(f"{done / 2**30:.1f} of {total / 2**30:.1f} GiB")
    writer.close()
    logger.info(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
