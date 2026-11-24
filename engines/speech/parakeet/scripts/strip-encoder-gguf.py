#!/usr/bin/env python3
"""Remove encoder.* tensors for use with matching Core ML encoder sidecars.

Requires numpy and gguf. Metadata and retained tensor payloads are copied without
conversion. The result requires a compatible Core ML encoder at runtime.
"""

import argparse
from pathlib import Path
import struct


ENCODER_PREFIX = "encoder."
COPY_BUFFER_BYTES = 1024 * 1024


def align_offset(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment


def copy_tensor(source, output, tensor):
    source.seek(tensor.data_offset)
    remaining = tensor.n_bytes
    while remaining:
        chunk = source.read(min(remaining, COPY_BUFFER_BYTES))
        if not chunk:
            raise ValueError(f"Truncated tensor payload: {tensor.name}")
        output.write(chunk)
        remaining -= len(chunk)


def write_tensor_directory(output, tensors, byte_order, alignment):
    offsets = []
    offset = 0
    for tensor in tensors:
        offsets.append(offset)
        # ReaderField parts retain the serialized name, dimensions and type.
        for part in tensor.field.parts[:-1]:
            output.write(part.tobytes())
        output.write(struct.pack(f"{byte_order}Q", offset))
        offset = align_offset(offset + tensor.n_bytes, alignment)
    return offsets


def copy_retained_tensors(source, output, tensors, offsets, alignment):
    data_start = align_offset(output.tell(), alignment)
    output.write(bytes(data_start - output.tell()))
    for tensor, offset in zip(tensors, offsets):
        output.write(bytes(data_start + offset - output.tell()))
        copy_tensor(source, output, tensor)
    output.write(bytes(align_offset(output.tell(), alignment) - output.tell()))


def strip_encoder(gguf_path, output_path):
    import gguf

    gguf_path, output_path = Path(gguf_path), Path(output_path)
    if gguf_path.resolve() == output_path.resolve():
        raise ValueError("Input and output GGUF paths must differ")
    reader = gguf.GGUFReader(gguf_path)
    tensors = [tensor for tensor in reader.tensors if not tensor.name.startswith(ENCODER_PREFIX)]
    removed = len(reader.tensors) - len(tensors)
    if removed == 0:
        raise ValueError("Input GGUF contains no encoder.* tensors")
    byte_order = "<" if reader.endianess == gguf.GGUFEndian.LITTLE else ">"
    metadata_end = reader.tensors[0].field.offset
    count_offset = reader.fields["GGUF.tensor_count"].offset
    with gguf_path.open("rb") as source:
        metadata = bytearray(source.read(metadata_end))
        struct.pack_into(f"{byte_order}Q", metadata, count_offset, len(tensors))
        # Exclusive creation also protects existing outputs and hard-linked input.
        output = output_path.open("xb")
        try:
            with output:
                output.write(metadata)
                offsets = write_tensor_directory(output, tensors, byte_order, reader.alignment)
                copy_retained_tensors(source, output, tensors, offsets, reader.alignment)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return removed, len(tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True, help="Full source GGUF")
    parser.add_argument("--out", type=Path, required=True, help="New encoder-less GGUF (must not exist)")
    args = parser.parse_args()
    removed, retained = strip_encoder(args.gguf, args.out)
    print(f"Removed {removed} encoder tensors; retained {retained} tensors: {args.out}")


if __name__ == "__main__":
    main()
