#!/usr/bin/env python3
"""Check the native quantizer's serialized blocks and alignment/error contracts.

Small synthetic streams exercise the real executable. The reader below uses
the file format's block sizes independently of the quantizer implementation.
This tests serialization, not model loading, inference, or model quality.
"""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


MAGIC = 0x67676D6C
WEIGHT = "encoder.blocks.0.attn.query.weight"
# name: (ggml tensor type, output model ftype, values/block, bytes/block,
#        position of the FP16 scale within a block)
FORMATS = {
    "q3_k": (11, 11, 256, 110, 108),
    "q4_k": (12, 12, 256, 144, 0),
    "q4_0": (2, 2, 32, 18, 0),
    "q8_0": (8, 7, 32, 34, 0),
}
BLOCKS = {0: (1, 4), 1: (1, 2)}
BLOCKS.update({t: (n, size) for t, _, n, size, _ in FORMATS.values()})


def write_model(path, width, source_type=0):
    tokens = [b"a", b"\xc3\xa9", b""]
    hparams = (len(tokens), 4, width, 1, 4, 4, width, 1, 4, 2, source_type)
    filters = (0.25, 0.5, 0.75, 1.0)
    tensors = [
        (WEIGHT, (width, 2)),
        ("encoder.positional_embedding", (width, 2)),  # explicitly skipped
        ("encoder.ln_post.weight", (width,)),  # only matrices are quantized
        ("encoder.conv1.weight", (3, 2, 1)),
    ]
    dtype = "f" if source_type == 0 else "e"
    with path.open("wb") as stream:
        stream.write(struct.pack("<I11i2i4fi", MAGIC, *hparams, 2, 2,
                                 *filters, len(tokens)))
        for token in tokens:
            stream.write(struct.pack("<I", len(token)))
            stream.write(token)
        for name, shape in tensors:
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<3i", len(shape), len(encoded), source_type))
            stream.write(struct.pack("<" + "i" * len(shape), *shape))
            stream.write(encoded)
            values = [((i % 37) - 18) / 16.0 for i in range(math.prod(shape))]
            stream.write(struct.pack("<" + dtype * len(values), *values))


def read_model(path):
    data = path.read_bytes()
    position = 0

    def take(size):
        nonlocal position
        if size < 0 or position + size > len(data):
            raise ValueError("truncated serialized model")
        result = data[position:position + size]
        position += size
        return result

    def integers(count):
        return struct.unpack("<" + "i" * count, take(4 * count))

    if integers(1)[0] != MAGIC:
        raise ValueError("invalid model magic")
    hparams = integers(11)
    n_mels, n_fft = integers(2)
    filters = take(n_mels * n_fft * 4)
    tokens = [take(integers(1)[0]) for _ in range(integers(1)[0])]
    tensors = {}
    while position < len(data):
        rank, name_size, dtype = integers(3)
        if not 1 <= rank <= 3 or name_size <= 0 or dtype not in BLOCKS:
            raise ValueError("invalid tensor header")
        shape = integers(rank)
        if any(n <= 0 for n in shape):
            raise ValueError("invalid tensor dimension")
        name = take(name_size).decode("utf-8")
        block, block_bytes = BLOCKS[dtype]
        if shape[0] % block:
            raise ValueError("tensor row is not block aligned")
        payload = take(math.prod(shape) // block * block_bytes)
        if name in tensors:
            raise ValueError("duplicate tensor")
        tensors[name] = (shape, dtype, payload)
    return hparams, (n_mels, n_fft, filters), tokens, tensors


class QuantizeTest(unittest.TestCase):
    quantizer = None

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="whisper-quantize-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def quantize(self, source, name, expected=0):
        output = self.root / (source.stem + "-" + name + ".bin")
        result = subprocess.run([str(self.quantizer), str(source), str(output), name],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, expected,
                         result.stdout + "\n" + result.stderr)
        return output, result.stderr

    def check_serialization(self, width, name, source_type=0):
        source = self.root / ("source-%d-%d.bin" % (width, source_type))
        write_model(source, width, source_type)
        original = read_model(source)
        output, _ = self.quantize(source, name)
        hparams, filters, tokens, tensors = read_model(output)
        tensor_type, ftype, block, size, scale_offset = FORMATS[name]
        self.assertEqual(hparams[:-1], original[0][:-1])
        self.assertEqual(hparams[-1], 2000 + ftype)  # quantization version 2
        self.assertEqual(filters, original[1])
        self.assertEqual(tokens, original[2])
        self.assertEqual(set(tensors), set(original[3]))
        for key, before in original[3].items():
            actual = tensors[key]
            if key != WEIGHT:
                self.assertEqual(actual, before, key)
                continue
            self.assertEqual(actual[:2], ((width, 2), tensor_type))
            self.assertEqual(len(actual[2]), width * 2 // block * size)
            for offset in range(0, len(actual[2]), size):
                scale = struct.unpack_from("<e", actual[2], offset + scale_offset)[0]
                self.assertTrue(math.isfinite(scale))

    def test_aligned_rows_serialize_complete_blocks(self):
        for source_type in (0, 1):
            for name in FORMATS:
                with self.subTest(name=name, source_type=source_type):
                    self.check_serialization(256, name, source_type)

    def test_width_384_accepts_32_element_blocks(self):
        for name in ("q4_0", "q8_0"):
            with self.subTest(name=name):
                self.check_serialization(384, name)

    def test_unaligned_k_quant_rows_fail_clearly(self):
        # Two rows contain 768 values, a multiple of 256. The ROW width384
        # still cannot contain complete K-quant blocks and must be rejected.
        for source_type in (0, 1):
            source = self.root / ("unaligned-%d.bin" % source_type)
            write_model(source, 384, source_type)
            for name in ("q3_k", "q4_k"):
                with self.subTest(name=name, source_type=source_type):
                    _, diagnostic = self.quantize(source, name, expected=1)
                    self.assertIn(WEIGHT, diagnostic)
                    self.assertRegex(diagnostic, r"row size 384.*block size 256")

    def test_unknown_formats_are_errors(self):
        source = self.root / "source.bin"
        write_model(source, 256)
        for name in ("q3", "q4", "q_unknown"):
            with self.subTest(name=name):
                _, diagnostic = self.quantize(source, name, expected=1)
                self.assertIn("unknown ftype", diagnostic)

    def test_requantizing_integer_weights_is_an_error(self):
        source = self.root / "source.bin"
        write_model(source, 256)
        quantized, _ = self.quantize(source, "q4_0")
        _, diagnostic = self.quantize(quantized, "q8_0", expected=1)
        self.assertIn("unsupported ttype 2", diagnostic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantizer", required=True, type=Path)
    options, unittest_args = parser.parse_known_args()
    QuantizeTest.quantizer = options.quantizer.resolve(strict=True)
    unittest.main(argv=[sys.argv[0], *unittest_args])
