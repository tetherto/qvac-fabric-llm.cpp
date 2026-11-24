"""Exercise invalid streaming fixtures without loading a model."""

import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = sys.argv.pop(1)


def save_tokens(path, tokens):
    header = f"{{'descr': '<i4', 'fortran_order': False, 'shape': ({len(tokens)},), }}\n"
    payload = struct.pack(f"<{len(tokens)}i", *tokens)
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + payload)


class StreamingValidationTests(unittest.TestCase):
    def reject(self, tokens, chunk, message):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            save_tokens(root / "speech_tokens.npy", tokens)
            if chunk is not None:
                save_tokens(root / "chunk_01_tokens.npy", chunk)
            result = subprocess.run(
                [BINARY, str(root / "unused.gguf"), str(root)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, result.stderr)

    def test_empty_global_tokens(self):
        self.reject([], None, "empty speech tokens")

    def test_no_chunks(self):
        self.reject([1, 2], None, "do not cover the complete token sequence")

    def test_empty_chunk(self):
        self.reject([1, 2], [], "chunk tokens must advance")

    def test_oversized_prefix(self):
        self.reject([1, 2], [1, 2, 3], "chunk tokens must advance")

    def test_wrong_prefix(self):
        self.reject([1, 2], [3], "chunk tokens diverge")

    def test_missing_noise(self):
        self.reject([1, 2], [1], "missing per-chunk CFM noise")


if __name__ == "__main__":
    unittest.main()
