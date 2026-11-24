"""Exercise the stage parity CLI with finite controls and invalid dumps."""

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
class StageCosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.a = self.root / "a"
        self.b = self.root / "b"
        self.a.mkdir()
        self.b.mkdir()
        self.script = pathlib.Path(__file__).parents[1] / "scripts" / "stage_cos.py"

    def tearDown(self):
        self.tmp.cleanup()

    def write_dump(self, directory, values, name="dit.bin", dims=None, suffix=b""):
        dims = dims if dims is not None else (2, 1, len(values))
        (directory / name).write_bytes(
            struct.pack("iii", *dims) + struct.pack("%df" % len(values), *values) + suffix)

    def run_cli(self, expected):
        proc = subprocess.run(
            [sys.executable, str(self.script), str(self.a), str(self.b)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        return proc

    def test_finite(self):
        for values, expected in (([1.0, 2.0], 0), ([-1.0, -2.0], 1)):
            with self.subTest(values=values):
                self.write_dump(self.a, [1.0, 2.0])
                self.write_dump(self.b, values)
                self.run_cli(expected)

    def test_zeros(self):
        for values, expected in (([0.0, 0.0], 0), ([1.0, 2.0], 1)):
            with self.subTest(values=values):
                self.write_dump(self.a, [0.0, 0.0])
                self.write_dump(self.b, values)
                self.run_cli(expected)

    def test_nonfinite(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for side in ("a", "b", "both"):
                with self.subTest(value=value, side=side):
                    self.write_dump(self.a, [value if side != "b" else 1.0, 2.0])
                    self.write_dump(self.b, [value if side != "a" else 1.0, 2.0])
                    proc = self.run_cli(1)
                    self.assertIn("nonfinite", proc.stderr)

    def test_roster(self):
        self.write_dump(self.a, [1.0, 2.0])
        self.write_dump(self.b, [1.0, 2.0])
        for side in (self.a, self.b):
            with self.subTest(side=side.name):
                self.write_dump(side, [1.0], name="lm.bin")
                self.assertIn("roster mismatch", self.run_cli(1).stderr)
                (side / "lm.bin").unlink()

    def test_header(self):
        for dims in ((3, 1, 2), (2, 0, 2), (2, -1, -2)):
            with self.subTest(dims=dims):
                self.write_dump(self.a, [1.0, 2.0], dims=dims)
                self.write_dump(self.b, [1.0, 2.0], dims=dims)
                self.assertIn("nonempty 2-D", self.run_cli(1).stderr)

    def test_payload(self):
        for values, suffix in (([1.0], b""), ([1.0, 2.0, 3.0], b""),
                               ([1.0, 2.0], b"x")):
            with self.subTest(values=values, suffix=suffix):
                self.write_dump(self.a, [1.0, 2.0])
                self.write_dump(self.b, values, dims=(2, 1, 2), suffix=suffix)
                self.assertIn("payload holds", self.run_cli(1).stderr)

    def test_shape(self):
        self.write_dump(self.a, [1.0, 2.0], dims=(2, 1, 2))
        self.write_dump(self.b, [1.0, 2.0], dims=(2, 2, 1))
        self.assertIn("shape mismatch", self.run_cli(1).stderr)

    def test_truncated(self):
        self.write_dump(self.a, [1.0, 2.0])
        (self.b / "dit.bin").write_bytes(b"1234")
        self.assertIn("truncated header", self.run_cli(1).stderr)

    def test_empty(self):
        self.assertIn("no *.bin", self.run_cli(1).stderr)


if __name__ == "__main__":
    unittest.main()
