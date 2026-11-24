#!/usr/bin/env python3
"""Tests for assemble-cosyvoice3-model.py (stdlib unittest, no fixtures).

Covers the branches that previously produced destructive or unusable
assemblies: cloning add-ons whose names the engine's prefix-based discovery
would never find must be rejected, and an input that already lives in --out
under its final name must be left untouched instead of being deleted before
the copy (or turned into a self-referencing symlink).  Plus the happy paths
for both copy and symlink modes.

Run directly or via ctest (test-assemble-cosyvoice3-model):
    python3 scripts/test_assemble_cosyvoice3_model.py

Method names are kept short on purpose: the TruffleHog CI lane's Lob detector
flags `test_` followed by a long lowercase run as an API key.
"""

import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "assemble-cosyvoice3-model.py")


class AssembleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self._tmp.name, "src")
        self.out = os.path.join(self._tmp.name, "out")
        os.makedirs(self.src)
        os.makedirs(self.out)
        self.required = {}
        for flag, fname in (("--llm", "llm.gguf"), ("--flow", "flow.gguf"),
                            ("--hift", "hift.gguf"), ("--voice", "voice.gguf"),
                            ("--vocab", "vocab.json"), ("--merges", "merges.txt")):
            path = os.path.join(self.src, fname)
            with open(path, "w") as f:
                f.write(fname)
            self.required[flag] = path

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, dirpath, name, content):
        path = os.path.join(dirpath, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _run(self, *extra):
        argv = [sys.executable, SCRIPT]
        for flag, path in self.required.items():
            argv += [flag, path]
        argv += list(extra) + ["--out", self.out]
        return subprocess.run(argv, capture_output=True, text=True)

    def test_copy_mode(self):
        s3tok = self._write(self.src, "cosyvoice3-s3tok-f16.gguf", "s3tok")
        campplus = self._write(self.src, "cosyvoice3-campplus-f32.gguf", "campplus")
        r = self._run("--s3tok", s3tok, "--campplus", campplus)
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ("cosyvoice3-llm-f32.gguf", "cosyvoice3-flow-f32.gguf",
                     "cosyvoice3-hift-f32.gguf", "voice.gguf", "vocab.json",
                     "merges.txt", "cosyvoice3-s3tok-f16.gguf",
                     "cosyvoice3-campplus-f32.gguf"):
            path = os.path.join(self.out, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertFalse(os.path.islink(path), name)
        with open(os.path.join(self.out, "cosyvoice3-s3tok-f16.gguf")) as f:
            self.assertEqual(f.read(), "s3tok")

    def test_symlink_mode(self):
        s3tok = self._write(self.src, "cosyvoice3-s3tok-q8_0.gguf", "s3tok")
        r = self._run("--s3tok", s3tok, "--symlink")
        self.assertEqual(r.returncode, 0, r.stderr)
        link = os.path.join(self.out, "cosyvoice3-s3tok-q8_0.gguf")
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.path.realpath(link), os.path.realpath(s3tok))
        with open(link) as f:
            self.assertEqual(f.read(), "s3tok")

    def test_in_place_input_kept(self):
        in_place = self._write(self.out, "cosyvoice3-campplus-f32.gguf", "campplus-bytes")
        for mode in ((), ("--symlink",)):
            r = self._run("--campplus", in_place, *mode)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("keep", r.stdout)
            self.assertFalse(os.path.islink(in_place), mode)
            with open(in_place) as f:
                self.assertEqual(f.read(), "campplus-bytes", mode)

    def test_bad_addon_names_rejected(self):
        for flag, name in (("--s3tok", "tokenizer.gguf"),
                           ("--s3tok", "cosyvoice3-s3tok-f16.bin"),
                           ("--campplus", "campplus.gguf")):
            bad = self._write(self.src, name, "x")
            r = self._run(flag, bad)
            self.assertNotEqual(r.returncode, 0, (flag, name))
            self.assertIn("does not match", r.stderr, (flag, name))
            self.assertFalse(
                os.path.exists(os.path.join(self.out, name)), (flag, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
