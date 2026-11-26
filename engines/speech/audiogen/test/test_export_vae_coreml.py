#!/usr/bin/env python3
"""Unit tests for scripts/export-vae-coreml.py that need no model file.

make_ane_safe() must be idempotent: one loaded decoder serves several exports
in one process (float16 plus palettized variants), so a second pass must not
re-wrap an already-rewritten upsample stage. Requires torch; skips without it,
mirroring the converter tests' dependency handling.
"""
import importlib.util
import pathlib
import unittest

try:
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'export-vae-coreml.py'


def load_exporter():
    spec = importlib.util.spec_from_file_location('export_vae_coreml', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class MakeAneSafeTest(unittest.TestCase):
    def setUp(self):
        self.exporter = load_exporter()
        self.model = self.exporter.OobleckDecoder()

    def broken_stride_blocks(self):
        return [i for i, stride in enumerate(self.exporter.STRIDES)
                if stride in self.exporter.ANE_BROKEN_DECONV_STRIDES]

    def test_rewrites_only_broken_strides(self):
        self.exporter.make_ane_safe(self.model)
        broken = self.broken_stride_blocks()
        self.assertTrue(broken)
        for i, block in enumerate(self.model.blocks):
            if i in broken:
                self.assertIsInstance(block.conv_t1, self.exporter.PhaseUpsample)
                self.assertIsInstance(block.conv_t1.conv, nn.Conv1d)
            else:
                self.assertIsInstance(block.conv_t1, nn.ConvTranspose1d)

    def test_second_pass_changes_nothing(self):
        self.exporter.make_ane_safe(self.model)
        first_pass = [id(block.conv_t1) for block in self.model.blocks]
        self.exporter.make_ane_safe(self.model)
        second_pass = [id(block.conv_t1) for block in self.model.blocks]
        self.assertEqual(first_pass, second_pass)


if __name__ == '__main__':
    unittest.main()
