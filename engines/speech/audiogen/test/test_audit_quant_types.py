"""Tests for scripts/audit_quant_types.py.

The audit is the last check between a quantized GGUF and a shipped model
pair, so the deny-list is pinned tensor by tensor: every protected name must
fail when quantized, and every legitimately quantized neighbour must pass.
A stub `gguf` module stands in for the reader, so no third-party package is
needed to run this.

Method names are kept short on purpose: the TruffleHog CI lane's Lob detector
flags `test_` followed by a long lowercase run as an API key.
"""

import importlib.util
import io
import pathlib
import sys
import types
import unittest
from unittest import mock

PLACEHOLDER_BYTES = 1024
UNQUANTIZED = ["F32", "F16", "BF16"]
PROTECTED = [
    "cond.proj.weight",
    "voc.conv_in.weight",
    "dit.time_fourier.weight",
    "depth.pos_embd.weight",
]
QUANTIZABLE = [
    "depth.blk.0.attn_v.weight",
    "depth.proj.weight",
    "depth.audio_embd.weight",
    "dit.blk.0.attn_qkv.weight",
    "blk.0.attn_v.weight",
]


def load_audit():
    stub = types.ModuleType("gguf")
    stub.GGUFReader = object
    sys.modules.setdefault("gguf", stub)
    path = pathlib.Path(__file__).parents[1] / "scripts" / "audit_quant_types.py"
    spec = importlib.util.spec_from_file_location("audit_quant_types", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_audit()


class FakeType:
    def __init__(self, name):
        self.name = name


class FakeTensor:
    def __init__(self, name, type_name, n_bytes=PLACEHOLDER_BYTES):
        self.name = name
        self.tensor_type = FakeType(type_name)
        self.n_bytes = n_bytes


class FakeReader:
    def __init__(self, tensors):
        self.tensors = tensors


def reader_with(names, type_name):
    return FakeReader([FakeTensor(name, type_name) for name in names])


def flagged_names(reader, deny=None):
    deny = audit.DEFAULT_DENY if deny is None else deny
    return [name for name, _ in audit.deny_list_violations(reader, deny)]


class DenyListTests(unittest.TestCase):
    def assert_flagged(self, names, type_name):
        self.assertEqual(flagged_names(reader_with(names, type_name)), names)

    def assert_clean(self, names, type_name):
        self.assertEqual(flagged_names(reader_with(names, type_name)), [])

    def test_pos_embd_quantized(self):
        self.assert_flagged(["depth.pos_embd.weight"], "Q8_0")

    def test_depth_attn_v_quantized(self):
        self.assert_clean(["depth.blk.0.attn_v.weight"], "Q8_0")

    def test_protected_quantized(self):
        self.assert_flagged(PROTECTED, "Q4_K")

    def test_protected_unquantized(self):
        for type_name in UNQUANTIZED:
            self.assert_clean(PROTECTED, type_name)

    def test_quantizable_pass(self):
        for type_name in ("Q8_0", "Q4_K", "Q6_K"):
            self.assert_clean(QUANTIZABLE, type_name)

    def test_mixed_file(self):
        reader = FakeReader([
            FakeTensor("depth.pos_embd.weight", "Q8_0"),
            FakeTensor("depth.blk.0.attn_v.weight", "Q8_0"),
            FakeTensor("cond.proj.weight", "F16"),
            FakeTensor("dit.blk.0.attn_qkv.weight", "Q4_K"),
        ])
        self.assertEqual(flagged_names(reader), ["depth.pos_embd.weight"])

    def test_custom_deny(self):
        reader = reader_with(["dit.blk.0.attn_qkv.weight"], "Q4_K")
        self.assertEqual(flagged_names(reader, ["dit.blk."]), ["dit.blk.0.attn_qkv.weight"])

    def test_empty_deny(self):
        self.assertEqual(flagged_names(reader_with(QUANTIZABLE, "Q4_K"), []), [])

    def test_reported_type(self):
        reader = reader_with(["depth.pos_embd.weight"], "Q8_0")
        self.assertEqual(audit.deny_list_violations(reader, audit.DEFAULT_DENY),
                         [("depth.pos_embd.weight", "Q8_0")])


class ByteSummaryTests(unittest.TestCase):
    def test_totals(self):
        reader = FakeReader([
            FakeTensor("a", "Q4_K", 100),
            FakeTensor("b", "Q4_K", 50),
            FakeTensor("c", "F32", 25),
        ])
        self.assertEqual(audit.bytes_by_type(reader), {"Q4_K": 150, "F32": 25})

    def test_empty(self):
        self.assertEqual(audit.bytes_by_type(FakeReader([])), {})


class MainTests(unittest.TestCase):
    def run_main(self, argv, reader):
        with mock.patch.object(audit, "GGUFReader", lambda path: reader), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(sys, "stdout", io.StringIO()) as out:
            return audit.main(), out.getvalue()

    def test_clean_exits_zero(self):
        reader = reader_with(["depth.blk.0.attn_v.weight"], "Q8_0")
        code, output = self.run_main(["audit_quant_types.py", "in.gguf"], reader)
        self.assertEqual(code, 0)
        self.assertIn("PASS", output)

    def test_violation_exits_one(self):
        reader = reader_with(["depth.pos_embd.weight"], "Q8_0")
        code, output = self.run_main(["audit_quant_types.py", "in.gguf"], reader)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_argv_deny(self):
        reader = reader_with(["dit.blk.0.attn_qkv.weight"], "Q4_K")
        code, _ = self.run_main(["audit_quant_types.py", "in.gguf", "dit.blk."], reader)
        self.assertEqual(code, 1)

    def test_argv_deny_drops_empty(self):
        # A trailing comma must not leave an empty substring, which would
        # match every tensor name and fail the whole file.
        reader = reader_with(["dit.blk.0.attn_qkv.weight"], "Q4_K")
        code, _ = self.run_main(["audit_quant_types.py", "in.gguf", "voc.,"], reader)
        self.assertEqual(code, 0)

    def test_usage(self):
        code, output = self.run_main(["audit_quant_types.py"], FakeReader([]))
        self.assertEqual(code, 1)
        self.assertIn("usage", output)


if __name__ == "__main__":
    sys.exit(unittest.main())
