#!/usr/bin/env python3
"""Exercise the real quality CLI against local generated models; no downloads."""

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ARGS = argparse.Namespace()


def diagnostic(process, tag):
    for line in process.stderr.decode(errors="replace").splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    raise AssertionError(f"missing {tag}: {process.stderr.decode(errors='replace')}")


def unpack_reference(data):
    if data[:8] != b"LLQLOGIT":
        raise AssertionError("stdout reference contaminated or missing")
    version, length = struct.unpack_from("<II", data, 8)
    if version != 1:
        raise AssertionError(f"unexpected reference version: {version}")
    meta = json.loads(data[16:16 + length])
    offset = 16 + length
    rows = []
    for _ in range(meta["rows"]):
        phase, position = struct.unpack_from("<II", data, offset)
        logits = struct.unpack_from(f"<{meta['vocab']}f", data, offset + 8)
        rows.append((phase, position, logits, offset))
        offset += 8 + meta["vocab"] * 4
    if offset != len(data):
        raise AssertionError("reference payload does not match declared rows")
    return meta, rows


class QualityToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="llama-quality-")
        cls.root = Path(cls.temporary.name)
        cls.tokens = [(i * 37 + 11) % 128 for i in range(49)]
        cls.token_file = cls.root / "tokens.json"
        cls.token_file.write_text(json.dumps(cls.tokens))
        cls.model = Path(ARGS.models) / "qwen35-dense.gguf"
        cls.dense_model = Path(ARGS.models) / "llama-dense.gguf"
        cls.record = cls.invoke("--quality-record", "-")
        if cls.record.returncode:
            raise AssertionError(cls.record.stderr.decode())
        cls.meta, cls.rows = unpack_reference(cls.record.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def invoke(cls, *extra, data=None, binary=None, model=None, token_file=None,
               prefix=24, width=4, rs=3, rollback=3, context=128, batch=64, ubatch=32, correct_slot=False):
        command = [binary or ARGS.tool, "-m", str(model or cls.model), "-ngl", "0",
                   "-t", "2", "-tb", "2", "-c", str(context), "-b", str(batch), "-ub", str(ubatch),
                   "--quality-tokens", str(token_file or cls.token_file),
                   "--quality-prefix", str(prefix), "--quality-width", str(width),
                   "--quality-rs", str(rs), "--quality-rollback", str(rollback), *extra]
        if correct_slot:
            command.insert(1, "--check-correct-slot")
        return subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)

    def compare(self, data=None, **kwargs):
        return self.invoke("--quality-reference", "-", data=self.record.stdout if data is None else data, **kwargs)

    def assert_error(self, result, text):
        self.assertEqual(result.returncode, 1, result.stderr.decode(errors="replace"))
        error = diagnostic(result, "quality_error")
        self.assertIn(text, error["error"])
        self.assertFalse(error["passed"])
        self.assertNotIn(b"quality_metrics ", result.stderr)

    def test_self_reference_and_next_token_alignment(self):
        result = self.compare()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"")
        metrics = diagnostic(result, "quality_metrics")
        self.assertEqual(metrics["exact_match_count"], len(self.rows))
        self.assertEqual(metrics["top_token_agreement"], 1)
        self.assertEqual(metrics["mean_kl"], 0)
        self.assertEqual(metrics["perplexity_ratio"], 1)
        self.assertTrue(metrics["replay"]["passed"])
        # Independent oracle for row labels and next-token NLL. In particular,
        # the final prefill logit predicts the FIRST continuation token.
        expected = [(0, 23)]
        for begin in range(24, len(self.tokens), 4):
            end = min(begin + 4, len(self.tokens))
            expected += [(1, pos) for pos in range(begin, min(end, len(self.tokens) - 1))]
            expected += [(2, pos) for pos in range(end - min(3, end - begin), min(end, len(self.tokens) - 1))]
        self.assertEqual([(phase, pos) for phase, pos, _, _ in self.rows], expected)
        nll = 0
        for _, position, logits, _ in self.rows:
            maximum = max(logits)
            nll += math.log(math.fsum(math.exp(x - maximum) for x in logits))
            nll -= logits[self.tokens[position + 1]] - maximum
        self.assertAlmostEqual(metrics["reference_mean_nll"], nll / len(self.rows), places=10)

    def test_dense_and_recurrent_schedules(self):
        for model, width, rs, rollback in [
            (self.dense_model, 1, 0, 0), (self.dense_model, 4, 3, 1),
            (self.model, 1, 0, 0), (self.model, 4, 3, 0),
            (self.model, 4, 3, 1), (self.model, 6, 5, 5), (self.model, 8, 7, 7),
        ]:
            with self.subTest(model=model.name, width=width, rollback=rollback):
                reference = self.invoke("--quality-record", "-", model=model, width=width, rs=rs, rollback=rollback)
                self.assertEqual(reference.returncode, 0, reference.stderr.decode())
                result = self.compare(reference.stdout, model=model, width=width, rs=rs, rollback=rollback)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                metrics = diagnostic(result, "quality_metrics")
                self.assertEqual(metrics["exact_match_count"], metrics["rows"])

    def test_corrupted_finite_logits_fail_gate(self):
        corrupted = bytearray(self.record.stdout)
        # Change every distribution to a sharply wrong token, not its row label.
        for _, _, logits, offset in self.rows:
            wrong = (max(range(len(logits)), key=logits.__getitem__) + 1) % len(logits)
            struct.pack_into("<f", corrupted, offset + 8 + 4 * wrong, max(logits) + 100)
        result = self.compare(corrupted)
        self.assertEqual(result.returncode, 1, result.stderr.decode())
        metrics = diagnostic(result, "quality_metrics")
        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["top_token_agreement"], 0)
        self.assertGreater(metrics["mean_kl"], 0.002)

    def test_nonfinite_reference_rejected(self):
        for bits in [0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001]:
            with self.subTest(bits=hex(bits)):
                corrupted = bytearray(self.record.stdout)
                struct.pack_into("<I", corrupted, self.rows[0][3] + 8, bits)
                self.assert_error(self.compare(corrupted), "non-finite logit")
                # The fault-helper uses fast-math; this first row is checked
                # before any rollback, so it independently exercises bit tests.
                self.assert_error(self.compare(corrupted, binary=ARGS.wrong_slot_tool), "non-finite logit")

    def test_header_row_and_stream_integrity(self):
        corruptions = [
            (self.record.stdout[:-1], "truncated"),
            (self.record.stdout + b"extra", "trailing bytes"),
            (b"badmagic" + self.record.stdout[8:], "magic"),
            (self.record.stdout[:8] + struct.pack("<I", 2) + self.record.stdout[12:], "version"),
        ]
        wrong_row = bytearray(self.record.stdout)
        struct.pack_into("<I", wrong_row, self.rows[0][3] + 4, 22)
        corruptions.append((wrong_row, "row schedule mismatch"))
        wrong_rows = dict(self.meta, rows=self.meta["rows"] + 1)
        header = json.dumps(wrong_rows, separators=(",", ":")).encode()
        corruptions.append((b"LLQLOGIT" + struct.pack("<II", 1, len(header)) + header +
                            self.record.stdout[self.rows[0][3]:], "metadata mismatch"))
        for data, error in corruptions:
            with self.subTest(error=error):
                self.assert_error(self.compare(data), error)
        self.assert_error(self.compare(width=6), "metadata mismatch")
        changed = self.root / "changed.json"
        changed.write_text(json.dumps([self.tokens[0] ^ 1, *self.tokens[1:]]))
        self.assert_error(self.compare(token_file=changed), "metadata mismatch")

    def test_invalid_inputs_and_schedule(self):
        for values, message in [
            (self.tokens[:24], "continuation"),
            ([*self.tokens[:-1], 128], "outside model vocabulary"),
            ([*self.tokens[:-1], -1], "invalid integer"),
            ([*self.tokens[:-1], 1.5], "invalid integer"),
            ([*self.tokens[:-1], True], "invalid integer"),
            ({"tokens": self.tokens}, "JSON array"),
        ]:
            with self.subTest(message=message):
                bad = self.root / "bad.json"
                bad.write_text(json.dumps(values))
                self.assert_error(self.compare(token_file=bad), message)
        for kwargs, message in [
            ({"width": 0}, "width"), ({"prefix": 0}, "prefix"),
            ({"rollback": 4}, "rollback"), ({"rs": 2}, "rollback"),
            ({"context": 32}, "context"), ({"width": 33}, "ubatch"),
        ]:
            with self.subTest(kwargs=kwargs):
                self.assert_error(self.compare(**kwargs), message)
        self.assert_error(self.invoke("--quality-record", "-", "--quality-reference", "-"), "exactly one")

    def test_named_files(self):
        path = self.root / "reference.bin"
        recorded = self.invoke("--quality-record", str(path))
        self.assertEqual(recorded.returncode, 0, recorded.stderr.decode())
        self.assertEqual(recorded.stdout, b"")
        compared = self.invoke("--quality-reference", str(path))
        self.assertEqual(compared.returncode, 0, compared.stderr.decode())
        self.assertEqual(diagnostic(compared, "quality_metrics")["exact_match_count"], len(self.rows))

    def test_planted_wrong_rollback_slot_fails(self):
        # The fixed numerical gate can legitimately accept this tiny model's
        # almost-invisible logit drift. A separate same-prefix cache invariant
        # must pass for correct replay and fail for the actual wrong-slot fault.
        correct = self.compare(binary=ARGS.wrong_slot_tool, correct_slot=True)
        self.assertEqual(correct.returncode, 0, correct.stderr.decode())
        self.assertTrue(diagnostic(correct, "quality_metrics")["passed"])
        self.assert_error(self.compare(binary=ARGS.wrong_slot_tool), "same-prefix replay state mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--wrong-slot-tool", required=True)
    parser.add_argument("--models", required=True)
    ARGS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__, *remaining])
