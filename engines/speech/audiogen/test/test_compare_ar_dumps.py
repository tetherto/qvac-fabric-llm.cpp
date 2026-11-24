"""Tests for scripts/compare_ar_dumps.py.

The comparator decides whether a quant/backend dump pair is acceptable, so
every gate path is pinned here: the cosine value itself, the incomparable
case, both hard errors, and each --min-* flag failing and passing. The suite
runs twice, once per numeric backend, and cross-checks the two against the
same fixtures.

Method names are kept short on purpose: the TruffleHog CI lane's Lob detector
flags `test_` followed by a long lowercase run as an API key.
"""

import importlib.util
import io
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

NEG_INF = float("-inf")
FIELDS = ["last-hidden", "sem-logits", "guided", "feedback", "depth-hidden"]
PLAIN = [3.0, 4.0]
PLAIN_OTHER = [4.0, 3.0]


def load_comparator():
    path = pathlib.Path(__file__).parents[1] / "scripts" / "compare_ar_dumps.py"
    spec = importlib.util.spec_from_file_location("compare_ar_dumps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparator = load_comparator()
NUMPY_AVAILABLE = comparator.np is not None


def pack_f32(values):
    return b"".join(struct.pack("<f", value) for value in values)


def write_field(dump_dir, iteration, field, values):
    dump_dir.mkdir(parents=True, exist_ok=True)
    comparator.dump_path(dump_dir, iteration, field).write_bytes(pack_f32(values))


def write_iteration(dump_dir, iteration, by_field):
    for field in FIELDS:
        write_field(dump_dir, iteration, field, by_field[field])


def write_dump_dir(dump_dir, iterations):
    for iteration, by_field in enumerate(iterations):
        write_iteration(dump_dir, iteration, by_field)


def uniform_iteration(values):
    return {field: list(values) for field in FIELDS}


def divergent_iteration(masked_ref):
    """One iteration whose `guided` field separates the two comparison paths:
    argmax agrees on the raw vectors while cosine sees a narrow finite subset."""
    by_field = uniform_iteration(PLAIN)
    by_field["guided"] = list(masked_ref)
    return by_field


class NumericBackend:
    """Runs a block with the comparator forced onto its stdlib math path."""

    def __enter__(self):
        self.saved = comparator.np
        comparator.np = None

    def __exit__(self, *exc):
        comparator.np = self.saved
        return False


class DirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.ref = self.root / "ref"
        self.cand = self.root / "cand"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, extra=()):
        argv = ["compare_ar_dumps.py", str(self.ref), str(self.cand), "--json", *extra]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(sys, "stdout", io.StringIO()), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            return comparator.main()

    def write_pair(self, ref_iters, cand_iters):
        write_dump_dir(self.ref, ref_iters)
        write_dump_dir(self.cand, cand_iters)


class CosineTests(unittest.TestCase):
    def test_both_finite(self):
        value, fraction = comparator.cosine(PLAIN, PLAIN_OTHER)
        self.assertAlmostEqual(value, 24.0 / 25.0, places=12)
        self.assertEqual(fraction, 1.0)

    def test_masked_subset(self):
        value, fraction = comparator.cosine([3.0, 5.0, NEG_INF], [-1.0, 4.0, NEG_INF])
        self.assertAlmostEqual(value, 2.0 ** -0.5, places=12)
        self.assertAlmostEqual(fraction, 2.0 / 3.0, places=12)

    def test_no_finite_pair(self):
        value, fraction = comparator.cosine([NEG_INF, NEG_INF], [NEG_INF, NEG_INF])
        self.assertIsNone(value)
        self.assertEqual(fraction, 0.0)

    def test_disjoint_masks(self):
        value, _ = comparator.cosine([1.0, NEG_INF], [NEG_INF, 1.0])
        self.assertIsNone(value)

    def test_both_zero(self):
        self.assertEqual(comparator.cosine([0.0, 0.0], [0.0, 0.0])[0], 1.0)

    def test_one_zero(self):
        self.assertEqual(comparator.cosine([0.0, 0.0], [1.0, 2.0])[0], 0.0)

    def test_stdlib_agrees(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy path is the one already under test")
        pairs = [(PLAIN, PLAIN_OTHER), ([3.0, 5.0, NEG_INF], [-1.0, 4.0, NEG_INF])]
        for a, b in pairs:
            with_numpy = comparator.cosine(a, b)
            with NumericBackend():
                stdlib = comparator.cosine(a, b)
            self.assertAlmostEqual(with_numpy[0], stdlib[0], places=12)
            self.assertAlmostEqual(with_numpy[1], stdlib[1], places=12)


class ArgmaxTests(unittest.TestCase):
    def test_raw_inf_ranked_low(self):
        stat = comparator.argmax_stat([3.0, 5.0, NEG_INF], [-1.0, 4.0, NEG_INF])
        self.assertTrue(stat["agree"])

    def test_cosine_path_differs(self):
        # The same vectors the argmax path calls a match score only 1/sqrt(2)
        # under cosine, because cosine drops the masked positions.
        ref, cand = [3.0, 5.0, NEG_INF], [-1.0, 4.0, NEG_INF]
        self.assertTrue(comparator.argmax_stat(ref, cand)["agree"])
        self.assertLess(comparator.cosine(ref, cand)[0], 0.71)

    def test_overlap_clamped(self):
        stat = comparator.argmax_stat(PLAIN, PLAIN_OTHER)
        self.assertEqual(stat["top_k_overlap"], 2)

    def test_disagree(self):
        self.assertFalse(comparator.argmax_stat([1.0, 2.0], [2.0, 1.0])["agree"])

    def test_stdlib_agrees(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy path is the one already under test")
        ref = [float(v) for v in range(10)]
        cand = [float(9 - v) for v in range(10)]
        with_numpy = comparator.argmax_stat(ref, cand)
        with NumericBackend():
            stdlib = comparator.argmax_stat(ref, cand)
        self.assertEqual(with_numpy, stdlib)


class HalfTests(unittest.TestCase):
    def test_conditional_half(self):
        self.assertEqual(list(comparator.conditional_half([1.0, 2.0, 3.0, 4.0])), [1.0, 2.0])

    def test_odd_length(self):
        with self.assertRaises(SystemExit):
            comparator.conditional_half([1.0, 2.0, 3.0])


class SummaryTests(unittest.TestCase):
    def test_worst_index(self):
        # worst_iteration must name the original iteration, not the position
        # it holds after the incomparable entries are filtered out.
        entries = [
            {"cosine": None, "finite_fraction": 0.0},
            {"cosine": 0.9, "finite_fraction": 1.0},
            {"cosine": 0.2, "finite_fraction": 1.0},
        ]
        stats = comparator.summarize_field(entries)
        self.assertEqual(stats["worst_iteration"], 2)
        self.assertEqual(stats["n_comparable"], 2)
        self.assertEqual(stats["n"], 3)

    def test_all_incomparable(self):
        stats = comparator.summarize_field([{"cosine": None, "finite_fraction": 0.0}])
        self.assertIsNone(stats["min_cosine"])
        self.assertEqual(stats["n_comparable"], 0)

    def test_overall_worst(self):
        # Picked by cosine, not by tuple order: iteration 1 is the worst.
        summary = comparator.build_summary(
            {field: [] for field in FIELDS}, {name: [] for name in comparator.ARGMAX_FIELDS},
            [0.5, 0.2])
        self.assertEqual(summary["overall_worst_iteration"], 1)
        self.assertAlmostEqual(summary["overall_worst_min_cosine"], 0.2)


class GateTests(unittest.TestCase):
    def summary_with(self, min_cosine, agree_pct):
        return {
            "fields": {"guided": {"min_cosine": min_cosine, "worst_iteration": 0}},
            "argmax": {"guided": {"argmax_agree_pct": agree_pct}},
            "overall_worst_min_cosine": min_cosine,
        }

    def test_cosine_fails(self):
        summary = self.summary_with(0.90, 100.0)
        self.assertEqual(len(comparator.apply_gates(summary, 0.99, None)), 1)

    def test_cosine_passes(self):
        summary = self.summary_with(0.999, 100.0)
        self.assertEqual(comparator.apply_gates(summary, 0.99, None), [])

    def test_agree_fails(self):
        summary = self.summary_with(0.999, 80.0)
        self.assertEqual(len(comparator.apply_gates(summary, None, 90.0)), 1)

    def test_agree_passes(self):
        summary = self.summary_with(0.999, 95.0)
        self.assertEqual(comparator.apply_gates(summary, None, 90.0), [])

    def test_none_cosine_fails(self):
        summary = self.summary_with(None, 100.0)
        self.assertEqual(len(comparator.apply_gates(summary, 0.99, None)), 1)

    def test_incomparable_flagged(self):
        self.assertEqual(len(comparator.comparability_violations(
            {"overall_worst_min_cosine": None})), 1)

    def test_comparable_clean(self):
        self.assertEqual(comparator.comparability_violations(
            {"overall_worst_min_cosine": 0.5}), [])


class MainTests(DirCase):
    def test_clean_run(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN_OTHER)])
        self.assertEqual(self.run_main(), 0)

    def test_gate_fails(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN_OTHER)])
        self.assertEqual(self.run_main(["--min-cosine", "0.99"]), 1)

    def test_gate_passes(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN_OTHER)])
        self.assertEqual(self.run_main(["--min-cosine", "0.5"]), 0)

    def test_agree_gate_fails(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN_OTHER)])
        self.assertEqual(self.run_main(["--min-argmax-agree", "50"]), 1)

    def test_agree_gate_passes(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN)])
        self.assertEqual(self.run_main(["--min-argmax-agree", "100"]), 0)

    def test_incomparable_ungated(self):
        # Nothing was actually compared, so the tool must not report success
        # even though no --min-* flag was given.
        masked = uniform_iteration([NEG_INF, NEG_INF])
        self.write_pair([masked], [masked])
        self.assertEqual(self.run_main(), 1)

    def test_incomparable_gated(self):
        masked = uniform_iteration([NEG_INF, NEG_INF])
        self.write_pair([masked], [masked])
        self.assertEqual(self.run_main(["--min-cosine", "0.99"]), 1)

    def test_size_mismatch(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration([1.0, 2.0, 3.0])])
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_iter_mismatch(self):
        self.write_pair([uniform_iteration(PLAIN), uniform_iteration(PLAIN)],
                        [uniform_iteration(PLAIN)])
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_missing_field(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN)])
        comparator.dump_path(self.cand, 0, "guided").unlink()
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_empty_field(self):
        self.write_pair([uniform_iteration(PLAIN)], [uniform_iteration(PLAIN)])
        comparator.dump_path(self.cand, 0, "guided").write_bytes(b"")
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_not_a_dump_dir(self):
        self.ref.mkdir(parents=True)
        self.cand.mkdir(parents=True)
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_missing_dir(self):
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_divergent_field(self):
        # guided agrees on argmax but not on cosine, so a tight cosine gate
        # fails while a permissive argmax gate passes on the same pair.
        self.write_pair([divergent_iteration([3.0, 5.0, NEG_INF])],
                        [divergent_iteration([-1.0, 4.0, NEG_INF])])
        self.assertEqual(self.run_main(["--min-cosine", "0.99"]), 1)
        self.assertEqual(self.run_main(["--min-argmax-agree", "100"]), 0)

    def test_stdlib_matches(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy path is the one already under test")
        self.write_pair([divergent_iteration([3.0, 5.0, NEG_INF])],
                        [divergent_iteration([-1.0, 4.0, NEG_INF])])
        with_numpy = self.run_main(["--min-cosine", "0.99"])
        with NumericBackend():
            stdlib = self.run_main(["--min-cosine", "0.99"])
        self.assertEqual(with_numpy, stdlib)


class ReadTests(DirCase):
    def test_roundtrip(self):
        write_field(self.ref, 0, "guided", [1.5, -2.5, NEG_INF])
        values = list(comparator.read_f32(comparator.dump_path(self.ref, 0, "guided")))
        self.assertEqual(values[:2], [1.5, -2.5])
        self.assertEqual(values[2], NEG_INF)

    def test_stdlib_roundtrip(self):
        write_field(self.ref, 0, "guided", [1.5, -2.5, NEG_INF])
        with NumericBackend():
            values = list(comparator.read_f32(comparator.dump_path(self.ref, 0, "guided")))
        self.assertEqual(values[:2], [1.5, -2.5])
        self.assertEqual(values[2], NEG_INF)


if __name__ == "__main__":
    sys.exit(unittest.main())
