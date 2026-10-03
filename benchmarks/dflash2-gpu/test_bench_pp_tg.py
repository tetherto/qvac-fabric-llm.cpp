"""Prevent incomplete or cached measurements from entering campaign results."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bench_pp_tg as bench
import collect_quality_tokens as quality

PROMPT_TOKENS = 10000
COMPLETION_TOKENS = 1024


def response(cache=None, include_usage=True, last=4.0, error=None):
    extras = {"timings": {"prompt_per_second": 250.0, "predicted_per_second": 50.0}}
    if cache is not None:
        extras["timings"]["cache_n"] = cache
    if include_usage:
        extras["usage"] = {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS}
    if error:
        extras["error"] = error
    return {"start": 1.0, "first": 2.0, "last": last, "extras": extras,
            "text": "answer", "chunks": []}


class AdmissionTests(unittest.TestCase):
    def measure(self, value, engine="fabric", evidence=None):
        with patch.object(bench, "stream", return_value=value):
            return bench.measure("http://unused", engine, "prompt", PROMPT_TOKENS, COMPLETION_TOKENS, evidence)

    def test_missing_cache_counter_is_not_uncached(self):
        result = self.measure(response())
        self.assertFalse(result["ok"])
        self.assertEqual(result["cache_status"], "unreported")
        self.assertFalse(self.measure(response(), evidence={"uncached": True})["ok"])

    def test_positive_cache_counter_overrides_external_evidence(self):
        result = self.measure(response(cache=128), engine="tf", evidence={"uncached": True})
        self.assertFalse(result["checks"]["cached_zero"])
        self.assertFalse(result["ok"])
        self.assertTrue(self.measure(response(cache=0))["ok"])

    def test_missing_counts_are_preserved_as_invalid_not_fabricated_rates(self):
        result = self.measure(response(cache=0, include_usage=False))
        self.assertFalse(result["ok"])
        self.assertIsNone(result["prefill_tps"])
        self.assertIsNone(result["decode_tps"])

    def test_single_chunk_cannot_supply_decode_rate(self):
        result = self.measure(response(cache=0, last=2.0))
        self.assertFalse(result["ok"])
        self.assertIsNone(result["decode_tps"])

    def test_server_error_rejects_even_complete_token_counts(self):
        result = self.measure(response(cache=0, error={"message": "device failure"}))
        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"]["server_error_free"])


class QualityFixtureTests(unittest.TestCase):
    def test_invalid_token_ids_are_not_saved_as_reference_inputs(self):
        for tokens in ([True, 1], [-1, 1], [1.0, 2], [1]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                quality.validate_tokens(tokens, 2)

    def test_cached_prefix_does_not_create_quality_artifact(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(base="http://unused", output_dir=directory)
            response = {"tokens": [1] * quality.COMPLETION_TOKENS, "truncated": False,
                        "timings": {"cache_n": 1, "prompt_n": 2}}
            with patch.object(quality, "post", side_effect=[{"tokens": [1, 2]}, response]):
                with self.assertRaises(ValueError):
                    quality.collect(args, {"prompts": ["prompt"], "prompt_tokens": 2}, 0, {})
            self.assertFalse((Path(directory) / "tokens-0.json").exists())


if __name__ == "__main__":
    unittest.main()
