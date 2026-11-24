#!/usr/bin/env python3
"""Hermetic tests for bench-load-time.py and fresh-host.sh (no Core ML, Metal or ANE).

  python3 -B scripts/coreml/test_bench_load_time.py
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_load_time", HERE / "bench-load-time.py")
blt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blt)

MANIFEST = blt.load_manifest(blt.DEFAULT_MANIFEST)

# Plan sections, work-unit ids and ticket ids: none belong in code or docs. The
# pattern is assembled from pieces so this file does not match it itself.
TASK_ID = re.compile("|".join(["QVAC" + "-[0-9]", "P" + "LAN", "\u00a7",
                               r"\b(S[0-9][a-c]?|P[0-9]|A[1]|Q[0-9]|H[0-9]|R[0-9])\b"]))


def errors_of(mutate):
    m = copy.deepcopy(MANIFEST)
    mutate(m)
    return blt.validate(m)


class ManifestTest(unittest.TestCase):
    def test_pinned_manifest_is_valid(self):
        self.assertEqual(blt.validate(MANIFEST), [])

    def test_dry_run_lists_every_measurement(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(blt.main(["--dry-run"]), 0)
        text = out.getvalue()
        self.assertIn("manifest ok", text)
        for name in blt.MEASUREMENTS:
            self.assertIn(f"  {name}", text)

    def test_dry_run_through_run_does_not_need_models(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = blt.main(["--dry-run", "--models-root", "/nonexistent", "run", "ttr", "--variant", "mtl"])
        self.assertEqual(rc, 0)
        self.assertIn("mtl/generate-ungated", out.getvalue())
        self.assertNotIn("turbo/", out.getvalue())

    def test_dry_run_fails_on_a_bad_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            bad = copy.deepcopy(MANIFEST)
            bad["protocol"]["runs"]["metal"] = 3
            p = Path(d) / "m.json"
            p.write_text(json.dumps(bad))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(blt.main(["--manifest", str(p), "--dry-run"]), 1)
            self.assertIn("protocol.runs.metal", err.getvalue())

    def test_unknown_measurement_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(blt.main(["--dry-run", "run", "nope"]), 64)

    def test_mutations_are_caught(self):
        def flip_sha(m):
            e = m["quality_sets"]["wer"][0]["audio"]
            e["sha256"] = ("0" if e["sha256"][0] != "0" else "1") + e["sha256"][1:]

        cases = [
            ("schema", lambda m: m.update(schema=2), "schema must be 1"),
            ("published run count", lambda m: m["protocol"]["runs"].update(cold_prebuilt=5), "protocol.runs.cold_prebuilt"),
            ("repo sha", flip_sha, "sha256"),
            ("repo bytes", lambda m: m["quality_sets"]["der"][0]["rttm"].update(bytes=1), "pinned 1"),
            ("traversal", lambda m: m["workloads"]["benchmark-prompt"]["text"].update(path="../x"), "without '..'"),
            ("bad hex", lambda m: m["variants"]["turbo"]["t3"].update(sha256="XYZ"), "64 lowercase hex"),
            ("no WER", lambda m: m["quality_sets"].update(wer=[]), "quality_sets.wer"),
            ("no DER", lambda m: m["quality_sets"].pop("der"), "quality_sets.der"),
            ("w3 size", lambda m: m["sidecar_sets"]["w3"].update(frames=[512, 768, 1024, 1536]), "3 widths"),
            ("w3 subset", lambda m: m["sidecar_sets"]["w3"].update(frames=[512, 1000, 1536]), "subset"),
            ("odd frames", lambda m: m["sidecar_sets"]["w3"].update(frames=[511, 1024, 1536]), "multiples"),
            ("exporter", lambda m: m["sidecar_sets"]["w7"].update(frames=[512, 640, 768, 896, 1024, 1280, 2048]),
             "DEFAULT_FRAMES"),
            ("hift_from", lambda m: m["sidecar_sets"]["w3"].pop("hift_from"), "hift_from"),
            ("stem", lambda m: m["variants"]["mtl"].update(sidecar_stem="chatterbox-s3gen"), "sidecar_stem"),
            ("variants", lambda m: m["variants"].pop("mtl"), "turbo and mtl"),
            ("hf pin", lambda m: m["variants"]["turbo"]["hf"].update(revision="main"), "40-hex"),
            ("sentence-split chars", lambda m: m["workloads"]["benchmark-prompt"].update(max_sentence_chars=200), "180"),
            ("sweep", lambda m: m["workloads"]["length-sweep"]["sentences"].append("One. Two."), "one sentence"),
            ("measurements", lambda m: m["measurements"].pop("sizes"), "measurements must be exactly"),
            ("ttr ungated", lambda m: m["measurements"]["ttr"].update(
                configs=[c for c in m["measurements"]["ttr"]["configs"] if c["mode"] != "ungated"]), "ungated"),
            ("ttr N", lambda m: m["measurements"]["ttr"]["configs"][0].update(runs=2), "time_to_ready"),
            ("synthesis", lambda m: m["measurements"]["synthesis"].update(
                configs=[c for c in m["measurements"]["synthesis"]["configs"] if c["name"] != "coreml-no-encoder"]),
             "coreml-no-encoder"),
            ("unknown set", lambda m: m["measurements"]["load"].update(sets=["w5"]), "unknown sidecar set"),
            ("units", lambda m: m["protocol"].update(units="ane"), "protocol.units"),
            ("sharing units", lambda m: m["measurements"]["load"].update(sharing_units=["gpu"]), "sharing_units"),
        ]
        for name, mutate, needle in cases:
            with self.subTest(name):
                errs = errors_of(mutate)
                self.assertTrue(any(needle in e for e in errs), f"{name}: {errs}")

    def test_model_files_are_checked_on_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = copy.deepcopy(MANIFEST)
            for v in m["variants"].values():
                for k in ("t3", "s3gen", "export_source"):
                    p = root / v[k]["path"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    data = f"{k}-{v['sidecar_stem']}".encode()
                    p.write_bytes(data)
                    v[k].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            self.assertEqual(blt.validate(m, models=root, check_models=True), [])
            # Not checked unless asked.
            self.assertEqual(blt.validate(m, models=root / "missing"), [])
            (root / m["variants"]["mtl"]["t3"]["path"]).write_bytes(b"x" * m["variants"]["mtl"]["t3"]["bytes"])
            errs = blt.validate(m, models=root, check_models=True)
            self.assertTrue(any("variants.mtl.t3" in e and "sha256" in e for e in errs), errs)


class NamingTest(unittest.TestCase):
    def test_function_names_follow_the_exporter(self):
        script = (blt.REPO / MANIFEST["exporter"]["script"]).read_text()
        self.assertIn('return f"tokens_{width}" if stage == "encoder" else f"frames_{width}"', script)
        self.assertEqual(blt.function_names("encoder", [512, 1024, 1536], 2), ["tokens_256", "tokens_512", "tokens_768"])
        self.assertEqual(blt.function_names("cfm", [512, 1024], 2), ["frames_512", "frames_1024"])
        self.assertEqual(blt.function_names("hift", [512, 1024], 2), ["main"])

    def test_sidecar_stem_matches_the_engine_rule(self):
        self.assertEqual(blt._model_stem("chatterbox-s3gen-q8_0.gguf"), "chatterbox-s3gen")
        self.assertEqual(blt._model_stem("chatterbox-s3gen-mtl-f32.gguf"), "chatterbox-s3gen-mtl")
        self.assertEqual(blt._model_stem("chatterbox-s3gen-mtl.gguf"), "chatterbox-s3gen-mtl")

    def test_w3_hift_comes_from_w7(self):
        root = Path("/models")
        self.assertEqual(blt.sidecar(MANIFEST, root, "mtl", "w3", "hift"),
                         root / "chatterbox" / "chatterbox-s3gen-mtl-hift.mlmodelc")
        self.assertEqual(blt.sidecar(MANIFEST, root, "turbo", "w3", "cfm", "mlpackage"),
                         root / "chatterbox" / "w3" / "chatterbox-s3gen-cfm.mlpackage")
        fns = dict(blt.set_functions(MANIFEST, "turbo", "w3"))
        self.assertEqual(fns["hift"], ["main"])
        self.assertEqual(len(fns["cfm"]), 3)


SAMPLE_STDERR = """\
BENCH: T3_LOAD_MS=812
BENCH: S3GEN_LOAD_MS=455
BENCH: T3_INFER_MS=1490 tokens=210
  [encoder] 52.1 ms (coreml-ane)
  [cfm_step0] 47.0 ms (coreml-ane)
  [cfm_step1] 45.9 ms (coreml-ane)
  [hift_decode] 120.3 ms (coreml-ane)
BENCH: S3GEN_INFER_MS=402 AUDIO_MS=5120
CHUNK_PROFILE n_total=402 T_mu=804 T_mel=804 enc_ms=52.1 cfm_ms=93.0 hift_ms=150.2 pipeline_ms=402.0 audio_ms=5120
BENCH: T3_INFER_MS=1210 tokens=160
  [encoder] 70.0 ms (ggml)
  [cfm_step0] 44.0 ms (coreml-ane)
  [hift_decode] 110.0 ms (coreml-ane)
BENCH: S3GEN_INFER_MS=380 AUDIO_MS=4000
CHUNK_PROFILE n_total=350 T_mu=700 T_mel=700 enc_ms=70.0 cfm_ms=88.0 hift_ms=140.0 pipeline_ms=380.0 audio_ms=4000
        3.21 real         2.10 user         0.80 sys
          1234567890  maximum resident set size
          987654321  peak memory footprint
"""


class ParsingTest(unittest.TestCase):
    def test_tts_stderr(self):
        r = blt.parse_tts_stderr(SAMPLE_STDERR)
        self.assertEqual(len(r["chunks"]), 2)
        self.assertEqual(r["chunks"][0]["T_mel"], 804)
        self.assertAlmostEqual(r["chunks"][1]["pipeline_ms"], 380.0)
        self.assertEqual(r["t3_infer_ms"], [1490, 1210])
        self.assertEqual(r["s3gen_infer_ms"], [402, 380])
        self.assertEqual(r["labels"]["encoder"], ["coreml-ane", "ggml"])
        self.assertEqual(r["labels"]["cfm"], ["coreml-ane"])
        self.assertEqual(r["max_rss"], 1234567890)
        self.assertEqual(r["peak_footprint"], 987654321)

    def test_labels(self):
        self.assertEqual(blt.check_labels({"encoder": ["coreml-ane"], "cfm": ["coreml-ane"], "hift": ["coreml-ane"]},
                                          ("encoder", "cfm", "hift")), [])
        self.assertEqual(blt.check_labels({"encoder": ["ggml"], "cfm": ["coreml-ane"], "hift": ["coreml-ane"]},
                                          ("cfm", "hift")), [])
        bad = blt.check_labels({"encoder": ["coreml-ane", "ggml"], "cfm": ["ggml"], "hift": []},
                               ("encoder", "cfm", "hift"))
        self.assertEqual(len(bad), 3)
        self.assertEqual(blt.check_labels({"encoder": ["ggml"], "cfm": ["ggml"], "hift": ["coreml-ane"]}, ()),
                         ["hift: expected ggml, got ['coreml-ane']"])


class AnalysisTest(unittest.TestCase):
    def test_linfit(self):
        a, b, r2 = blt.linfit([100, 200, 300], [15, 25, 35])
        self.assertAlmostEqual(a, 5.0)
        self.assertAlmostEqual(b, 0.1)
        self.assertAlmostEqual(r2, 1.0)
        with self.assertRaises(ValueError):
            blt.linfit([1], [1])
        with self.assertRaises(ValueError):
            blt.linfit([2, 2], [1, 3])

    def test_median_mad(self):
        self.assertEqual(blt.median([3, 1, 2]), 2)
        self.assertEqual(blt.mad([1, 2, 3, 4, 100]), 1)
        self.assertIsNone(blt.median([]))

    def test_waits(self):
        ttr = {"detail": {"t_issue_ms": 100.0}, "timeline": [
            {"kind": "prepare", "label": "encoder", "start_ms": 100.0, "end_ms": 1100.0},
            {"kind": "load", "label": "encoder/tokens_256", "start_ms": 1100.0, "end_ms": 9000.0},
            {"kind": "prepare", "label": "cfm", "start_ms": 9000.0, "end_ms": 10000.0},
            {"kind": "load", "label": "encoder/tokens_512", "start_ms": 10000.0, "end_ms": 12000.0},
            {"kind": "warm", "label": "encoder/tokens_256", "start_ms": 9000.0, "end_ms": 9001.0},
        ]}
        self.assertEqual(blt.prepare_waits(ttr), [0.0, 8900.0])
        self.assertEqual(blt.load_gate_waits(ttr), [0.0, 8900.0])
        pre = {"detail": {"t_issue_ms": 5.0}, "timeline": [
            {"kind": "load", "label": "cfm/frames_512", "start_ms": 5.0, "end_ms": 50.0}]}
        self.assertEqual(blt.load_gate_waits(pre), [0.0])

    def test_sharing_verdict(self):
        w = 400_000_000
        self.assertEqual(blt.sharing_verdict([w, 10_000_000, 12_000_000], w), "shared")
        self.assertEqual(blt.sharing_verdict([w, w, w], w), "not shared")
        self.assertEqual(blt.sharing_verdict([w], w), "n/a")

    def test_report_renders_every_section(self):
        recs = [
            {"kind": "env", "git": "abc123", "chip": "Apple M2", "memsize": 24 * 2**30, "os_version": "15.7.3",
             "os_build": "24G419", "ac_power": True, "units": "cpu_and_ane", "manifest_sha256": "f" * 64},
            {"kind": "sizes", "variant": "turbo", "set": "w7", "stage": "cfm", "functions": 7, "mil_bytes": 2_000_000,
             "mil_zlib": 300_000, "mil_lzfse": 280_000, "weight_bytes": 200_000_000, "mlmodelc_bytes": 202_000_000,
             "mlmodel_bytes": 1_500_000, "mlmodel_zlib": 200_000, "mlmodel_lzfse": 190_000},
            {"kind": "memory", "samples": [8e9, 9e9], "min": 8e9, "median": 8.5e9, "max": 9e9, "low_power": False,
             "reserves": {"turbo/cfm": 320_000_000, "mtl/cfm": 9_000_000_000}, "memsize": 1},
            {"kind": "load_cold", "variant": "turbo", "set": "w7", "stage": "cfm", "function": "frames_512",
             "load_ms": 12000.0},
            {"kind": "load_warm", "variant": "turbo", "set": "w7", "stage": "cfm", "function": "frames_512",
             "load_ms": 300.0, "predict_ms": [40.0, 30.0, 31.0]},
            {"kind": "load_warm", "variant": "turbo", "set": "w7", "stage": "cfm", "function": "frames_512",
             "load_ms": 320.0, "predict_ms": [42.0, 32.0, 33.0]},
            {"kind": "sharing", "variant": "turbo", "set": "w7", "stage": "cfm", "units": "cpu_and_ane",
             "weight_bytes": 200_000_000, "metrics": {
                 "footprint": {"increments": [10e6, 5e6, 6e6], "alone": 9e6, "verdict": "shared"},
                 "wired": {"increments": [210e6, 190e6, 200e6], "alone": 205e6, "verdict": "not shared"}}},
            {"kind": "metal_cost", "variant": "turbo", "T_mu": 600, "T_mel": 600, "enc_ms": 60.0, "cfm_ms": 200.0,
             "hift_ms": 400.0, "pipeline_ms": 700.0},
            {"kind": "metal_cost", "variant": "turbo", "T_mu": 1200, "T_mel": 1200, "enc_ms": 80.0, "cfm_ms": 300.0,
             "hift_ms": 700.0, "pipeline_ms": 1100.0},
            {"kind": "ttr", "variant": "turbo", "config": "generate-gated", "all_ready": True,
             "time_to_ready_ms": 200000.0, "prepare_waits_ms": [0.0, 30000.0], "load_gate_waits_ms": [0.0, 5000.0],
             "deferred_memory_ms": 0.0},
            {"kind": "ttr", "variant": "turbo", "config": "generate-gated", "all_ready": False,
             "time_to_ready_ms": -1, "prepare_waits_ms": [], "load_gate_waits_ms": [], "deferred_memory_ms": 2000.0},
            {"kind": "ttr_cached", "variant": "turbo", "config": "generate-gated", "all_ready": True,
             "time_to_ready_ms": 9000.0},
            {"kind": "synthesis", "variant": "turbo", "config": "coreml-w7", "wall_ms": 3680.0, "t3_infer_ms": 2660,
             "s3gen_infer_ms": 720, "peak_footprint": 2_000_000_000},
        ]
        s = blt.summarize(recs)
        self.assertAlmostEqual(s["metal_fit"]["turbo"]["pipeline_ms"]["b"], 400 / 600)
        self.assertEqual(s["cold"][("turbo", "w7")]["total_ms"], 12000.0)
        self.assertEqual(blt.median(s["warm"][("turbo", "w7", "cfm", "frames_512")]["warm"]), 351.0)
        doc = blt.render(s)
        for needle in ("Apple M2", "| turbo | w7 | cfm | 7 | 2.00 |", "| turbo | w7 | 1 | 12.0 | 351 |",
                       "| turbo | w7 | cfm | frames_512 | 12,000 | 310 | 351 | 362 | 31.5 |",
                       "| turbo | w7 | cfm | cpu_and_ane | 200.00 | footprint | 10.00 | 5.50 | 9.00 | shared |",
                       "| turbo | w7 | cfm | cpu_and_ane | 200.00 | wired | 210.00 | 195.00 | 205.00 | not shared |", "| turbo | pipeline | 300.0 | 0.6667 | 1.0000 | 2 |",
                       "| turbo | generate-gated | 2 | 1/2 | 200.0 |", "| 30.0 | 5.0 | 2.0 | 9.00 |",
                       "| mtl/cfm | 9,000.00 | yes |", "| turbo/cfm | 320.00 | no |",
                       "| turbo | coreml-w7 | 1 | 3,680 | 2,660 | 720 |", "Where each number goes"):
            self.assertIn(needle, doc)
        self.assertIsNone(TASK_ID.search(doc), TASK_ID.search(doc))

    def test_two_cold_runs_into_one_results_file(self):
        # Two `run load` passes with the same --out: two identities, each with
        # its own cold load of the same functions.
        def cold(identity, fn, ms):
            return {"kind": "load_cold", "variant": "turbo", "set": "w3", "stage": "cfm", "function": fn,
                    "load_ms": ms, "identity": identity}

        sizes = {"kind": "sizes", "variant": "turbo", "set": "w3", "stage": "cfm", "functions": 3,
                 "mil_bytes": 3_940_000, "mil_zlib": 330_000, "weight_bytes": 139_900_000, "mlmodelc_bytes": 1}
        recs = [cold("id-a", "frames_512", 10000.0), cold("id-a", "frames_1024", 20000.0),
                cold("id-b", "frames_512", 12000.0), cold("id-b", "frames_1024", 22000.0),
                {"kind": "load_warm", "variant": "turbo", "set": "w3", "stage": "cfm", "function": "frames_512",
                 "load_ms": 470.0, "predict_ms": [40.0, 23.0]}, sizes, dict(sizes)]
        s = blt.summarize(recs)
        c = s["cold"][("turbo", "w3")]
        self.assertEqual(c["total_ms"], 32000.0)  # median of the per-identity totals 30 s and 34 s
        self.assertEqual(c["runs"], 2)
        self.assertEqual(len(c["functions"]), 2)
        doc = blt.render(s)
        self.assertIn("| turbo | w3 | 2 | 32.0 | 510 |", doc)
        self.assertIn("| turbo | w3 | cfm | frames_512 | 11,000 | 470 | 510 |", doc)
        self.assertEqual(doc.count("| turbo | w3 | cfm | 3 |"), 1)

    def test_report_without_results(self):
        doc = blt.render(blt.summarize([]))
        self.assertIn("Not measured yet", doc)
        self.assertIn("## Time to ready", doc)

    def test_readme_links_the_setup_and_usage_section(self):
        doc = blt.render(blt.summarize([]))
        self.assertIn("\n## Setup and usage\n", doc)
        self.assertIn("docs/COREML-LOAD-TIME.md#setup-and-usage", (blt.REPO / "README.md").read_text())
        for cmd in ("--dry-run --check-models", "build-driver", "smoke", "run --tts-cli", "report --results",
                    "cleanup"):
            self.assertIn(cmd, doc)

    def test_sources_cite_no_task_identifiers(self):
        files = sorted(p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".mm", ".h", ".cpp",
                                                                                 ".json"))
        files.append(blt.REPO / "docs" / "COREML-LOAD-TIME.md")
        hits = [f"{p.name}:{i}: {ln.strip()}" for p in files
                for i, ln in enumerate(p.read_text().splitlines(), 1) if TASK_ID.search(ln)]
        self.assertGreaterEqual(len(files), 8)
        self.assertEqual(hits, [])
        for sample in ("QVAC" + "-12345", "the " + "P" + "LAN", "\u00a7" + "2.9", "unit " + "S" + "1c", "A" + "1"):
            self.assertIsNotNone(TASK_ID.search(sample), sample)

    def test_report_cites_no_task_identifiers(self):
        doc = blt.render(blt.summarize([]))
        self.assertTrue(doc.startswith("# Core ML load time measurements\n"))
        self.assertIsNone(TASK_ID.search(doc), TASK_ID.search(doc))


class ReportMergeTest(unittest.TestCase):
    @staticmethod
    def results(d: Path, cold_ms: float) -> Path:
        d.mkdir()
        (d / "results.jsonl").write_text(json.dumps(
            {"kind": "load_cold", "variant": "turbo", "set": "w7", "stage": "cfm", "function": "frames_512",
             "load_ms": cold_ms, "identity": "id-a"}) + "\n")
        return d

    @staticmethod
    def report(*args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = blt.main(["report", *map(str, args)])
        return rc, err.getvalue()

    def test_report_keeps_the_hand_written_text(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = self.results(Path(d) / "a", 12000.0), self.results(Path(d) / "b", 30000.0)
            doc = Path(d) / "doc.md"
            self.assertEqual(self.report("--results", a, "--out", doc)[0], 0)
            first = doc.read_text()
            self.assertIn("| turbo | w7 | 1 | 12.0 |", first)
            # Hand edits outside the generated blocks: a reading of the numbers
            # and a corrected host line.
            i = first.index("<!-- report:")
            edited = (first[:i] + "## Reading the numbers\n\nWritten by hand.\n\n" + first[i:]).replace(
                "- Not measured yet", "- Power: on battery for part of the run")
            doc.write_text(edited)
            self.assertEqual(self.report("--results", b, "--out", doc)[0], 0)
            second = doc.read_text()
            self.assertIn("## Reading the numbers\n\nWritten by hand.\n", second)
            self.assertIn("- Power: on battery for part of the run", second)
            self.assertIn("| turbo | w7 | 1 | 30.0 |", second)
            self.assertNotIn("| turbo | w7 | 1 | 12.0 |", second)
            # Same results again: nothing changes.
            self.assertEqual(self.report("--results", b, "--out", doc)[0], 0)
            self.assertEqual(doc.read_text(), second)

    def test_report_refuses_a_file_without_generated_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.results(Path(d) / "a", 12000.0)
            doc = Path(d) / "notes.md"
            doc.write_text("# Notes\n\nWritten by hand.\n")
            rc, err = self.report("--results", a, "--out", doc)
            self.assertEqual(rc, 1)
            self.assertIn("generated", err)
            self.assertEqual(doc.read_text(), "# Notes\n\nWritten by hand.\n")

    def test_committed_doc_keeps_its_text_on_regeneration(self):
        committed = (blt.REPO / "docs" / "COREML-LOAD-TIME.md").read_text()
        merged = blt.merge_report(committed, blt.render(blt.summarize([])) + "\n")
        for kept in ("## Reading the numbers", "Limits of this run", "on battery from 12:35 to 14:41"):
            self.assertIn(kept, merged)
        self.assertIn("Not measured yet.", merged)


class DriverSourceTest(unittest.TestCase):
    """The ttr readings the driver emits go through bench_ttr::with_readings
    (sampling order tested in test_bench_ttr.cpp), not straight into the output."""

    def test_ttr_readings_are_taken_around_the_run(self):
        src = blt.DRIVER_SRC.read_text()
        body = src[src.index("int cmd_ttr("):]
        body = body[:body.index("\n}\n")]
        call = body.find("bench_ttr::with_readings(")
        self.assertGreaterEqual(call, 0, "cmd_ttr takes its readings through bench_ttr::with_readings")
        depth, end = 0, None
        for i in range(body.index("(", call), len(body)):
            depth += {"(": 1, ")": -1}.get(body[i], 0)
            if depth == 0:
                end = i
                break
        outside = body[:call] + body[end + 1:]
        for reading in ("phys_footprint()", "lifetime_max_footprint()", "device_available_memory()",
                        "g_models_live"):
            self.assertIn(reading, body[call:end])
            self.assertNotIn(reading, outside)
        for helper in ("json ttr_manager(", "json ttr_manual("):
            fn = src[src.index(helper):]
            self.assertNotIn("models_live", fn[:fn.index("\n}\n")], helper)


class TtrSummaryTest(unittest.TestCase):
    @staticmethod
    def run_record(ttr_ms, live, n=3):
        slots = [{"slot": f"cfm/f{i}", "ready_ms": 10.0 * i} for i in range(n)]
        return {"time_to_ready_ms": ttr_ms, "models_live": live, "available_bytes_at_start": 11_000_000_000,
                "detail": {"t_issue_ms": 0.0, "slots": slots}}

    def test_ready_needs_every_model_resident_at_the_end(self):
        s = blt.Runner._ttr_summary(self.run_record(30.0, 3))
        self.assertTrue(s["all_ready"])
        self.assertEqual(s["models_live"], 3)
        self.assertEqual(s["available_bytes_at_start"], 11_000_000_000)
        self.assertFalse(blt.Runner._ttr_summary(self.run_record(30.0, 0))["all_ready"])
        self.assertFalse(blt.Runner._ttr_summary(self.run_record(30.0, None))["all_ready"])
        self.assertFalse(blt.Runner._ttr_summary(self.run_record(-1, 3))["all_ready"])


@contextlib.contextmanager
def chdir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class RunnerFixture:
    """A Runner over fake sidecars and GGUFs with the device-facing calls
    (fresh-host.sh identities, the driver, tts-cli) replaced by recorders."""

    def __init__(self, root: Path, models: str = "models"):
        self.root = root
        self.models = Path(models) if Path(models).is_absolute() else root / models
        for v in MANIFEST["variants"]:
            for s, spec in MANIFEST["sidecar_sets"].items():
                for st in spec["stages"]:
                    for kind in ("mlmodelc", "mlpackage"):
                        p = blt.sidecar(MANIFEST, self.models, v, s, st, kind)
                        (p / "weights").mkdir(parents=True, exist_ok=True)
                        (p / "weights" / "weight.bin").write_text(s)  # which set a clone came from
            gguf = blt.resolve(MANIFEST["variants"][v]["s3gen"], self.models)
            gguf.parent.mkdir(parents=True, exist_ok=True)
            gguf.write_bytes(b"GGUF")
        for exe in ("bench-load-driver", "tts-cli"):
            (root / exe).write_text("#!/bin/sh\n")
        self.args = argparse.Namespace(
            models_root=models, out="out", driver=str(root / "bench-load-driver"), tts_cli=str(root / "tts-cli"),
            units=None, no_quiet_check=True, runs=None, variant=["turbo"], manifest=blt.DEFAULT_MANIFEST,
            allow_battery=False)
        self.identities = []  # [{"id", "binary", "open"}]
        self.calls = []       # driver calls: {"cmd", "identity", "artifacts"}
        self.ready = lambda call: True

    @contextlib.contextmanager
    def fresh_identity(self, binary, hosts_dir):
        ident = {"id": f"qvac-bench-{len(self.identities):08x}", "binary": str(self.root / "host"), "open": True}
        self.identities.append(ident)
        try:
            yield ident
        finally:
            ident["open"] = False

    def run_json(self, cmd, env=None, timeout=3600, ok_codes=(0,)):
        cmd = [str(c) for c in cmd]
        arts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--artifact"]
        parsed = []
        for a in arts:
            stage, rest = a.split("=", 1)
            path, fns = rest.rsplit(":", 1)
            weight = Path(path) / "weights" / "weight.bin"
            parsed.append({"stage": stage, "path": path, "functions": fns.split(","), "exists": Path(path).is_dir(),
                           "origin": weight.read_text() if weight.is_file() else None})
        call = {"cmd": cmd, "identity": self.identities[-1]["id"], "open": self.identities[-1]["open"],
                "artifacts": parsed, "timeout": timeout, "ok_codes": ok_codes}
        self.calls.append(call)
        slots = [{"slot": f"{p['stage']}/{f}", "ready_ms": 1.0} for p in parsed for f in p["functions"]]
        ok = self.ready(call)
        return {"time_to_ready_ms": 1000.0 if ok else -1.0, "models_live": len(slots) if ok else 0,
                "available_bytes_at_start": 11_000_000_000, "footprint_max": 1, "timeline": [],
                "detail": {"t_issue_ms": 0.0, "slots": slots}}

    def runner(self):
        with chdir(self.root):
            return blt.Runner(self.args, MANIFEST)

    def records(self, runner, kind=None):
        recs = blt.read_results(self.root / runner.results)
        return [r for r in recs if kind is None or r["kind"] == kind]

    def patched(self):
        stack = contextlib.ExitStack()
        stack.enter_context(chdir(self.root))
        stack.enter_context(mock.patch.object(blt, "fresh_identity", self.fresh_identity))
        stack.enter_context(mock.patch.object(blt, "run_json", self.run_json))
        return stack


class TtrRunnerTest(unittest.TestCase):
    CFG = {"memory_admission": True, "timeout_s": 60, "configs": [
        {"name": "prebuilt-gated-w3", "route": "prebuilt", "mode": "manager", "set": "w3", "runs": 2,
         "cached_startup_runs": 2},
        {"name": "generate-ungated", "route": "generate", "mode": "ungated", "set": "w7", "runs": 1}]}

    def run_ttr(self, cfg, ready=None):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        fx = RunnerFixture(Path(d.name))
        if ready:
            fx.ready = ready
        r = fx.runner()
        with fx.patched():
            r.ttr(cfg)
        return fx, r

    @staticmethod
    def arg(cmd, name):
        return cmd[cmd.index(name) + 1]

    def test_each_run_gets_a_fresh_identity_and_path(self):
        fx, r = self.run_ttr(self.CFG)
        self.assertEqual(len(fx.identities), 3)  # 2 prebuilt runs + 1 generate run
        self.assertEqual(len({i["id"] for i in fx.identities}), 3)
        self.assertTrue(all(not i["open"] for i in fx.identities), "every identity is cleaned up")
        dirs = {c["identity"]: Path(c["artifacts"][0]["path"]).parent for c in fx.calls}
        self.assertEqual(len(set(dirs.values())), 3, "every run clones the artifacts to a new path")
        dirs = set(dirs.values())
        for c in fx.calls:
            self.assertTrue(c["open"], "the driver runs while its identity exists")
            self.assertTrue(all(a["exists"] for a in c["artifacts"]), c["artifacts"])
            self.assertEqual(self.arg(c["cmd"], "--work-dir"), str(Path(c["artifacts"][0]["path"]).parent / "work"))
            self.assertEqual(self.arg(c["cmd"], "--cache-dir"), str(Path(c["artifacts"][0]["path"]).parent / "cache"))
        for p in dirs:
            self.assertFalse((fx.root / p).exists(), f"{p} is removed after its run")
        recs = fx.records(r, "ttr")
        self.assertEqual([x["identity"] for x in recs], [i["id"] for i in fx.identities])
        self.assertEqual([x["run"] for x in recs], [0, 1, 0])

    def test_artifacts_follow_the_route_and_set(self):
        fx, _ = self.run_ttr(self.CFG)
        for c in fx.calls:
            route = self.arg(c["cmd"], "--route")
            mode = self.arg(c["cmd"], "--mode")
            want = blt.set_functions(MANIFEST, "turbo", "w3" if route == "prebuilt" else "w7")
            self.assertEqual([(a["stage"], a["functions"]) for a in c["artifacts"]], want)
            suffix = ".mlmodelc" if route == "prebuilt" else ".mlpackage"
            self.assertTrue(all(a["path"].endswith(suffix) for a in c["artifacts"]), c["artifacts"])
            self.assertEqual(mode, "manager" if route == "prebuilt" else "ungated")
            names = [Path(a["path"]).name for a in c["artifacts"]]
            self.assertEqual(names, [f"chatterbox-s3gen-{st}{suffix}" for st, _ in want])
            self.assertEqual(self.arg(c["cmd"], "--units"), MANIFEST["protocol"]["units"])
            self.assertEqual(self.arg(c["cmd"], "--timeout-s"), "60")
            self.assertEqual(c["timeout"], 180)
            self.assertEqual(c["ok_codes"], (0, 3))
            # w3 has no HiFT of its own: it is cloned from w7's.
            origin = "w3" if route == "prebuilt" else "w7"
            self.assertEqual([a["origin"] for a in c["artifacts"]],
                             [origin if st != "hift" else "w7" for st, _ in want])

    def test_memory_admission_flag_follows_the_config(self):
        fx, _ = self.run_ttr(self.CFG)
        self.assertTrue(all("--memory-admission" in c["cmd"] for c in fx.calls))
        fx, _ = self.run_ttr({**self.CFG, "memory_admission": False})
        self.assertTrue(fx.calls)
        self.assertFalse(any("--memory-admission" in c["cmd"] for c in fx.calls))

    def test_cached_startup_reuses_the_identity_after_a_ready_run(self):
        fx, r = self.run_ttr(self.CFG)
        by_id = {}
        for c in fx.calls:
            by_id.setdefault(c["identity"], []).append(c["cmd"])
        counts = sorted(len(v) for v in by_id.values())
        self.assertEqual(counts, [1, 3, 3])  # generate: no cached startup; prebuilt: 1 + 2
        for cmds in by_id.values():
            self.assertTrue(all(cmd == cmds[0] for cmd in cmds), "cached startup reruns the same command")
        cached = fx.records(r, "ttr_cached")
        self.assertEqual([(x["run"], x["rep"]) for x in cached], [(0, 0), (0, 1), (1, 0), (1, 1)])
        prime_ids = [x["identity"] for x in fx.records(r, "ttr") if x["config"] == "prebuilt-gated-w3"]
        self.assertEqual([x["identity"] for x in cached], [prime_ids[0]] * 2 + [prime_ids[1]] * 2)

    def test_no_cached_startup_after_a_run_that_was_not_ready(self):
        # The first prebuilt run never gets every function ready; the second does.
        seen = []

        def ready(call):
            seen.append(call["identity"])
            return len(set(seen)) != 1

        fx, r = self.run_ttr(self.CFG, ready)
        ttr = fx.records(r, "ttr")
        self.assertEqual([x["all_ready"] for x in ttr], [False, True, True])
        cached = fx.records(r, "ttr_cached")
        self.assertEqual([x["identity"] for x in cached], [ttr[1]["identity"]] * 2)
        self.assertEqual(sum(1 for c in fx.calls if c["identity"] == ttr[0]["identity"]), 1)


class SynthesisRunnerTest(unittest.TestCase):
    def test_relative_models_root_links_the_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RunnerFixture(Path(d), models="models")
            r = fx.runner()
            seen = []

            def tts(binary, v, text, max_chars, s3gen, coreml, tag):
                p = Path(s3gen)
                seen.append({"name": p.name, "coreml": coreml, "readable": p.is_file() and p.read_bytes() == b"GGUF",
                             "beside_sidecars": (p.parent / "chatterbox-s3gen-cfm.mlmodelc").is_dir()})
                labels = {"encoder": ["coreml-ane"], "cfm": ["coreml-ane"], "hift": ["coreml-ane"]}
                return {"labels": labels, "wall_ms": 1.0, "t3_infer_ms": [1], "s3gen_infer_ms": [1],
                        "t3_load_ms": [1], "s3gen_load_ms": [1], "chunks": [], "peak_footprint": 1, "max_rss": 1}

            cfg = {"workload": "benchmark-prompt", "runs": 1, "configs": [
                {"name": "coreml-w7", "coreml": True, "set": "w7", "stages": ["encoder", "cfm", "hift"]}]}
            with fx.patched(), mock.patch.object(r, "tts", tts):
                r.synthesis(cfg)
            # The priming run and one measured run, both on the GGUF linked beside
            # the cloned sidecars (the engine finds the sidecars from that path).
            want = {"name": "chatterbox-s3gen-q8_0.gguf", "coreml": True, "readable": True, "beside_sidecars": True}
            self.assertEqual(seen, [want, want])
            self.assertEqual(len(fx.records(r, "synthesis")), 1)


class NativeTest(unittest.TestCase):
    """Builds and runs test_bench_ttr.cpp: the driver's time-to-ready bookkeeping."""

    def test_bench_ttr(self):
        cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if cxx is None:
            self.skipTest("no C++ compiler")
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "test_bench_ttr"
            build = subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(HERE),
                                    str(HERE / "test_bench_ttr.cpp"), "-o", str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("test_bench_ttr: ok", run.stdout)


@unittest.skipUnless(sys.platform != "win32", "bash")
class FreshHostTest(unittest.TestCase):
    def run_fh(self, *args, env):
        return subprocess.run(["bash", str(blt.FRESH_HOST), *args], capture_output=True, text=True, env=env)

    def test_cleanup_only_touches_bench_identities(self):
        with tempfile.TemporaryDirectory() as d:
            hosts, caches = Path(d) / "hosts", Path(d) / "caches"
            ident = "qvac-bench-0123abcd-0000-4000-8000-0123456789ab"
            (hosts / ident).mkdir(parents=True)
            (hosts / ident / ident).write_text("bin")
            (caches / ident / "com.apple.e5rt.e5bundlecache").mkdir(parents=True)
            (caches / "tts-cli" / "com.apple.e5rt.e5bundlecache").mkdir(parents=True)
            env = dict(os.environ, QVAC_BENCH_HOSTS=str(hosts), QVAC_BENCH_CACHES=str(caches))
            r = self.run_fh("caches", ident, env=env)
            self.assertEqual(r.stdout.strip(), str(caches / ident))
            r = self.run_fh("cleanup", str(hosts / ident / ident), env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["removed"], 2)
            self.assertFalse((caches / ident).exists())
            self.assertTrue((caches / "tts-cli").exists())
            # Not an identity: refused, nothing deleted.
            r = self.run_fh("cleanup", "tts-cli", env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertTrue((caches / "tts-cli").exists())
            other = "qvac-bench-ffffffff-0000-4000-8000-0123456789ab"
            (caches / other).mkdir()
            r = self.run_fh("cleanup-all", env=env)
            self.assertEqual(json.loads(r.stdout)["removed"], 1)
            self.assertFalse((caches / other).exists())
            self.assertTrue((caches / "tts-cli").exists())

    def test_cleanup_with_spaces_in_the_paths(self):
        with tempfile.TemporaryDirectory() as d:
            hosts, caches = Path(d) / "bench hosts", Path(d) / "cache root"
            ident = "qvac-bench-0123abcd-0000-4000-8000-0123456789ab"
            (hosts / ident).mkdir(parents=True)
            (hosts / ident / ident).write_text("bin")
            (caches / ident / "com.apple.e5rt.e5bundlecache").mkdir(parents=True)
            env = dict(os.environ, QVAC_BENCH_HOSTS=str(hosts), QVAC_BENCH_CACHES=str(caches))
            r = self.run_fh("caches", ident, env=env)
            self.assertEqual(r.stdout.splitlines(), [str(caches / ident)])
            r = self.run_fh("cleanup", str(hosts / ident / ident), env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["removed"], 2)
            self.assertFalse((caches / ident).exists())
            self.assertFalse((hosts / ident).exists())
            self.assertTrue(caches.is_dir())

    def test_new_rejects_a_non_executable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f"
            p.write_text("x")
            r = self.run_fh("new", str(p), env=dict(os.environ, QVAC_BENCH_HOSTS=d))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not an executable", r.stderr)


if __name__ == "__main__":
    unittest.main()
