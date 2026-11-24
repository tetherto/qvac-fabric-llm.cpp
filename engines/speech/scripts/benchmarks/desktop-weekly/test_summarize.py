#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest

import summarize

FAMILIES = {
    "_schema": {},
    "asr": {"task": "ASR", "model_label": "ASR model", "weights": "Q8_0", "audio_duration_seconds": 11.0},
    "tts": {"task": "TTS", "model_label": "TTS model", "weights": "F16", "audio_duration_seconds": None},
}
ASR_RESULT = {"status": "ok", "backend": "CUDA", "runner": "host-cuda", "wall_ms_median": 110.0, "wall_ms_min": 100.0,
              "wall_ms_max": 120.0, "rtf_median": 0.01, "peak_rss_mib": 900.0, "runs": 10, "wer_median": 0.0, "notes": "n"}
TTS_RESULT = {"status": "ok", "backend": "CUDA", "runner": "host-cuda", "wall_ms_median": 2000.0, "wall_ms_min": 1900.0,
              "wall_ms_max": 2100.0, "rtf_median": None, "peak_rss_mib": 1500.0, "runs": 5,
              "tts_intelligibility": {"wer": 0.05, "audio_metadata": {"duration_seconds": 10.0}}, "notes": ""}
GPU_CSV = "2026/11/01 06:00:00.000, 500, 0\n2026/11/01 06:00:00.250, 2500, 90\nbroken line\n"


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = pathlib.Path(self.tmp.name) / "2026-11-01"
        (self.run_dir / "results" / "cuda").mkdir(parents=True)
        (self.run_dir / "machine.json").write_text(json.dumps({"hostname": "box"}))
        (self.run_dir / "pins.json").write_text(json.dumps({"engine": {"sha": "abc"}}))
        self.write_result("cuda", "asr", ASR_RESULT)
        self.write_result("cuda", "tts", TTS_RESULT)
        (self.run_dir / "results" / "cuda" / "asr.gpu.csv").write_text(GPU_CSV)

    def tearDown(self):
        self.tmp.cleanup()

    def write_result(self, backend, family, result):
        (self.run_dir / "results" / backend / f"{family}.json").write_text(json.dumps(result))

    def specs(self):
        return {name: spec for name, spec in FAMILIES.items() if not name.startswith("_")}

    def test_rows_carry_speed_memory_and_quality(self):
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda"], ["asr", "tts"], "2026-11-01T06:00:00Z")
        asr, tts = summary["results"]
        self.assertAlmostEqual(asr["speed_x"], 100.0)
        self.assertEqual(asr["gpu_memory_rise_mib"], 2000.0)
        self.assertEqual(asr["gpu_memory_peak_mib"], 2500.0)
        self.assertEqual(asr["gpu_util_max"], 90.0)
        self.assertEqual(asr["quality"], {"kind": "wer", "value": 0.0})
        self.assertAlmostEqual(tts["speed_x"], 5.0)
        self.assertEqual(tts["audio_seconds"], 10.0)
        self.assertEqual(tts["quality"], {"kind": "tts_wer", "value": 0.05})
        self.assertIsNone(tts["gpu_memory_rise_mib"])

    def test_rows_carry_the_core_ml_encoder_and_the_forced_ggml_time(self):
        result = {"status": "ok", "backend": "ggml-mtl0", "encoder_backend": "coreml-all", "baseline_inference_ms_median": 37.034,
                  "wall_ms_median": 36.669, "wall_ms_min": 36.0, "wall_ms_max": 37.5, "rtf_median": 0.0033, "runs": 10, "notes": ""}
        row = summarize.measured_row("metal", "parakeet-tdt-coreml", {"task": "ASR", "audio_duration_seconds": 11.0}, result, {})
        self.assertEqual((row["encoder_backend"], row["baseline_inference_ms_median"]), ("coreml-all", 37.034))
        self.assertEqual((summarize.unrun_row("metal", "f", {}, "timeout")["encoder_backend"], summarize.unrun_row("metal", "f", {}, "timeout")["baseline_inference_ms_median"]), (None, None))

    def test_status_counts_and_problems(self):
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda", "cpu"], ["asr", "tts"], "")
        self.assertEqual(summary["counts"], {"expected": 4, "ok": 2, "failed": 2})
        self.assertEqual(summary["run"]["status"], "partial")
        statuses = {(row["backend"], row["family"]): row["status"] for row in summary["results"]}
        self.assertEqual(statuses[("cpu", "asr")], "not-run")
        self.assertIn("asr on cpu not-run", summary["problems"])
        self.assertEqual(summary["run"]["host"], "box")
        self.assertEqual(summary["pins"]["engine"]["sha"], "abc")

    def test_a_metal_only_family_is_not_expected_on_the_other_backends(self):
        specs = {**self.specs(), "mlx": {"task": "ASR", "model_label": "MLX model", "weights": "BF16", "backends": ["metal"], "audio_duration_seconds": 11.0}}
        self.write_result("cuda", "mlx", ASR_RESULT)
        summary = summarize.build_summary(self.run_dir, specs, ["cuda", "cpu"], ["asr", "mlx"], "")
        pairs = [(row["backend"], row["family"]) for row in summary["results"]]
        self.assertEqual(pairs, [("cuda", "asr"), ("cpu", "asr")])
        (self.run_dir / "results" / "metal").mkdir()
        self.write_result("metal", "asr", ASR_RESULT)
        self.write_result("metal", "mlx", ASR_RESULT)
        summary = summarize.build_summary(self.run_dir, specs, ["cpu", "metal"], ["asr", "mlx"], "")
        self.assertEqual(summary["counts"], {"expected": 3, "ok": 2, "failed": 1})
        self.assertEqual(summary["problems"], ["asr on cpu not-run"])

    def test_a_skipped_cell_is_not_expected_but_its_family_stays_on_the_other_backends(self):
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda", "cpu"], ["asr", "tts"], "", ["asr:cpu", "tts:cuda"])
        pairs = [(row["backend"], row["family"]) for row in summary["results"]]
        self.assertEqual(pairs, [("cuda", "asr"), ("cpu", "tts")])
        self.assertEqual(summary["counts"], {"expected": 2, "ok": 1, "failed": 1})
        self.assertEqual(summary["problems"], ["tts on cpu not-run"])

    def test_a_cell_sampled_under_load_is_flagged_and_listed_as_a_problem(self):
        (self.run_dir / "results" / "cuda" / "asr.load.json").write_text(json.dumps({"busy_before_pct": 35.2, "load1_before": 9.8, "cpu_count": 28}))
        (self.run_dir / "results" / "cuda" / "tts.load.json").write_text(json.dumps({"busy_before_pct": 3.1, "load1_before": 1.9, "cpu_count": 28}))
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda"], ["asr", "tts"], "")
        asr, tts = summary["results"]
        self.assertEqual((asr["contended"], asr["busy_before_pct"], asr["load1_before"]), (True, 35.2, 9.8))
        self.assertEqual((tts["contended"], tts["busy_before_pct"]), (False, 3.1))
        self.assertEqual(summary["run"]["status"], "ok")
        self.assertEqual(summary["problems"], ["asr on cuda measured under load: 35% of the CPUs were busy with other work just before the cell"])

    def test_build_failure_marks_every_family_on_that_backend(self):
        (self.run_dir / "build-failures.txt").write_text("cpu\n")
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda", "cpu"], ["asr", "tts"], "")
        cpu_rows = [row for row in summary["results"] if row["backend"] == "cpu"]
        self.assertTrue(all(row["status"] == "build-failed" for row in cpu_rows))
        self.assertEqual(summary["problems"][0], "build failed on cpu")

    def test_all_ok_and_none_ok(self):
        summary = summarize.build_summary(self.run_dir, self.specs(), ["cuda"], ["asr", "tts"], "")
        self.assertEqual(summary["run"]["status"], "ok")
        summary = summarize.build_summary(self.run_dir, self.specs(), ["vulkan"], ["asr"], "")
        self.assertEqual(summary["run"]["status"], "failed")

    def test_main_writes_summary_file(self):
        families = pathlib.Path(self.tmp.name) / "families.json"
        families.write_text(json.dumps(FAMILIES))
        summarize.main(["--run-dir", str(self.run_dir), "--families", str(families), "--backends", "cuda",
                        "--families-selected", "asr,tts", "--started", "2026-11-01T06:00:00Z"])
        summary = json.loads((self.run_dir / "summary.json").read_text())
        self.assertEqual(summary["schema"], summarize.SCHEMA_VERSION)
        self.assertEqual(summary["run"]["started"], "2026-11-01T06:00:00Z")
        self.assertEqual(summary["families"], ["asr", "tts"])


if __name__ == "__main__":
    unittest.main()
