#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module("build_dashboard_data", "build-dashboard-data.py")

HOST = "qvac-dev-linux-x64"
FAMILIES = {
    "parakeet-tdt": {"task": "ASR", "model_label": "Parakeet TDT 0.6B v3", "weights": "Q8_0", "bench_kind": "native", "binary": "parakeet/parakeet",
                     "args": ["--model", "${MODEL_DIR}/parakeet.gguf"], "runs": 10, "warmup": 2, "cpu_runs": 5, "cpu_warmup": 1},
    "audio8-lm-f16-dec-f16": {"task": "TTS", "model_label": "Audio8 TTS Preview 0.6B", "weights": "LM F16, codec decoder F16", "bench_kind": "time-wrapped",
                              "binary": "tts/audio8-cli", "args": ["--lm", "${MODEL_DIR}/lm.gguf"], "runs": 5, "warmup": 1, "cpu_runs": 3, "cpu_warmup": 1},
    "audio8-clone-enc-f16": {"task": "TTS cloning", "model_label": "Audio8 TTS Preview 0.6B", "weights": "LM Q8_0, codec encoder F16", "bench_kind": "time-wrapped",
                             "binary": "tts/audio8-cli", "args": [], "runs": 5, "warmup": 1, "cpu_runs": 3, "cpu_warmup": 1},
    "moss-tts-f16": {"task": "TTS", "model_label": "MOSS-TTS v1.5 8.5B", "weights": "F16", "bench_kind": "time-wrapped", "binary": "tts/moss-cli",
                     "args": [], "runs": 3, "warmup": 1, "cpu_runs": 1, "cpu_warmup": 0},
    "parakeet-tdt-abcba": {"task": "ASR", "model_label": "Parakeet TDT 0.6B v3", "weights": "Q8_0", "bench_kind": "native", "binary": "parakeet/parakeet",
                           "args": ["--model", "${MODEL_DIR}/parakeet.gguf"], "scenario": "abcba", "runs": 5, "warmup": 1, "cpu_runs": 3, "cpu_warmup": 1},
    "cosyvoice3-clone": {"task": "TTS cloning", "model_label": "CosyVoice3 0.5B", "weights": "LM Q8_0", "bench_kind": "native", "binary": "tts/cosyvoice-bench",
                         "args": [], "runs": 5, "warmup": 1, "cpu_runs": 3, "cpu_warmup": 1},
    "whisper-small": {"task": "ASR", "model_label": "Whisper small multilingual", "weights": "GGML small", "bench_kind": "time-wrapped",
                      "binary": "engines/whisper/whisper-cli", "args": ["-m", "${MODEL_DIR}/ggml-small.bin"], "runs": 10, "warmup": 2, "cpu_runs": 5, "cpu_warmup": 1},
    "mlx-whisper": {"task": "ASR", "model_label": "Whisper small multilingual", "weights": "FP16 (MLX conversion)", "engine": "mlx-whisper", "platform": "Darwin",
                    "backends": ["metal"], "bench_kind": "native", "binary": "${BENCH_HOME}/mlx/venv/bin/python3", "args": ["${BENCH_HOME}/tools/mlx-bench.py"],
                    "runs": 10, "warmup": 2, "cpu_runs": 1, "cpu_warmup": 0},
    "whisper-small-abcba": {"task": "ASR", "model_label": "Whisper small multilingual", "weights": "GGML small", "bench_kind": "time-wrapped",
                            "binary": "engines/whisper/whisper-cli", "args": ["-m", "${MODEL_DIR}/ggml-small.bin"], "scenario": "abcba", "runs": 5, "warmup": 1,
                            "cpu_runs": 3, "cpu_warmup": 1},
    "mlx-whisper-abcba": {"task": "ASR", "model_label": "Whisper small multilingual", "weights": "FP16 (MLX conversion)", "engine": "mlx-whisper", "platform": "Darwin",
                          "backends": ["metal"], "bench_kind": "native", "binary": "${BENCH_HOME}/mlx/venv/bin/python3", "args": ["${BENCH_HOME}/tools/mlx-bench.py"],
                          "scenario": "abcba", "runs": 5, "warmup": 1, "cpu_runs": 1, "cpu_warmup": 0},
    "parakeet-mlx-abcba": {"task": "ASR", "model_label": "Parakeet TDT 0.6B v3", "weights": "BF16 (MLX conversion)", "engine": "parakeet-mlx", "platform": "Darwin",
                           "backends": ["metal"], "bench_kind": "native", "binary": "${BENCH_HOME}/mlx/venv/bin/python3", "args": ["${BENCH_HOME}/tools/mlx-bench.py"],
                           "scenario": "abcba", "runs": 5, "warmup": 1, "cpu_runs": 1, "cpu_warmup": 0},
}
COMPARISONS = [
    {"task": "ASR", "engine": "transcribe.cpp", "model": "Parakeet TDT 0.6B v3", "quantization": "q8_0", "device": "RTX 5090 box", "backend": "cuda",
     "qvac_rtf": "0.0007 / 0.0006", "ratio": "7.11× / 8.54×", "measured": "2026-09-22", "source": "campaign", "harness": "h",
     "qvac_version": "46afe7d9", "engine_version": "v0.2.3"},
    {"task": "ASR", "engine": "audio.cpp", "model": "Parakeet TDT 0.6B v3", "quantization": "q8_0", "device": "RTX 5090 box", "backend": "cuda",
     "qvac_rtf": "0.0007 / 0.0006", "ratio": "1.84× / 1.85×", "measured": "2026-09-22", "source": "campaign", "harness": "h"},
    {"task": "TTS", "engine": "audio.cpp", "model": "Audio8 TTS Preview 0.6B", "quantization": "LM q8_0", "device": "RTX 5090 box", "backend": "vulkan",
     "qvac_rtf": "2.634 / 2.060", "ratio": "2.22× / OOM", "measured": "2026-09-30", "source": "campaign", "harness": "h"},
    {"task": "Music", "engine": "acestep.cpp", "model": "ACE-Step 1.5 Turbo", "quantization": "Q8_0", "device": "RTX 5090 box", "backend": "cuda",
     "qvac_rtf": "0.139 single set", "ratio": "1.91× single set", "measured": "2026-09-14", "source": "campaign", "harness": "h"},
    {"task": "ASR", "engine": "audio.cpp", "model": "Parakeet TDT 0.6B v3", "quantization": "q8_0", "device": "MacBook Air M5", "backend": "metal",
     "qvac_rtf": "0.0055 / 0.0077", "ratio": "1.46× / 1.14×", "measured": "2026-09-22", "source": "campaign", "harness": "h"},
]
ENGINES = [{"engine": build.QVAC, "role": "QVAC engine", "ggml": "ggml-based", "details": "weekly", "source": "bench"},
           {"engine": "audio.cpp", "role": "prior-report competitor", "ggml": "ggml-based", "details": "campaign", "source": "report"},
           {"engine": "SGLang-Omni", "role": "new competitor, measured", "ggml": "Not ggml-based", "details": "none here", "source": "report"}]


def result(backend, family, speed=10.0, status="ok", wer=0.0125):
    spec = FAMILIES[family]
    return {"backend": backend, "family": family, "task": spec["task"], "model_label": spec["model_label"], "weights": spec["weights"],
            "status": status, "observed_backend": backend.upper(), "wall_ms_median": 1100.0, "wall_ms_min": 1000.0, "wall_ms_max": 1200.0,
            "speed_x": speed if status == "ok" else None, "audio_seconds": 11.0, "peak_rss_mib": 1024.0,
            "gpu_memory_rise_mib": 512.0 if backend == "cuda" else None,
            "quality": {"kind": "wer" if spec["task"] == "ASR" else "tts_wer", "value": wer}, "runs": 3, "notes": f"note {family}"}


def summary(name, started, results, families=None, backends=("cpu", "cuda")):
    return {"schema": 1, "_host": HOST, "_file": f"{HOST}/{name}.json",
            "run": {"name": name, "host": HOST, "started": started, "finished": started, "status": "ok"},
            "machine": {"os_version": "Ubuntu 25.10", "cpu": "Ryzen", "memory_gib": 60,
                        "gpu": {"name": "RTX 5090", "driver": "595", "cuda_toolkit": "13.3", "unified_memory": False}},
            "pins": {"engine": {"sha": "abcdef1234567890", "commit_date": "2026-11-01"}, "ggml": {"sha": "fedcba0987654321", "commit_date": "2026-11-01"},
                     "build": {"cuda_arch": "120a-real"}, "models": [{"name": "parakeet.gguf", "sha256": "0123456789abcdef", "source": "registry parakeet"}]},
            "backends": list(backends), "families": families or sorted({r["family"] for r in results}), "results": results,
            "counts": {"expected": len(results), "ok": sum(1 for r in results if r["status"] == "ok")}, "problems": []}


class WeeklyRowsTest(unittest.TestCase):
    def setUp(self):
        self.older = summary("2026-11-01", "2026-11-01T17:00:00Z", [result("cpu", "parakeet-tdt", 9.0), result("cuda", "audio8-clone-enc-f16", 2.0)])
        self.newer = summary("2026-11-08", "2026-11-08T17:00:00Z", [result("cpu", "parakeet-tdt", 12.0),
                                                                    result("cuda", "audio8-lm-f16-dec-f16", status="run-failed")])

    def test_weekly_row_carries_units_pins_and_provenance(self):
        row = build.weekly_row(self.newer, result("cuda", "parakeet-tdt", 12.0), FAMILIES, "2026-11-08")
        self.assertEqual((row["machine"], row["variant"], row["timing_class"]), ("RTX 5090", "Q8_0", "native_excl_load"))
        self.assertEqual((row["runs"], row["warmups"]), (3, 2))
        self.assertAlmostEqual(row["rtf"], 1 / 12.0)
        self.assertEqual((row["gpu_mib"], row["gpu_kind"]), (512.0, "sampled rise"))
        self.assertEqual((row["wer_pct"], row["wer_kind"]), (1.25, "ASR ground truth"))
        self.assertEqual(row["hashes"], "parakeet.gguf 0123456789")
        self.assertEqual(row["pins"], "qvac abcdef1234 (2026-11-01), ggml fedcba0987 (2026-11-01)")
        self.assertEqual(row["caveats"], "note parakeet-tdt")

    def test_a_cell_measured_under_load_carries_the_flag_a_caveat_and_a_comment(self):
        contended = dict(result("cuda", "parakeet-tdt", 12.0), contended=True, busy_before_pct=35.2)
        row = build.weekly_row(self.newer, contended, FAMILIES, "2026-11-08")
        self.assertEqual((row["contended"], row["busy_before_pct"]), (True, 35.2))
        self.assertIn("Measured under load: 35% of the CPUs", row["caveats"])
        plain = build.weekly_row(self.newer, result("cuda", "parakeet-tdt", 12.0), FAMILIES, "2026-11-08")
        self.assertEqual((plain["contended"], plain["busy_before_pct"]), (False, None))
        items = build.contention_items([row, plain])
        self.assertEqual(items[0][:2], ("summary", "Cells under load"))
        self.assertIn("35% busy", items[1][2])
        self.assertEqual(build.contention_items([plain]), [])

    def test_synthesis_and_cloning_get_short_variants_and_separate_charts(self):
        tts = build.weekly_row(self.newer, result("cuda", "audio8-lm-f16-dec-f16", 3.0), FAMILIES, "2026-11-08")
        clone = build.weekly_row(self.newer, result("cuda", "audio8-clone-enc-f16", 2.0), FAMILIES, "2026-11-08")
        self.assertEqual((tts["variant"], tts["precision"]), ("LM F16 · decoder F16", "LM F16, codec decoder F16"))
        self.assertEqual(clone["variant"], "encoder F16")
        self.assertNotEqual(tts["group"], clone["group"])
        self.assertEqual((tts["timing_class"], clone["timing_class"]), ("fresh_process", "fresh_process_ref"))
        self.assertEqual((tts["dashboard"], clone["dashboard"]), ("tts", "tts"))

    def test_only_time_wrapped_cloning_includes_load_and_reference_encoding(self):
        native = build.weekly_row(self.newer, result("cuda", "cosyvoice3-clone", 3.5), FAMILIES, "2026-11-08")
        wrapped = build.weekly_row(self.newer, result("cuda", "audio8-clone-enc-f16", 2.0), FAMILIES, "2026-11-08")
        self.assertEqual((native["timing_class"], native["timing_label"]), build.TIMING["native"][:2])
        self.assertEqual(wrapped["timing_class"], "fresh_process_ref")

    def test_a_family_with_its_own_clip_gets_that_scenario_and_its_own_chart(self):
        jfk = build.weekly_row(self.newer, result("cuda", "parakeet-tdt", 12.0), FAMILIES, "2026-11-08")
        long = build.weekly_row(self.newer, result("cuda", "parakeet-tdt-abcba", 6.0), FAMILIES, "2026-11-08")
        self.assertEqual((jfk["scenario"], long["scenario"]), ("jfk", "abcba"))
        self.assertEqual(long["scenario_label"], "ABCBA multi-speaker clip, 160.6 s (no reference transcript)")
        self.assertNotEqual(jfk["group"], long["group"])
        self.assertEqual(build.skipped_cell(self.newer, "parakeet-tdt-abcba", FAMILIES["parakeet-tdt-abcba"], "cpu", 1)["scenario"], long["scenario_label"])

    def test_a_core_ml_cell_names_its_encoder_and_keeps_the_forced_ggml_time(self):
        cell = result("metal", "parakeet-tdt", 12.0)
        cell.update({"encoder_backend": "coreml-all", "baseline_inference_ms_median": 37.034})
        row = build.weekly_row(self.newer, cell, FAMILIES, "2026-11-08")
        self.assertEqual(row["observed_backend"], "METAL, encoder coreml-all")
        self.assertIn("The same process forced to the ggml encoder: 37.0 ms median.", row["caveats"])
        plain = build.weekly_row(self.newer, result("metal", "parakeet-tdt", 12.0), FAMILIES, "2026-11-08")
        self.assertEqual((plain["observed_backend"], "ggml" in plain["caveats"]), ("METAL", False))

    def test_a_family_of_another_engine_shares_the_chart_of_the_qvac_family(self):
        qvac = build.weekly_row(self.newer, result("metal", "parakeet-tdt-abcba", 6.0), FAMILIES, "2026-11-08")
        mlx = build.weekly_row(self.newer, result("metal", "parakeet-mlx-abcba", 4.0), FAMILIES, "2026-11-08")
        self.assertEqual((mlx["engine"], mlx["is_qvac"], mlx["variant"], mlx["precision"]), ("parakeet-mlx", False, "BF16", "BF16 (MLX conversion)"))
        self.assertEqual((qvac["engine"], qvac["is_qvac"]), (build.QVAC, True))
        self.assertEqual(mlx["group"], qvac["group"])
        self.assertEqual(mlx["command"], "${BENCH_HOME}/mlx/venv/bin/python3 ${BENCH_HOME}/tools/mlx-bench.py")

    def test_rows_of_different_timing_scopes_never_share_a_chart(self):
        for qvac_family, rival_family in (("whisper-small", "mlx-whisper"), ("whisper-small-abcba", "mlx-whisper-abcba")):
            qvac = build.weekly_row(self.newer, result("metal", qvac_family, 6.0), FAMILIES, "2026-11-08")
            rival = build.weekly_row(self.newer, result("metal", rival_family, 4.0), FAMILIES, "2026-11-08")
            self.assertEqual((qvac["timing_class"], rival["timing_class"]), ("fresh_process", "native_excl_load"))
            self.assertNotEqual(qvac["group"], rival["group"])
            self.assertIn(qvac["timing_class"], qvac["group"])

    def test_families_outside_the_hosts_platform_or_backends_are_not_coverage_gaps(self):
        darwin = summary("2026-11-08", "2026-11-08T17:00:00Z", [result("metal", "parakeet-tdt", 12.0)], backends=("cpu", "metal"))
        darwin["machine"]["os"] = "Darwin"
        linux = summary("2026-11-08", "2026-11-08T17:00:00Z", [result("cuda", "parakeet-tdt", 12.0)])
        linux["machine"]["os"] = "Linux"
        self.assertEqual([(c["model"], c["backend"], c["engine"]) for c in build.skipped_cells([darwin], [], {"parakeet-mlx-abcba": FAMILIES["parakeet-mlx-abcba"]}, 0)],
                         [("Parakeet TDT 0.6B v3", "metal", "parakeet-mlx")])
        self.assertEqual(build.skipped_cells([linux], [], {"parakeet-mlx-abcba": FAMILIES["parakeet-mlx-abcba"]}, 0), [])
        self.assertEqual(len(build.skipped_cells([linux], [], {"moss-tts-f16": FAMILIES["moss-tts-f16"]}, 0)), 2)

    def test_a_family_with_a_stored_row_from_an_older_run_is_not_a_coverage_gap(self):
        runs = build.newest_cells([self.older, self.newer], FAMILIES)
        cells = build.skipped_cells([self.older, self.newer], runs, {"parakeet-tdt": FAMILIES["parakeet-tdt"], "audio8-clone-enc-f16": FAMILIES["audio8-clone-enc-f16"],
                                                                   "moss-tts-f16": FAMILIES["moss-tts-f16"]}, 0)
        self.assertEqual([(c["model"], c["backend"]) for c in cells], [("Parakeet TDT 0.6B v3", "cuda"), ("Audio8 TTS Preview 0.6B", "cpu"),
                                                                       ("MOSS-TTS v1.5 8.5B", "cpu"), ("MOSS-TTS v1.5 8.5B", "cuda")])
        self.assertEqual(cells[0]["reason"], build.SKIPPED_REASON)

    def test_an_engine_can_declare_what_it_cannot_measure_on_a_host(self):
        engines = [{"engine": "mlx-audio", "not_measured": [{"host": HOST, "task": "TTS", "model": "Supertonic 3", "backend": "metal", "reason": "no loader"},
                                                             {"host": "other", "task": "TTS", "model": "X", "backend": "metal", "reason": "elsewhere"}]}]
        cells = build.coverage_rows([self.newer], [], {}, {}, engines)
        self.assertEqual([(c["id"], c["engine"], c["model"], c["backend"], c["status"], c["reason"], c["source"]) for c in cells],
                         [("k000", "mlx-audio", "Supertonic 3", "metal", "not attempted", "no loader", "engines.json")])

    def test_newest_result_wins_and_an_older_cell_says_where_it_comes_from(self):
        cells = {(r["backend"], r["family"]): r for r in build.newest_cells([self.older, self.newer], FAMILIES)}
        self.assertEqual(cells[("cpu", "parakeet-tdt")]["speed_x"], 12.0)
        self.assertIn("earlier run 2026-11-01", cells[("cuda", "audio8-clone-enc-f16")]["caveats"])
        failed = cells[("cuda", "audio8-lm-f16-dec-f16")]
        self.assertEqual((failed["status"], failed["accepted"], failed["speed_x"]), ("failed", False, None))

    def test_history_keeps_every_run_with_its_rtf(self):
        rows = build.history_rows([self.older, self.newer], FAMILIES)
        self.assertEqual([r["run"] for r in rows], ["2026-11-01", "2026-11-01", "2026-11-08", "2026-11-08"])
        self.assertAlmostEqual(rows[0]["rtf"], 1 / 9.0)


class CampaignRowsTest(unittest.TestCase):
    def setUp(self):
        self.rows = build.campaign_rows(HOST, COMPARISONS)

    def test_one_qvac_row_per_chart_and_competitor_time_from_the_ratio(self):
        jfk = [r for r in self.rows if r["task"] == "ASR" and r["scenario"] == "jfk"]
        self.assertEqual(sorted(r["engine"] for r in jfk), ["audio.cpp", build.QVAC, "transcribe.cpp"])
        self.assertEqual(len({r["group"] for r in jfk}), 1)
        qvac = next(r for r in jfk if r["is_qvac"])
        transcribe = next(r for r in jfk if r["engine"] == "transcribe.cpp")
        self.assertEqual((qvac["latency_ms"], qvac["pins"]), (7.7, "46afe7d9"))
        self.assertAlmostEqual(transcribe["rtf"], 0.0007 * 7.11)
        self.assertAlmostEqual(transcribe["speed_x"], 1 / (0.0007 * 7.11), places=2)
        self.assertEqual(transcribe["pins"], "v0.2.3")

    def test_only_the_qvac_row_carries_the_campaign_quantization(self):
        jfk = [r for r in self.rows if r["task"] == "ASR" and r["scenario"] == "jfk"]
        self.assertEqual({r["engine"]: r["precision"] for r in jfk},
                         {build.QVAC: COMPARISONS[0]["quantization"], "audio.cpp": build.NOT_RECORDED, "transcribe.cpp": build.NOT_RECORDED})

    def test_a_competitor_without_a_ratio_is_a_failed_row(self):
        competitor = next(r for r in self.rows if r["scenario"] == "long_prompt" and not r["is_qvac"])
        self.assertEqual((competitor["status"], competitor["status_detail"], competitor["accepted"]), ("failed", "OOM", False))
        self.assertIsNone(competitor["speed_x"])
        self.assertIn("Campaign result: OOM.", competitor["caveats"])

    def test_music_rows_keep_the_rtf_without_a_duration(self):
        music = [r for r in self.rows if r["task"] == "Music"]
        self.assertEqual(len(music), 2)
        self.assertTrue(all(r["latency_ms"] is None for r in music))
        self.assertAlmostEqual(next(r for r in music if not r["is_qvac"])["rtf"], 0.139 * 1.91)

    def test_the_strix_halo_is_strix_1_and_takes_the_campaign_rows(self):
        rows = build.campaign_rows("qvac-dev-strix-1", [dict(COMPARISONS[0], device="Strix Halo")])
        self.assertEqual(build.machine_label("qvac-dev-strix-1"), "Strix Halo")
        self.assertEqual({r["machine"] for r in rows}, {"Strix Halo"})
        self.assertEqual(build.campaign_rows("qvac-dev-strix-0", [dict(COMPARISONS[0], device="Strix Halo")]), [])

    def test_devices_outside_the_lab_are_left_out(self):
        self.assertEqual(len(self.rows), 12)
        self.assertFalse([r for r in self.rows if r["backend"] == "metal"])


def report_row(engine, is_qvac, speed, wer=None, host=HOST, backend="cuda"):
    return {"id": f"r-{engine}", "host": host, "machine": build.machine_label(host), "task": "TTS", "dashboard": "tts", "family": "", "model": "Supertonic 3",
            "model_detail": "", "required_family": True, "engine": engine, "is_qvac": is_qvac, "variant": "", "backend": backend, "observed_backend": "",
            "precision": "Q8_0", "precision_source": "report page", "scenario": "fixed_prompt", "scenario_label": "fixed prompt",
            "timing_class": "fresh_process", "timing_label": "fresh process including load", "timing_includes": "", "comparison_set": "report",
            "status": "measured", "status_detail": "measured", "accepted": True, "latency_ms": 612.6, "min_ms": None, "max_ms": None, "samples_ms": "",
            "audio_s": 12.47, "speed_x": speed, "rtf": 1 / speed, "load_ms": None, "rss_mib": 799.0, "gpu_mib": None, "gpu_kind": "unavailable", "gpu_note": "",
            "wer_pct": wer, "wer_kind": "TTS intelligibility (Whisper-tiny)", "wer_scope": "", "wer_scored_outputs": 1 if wer is not None else None,
            "wer_reason": "", "runs": 3, "warmups": 1, "threads": 5, "date": "2026-11-04", "date_source": "report page", "started_utc": "", "finished_utc": "",
            "pins": "", "model_revisions": "", "hashes": "", "command": "", "config": "", "caveats": "", "comparison_limitation": "", "host_conditions": "",
            "source_file": "report-rows.json", "source_row": 0, "raw_source": "", "group": f"{host}|{backend}|Supertonic 3|TTS|fixed_prompt|report|0",
            "ratio_qualifier": "", "run": "report"}


REPORT = {"rows": [report_row(build.QVAC, True, 20.4, 2.9), report_row("audio.cpp", False, 16.4, 12.0), report_row("audio.cpp", False, 9.0, host="spark-d9ca")],
          "coverage": [{"id": "r000", "host": HOST, "machine": "RTX 5090", "task": "TTS", "model": "Indic original reference", "engine": "all engines",
                        "backend": "", "status": "not attempted", "rows": 0, "reason": "HTTP 401", "source": "report-rows.json", "scenario": ""}]}


class ReportRowsTest(unittest.TestCase):
    def setUp(self):
        self.summary = summary("2026-11-08", "2026-11-08T17:00:00Z", [result("cpu", "parakeet-tdt")])
        self.runs = build.newest_cells([self.summary], FAMILIES) + build.report_rows(HOST, REPORT)

    def test_only_the_machines_own_report_rows_join_its_runs(self):
        self.assertEqual([r["engine"] for r in self.runs if r["comparison_set"] == "report"], [build.QVAC, "audio.cpp"])
        self.assertEqual(build.report_rows("qvac-dev-strix-1", REPORT), [])

    def test_report_coverage_cells_follow_the_weekly_ones_with_fresh_ids(self):
        cells = build.coverage_rows([self.summary], self.runs, FAMILIES, REPORT, [])
        self.assertEqual((cells[-1]["id"], cells[-1]["model"], cells[-1]["status"]), (f"k{len(cells) - 1:03d}", "Indic original reference", "not attempted"))
        self.assertEqual(len(build.coverage_rows([self.summary], self.runs, FAMILIES, {}, [])), len(cells) - 1)

    def test_commentary_and_machine_row_name_the_report(self):
        notes = [n["comment"] for n in build.commentary_rows([self.summary], self.runs, FAMILIES) if n["topic"] == "Comparison report"]
        self.assertEqual(notes, ["2 rows copied from the comparison report of 2026-11-04: QVAC against audio.cpp with that report's own builds, inputs and "
                                 "timing scopes; its charts are never mixed with the weekly ones."])
        self.assertIn("no campaign rows for this machine; 2 report rows from 2026-11-04", build.machine_row([self.summary], self.runs)[0]["work"])

    def test_the_hub_counts_report_engines_and_flags_their_wer(self):
        hub = build.hub_document({HOST: [self.summary]}, {HOST: self.runs}, ENGINES, {})
        self.assertEqual(hub["machines"][0]["tasks"]["tts"], {"models": 1, "backends": ["cuda"], "others": 1, "rows": 2})
        self.assertEqual([(f["engine"], f["wer_pct"]) for f in hub["quality_flags"]], [("audio.cpp", 12.0)])

    def test_build_reads_the_report_next_to_the_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "summaries" / HOST).mkdir(parents=True)
            (root / "summaries" / HOST / "2026-11-08.json").write_text(json.dumps({k: v for k, v in self.summary.items() if not k.startswith("_")}))
            (root / "report-rows.json").write_text(json.dumps(REPORT))
            args = build.parse_args([str(root / "summaries"), "-o", str(root / "out")])
            self.assertEqual(build.report_path(args), str(root / "report-rows.json"))
            build.main([str(root / "summaries"), "-o", str(root / "out"), "--comparisons", "", "--dashboards", ""])
            runs = json.loads((root / "out" / HOST / "runs.json").read_text())
        self.assertEqual(sum(1 for r in runs if r["comparison_set"] == "report"), 2)


class CommentaryTest(unittest.TestCase):
    def setUp(self):
        first = summary("2026-11-01", "2026-11-01T17:00:00Z", [result("cpu", "parakeet-tdt", 10.0), result("cpu", "audio8-lm-f16-dec-f16", 1.0)])
        quick = summary("2026-11-02-quick", "2026-11-02T10:00:00Z", [result("cpu", "parakeet-tdt", 50.0)])
        newest = summary("2026-11-08", "2026-11-08T17:00:00Z",
                         [result("cpu", "parakeet-tdt", 12.0), result("cpu", "audio8-lm-f16-dec-f16", 1.02),
                          result("cuda", "audio8-lm-f16-dec-f16", status="run-failed"), result("cuda", "audio8-clone-enc-f16", status="run-failed")],
                         families=["parakeet-tdt", "audio8-lm-f16-dec-f16", "audio8-clone-enc-f16"])
        self.summaries = [first, quick, newest]
        self.runs = build.newest_cells(self.summaries, FAMILIES) + build.campaign_rows(HOST, COMPARISONS)
        self.notes = build.commentary_rows(self.summaries, self.runs, FAMILIES)

    def topics(self, key):
        return {n["topic"]: n["comment"] for n in self.notes if n["dashboard"] == key}

    def comments(self, key, topic):
        return [n["comment"] for n in self.notes if n["dashboard"] == key and n["topic"] == topic]

    def test_summary_names_fastest_cells_failures_moves_coverage_and_pins(self):
        notes = self.topics(build.SUMMARY)
        self.assertIn("ASR: Parakeet TDT 0.6B v3 (Q8_0) on CPU, 12.0× real time", notes["Fastest cells this week"])
        self.assertIn("2 weekly cells failed", notes["Failed cells"])
        self.assertIn("previous full run (2026-11-01)", notes["Week over week"])
        self.assertIn("+20%", notes["Week over week"])
        self.assertNotIn("Audio8", notes["Week over week"])
        self.assertIn("2 of 4 weekly cells measured on CPU and CUDA", notes["Coverage"])
        self.assertIn("No stored result on this machine for CosyVoice3 0.5B, MOSS-TTS v1.5 8.5B, Parakeet TDT 0.6B v3 and Whisper small multilingual", notes["Coverage"])
        self.assertIn("qvac abcdef1234", notes["Pins"])

    def test_campaign_lines_give_the_ratio_range_per_task(self):
        self.assertEqual(self.comments("asr", "Against other engines"),
                         ["ASR campaign of 2026-09-22 against audio.cpp and transcribe.cpp: competitor ÷ QVAC time 1.84 to 8.54 over 4 pairs; "
                          "QVAC took less time in 4 of 4."])
        self.assertEqual(self.comments("tts", "Against other engines"),
                         ["TTS campaign of 2026-09-30 against audio.cpp: competitor ÷ QVAC time 2.22 over 1 pair; QVAC took less time in 1 of 1."])

    def test_failures_of_synthesis_and_cloning_share_the_tts_tab(self):
        failures = self.comments("tts", "Failed cells")
        self.assertEqual(len(failures), 1)
        self.assertIn("(LM F16 · decoder F16) on CUDA and Audio8 TTS Preview 0.6B (encoder F16) on CUDA", failures[0])

    def test_rules_are_the_general_notes(self):
        self.assertEqual(len(self.topics(build.GENERAL)), len(build.RULES))

    def test_quick_runs_are_never_a_baseline(self):
        notes = build.commentary_rows(self.summaries[1:], self.runs, FAMILIES)
        moves = [n["comment"] for n in notes if n["topic"] == "Week over week"]
        self.assertEqual(moves, ["First full run of this machine: no earlier full run to compare with (quick runs are not baselines)."])


class HubAndFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.summary = summary("2026-11-08", "2026-11-08T17:00:00Z",
                               [result("cpu", "parakeet-tdt"), result("cuda", "parakeet-tdt"), result("cuda", "audio8-lm-f16-dec-f16", 0.7, wer=0.147)])
        self.rows = {HOST: build.newest_cells([self.summary], FAMILIES) + build.campaign_rows(HOST, COMPARISONS)}
        self.links = {HOST: "https://example.test/rtx", "landing": "https://example.test/hub"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_hub_counts_models_backends_and_other_engines_per_task(self):
        hub = build.hub_document({HOST: [self.summary]}, self.rows, ENGINES, self.links)
        machine = hub["machines"][0]
        self.assertEqual((machine["machine"], machine["url"], hub["hub_url"]), ("RTX 5090", "https://example.test/rtx", "https://example.test/hub"))
        self.assertEqual(machine["tasks"]["asr"], {"models": 1, "backends": ["cpu", "cuda"], "others": 2, "rows": 8})
        self.assertEqual(machine["tasks"]["music"]["others"], 1)
        self.assertEqual((hub["measured"], hub["accepted"]), ("2026-11-08", machine["accepted"]))

    def test_quality_flags_keep_tts_rows_at_or_over_the_threshold(self):
        flags = build.hub_document({HOST: [self.summary]}, self.rows, ENGINES, self.links)["quality_flags"]
        self.assertEqual([(f["model"], f["variant"], f["wer_pct"]) for f in flags], [("Audio8 TTS Preview 0.6B", "LM F16 · decoder F16", 14.7)])

    def test_engines_say_where_they_have_rows(self):
        engines = {e["engine"]: e for e in build.engine_rows(ENGINES, self.rows)}
        self.assertEqual((engines["audio.cpp"]["measured_on"], engines["audio.cpp"]["status"]), ("RTX 5090", "measured"))
        self.assertEqual((engines["SGLang-Omni"]["measured_on"], engines["SGLang-Omni"]["status"]), ("", "not measured"))

    def test_build_writes_the_machine_files_and_the_hub(self):
        written = build.build([self.summary], COMPARISONS, ENGINES, FAMILIES, self.links, self.root / "out")
        self.assertEqual(sorted(written[HOST]), ["commentary", "coverage", "engines", "history", "machine", "runs"])
        machine = json.loads((self.root / "out" / HOST / "machine.json").read_text())[0]
        self.assertEqual((machine["backends"], machine["failed"]), ("cpu, cuda", 1))
        coverage = json.loads((self.root / "out" / HOST / "coverage.json").read_text())
        self.assertIn("not attempted", {c["status"] for c in coverage})
        self.assertTrue((self.root / "out" / "hub.json").is_file())

    def test_retired_hosts_are_left_out_of_the_summaries(self):
        for host in ("qvac-dev-strix-0", "qvac-dev-strix-1"):
            (self.root / "in" / host).mkdir(parents=True)
            (self.root / "in" / host / "2026-11-02.json").write_text(json.dumps({k: v for k, v in self.summary.items() if not k.startswith("_")}))
        self.assertEqual([s["_host"] for s in build.load_summaries(self.root / "in")], ["qvac-dev-strix-1"])

    def test_main_reads_summaries_and_tolerates_a_missing_comparison_file(self):
        folder = self.root / "summaries" / HOST
        folder.mkdir(parents=True)
        (folder / "2026-11-08.json").write_text(json.dumps({k: v for k, v in self.summary.items() if not k.startswith("_")}))
        families, engines = self.root / "families.json", self.root / "engines.json"
        families.write_text(json.dumps(FAMILIES))
        engines.write_text(json.dumps({"rows": ENGINES}))
        build.main([str(self.root / "summaries"), "-o", str(self.root / "out"), "--comparisons", str(self.root / "missing.json"),
                    "--families", str(families), "--engines", str(engines), "--dashboards", str(self.root / "none.json")])
        runs = json.loads((self.root / "out" / HOST / "runs.json").read_text())
        self.assertEqual({r["comparison_set"] for r in runs}, {"weekly"})


if __name__ == "__main__":
    unittest.main()
