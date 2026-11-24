#!/usr/bin/env python3
import copy
import unittest

import report


def row(backend, family, status="ok", wall=1000.0, rss=1024.0, speed=11.0):
    return {"backend": backend, "family": family, "task": "ASR", "model_label": f"Model {family}", "weights": "Q8_0",
            "status": status, "observed_backend": backend.upper(), "runner": f"host-{backend}", "wall_ms_median": wall,
            "wall_ms_min": None if wall is None else wall - 10, "wall_ms_max": None if wall is None else wall + 10, "rtf_median": None, "audio_seconds": 11.0,
            "speed_x": speed, "peak_rss_mib": rss, "gpu_memory_peak_mib": 3072.0, "gpu_memory_rise_mib": 2048.0,
            "gpu_util_max": 95.0, "quality": {"kind": "wer", "value": 0.0}, "runs": 10, "notes": ""}


def summary(name="2026-11-08"):
    return {"schema": 1, "run": {"name": name, "host": "box", "started": "s", "finished": "f", "status": "partial"},
            "machine": {"hostname": "box", "os_version": "Ubuntu", "kernel": "k", "arch": "x86_64", "cpu": "cpu", "cpu_count": 10,
                        "memory_gib": 60, "gpu": {"name": "RTX", "memory_mib": "32607", "driver": "595", "cuda_toolkit": "13.3",
                                                  "vulkan": "deviceName = RTX", "index": "0", "unified_memory": False},
                        "tools": {"cmake": "cmake 3.31", "ninja": "1.12", "compiler": "gcc 15", "python": "3.13", "node": "v20"}},
            "pins": {"engine": {"repo": "r", "branch": "master", "sha": "abc123", "commit_date": "2026-11-01", "pin": ""},
                     "ggml": {"repo": "g", "branch": "speech", "sha": "def456", "commit_date": "2026-11-01", "pin": ""},
                     "build": {"cuda_arch": "120a-real", "ggml_native": "default (ON)"}, "registry_client": "0.7.0",
                     "tts_asr_scorer": {"repo": "ggerganov/whisper.cpp", "revision": "5359861c", "name": "ggml-tiny.bin"},
                     "families_sha256": "ffff", "models": [{"name": "m.gguf", "present": True, "bytes": 1073741824, "sha256": "0123456789abcdef", "source": "registry x"}]},
            "backends": ["cuda", "cpu"], "families": ["a", "b"],
            "results": [row("cuda", "a"), row("cuda", "b", status="timeout", wall=None, rss=None, speed=None),
                        row("cpu", "a", wall=5000.0, speed=2.2), row("cpu", "b", wall=6000.0, speed=1.8)],
            "counts": {"expected": 4, "ok": 3, "failed": 1}, "build_failures": [],
            "problems": ["b on cuda timeout"]}


class ReportTest(unittest.TestCase):
    def test_render_has_every_section_and_rows(self):
        text = report.render(summary(), None, 5.0)
        for heading in ("## Machine", "## Pins", "### Model files", "## Results on cuda", "## Results on cpu",
                        "## Speed across backends", "## Changes since the previous run", "## Problems"):
            self.assertIn(heading, text)
        self.assertIn("**Status: partial** (3 of 4 cells ok)", text)
        self.assertIn("| Model a | Q8_0 | ok | 1,000 ms (990 ms to 1,010 ms) | 11x | 1.0 GiB | 2.0 GiB rise (peak 3.0 GiB) | WER 0.0% | CUDA | 10 |", text)
        self.assertIn("| Model b | Q8_0 | **timeout** | - | - | - | - | - | CUDA | 10 |", text)
        self.assertIn("| Model a | Q8_0 | 11x | 2.20x |", text)
        self.assertIn("| Model b | Q8_0 | timeout | 1.80x |", text)
        self.assertIn("| m.gguf | 1.0 GiB | `0123456789ab` | registry x |", text)
        self.assertIn("- b on cuda timeout", text)
        self.assertIn("- no previous run to compare with", text)

    def test_a_family_expected_on_one_backend_only_shows_an_empty_speed_cell_on_the_others(self):
        data = summary()
        data["families"].append("c")
        data["results"].append(row("cuda", "c"))
        text = report.render(data, None, 5.0)
        self.assertIn("| Model c | Q8_0 | 11x | n/a |", text)

    def test_a_family_without_any_row_in_the_run_is_left_out_of_the_speed_table(self):
        data = summary()
        data["families"].append("metal-only")
        text = report.render(data, None, 5.0)
        self.assertNotIn("metal-only", text)
        self.assertIn("| Model b | Q8_0 | timeout | 1.80x |", text)

    def test_a_cell_measured_under_load_says_so_in_its_status(self):
        data = summary()
        data["results"][2]["contended"] = True
        data["results"][2]["busy_before_pct"] = 35.2
        text = report.render(data, None, 5.0)
        self.assertIn("| Model a | Q8_0 | ok (under load) |", text)
        self.assertIn("| Model a | Q8_0 | ok | 1,000 ms", text)

    def test_cpu_rows_hide_gpu_memory(self):
        text = report.render(summary(), None, 5.0)
        self.assertIn("| Model a | Q8_0 | ok | 5,000 ms (4,990 ms to 5,010 ms) | 2.20x | 1.0 GiB | - | WER 0.0% | CPU | 10 |", text)

    def test_changes_list_only_moves_over_threshold(self):
        previous = summary("2026-11-01")
        current = summary()
        current["results"][0]["wall_ms_median"] = 1100.0
        current["results"][2]["peak_rss_mib"] = 1030.0
        text = report.render(current, previous, 5.0)
        self.assertIn("## Changes over 5% since 2026-11-01", text)
        self.assertIn("| Model a (Q8_0) on cuda: wall median | 1,000 ms | 1,100 ms | +10.0% |", text)
        self.assertNotIn("Model a (Q8_0) on cpu: peak RSS", text)
        self.assertIn("Pins that changed: none.", text)

    def test_pin_changes_are_named(self):
        previous = summary("2026-11-01")
        current = summary()
        current["pins"]["engine"]["sha"] = "new"
        current["pins"]["models"][0]["sha256"] = "changed"
        text = report.render(current, previous, 5.0)
        self.assertIn("Pins that changed: engine, models.", text)

    def test_failed_cells_are_not_compared(self):
        previous = summary("2026-11-01")
        current = copy.deepcopy(previous)
        current["run"]["name"] = "2026-11-08"
        previous["results"][1] = row("cuda", "b", wall=100.0)
        text = report.render(current, previous, 5.0)
        self.assertNotIn("Model b (Q8_0) on cuda", text)

    def test_build_line_omits_cuda_arch_without_cuda(self):
        self.assertEqual(report.build_line({"cuda_arch": "120a-real", "ggml_native": "ON"}), "- **Build:** CUDA arch 120a-real, GGML_NATIVE ON")
        self.assertEqual(report.build_line({"cuda_arch": "", "ggml_native": "ON"}), "- **Build:** GGML_NATIVE ON")

    def test_formatters(self):
        self.assertEqual(report.fmt_ms(12345.0), "12.3 s")
        self.assertEqual(report.fmt_ms(999.4), "999 ms")
        self.assertEqual(report.fmt_speed(0.5), "0.50x")
        self.assertEqual(report.fmt_speed(250.0), "250x")
        self.assertEqual(report.fmt_gib(1536.0), "1.5 GiB")
        self.assertEqual(report.fmt_percent(0.123), "12.3%")


if __name__ == "__main__":
    unittest.main()
