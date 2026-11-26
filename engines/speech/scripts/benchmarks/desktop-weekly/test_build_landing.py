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


landing = load_module("build_landing", "build-landing.py")


def tasks(asr=(0, (), 0), tts=(0, (), 0), music=(0, (), 0)):
    return {key: {"models": models, "backends": list(backends), "others": others, "rows": models * 2}
            for key, (models, backends, others) in (("asr", asr), ("tts", tts), ("music", music))}


def hub(flags=()):
    return {"measured": "2026-11-08", "accepted": 12, "hub_url": "https://example.test/hub", "quality_flags": list(flags),
            "machines": [
                {"host": "mac-mini-m4-gpu", "machine": "Mac mini <M4>", "url": "", "newest_run": "2026-11-08", "status": "ok", "ok": 4, "expected": 4,
                 "pins": "qvac abc", "tasks": tasks(asr=(1, ("metal",), 0))},
                {"host": "qvac-dev-linux-x64", "machine": "RTX 5090", "url": "https://example.test/rtx", "newest_run": "2026-11-08", "status": "partial",
                 "ok": 9, "expected": 10, "pins": "qvac abc", "tasks": tasks(asr=(2, ("cpu", "cuda"), 3), tts=(1, ("cuda",), 1))}],
            "engines": [{"engine": "qvac-fabric-speech.cpp", "role": "QVAC engine", "status": "measured", "measured_on": "RTX 5090", "details": "weekly", "source": "bench"},
                        {"engine": "audio.cpp", "role": "prior-report competitor", "status": "measured", "measured_on": "RTX 5090", "details": "ggml-based. <x>", "source": "report"},
                        {"engine": "SGLang-Omni", "role": "new competitor, measured", "status": "not measured", "measured_on": "", "details": "none", "source": "report"}]}


FLAG = {"engine": "qvac-fabric-speech.cpp", "model": "Audio8 TTS Preview 0.6B", "variant": "LM Q4_0 · decoder F16", "machine": "RTX 5090", "backend": "cpu",
        "task": "TTS", "wer_pct": 14.71, "speed_x": 0.721, "wer_kind": "TTS intelligibility"}


ROW = {"engine": "qvac-fabric-speech.cpp", "is_qvac": True, "variant": "Q8_0", "model": "Parakeet TDT 0.6B v3", "task": "ASR", "backend": "cuda",
       "scenario": "jfk", "scenario_label": "JFK clip", "timing_class": "native_excl_load", "timing_label": "Native bench", "comparison_set": "weekly",
       "group": "rtx|cuda|parakeet|ASR|weekly", "accepted": True, "status": "measured", "latency_ms": 16.9, "audio_s": 11.0, "speed_x": 648.0,
       "rtf": 0.0015, "date": "2026-11-08", "run": "2026-11-08", "precision": "Q8_0"}


class BuildLandingTest(unittest.TestCase):
    def setUp(self):
        self.page = landing.render(hub(), "2026-11-09 08:00 UTC")

    def test_task_boxes_open_the_task_page_of_each_machine(self):
        self.assertIn('<a href="#rtx-5090-asr"><b>RTX 5090</b><small>2 models · CPU, CUDA · 3 other engines</small></a>', self.page)
        self.assertIn("1 model · CUDA · 1 other engine<", self.page)
        self.assertIn('<a href="#mac-mini-m4-asr"><b>Mac mini &lt;M4&gt;</b>', self.page)
        self.assertLess(self.page.index("RTX 5090"), self.page.index("Mac mini"))

    def test_machines_without_rows_say_so(self):
        self.assertIn(f"<b>RTX 5090</b><small>{landing.NO_ROWS}</small>", self.page)

    def test_task_pages_are_hidden_views_behind_the_hub(self):
        page = landing.render(hub(), "2026-11-09 08:00 UTC", {"qvac-dev-linux-x64": [ROW]}, {})
        self.assertLess(page.index('data-view="hub"'), page.index('data-view="rtx-5090-tts"'))
        self.assertIn('<div class="view" data-view="rtx-5090-asr" hidden>', page)
        self.assertIn("RTX 5090: ASR", page)
        self.assertIn("Parakeet TDT 0.6B v3: transcription speed", page)
        self.assertIn(".msec {", page)
        self.assertIn("document.addEventListener('click', follow);", page)

    def test_quality_flags_list_the_flagged_rows(self):
        self.assertIn("No TTS row reaches the 10% intelligibility WER flag", self.page)
        flagged = landing.render(hub([FLAG]), "2026-11-09 08:00 UTC")
        self.assertIn("intelligibility WER <b>14.7%</b> at 0.72×", flagged)

    def test_engines_are_split_by_role_and_escaped(self):
        for title in ("Weekly engine", "From the previous comparison", "New competitors"):
            self.assertIn(f"<h3>{title}</h3>", self.page)
        self.assertIn("ggml-based. &lt;x&gt;", self.page)
        self.assertIn("<td>not measured</td><td>none</td>", self.page)

    def test_header_and_method_name_the_run_pins_and_page_links(self):
        self.assertIn("<title>Speech Bench</title>", self.page)
        self.assertIn("2 machines · 12 accepted measurements", self.page)
        self.assertIn("run 2026-11-08 (partial, 9 of 10 cells) at qvac abc", self.page)
        self.assertIn("https://example.test/hub#dgx-spark-tts", self.page)
        self.assertIn('<a href="https://example.test/rtx" target="_blank" rel="noopener">RTX 5090</a>: the machine dashboards', self.page)

    def test_main_reads_the_machine_rows_next_to_hub_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            source, out = folder / "hub.json", folder / "landing.html"
            source.write_text(json.dumps(hub()))
            (folder / "qvac-dev-linux-x64").mkdir()
            (folder / "qvac-dev-linux-x64" / "runs.json").write_text(json.dumps([ROW]))
            landing.main([str(source), "-o", str(out)])
            self.assertIn("Parakeet TDT 0.6B v3: transcription speed", out.read_text())


if __name__ == "__main__":
    unittest.main()
