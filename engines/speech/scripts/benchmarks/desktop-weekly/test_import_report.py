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


importer = load_module("import_report", "import-report.py")

SOURCES = ('<details class="src"><summary>Sources and exact values</summary>'
           '<dl><dt>Run</dt><dd>r502 · Supertonic 3 · qvac-fabric-speech.cpp · Jetson Orin cpu · measured</dd>'
           '<dt>Source</dt><dd>qvac-dev-linux-arm64/followup.json row 0 · raw: results/cpu.json</dd>'
           '<dt>Samples (ms)</dt><dd>6418.9, 6442.8, 6352.8</dd><dt>Audio per run (s)</dt><dd>12.47</dd>'
           '<dt>Precision</dt><dd>Prequantized Q8_0 (source label)</dd><dt>Timing scope</dt><dd>Fresh process incl. model load.</dd>'
           '<dt>Iterations</dt><dd>3 timed · 1 warmup · 6 CPU threads</dd><dt>Device memory</dt><dd>not applicable · CPU backend</dd>'
           '<dt>WER</dt><dd>2.9% · TTS intelligibility (Whisper-tiny) · scorer errors included</dd><dt>Measured</dt><dd>2026-11-04 · snapshot</dd>'
           '<dt>Engine pins</dt><dd><code>qvac 6805d8cb; ggml de1d0f5c</code></dd><dt>Hashes</dt><dd><code>model sha256:be07</code></dd>'
           '<dt>Command</dt><dd><code>supertonic-bench --text &quot;a&quot;</code></dd><dt>Caveats</dt><dd>One short prompt.</dd></dl>'
           '<dl><dt>Run</dt><dd>r504 · Supertonic 3 · audio.cpp · Jetson Orin cpu · measured</dd><dt>Source</dt><dd>row 2</dd>'
           '<dt>Samples (ms)</dt><dd>6716.5, 6490.1, 6980.2</dd><dt>Audio per run (s)</dt><dd>12.39</dd>'
           '<dt>Iterations</dt><dd>3 timed · 1 warmup · 6 CPU threads</dd><dt>Device memory</dt><dd>≥ 960 MiB · nvidia-smi peak</dd>'
           '<dt>WER</dt><dd>n/a · TTS intelligibility (Whisper-tiny) · not scored</dd><dt>Measured</dt><dd>2026-11-04 · snapshot</dd></dl></details>')

PAGE = ('<title>Jetson Orin TTS</title><header><h1>Jetson Orin: TTS and voice cloning</h1></header>'
        '<section class="msec"><h2>Supertonic 3: speech synthesis speed <span class="hib">(higher is better)</span> <span class="tag">required</span></h2>'
        '<div class="panels"><div class="panel"><h4>CPU · fixed prompt</h4></div></div>'
        '<details class="md"><summary>Details</summary>'
        '<div class="gd"><h4>CPU · fixed prompt · fresh process including load</h4><p class="cap"><b>Input</b> Fixed English prompt · <b>Measured</b> 2026-11-04</p>'
        '<p class="cap">Timed region: Process wall time, model load included.</p><div class="scroll"><table><thead><tr><th>Engine</th></tr></thead><tbody>'
        '<tr><td><i class="dot e-qvac"></i> qvac-fabric-speech.cpp</td><td>6.42 s (6.35 s to 6.44 s)</td><td>12.47 s</td><td>1.94×</td><td>0.515</td><td>1,688 MiB</td><td>not applicable</td><td>2.9%</td><td>3 / 1</td></tr>'
        '<tr><td><i class="dot e-audiocpp"></i> audio.cpp (runtime Q8 conversion)</td><td>6.72 s (6.49 s to 6.98 s)</td><td>12.39 s</td><td>1.85×</td><td>0.542</td><td>n/a</td><td>≥ 960 MiB</td><td>n/a</td><td>3 / 1</td></tr>'
        '</tbody></table></div><p class="cap">Precision by row</p><ul class="plain"><li><i class="dot e-qvac"></i> qvac-fabric-speech.cpp: Q8_0 tensors</li>'
        '<li><i class="dot e-audiocpp"></i> audio.cpp (runtime Q8 conversion): runtime Q8</li></ul>'
        '<div class="ratio"><p><b>Competitor ÷ QVAC time</b></p><ul><li>audio.cpp: <b>1.05</b></li></ul><p class="cap">Qualifier: Deployment pair</p></div>'
        + SOURCES + '</div>'
        '<div class="gd"><h4>QVAC-only timings for this model (no competitor in the same setup)</h4><ul class="plain">'
        '<li><i class="dot e-qvac"></i> whisper.cpp (upstream) (NVIDIA driver library retained) · Vulkan · JFK 11 s · native, excluding load: 377.4 ms, 29.1×</li>'
        '<li><i class="dot e-qvac"></i> qvac-fabric-speech.cpp · Vulkan · 30 s request · fresh process including load: failed</li></ul>'
        '<details class="src"><summary>Sources and exact values</summary>'
        '<dl><dt>Run</dt><dd>r600 · Supertonic 3 · whisper.cpp · Jetson Orin vulkan · measured</dd><dt>Samples (ms)</dt><dd>377.4, 376.8, 379.1</dd>'
        '<dt>Audio per run (s)</dt><dd>11.00</dd><dt>Iterations</dt><dd>10 timed · 2 warmup · 4 CPU threads</dd><dt>Measured</dt><dd>2026-11-04</dd></dl>'
        '<dl><dt>Run</dt><dd>r601 · Supertonic 3 · qvac-fabric-speech.cpp · Jetson Orin vulkan · failed</dd><dt>Measured</dt><dd>2026-11-04</dd></dl>'
        '</details></div></details></section>'
        '<details class="md"><summary>Not measured here, and why</summary><ul class="plain">'
        '<li><span class="pill">budget-excluded</span> MOSS-TTS v1.5 8.5B (OpenMOSS) · all engines: resource-budget policy.</li>'
        '<li><span class="pill bad">failed</span> Parakeet Realtime EOU 120M · audio.cpp: unsupported by competitor</li></ul></details>')


class ImportReportTest(unittest.TestCase):
    def setUp(self):
        context = importer.page_context(PAGE, "2026-11-04", "report-rows.json")
        self.rows = importer.page_rows(context, PAGE, 0)
        self.coverage = importer.page_coverage(context, PAGE, 0)

    def test_chart_rows_take_the_table_and_the_source_list(self):
        qvac, audio = self.rows[:2]
        self.assertEqual((qvac["id"], qvac["host"], qvac["machine"], qvac["task"], qvac["model"]), ("r-r502", "qvac-dev-linux-arm64", "Jetson Orin", "TTS", "Supertonic 3"))
        self.assertEqual((qvac["engine"], qvac["variant"], qvac["is_qvac"], qvac["required_family"]), ("qvac-fabric-speech.cpp", "", True, True))
        self.assertEqual((qvac["backend"], qvac["scenario"], qvac["timing_class"], qvac["comparison_set"]), ("cpu", "fixed_prompt", "fresh_process", "report"))
        self.assertEqual((qvac["latency_ms"], qvac["min_ms"], qvac["max_ms"], qvac["audio_s"]), (6418.9, 6352.8, 6442.8, 12.47))
        self.assertEqual((qvac["speed_x"], qvac["rtf"], qvac["rss_mib"], qvac["gpu_kind"]), (1.94, 0.515, 1688.0, "not applicable"))
        self.assertEqual((qvac["wer_pct"], qvac["wer_kind"], qvac["runs"], qvac["warmups"], qvac["threads"]), (2.9, "TTS intelligibility (Whisper-tiny)", 3, 1, 6))
        self.assertEqual((qvac["precision"], qvac["pins"], qvac["command"], qvac["ratio_qualifier"]), ("Q8_0 tensors", "qvac 6805d8cb; ggml de1d0f5c", 'supertonic-bench --text "a"', "Deployment pair"))
        self.assertEqual(qvac["timing_includes"], "Process wall time, model load included.")
        self.assertEqual((audio["engine"], audio["variant"], audio["is_qvac"], audio["precision"]), ("audio.cpp", "runtime Q8 conversion", False, "runtime Q8"))
        self.assertEqual((audio["gpu_mib"], audio["gpu_kind"], audio["wer_pct"], audio["wer_reason"]), (960.0, "≥ 960 MiB", None, "not scored"))
        self.assertEqual(qvac["group"], audio["group"])

    def test_qvac_only_items_become_rows_without_a_shared_chart(self):
        whisper, failed = self.rows[2:]
        self.assertEqual((whisper["engine"], whisper["variant"], whisper["backend"], whisper["scenario"]), ("whisper.cpp (upstream)", "NVIDIA driver library retained", "vulkan", "jfk"))
        self.assertEqual((whisper["timing_class"], whisper["latency_ms"], whisper["speed_x"], whisper["runs"]), ("native_excl_load", 377.4, 29.1, 10))
        self.assertEqual((failed["status"], failed["accepted"], failed["latency_ms"], failed["scenario"]), ("failed", False, None, "music_30s"))
        self.assertNotEqual(whisper["group"], failed["group"])
        self.assertEqual((whisper["no_chart"], failed["no_chart"], self.rows[0]["no_chart"]), (True, True, False))
        self.assertNotEqual(whisper["group"], self.rows[0]["group"])

    def test_a_model_qualifier_moves_into_the_model_detail(self):
        self.assertEqual(importer.split_model("MOSS-TTS v1.5 8.5B (OpenMOSS)"), ("MOSS-TTS v1.5 8.5B", "OpenMOSS"))
        self.assertEqual(importer.split_model("Chatterbox Turbo (Q8 package pair)"), ("Chatterbox Turbo", "Q8 package pair"))
        self.assertEqual(importer.split_model("Whisper tiny multilingual"), ("Whisper tiny multilingual", ""))

    def test_not_measured_items_become_coverage_cells(self):
        self.assertEqual([(c["status"], c["model"], c["engine"], c["task"]) for c in self.coverage],
                         [("budget-excluded", "MOSS-TTS v1.5 8.5B (OpenMOSS)", "all engines", "TTS"), ("failed", "Parakeet Realtime EOU 120M", "audio.cpp", "TTS")])
        self.assertEqual(self.coverage[0]["reason"], "resource-budget policy.")

    def test_main_writes_one_row_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "Jetson-Orin-TTS.html").write_text(PAGE, encoding="utf-8")
            importer.main([str(folder), "--date", "2026-11-04", "--hub", "https://example.test/hub", "-o", str(folder / "report-rows.json")])
            doc = json.loads((folder / "report-rows.json").read_text(encoding="utf-8"))
        self.assertEqual((len(doc["rows"]), len(doc["coverage"]), doc["_schema"]["report"]["pages"]), (4, 2, ["Jetson-Orin-TTS.html"]))
        self.assertEqual(doc["rows"][0]["source_file"], "report-rows.json")


if __name__ == "__main__":
    unittest.main()
