#!/usr/bin/env python3
import importlib.util
import pathlib
import re
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


views = load_module("task_views", "task_views.py")

QVAC = "qvac-fabric-speech.cpp"
MACHINE = {"host": "spark-d9ca", "machine": "DGX Spark", "url": "https://example.test/spark", "date": "2026-11-08",
           "newest_run": "2026-11-08", "pins": "qvac abc, ggml def"}


def weekly(**overrides):
    row = {"engine": QVAC, "is_qvac": True, "variant": "Q8_0", "model": "Parakeet TDT 0.6B v3", "task": "ASR", "backend": "cpu",
           "scenario": "jfk", "scenario_label": "JFK clip, 11.0 s", "timing_class": "native_excl_load", "timing_label": "Native bench",
           "timing_includes": "Mel, encoder and decoder", "comparison_set": "weekly", "group": "spark|cpu|Parakeet|ASR|weekly",
           "accepted": True, "status": "measured", "status_detail": "ok", "latency_ms": 385.5, "min_ms": 372.5, "max_ms": 395.8,
           "audio_s": 11.0, "speed_x": 28.5, "rtf": 0.035, "rss_mib": 581.0, "gpu_mib": None, "gpu_kind": "unavailable", "gpu_note": "",
           "wer_pct": 0.0, "wer_kind": "ASR ground truth", "wer_scope": "JFK", "runs": 5, "warmups": 1, "date": "2026-11-08",
           "run": "2026-11-08", "precision": "Q8_0", "precision_source": "families.json", "required_family": True,
           "source_file": "spark-d9ca/2026-11-08.json", "source_row": "parakeet-tdt", "raw_source": "", "samples_ms": "", "command": "parakeet --bench"}
    row.update(overrides)
    return row


def campaign(engine, **overrides):
    row = weekly(engine=engine, is_qvac=engine == QVAC, variant="", scenario="fixed_prompt_short", scenario_label="9-word prompt",
                 model="Audio8 TTS Preview 0.6B", task="TTS", backend="vulkan", timing_class="campaign", timing_label="Campaign measurement",
                 timing_includes="Ratios are competitor time divided by QVAC time.", comparison_set="campaign", model_detail="LM q8_0",
                 group="spark|vulkan|Audio8|LM q8_0|fixed_prompt_short|campaign", min_ms=None, max_ms=None, audio_s=3.1, date="2026-09-30",
                 run="campaign", raw_source="QVAC speech engines report", required_family=False, runs=None, warmups=None, wer_pct=None,
                 ratio_qualifier="qualified: campaign inputs", rss_mib=None)
    row.update(overrides)
    return row


def page(rows, coverage=(), key="asr"):
    page_spec = next(p for p in views.TASK_PAGES if p[0] == key)
    return views.task_view(MACHINE, page_spec, rows, list(coverage))


class AxisTest(unittest.TestCase):
    def test_axis_rounds_up_to_a_round_top(self):
        cases = {(0.82, 0.70): 1.0, (3.21,): 4.0, (611.0,): 800.0, (803.0,): 1000.0, (0.0798,): 0.08}
        for speeds, top in cases.items():
            self.assertAlmostEqual(views.axis_top([weekly(speed_x=s) for s in speeds]), top)
        self.assertEqual(views.axis_top([weekly(accepted=False, speed_x=None)]), views.DEFAULT_AXIS_TOP)

    def test_axis_labels_drop_trailing_zeros(self):
        self.assertIn('<b style="left:25.00%">0.25</b>', views.axis_row(1.0))
        self.assertIn('<b style="left:100.00%">800</b>', views.axis_row(800.0))


class ChartTest(unittest.TestCase):
    def test_bars_sort_by_speed_scale_to_the_axis_and_mark_failures(self):
        rows = [weekly(speed_x=0.70), weekly(engine="audio.cpp", is_qvac=False, variant="", speed_x=0.82),
                weekly(variant="F16", accepted=False, speed_x=None, status_detail="run-failed")]
        html = views.panel_html(rows)
        self.assertLess(html.index("audio.cpp"), html.index(f"{QVAC}<small>Q8_0</small>"))
        self.assertIn('class="bar e-audiocpp" style="width:82.00%"', html)
        self.assertIn('class="bar e-qvac" style="width:70.00%"', html)
        self.assertIn('<span class="hv na">failed</span>', html)
        self.assertLess(html.index("0.70×"), html.index("hv na"))

    def test_a_cell_measured_under_load_is_flagged_on_its_bar(self):
        html = views.panel_html([weekly(contended=True, busy_before_pct=35.2), weekly(engine="audio.cpp", is_qvac=False, variant="", speed_x=20.0)])
        self.assertIn(f'{QVAC}<small>Q8_0</small><small class="ul">under load</small>', html)
        self.assertIn("under load: 35% of the CPUs busy with other work before the cell", html)
        self.assertEqual(html.count("under load"), 2)

    def test_sections_split_synthesis_and_cloning_in_first_appearance_order(self):
        rows = [weekly(model="Supertonic 3", task="TTS", group="a"), weekly(model="Audio8", task="TTS", group="b"),
                weekly(model="Audio8", task="TTS cloning", group="c")]
        titles = re.findall(r"<h2>(.*?) <span", page(rows, key="tts"))
        self.assertEqual(titles, ["Supertonic 3: speech synthesis speed", "Audio8: speech synthesis speed", "Audio8: voice cloning speed"])

    def test_weekly_and_campaign_rows_never_share_a_chart(self):
        rows = [campaign(QVAC), campaign("audio.cpp", latency_ms=500.0, speed_x=1.0), weekly(model="Audio8 TTS Preview 0.6B", task="TTS", backend="vulkan")]
        html = page(rows, key="tts")
        self.assertEqual(re.findall(r'<div class="panel"><h4>(.*?)</h4>', html), ["Vulkan · JFK 11 s", "Vulkan · 9-word prompt · LM q8_0"])
        self.assertIn("inference only · weekly run 2026-11-08", html)
        self.assertIn("campaign harness · campaign 2026-09-30", html)
        self.assertIn(views.esc(views.SETS_NOTE), html)
        self.assertIn(' <span class="tag">required</span>', html)

    def test_the_long_clip_is_its_own_panel(self):
        rows = [weekly(), weekly(scenario="abcba", scenario_label="ABCBA clip, 160.6 s", audio_s=160.6, speed_x=20.0, group="abcba")]
        self.assertEqual(re.findall(r'<div class="panel"><h4>(.*?)</h4>', page(rows)), ["CPU · JFK 11 s", "CPU · ABCBA 160.6 s"])

    def test_a_chart_without_an_accepted_row_moves_into_the_details(self):
        rows = [weekly(), weekly(backend="vulkan", group="vulkan", accepted=False, speed_x=None, status_detail="run-failed")]
        html = page(rows)
        self.assertEqual(re.findall(r'<div class="panel"><h4>(.*?)</h4>', html), ["CPU · JFK 11 s"])
        self.assertIn(views.NO_CHART_TITLE, html)
        self.assertIn("Vulkan · JFK 11 s · inference only: failed (run-failed)", html)


class DetailsTest(unittest.TestCase):
    def test_table_uses_the_report_formats(self):
        row = weekly(latency_ms=17169.9, min_ms=17080.1, max_ms=17221.4, audio_s=12.07, rss_mib=1688.0, rtf=1.4225, runs=3, warmups=1)
        cells = re.findall(r"<td>(.*?)</td>", views.table_row(row))
        self.assertEqual(cells[1:], ["17.17 s (17.08 s to 17.22 s)", "12.07 s", "28.5×", "1.423", "1,688 MiB", "not applicable", "0.0%", "3 / 1"])
        cuda = re.findall(r"<td>(.*?)</td>", views.table_row(weekly(backend="cuda", gpu_mib=1562.0, latency_ms=13.7, min_ms=None)))
        self.assertEqual((cuda[1], cuda[6]), ("13.7 ms", "1,562 MiB"))

    def test_campaign_details_give_the_competitor_ratio_and_failures(self):
        rows = [campaign(QVAC, latency_ms=100.0), campaign("audio.cpp", latency_ms=183.0, speed_x=1.7),
                campaign("transcribe.cpp", accepted=False, speed_x=None, latency_ms=None, status_detail="OOM")]
        html = views.detail_block(rows)
        self.assertIn("audio.cpp: <b>1.83</b>", html)
        self.assertIn("transcribe.cpp: <b>n/a</b> (OOM)", html)
        self.assertIn("Qualifier: Qualified: campaign inputs", html)
        self.assertIn("Campaign method: Ratios are competitor time divided by QVAC time.", html)
        self.assertLess(html.index(f"{QVAC}</td>"), html.index("audio.cpp</td>"))

    def test_the_ratio_falls_back_to_the_rtf_without_a_latency(self):
        qvac = campaign(QVAC, latency_ms=None, rtf=0.139)
        self.assertAlmostEqual(views.time_ratio(campaign("acestep.cpp", latency_ms=None, rtf=0.265), qvac), 0.265 / 0.139)

    def test_weekly_charts_have_no_competitor_ratio(self):
        self.assertNotIn(views.RATIO_TITLE, views.detail_block([weekly(), weekly(variant="F16", speed_x=20.0)]))

    def test_generated_durations_that_differ_are_called_out(self):
        rows = [weekly(task="TTS", audio_s=12.07), weekly(task="TTS", variant="F16", audio_s=14.21)]
        self.assertIn(f"Generated durations differ ({QVAC} (Q8_0) 12.07 s, {QVAC} (F16) 14.21 s)", views.notes_html(rows))
        self.assertEqual(views.notes_html([weekly(), weekly(variant="F16")]), "")

    def test_sources_skip_empty_fields_and_escape_values(self):
        html = views.source_list(weekly(command="parakeet --text <b>", samples_ms=""))
        self.assertIn("<code>parakeet --text &lt;b&gt;</code>", html)
        self.assertNotIn("Samples (ms)", html)
        self.assertIn("<dt>Source</dt><dd>spark-d9ca/2026-11-08.json row parakeet-tdt</dd>", html)


class ReportSetTest(unittest.TestCase):
    def test_report_charts_sit_between_weekly_and_campaign_and_carry_their_labels(self):
        report = [weekly(engine="transcribe.cpp", is_qvac=False, variant="", comparison_set="report", group="r", timing_class="resident_api", date="2026-11-04",
                         scenario="abcba", scenario_label="ABCBA clip", speed_x=20.0, audio_s=160.6),
                  weekly(variant="", comparison_set="report", group="r", timing_class="resident_api", date="2026-11-04", scenario="abcba", speed_x=10.0, audio_s=160.6)]
        rows = [campaign(QVAC), campaign("audio.cpp", latency_ms=500.0, speed_x=1.0)] + report + [weekly()]
        html = page(rows, key="tts") + page(rows)
        self.assertLess(html.index("weekly run 2026-11-08"), html.index("report 2026-11-04"))
        self.assertIn('<h4>CPU · ABCBA 160.6 s</h4><p class="cap">resident API · report 2026-11-04</p>', html)
        self.assertIn(views.esc(views.SETS_NOTE), page(report + [weekly()]))
        self.assertIn("Report charts are copied from the comparison report of 2026-11-04: QVAC with transcribe.cpp", page(report + [weekly()]))

    def test_qvac_only_report_rows_are_listed_in_the_details_without_a_chart(self):
        listed = weekly(variant="", comparison_set="report", group="only", timing_class="native_excl_load", no_chart=True, speed_x=559.0, latency_ms=287.0,
                        min_ms=None, max_ms=None)
        html = page([weekly(), listed])
        self.assertEqual(re.findall(r'<div class="panel"><h4>(.*?)</h4>', html), ["CPU · JFK 11 s"])
        self.assertIn(f"<h4>{views.QVAC_ONLY_TITLE}</h4>", html)
        self.assertIn("qvac-fabric-speech.cpp · CPU · JFK 11 s · inference only: 287.0 ms, 559×", html)

    def test_coverage_without_a_backend_names_no_backend(self):
        coverage = [{"task": "TTS", "status": "not attempted", "model": "Indic original reference", "engine": "all engines", "backend": "", "reason": "HTTP 401"}]
        self.assertIn("Indic original reference · all engines: HTTP 401", views.not_measured_html(coverage, ("TTS",)))


class PageTest(unittest.TestCase):
    def test_header_links_the_sibling_pages_and_the_dashboard(self):
        html = page([weekly()])
        self.assertIn('<a href="#">Speech Bench</a> · DGX Spark · measured 2026-11-08', html)
        self.assertIn('<span class="cur">ASR</span>', html)
        self.assertIn('<a href="#dgx-spark-tts">TTS and voice cloning</a>', html)
        self.assertIn('<a href="https://example.test/spark" target="_blank" rel="noopener">DGX Spark dashboard</a>', html)
        self.assertIn("QVAC from the weekly run 2026-11-08 at qvac abc, ggml def: 1 model on CPU.", html)

    def test_not_measured_groups_backends_and_marks_failures(self):
        coverage = [{"task": "ASR", "status": "not attempted", "model": "Nemotron", "engine": QVAC, "backend": b, "reason": "Skipped"} for b in ("cuda", "cpu")]
        coverage.append({"task": "ASR", "status": "failed", "model": "Parakeet", "engine": QVAC, "backend": "vulkan", "reason": "run-failed"})
        coverage.append({"task": "Music", "status": "failed", "model": "ACE-Step", "engine": QVAC, "backend": "cpu", "reason": "timeout"})
        html = views.not_measured_html(coverage, ("ASR",))
        self.assertIn('<span class="pill">not attempted</span> Nemotron · qvac-fabric-speech.cpp on CPU and CUDA: Skipped', html)
        self.assertIn('<span class="pill bad">failed</span> Parakeet', html)
        self.assertNotIn("ACE-Step", html)
        self.assertEqual(views.not_measured_html([], ("ASR",)), "")

    def test_a_page_without_measurements_opens_the_not_measured_list(self):
        coverage = [{"task": "Music", "status": "not attempted", "model": "ACE-Step", "engine": QVAC, "backend": "cpu", "reason": "Skipped"}]
        self.assertIn('<details class="md" open>', page([], coverage, key="music"))
        self.assertIn(views.NO_WEEKLY, page([], coverage, key="music"))
        self.assertIn(views.EMPTY_PAGE, page([], key="music"))
        self.assertNotIn(" open>", page([weekly()], coverage[:0]))

    def test_every_machine_gets_three_hidden_views(self):
        html = views.render_views([MACHINE, dict(MACHINE, host="mac-mini-m4-gpu", machine="Mac mini M4")], {"spark-d9ca": [weekly()]}, {})
        self.assertEqual(re.findall(r'data-view="([^"]+)" hidden', html),
                         ["dgx-spark-tts", "dgx-spark-asr", "dgx-spark-music", "mac-mini-m4-tts", "mac-mini-m4-asr", "mac-mini-m4-music"])
        self.assertEqual(views.view_token("Mac Studio M3 Ultra", "tts"), "mac-studio-m3-ultra-tts")


if __name__ == "__main__":
    unittest.main()
