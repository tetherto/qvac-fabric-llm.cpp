#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


highlights = load_module("build_highlights", "build-highlights.py")

QVAC = "qvac-fabric-speech.cpp"
HUB = {"hub_url": "https://example.test/hub", "measured": "2026-11-04"}
MACHINE = [{"host": "spark-d9ca", "machine": "DGX Spark", "hardware": "NVIDIA GB10"}]
CARD = {"task": "ASR", "model": "Parakeet TDT 0.6B v3", "scenario": "abcba", "timing_class": "resident_api", "set": "report"}
CONFIG = {"title": "Speech Bench highlights", "lede": "A few results.", "machines": [{"host": "spark-d9ca", "backend": "cuda", "cards": [CARD]}]}


def row(engine, latency_ms, **overrides):
    base = {"host": "spark-d9ca", "machine": "DGX Spark", "task": "ASR", "dashboard": "asr", "model": "Parakeet TDT 0.6B v3", "engine": engine,
            "is_qvac": engine == QVAC, "variant": "", "backend": "cuda", "scenario": "abcba", "scenario_label": "ABCBA", "timing_class": "resident_api",
            "timing_label": "resident API", "comparison_set": "report", "group": "spark-d9ca|cuda|Parakeet TDT 0.6B v3|ASR|abcba|report|2",
            "accepted": True, "status": "measured", "latency_ms": latency_ms, "audio_s": 160.6, "speed_x": 160600 / latency_ms, "rtf": 0.002,
            "date": "2026-11-03", "ratio_qualifier": "Same checkpoint, separate conversions", "no_chart": False}
    base.update(overrides)
    return base


ROWS = [row(QVAC, 282.6, variant="Parakeet resident C++ API"), row("audio.cpp", 551.2), row("transcribe.cpp", 1896.4),
        row(QVAC, 287.0, timing_class="native_excl_load", group="spark-d9ca|cuda|Parakeet TDT 0.6B v3|ASR|abcba|report|4-2", no_chart=True),
        row(QVAC, 5000.0, group="spark-d9ca|cuda|Parakeet TDT 0.6B v3|ASR|abcba|report|9", variant="160 ms streaming policy"),
        row("NeMo-Speech.cpp", 9000.0, group="spark-d9ca|cuda|Parakeet TDT 0.6B v3|ASR|abcba|report|9")]


class SelectionTest(unittest.TestCase):
    def test_rtf_comes_from_latency_and_audio_not_the_rounded_field(self):
        self.assertAlmostEqual(highlights.rtf_of(ROWS[0]), 0.00176, places=5)
        self.assertEqual(highlights.rtf_of({"rtf": 0.5}), 0.5)

    def test_a_card_without_a_variant_prefers_the_group_with_a_plain_qvac_row_then_the_first_group(self):
        machine = {"host": "spark-d9ca", "backend": "cuda"}
        selected = highlights.select_rows(ROWS, machine, CARD)
        self.assertEqual([r["engine"] for r in selected], [QVAC, "audio.cpp", "transcribe.cpp"])
        plain = highlights.select_rows(ROWS + [row(QVAC, 300.0, group="g-plain"), row("audio.cpp", 400.0, group="g-plain")], machine, CARD)
        self.assertEqual(plain[0]["group"], "g-plain")
        streaming = highlights.select_rows(ROWS, machine, {**CARD, "variant": "160 ms streaming policy"})
        self.assertEqual([r["engine"] for r in streaming], [QVAC, "NeMo-Speech.cpp"])

    def test_rows_without_a_chart_or_without_acceptance_are_not_drawn(self):
        machine = {"host": "spark-d9ca", "backend": "cuda"}
        self.assertEqual(highlights.select_rows(ROWS, machine, {**CARD, "timing_class": "native_excl_load"}), [])
        self.assertEqual(highlights.select_rows([row(QVAC, 10.0, accepted=False)], machine, CARD), [])


class CardTest(unittest.TestCase):
    def test_card_states_the_lead_over_the_fastest_other_engine_with_bars_and_ratios(self):
        page = highlights.card_html({"host": "spark-d9ca", "machine": "DGX Spark", "backend": "cuda"}, ROWS[:3], HUB["hub_url"])
        self.assertIn('<p class="win"><b>2.0×</b> lower RTF than the fastest other engine</p>', page)
        self.assertIn("QVAC RTF 0.00176 (283 ms for 160.6 s of audio)", page)
        self.assertIn('<span class="lab">audio.cpp</span>', page)
        self.assertIn('<small>2.0× QVAC</small>', page)
        self.assertIn('<small>6.7× QVAC</small>', page)
        self.assertIn('style="width:100.00%"', page)
        self.assertIn('<p class="note">Same checkpoint, separate conversions</p>', page)
        self.assertIn('href="https://example.test/hub#dgx-spark-asr"', page)
        self.assertIn("ABCBA 160.6 s · resident API · report 2026-11-03", page)

    def test_card_names_the_engine_that_leads_when_qvac_is_slower(self):
        page = highlights.card_html({"host": "spark-d9ca", "machine": "DGX Spark", "backend": "cuda"}, [row(QVAC, 487.0), row("audio.cpp", 450.6)], "")
        self.assertIn('<p class="win loss"><b>1.1×</b> higher RTF than audio.cpp, which leads here</p>', page)
        self.assertIn("<small>0.9× QVAC</small>", page)

    def test_weekly_qvac_bars_keep_their_variants_and_every_qvac_variant_is_drawn(self):
        rows = [row(QVAC, 100.0, comparison_set="weekly", variant="Q8_0"), row("parakeet-mlx", 150.0, comparison_set="weekly", variant="bf16"),
                row(QVAC, 90.0, comparison_set="weekly", variant="Q8_0, Core ML encoder on the ANE")]
        page = highlights.card_html({"host": "spark-d9ca", "machine": "DGX Spark", "backend": "cuda"}, rows, "")
        self.assertIn('<span class="lab">QVAC Q8_0</span>', page)
        self.assertIn('<span class="lab">QVAC Q8_0, Core ML encoder on the ANE</span>', page)
        self.assertIn('<span class="lab">parakeet-mlx (bf16)</span>', page)
        self.assertEqual(page.count('class="row q"'), 2)
        self.assertEqual(page.count("× QVAC</small>"), 1)
        self.assertIn("QVAC RTF 0.00056 (90 ms", page)


class PageTest(unittest.TestCase):
    def test_build_writes_the_page_and_names_the_cards_it_could_not_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "hub.json").write_text(json.dumps(HUB), encoding="utf-8")
            (folder / "spark-d9ca").mkdir()
            (folder / "spark-d9ca" / "machine.json").write_text(json.dumps(MACHINE), encoding="utf-8")
            (folder / "spark-d9ca" / "runs.json").write_text(json.dumps(ROWS), encoding="utf-8")
            config = {**CONFIG, "machines": [{"host": "spark-d9ca", "backend": "cuda", "cards": [CARD, {**CARD, "model": "Nemotron 3.5 ASR Streaming 0.6B"}]}]}
            (folder / "highlights.json").write_text(json.dumps(config), encoding="utf-8")
            highlights.main([str(folder), "--config", str(folder / "highlights.json"), "-o", str(folder / "out.html")])
            page = (folder / "out.html").read_text(encoding="utf-8")
            _, skipped = highlights.build(str(folder), config)
        self.assertIn("<title>Speech Bench highlights</title>", page)
        self.assertIn('<h2>DGX Spark</h2><p><span class="chip b-cuda">CUDA</span> NVIDIA GB10</p>', page)
        self.assertEqual(page.count('<article class="card">'), 1)
        self.assertIn("ASR is timed as resident API.", page)
        self.assertIn("Charts labelled report come from the comparison report of 2026-11-04", page)
        self.assertIn('href="https://example.test/hub" target="_blank" rel="noopener">full Speech Bench hub</a>', page)
        self.assertEqual(skipped, ["spark-d9ca cuda ASR Nemotron 3.5 ASR Streaming 0.6B abcba resident_api report"])

    def test_a_machine_without_data_is_skipped_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "hub.json").write_text(json.dumps(HUB), encoding="utf-8")
            (folder / "spark-d9ca").mkdir()
            (folder / "spark-d9ca" / "machine.json").write_text(json.dumps(MACHINE), encoding="utf-8")
            (folder / "spark-d9ca" / "runs.json").write_text(json.dumps(ROWS), encoding="utf-8")
            config = {**CONFIG, "machines": [{"host": "qvac-dev-strix-1", "backend": "vulkan", "cards": [CARD]},
                                             {"host": "spark-d9ca", "backend": "cuda", "cards": [CARD]}]}
            page, skipped = highlights.build(str(folder), config)
        self.assertIn("<h2>DGX Spark</h2>", page)
        self.assertNotIn("Strix", page)
        self.assertEqual(skipped, ["qvac-dev-strix-1 (no data for this machine)"])


if __name__ == "__main__":
    unittest.main()
