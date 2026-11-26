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


build = load_module("build_artifact", "build-artifact.py")


def row(backend, family, speed=10.0, status="ok", task="ASR"):
    return {"backend": backend, "family": family, "task": task, "model_label": f"Model {family}", "weights": "Q8_0", "status": status,
            "observed_backend": backend.upper(), "wall_ms_median": 1100.0, "wall_ms_min": 1000.0, "wall_ms_max": 1200.0,
            "speed_x": speed if status == "ok" else None, "peak_rss_mib": 1024.0, "gpu_memory_rise_mib": 512.0, "gpu_memory_peak_mib": 600.0,
            "quality": {"kind": "wer", "value": 0.0}, "runs": 3, "notes": ""}


def summary(name, started, speeds, os_name="Linux", backends=("cpu", "cuda"), host="box", tasks=None):
    tasks = tasks or {}
    results = [row(backend, family, speed, task=tasks.get(family, "ASR")) for family, speed in speeds.items() for backend in backends]
    return {"schema": 1, "run": {"name": name, "host": host, "started": started, "finished": started, "status": "ok"},
            "machine": {"hostname": host, "os": os_name, "os_version": "Ubuntu", "cpu": "cpu", "memory_gib": 60,
                        "gpu": {"name": "RTX", "driver": "595", "cuda_toolkit": "13.3", "unified_memory": False}, "tools": {"compiler": "gcc"}},
            "pins": {"engine": {"branch": "master", "sha": "abcdef1234567890", "commit_date": "2026-11-01"},
                     "ggml": {"branch": "speech", "sha": "fedcba0987654321", "commit_date": "2026-11-01"},
                     "models": [{"name": "m.gguf", "present": True, "bytes": 1073741824, "sha256": "0123456789abcdef", "source": "registry x"}]},
            "backends": list(backends), "families": list(speeds), "results": results,
            "counts": {"expected": len(results), "ok": len(results), "failed": 0}, "build_failures": [], "problems": []}


COMPARISON = {"task": "Music", "engine": "other.cpp", "model": "M", "quantization": "Q8_0", "host": "qvac-dev-linux-x64", "device": "RTX 5090 box",
              "backend": "cuda", "qvac_rtf": "0.139 single set", "ratio": "1.91\u00d7 single set", "measured": "2026-09-14", "source": "campaign x",
              "qvac_version": "46afe7d9", "engine_version": "daf7644d", "harness": "harness text"}


class BuildArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "box").mkdir()
        self.write("box", "2026-11-01", summary("2026-11-01", "2026-11-01T17:00:00Z", {"a": 10.0, "b": 100.0}))
        self.write("box", "2026-11-08", summary("2026-11-08", "2026-11-08T17:00:00Z", {"a": 12.0, "b": 90.0}))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, host, name, data):
        (self.root / host / f"{name}.json").write_text(json.dumps(data))

    def test_latest_run_and_history_chips(self):
        summaries = build.load_summaries(self.root)
        hosts = build.runs_by_host(summaries)
        self.assertEqual(build.latest_run(hosts["box"])["run"]["name"], "2026-11-08")
        text = build.render(summaries, 5.0, [])
        self.assertIn("<title>Speech Fabric Weekly Bench</title>", text)
        self.assertIn('<section id="box">', text)
        self.assertIn('<section id="asr">', text)
        self.assertIn('<span class="chip good">+20%</span>', text)
        self.assertIn('<span class="chip warn">-10%</span>', text)
        self.assertIn("Speed on cuda", text)
        self.assertIn('id="bench-data"', text)
        self.write("box", "2026-11-15", summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 12.0}))
        text = build.render(build.load_summaries(self.root), 5.0, [])
        self.assertIn("Model b", text.split("<h3>History</h3>")[1])

    def test_task_tables_use_gpu_lanes_per_machine(self):
        (self.root / "qvac-dev-mac-arm64").mkdir()
        self.write("qvac-dev-mac-arm64", "2026-11-08", summary("2026-11-08", "2026-11-08T17:00:00Z", {"a": 50.0, "m": 2.0}, os_name="Darwin",
                                                               backends=("cpu", "metal"), host="qvac-dev-mac-arm64", tasks={"m": "Music"}))
        hosts = build.runs_by_host(build.load_summaries(self.root))
        self.assertEqual(list(hosts), ["qvac-dev-mac-arm64", "box"])
        self.assertEqual(build.machine_label("qvac-dev-linux-arm64"), "Jetson Orin Nano")
        self.assertEqual(build.lanes(hosts, build.GPU_BACKENDS), [("qvac-dev-mac-arm64", "metal"), ("box", "cuda")])
        self.assertEqual(build.task_families(hosts, "Music"), ["m"])
        text = build.task_section(hosts, "ASR", [])
        self.assertIn("Mac Studio M3 Ultra", text)
        self.assertIn('<b class="num">50x</b>', text)
        self.assertIn('<b class="num">12x</b>', text)
        self.assertIn("<summary>CPU lanes</summary>", text)
        self.assertNotIn("Model m", text)
        self.assertIn("No paired measurement", text)

    def test_comparisons_table(self):
        hosts = build.runs_by_host(build.load_summaries(self.root))
        text = build.task_section(hosts, "Music", [COMPARISON])
        self.assertIn("other.cpp", text)
        self.assertIn("RTX 5090 box · cuda", text)
        self.assertIn('<span class="num good">1.91\u00d7 single set</span>', text)
        self.assertIn("Source: campaign x (qvac 46afe7d9)", text)
        self.assertEqual(build.ratio_tone("0.87\u00d7 / 1.23\u00d7"), "warn")
        self.assertEqual(build.ratio_tone("1.00\u00d7 / 0.92\u00d7"), "")
        self.assertEqual(build.ratio_tone("n/a"), "")
        self.assertEqual(build.load_comparisons(None), [])
        rows = build.load_comparisons(build.DEFAULT_COMPARISONS)
        self.assertTrue(all(row["ratio"] and row["qvac_rtf"] for row in rows))
        self.assertEqual({row["task"] for row in rows}, {"ASR", "TTS", "Music"})

    def test_results_cells_and_failed_rows(self):
        data = summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 10.0})
        data["results"][1] = row("cuda", "a", status="timeout")
        self.assertEqual(build.cell_html(data["results"][1]), '<td class="cell bad">timeout</td>')
        self.assertIn('<b class="num">10x</b>', build.cell_html(data["results"][0]))
        self.assertIn("1,100 ms · 1.0 GiB", build.cell_html(data["results"][0]))
        self.assertEqual(build.cell_html(None), '<td class="cell muted">not run</td>')

    def test_chart_scales_logarithmically_and_lists_backends(self):
        data = summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 1.0, "b": 10.0, "c": 100.0})
        scale_max = build.chart_scale_max(data)
        self.assertEqual(scale_max, 100.0)
        self.assertAlmostEqual(build.bar_width(10.0, scale_max), build.CHART_PLOT_WIDTH / 2)
        self.assertAlmostEqual(build.bar_width(100.0, scale_max), build.CHART_PLOT_WIDTH)
        self.assertEqual(build.chart_ticks(scale_max), [1.0, 10.0, 100.0])
        chart = build.speed_chart(data)
        self.assertIn('class="bar cuda"', chart)
        self.assertIn("log scale", chart)

    def test_pins_list_sizes_in_gib(self):
        text = build.pins_list(summary("x", "", {"a": 1.0}))
        self.assertIn("<code>m.gguf</code> 1.0 GiB", text)

    def test_primary_backend_by_os(self):
        self.assertEqual(build.primary_backend(summary("x", "", {"a": 1.0})), "cuda")
        self.assertEqual(build.primary_backend(summary("x", "", {"a": 1.0}, os_name="Darwin", backends=("cpu", "metal"))), "metal")

    def test_main_writes_page(self):
        out = self.root / "page.html"
        build.main([str(self.root), "-o", str(out)])
        self.assertTrue(out.read_text().startswith("<title>"))


if __name__ == "__main__":
    unittest.main()
