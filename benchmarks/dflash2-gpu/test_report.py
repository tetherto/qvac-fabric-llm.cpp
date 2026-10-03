"""Deterministic behavioral tests of the campaign acceptance/report gates."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import make_table as table
from render_report import END, START, fill


def save(path, data):
    path.write_text(json.dumps(data))


def make_result(lane_id, arm, label, timestamp, pp: float = 100, tg: float = 10):
    device, backend = table.LANES[lane_id]
    config = {"source_sha": table.BASE_SHA if arm.endswith("base") else "a" * 40,
              "model_sha256": table.MODEL_SHA, "draft_sha256": table.DRAFT_SHA,
              "fixture_sha256": table.FIXTURE_SHA, "context": 16384, "batch": 2048, "ubatch": 512,
              "cache_k": "f16", "cache_v": "f16", "cache_k_draft": "f16", "cache_v_draft": "f16",
              "threads": 8, "threads_batch": 8, "flash_attn": "on", "offload_layers": 99,
              "parallel": 1, "greedy": True, "gpu_id": device + "-physical-gpu",
              "backend_config": {"precision": "stock", "compiler": "pinned"},
              "toolchain": "fixture compiler", "evidence": ["config.log"]}
    if arm.startswith("serial_"):
        for key in table.DRAFT_FIELDS:
            del config[key]
    if arm == "tf":
        config.update(source_sha=table.TF_SHA, model_sha256="d" * 64, draft_sha256="e" * 64, drafting_active=True)
    runtime = {"target_gpu_layers": 66, "target_layers": 66, "draft_gpu_layers": 6, "draft_layers": 6,
               "target_n_rs_seq": 7, "draft_n_rs_seq": 0, "evidence": ["config.log"]}
    if arm.startswith("serial_"):
        runtime.update(draft_gpu_layers=0, draft_layers=0, target_n_rs_seq=0)
    manifest = {"label": label, "device": device, "backend": "mlx" if arm == "tf" and backend == "metal" else backend,
                "lane": {"report_config": config, "runtime_evidence": runtime}, "draft_width": 0 if arm.startswith("serial_") else 7,
                "binary_sha256": ("b" if arm.endswith("base") else "c") * 64,
                "command": ["/fixture/llama-server", "-c", "16384"], "environment": {}, "created_unix_s": timestamp}
    runs = []
    for prompt_index in range(3):
        request = {"prompt": f"fixture-{prompt_index}", "temperature": 0, "max_tokens": 1024,
                   "ignore_eos": True, "stream": True, "cache_prompt": False}
        runs.append({"prompt_index": prompt_index, "prompt_tokens": 10000, "completion_tokens": 1024,
                     "cached_tokens": 0, "cache_status": "reported", "request": request, "ok": True,
                     "checks": {k: True for k in ("prompt_tokens", "completion_tokens", "cached_zero", "server_error_free", "positive_intervals")},
                     "text": "complete synthetic answer",
                     "text_sha256": hashlib.sha256(b"complete synthetic answer").hexdigest(),
                     "ttft_s": 10000 / pp, "decode_s": 1023 / tg,
                     "server": {"prefill_tps": pp, "decode_tps": tg}, "prefill_tps": pp, "decode_tps": tg})
    return {"label": label, "engine": "tf" if arm == "tf" else "fabric", "complete": True, "error": None,
            "runs": runs, "summary": {"all_ok": True, "server_prefill_tps": pp, "decode_tps": tg,
                                       "prefill_tps": pp, "server_decode_tps": tg},
            "provenance": {"manifest": manifest, "fixture_sha256": table.FIXTURE_SHA, "client_sha256": "0" * 64}}


def make_fixture(root, lane_ids=None, blocks=3):
    """Synthetic only: fixed rates make the expected ratios independently known."""
    (root / "config.log").write_text("Synthetic configuration evidence, not a measured GPU run.\n")
    index = {"schema_version": 1, "campaign": "SYNTHETIC report gate smoke", "baseline_sha": table.BASE_SHA,
             "candidate_sha": "a" * 40, "prompt_indices": [0, 1, 2],
             "bootstrap": {"seed": 20261003, "resamples": 1000}, "lanes": {}}
    for lane_id in table.LANES if lane_ids is None else lane_ids:
        lane = {"draft_width": 7, "base": [], "candidate": [], "blocks": []}
        index["lanes"][lane_id] = lane
        for block_index in range(blocks):
            block = {"id": f"b{block_index}", "launches": []}
            lane["blocks"].append(block)
            for position, arm in enumerate(("base", "candidate", "candidate", "base")):
                label = f"{lane_id}-{block_index}-{position}-{arm}"
                path = label + ".json"
                lane[arm].append(path)
                block["launches"].append(path)
                factor_pp, factor_tg = (2, 1.5) if arm == "candidate" else (1, 1)
                save(root / path, make_result(lane_id, arm, label, 100 + block_index * 4 + position,
                                              pp=100 * factor_pp, tg=10 * factor_tg))
    save(root / "index.json", index)
    return index


class ReportGates(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.index = make_fixture(self.root)
        self.path = self.root / "index.json"

    def mutate(self, callback, lane="metal", arm="candidate"):
        path = self.root / self.index["lanes"][lane][arm][0]
        data = table.load_json(path)
        callback(data)
        save(path, data)
        return path

    def assert_withheld(self):
        result = table.analyze(self.path)
        self.assertEqual(result["aggregate"], {})
        self.assertIn("metal", result["aggregate_error"])
        return result

    def test_all_six_lanes_equal_weight_and_source_pins(self):
        result = table.analyze(self.path)
        self.assertIsNone(result["aggregate_error"])
        self.assertAlmostEqual(result["aggregate"]["server_prefill_tps"]["median"], 2)
        self.assertAlmostEqual(result["aggregate"]["decode_tps"]["median"], 1.5)
        self.assertAlmostEqual(result["aggregate"]["decode_tps"]["ci"][0], 1.5)
        unequal = table.aggregate([[math.log(64)] * 3] + [[0] * 9] * 5, seed=7, resamples=1000)
        self.assertAlmostEqual(unequal["median"], 2)
        self.assertAlmostEqual(unequal["ci"][0], 2)

    def test_missing_lane_never_shrinks_denominator(self):
        del self.index["lanes"]["metal"]
        save(self.path, self.index)
        result = self.assert_withheld()
        self.assertEqual(len(result["lanes"]), 6)
        self.assertIn("| metal | UNMEASURED |", table.report(result))
        with self.assertRaises(ValueError):
            table.aggregate([[0] * 3] * 5, seed=1, resamples=1000)

    def test_cached_partial_and_unproven_rows_fail_despite_summary(self):
        path = self.root / self.index["lanes"]["metal"]["candidate"][0]
        original = table.load_json(path)
        changes = [lambda d: d["runs"][0].update(cached_tokens=1),
                   lambda d: d["runs"][0].update(cached_tokens=None, cache_status="unreported"),
                   lambda d: d["runs"].pop(), lambda d: d.update(complete=False),
                   lambda d: d["runs"][0].update(completion_tokens=1023),
                   lambda d: d["runs"][0]["checks"].update(server_error_free=False),
                   lambda d: d["runs"][0].pop("text"),
                   lambda d: d["runs"][0].update(text="truncated answer"),
                   lambda d: d["provenance"]["manifest"]["lane"].pop("report_config")]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(original)
                change(data)
                save(path, data)
                self.assert_withheld()

    def test_workload_and_configuration_changes_are_not_speedups(self):
        path = self.root / self.index["lanes"]["metal"]["candidate"][0]
        original = table.load_json(path)
        changes = [lambda d: d["provenance"]["manifest"]["lane"]["report_config"].update(context=8192),
                   lambda d: d["provenance"]["manifest"]["lane"]["report_config"].update(model_sha256="9" * 64),
                   lambda d: d["provenance"]["manifest"]["lane"]["report_config"].update(threads=4),
                   lambda d: d["provenance"]["manifest"]["lane"]["report_config"].update(source_sha="9" * 40),
                   lambda d: d["provenance"]["manifest"]["lane"]["report_config"]["backend_config"].update(precision="narrow"),
                   lambda d: d["runs"][0].update(prompt_index=4),
                   lambda d: d["runs"][0]["request"].update(prompt="different prompt"),
                   lambda d: d["summary"].update(server_prefill_tps=900),
                   lambda d: d["runs"][0].update(decode_s=1)]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(original)
                change(data)
                save(path, data)
                self.assert_withheld()

    def test_wrong_order_reused_launch_and_omitted_selection_rejected(self):
        original = copy.deepcopy(self.index)
        changes = [lambda lane: lane["blocks"][0]["launches"].reverse(),
                   lambda lane: lane["blocks"][0]["launches"].__setitem__(2, lane["blocks"][0]["launches"][1]),
                   lambda lane: lane["blocks"].pop()]
        # Reversing ABBA is still ABBA, but its timestamps are reversed.
        for change in changes:
            self.index = copy.deepcopy(original)
            change(self.index["lanes"]["metal"])
            save(self.path, self.index)
            self.assert_withheld()

    def test_two_blocks_have_no_confidence_claim(self):
        self.index = make_fixture(self.root, blocks=2)
        result = self.assert_withheld()
        ratio = result["lanes"]["metal"]["spec"]["metrics"]["decode_tps"]
        self.assertAlmostEqual(ratio["median"], 1.5)
        self.assertIsNone(ratio["ci"])
        self.assertEqual(table.verdict(ratio), "UNMEASURED")

    def test_abba_math_uses_paired_logs_not_best_retry(self):
        group = [{"data": {"runs": [{"decode_tps": rate * scale} for scale in (1, 10, 100)]}}
                 for rate in (10, 20, 80, 40)]
        self.assertAlmostEqual(math.exp(table.block_logs(group, "decode_tps")), 2)
        logs = [math.log(x) for x in (1, 2, 4)]
        result = table.bootstrap(logs, seed=92, resamples=1000)
        self.assertEqual(result, table.bootstrap(logs, seed=92, resamples=1000))
        self.assertAlmostEqual(result["median"], 2)
        self.assertEqual(result["ci"], [1, 4])
        (self.root / "metal-unselected-retry.json").write_text("not valid JSON")
        self.assertIsNone(table.analyze(self.path)["aggregate_error"])

    def test_one_regression_prevents_target_pass(self):
        for path in self.index["lanes"]["metal"]["candidate"]:
            data = table.load_json(self.root / path)
            for run in data["runs"]:
                run.update(decode_tps=9, decode_s=1023 / 9)
            data["summary"]["decode_tps"] = 9
            save(self.root / path, data)
        result = table.analyze(self.path)
        self.assertGreater(result["aggregate"]["decode_tps"]["median"], 1.35)
        self.assertIn("REGRESSION", table.verdict(result["lanes"]["metal"]["spec"]["metrics"]["decode_tps"]))
        self.assertIn("| decode_tps |", table.report(result))
        row = next(line for line in table.report(result).splitlines() if line.startswith("| decode_tps |"))
        self.assertIn("TARGET NOT MET", row)

    def test_serial_without_draft_fields(self):
        data = make_result("metal", "serial_base", "serial", 500)
        save(self.root / "serial.json", data)
        record = table.load_record(self.root / "serial.json", "serial_base", "metal", self.index, self.root)
        self.assertIsNone(record["error"])
        self.assertIsNone(record["proof_error"])
        with_nulls = copy.deepcopy(data)
        with_nulls["provenance"]["manifest"]["lane"]["report_config"].update({k: None for k in table.DRAFT_FIELDS})
        save(self.root / "serial-nulls.json", with_nulls)
        null_record = table.load_record(self.root / "serial-nulls.json", "serial_base", "metal", self.index, self.root)
        table.comparable([record, null_record])
        with_nulls["provenance"]["manifest"]["command"] += ["--spec-type=draft-dflash"]
        save(self.root / "serial-nulls.json", with_nulls)
        self.assertIsNotNone(table.load_record(self.root / "serial-nulls.json", "serial_base", "metal", self.index, self.root)["proof_error"])

    def test_unproven_measurements_remain_visible_without_acceptance(self):
        path = self.mutate(lambda d: d["provenance"]["manifest"]["lane"].pop("report_config"))
        result = self.assert_withheld()
        record = result["lanes"]["metal"]["records"][path.resolve()]
        self.assertIsNone(record["error"])
        self.assertIsNotNone(record["proof_error"])
        self.assertIn("200.00 / 15.00 (PROVISIONAL", table.report(result))
        self.assertIn("PROVISIONAL: 'report_config'", table.report(result))

    def test_actual_offload_and_state_proof_are_required(self):
        path = self.root / self.index["lanes"]["metal"]["candidate"][0]
        original = table.load_json(path)
        for field, value in (("target_gpu_layers", 65), ("draft_gpu_layers", 5),
                             ("target_n_rs_seq", 0), ("draft_n_rs_seq", 7)):
            data = copy.deepcopy(original)
            data["provenance"]["manifest"]["lane"]["runtime_evidence"][field] = value
            save(path, data)
            self.assert_withheld()

    def test_artifact_paths_are_provenance_not_config_invariants(self):
        def change(data):
            manifest = data["provenance"]["manifest"]
            manifest["cmake_cache_sha256"] = "9" * 64
            manifest["command"][0] = "/another/build/bin/llama-server"
            manifest["environment"] = {"CUDA_CACHE_PATH": "/another/build/cache"}
        self.mutate(change)
        self.assertIsNone(table.analyze(self.path)["aggregate_error"])

    def test_tensorfold_format_difference_and_no_vulkan_ratio(self):
        for lane_id in ("metal", "rtx5090-vulkan"):
            name = lane_id + "-tf.json"
            data = make_result(lane_id, "tf", name, 500)
            config = data["provenance"]["manifest"]["lane"]["report_config"]
            data["provenance"]["manifest"]["lane"]["report_config"] = {k: config[k] for k in table.TF_CONFIG_FIELDS}
            save(self.root / name, data)
            self.index["lanes"][lane_id]["tf"] = [name]
        save(self.path, self.index)
        result = table.analyze(self.path)
        self.assertIsNone(result["aggregate_error"])
        text = table.report(result)
        tf_section = text.split("## TensorFold comparison", 1)[1].split("## Block evidence", 1)[0]
        self.assertIn("| metal / candidate | 100.00 / 10.00 | 2.0000x | 1.5000x |", tf_section)
        self.assertIn("| rtx5090-vulkan | n/a: no TensorFold Vulkan backend | n/a | n/a |", tf_section)
        self.assertIn("engine-plus-format", text)

    def test_tensorfold_uncached_evidence_must_exist_at_pinned_revision(self):
        data = make_result("metal", "tf", "tf-evidenced", 500)
        for run in data["runs"]:
            run.update(cached_tokens=None, cache_status="evidenced",
                       cache_evidence={"uncached": True, "source_revision": table.TF_SHA, "artifact": "config.log"})
        save(self.root / "tf.json", data)
        self.assertIsNone(table.load_record(self.root / "tf.json", "tf", "metal", self.index, self.root)["error"])
        data["runs"][0]["cache_evidence"]["artifact"] = "missing.log"
        save(self.root / "tf.json", data)
        self.assertIsNotNone(table.load_record(self.root / "tf.json", "tf", "metal", self.index, self.root)["error"])

    def test_quality_is_separate_and_proxy_is_not_production_time(self):
        save(self.root / "quality.json", {"entries": [{"lane": "metal", "schedule": "width8/rs7/R1", "reference_sha": table.BASE_SHA,
                                                       "candidate_sha": "a" * 40, "artifact": "config.log"}]})
        save(self.root / "profile.json", {"entries": [{"lane": "metal", "stage": "prefill", "kind": "proxy", "artifact": "config.log",
                                                       "ops": [{"family": f"op{i}", "shape": "128x128", "dtype": "f32", "time_ms": i + 1}
                                                               for i in range(12)]}]})
        self.index.update(quality_index="quality.json", profile_index="profile.json")
        save(self.path, self.index)
        text = table.report(table.analyze(self.path))
        self.assertIn("INDEXED ONLY", text)
        self.assertIn("Proxy only: no production share", text)
        self.assertIn("| op11 |", text)
        self.assertIn("| op2 |", text)
        self.assertNotIn("| op1 |", text)
        self.assertIn("Residual CPU/sync: UNMEASURED", text)


class HistoryPreservation(unittest.TestCase):
    def test_append_replace_and_preserve_all_history(self):
        history = "# Old report\r\n<!-- table -->keep this<!-- /table -->\r\n"
        appended = fill(history, "new\n")
        self.assertTrue(appended.startswith(history))
        original = history + START + "\nold campaign\n" + END + "\r\nHistoric appendix\r\n"
        expected = history + START + "\nnew\n" + END + "\r\nHistoric appendix\r\n"
        self.assertEqual(fill(original, "new\n"), expected)
        self.assertEqual(fill(expected, "new\n"), expected)

    def test_ambiguous_markers_refuse_to_overwrite(self):
        for text in (START, END, END + START, START + END + START + END):
            with self.subTest(text=text), self.assertRaises(ValueError):
                fill(text, "new")


if __name__ == "__main__":
    unittest.main()
