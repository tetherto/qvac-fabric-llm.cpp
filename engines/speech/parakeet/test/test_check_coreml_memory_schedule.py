"""Unit tests for scripts/check-coreml-memory-schedule.py: what each check
compares, that the stage constants must keep the 1.25 margin (rounded up to
1 MiB), that a report missing a measurement fails, and that a stage seen more
than once in a phase is checked at its largest value."""
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-coreml-memory-schedule.py"
spec = importlib.util.spec_from_file_location("check_coreml_memory_schedule", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

MIB = 1 << 20
STAGES = ("diarizer-358", "diarizer-745")

# I/O: 1 x 4 x 256 fp32 in (4096 bytes), 1 x 512 fp16 out (1024 bytes).
IO_BYTES = 4096 + 1024


def template(stage, **memory):
    m = {"runtime_weight_bytes": 10 * MIB, "workspace_bytes": 2 * MIB, "compile_peak_bytes": 20 * MIB,
         "ane_compile_reserve_bytes": 300 * MIB,
         "load_peak_bytes": {"cpu_and_ane": 30 * MIB, "all": None, "cpu_and_gpu": None, "cpu_only": None},
         "margin": 1.0}
    m.update(memory)
    return {"stage": stage, "weight_bin_size": 100 * MIB, "generation_peak_bytes": 2 * MIB, "state_bytes": 0,
            "functions": [{"name": "main", "inputs": [{"name": "x", "dtype": "fp32", "shape": [1, 4, 256]}],
                           "outputs": [{"name": "y", "dtype": "fp16", "shape": [1, 512]}]}],
            "memory": m}


# Charged per stage by template(): prepare 2 + 8 + 20 = 30 MiB, load 30 MiB,
# reserve 300 MiB, resident 10 MiB + 2 x I/O + 2 MiB.
RESIDENT = 12 * MIB + 2 * IO_BYTES


def segment(stage, phase, t0, before, peak, **extra):
    s = {"stage": stage, "phase": phase, "t0": t0, "t1": t0 + 1, "before": before, "peak": peak,
         "increment": peak - before, "ane_compiler_peak": 0}
    s.update(extra)
    return s


def run_report():
    segs, t = [], 0.0
    for stage in STAGES:
        segs.append(segment(stage, "generate", t, 100 * MIB, 110 * MIB))
        segs.append(segment(stage, "compile", t + 1, 110 * MIB, 120 * MIB))
        t += 2
    for i, stage in enumerate(STAGES):
        segs.append(segment(stage, "load", t, 100 * MIB, 120 * MIB, ane_compiler_peak=(450 + 50 * i) * MIB,
                            ane_compiler_before=(280 + 20 * i) * MIB,
                            settled_after_ready=(105 + 5 * i) * MIB, resident_increment=5 * MIB))
        t += 1
    return {"mode": "coreml", "baseline_bytes": 100 * MIB, "fit_coreml_bytes": 1000 * MIB,
            "peak_increment_over_baseline": 50 * MIB, "steady_increment_over_baseline": 10 * MIB,
            "ane_compiler_before_loads": 280 * MIB, "coreml_passes": 10, "ggml_passes_after_ready": 0,
            "sampler_errors": 0, "all_ready_s": 10.0,
            "coreml_passes_by_capacity": {s.rsplit("-", 1)[-1]: 3 for s in STAGES},
            # A ggml pass during the first generate, one during the first
            # load, and Core ML passes after every load.
            "pass_intervals": [[0.5, 0.6, 0], [2 * len(STAGES) + 0.5, 2 * len(STAGES) + 0.6, 0],
                               [t + 1, t + 1.1, 1], [t + 2, t + 2.1, 1]],
            "segments": segs,
            "status": {"stages": [{"stage": s, "compute_units": "cpu_and_ane", "state": "ready"} for s in STAGES]}}


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.templates = [template(s) for s in STAGES]

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, *reports):
        args = []
        for i, t in enumerate(self.templates):
            d = self.dir / f"t{i}"
            d.mkdir(exist_ok=True)
            (d / "template.json").write_text(json.dumps(t))
            args += ["--template-dir", str(d)]
        for i, r in enumerate(reports):
            p = self.dir / f"run-{i}.json"
            p.write_text(json.dumps(r))
            args.append(str(p))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = checker.main(args)
        return rc, out.getvalue()

    def assertFails(self, report, needle):
        rc, out = self.check(report)
        self.assertEqual(rc, 1, out)
        self.assertIn(needle, out)

    def test_charged_stage_constants(self):
        self.assertEqual(checker.io_bytes(self.templates[0]), IO_BYTES)
        self.assertEqual(checker.charged(self.templates[0], "cpu_and_ane"),
                         (30 * MIB, 30 * MIB, 300 * MIB, RESIDENT))

    def test_null_constants_use_the_runtime_defaults(self):
        t = template("diarizer-358", runtime_weight_bytes=0, workspace_bytes=0, compile_peak_bytes=None,
                     ane_compile_reserve_bytes=None,
                     load_peak_bytes={"cpu_and_ane": None, "all": None, "cpu_and_gpu": None, "cpu_only": None})
        self.assertEqual(checker.charged(t, "cpu_and_ane"),
                         (2 * MIB + 8 * MIB + 50 * MIB, 25 * MIB, int(1.6 * 100 * MIB), 100 * MIB + 2 * IO_BYTES))
        self.assertEqual(checker.charged(t, "all")[1], 100 * MIB)

    def test_a_run_within_every_charge_passes(self):
        rc, out = self.check(run_report(), run_report())
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS", out)

    def test_a_stage_without_a_load_segment_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if not (s["stage"] == "diarizer-745" and s["phase"] == "load")]
        self.assertFails(r, "diarizer-745 has no load segment")

    def test_a_stage_without_a_generate_segment_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if not (s["stage"] == "diarizer-358" and s["phase"] == "generate")]
        self.assertFails(r, "diarizer-358 has no generate segment")

    def test_a_template_stage_absent_from_the_run_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if s["stage"] != "diarizer-745"]
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("diarizer-745 has no generate segment", out)
        self.assertIn("diarizer-745 has no load segment", out)

    def test_a_stage_without_a_template_fails(self):
        r = run_report()
        r["segments"].append(segment("diarizer-1125", "load", 99, 100 * MIB, 101 * MIB, resident_increment=0))
        self.assertFails(r, "no template for diarizer-1125")

    def test_repeated_loads_are_checked_at_the_largest(self):
        r = run_report()
        # A first load over the charge, then a retry within it.
        r["segments"].insert(0, segment("diarizer-358", "load", -5, 100 * MIB, 140 * MIB, resident_increment=0))
        self.assertFails(r, "diarizer-358 load +40.0 MiB (x 1.25 = 50.0 MiB) over the charged 30.0 MiB")

    def test_repeated_prepares_are_checked_at_the_largest(self):
        r = run_report()
        r["segments"][0:0] = [segment("diarizer-745", "generate", -10, 100 * MIB, 105 * MIB),
                              segment("diarizer-745", "compile", -9, 105 * MIB, 135 * MIB)]
        self.assertFails(r, "diarizer-745 prepare +35.0 MiB (x 1.25 = 44.0 MiB) over the charged 30.0 MiB")

    def test_a_prepare_spans_its_compile(self):
        r = run_report()
        r["segments"][1]["peak"] = 125 * MIB  # diarizer-358 compile: 125 - 100 (generate before)
        self.assertFails(r, "diarizer-358 prepare +25.0 MiB (x 1.25 = 32.0 MiB) over the charged 30.0 MiB")

    def test_the_compiler_is_checked_on_its_growth_not_its_size(self):
        # Peak 500 MiB over a reserve of 300 MiB, but it held 280 MiB before
        # the loads: growth 220 MiB (x 1.25 = 275 MiB) passes.
        rc, out = self.check(run_report())
        self.assertEqual(rc, 0, out)
        r = run_report()
        r["ane_compiler_before_loads"] = 150 * MIB
        self.assertFails(r, "ANECompilerService grew 350.0 MiB during the loads (x 1.25 = 438.0 MiB), "
                            "over the reserve 300.0 MiB")

    def test_the_compiler_peak_is_the_largest_over_the_loads(self):
        r = run_report()
        r["segments"][-2]["ane_compiler_peak"] = 600 * MIB  # the first load, not the last
        self.assertFails(r, "ANECompilerService grew 320.0 MiB")

    def test_a_report_without_the_compiler_baseline_fails(self):
        r = run_report()
        del r["ane_compiler_before_loads"]
        self.assertFails(r, "no ane_compiler_before_loads")

    def test_each_row_labels_the_compiler_baseline(self):
        dirty, clean = run_report(), run_report()
        dirty["ane_compiler_baseline"] = "dirty"
        clean["ane_compiler_baseline"] = "clean"
        rc, out = self.check(dirty, clean, run_report())
        self.assertEqual(rc, 0, out)
        rows = [line for line in out.splitlines() if line.startswith("| run-")]
        self.assertEqual(len(rows), 3, out)
        self.assertIn("dirty baseline", rows[0])
        self.assertIn("clean baseline", rows[1])
        self.assertIn("unknown baseline", rows[2])  # a report without the label

    def test_a_resident_step_over_the_charge_fails(self):
        r = run_report()
        r["segments"][-1]["resident_increment"] = RESIDENT + 1
        self.assertFails(r, "diarizer-745 resident step")

    def test_a_load_without_a_resident_step_fails(self):
        r = run_report()
        del r["segments"][-1]["resident_increment"]
        self.assertFails(r, "diarizer-745 load segment without resident_increment")

    def test_steady_growth_over_the_resident_parts_fails(self):
        r = run_report()
        r["steady_increment_over_baseline"] = 2 * RESIDENT + 1
        self.assertFails(r, "steady growth")
        r["steady_increment_over_baseline"] = 2 * RESIDENT
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)

    def test_a_peak_increment_over_the_fit_charge_fails(self):
        r = run_report()
        r["peak_increment_over_baseline"] = r["fit_coreml_bytes"] + 1
        self.assertFails(r, "over the fit charge")

    def test_a_run_without_a_baseline_fails(self):
        r = run_report()
        r["baseline_bytes"] = 0
        self.assertFails(r, "not a Core ML run with a ggml-only baseline")

    # ---- the margin: constants hold 1.25 x the measurement, rounded up to 1 MiB

    def test_a_load_within_the_constant_but_not_its_margin_fails(self):
        r = run_report()
        r["segments"][-1]["increment"] = 24 * MIB  # x 1.25 = 30 MiB: exactly the constant
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)
        r["segments"][-1]["increment"] = 24 * MIB + 1  # 30 MiB + 1.25 B: rounds up to 31 MiB
        self.assertFails(r, "diarizer-745 load +24.0 MiB (x 1.25 = 31.0 MiB) over the charged 30.0 MiB")

    def test_the_nemotron_load_constant_needs_its_margin(self):
        # A 110 MiB load against the 121 MiB constant: x 1.25 = 137.5, 138 MiB.
        self.templates = [template(s, load_peak_bytes={"cpu_and_ane": 121 * MIB}) for s in STAGES]
        r = run_report()
        r["segments"][-1]["peak"] = r["segments"][-1]["before"] + 110 * MIB
        r["segments"][-1]["increment"] = 110 * MIB
        self.assertFails(r, "diarizer-745 load +110.0 MiB (x 1.25 = 138.0 MiB) over the charged 121.0 MiB")

    def test_prepare_resident_and_reserve_keep_the_margin(self):
        r = run_report()
        r["segments"][1]["peak"] = 124 * MIB  # prepare 24 MiB: x 1.25 = 30, the constant
        r["segments"][-1]["resident_increment"] = 9 * MIB  # x 1.25 = 11.25 -> 12 MiB <= 12 MiB + I/O
        r["ane_compiler_before_loads"] = 260 * MIB  # growth 240 MiB: x 1.25 = 300, the reserve
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)
        r["segments"][1]["peak"] = 125 * MIB
        self.assertFails(r, "diarizer-358 prepare +25.0 MiB (x 1.25")
        r = run_report()
        r["segments"][-1]["resident_increment"] = 10 * MIB  # x 1.25 = 12.5 -> 13 MiB
        self.assertFails(r, "diarizer-745 resident step +10.0 MiB (x 1.25 = 13.0 MiB)")
        r = run_report()
        r["ane_compiler_before_loads"] = 259 * MIB  # growth 241 MiB: x 1.25 = 301.25 -> 302 MiB
        self.assertFails(r, "(x 1.25 = 302.0 MiB), over the reserve 300.0 MiB")

    def test_a_recorded_margin_above_the_rule_is_applied(self):
        self.templates = [template(s, measured={"margin": 1.5}) for s in STAGES]
        r = run_report()  # load 20 MiB: x 1.5 = 30 MiB, the constant
        r["ane_compiler_before_loads"] = 300 * MIB  # growth 200 MiB: x 1.5 = 300 MiB, the reserve
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)
        r["ane_compiler_before_loads"] = 299 * MIB
        self.assertFails(r, "(x 1.5 = 302.0 MiB), over the reserve 300.0 MiB")
        r["ane_compiler_before_loads"] = 300 * MIB
        r["segments"][-1]["increment"] = 20 * MIB + 1
        self.assertFails(r, "diarizer-745 load +20.0 MiB (x 1.5 = 31.0 MiB)")
        r["segments"][-1]["increment"] = 20 * MIB
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)
        self.templates = [template(s, measured={"margin": 1.0}) for s in STAGES]  # never below 1.25
        r["segments"][-1]["increment"] = 29 * MIB
        self.assertFails(r, "diarizer-745 load +29.0 MiB (x 1.25 = 37.0 MiB)")

    def test_whole_process_checks_stay_raw(self):
        r = run_report()
        r["peak_increment_over_baseline"] = r["fit_coreml_bytes"]
        r["steady_increment_over_baseline"] = 2 * RESIDENT
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)

    # ---- a measurement missing from the report fails

    def test_a_stage_without_a_compile_segment_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if s["phase"] != "compile"]
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("diarizer-358 has no compile segment", out)
        self.assertIn("diarizer-745 has no compile segment", out)

    def test_a_segment_without_footprint_samples_fails(self):
        r = run_report()
        r["segments"][1].update(before=0, peak=0, increment=0)
        self.assertFails(r, "diarizer-358 compile segment without footprint samples")

    def test_a_load_without_a_compiler_sample_fails(self):
        r = run_report()
        r["segments"][-1]["ane_compiler_peak"] = 0
        self.assertFails(r, "no ANECompilerService sample during a diarizer-745 load")

    def test_a_report_with_no_compiler_observation_fails(self):
        r = run_report()
        r["ane_compiler_before_loads"] = 0
        for s in r["segments"]:
            s["ane_compiler_peak"] = 0
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("no ANECompilerService sample during a diarizer-358 load", out)

    def test_sampler_errors_fail(self):
        r = run_report()
        r["sampler_errors"] = 3
        self.assertFails(r, "3 sampler errors")
        del r["sampler_errors"]
        self.assertFails(r, "no sampler_errors (an older tool's report)")

    # ---- completeness: every stage ready, served after readiness, cold

    def test_a_stage_that_ended_failed_fails(self):
        r = run_report()
        r["status"]["stages"][1]["state"] = "failed"
        r["all_ready_s"] = -1.0
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("diarizer-745 ended the run failed, not ready", out)
        self.assertIn("not every capacity became ready (all_ready_s -1.0)", out)

    def test_a_capacity_with_no_core_ml_pass_after_readiness_fails(self):
        r = run_report()
        r["coreml_passes_by_capacity"] = {"745": 6}  # 745 served the 358 passes too
        self.assertFails(r, "diarizer-358 served no Core ML pass after every capacity was ready")
        r["coreml_passes_by_capacity"] = {}
        rc, out = self.check(r)
        self.assertIn("diarizer-745 served no Core ML pass", out)
        del r["coreml_passes_by_capacity"]
        self.assertFails(r, "no coreml_passes_by_capacity")

    def test_a_compile_without_a_generate_before_it_fails(self):
        r = run_report()
        # A compile of 358 before its generate (a +512 MiB peak the prepare would not see).
        r["segments"].insert(0, segment("diarizer-358", "compile", -5, 100 * MIB, 612 * MIB))
        self.assertFails(r, "diarizer-358 has a compile segment without a generate before it")

    def test_the_compiler_is_checked_per_load(self):
        # 500 MiB before the loads, 10 MiB when the second load starts, 810 MiB
        # during it: 310 MiB over the run, but that load grew 800 MiB.
        r = run_report()
        r["ane_compiler_before_loads"] = 500 * MIB
        r["segments"][-2].update(ane_compiler_peak=520 * MIB, ane_compiler_before=500 * MIB)
        r["segments"][-1].update(ane_compiler_peak=810 * MIB, ane_compiler_before=10 * MIB)
        self.assertFails(r, "ANECompilerService grew 800.0 MiB during one load")
        del r["segments"][-1]["ane_compiler_before"]
        self.assertFails(r, "load segment without ane_compiler_before")

    def test_a_warm_run_fails(self):
        r = run_report()
        r["ane_compiler_before_loads"] = 470 * MIB  # growth 30 MiB: the compile was cached
        for s in r["segments"]:
            if s["phase"] == "load":
                s["ane_compiler_before"] = s["ane_compiler_peak"] - 20 * MIB
        self.assertFails(r, "below the 64.0 MiB of a cold run")

    def test_a_run_without_a_ggml_pass_during_a_prepare_fails(self):
        r = run_report()
        r["pass_intervals"][0] = [-1.0, -0.9, 0]  # before the first prepare
        self.assertFails(r, "no ggml pass ran during a prepare: the schedule was not overlapped")
        rc, out = self.check(r)
        self.assertNotIn("during a load or warm", out)

    def test_a_run_without_a_ggml_pass_during_a_load_fails(self):
        r = run_report()
        r["pass_intervals"][1] = [100.0, 100.1, 0]  # after every load
        self.assertFails(r, "no ggml pass ran during a load or warm: the schedule was not overlapped")
        rc, out = self.check(r)
        self.assertNotIn("during a prepare", out)

    def test_a_pass_on_core_ml_is_not_an_overlap(self):
        # Every pass on Core ML (the passes ran only after the loads finished).
        r = run_report()
        r["pass_intervals"] = [[p[0], p[1], 1] for p in r["pass_intervals"]]
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("no ggml pass ran during a prepare", out)
        self.assertIn("no ggml pass ran during a load or warm", out)

    def test_one_pass_spanning_a_prepare_and_a_load_counts_for_both(self):
        r = run_report()
        r["pass_intervals"] = [[0.5, 2 * len(STAGES) + 0.5, 0]] + r["pass_intervals"][2:]
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)

    def test_a_report_without_pass_intervals_fails(self):
        r = run_report()
        del r["pass_intervals"]
        self.assertFails(r, "no pass_intervals (an older tool's report)")

    def test_every_run_is_checked(self):
        bad = run_report()
        bad["segments"][-1]["increment"] = 31 * MIB
        rc, out = self.check(run_report(), copy.deepcopy(bad))
        self.assertEqual(rc, 1, out)
        self.assertIn("run-1.json: diarizer-745 load +31.0 MiB", out)


if __name__ == "__main__":
    unittest.main()
