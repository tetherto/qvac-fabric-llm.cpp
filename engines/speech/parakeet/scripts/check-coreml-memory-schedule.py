#!/usr/bin/env python3
"""Checks overlapped memory schedules against the templates' memory envelope.

Reads the JSON reports of diarizer-coreml-memory (one per run,
generated schedule) and the templates they exercised, prints one table row per
run plus the maxima over the runs, and fails when a measurement exceeds what
the envelope charges.

The per-stage constants (prepare, load, resident part, compiler reserve) must
keep the measurement margin: each is checked against the measurement x 1.25
(or the template's memory.measured.margin when larger), rounded up to 1 MiB,
the rule the constants were set by. The whole-process checks (peak increment
against the fit charge, steady growth against the summed resident parts)
compare the raw measurement, since those totals already carry the margin.

  - the run's peak footprint increment over the ggml-only run of the same
    passes, against the Core ML bytes fit_params charged for the same options;
  - each prepare (from the start of a capacity's generate phase to the end of
    its compile), against the template's prepare transient
    generation_peak_bytes + 8 MiB + compile_peak_bytes;
  - each load and warm increment, against load_peak_bytes for the resolved
    compute units;
  - each capacity's resident step (the settled footprint once it is ready,
    less the level before it), against the template's resident part
    runtime_weight_bytes + 2 x I/O + state_bytes + workspace_bytes, and the
    run's steady growth over ggml against the sum of those parts;
  - ANECompilerService's growth during the loads (its peak less its size when
    the first load started), against ane_compile_reserve_bytes (the runtime's
    cold-load memory admission). Each row labels that baseline clean (the
    service held at most 32 MiB), dirty (it already held more, so its growth
    understates a cold compile on a clean host) or unknown (no sample before
    the load, or a report without the label).

Every template stage must show a generate, a compile and a load segment in
every run, each with footprint samples, every load must have sampled
ANECompilerService, and the tool must report no sampler errors; otherwise the
checks would pass without measuring anything. A compile segment must follow a
generate of its stage. Every template stage must end the run ready (and
all_ready_s be set), and serve at least one Core ML pass after that, counted
by the capacity that served it. The compiler's growth is checked over the
whole run and per load (each load from the service's size when it started, so
a service that shrank between loads cannot hide a later compile), and a run
whose compiler grew less than 64 MiB is not a cold run and fails. A stage seen
more than once in a phase is checked at its largest value. The run must have
overlapped inference with the background work: among its pass_intervals, a
ggml pass ran during a prepare (generate or compile) and one during a load or
warm (one pass spanning both counts for both).

A null constant means the runtime's default: runtime weights = weight.bin;
compile 0.5 x weight.bin; load 0.25 x weight.bin for cpu_and_ane and 1.0 x for
the other unit classes; the compiler reserve 1.6 x weight.bin.

  check-coreml-memory-schedule.py --template-dir DIR [--template-dir DIR ...] RUN.json [RUN.json ...]
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

MIB = 1 << 20
PREPARE_FIXED = 8 * MIB
MARGIN = 1.25  # each stage constant is at least 1.25 x the largest measurement
COLD_COMPILER_FLOOR = 64 * MIB  # a cold run's ANECompilerService grows at least this much


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def dtype_bytes(dtype):
    return {"fp16": 2, "fp64": 8}.get(dtype, 4)


def io_bytes(template):
    total = 0
    for fn in template.get("functions", []):
        for side in ("inputs", "outputs"):
            for x in fn.get(side, []):
                n = dtype_bytes(x.get("dtype", "fp32"))
                for d in x.get("shape", []):
                    n *= int(d)
                total += n
    return total


def charged(template, units):
    """(prepare, load, reserve, resident) the envelope charges one stage."""
    t, m = template, template["memory"]
    weights = int(t.get("weight_bin_size", 0))
    compile_peak = m.get("compile_peak_bytes")
    compile_peak = weights // 2 if compile_peak is None else int(compile_peak)
    load = (m.get("load_peak_bytes") or {}).get(units)
    if load is None:
        load = weights // 4 if units == "cpu_and_ane" else weights
    reserve = m.get("ane_compile_reserve_bytes")
    reserve = int(1.6 * weights) if reserve is None else int(reserve)
    prepare = int(t.get("generation_peak_bytes", 0)) + PREPARE_FIXED + compile_peak
    resident = (int(m.get("runtime_weight_bytes") or weights) + 2 * io_bytes(t) + int(t.get("state_bytes", 0))
                + int(m.get("workspace_bytes") or 0))
    return prepare, int(load), reserve, resident


def mib(v):
    return f"{v / MIB:.1f}"


def margin_of(template):
    measured = template.get("memory", {}).get("measured") or {}
    return max(MARGIN, float(measured.get("margin") or MARGIN))


def with_margin(raw, margin):
    """What a constant must hold for a raw measurement: x margin, rounded up to 1 MiB."""
    return -(-int(raw * margin) // MIB) * MIB if raw > 0 else 0


def over(path, what, raw, margin, limit):
    """A failure message when raw x margin (rounded up to 1 MiB) exceeds limit, else None."""
    need = with_margin(raw, margin)
    if need <= limit:
        return None
    return (f"{path}: {what} +{mib(raw)} MiB (x {margin:g} = {mib(need)} MiB) over the charged "
            f"{mib(limit)} MiB")


def orphan_compiles(segs):
    """Compile segments with no generate of the stage before them (since the last load)."""
    orphans, generating = 0, False
    for s in sorted(segs, key=lambda s: s["t0"]):
        if s["phase"] == "generate":
            generating = True
        elif s["phase"] == "load":
            generating = False
        elif s["phase"] == "compile" and not generating:
            orphans += 1
    return orphans


def prepares(segs):
    """The increment of each prepare: a generate segment and the compile
    segments after it, up to the next generate or load."""
    out, current = [], None
    for s in sorted(segs, key=lambda s: s["t0"]):
        if s["phase"] == "generate":
            if current:
                out.append(current)
            current = {"before": s["before"], "peak": s["peak"]}
        elif s["phase"] == "compile" and current:
            current["peak"] = max(current["peak"], s["peak"])
        elif s["phase"] == "load" and current:
            out.append(current)
            current = None
    if current:
        out.append(current)
    return [max(0, p["peak"] - p["before"]) for p in out]


def overlap_failures(path, r):
    """The schedule must overlap inference with the background work: a ggml
    pass ([t0, t1, coreml] in pass_intervals) running during a prepare
    (generate or compile segment) and one during a load or warm (one pass
    spanning both counts for both)."""
    intervals = r.get("pass_intervals")
    if intervals is None:
        return [f"{path}: no pass_intervals (an older tool's report)"]
    segments = r.get("segments", [])
    out = []
    for label, phases in (("prepare", ("generate", "compile")), ("load or warm", ("load",))):
        spans = [(float(s["t0"]), float(s["t1"])) for s in segments if s["phase"] in phases]
        if not any(not int(p[2]) and any(float(p[0]) < b and a < float(p[1]) for a, b in spans) for p in intervals):
            out.append(f"{path}: no ggml pass ran during a {label}: the schedule was not overlapped")
    return out


def check_run(path, r, templates):
    """Returns (row, failures) for one report."""
    failures = []
    if r.get("mode") != "coreml" or not r.get("baseline_bytes"):
        return None, [f"{path}: not a Core ML run with a ggml-only baseline"]
    units = {s["stage"]: s.get("compute_units") for s in r.get("status", {}).get("stages", [])}
    states = {s["stage"]: s.get("state") for s in r.get("status", {}).get("stages", [])}
    if float(r.get("all_ready_s", -1)) < 0:
        failures.append(f"{path}: not every capacity became ready (all_ready_s {r.get('all_ready_s')})")
    by_capacity = r.get("coreml_passes_by_capacity")
    if by_capacity is None:
        failures.append(f"{path}: no coreml_passes_by_capacity (an older tool's report)")
    if "sampler_errors" not in r:
        failures.append(f"{path}: no sampler_errors (an older tool's report)")
    elif int(r["sampler_errors"]) != 0:
        failures.append(f"{path}: {r['sampler_errors']} sampler errors: the measurements are incomplete")
    by_stage = {}
    for s in r.get("segments", []):
        by_stage.setdefault(s["stage"], []).append(s)
        if int(s.get("before", 0)) <= 0 or int(s.get("peak", 0)) <= 0:
            failures.append(f"{path}: {s['stage']} {s['phase']} segment without footprint samples")
    for stage in sorted(set(by_stage) - set(templates)):
        failures.append(f"{path}: no template for {stage}")
    failures += overlap_failures(path, r)

    row = {"prepare": 0, "load": 0, "resident": 0, "prepare_lim": None, "load_lim": None, "resident_lim": None,
           "reserve_lim": None, "resident_sum": 0}
    for stage, template in sorted(templates.items()):
        segs = by_stage.get(stage, [])
        phases = {s["phase"] for s in segs}
        for phase in ("generate", "compile", "load"):
            if phase not in phases:
                failures.append(f"{path}: {stage} has no {phase} segment")
        if orphan_compiles(segs):
            failures.append(f"{path}: {stage} has a compile segment without a generate before it")
        if states.get(stage) != "ready":
            failures.append(f"{path}: {stage} ended the run {states.get(stage)}, not ready")
        frames = stage.rsplit("-", 1)[-1]
        if by_capacity is not None and int(by_capacity.get(frames, 0)) <= 0:
            failures.append(f"{path}: {stage} served no Core ML pass after every capacity was ready")
        margin = margin_of(template)
        p_lim, l_lim, r_lim, res_lim = charged(template, units.get(stage) or "cpu_and_ane")
        row["resident_sum"] += res_lim
        for key, lim in (("prepare_lim", p_lim), ("load_lim", l_lim), ("reserve_lim", r_lim),
                         ("resident_lim", res_lim)):
            row[key] = lim if row[key] is None else min(row[key], lim)
        inc = max(prepares(segs), default=0)
        row["prepare"] = max(row["prepare"], inc)
        failures.append(over(path, f"{stage} prepare", inc, margin, p_lim))
        loads = [s for s in segs if s["phase"] == "load"]
        inc = max((s["increment"] for s in loads), default=0)
        row["load"] = max(row["load"], inc)
        failures.append(over(path, f"{stage} load", inc, margin, l_lim))
        if any("resident_increment" not in s for s in loads):
            failures.append(f"{path}: {stage} load segment without resident_increment (an older tool's report)")
        # Every load in a cold generated schedule is a cold ANE load: the
        # compiler must have been sampled while it ran.
        if any(int(s.get("ane_compiler_peak", 0)) <= 0 for s in loads):
            failures.append(f"{path}: no ANECompilerService sample during a {stage} load")
        if any("ane_compiler_before" not in s for s in loads):
            failures.append(f"{path}: {stage} load segment without ane_compiler_before (an older tool's report)")
        load_growth = max((max(0, int(s.get("ane_compiler_peak", 0)) - int(s.get("ane_compiler_before", 0)))
                           for s in loads), default=0)
        row["ane_load_growth"] = max(row.get("ane_load_growth", 0), load_growth)
        inc = max((s.get("resident_increment", 0) for s in loads), default=0)
        row["resident"] = max(row["resident"], inc)
        failures.append(over(path, f"{stage} resident step", inc, margin, res_lim))
        row["margin"] = max(row.get("margin", MARGIN), margin)

    loads = [s for s in r.get("segments", []) if s["phase"] == "load"]
    peak = max((s["ane_compiler_peak"] for s in loads), default=0)
    if "ane_compiler_before_loads" not in r:
        failures.append(f"{path}: no ane_compiler_before_loads (an older tool's report)")
    growth = max(0, peak - int(r.get("ane_compiler_before_loads", 0)))
    row["ane_growth"], row["ane_peak"] = growth, peak
    row["ane_baseline"] = r.get("ane_compiler_baseline", "unknown")
    if growth < COLD_COMPILER_FLOOR:
        failures.append(f"{path}: ANECompilerService grew {mib(growth)} MiB during the loads, below the "
                        f"{mib(COLD_COMPILER_FLOOR)} MiB of a cold run (a warm run measures no compile)")
    load_growth = row.get("ane_load_growth", 0)
    if row["reserve_lim"] is not None:
        need = with_margin(load_growth, row.get("margin", MARGIN))
        if need > row["reserve_lim"]:
            failures.append(f"{path}: ANECompilerService grew {mib(load_growth)} MiB during one load, from its "
                            f"size when that load started (x {row.get('margin', MARGIN):g} = {mib(need)} MiB), "
                            f"over the reserve {mib(row['reserve_lim'])} MiB")
    if row["reserve_lim"] is not None:
        need = with_margin(growth, row.get("margin", MARGIN))
        if need > row["reserve_lim"]:
            failures.append(f"{path}: ANECompilerService grew {mib(growth)} MiB during the loads "
                            f"(x {row.get('margin', MARGIN):g} = {mib(need)} MiB), over the reserve "
                            f"{mib(row['reserve_lim'])} MiB")
    steady = int(r.get("steady_increment_over_baseline", 0))
    row["steady"] = steady
    if steady > row["resident_sum"]:
        failures.append(f"{path}: steady growth +{mib(steady)} MiB over the charged resident parts "
                        f"{mib(row['resident_sum'])} MiB")
    row["increment"], row["envelope"] = r["peak_increment_over_baseline"], r["fit_coreml_bytes"]
    if row["increment"] > row["envelope"]:
        failures.append(f"{path}: peak increment +{mib(row['increment'])} MiB over the fit charge "
                        f"{mib(row['envelope'])} MiB")
    if r.get("ggml_passes_after_ready", 0) and r.get("coreml_passes", 0) == 0:
        failures.append(f"{path}: no pass ran on Core ML")
    row["units"] = ",".join(sorted({u for u in units.values() if u}))
    return row, [f for f in failures if f]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template-dir", action="append", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)

    templates = {}
    for d in a.template_dir:
        t = json.loads((Path(d) / "template.json").read_text())
        templates[t["stage"]] = t
    host = f"{sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}, macOS {sh(['sw_vers', '-productVersion'])}"
    if platform.system() != "Darwin":
        host = platform.platform()

    failures = []
    rows = []
    print(f"host: {host}")
    print("| Run | Units | Peak increment (MiB) | Fit charge (MiB) | Largest prepare (MiB) | Charged prepare (MiB) "
          "| Largest load+warm (MiB) | Charged load (MiB) | Largest resident step (MiB) | Charged resident (MiB) "
          "| Steady growth (MiB) | ANECompilerService growth (MiB) | Reserve (MiB) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in a.runs:
        row, f = check_run(path, json.loads(Path(path).read_text()), templates)
        failures += f
        if row is None:
            continue
        rows.append(row)
        print(f"| {Path(path).name} | {row['units']} | {mib(row['increment'])} | {mib(row['envelope'])} | "
              f"{mib(row['prepare'])} | {mib(row['prepare_lim'] or 0)} | {mib(row['load'])} | "
              f"{mib(row['load_lim'] or 0)} | {mib(row['resident'])} | {mib(row['resident_lim'] or 0)} | "
              f"{mib(row['steady'])} | {mib(row['ane_growth'])} (peak {mib(row['ane_peak'])}, "
              f"{row['ane_baseline']} baseline) | "
              f"{mib(row['reserve_lim'] or 0)} |")
    if rows:
        worst = {k: max(r[k] for r in rows) for k in ("increment", "prepare", "load", "resident", "steady",
                                                       "ane_growth")}
        print(f"maxima: peak increment {mib(worst['increment'])} MiB, prepare {mib(worst['prepare'])} MiB, "
              f"load+warm {mib(worst['load'])} MiB, resident step {mib(worst['resident'])} MiB, "
              f"steady growth {mib(worst['steady'])} MiB, ANECompilerService growth {mib(worst['ane_growth'])} MiB")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
