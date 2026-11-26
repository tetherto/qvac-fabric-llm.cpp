#!/usr/bin/env python3
"""Write a Markdown report from a run's summary.json, optionally compared with an earlier run.

usage: report.py summary.json [--previous summary.json] [--threshold 5] [-o report.md]

Sections: status, machine, pins (engine, ggml, toolchain, every model file with its sha256 and origin), one results
table per backend, speed across backends, every metric that moved by more than the threshold since the previous run
(with whether the engine, ggml or model pins changed), and the problem list.
"""
from __future__ import annotations

import argparse
import json
import sys

STATUS_OK = "ok"
CELL_NOT_EXPECTED = "n/a"
MS_PER_SECOND = 1000.0
MIB_PER_GIB = 1024.0
BYTES_PER_MIB = 1024.0 * 1024.0
SLOW_MS_LIMIT = 10000.0
FAST_SPEED_LIMIT = 10.0
PERCENT = 100.0
SHA_SHORT = 12
DEFAULT_THRESHOLD_PERCENT = 5.0
GPU_BACKENDS = ("cuda", "vulkan", "metal")
COMPARED_METRICS = (("wall_ms_median", "wall median"), ("peak_rss_mib", "peak RSS"))


def load(path):
    with open(path) as handle:
        return json.load(handle)


def table(head, rows):
    if not rows:
        return ["- none", ""]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
    return lines + [""]


def fmt_ms(value):
    if value is None:
        return "-"
    if value < SLOW_MS_LIMIT:
        return f"{value:,.0f} ms"
    return f"{value / MS_PER_SECOND:,.1f} s"


def fmt_range(row):
    if row.get("wall_ms_median") is None:
        return "-"
    return f"{fmt_ms(row['wall_ms_median'])} ({fmt_ms(row.get('wall_ms_min'))} to {fmt_ms(row.get('wall_ms_max'))})"


def fmt_speed(value):
    if value is None:
        return "-"
    return f"{value:,.0f}x" if value >= FAST_SPEED_LIMIT else f"{value:.2f}x"


def fmt_gib(mib):
    return "-" if mib is None else f"{mib / MIB_PER_GIB:.1f} GiB"


def fmt_percent(value):
    return "-" if value is None else f"{value * PERCENT:.1f}%"


def fmt_quality(row):
    quality = row.get("quality")
    if not quality:
        return "-"
    label = "WER" if quality["kind"] == "wer" else "TTS WER"
    return f"{label} {fmt_percent(quality['value'])}"


def fmt_gpu_memory(row):
    if row["backend"] not in GPU_BACKENDS or row.get("gpu_memory_rise_mib") is None:
        return "-"
    return f"{fmt_gib(row['gpu_memory_rise_mib'])} rise (peak {fmt_gib(row['gpu_memory_peak_mib'])})"


def status_cell(row):
    if row["status"] != STATUS_OK:
        return f"**{row['status']}**"
    return f"{row['status']} (under load)" if row.get("contended") else row["status"]


def header(summary):
    run, counts = summary["run"], summary["counts"]
    return [f"# Speech weekly benchmark: {run.get('host', '?')} {run.get('name', '?')}", "",
            f"**Status: {run.get('status')}** ({counts['ok']} of {counts['expected']} cells ok). "
            f"Started {run.get('started') or '?'}, finished {run.get('finished') or '?'} (UTC).", ""]


def gpu_lines(gpu):
    if not gpu:
        return []
    if gpu.get("unified_memory"):
        return [f"- **GPU:** {gpu.get('name', '?')}, {gpu.get('gpu_cores', '?')} GPU cores, unified memory"]
    lines = [f"- **GPU:** {gpu.get('name', '?')} ({gpu.get('memory_mib', '?')} MiB, index {gpu.get('index', '?')}), "
             f"driver {gpu.get('driver', '?')}, CUDA toolkit {gpu.get('cuda_toolkit', '?')}"]
    if gpu.get("vulkan"):
        lines.append(f"- **Vulkan:** {gpu['vulkan']}")
    return lines


def machine_section(machine):
    tools = machine.get("tools", {})
    lines = ["## Machine", "",
             f"- **Host:** {machine.get('hostname', '?')}, {machine.get('os_version', '?')}, kernel {machine.get('kernel', '?')}, "
             f"{machine.get('arch', '?')}",
             f"- **CPU:** {machine.get('cpu', '?')}, {machine.get('cpu_count', '?')} logical cores, {machine.get('memory_gib', '?')} GiB RAM"]
    lines += gpu_lines(machine.get("gpu", {}))
    lines.append(f"- **Toolchain:** {tools.get('compiler', '?')}; {tools.get('cmake', '?')}; ninja {tools.get('ninja', '?')}; "
                 f"{tools.get('python', '?')}; node {tools.get('node', '?')}")
    return lines + [""]


def repo_line(label, repo):
    pin = f", pinned to {repo['pin']}" if repo.get("pin") else ""
    return f"- **{label}:** {repo.get('repo', '?')} {repo.get('branch', '?')} @ `{repo.get('sha', '?')}` ({repo.get('commit_date', '?')}{pin})"


def model_rows(models):
    return [[model["name"], fmt_gib(model["bytes"] / BYTES_PER_MIB) if model.get("present") else "missing",
             f"`{model.get('sha256', '')[:SHA_SHORT]}`" if model.get("present") else "-", model.get("source", "")]
            for model in models]


def build_line(build):
    arch = f"CUDA arch {build['cuda_arch']}, " if build.get("cuda_arch") else ""
    return f"- **Build:** {arch}GGML_NATIVE {build.get('ggml_native', '-')}"


def pins_section(pins):
    scorer = pins.get("tts_asr_scorer") or {}
    lines = ["## Pins", "", repo_line("Engine", pins.get("engine", {})), repo_line("ggml", pins.get("ggml", {})),
             build_line(pins.get("build", {})),
             f"- **Registry client:** @qvac/registry-client {pins.get('registry_client') or 'not installed in this bench home'}",
             f"- **Requantize venv:** {pins.get('venv_packages') or 'not created'}",
             f"- **TTS intelligibility scorer:** {scorer.get('repo', '-')} @ {scorer.get('revision', '-')} {scorer.get('name', '')}",
             f"- **families.json:** sha256 `{(pins.get('families_sha256') or '')[:SHA_SHORT]}`", "", "### Model files", ""]
    return lines + table(["file", "size", "sha256", "origin"], model_rows(pins.get("models", [])))


def measured_cells(row):
    if row["status"] != STATUS_OK:
        return ["-", "-", "-", "-", "-"]
    return [fmt_range(row), fmt_speed(row.get("speed_x")), fmt_gib(row.get("peak_rss_mib")), fmt_gpu_memory(row), fmt_quality(row)]


def result_row(row):
    return [row["model_label"], row["weights"], status_cell(row)] + measured_cells(row) + \
        [row.get("observed_backend") or "-", row.get("runs", 0)]


def backend_section(backend, rows):
    head = ["model", "weights", "status", "wall median (min to max)", "speed", "peak RSS", "GPU memory", "quality",
            "observed backend", "runs"]
    return [f"## Results on {backend}", "",
            "Speed is seconds of audio per second of wall clock: input audio for ASR, generated audio for TTS and music. "
            "Time-wrapped families include model load in the wall clock; native ASR and Supertonic benches time inference only.", ""] + \
        table(head, [result_row(row) for row in rows if row["backend"] == backend])


def speed_cell(row):
    if row is None:
        return CELL_NOT_EXPECTED
    return fmt_speed(row["speed_x"]) if row["status"] == STATUS_OK else row["status"]


def first_row_of(summary, family):
    return next((row for row in summary["results"] if row["family"] == family), None)


def speed_row(summary, family, by_key):
    first = first_row_of(summary, family)
    if first is None:
        return None
    return [first["model_label"], first["weights"]] + [speed_cell(by_key.get((backend, family))) for backend in summary["backends"]]


def speed_rows(summary):
    by_key = {(row["backend"], row["family"]): row for row in summary["results"]}
    rows = [speed_row(summary, family, by_key) for family in summary["families"]]
    return [row for row in rows if row is not None]


def speed_section(summary):
    return ["## Speed across backends", ""] + table(["model", "weights"] + summary["backends"], speed_rows(summary))


def change_percent(previous, current):
    if not previous or current is None:
        return None
    return PERCENT * (current - previous) / previous


def pin_changes(current, previous):
    changes = []
    for label in ("engine", "ggml"):
        if current.get("pins", {}).get(label, {}).get("sha") != previous.get("pins", {}).get(label, {}).get("sha"):
            changes.append(label)
    if model_shas(current) != model_shas(previous):
        changes.append("models")
    return changes


def model_shas(summary):
    return {model["name"]: model.get("sha256") for model in summary.get("pins", {}).get("models", [])}


def changed_rows(current, previous, threshold):
    old = {(row["backend"], row["family"]): row for row in previous["results"] if row["status"] == STATUS_OK}
    found = []
    for row in current["results"]:
        before = old.get((row["backend"], row["family"]))
        if row["status"] != STATUS_OK or before is None:
            continue
        found += metric_changes(row, before, threshold)
    return sorted(found, key=lambda item: -abs(item[0]))


def metric_changes(row, before, threshold):
    found = []
    for key, label in COMPARED_METRICS:
        delta = change_percent(before.get(key), row.get(key))
        if delta is not None and abs(delta) > threshold:
            found.append((delta, [f"{row['model_label']} ({row['weights']}) on {row['backend']}: {label}",
                                  fmt_metric(key, before.get(key)), fmt_metric(key, row.get(key)), f"{delta:+.1f}%"]))
    return found


def fmt_metric(key, value):
    return fmt_ms(value) if key == "wall_ms_median" else fmt_gib(value)


def changes_section(current, previous, threshold):
    if previous is None:
        return ["## Changes since the previous run", "", "- no previous run to compare with", ""]
    pins = ", ".join(pin_changes(current, previous)) or "none"
    lines = [f"## Changes over {threshold:g}% since {previous['run'].get('name', '?')}", "", f"Pins that changed: {pins}.", ""]
    lines += table(["metric", "previous", "current", "change"], [cells for _, cells in changed_rows(current, previous, threshold)])
    lines += ["A slower wall clock with unchanged pins usually means another user loaded the shared machine; a changed pin "
              "points at the engine, ggml or model cut named above.", ""]
    return lines


def problems_section(summary):
    found = summary.get("problems") or []
    return ["## Problems", ""] + ([f"- {problem}" for problem in found] or ["- none"]) + [""]


def render(summary, previous, threshold):
    lines = header(summary) + machine_section(summary.get("machine", {})) + pins_section(summary.get("pins", {}))
    for backend in summary["backends"]:
        lines += backend_section(backend, summary["results"])
    lines += speed_section(summary) + changes_section(summary, previous, threshold) + problems_section(summary)
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary")
    parser.add_argument("--previous")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PERCENT)
    parser.add_argument("-o", "--out")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    text = render(load(args.summary), load(args.previous) if args.previous else None, args.threshold)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
