#!/usr/bin/env python3
"""Build the weekly results page from a folder of summary.json files.

usage: build-artifact.py <summaries dir> -o page.html [--threshold 5] [--comparisons comparisons.json]

The folder holds one file per run, <host>/<run>.json, as written by summarize.py and collected by the speech-bench
skill. The page is organized by task (ASR, TTS, TTS cloning, Music): for every task one table with a row per model
and quantization and a column per GPU lane (machine and backend) of the newest run of each machine, the CPU lanes
in a collapsed table, and the paired measurements against other engines from comparisons.json. Below that, one
block per machine with its history across the stored runs (moves over the threshold marked), pins and problems. It
is a self-contained HTML body (no document skeleton) in the shape the Artifact tool publishes.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import math
import pathlib
import re

STATUS_OK = "ok"
PERCENT = 100.0
MIB_PER_GIB = 1024.0
BYTES_PER_MIB = 1024.0 * 1024.0
MS_PER_SECOND = 1000.0
SLOW_MS_LIMIT = 10000.0
FAST_SPEED_LIMIT = 10.0
DEFAULT_THRESHOLD_PERCENT = 5.0
SHA_SHORT = 10
GPU_BACKENDS = ("cuda", "vulkan", "metal")
BACKEND_ORDER = ("cpu", "cuda", "vulkan", "metal")
PRIMARY_BACKEND = {"Linux": "cuda", "Darwin": "metal"}
TASK_ORDER = ("ASR", "TTS", "TTS cloning", "Music")
MACHINE_LABELS = {"spark-d9ca": "DGX Spark", "qvac-dev-linux-x64": "RTX 5090", "qvac-dev-strix-1": "Strix Halo", "qvac-dev-linux-arm64": "Jetson Orin Nano",
                  "qvac-dev-mac-arm64": "Mac Studio M3 Ultra", "mac-mini-m4-gpu": "Mac mini M4"}
CHART_MIN_SPEED = 1.0
CHART_MAX_SPEED_FLOOR = 10.0
CHART_TICKS = (1.0, 10.0, 100.0, 1000.0)
CHART_LABEL_WIDTH = 230
CHART_PLOT_WIDTH = 520
CHART_ROW_HEIGHT = 18
CHART_GROUP_GAP = 10
CHART_TOP = 24
CHART_BAR_HEIGHT = 12
PAGE_TITLE = "Speech Fabric Weekly Bench"
DEFAULT_COMPARISONS = pathlib.Path(__file__).resolve().parent / "comparisons.json"


def load_summaries(folder):
    found = []
    for path in sorted(pathlib.Path(folder).glob("*/*.json")):
        with path.open() as handle:
            summary = json.load(handle)
        summary["_host"] = path.parent.name
        found.append(summary)
    return found


def load_comparisons(path):
    if not path or not pathlib.Path(path).is_file():
        return []
    with open(path) as handle:
        return json.load(handle).get("rows", [])


def started_at(summary):
    return summary.get("run", {}).get("started") or ""


def host_rank(host):
    known = list(MACHINE_LABELS)
    return (known.index(host), host) if host in known else (len(known), host)


def runs_by_host(summaries):
    hosts = {}
    for summary in summaries:
        hosts.setdefault(summary["_host"], []).append(summary)
    return {host: sorted(hosts[host], key=started_at) for host in sorted(hosts, key=host_rank)}


def latest_run(runs):
    return runs[-1]


def machine_label(host):
    return MACHINE_LABELS.get(host, host)


def ordered_backends(summary):
    return [backend for backend in BACKEND_ORDER if backend in summary.get("backends", [])]


def family_rows(summary):
    rows = {}
    for row in summary.get("results", []):
        rows.setdefault(row["family"], {})[row["backend"]] = row
    return rows


def family_meta(summary):
    meta = {}
    for row in summary.get("results", []):
        meta.setdefault(row["family"], (row.get("model_label", row["family"]), row.get("weights", ""), row.get("task", "")))
    return meta


def fmt_ms(value):
    if value is None:
        return "-"
    if value < SLOW_MS_LIMIT:
        return f"{value:,.0f} ms"
    return f"{value / MS_PER_SECOND:,.1f} s"


def fmt_speed(value):
    if value is None:
        return "-"
    return f"{value:,.0f}x" if value >= FAST_SPEED_LIMIT else f"{value:.2f}x"


def fmt_gib(mib):
    return "-" if mib is None else f"{mib / MIB_PER_GIB:.1f} GiB"


def fmt_quality(row):
    quality = row.get("quality")
    if not quality:
        return ""
    label = "WER" if quality["kind"] == "wer" else "TTS WER"
    return f"{label} {quality['value'] * PERCENT:.1f}%"


def esc(text):
    return html.escape(str(text))


def change_percent(previous, current):
    if not previous or current is None:
        return None
    return PERCENT * (current - previous) / previous


def change_chip(delta, threshold, higher_is_better):
    if delta is None or abs(delta) <= threshold:
        return ""
    better = (delta > 0) == higher_is_better
    tone = "good" if better else "warn"
    return f'<span class="chip {tone}">{delta:+.0f}%</span>'


def cell_html(row):
    if row is None:
        return '<td class="cell muted">not run</td>'
    if row.get("status") != STATUS_OK:
        return f'<td class="cell bad">{esc(row["status"])}</td>'
    memory = fmt_gib(row.get("peak_rss_mib"))
    if row["backend"] in GPU_BACKENDS and row.get("gpu_memory_rise_mib") is not None:
        memory += f" + {fmt_gib(row['gpu_memory_rise_mib'])} GPU"
    return (f'<td class="cell"><b class="num">{fmt_speed(row.get("speed_x"))}</b>'
            f'<span class="sub">{fmt_ms(row.get("wall_ms_median"))} · {memory}</span>'
            f'<span class="sub">{esc(fmt_quality(row))}</span></td>')


def lanes(hosts, backends):
    found = []
    for host, runs in hosts.items():
        latest = latest_run(runs)
        found += [(host, backend) for backend in ordered_backends(latest) if backend in backends]
    return found


def lane_header(host, backend):
    return f'<th>{esc(machine_label(host))}<span class="sub">{esc(backend)}</span></th>'


def task_families(hosts, task):
    found = []
    for runs in hosts.values():
        latest = latest_run(runs)
        meta = family_meta(latest)
        found += [family for family in latest.get("families", []) if meta.get(family, ("", "", ""))[2] == task and family not in found]
    return found


def task_meta(hosts):
    meta = {}
    for runs in hosts.values():
        meta.update(family_meta(latest_run(runs)))
    return meta


def lane_table(hosts, families, meta, lane_list, css):
    if not lane_list or not families:
        return '<p class="note">no lane of this kind in the stored runs</p>'
    latest = {host: family_rows(latest_run(runs)) for host, runs in hosts.items()}
    head = "".join(lane_header(host, backend) for host, backend in lane_list)
    body = []
    for family in families:
        label, weights, _ = meta.get(family, (family, "", ""))
        cells = "".join(cell_html(latest[host].get(family, {}).get(backend)) for host, backend in lane_list)
        body.append(f'<tr><th scope="row"><span class="model">{esc(label)}</span><span class="sub">{esc(weights)}</span></th>{cells}</tr>')
    return f'<div class="scroll"><table class="{css}"><thead><tr><th>model</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


RATIO_TIE_LOW = 0.95
RATIO_TIE_HIGH = 1.05


def ratio_tone(ratio_text):
    match = re.match(r"\s*([0-9.]+)", ratio_text or "")
    if not match:
        return ""
    first = float(match.group(1))
    if first > RATIO_TIE_HIGH:
        return "good"
    return "warn" if first < RATIO_TIE_LOW else ""


def comparison_row(row):
    lane = f"{row.get('device') or machine_label(row.get('host', '?'))} · {row.get('backend', '?')}"
    return [esc(row.get("engine", "?")), f'{esc(row.get("model", ""))}<span class="sub">{esc(row.get("quantization", ""))}</span>', esc(lane),
            f'<span class="num">{esc(row.get("qvac_rtf", "-"))}</span>',
            f'<span class="num {ratio_tone(row.get("ratio", ""))}">{esc(row.get("ratio", "-"))}</span>',
            esc(row.get("measured", "")), f'<span class="sub">{esc(row.get("engine_version", ""))}</span>']


def comparison_sort_key(row):
    return (row.get("model", ""), row.get("quantization", ""), host_rank(row.get("host", "")), row.get("backend", ""), row.get("engine", ""))


def comparisons_table(rows):
    if not rows:
        return ('<p class="note">No paired measurement against another engine is recorded for this task yet. Add rows to '
                '<code>comparisons.json</code> from a campaign that measured both engines the same way.</p>')
    sources = sorted({f"{row.get('source', '')} (qvac {row.get('qvac_version', '?')})" for row in rows})
    head = "".join(f"<th>{column}</th>" for column in ("engine", "model", "lane", "qvac RTF short / long", "competitor time \u00f7 qvac time", "measured", "competitor build"))
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in comparison_row(row)) + "</tr>" for row in sorted(rows, key=comparison_sort_key))
    notes = "".join(f'<p class="note">Source: {esc(text)}</p>' for text in sources)
    return (f'{notes}<div class="scroll"><table class="compare"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            '<p class="note">A ratio above 1\u00d7 means qvac was faster than the competitor on that lane (end-to-end process wall, '
            'short / long input). These campaign rows are dated and keep their own prompts, so compare engines within a row, '
            'not against the weekly table.</p>')


def task_section(hosts, task, comparisons):
    families = task_families(hosts, task)
    meta = task_meta(hosts)
    gpu = lane_table(hosts, families, meta, lanes(hosts, GPU_BACKENDS), "results")
    cpu = lane_table(hosts, families, meta, lanes(hosts, ("cpu",)), "results")
    paired = [row for row in comparisons if row.get("task") == task]
    return (f'<section id="{esc(task.lower().replace(" ", "-"))}"><h2>{esc(task)}</h2>'
            f'<h3>GPU lanes, newest run per machine</h3>{gpu}'
            f'<details><summary>CPU lanes</summary>{cpu}</details>'
            f'<h3>Other engines</h3>{comparisons_table(paired)}</section>')


def chart_scale_max(summary):
    speeds = [row["speed_x"] for row in summary.get("results", []) if row.get("status") == STATUS_OK and row.get("speed_x")]
    return max([CHART_MAX_SPEED_FLOOR] + speeds)


def bar_width(speed, scale_max):
    if not speed or speed <= CHART_MIN_SPEED:
        return 2.0
    return CHART_PLOT_WIDTH * math.log10(speed / CHART_MIN_SPEED) / math.log10(scale_max / CHART_MIN_SPEED)


def chart_ticks(scale_max):
    return [tick for tick in CHART_TICKS if tick <= scale_max]


def chart_bar(row, y, scale_max, backend):
    if row is None or row.get("status") != STATUS_OK or not row.get("speed_x"):
        return (f'<text x="{CHART_LABEL_WIDTH + 4}" y="{y + CHART_BAR_HEIGHT - 2}" class="chart-missing">'
                f'{esc(backend)}: {esc((row or {}).get("status", "not run"))}</text>')
    width = bar_width(row["speed_x"], scale_max)
    return (f'<rect class="bar {esc(backend)}" x="{CHART_LABEL_WIDTH}" y="{y}" width="{width:.1f}" height="{CHART_BAR_HEIGHT}" rx="2">'
            f'<title>{esc(backend)}: {fmt_speed(row["speed_x"])}, {fmt_ms(row.get("wall_ms_median"))}</title></rect>')


def chart_group(meta, rows, backends, y, scale_max):
    label, weights, _ = meta
    parts = [f'<text x="{CHART_LABEL_WIDTH - 8}" y="{y + CHART_BAR_HEIGHT}" text-anchor="end" class="chart-label">{esc(label)}</text>',
             f'<text x="{CHART_LABEL_WIDTH - 8}" y="{y + CHART_BAR_HEIGHT + 12}" text-anchor="end" class="chart-sub">{esc(weights[:40])}</text>']
    for index, backend in enumerate(backends):
        parts.append(chart_bar(rows.get(backend), y + index * CHART_ROW_HEIGHT, scale_max, backend))
    return "".join(parts)


def chart_axis(scale_max, height):
    parts = []
    for tick in chart_ticks(scale_max):
        x = CHART_LABEL_WIDTH + bar_width(tick, scale_max)
        parts.append(f'<line x1="{x:.1f}" y1="{CHART_TOP - 6}" x2="{x:.1f}" y2="{height}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{CHART_TOP - 10}" text-anchor="middle" class="chart-sub">{tick:g}x</text>')
    return "".join(parts)


def speed_chart(summary):
    backends = ordered_backends(summary)
    rows = family_rows(summary)
    meta = family_meta(summary)
    scale_max = chart_scale_max(summary)
    group_height = len(backends) * CHART_ROW_HEIGHT + CHART_GROUP_GAP
    families = summary.get("families", [])
    height = CHART_TOP + len(families) * group_height
    groups = [chart_group(meta.get(family, (family, "", "")), rows.get(family, {}), backends, CHART_TOP + index * group_height, scale_max)
              for index, family in enumerate(families)]
    legend = "".join(f'<span class="legend-item"><i class="swatch {esc(b)}"></i>{esc(b)}</span>' for b in backends)
    width = CHART_LABEL_WIDTH + CHART_PLOT_WIDTH + 40
    return (f'<div class="legend">{legend}<span class="legend-note">seconds of audio per second of wall clock, log scale</span></div>'
            f'<div class="scroll"><svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="speed per family and backend">{chart_axis(scale_max, height)}{"".join(groups)}</svg></div>')


def primary_backend(summary):
    backends = ordered_backends(summary)
    return PRIMARY_BACKEND.get(summary.get("machine", {}).get("os", ""), backends[-1] if backends else "")


def history_cells(runs, family, backend, threshold):
    cells, previous = [], None
    for summary in runs:
        row = family_rows(summary).get(family, {}).get(backend)
        if row is None or row.get("status") != STATUS_OK:
            cells.append(f'<td class="muted">{esc((row or {}).get("status", "-"))}</td>')
            continue
        speed = row.get("speed_x")
        cells.append(f'<td><span class="num">{fmt_speed(speed)}</span>{change_chip(change_percent(previous, speed), threshold, True)}</td>')
        previous = speed
    return "".join(cells)


def family_meta_across(runs):
    meta = {}
    for summary in runs:
        meta.update(family_meta(summary))
    return meta


def history_table(runs, threshold):
    latest = latest_run(runs)
    backend = primary_backend(latest)
    meta = family_meta_across(runs)
    families = []
    for summary in runs:
        families += [family for family in summary.get("families", []) if family not in families]
    head = "".join(f'<th>{esc(summary["run"].get("name", "?"))}</th>' for summary in runs)
    body = []
    for family in families:
        label, weights, _ = meta.get(family, (family, "", ""))
        body.append(f'<tr><th scope="row"><span class="model">{esc(label)}</span><span class="sub">{esc(weights)}</span></th>'
                    f'{history_cells(runs, family, backend, threshold)}</tr>')
    return (f'<p class="note">Speed on {esc(backend)}; a chip marks a move over {threshold:g}% against the previous stored run.</p>'
            f'<div class="scroll"><table class="history"><thead><tr><th>model</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def status_tile(summary):
    run, counts, pins = summary.get("run", {}), summary.get("counts", {}), summary.get("pins", {})
    status = run.get("status", "?")
    tone = {"ok": "good", "partial": "warn"}.get(status, "bad")
    engine = (pins.get("engine") or {}).get("sha", "")[:SHA_SHORT]
    ggml = (pins.get("ggml") or {}).get("sha", "")[:SHA_SHORT]
    return (f'<div class="tile"><span class="tile-host">{esc(machine_label(summary["_host"]))}<span class="sub">{esc(summary["_host"])}</span></span>'
            f'<span class="tile-status {tone}">{esc(status)}</span>'
            f'<span class="tile-big num">{counts.get("ok", 0)}<small>/{counts.get("expected", 0)} cells ok</small></span>'
            f'<span class="sub">{esc(run.get("name", "?"))} · engine <code>{esc(engine)}</code> · ggml <code>{esc(ggml)}</code></span></div>')


def machine_line(summary):
    machine = summary.get("machine", {})
    gpu = machine.get("gpu", {})
    gpu_text = f"{gpu.get('name', '?')}, {gpu.get('gpu_cores', '?')} GPU cores" if gpu.get("unified_memory") else \
        f"{gpu.get('name', '?')}, driver {gpu.get('driver', '?')}, CUDA {gpu.get('cuda_toolkit', '?')}"
    return (f"{machine.get('os_version', '?')} · {machine.get('cpu', '?')} · {machine.get('memory_gib', '?')} GiB · {gpu_text} · "
            f"{machine.get('tools', {}).get('compiler', '?')}")


def pins_list(summary):
    pins = summary.get("pins", {})
    items = []
    for label in ("engine", "ggml"):
        repo = pins.get(label) or {}
        items.append(f"<li><b>{label}</b> {esc(repo.get('branch', '?'))} <code>{esc(repo.get('sha', '?'))}</code> ({esc(repo.get('commit_date', '?'))})</li>")
    for model in pins.get("models", []):
        size = fmt_gib(model["bytes"] / BYTES_PER_MIB) if model.get("present") else "missing"
        items.append(f"<li><code>{esc(model['name'])}</code> {size} <code>{esc((model.get('sha256') or '')[:SHA_SHORT])}</code> "
                     f"<span class=\"sub\">{esc(model.get('source', ''))}</span></li>")
    return "<ul class=\"pins\">" + "".join(items) + "</ul>"


def problems_list(summary):
    problems = summary.get("problems") or []
    if not problems:
        return "<p class=\"note\">none</p>"
    return "<ul>" + "".join(f"<li>{esc(problem)}</li>" for problem in problems) + "</ul>"


def machine_section(host, runs, threshold):
    latest = latest_run(runs)
    run = latest.get("run", {})
    return (f'<section id="{esc(host)}"><h2>{esc(machine_label(host))} <span class="sub">{esc(host)}</span></h2>'
            f'<p class="note">{esc(machine_line(latest))}</p>'
            f'<p class="note">Newest run <b>{esc(run.get("name", "?"))}</b>, {esc(run.get("started", "?"))} to {esc(run.get("finished", "?"))} UTC; '
            f'{len(runs)} stored run{"s" if len(runs) != 1 else ""}.</p>'
            f'<h3>Speed</h3>{speed_chart(latest)}<h3>History</h3>{history_table(runs, threshold)}'
            f'<details><summary>Pins</summary>{pins_list(latest)}</details>'
            f'<h3>Problems</h3>{problems_list(latest)}</section>')


def data_script(hosts):
    compact = {host: [{"run": summary["run"], "results": summary.get("results", []), "pins": {k: v for k, v in summary.get("pins", {}).items() if k in ("engine", "ggml")}}
                      for summary in runs] for host, runs in hosts.items()}
    return '<script type="application/json" id="bench-data">' + json.dumps(compact).replace("</", "<\\/") + "</script>"


STYLE = """
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* layout: one measured column; summary tiles, then one section per task with lane columns, then one block per machine */
:root { --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2420; --fg-2: #5a645e; --line: #d9ded9; --accent: #1f6f78; --good: #1b7f4a; --warn: #b86a00; --bad: #b3261e;
  --cpu: #2a78d6; --cuda: #eb6834; --vulkan: #1baf7a; --metal: #eda100;
  --display: "Barlow Semi Condensed", "Arial Narrow", sans-serif; --body: "Source Sans 3", "Segoe UI", sans-serif; --mono: "JetBrains Mono", Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141817; --surface: #1d2321; --fg: #e8ebe8; --fg-2: #a7b0aa; --line: #333b38; --accent: #6cc3cc; --good: #52c27a; --warn: #e3a24a; --bad: #ef7a73;
  --cpu: #3987e5; --cuda: #d95926; --vulkan: #199e70; --metal: #c98500; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #141817; --surface: #1d2321; --fg: #e8ebe8; --fg-2: #a7b0aa; --line: #333b38; --accent: #6cc3cc; --good: #52c27a; --warn: #e3a24a; --bad: #ef7a73;
  --cpu: #3987e5; --cuda: #d95926; --vulkan: #199e70; --metal: #c98500; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.45; margin: 0;
  padding-block: 24px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1240px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.1; }
h1 { font-size: 2.4rem; } h2 { font-size: 1.7rem; color: var(--accent); margin-top: 8px; } h2 .sub { display: inline; font-family: var(--mono); font-size: .8rem; margin-left: 8px; }
h3 { font-size: 1.15rem; margin-top: 18px; text-transform: uppercase; letter-spacing: .06em; color: var(--fg-2); }
p { margin: 0; max-width: 72ch; } .note { color: var(--fg-2); font-size: .95rem; }
header { display: grid; gap: 10px; } nav.jump { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: .95rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 4px; }
.tile-host { font-weight: 600; } .tile-host .sub { font-family: var(--mono); }
.tile-status { font-family: var(--display); font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.tile-big { font-size: 1.8rem; font-weight: 600; } .tile-big small { font-size: .9rem; color: var(--fg-2); font-weight: 400; margin-left: 4px; }
.good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); } .muted { color: var(--fg-2); }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.sub { display: block; color: var(--fg-2); font-size: .82rem; } .model { display: block; font-weight: 600; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; } th, td { text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid var(--line); }
thead th { font-family: var(--display); font-size: .95rem; text-transform: uppercase; letter-spacing: .05em; color: var(--fg-2); }
thead th .sub { text-transform: none; letter-spacing: 0; font-family: var(--mono); }
tbody th { font-weight: 400; min-width: 180px; } td.cell { min-width: 150px; } td.bad { color: var(--bad); font-weight: 600; }
.chip { display: inline-block; margin-left: 6px; padding: 0 6px; border-radius: 999px; font-size: .75rem; font-family: var(--mono); border: 1px solid currentColor; }
.chip.good { color: var(--good); } .chip.warn { color: var(--warn); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: .85rem; color: var(--fg-2); margin: 8px 0; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; } .legend-note { margin-left: auto; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
.swatch.cpu, .bar.cpu { background: var(--cpu); fill: var(--cpu); } .swatch.cuda, .bar.cuda { background: var(--cuda); fill: var(--cuda); }
.swatch.vulkan, .bar.vulkan { background: var(--vulkan); fill: var(--vulkan); } .swatch.metal, .bar.metal { background: var(--metal); fill: var(--metal); }
svg { font-family: var(--body); } .chart-label { font-size: 12px; fill: var(--fg); } .chart-sub { font-size: 10px; fill: var(--fg-2); }
.chart-missing { font-size: 10px; fill: var(--fg-2); } .grid { stroke: var(--line); stroke-width: 1; }
details { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 16px; } summary { cursor: pointer; font-family: var(--display); color: var(--fg-2); text-transform: uppercase; letter-spacing: .06em; }
ul.pins { font-size: .85rem; padding-left: 18px; } ul.pins li { margin: 3px 0; } code { font-family: var(--mono); font-size: .85em; }
section { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; min-width: 0; }
a { color: var(--accent); } :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""


def jump_links(hosts):
    tasks = "".join(f'<a href="#{task.lower().replace(" ", "-")}">{esc(task)}</a>' for task in TASK_ORDER)
    machines = "".join(f'<a href="#{esc(host)}">{esc(machine_label(host))}</a>' for host in hosts)
    return f'<nav class="jump">{tasks}{machines}<a href="#method">How to read it</a></nav>'


def page_header(hosts, built_at):
    tiles = "".join(status_tile(latest_run(runs)) for runs in hosts.values())
    return (f'<header><h1>{PAGE_TITLE}</h1>'
            f'<p>Every Sunday from 11:00 CET, cron on the shared Tailscale machines benchmarks the qvac-fabric-speech.cpp engines on the '
            f'current engine and ggml heads: Parakeet TDT and Unified, TranscribePsy, Nemotron 3.5 ASR, Supertonic 3, the Audio8 LM and codec '
            f'grid with voice cloning, MOSS-TTS v1.5 F16 and Q8_0 with cloning, and the ACE-Step 1.5 Turbo, SFT and Base DiTs. One table per '
            f'task, one column per machine and backend; the fabric-llm engine bench owns Saturday on the same machines. Rebuilt from each '
            f'machine\'s <code>summary.json</code>; last build {esc(built_at)} UTC.</p>{jump_links(hosts)}'
            f'<div class="tiles">{tiles}</div></header>')


METHOD = """<section id="method"><h2>How to read it</h2>
<p>Speed is seconds of audio per second of wall clock: the 11 s JFK clip for ASR, the generated audio for TTS, a 30 s render for music.
ASR and Supertonic rows time inference only through the engines' own bench modes; Audio8, MOSS-TTS and ACE-Step rows are the whole process
including model load, so their speed is a throughput figure. Memory is the peak RSS of the process and, on NVIDIA backends, the rise of
<code>nvidia-smi</code> memory in use over the run (system memory rise on unified-memory hosts). WER is scored against the checked-in JFK
transcript; TTS WER transcribes the last generated clip with pinned Whisper tiny on CPU, so it includes the scorer's own errors. A slower
cell with unchanged pins usually means another user loaded the machine during the run; a changed pin names the engine, ggml or model cut
to look at. Two rows differ from the desktop target set because the registry has no such cut: ACE-Step Base runs at Q4_K_M, and MOSS-TTS
Q8_0 is the F16 backbone with its transformer blocks requantized on the machine. The "Other engines" tables are paired harness runs
recorded in <code>comparisons.json</code> (currently the October 2026 QVAC speech engines campaign); they compare engines under one
protocol and are not on the weekly clock.</p></section>"""


def render(summaries, threshold, comparisons):
    hosts = runs_by_host(summaries)
    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    tasks = "".join(task_section(hosts, task, comparisons) for task in TASK_ORDER)
    machines = "".join(machine_section(host, runs, threshold) for host, runs in hosts.items())
    return f"<title>{PAGE_TITLE}</title>{STYLE}<main>{page_header(hosts, built_at)}{tasks}{machines}{METHOD}</main>{data_script(hosts)}\n"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PERCENT)
    parser.add_argument("--comparisons", default=str(DEFAULT_COMPARISONS))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summaries = load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"no summary.json files under {args.summaries}")
    pathlib.Path(args.out).write_text(render(summaries, args.threshold, load_comparisons(args.comparisons)))
    print(f"wrote {args.out} from {len(summaries)} runs on {len(runs_by_host(summaries))} hosts")


if __name__ == "__main__":
    main()
