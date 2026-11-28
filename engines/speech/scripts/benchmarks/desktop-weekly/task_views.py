#!/usr/bin/env python3
"""Render the task pages of the Speech Bench hub from the rows build-dashboard-data.py writes per machine.

Every machine gets three pages (TTS and voice cloning, ASR, music generation) in the layout of the approved Speech Bench
report: one section per model with a horizontal speed chart per backend and input, the engines in their colors, a
details panel per chart (exact values, precision per row, the competitor-over-QVAC time ratio, notes and sources) and the
cells that were not measured with the reason. build-landing.py embeds the pages in the hub as views; each one is reached
at the hub's link plus #<machine>-<task>, for example #dgx-spark-tts.
"""
from __future__ import annotations

import html
import math
import re

WEEKLY = "weekly"
REPORT = "report"
CAMPAIGN = "campaign"
SET_ORDER = (WEEKLY, REPORT, CAMPAIGN)
COPIED_SETS = (REPORT, CAMPAIGN)
TASK_PAGES = (("tts", "TTS and voice cloning", ("TTS", "TTS cloning")), ("asr", "ASR", ("ASR",)), ("music", "Music generation", ("Music",)))
BACKEND_LABELS = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal", "coreml": "Core ML"}
BACKEND_ORDER = tuple(BACKEND_LABELS)
ENGINE_CLASSES = {"qvac-fabric-speech.cpp": "e-qvac", "QVAC bundled whisper.cpp": "e-qvac", "audio.cpp": "e-audiocpp",
                  "LocalAI / parakeet.cpp": "e-localai", "transcribe.cpp": "e-transcribe", "NeMo-Speech.cpp": "e-nemo",
                  "whisper.cpp (upstream)": "e-whispercpp", "cosyvoice.cpp": "e-cosy", "TTS.cpp": "e-ttscpp", "acestep.cpp": "e-acestep",
                  "minimaxmusic.cpp": "e-minimax", "faster-whisper (CTranslate2)": "e-fw", "vLLM": "e-vllm", "SGLang-Omni": "e-sglang",
                  "OpenMOSS PyTorch reference": "e-moss", "parakeet-mlx": "e-mlx", "mlx-whisper": "e-mlx", "mlx-audio": "e-mlx", "FluidAudio": "e-fluid"}
OTHER_ENGINE = "e-other"
METRICS = {"ASR": "transcription speed", "TTS": "speech synthesis speed", "TTS cloning": "voice cloning speed", "Music": "music generation speed"}
DEFAULT_METRIC = "speed"
METRIC_NOTES = {
    "ASR": "Seconds of audio transcribed per second of processing, higher is better. Each chart is one backend and clip, and every row in it ran the same input under the same timing scope.",
    "TTS": "Seconds of speech generated per second of processing, higher is better. Each chart is one backend, and every row in it synthesized the same prompt under the same timing scope.",
    "TTS cloning": "Seconds of cloned speech generated per second of processing, higher is better. Each chart is one backend and reference recording, under the same timing scope for every row.",
    "Music": "Seconds of music generated per second of processing, higher is better. Each chart is one backend, and every row in it received the same request as a fresh process including model load.",
}
SETS_NOTE = ("Weekly charts hold this week's QVAC variants and the engines the weekly run drives beside them (the MLX engines on the Mac Studio); "
             "report and campaign charts hold QVAC and its competitors as that comparison measured them, and the sets are never mixed.")
SCENARIO_SHORT = {"jfk": "JFK 11 s", "abcba": "ABCBA 160.6 s", "long_clip": "98.49 s clip", "fixed_prompt": "fixed prompt", "fixed_prompt_short": "9-word prompt",
                  "long_prompt": "76-word prompt", "clone_jfk": "cloning, JFK reference", "clone_bwav": "cloning, b.wav reference",
                  "music_30s": "30 s request", "music_10s": "10 s request", "music_set": "campaign prompt set"}
SCENARIO_ORDER = tuple(SCENARIO_SHORT)
TIMING_SHORT = {"native_excl_load": "inference only", "fresh_process": "fresh process including load",
                "fresh_process_ref": "fresh process including load and reference encoding", "resident_api": "resident API",
                "http": "HTTP, full response", "campaign": "campaign harness"}
REGION_LABELS = {WEEKLY: "Timed region", REPORT: "Timed region", CAMPAIGN: "Campaign method"}
DURATION_TASKS = ("TTS", "TTS cloning", "Music")
TABLE_HEAD = ("Engine", "Latency median (min to max)", "Audio", "Speed ×", "RTF", "Host RSS", "Device", "WER", "Runs / warmups")
AXIS_TICKS = 4
AXIS_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
DEFAULT_AXIS_TOP = 1.0
PERCENT = 100.0
MS_PER_S = 1000.0
HUB_HREF = "#"
DETAILS_SUMMARY = "Details: exact values, precision, time ratios, comparison limits and sources"
SOURCES_SUMMARY = "Sources and exact values"
NOT_MEASURED_SUMMARY = "Not measured here, and why"
NO_CHART_TITLE = "Cells without a chart (no accepted measurement in this setup)"
QVAC_ONLY_TITLE = "QVAC-only timings for this model (no competitor in the same setup)"
NO_CHART_NOTE = "No accepted measurement for this model on this machine; the details name each cell and why."
EMPTY_PAGE = "Nothing for this task on this machine yet."
NO_WEEKLY = "No weekly row for this task on this machine; the end of the page says why."
RATIO_TITLE = "Competitor ÷ QVAC time"
RATIO_NOTE = "(median latency; above 1 means QVAC took less time)"
FAILED = "failed"
MEASURED = "measured"
NOT_SCORED = "not measured"
NOT_APPLICABLE = "not applicable"
NA = "n/a"


def esc(text):
    return html.escape(str(text if text is not None else ""), quote=True)


def code(text):
    return f"<code>{esc(text)}</code>" if text else ""


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def view_token(machine, page_key):
    return f"{slug(machine)}-{page_key}"


def rank(order, value):
    return order.index(value) if value in order else len(order)


def ordered(order, values):
    return sorted(values, key=lambda value: (rank(order, value), value))


def join_words(words):
    words = list(words)
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def count_text(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def backend_label(backend):
    return BACKEND_LABELS.get(backend, backend)


def engine_class(engine):
    return ENGINE_CLASSES.get(engine, OTHER_ENGINE)


def dot(engine):
    return f'<i class="dot {engine_class(engine)}"></i>'


def engine_name(row):
    return f'{row["engine"]} ({row["variant"]})' if row.get("variant") else row["engine"]


UNDER_LOAD = "under load"


def engine_label(row):
    small = f'<small>{esc(row["variant"])}</small>' if row.get("variant") else ""
    flag = f'<small class="ul">{UNDER_LOAD}</small>' if row.get("contended") else ""
    return f'{esc(row["engine"])}{small}{flag}'


def fmt_speed(value):
    return f"{value:,.0f}×" if value >= 100 else f"{value:.1f}×" if value >= 10 else f"{value:.2f}×"


def fmt_latency(ms):
    return f"{ms / MS_PER_S:,.2f} s" if ms >= MS_PER_S else f"{ms:.1f} ms"


def fmt_mib(value):
    return f"{value:,.0f} MiB"


def is_accepted(row):
    return bool(row.get("accepted")) and bool(row.get("speed_x"))


def scenario_short(row):
    return SCENARIO_SHORT.get(row.get("scenario"), row.get("scenario_label", ""))


def timing_short(row):
    return TIMING_SHORT.get(row.get("timing_class"), row.get("timing_label", ""))


def page_rows(rows, tasks):
    return [row for row in rows if row.get("task") in tasks]


def sections(rows):
    found = {}
    for row in rows:
        found.setdefault((row["model"], row["task"]), []).append(row)
    return found


def chart_order(rows):
    first = rows[0]
    return (rank(SET_ORDER, first.get("comparison_set")), rank(BACKEND_ORDER, first.get("backend")), rank(SCENARIO_ORDER, first.get("scenario")), first.get("group", ""))


def charts(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.get("group", ""), []).append(row)
    return sorted(groups.values(), key=chart_order)


def flatten(groups):
    return [row for rows in groups for row in rows]


def has_accepted(rows):
    return any(is_accepted(row) for row in rows)


def bar_order(rows):
    return sorted(rows, key=lambda row: (not is_accepted(row), -(row.get("speed_x") or 0)))


def table_order(rows):
    return sorted(rows, key=lambda row: (not row.get("is_qvac"), not is_accepted(row), -(row.get("speed_x") or 0)))


def axis_step(value):
    raw = value / AXIS_TICKS
    base = 10 ** math.floor(math.log10(raw))
    return next(step * base for step in AXIS_STEPS if step * base >= raw)


def axis_top(rows):
    speeds = [row["speed_x"] for row in rows if is_accepted(row)]
    top = max(speeds) if speeds else 0
    return axis_step(top) * AXIS_TICKS if top > 0 else DEFAULT_AXIS_TOP


def tick_percent(index):
    return index * PERCENT / AXIS_TICKS


def tick_label(top, index):
    return f"{top * index / AXIS_TICKS:g}"


def grid_lines():
    return "".join(f'<i class="g" style="left:{tick_percent(i):.2f}%"></i>' for i in range(AXIS_TICKS + 1))


def axis_row(top):
    labels = "".join(f'<b style="left:{tick_percent(i):.2f}%">{tick_label(top, i)}</b>' for i in range(AXIS_TICKS + 1))
    return f'<div class="hr ax"><span></span><span class="ht">{labels}</span><span></span></div>'


def bar_title(row):
    if not is_accepted(row):
        return f"{engine_name(row)} · {FAILED}: {row.get('status_detail', '')}"
    parts = [engine_name(row)]
    if row.get("contended"):
        parts.append(f"{UNDER_LOAD}: {row['busy_before_pct']:.0f}% of the CPUs busy with other work before the cell")
    if row.get("latency_ms"):
        parts.append(f"latency {fmt_latency(row['latency_ms'])}")
    if row.get("audio_s"):
        parts.append(f"{row['audio_s']:.2f} s audio")
    return " · ".join(parts)


def bar_row(row, top):
    if not is_accepted(row):
        return f'<div class="hr" title="{esc(bar_title(row))}"><span class="hl">{engine_label(row)}</span><span class="ht">{grid_lines()}</span><span class="hv na">{FAILED}</span></div>'
    width = min(row["speed_x"] / top * PERCENT, PERCENT)
    return (f'<div class="hr" title="{esc(bar_title(row))}"><span class="hl">{engine_label(row)}</span><span class="ht">{grid_lines()}'
            f'<span class="bar {engine_class(row["engine"])}" style="width:{width:.2f}%"></span></span><span class="hv">{fmt_speed(row["speed_x"])}</span></div>')


def bars(rows, top):
    return "".join(bar_row(row, top) for row in bar_order(rows))


def chart_title(row):
    detail = f" · {row['model_detail']}" if row.get("comparison_set") == CAMPAIGN and row.get("model_detail") else ""
    return f"{backend_label(row.get('backend', ''))} · {scenario_short(row)}{detail}"


def chart_date(rows):
    return max((row.get("date") or "" for row in rows), default="")


def set_label(rows):
    name = rows[0].get("comparison_set")
    if name in COPIED_SETS:
        return f"{name} {chart_date(rows)}"
    return f"weekly run {chart_date(rows)}"


def panel_html(rows):
    top = axis_top(rows)
    return (f'<div class="panel"><h4>{esc(chart_title(rows[0]))}</h4><p class="cap">{esc(timing_short(rows[0]))} · {esc(set_label(rows))}</p>'
            f'<div class="hb">{bars(rows, top)}{axis_row(top)}</div></div>')


def panels_html(groups):
    if not groups:
        return f'<p class="cap">{esc(NO_CHART_NOTE)}</p>'
    return '<div class="panels">' + "".join(panel_html(rows) for rows in groups) + "</div>"


def latency_cell(row):
    if not row.get("accepted"):
        detail = row.get("status_detail") or ""
        return f"{FAILED} ({detail})" if detail and detail != FAILED else FAILED
    if row.get("latency_ms") is None:
        return NA
    if row.get("min_ms") is None or row.get("max_ms") is None:
        return fmt_latency(row["latency_ms"])
    return f"{fmt_latency(row['latency_ms'])} ({fmt_latency(row['min_ms'])} to {fmt_latency(row['max_ms'])})"


def audio_cell(row):
    return f"{row['audio_s']:.2f} s" if row.get("audio_s") else NA


def speed_cell(row):
    return fmt_speed(row["speed_x"]) if is_accepted(row) else NA


def rtf_cell(row):
    return f"{row['rtf']:.3f}" if row.get("rtf") and row.get("accepted") else NA


def rss_cell(row):
    return fmt_mib(row["rss_mib"]) if row.get("rss_mib") is not None else NA


def device_cell(row):
    if row.get("backend") == "cpu":
        return NOT_APPLICABLE
    if row.get("gpu_mib") is not None:
        return fmt_mib(row["gpu_mib"])
    return row.get("gpu_kind") or NA


def wer_cell(row):
    return f"{row['wer_pct']:.1f}%" if row.get("wer_pct") is not None else NA


def runs_cell(row):
    if row.get("runs") is None:
        return NA
    return f"{row['runs']} / {row['warmups'] if row.get('warmups') is not None else NA}"


def table_row(row):
    cells = [audio_cell(row), speed_cell(row), rtf_cell(row), rss_cell(row), device_cell(row), wer_cell(row), runs_cell(row)]
    return (f'<tr><td>{dot(row["engine"])} {esc(engine_name(row))}</td><td>{esc(latency_cell(row))}</td>'
            + "".join(f"<td>{esc(cell)}</td>" for cell in cells) + "</tr>")


def results_table(rows):
    head = "".join(f"<th>{esc(title)}</th>" for title in TABLE_HEAD)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(table_row(row) for row in rows)}</tbody></table></div>'


def precision_item(row):
    return f'<li>{dot(row["engine"])} {esc(row["engine"])}: {esc(row.get("precision") or NA)}</li>'


def precision_list(rows):
    return '<p class="cap">Precision by row</p><ul class="plain">' + "".join(precision_item(row) for row in rows) + "</ul>"


def time_ratio(row, qvac):
    if not is_accepted(row):
        return None
    if row.get("latency_ms") and qvac.get("latency_ms"):
        return row["latency_ms"] / qvac["latency_ms"]
    if row.get("rtf") and qvac.get("rtf"):
        return row["rtf"] / qvac["rtf"]
    return None


def ratio_item(row, qvac):
    ratio = time_ratio(row, qvac)
    value = f"<b>{ratio:.2f}</b>" if ratio is not None else f"<b>{NA}</b> ({esc(row.get('status_detail', ''))})"
    return f"<li>{dot(row['engine'])} {esc(engine_name(row))}: {value}</li>"


def qualifier_text(rows):
    return next((row["ratio_qualifier"] for row in rows if row.get("ratio_qualifier")), "")


def ratio_block(rows):
    qvac = [row for row in rows if row.get("is_qvac") and is_accepted(row)]
    others = [row for row in rows if not row.get("is_qvac")]
    if len(qvac) != 1 or not others:
        return ""
    qualifier = qualifier_text(rows)
    qualifier_html = f'<p class="cap">Qualifier: {esc(qualifier[:1].upper() + qualifier[1:])}</p>' if qualifier else ""
    return (f'<div class="ratio"><p><b>{esc(RATIO_TITLE)}</b> <span class="cap">{esc(RATIO_NOTE)}</span></p>'
            f'<ul>{"".join(ratio_item(row, qvac[0]) for row in others)}</ul>{qualifier_html}</div>')


def duration_note(rows):
    timed = [row for row in rows if is_accepted(row) and row.get("audio_s") and row.get("task") in DURATION_TASKS]
    if len({round(row["audio_s"], 2) for row in timed}) < 2:
        return ""
    listed = ", ".join(f"{engine_name(row)} {row['audio_s']:.2f} s" for row in timed)
    return f'<p class="note">Generated durations differ ({esc(listed)}), so latency and speed ratios are not identical.</p>'


def failure_note(row):
    caveats = f" {row['caveats']}" if row.get("caveats") else ""
    return f'<p class="note">{esc(engine_name(row))} {esc(latency_cell(row))}.{esc(caveats)}</p>'


def failure_notes(rows):
    return "".join(failure_note(row) for row in rows if not row.get("accepted"))


def context_notes(rows):
    texts = dict.fromkeys(text for row in rows for text in (row.get("comparison_limitation"), row.get("host_conditions")) if text)
    return "".join(f'<p class="note info">{esc(text)}</p>' for text in texts)


def notes_html(rows):
    return duration_note(rows) + failure_notes(rows) + context_notes(rows)


def source_text(row):
    raw = f" · raw: {row['raw_source']}" if row.get("raw_source") else ""
    return f"{row.get('source_file', '')} row {row.get('source_row', '')}{raw}"


def iterations_text(row):
    if row.get("runs") is None:
        return ""
    threads = f" · {row['threads']} CPU threads" if row.get("threads") else ""
    return f"{row['runs']} timed · {row['warmups'] if row.get('warmups') is not None else NA} warmup{threads}"


def device_text(row):
    if row.get("backend") == "cpu":
        return f"{NOT_APPLICABLE} · CPU backend"
    value = f"{fmt_mib(row['gpu_mib'])} · " if row.get("gpu_mib") is not None else ""
    return " · ".join(part for part in (f"{value}{row.get('gpu_kind', '')}", row.get("gpu_note", "")) if part)


def wer_text(row):
    if row.get("wer_pct") is None:
        return f"{NA} · {row.get('wer_reason') or NOT_SCORED}"
    return " · ".join(part for part in (f"{row['wer_pct']:.1f}%", row.get("wer_kind", ""), row.get("wer_scope", "")) if part)


def measured_text(row):
    window = f"{row['started_utc']} to {row['finished_utc']}" if row.get("started_utc") and row.get("finished_utc") else ""
    return " · ".join(part for part in (row.get("date", ""), window, row.get("date_source", "")) if part)


def run_text(row):
    return f"{row.get('run', '')} · {row.get('model', '')} · {row['engine']} · {row.get('machine', '')} {row.get('backend', '')} · {row.get('status', '')}"


def precision_text(row):
    if not row.get("precision"):
        return ""
    source = f" ({row['precision_source']})" if row.get("precision_source") else ""
    return f"{row['precision']}{source}"


def timing_text(row):
    return ". ".join(part for part in (row.get("timing_label", ""), row.get("timing_includes", "")) if part)


def source_fields(row):
    return [("Run", esc(run_text(row))), ("Source", esc(source_text(row))), ("Samples (ms)", esc(row.get("samples_ms", ""))),
            ("Audio per run (s)", f"{row['audio_s']:.2f}" if row.get("audio_s") else ""), ("Precision", esc(precision_text(row))),
            ("Timing scope", esc(timing_text(row))), ("Iterations", esc(iterations_text(row))), ("Device memory", esc(device_text(row))),
            ("WER", esc(wer_text(row))), ("Measured", esc(measured_text(row))), ("Engine pins", code(row.get("pins"))),
            ("Model revisions", esc(row.get("model_revisions", ""))), ("Hashes", code(row.get("hashes"))), ("Configuration", code(row.get("config"))),
            ("Command", code(row.get("command"))), ("Caveats", esc(row.get("caveats", "")))]


def source_list(row):
    return "<dl>" + "".join(f"<dt>{esc(label)}</dt><dd>{value}</dd>" for label, value in source_fields(row) if value) + "</dl>"


def sources_html(rows):
    return f'<details class="src"><summary>{esc(SOURCES_SUMMARY)}</summary>{"".join(source_list(row) for row in rows)}</details>'


def source_label(row):
    return row.get("run", "") if row.get("comparison_set") == WEEKLY else row.get("raw_source") or row.get("run", "")


def detail_caption(first, rows):
    return (f'<p class="cap"><b>Input</b> {esc(first.get("scenario_label", ""))} · <b>Timing</b> {esc(first.get("timing_label", ""))} · '
            f'<b>Source</b> {esc(source_label(first))} · <b>Measured</b> {esc(chart_date(rows))}</p>')


def region_caption(first):
    if not first.get("timing_includes"):
        return ""
    return f'<p class="cap">{esc(REGION_LABELS.get(first.get("comparison_set"), REGION_LABELS[WEEKLY]))}: {esc(first["timing_includes"])}</p>'


def detail_block(rows):
    rows = table_order(rows)
    first = rows[0]
    return (f'<div class="gd"><h4>{esc(chart_title(first))} · {esc(timing_short(first))}</h4>{detail_caption(first, rows)}{region_caption(first)}'
            f'{results_table(rows)}{precision_list(rows)}{ratio_block(rows)}{notes_html(rows)}{sources_html(rows)}</div>')


def listed_value(row):
    if not is_accepted(row) or not row.get("speed_x"):
        return latency_cell(row)
    return f"{latency_cell(row)}, {fmt_speed(row['speed_x'])}"


def blank_item(row):
    return (f'<li>{dot(row["engine"])} {esc(engine_name(row))} · {esc(backend_label(row.get("backend", "")))} · {esc(scenario_short(row))} · '
            f'{esc(timing_short(row))}: {esc(listed_value(row))}</li>')


def list_block(title, groups):
    rows = flatten(groups)
    return f'<div class="gd"><h4>{esc(title)}</h4><ul class="plain">{"".join(blank_item(row) for row in rows)}</ul>{sources_html(rows)}</div>'


def details_html(drawn, listed, blank):
    blocks = [detail_block(rows) for rows in drawn]
    blocks += [list_block(QVAC_ONLY_TITLE, listed)] if listed else []
    blocks += [list_block(NO_CHART_TITLE, blank)] if blank else []
    return f'<details class="md"><summary>{esc(DETAILS_SUMMARY)}</summary>{"".join(blocks)}</details>'


def is_listed(rows):
    return all(row.get("no_chart") for row in rows)


def required_tag(rows):
    return ' <span class="tag">required</span>' if any(row.get("required_family") for row in rows) else ""


def section_note(task, rows):
    sets = {row.get("comparison_set") for row in rows}
    extra = f" {SETS_NOTE}" if WEEKLY in sets and sets & set(COPIED_SETS) else ""
    return METRIC_NOTES.get(task, "") + extra


def section_html(model, task, rows):
    groups = charts(rows)
    drawn = [group for group in groups if has_accepted(group) and not is_listed(group)]
    listed = [group for group in groups if is_listed(group)]
    blank = [group for group in groups if not has_accepted(group) and not is_listed(group)]
    return (f'<section class="msec"><h2>{esc(model)}: {esc(METRICS.get(task, DEFAULT_METRIC))} <span class="hib">(higher is better)</span>{required_tag(rows)}</h2>'
            f'<p class="one">{esc(section_note(task, rows))}</p>{panels_html(drawn)}{details_html(drawn, listed, blank)}</section>')


def sections_html(rows):
    return "".join(section_html(model, task, group) for (model, task), group in sections(rows).items())


def missing_cells(coverage, tasks):
    return [cell for cell in coverage if cell.get("task") in tasks and cell.get("status") != MEASURED]


def missing_groups(cells):
    groups = {}
    for cell in cells:
        groups.setdefault((cell.get("status", ""), cell.get("model", ""), cell.get("engine", ""), cell.get("reason", "")), []).append(cell.get("backend", ""))
    return groups


def missing_item(key, backends):
    status, model, engine, reason = key
    pill = "pill bad" if status == FAILED else "pill"
    names = join_words(backend_label(backend) for backend in ordered(BACKEND_ORDER, {b for b in backends if b}))
    where = f" on {esc(names)}" if names else ""
    return f'<li><span class="{pill}">{esc(status)}</span> {esc(model)} · {esc(engine)}{where}: {esc(reason)}</li>'


def not_measured_html(coverage, tasks, expanded=False):
    groups = missing_groups(missing_cells(coverage, tasks))
    if not groups:
        return ""
    items = "".join(missing_item(key, backends) for key, backends in groups.items())
    state = " open" if expanded else ""
    return f'<details class="md"{state}><summary>{esc(NOT_MEASURED_SUMMARY)}</summary><ul class="plain">{items}</ul></details>'


def weekly_sentence(machine, rows):
    if not rows:
        return esc(NO_WEEKLY)
    models = count_text(len({row["model"] for row in rows}), "model")
    backends = join_words(backend_label(backend) for backend in ordered(BACKEND_ORDER, {row["backend"] for row in rows}))
    scopes = "; ".join(dict.fromkeys(timing_short(row) for row in rows))
    return esc(f"QVAC from the weekly run {machine.get('newest_run', '')} at {machine.get('pins', '')}: {models} on {backends}. Timing scopes: {scopes}.")


def campaign_sentence(rows):
    engines = join_words(sorted({row["engine"] for row in rows if not row.get("is_qvac")}))
    if not engines:
        return ""
    dates = join_words(sorted({row["date"] for row in rows if row.get("date")}))
    return esc(f"Campaign charts compare QVAC with {engines} as the campaign of {dates} measured them, with that campaign's own QVAC build.")


def dashboard_sentence(machine):
    if not machine.get("url"):
        return ""
    return (f'Week-over-week lines and the run log are on the <a href="{esc(machine["url"])}" target="_blank" rel="noopener">'
            f'{esc(machine["machine"])} dashboard</a>.')


def report_sentence(rows):
    engines = join_words(sorted({row["engine"] for row in rows if not row.get("is_qvac")}))
    if not rows:
        return ""
    dates = join_words(sorted({row["date"] for row in rows if row.get("date")}))
    against = f" with {engines}" if engines else ""
    return esc(f"Report charts are copied from the comparison report of {dates}: QVAC{against} as that report measured them, with its own builds, inputs and timing scopes.")


def lede_html(machine, rows):
    weekly = [row for row in rows if row.get("comparison_set") == WEEKLY]
    report = [row for row in rows if row.get("comparison_set") == REPORT]
    campaign = [row for row in rows if row.get("comparison_set") == CAMPAIGN]
    parts = (weekly_sentence(machine, weekly), report_sentence(report), campaign_sentence(campaign), dashboard_sentence(machine))
    return " ".join(part for part in parts if part)


def nav_item(machine, page, current):
    key, title, _ = page
    if key == current:
        return f'<span class="cur">{esc(title)}</span>'
    return f'<a href="#{view_token(machine["machine"], key)}">{esc(title)}</a>'


def nav_html(machine, current):
    return '<nav class="jump">' + "".join(nav_item(machine, page, current) for page in TASK_PAGES) + "</nav>"


def header_html(machine, page, rows):
    key, title, _ = page
    eyebrow = f'<div class="eyebrow"><a href="{HUB_HREF}">Speech Bench</a> · {esc(machine["machine"])} · measured {esc(machine.get("date", ""))}</div>'
    return f'<header>{eyebrow}<h1>{esc(machine["machine"])}: {esc(title)}</h1><p class="lede">{lede_html(machine, rows)}</p>{nav_html(machine, key)}</header>'


def page_body(rows, coverage, tasks):
    measured = sections_html(rows)
    body = measured + not_measured_html(coverage, tasks, expanded=not measured)
    return body or f'<p class="cap">{esc(EMPTY_PAGE)}</p>'


def task_view(machine, page, rows, coverage):
    key, _, tasks = page
    task_rows = page_rows(rows, tasks)
    return (f'<div class="view" data-view="{view_token(machine["machine"], key)}" hidden><div class="wrap tp">{header_html(machine, page, task_rows)}'
            f'{page_body(task_rows, coverage, tasks)}</div></div>')


def machine_views(machine, rows, coverage):
    return "".join(task_view(machine, page, rows, coverage) for page in TASK_PAGES)


def render_views(machines, rows_by_host, coverage_by_host):
    return "".join(machine_views(machine, rows_by_host.get(machine["host"], []), coverage_by_host.get(machine["host"], [])) for machine in machines)
