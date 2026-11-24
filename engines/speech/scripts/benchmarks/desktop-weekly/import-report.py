#!/usr/bin/env python3
"""Copy the measurements of a comparison report into report-rows.json.

usage: import-report.py <pages dir> --date <report date> --hub <hub url> -o <history dir>/report-rows.json

<pages dir> holds the saved HTML of the report's task pages (one per machine and task, as the Speech Bench report
publishes them). Every chart of every page becomes rows in the shape of runs.json (comparison_set "report"), the
QVAC-only lists become rows without a chart, and the "Not measured here" lists become coverage cells, so that
build-dashboard-data.py (--report) can show the report next to the weekly rows of the same machine.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
REPORT = "report"
QVAC = "qvac-fabric-speech.cpp"
QVAC_ENGINES = (QVAC, "QVAC bundled whisper.cpp")
MACHINE_HOSTS = {"DGX Spark": "spark-d9ca", "RTX 5090": "qvac-dev-linux-x64", "Strix Halo": "qvac-dev-strix-1", "Jetson Orin": "qvac-dev-linux-arm64",
                 "Mac Studio M3 Ultra": "qvac-dev-mac-arm64", "Mac mini M4": "mac-mini-m4-gpu"}
TASKS = {"transcription speed": "ASR", "speech synthesis speed": "TTS", "voice cloning speed": "TTS cloning", "music generation speed": "Music"}
TASK_KEYS = {"ASR": "asr", "TTS": "tts", "TTS cloning": "tts", "Music": "music"}
BACKENDS = {"CPU": "cpu", "CUDA": "cuda", "Vulkan": "vulkan", "Metal": "metal", "Core ML": "coreml"}
SCENARIOS = {"JFK 11 s": "jfk", "ABCBA 160.6 s": "abcba", "fixed prompt": "fixed_prompt", "cloning, JFK reference": "clone_jfk",
             "cloning, b.wav reference": "clone_bwav", "30 s request": "music_30s", "10 s request": "music_10s"}
TIMINGS = {"fresh process including load": "fresh_process", "fresh process including load and reference encoding": "fresh_process_ref",
           "resident API": "resident_api", "native stage sum": "native_excl_load", "native, excluding load": "native_excl_load",
           "HTTP, full response": "http"}
ENGINES_WITH_PARENTHESES = ("whisper.cpp (upstream)", "faster-whisper (CTranslate2)")
MODEL_QUALIFIERS = ("OpenMOSS", "QVAC registry", "Q8 package pair")
QVAC_ONLY_TITLE = "QVAC-only timings for this model"
NOT_MEASURED_SUMMARY = "Not measured here, and why"
MEASURED = "measured"
FAILED = "failed"
SOURCE_LABEL_SUFFIX = " (source label)"
MS_PER_S = 1000.0
PERCENT = 100.0
SEPARATOR = " · "
TAG = re.compile(r"<[^>]+>")
SECTION = re.compile(r'<section class="msec"><h2>(.*?)</h2>(.*?)</section>', re.S)
CHART = re.compile(r'<div class="gd"><h4>(.*?)</h4>(.*?)(?=<div class="gd">|</details></div>|</details></section>)', re.S)
TABLE_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<td>(.*?)</td>", re.S)
LIST_ITEM = re.compile(r"<li>(.*?)</li>", re.S)
TERM = re.compile(r"<dt>(.*?)</dt><dd>(.*?)</dd>", re.S)
DEFINITION_LIST = re.compile(r"<dl>(.*?)</dl>", re.S)
NOT_MEASURED = re.compile(r"<summary>" + re.escape(NOT_MEASURED_SUMMARY) + r"</summary><ul class=\"plain\">(.*?)</ul>", re.S)
PILL = re.compile(r'<span class="pill[^"]*">(.*?)</span>(.*)', re.S)
QUALIFIER = re.compile(r'<p class="cap">Qualifier: (.*?)</p>', re.S)
MEASURED_AT = re.compile(r'<b>Measured</b> (\d{4}-\d{2}-\d{2})')
THREADS = re.compile(r"(\d+) CPU threads")
ITERATIONS = re.compile(r"(\d+) timed · (\d+) warmup")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
NAME_WITH_VARIANT = re.compile(r"^(.*?) \((.*)\)$")
QVAC_ONLY_ITEM = re.compile(r"^(.*?) · (.*?) · (.*?) · (.*?): (.*)$")
QVAC_ONLY_VALUES = re.compile(r"^(.*), (.*)×$")


def text(fragment):
    return html.unescape(TAG.sub("", fragment)).strip()


def number(fragment):
    found = NUMBER.search(fragment or "")
    return float(found.group(0).replace(",", "")) if found else None


def milliseconds(fragment):
    value = number(fragment)
    if value is None:
        return None
    return value * MS_PER_S if " s" in fragment and " ms" not in fragment else value


def split_known_name(label):
    engine = next((name for name in ENGINES_WITH_PARENTHESES if label.startswith(name)), "")
    rest = label[len(engine):].strip()
    return engine, rest[1:-1] if rest.startswith("(") and rest.endswith(")") else rest


def split_name(label):
    engine, variant = split_known_name(label)
    if engine:
        return engine, variant
    found = NAME_WITH_VARIANT.match(label)
    return (found.group(1), found.group(2)) if found else (label, "")


def page_machine(page):
    title = text(re.search(r"<title>(.*?)</title>", page, re.S).group(1))
    return next(name for name in MACHINE_HOSTS if title.startswith(name))


def section_task(heading):
    metric = text(heading).split(": ", 1)[1].split(" (")[0].strip()
    return TASKS[metric]


def split_model(name):
    found = NAME_WITH_VARIANT.match(name)
    if found and found.group(2) in MODEL_QUALIFIERS:
        return found.group(1), found.group(2)
    return name, ""


def section_model(heading):
    return split_model(text(heading).split(": ", 1)[0])


def section_required(heading):
    return 'class="tag"' in heading


def chart_parts(title):
    parts = text(title).split(SEPARATOR)
    return BACKENDS[parts[0]], parts[1], SEPARATOR.join(parts[2:])


def definitions(fragment):
    return {text(term): html.unescape(TAG.sub("", value)).strip() for term, value in TERM.findall(fragment)}


def source_lists(fragment):
    return [definitions(found) for found in DEFINITION_LIST.findall(fragment)]


def table_rows(fragment):
    rows = [[text(cell) for cell in CELL.findall(cells)] for cells in TABLE_ROW.findall(fragment)]
    return [cells for cells in rows if cells]


def precision_by_label(fragment):
    found = re.search(r'<p class="cap">Precision by row</p><ul class="plain">(.*?)</ul>', fragment, re.S)
    if not found:
        return {}
    pairs = (text(item).split(": ", 1) for item in LIST_ITEM.findall(found.group(1)))
    return {label: precision for label, precision in pairs}


def qualifier(fragment):
    found = QUALIFIER.search(fragment)
    return text(found.group(1)) if found else ""


def run_parts(definition):
    parts = definition.get("Run", "").split(SEPARATOR)
    return {"id": parts[0], "model": parts[1], "engine": parts[2], "status": parts[-1]} if len(parts) >= 5 else {}


def samples(definition):
    values = [float(v) for v in definition.get("Samples (ms)", "").split(", ") if v.strip()]
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def iterations(definition):
    found = ITERATIONS.search(definition.get("Iterations", ""))
    threads = THREADS.search(definition.get("Iterations", ""))
    return (int(found.group(1)), int(found.group(2))) if found else (None, None), int(threads.group(1)) if threads else None


def device_memory(definition):
    parts = definition.get("Device memory", "").split(SEPARATOR)
    kind = parts[0]
    return number(kind) if "MiB" in kind else None, kind, SEPARATOR.join(parts[1:])


def wer(definition):
    parts = definition.get("WER", "").split(SEPARATOR)
    value = number(parts[0]) if "%" in parts[0] else None
    kind = parts[1] if len(parts) > 1 else ""
    return value, kind, SEPARATOR.join(parts[2:])


def measured_date(definition):
    return definition.get("Measured", "").split(SEPARATOR)[0]


def load_ms(definition):
    return milliseconds(definition.get("Model load", "")) if definition.get("Model load") else None


def precision_text(definition, listed):
    return listed or definition.get("Precision", "").replace(SOURCE_LABEL_SUFFIX, "")


def latency_from_table(cell):
    if " (" not in cell:
        return milliseconds(cell), None, None
    median, span = cell.split(" (", 1)
    low, high = span.rstrip(")").split(" to ")
    return milliseconds(median), milliseconds(low), milliseconds(high)


def table_metrics(cells):
    if not cells:
        return {}
    median, low, high = latency_from_table(cells[1])
    return {"latency_ms": median, "min_ms": low, "max_ms": high, "audio_s": number(cells[2]), "speed_x": number(cells[3]),
            "rtf": number(cells[4]), "rss_mib": number(cells[5]) if "MiB" in cells[5] else None}


def list_metrics(item):
    values = QVAC_ONLY_VALUES.match(QVAC_ONLY_ITEM.match(text(item)).group(5))
    if not values:
        return {}
    return {"latency_ms": milliseconds(values.group(1)), "min_ms": None, "max_ms": None, "audio_s": None, "speed_x": number(values.group(2)),
            "rtf": None, "rss_mib": None}


def speed_of(metrics, audio_s):
    if metrics.get("speed_x"):
        return metrics["speed_x"]
    if metrics.get("latency_ms") and audio_s:
        return round(audio_s * MS_PER_S / metrics["latency_ms"], 4)
    return None


def build_row(context, chart, label, metrics, definition, index):
    engine, variant = split_name(label)
    run = run_parts(definition)
    median, low, high = samples(definition)
    (runs, warmups), threads = iterations(definition)
    gpu_mib, gpu_kind, gpu_note = device_memory(definition)
    wer_pct, wer_kind, wer_scope = wer(definition)
    audio_s = number(definition.get("Audio per run (s)", "")) or metrics.get("audio_s")
    latency = median if median is not None else metrics.get("latency_ms")
    speed = speed_of(metrics, audio_s)
    accepted = run.get("status", MEASURED) == MEASURED and latency is not None
    return {"id": f"r-{run.get('id', index)}", "host": context["host"], "machine": context["machine"], "task": chart["task"],
            "dashboard": TASK_KEYS[chart["task"]], "family": "", "model": chart["model"], "model_detail": chart["model_detail"], "required_family": chart["required"],
            "engine": engine, "is_qvac": engine in QVAC_ENGINES, "variant": variant, "backend": chart["backend"],
            "observed_backend": definition.get("Backend observed", ""), "precision": precision_text(definition, chart["precision"].get(label, "")),
            "precision_source": "report page", "scenario": chart["scenario"], "scenario_label": chart["scenario_label"],
            "timing_class": chart["timing_class"], "timing_label": chart["timing_label"], "timing_includes": chart["timing_includes"],
            "comparison_set": REPORT, "status": MEASURED if accepted else FAILED, "status_detail": run.get("status", ""), "accepted": accepted,
            "latency_ms": latency, "min_ms": low if low is not None else metrics.get("min_ms"), "max_ms": high if high is not None else metrics.get("max_ms"),
            "samples_ms": definition.get("Samples (ms)", ""), "audio_s": audio_s, "speed_x": speed, "rtf": metrics.get("rtf") or (round(1 / speed, 6) if speed else None),
            "load_ms": load_ms(definition), "rss_mib": metrics.get("rss_mib"), "gpu_mib": gpu_mib, "gpu_kind": gpu_kind, "gpu_note": gpu_note,
            "wer_pct": wer_pct, "wer_kind": wer_kind, "wer_scope": wer_scope, "wer_scored_outputs": 1 if wer_pct is not None else None,
            "wer_reason": "" if wer_pct is not None else wer_scope, "runs": runs, "warmups": warmups, "threads": threads,
            "date": measured_date(definition) or context["date"], "date_source": "report page", "started_utc": "", "finished_utc": "",
            "pins": definition.get("Engine pins", ""), "model_revisions": "", "hashes": definition.get("Hashes", ""),
            "command": definition.get("Command", ""), "config": definition.get("Configuration", ""), "caveats": definition.get("Caveats", ""),
            "comparison_limitation": context["limitation"], "host_conditions": "", "source_file": context["source_file"], "source_row": index,
            "raw_source": definition.get("Source", ""), "group": chart["group"], "ratio_qualifier": chart["qualifier"], "run": REPORT,
            "no_chart": chart.get("no_chart", False)}


def chart_context(context, section, title, body, chart_index):
    backend, scenario_label, timing_label = chart_parts(title)
    region = re.search(r'<p class="cap">Timed region: (.*?)</p>', body, re.S)
    (model, model_detail), task = section_model(section), section_task(section)
    return {"model": model, "model_detail": model_detail, "task": task, "required": section_required(section), "backend": backend,
            "scenario": SCENARIOS.get(scenario_label, scenario_label), "scenario_label": scenario_label,
            "timing_class": TIMINGS.get(timing_label, timing_label), "timing_label": timing_label, "timing_includes": text(region.group(1)) if region else "",
            "precision": precision_by_label(body), "qualifier": qualifier(body),
            "group": f"{context['host']}|{backend}|{model}|{task}|{SCENARIOS.get(scenario_label, scenario_label)}|{REPORT}|{chart_index}"}


def pair_rows(labels, metrics, definitions_list):
    return list(zip(labels, metrics, definitions_list + [{}] * (len(labels) - len(definitions_list))))


def chart_rows(context, section, title, body, chart_index, start):
    chart = chart_context(context, section, title, body, chart_index)
    rows = table_rows(body)
    labels = [cells[0] for cells in rows]
    metrics = [table_metrics(cells) for cells in rows]
    return [build_row(context, chart, label, metric, definition, start + offset)
            for offset, (label, metric, definition) in enumerate(pair_rows(labels, metrics, source_lists(body)))]


def qvac_only_chart(context, section, item, definition, chart_index):
    found = QVAC_ONLY_ITEM.match(text(item))
    title = SEPARATOR.join(found.group(2, 3, 4))
    return dict(chart_context(context, section, title, "", chart_index), no_chart=True)


def qvac_only_rows(context, section, body, chart_index, start):
    items = LIST_ITEM.findall(body.split("<details", 1)[0])
    out = []
    for offset, (item, definition) in enumerate(zip(items, source_lists(body))):
        chart = qvac_only_chart(context, section, item, definition, f"{chart_index}-{offset}")
        label = QVAC_ONLY_ITEM.match(text(item)).group(1)
        out.append(build_row(context, chart, label, list_metrics(item), definition, start + offset))
    return out


def section_rows(context, heading, body, start):
    out = []
    for chart_index, (title, chart_body) in enumerate(CHART.findall(body)):
        if text(title).startswith(QVAC_ONLY_TITLE):
            out += qvac_only_rows(context, heading, chart_body, chart_index, start + len(out))
        else:
            out += chart_rows(context, heading, title, chart_body, chart_index, start + len(out))
    return out


def page_rows(context, page, start):
    out = []
    for heading, body in SECTION.findall(page):
        if ": " in text(heading):
            out += section_rows(context, heading, body, start + len(out))
    return out


def coverage_cell(context, item, task, index):
    found = PILL.match(item.strip())
    subject, reason = text(found.group(2)).split(": ", 1) if ": " in text(found.group(2)) else (text(found.group(2)), "")
    model, engine = subject.split(SEPARATOR, 1) if SEPARATOR in subject else (subject, "")
    return {"id": f"r{index:03d}", "host": context["host"], "machine": context["machine"], "task": task, "model": model, "engine": engine,
            "backend": "", "status": text(found.group(1)), "rows": 0, "reason": reason, "source": context["source_file"], "scenario": ""}


def page_task(page):
    heading = text(re.search(r"<h1>(.*?)</h1>", page, re.S).group(1)).split(": ", 1)[1]
    return {"TTS and voice cloning": "TTS", "ASR": "ASR", "Music generation": "Music"}[heading]


def page_coverage(context, page, start):
    found = NOT_MEASURED.search(page)
    if not found:
        return []
    return [coverage_cell(context, item, page_task(page), start + index) for index, item in enumerate(LIST_ITEM.findall(found.group(1)))]


def page_context(page, date, source_file):
    machine = page_machine(page)
    return {"machine": machine, "host": MACHINE_HOSTS[machine], "date": date, "source_file": source_file,
            "limitation": f"Comparison report of {date}: its own builds, inputs and timing scopes, not the weekly run; compare only within this chart."}


def import_pages(paths, date, source_file):
    rows, coverage = [], []
    for path in paths:
        page = path.read_text(encoding="utf-8")
        context = page_context(page, date, source_file)
        rows += page_rows(context, page, len(rows))
        coverage += page_coverage(context, page, len(coverage))
    return rows, coverage


def document(rows, coverage, date, hub, pages):
    return {"_schema": {"description": "Rows copied from the task pages of a comparison report by import-report.py, in the shape of runs.json "
                                       "(comparison_set report) plus the report's not-measured lists as coverage cells.",
                        "report": {"date": date, "hub": hub, "pages": [p.name for p in pages]}},
            "rows": rows, "coverage": coverage}


def dump(doc):
    lines = [json.dumps(row, ensure_ascii=False) for row in doc["rows"]]
    cells = [json.dumps(cell, ensure_ascii=False) for cell in doc["coverage"]]
    return ('{"_schema": ' + json.dumps(doc["_schema"], ensure_ascii=False) + ',\n"rows": [\n' + ",\n".join(lines) + '\n],\n"coverage": [\n'
            + ",\n".join(cells) + "\n]}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pages", type=pathlib.Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--hub", default="")
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pages = sorted(args.pages.glob("*.html"))
    rows, coverage = import_pages(pages, args.date, args.out.name)
    args.out.write_text(dump(document(rows, coverage, args.date, args.hub, pages)), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} rows and {len(coverage)} coverage cells from {len(pages)} pages")


if __name__ == "__main__":
    main()
