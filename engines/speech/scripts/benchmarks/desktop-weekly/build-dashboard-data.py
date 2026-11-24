#!/usr/bin/env python3
"""Write the data files of the per-machine dashboards and of the hub page from the stored summaries.

usage: build-dashboard-data.py <summaries dir> -o <out dir> [--comparisons comparisons.json] [--engines engines.json]
       [--families families.json] [--dashboards dashboards.json] [--report report-rows.json]

For every machine with a stored run it writes <out dir>/<host>/: runs.json (one row per backend and model of the
machine's newest result, plus the campaign rows measured on that machine and the rows of the comparison report that
import-report.py copied for it), history.json (every stored result, for the
week-over-week lines), coverage.json (measured, failed and skipped cells with their reasons), engines.json (the engines
with where they were measured), machine.json (hardware, counts, snapshot) and commentary.json (the overview summary,
the notes of each task tab and the comparison rules). It also writes <out dir>/hub.json, the input of build-landing.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
QVAC = "qvac-fabric-speech.cpp"
TASK_ORDER = ("ASR", "TTS", "TTS cloning", "Music")
TASK_KEYS = {"ASR": "asr", "TTS": "tts", "TTS cloning": "tts", "Music": "music"}
TASK_NAMES = {"ASR": "ASR", "TTS": "TTS", "TTS cloning": "voice cloning", "Music": "music"}
SUMMARY = "summary"
GENERAL = "all"
RETIRED_HOSTS = ("qvac-dev-strix-0",)
MACHINE_LABELS = {"spark-d9ca": "DGX Spark", "qvac-dev-linux-x64": "RTX 5090", "qvac-dev-strix-1": "Strix Halo", "qvac-dev-linux-arm64": "Jetson Orin",
                  "qvac-dev-mac-arm64": "Mac Studio M3 Ultra", "mac-mini-m4-gpu": "Mac mini M4"}
BACKEND_NAMES = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal", "coreml": "Core ML"}
VARIANT_LABELS = {"parakeet-tdt": "Q8_0", "parakeet-unified": "Q8_0", "transcribepsy": "Q8_0", "nemotron": "F16", "supertonic3": "Q8_0",
                  "audio8-lm-f16-dec-f16": "LM F16 · decoder F16", "audio8-lm-f16-dec-q8_0": "LM F16 · decoder Q8_0",
                  "audio8-lm-q8_0-dec-f16": "LM Q8_0 · decoder F16", "audio8-lm-q8_0-dec-q8_0": "LM Q8_0 · decoder Q8_0",
                  "audio8-lm-q4_0-dec-f16": "LM Q4_0 · decoder F16", "audio8-lm-q4_0-dec-q8_0": "LM Q4_0 · decoder Q8_0",
                  "audio8-clone-enc-f16": "encoder F16", "audio8-clone-enc-q8_0": "encoder Q8_0", "moss-tts-f16": "F16",
                  "moss-tts-q8_0": "Q8_0 blocks", "moss-tts-clone-f16": "F16", "acestep-turbo-q8_0": "DiT Q8_0",
                  "acestep-turbo-q4_k_m": "DiT Q4_K_M", "acestep-sft-q8_0": "DiT Q8_0", "acestep-base-q4_k_m": "DiT Q4_K_M",
                  "parakeet-eou": "F16", "whisper-small": "F16", "parakeet-tdt-abcba": "Q8_0", "parakeet-unified-abcba": "Q8_0",
                  "transcribepsy-abcba": "Q8_0", "nemotron-abcba": "F16", "parakeet-eou-abcba": "F16", "whisper-small-abcba": "F16",
                  "chatterbox-turbo": "Q8_0", "chatterbox-mtl-clone": "Q8_0", "cosyvoice3-clone": "LM Q8_0", "parler-mini": "Q8_0",
                  "parler-indic": "Q8_0", "parakeet-mlx": "BF16", "parakeet-mlx-abcba": "BF16", "mlx-whisper": "F16", "mlx-whisper-abcba": "F16",
                  "nemotron-mlx-audio": "BF16", "nemotron-mlx-audio-abcba": "BF16", "chatterbox-turbo-mlx-audio": "F16",
                  "audio8-mlx-audio": "BF16", "audio8-clone-mlx-audio": "BF16", "moss-tts-mlx-audio": "BF16",
                  "minimax-music3-mlx-audio": "8-bit", "parakeet-fluidaudio": "Core ML int8, ANE", "parakeet-fluidaudio-abcba": "Core ML int8, ANE",
                  "parakeet-unified-fluidaudio": "Core ML int8, ANE", "parakeet-unified-fluidaudio-abcba": "Core ML int8, ANE",
                  "parakeet-tdt-coreml": "Q8_0, Core ML encoder and decoder on the ANE", "parakeet-unified-coreml": "Q8_0, Core ML encoder on the ANE",
                  "parakeet-eou-coreml": "F16, Core ML encoder on the ANE", "nemotron-coreml": "F16, Core ML encoder on the ANE",
                  "parakeet-tdt-abcba-coreml": "Q8_0, Core ML encoder and decoder on the ANE", "parakeet-unified-abcba-coreml": "Q8_0, Core ML encoder on the ANE",
                  "parakeet-eou-abcba-coreml": "F16, Core ML encoder on the ANE", "nemotron-abcba-coreml": "F16, Core ML encoder on the ANE",
                  "supertonic3-coreml": "Q8_0, Core ML vocoder on the ANE", "audio8-lm-q8_0-dec-q8_0-coreml": "LM Q8_0 · decoder Q8_0, Core ML codec on the ANE",
                  "minimax-music3-q8_0": "LM Q8_0 · synth Q8_0"}
CAMPAIGN_HOSTS = {"RTX 5090 box": "qvac-dev-linux-x64", "Strix Halo": "qvac-dev-strix-1", "DGX Spark GB10": "spark-d9ca",
                  "Mac Studio M3 Ultra": "qvac-dev-mac-arm64", "Jetson Orin 8 GB": "qvac-dev-linux-arm64"}
HOST_CONDITIONS = {"qvac-dev-linux-x64": "Shared box: the bench uses GPU 0 while other people's jobs may run on GPU 1."}
SCENARIOS = {"ASR": ("jfk", "JFK clip, 11.0 s (has reference transcript)"), "TTS": ("fixed_prompt", "Fixed English prompt"),
             "TTS cloning": ("clone_jfk", "Voice cloning from JFK reference, fixed English target text"),
             "Music": ("music_30s", "30 s instrumental request, seed 42")}
FAMILY_SCENARIOS = {"abcba": ("abcba", "ABCBA multi-speaker clip, 160.6 s (no reference transcript)")}
TIMING = {"native": ("native_excl_load", "Native bench, inference only (excludes model load)",
                     "Mel, encoder and decoder of one clip; the model is loaded before the timed runs."),
          "time-wrapped": ("fresh_process", "Fresh process incl. model load (warm filesystem cache)",
                           "Whole process wall clock from launch to exit, model load and output writing included.")}
TIMING_CLONING = ("fresh_process_ref", "Fresh process incl. load and reference encoding (warm filesystem cache)",
                  "Whole process wall clock including model load and the encoding of the reference recording.")
CAMPAIGN_TIMING = ("campaign", "Campaign measurement (see the harness note)", "")
REPORT = "report"
REPORT_FILE = "report-rows.json"
ENGINES_FILE = "engines.json"
CAMPAIGN_CLIPS = {"ASR": (("jfk", "JFK clip, 11.0 s", 11.0), ("long_clip", "98.49 s clip", 98.49)),
                  "TTS": (("fixed_prompt_short", "9-word prompt (about 3.1 s of speech)", 3.1), ("long_prompt", "76-word prompt (about 26 s of speech)", 26.0)),
                  "Music": (("music_set", "Ten fixed prompts of the campaign", None),)}
WER_KINDS = {"wer": ("ASR ground truth", "JFK reference transcript, single clip; smoke check, not a corpus benchmark"),
             "tts_wer": ("TTS intelligibility", "Pinned Whisper tiny transcribes the generated clip; scorer errors included")}
QUALITY_FLAG_PCT = 10.0
MOVE_THRESHOLD = 0.05
QUICK_MARK = "-quick"
PERCENT = 100.0
MS_PER_S = 1000.0
MS_SECONDS_FROM = 10000.0
SHA_SHORT = 10
RTF_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
PAIR_SEPARATOR = "/"
NO_RATIO = "no ratio recorded"
NOT_RECORDED = "not recorded"
RULES = [
    ("Weekly rows and campaign rows are separate", "The weekly rows come from the Sunday run at the engine master head and the ggml commit it pins; the campaign rows keep the build, inputs and dates of the report they come from. Each chart compares rows of one source only: a weekly chart holds QVAC variants, a campaign chart holds QVAC and its competitors as the campaign measured them."),
    ("Timing scopes", "ASR, Supertonic and Nemotron rows time inference only. Audio8, MOSS-TTS and ACE-Step rows are whole fresh processes including model load, so their speed is a throughput figure, not a latency. Rows of different scopes are never in one chart."),
    ("Competitor ÷ QVAC time", "In a campaign chart, the competitor's time divided by QVAC's time on the same device, backend and input; above 1 means QVAC took less time. The campaign measured its own QVAC build, so the ratio describes that build, not this week's."),
    ("Reading a move", "A slower cell with unchanged engine and ggml commits points at load from another user that night; a changed commit names what to look at."),
]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_rows(path, default=()):
    if not path or not pathlib.Path(path).is_file():
        return list(default)
    data = load_json(path)
    return data.get("rows", []) if isinstance(data, dict) else data


def current_summary_paths(folder):
    return [path for path in sorted(pathlib.Path(folder).glob("*/*.json")) if path.parent.name not in RETIRED_HOSTS]


def load_summaries(folder):
    found = []
    for path in current_summary_paths(folder):
        summary = load_json(path)
        summary["_host"] = path.parent.name
        summary["_file"] = f"{path.parent.name}/{path.name}"
        found.append(summary)
    return sorted(found, key=lambda s: (s["_host"], run_started(s)))


def load_families(path):
    return {k: v for k, v in load_json(path).items() if not k.startswith("_")}


def machine_label(host):
    return MACHINE_LABELS.get(host, host)


def backend_name(backend):
    return BACKEND_NAMES.get(backend, backend)


def run_started(summary):
    return summary.get("run", {}).get("started") or ""


def run_name(summary):
    return summary.get("run", {}).get("name", "")


def short_sha(pins, key):
    return ((pins.get(key) or {}).get("sha") or "")[:SHA_SHORT]


def pins_text(summary):
    pins = summary.get("pins", {})
    engine, ggml = pins.get("engine") or {}, pins.get("ggml") or {}
    return f"qvac {short_sha(pins, 'engine')} ({engine.get('commit_date', '')}), ggml {short_sha(pins, 'ggml')} ({ggml.get('commit_date', '')})"


def variant_label(row):
    return VARIANT_LABELS.get(row["family"]) or row.get("weights", "")


def scenario_of(spec):
    task_default = SCENARIOS.get(spec.get("task", ""), SCENARIOS["ASR"])
    return FAMILY_SCENARIOS.get(spec.get("scenario", ""), task_default)


def family_models(summary, family, families):
    names = {pathlib.Path(a.split("/")[-1]).name for a in families.get(family, {}).get("args", []) if "${MODEL_DIR}/" in a}
    return [m for m in summary.get("pins", {}).get("models", []) if m.get("name") in names]


def hashes_text(models):
    return "; ".join(f"{m['name']} {(m.get('sha256') or '')[:SHA_SHORT]}" for m in models)


def revisions_text(models):
    return "; ".join(f"{m['name']}: {m.get('source', '')}" for m in models)


def command_text(family, families):
    spec = families.get(family, {})
    return f"{spec.get('binary', '')} {' '.join(spec.get('args', []))}".strip()


def iteration_counts(family, backend, families):
    spec = families.get(family, {})
    if backend == "cpu":
        return spec.get("cpu_runs"), spec.get("cpu_warmup")
    return spec.get("runs"), spec.get("warmup")


def engine_of(family, families):
    return families.get(family, {}).get("engine") or QVAC


def timing_of(row, families):
    bench_kind = families.get(row["family"], {}).get("bench_kind", "native")
    if row.get("task") == "TTS cloning" and bench_kind == "time-wrapped":
        return TIMING_CLONING
    return TIMING.get(bench_kind, TIMING["native"])


def gpu_fields(row, summary):
    rise = row.get("gpu_memory_rise_mib")
    unified = (summary.get("machine", {}).get("gpu") or {}).get("unified_memory")
    if rise is not None:
        return rise, "sampled rise", "Rise of the GPU's used memory while the cell ran, sampled with nvidia-smi."
    if unified:
        return None, "unavailable", "Unified memory: the GPU reports no per-process memory use; host RSS carries the model."
    return None, "unavailable", "No GPU memory sampler on this lane."


def wer_fields(row):
    quality = row.get("quality") or {}
    kind, scope = WER_KINDS.get(quality.get("kind"), ("", ""))
    value = quality.get("value")
    if value is None:
        return None, kind, scope, "Not measured for this family"
    return round(value * PERCENT, 2), kind, scope, ""


COREML_PREFIX = "coreml"


def observed_backend(row):
    encoder = row.get("encoder_backend") or ""
    observed = row.get("observed_backend") or ""
    if encoder.startswith(COREML_PREFIX):
        return f"{observed}, encoder {encoder}"
    return observed


def baseline_note(row):
    baseline = row.get("baseline_inference_ms_median")
    if baseline is None:
        return ""
    return f"The same process forced to the ggml encoder: {baseline:.1f} ms median."


def contention_note(row):
    if not row.get("contended"):
        return ""
    return f"Measured under load: {row['busy_before_pct']:.0f}% of the CPUs were busy with other work just before the cell, so the time can be slower than the machine's own."


def stale_note(summary, newest_name):
    if run_name(summary) == newest_name:
        return ""
    return f"From the earlier run {run_name(summary)}; the machine's newest run {newest_name} has no result for this cell."


def weekly_row(summary, row, families, newest_name):
    host, run = summary["_host"], summary.get("run", {})
    task = row.get("task", "")
    model = row.get("model_label", row["family"])
    scenario, scenario_label = scenario_of(families.get(row["family"], {"task": task}))
    timing_class, timing_label, timing_includes = timing_of(row, families)
    runs, warmups = iteration_counts(row["family"], row["backend"], families)
    gpu_mib, gpu_kind, gpu_note = gpu_fields(row, summary)
    wer_pct, wer_kind, wer_scope, wer_reason = wer_fields(row)
    models = family_models(summary, row["family"], families)
    engine = engine_of(row["family"], families)
    accepted = row.get("status") == "ok"
    speed = row.get("speed_x") or None
    caveats = " ".join(x for x in [row.get("notes", ""), baseline_note(row), contention_note(row), stale_note(summary, newest_name)] if x)
    return {"id": f"w-{row['backend']}-{row['family']}", "host": host, "machine": machine_label(host), "task": task,
            "dashboard": TASK_KEYS.get(task, "asr"), "family": row["family"], "model": model,
            "model_detail": row.get("weights", ""), "required_family": True, "engine": engine, "is_qvac": engine == QVAC,
            "variant": variant_label(row), "backend": row["backend"], "observed_backend": observed_backend(row),
            "precision": row.get("weights", ""), "precision_source": "families.json", "scenario": scenario, "scenario_label": scenario_label,
            "timing_class": timing_class, "timing_label": timing_label, "timing_includes": timing_includes, "comparison_set": "weekly",
            "status": "measured" if accepted else "failed", "status_detail": row.get("status", ""), "accepted": accepted,
            "contended": bool(row.get("contended")), "busy_before_pct": row.get("busy_before_pct"),
            "latency_ms": row.get("wall_ms_median"), "min_ms": row.get("wall_ms_min"), "max_ms": row.get("wall_ms_max"), "samples_ms": "",
            "audio_s": row.get("audio_seconds"), "speed_x": speed, "rtf": None if not speed else 1 / speed, "load_ms": None,
            "rss_mib": row.get("peak_rss_mib"), "gpu_mib": gpu_mib, "gpu_kind": gpu_kind, "gpu_note": gpu_note,
            "wer_pct": wer_pct, "wer_kind": wer_kind, "wer_scope": wer_scope, "wer_scored_outputs": 1 if wer_pct is not None else None,
            "wer_reason": wer_reason, "runs": row.get("runs") or runs, "warmups": warmups, "threads": None,
            "date": run_started(summary)[:10], "date_source": "run start", "started_utc": run.get("started", ""), "finished_utc": run.get("finished", ""),
            "pins": pins_text(summary), "model_revisions": revisions_text(models), "hashes": hashes_text(models),
            "command": command_text(row["family"], families), "config": f"BENCH backends {', '.join(summary.get('backends', []))}; CUDA arch {(summary.get('pins', {}).get('build') or {}).get('cuda_arch', '')}",
            "caveats": caveats, "comparison_limitation": "", "host_conditions": HOST_CONDITIONS.get(host, ""),
            "source_file": summary["_file"], "source_row": row["family"], "raw_source": f"runs/{run_name(summary)}/results/{row['backend']}/{row['family']}.json",
            "group": f"{host}|{row['backend']}|{model}|{task}|{scenario}|{timing_class}|weekly", "ratio_qualifier": "", "run": run_name(summary)}


def newest_cells(host_summaries, families):
    newest_name = run_name(host_summaries[-1])
    cells = {}
    for summary in host_summaries:
        for row in summary.get("results", []):
            cells[(row["backend"], row["family"])] = weekly_row(summary, row, families, newest_name)
    return list(cells.values())


def history_rows(host_summaries, families):
    return [history_row(weekly_row(summary, row, families, run_name(host_summaries[-1])))
            for summary in host_summaries for row in summary.get("results", [])]


def history_row(row):
    keys = ("host", "machine", "run", "date", "started_utc", "backend", "family", "model", "variant", "task", "dashboard", "status",
            "latency_ms", "speed_x", "rtf", "rss_mib", "gpu_mib", "wer_pct", "pins")
    return {k: row.get(k) for k in keys}


def pair_parts(text):
    return [part.strip() for part in (text or "").split(PAIR_SEPARATOR)]


def part_number(part):
    found = RTF_PATTERN.findall(part)
    return float(found[0]) if found else None


def sentence_case(text):
    return text[:1].upper() + text[1:]


def campaign_latency(rtf, seconds):
    return None if rtf is None or seconds is None else round(rtf * seconds * MS_PER_S, 1)


def campaign_rtf(rtf, ratio, is_qvac):
    if is_qvac:
        return rtf
    return None if ratio is None else rtf * ratio


def campaign_row(host, row, index, clip, rtf, ratio, engine, failure=""):
    scenario, scenario_label, seconds = clip
    is_qvac = engine == QVAC
    own_rtf = campaign_rtf(rtf, ratio, is_qvac)
    task = row.get("task", "")
    group = f"{host}|{row.get('backend', '')}|{row.get('model', '')}|{row.get('quantization', '')}|{scenario}|campaign"
    return {"id": f"c-{index}-{scenario}-{'qvac' if is_qvac else engine_slug(engine)}", "host": host, "machine": machine_label(host), "task": task,
            "dashboard": TASK_KEYS.get(task, "asr"), "family": "", "model": row.get("model", ""), "model_detail": row.get("quantization", ""),
            "required_family": False, "engine": engine, "is_qvac": is_qvac, "variant": "", "backend": row.get("backend", ""), "observed_backend": "",
            "precision": row.get("quantization", "") if is_qvac else NOT_RECORDED, "precision_source": "campaign report", "scenario": scenario, "scenario_label": scenario_label,
            "timing_class": CAMPAIGN_TIMING[0], "timing_label": CAMPAIGN_TIMING[1], "timing_includes": row.get("harness", ""), "comparison_set": "campaign",
            "status": "measured" if own_rtf is not None else "failed", "status_detail": "campaign row" if own_rtf is not None else (failure or NO_RATIO),
            "accepted": own_rtf is not None,
            "latency_ms": campaign_latency(own_rtf, seconds), "min_ms": None, "max_ms": None, "samples_ms": "", "audio_s": seconds,
            "speed_x": None if not own_rtf else round(1 / own_rtf, 4), "rtf": own_rtf, "load_ms": None, "rss_mib": None, "gpu_mib": None,
            "gpu_kind": "not recorded", "gpu_note": "The campaign report carries no memory figures.", "wer_pct": None, "wer_kind": "", "wer_scope": "",
            "wer_scored_outputs": None, "wer_reason": "Not part of the campaign rows", "runs": None, "warmups": None, "threads": None,
            "date": row.get("measured", ""), "date_source": "campaign row", "started_utc": "", "finished_utc": "",
            "pins": row.get("qvac_version", "") if is_qvac else row.get("engine_version", ""), "model_revisions": "", "hashes": "", "command": "",
            "config": row.get("device", ""), "caveats": " ".join(x for x in [f"Campaign result: {failure}." if failure else "", row.get("harness", "")] if x),
            "comparison_limitation": f"Campaign of {row.get('measured', '')}: its own QVAC build and inputs, not the weekly run; compare only within this chart.",
            "host_conditions": "", "source_file": "comparisons.json", "source_row": index, "raw_source": row.get("source", ""),
            "group": group, "ratio_qualifier": "qualified: campaign inputs and builds of the measured date; seconds of speech of the TTS prompts are approximate",
            "run": "campaign"}


def engine_slug(engine):
    return re.sub(r"[^a-z0-9]+", "-", engine.lower()).strip("-")


def comparison_rows(host, row, index, seen):
    rows = []
    for clip, qvac_part, ratio_part in zip(CAMPAIGN_CLIPS.get(row.get("task", ""), ()), pair_parts(row.get("qvac_rtf", "")), pair_parts(row.get("ratio", ""))):
        qvac_rtf, ratio = part_number(qvac_part), part_number(ratio_part)
        if qvac_rtf is None:
            continue
        key = (row.get("backend"), row.get("model"), row.get("quantization"), clip[0])
        if key not in seen:
            seen.add(key)
            rows.append(campaign_row(host, row, index, clip, qvac_rtf, None, QVAC))
        rows.append(campaign_row(host, row, index, clip, qvac_rtf, ratio, row.get("engine", ""), "" if ratio is not None else ratio_part))
    return rows


def campaign_rows(host, comparisons):
    rows, seen = [], set()
    for index, row in enumerate(comparisons):
        if CAMPAIGN_HOSTS.get(row.get("device", "")) == host:
            rows += comparison_rows(host, row, index, seen)
    return rows


def report_rows(host, report):
    return [row for row in report.get("rows", []) if row.get("host") == host]


def report_coverage(host, report, start):
    cells = [cell for cell in report.get("coverage", []) if cell.get("host") == host]
    return [dict(cell, id=f"k{start + index:03d}") for index, cell in enumerate(cells)]


def coverage_reason(row):
    if row["status"] == "measured":
        return "Accepted measurement"
    return f"{row['status_detail']}: {row['caveats'] or 'no driver note'}"


def coverage_cell(index, row):
    return {"id": f"k{index:03d}", "host": row["host"], "machine": row["machine"], "task": row["task"], "model": row["model"], "engine": row["engine"],
            "backend": row["backend"], "status": row["status"], "rows": 1, "reason": coverage_reason(row), "source": row["source_file"],
            "scenario": row["scenario_label"]}


def family_fits(spec, newest, backend):
    platform = spec.get("platform")
    backends = spec.get("backends")
    return (not platform or platform == newest.get("machine", {}).get("os")) and (not backends or backend in backends)


SKIPPED_REASON = "No stored result on this machine: SKIP_FAMILIES in its config.env, or the family has not run here yet"


def stored_families(runs):
    return {row["family"] for row in runs if row.get("comparison_set") == "weekly"}


def fits_somewhere(spec, newest):
    return any(family_fits(spec, newest, backend) for backend in newest.get("backends", []))


def skipped_families(runs, families, newest):
    stored = stored_families(runs)
    return [family for family, spec in families.items() if family not in stored and fits_somewhere(spec, newest)]


def skipped_cell(newest, family, spec, backend, number):
    return {"id": f"k{number:03d}", "host": newest["_host"], "machine": machine_label(newest["_host"]), "task": spec.get("task", ""),
            "model": spec.get("model_label", family), "engine": spec.get("engine") or QVAC, "backend": backend, "status": "not attempted",
            "rows": 0, "reason": SKIPPED_REASON, "source": newest["_file"], "scenario": scenario_of(spec)[1]}


def stored_cells(runs):
    return {(row["backend"], row["family"]) for row in runs if row.get("comparison_set") == "weekly"}


def skipped_pairs(runs, families, newest):
    stored = stored_cells(runs)
    return [(family, backend) for family in families for backend in newest.get("backends", [])
            if (backend, family) not in stored and family_fits(families[family], newest, backend)]


def skipped_cells(host_summaries, runs, families, start):
    newest = host_summaries[-1]
    return [skipped_cell(newest, family, families[family], backend, start + i) for i, (family, backend) in enumerate(skipped_pairs(runs, families, newest))]


def engine_gap_cell(host, seed, gap, number):
    return {"id": f"k{number:03d}", "host": host, "machine": machine_label(host), "task": gap.get("task", ""), "model": gap.get("model", ""),
            "engine": seed["engine"], "backend": gap.get("backend", ""), "status": "not attempted", "rows": 0, "reason": gap.get("reason", ""),
            "source": ENGINES_FILE, "scenario": ""}


def engine_gaps(host, engines, start):
    gaps = [(seed, gap) for seed in engines for gap in seed.get("not_measured", []) if gap.get("host") == host]
    return [engine_gap_cell(host, seed, gap, start + i) for i, (seed, gap) in enumerate(gaps)]


def coverage_rows(host_summaries, runs, families, report, engines):
    cells = [coverage_cell(i, row) for i, row in enumerate(runs)]
    cells += skipped_cells(host_summaries, runs, families, len(cells))
    cells += report_coverage(host_summaries[-1]["_host"], report, len(cells))
    return cells + engine_gaps(host_summaries[-1]["_host"], engines, len(cells))


def machines_of(engine, rows_by_host):
    return sorted({machine_label(h) for h, rows in rows_by_host.items() if any(r["engine"] == engine and r["accepted"] for r in rows)})


def engine_rows(engines, rows_by_host):
    out = []
    for seed in engines:
        measured_on = machines_of(seed["engine"], rows_by_host)
        out.append({"engine": seed["engine"], "role": seed.get("role", ""), "status": "measured" if measured_on else "not measured",
                    "ggml": seed.get("ggml", ""), "measured_on": ", ".join(measured_on), "details": f"{seed.get('ggml', '')}. {seed.get('details', '')}".strip(". "),
                    "source": seed.get("source", "")})
    return out


def hardware_text(summary):
    machine = summary.get("machine", {})
    gpu = machine.get("gpu") or {}
    parts = [machine.get("os_version") or machine.get("os", ""), machine.get("cpu", ""),
             f"{machine.get('memory_gib', '')} GiB" if machine.get("memory_gib") else "", gpu.get("name", ""),
             f"driver {gpu.get('driver')}" if gpu.get("driver") else "", f"CUDA {gpu.get('cuda_toolkit')}" if gpu.get("cuda_toolkit") else ""]
    return " · ".join(p for p in parts if p)


def machine_row(host_summaries, runs):
    newest = host_summaries[-1]
    run, counts = newest.get("run", {}), newest.get("counts", {})
    weekly = [r for r in runs if r["comparison_set"] == "weekly"]
    accepted = sum(1 for r in runs if r["accepted"])
    campaign_text = "; ".join(copied_rows_text(runs, name) for name in ("campaign", REPORT))
    return [{"host": newest["_host"], "machine": machine_label(newest["_host"]), "hardware": hardware_text(newest), "accepted": accepted,
             "failed": len(runs) - accepted, "superseded": 0, "backends": ", ".join(newest.get("backends", [])),
             "status": f"Newest run {run.get('name', '')} {run.get('status', '')}: {counts.get('ok', 0)} of {counts.get('expected', 0)} cells ok; {len(host_summaries)} stored runs",
             "work": f"{sum(1 for r in weekly if r['accepted'])} weekly rows at {pins_text(newest)}; {campaign_text}",
             "conditions": HOST_CONDITIONS.get(newest["_host"], ""), "snapshot_utc": run.get("finished", "")}]


def copied_rows_text(runs, name):
    rows = [r for r in runs if r["comparison_set"] == name]
    if not rows:
        return f"no {name} rows for this machine"
    return f"{len(rows)} {name} rows from {', '.join(sorted({r['date'] for r in rows}))}"


def fmt_speed(value):
    return f"{value:,.0f}×" if value >= 100 else f"{value:.1f}×" if value >= 10 else f"{value:.2f}×"


def fmt_ms(value):
    return f"{value / MS_PER_S:,.1f} s" if value >= MS_SECONDS_FROM else f"{value:,.0f} ms"


def join_words(words):
    words = list(words)
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def fastest_cell(runs, task):
    rows = [r for r in runs if r["comparison_set"] == "weekly" and r["task"] == task and r["accepted"] and r["speed_x"]]
    return max(rows, key=lambda r: r["speed_x"]) if rows else None


def cell_text(row):
    audio = f" for {row['audio_s']:.1f} s of audio" if row["audio_s"] else ""
    return f"{row['model']} ({row['variant']}) on {backend_name(row['backend'])}, {fmt_speed(row['speed_x'])} real time, {fmt_ms(row['latency_ms'])}{audio}"


def fastest_items(runs):
    items, parts = [], []
    for task in TASK_ORDER:
        best = fastest_cell(runs, task)
        if best is None:
            continue
        items.append((TASK_KEYS[task], f"Fastest {TASK_NAMES[task]} cell", f"{cell_text(best)}."))
        parts.append(f"{sentence_case(TASK_NAMES[task])}: {cell_text(best)}")
    if parts:
        items.insert(0, (SUMMARY, "Fastest cells this week", "; ".join(parts) + "."))
    return items


def campaign_groups(runs):
    groups = {}
    for row in runs:
        if row["comparison_set"] == "campaign" and row["accepted"] and row["rtf"]:
            groups.setdefault(row["group"], []).append(row)
    return groups.values()


def campaign_pairs(runs):
    pairs = []
    for rows in campaign_groups(runs):
        qvac = [r for r in rows if r["is_qvac"]]
        if len(qvac) == 1:
            pairs += [(r, r["rtf"] / qvac[0]["rtf"]) for r in rows if not r["is_qvac"]]
    return pairs


def campaign_text(task, pairs):
    ratios = [ratio for _, ratio in pairs]
    engines = sorted({row["engine"] for row, _ in pairs})
    dates = sorted({row["date"] for row, _ in pairs})
    faster = sum(1 for ratio in ratios if ratio > 1)
    return (f"{sentence_case(TASK_NAMES[task])} campaign of {', '.join(dates)} against {join_words(engines)}: competitor ÷ QVAC time {ratio_range(ratios)} "
            f"over {pairs_text(len(ratios))}; QVAC took less time in {faster} of {len(ratios)}.")


def ratio_range(ratios):
    low, high = f"{min(ratios):.2f}", f"{max(ratios):.2f}"
    return low if low == high else f"{low} to {high}"


def pairs_text(count):
    return "1 pair" if count == 1 else f"{count} pairs"


def campaign_items(runs):
    items = []
    pairs = campaign_pairs(runs)
    for task in TASK_ORDER:
        task_pairs = [(row, ratio) for row, ratio in pairs if row["task"] == task]
        if task_pairs:
            text = campaign_text(task, task_pairs)
            items += [(SUMMARY, "Against other engines", text), (TASK_KEYS[task], "Against other engines", text)]
    return items


def task_keys():
    return list(dict.fromkeys(TASK_KEYS[task] for task in TASK_ORDER))


def failure_items(runs):
    failed = [r for r in runs if r["comparison_set"] == "weekly" and not r["accepted"]]
    items = []
    for key in task_keys():
        cells = [f"{r['model']} ({r['variant']}) on {backend_name(r['backend'])}" for r in failed if r["dashboard"] == key]
        if cells:
            items.append((key, "Failed cells", f"{join_words(cells)}. The driver notes are in the Runs and coverage tab."))
    if failed:
        items.insert(0, (SUMMARY, "Failed cells", f"{len(failed)} weekly cells failed in the newest result; each task tab and the Runs and coverage tab name them with the driver note."))
    return items


def contention_items(runs):
    under_load = [r for r in runs if r["comparison_set"] == "weekly" and r.get("contended")]
    items = []
    for key in task_keys():
        cells = [f"{r['model']} ({r['variant']}) on {backend_name(r['backend'])}, {r['busy_before_pct']:.0f}% busy" for r in under_load if r["dashboard"] == key]
        if cells:
            items.append((key, "Cells under load", f"{join_words(cells)}: other work was running on the machine just before the cell, so the time can be slower than the machine's own."))
    if under_load:
        items.insert(0, (SUMMARY, "Cells under load", f"{len(under_load)} weekly cells were measured while other work kept the CPUs busy; each task tab names them with the busy share."))
    return items


def baseline_runs(host_summaries):
    return [s for s in host_summaries[:-1] if QUICK_MARK not in run_name(s)]


def previous_values(host_summaries):
    baselines = baseline_runs(host_summaries)
    if not baselines:
        return {}
    return {(r["backend"], r["family"]): r.get("speed_x") for r in baselines[-1].get("results", []) if r.get("speed_x")}


def moves(host_summaries, runs):
    previous = previous_values(host_summaries)
    found = []
    for row in runs:
        before = previous.get((row["backend"], row["family"]))
        if row["comparison_set"] != "weekly" or not row["speed_x"] or not before:
            continue
        change = row["speed_x"] / before - 1
        if abs(change) > MOVE_THRESHOLD:
            found.append(f"{row['model']} ({row['variant']}) on {backend_name(row['backend'])} {change:+.0%} ({fmt_speed(before)} to {fmt_speed(row['speed_x'])})")
    return found


def move_items(host_summaries, runs):
    baselines = baseline_runs(host_summaries)
    if not baselines:
        return [(SUMMARY, "Week over week", "First full run of this machine: no earlier full run to compare with (quick runs are not baselines).")]
    found = moves(host_summaries, runs)
    if not found:
        return [(SUMMARY, "Week over week", f"No cell moved more than {MOVE_THRESHOLD:.0%} against the previous full run ({run_name(baselines[-1])}).")]
    return [(SUMMARY, "Week over week", f"Moves over {MOVE_THRESHOLD:.0%} against the previous full run ({run_name(baselines[-1])}): {'; '.join(found)}.")]


def coverage_item(host_summaries, runs, families):
    newest = host_summaries[-1]
    weekly = [r for r in runs if r["comparison_set"] == "weekly"]
    measured = sum(1 for r in weekly if r["accepted"])
    skipped = sorted({families[f].get("model_label", f) for f in skipped_families(runs, families, newest)})
    skipped_text = f" No stored result on this machine for {join_words(skipped)} (SKIP_FAMILIES in its config.env, or not run here yet)." if skipped else ""
    backends = join_words([backend_name(b) for b in newest.get("backends", [])]) if newest.get("backends") else "no backend"
    return (SUMMARY, "Coverage", f"{measured} of {len(weekly)} weekly cells measured on {backends}.{skipped_text}")


def pins_item(newest):
    cuda_arch = (newest.get("pins", {}).get("build") or {}).get("cuda_arch", "") or "n/a"
    return (SUMMARY, "Pins", f"Newest run {run_name(newest)} at {pins_text(newest)}; CUDA arch {cuda_arch}.")


def report_item(runs):
    rows = [r for r in runs if r["comparison_set"] == REPORT and r["accepted"]]
    if not rows:
        return []
    engines = join_words(sorted({r["engine"] for r in rows if not r["is_qvac"]}))
    dates = join_words(sorted({r["date"] for r in rows}))
    return [(SUMMARY, "Comparison report", f"{len(rows)} rows copied from the comparison report of {dates}: QVAC against {engines} with that report's own builds, inputs and timing scopes; its charts are never mixed with the weekly ones.")]


def commentary_rows(host_summaries, runs, families):
    newest = host_summaries[-1]
    items = fastest_items(runs) + campaign_items(runs) + report_item(runs) + failure_items(runs) + contention_items(runs) + move_items(host_summaries, runs)
    items += [coverage_item(host_summaries, runs, families), pins_item(newest)]
    items += [(GENERAL, topic, text) for topic, text in RULES]
    sources = {GENERAL: "comparison rules of the weekly bench"}
    return [{"id": f"c{i:02d}", "dashboard": key, "topic": topic, "comment": text, "scope": "", "source": sources.get(key, newest["_file"])}
            for i, (key, topic, text) in enumerate(items)]


def quality_flags(rows_by_host):
    flagged = [r for rows in rows_by_host.values() for r in rows
               if r["accepted"] and r["task"] in ("TTS", "TTS cloning") and r["wer_pct"] is not None and r["wer_pct"] >= QUALITY_FLAG_PCT]
    return [{"engine": r["engine"], "model": r["model"], "variant": r["variant"], "machine": r["machine"], "backend": r["backend"], "task": r["task"],
             "wer_pct": r["wer_pct"], "speed_x": r["speed_x"], "wer_kind": r["wer_kind"]} for r in sorted(flagged, key=lambda r: -r["wer_pct"])]


def hub_task(rows, task_key):
    task_rows = [r for r in rows if r["dashboard"] == task_key and r["accepted"]]
    return {"models": len({r["model"] for r in task_rows}), "backends": sorted({r["backend"] for r in task_rows}),
            "others": len({r["engine"] for r in task_rows if not r["is_qvac"]}), "rows": len(task_rows)}


def hub_machine(host, host_summaries, rows, dashboards):
    newest = host_summaries[-1]
    return {"host": host, "machine": machine_label(host), "url": dashboards.get(host, ""), "hardware": hardware_text(newest),
            "newest_run": run_name(newest), "date": run_started(newest)[:10], "status": newest.get("run", {}).get("status", ""),
            "ok": newest.get("counts", {}).get("ok", 0), "expected": newest.get("counts", {}).get("expected", 0), "pins": pins_text(newest),
            "accepted": sum(1 for r in rows if r["accepted"]), "tasks": {key: hub_task(rows, key) for key in ("asr", "tts", "music")}}


def hub_document(summaries_by_host, rows_by_host, engines, dashboards):
    machines = [hub_machine(host, summaries_by_host[host], rows_by_host[host], dashboards) for host in summaries_by_host]
    newest = max(m["date"] for m in machines)
    return {"measured": newest, "machines": machines, "accepted": sum(m["accepted"] for m in machines),
            "quality_flags": quality_flags(rows_by_host), "engines": engine_rows(engines, rows_by_host), "hub_url": dashboards.get("landing", "")}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")


def group_by_host(summaries):
    by_host = {}
    for summary in summaries:
        by_host.setdefault(summary["_host"], []).append(summary)
    return by_host


def machine_files(host_summaries, runs, engines, rows_by_host, families, report):
    return {"runs": runs, "history": history_rows(host_summaries, families), "coverage": coverage_rows(host_summaries, runs, families, report, engines),
            "engines": engine_rows(engines, rows_by_host), "machine": machine_row(host_summaries, runs),
            "commentary": commentary_rows(host_summaries, runs, families)}


def write_machine_files(folder, files):
    for name, rows in files.items():
        write_rows(folder / f"{name}.json", rows)
    return {name: len(rows) for name, rows in files.items()}


def build(summaries, comparisons, engines, families, dashboards, out_dir, report=None):
    report = report or {}
    by_host = group_by_host(summaries)
    rows_by_host = {host: newest_cells(host_summaries, families) + campaign_rows(host, comparisons) + report_rows(host, report)
                    for host, host_summaries in by_host.items()}
    written = {}
    for host, host_summaries in by_host.items():
        written[host] = write_machine_files(out_dir / host, machine_files(host_summaries, rows_by_host[host], engines, rows_by_host, families, report))
    write_rows(out_dir / "hub.json", hub_document(by_host, rows_by_host, engines, dashboards))
    return written


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--comparisons", default=str(HERE / "comparisons.json"))
    parser.add_argument("--engines", default=str(HERE / "engines.json"))
    parser.add_argument("--families", default=str(HERE / "families.json"))
    parser.add_argument("--dashboards", default=str(HERE / "dashboards.json"))
    parser.add_argument("--report", default="", help=f"rows written by import-report.py; default: {REPORT_FILE} next to the summaries folder")
    return parser.parse_args(argv)


def report_path(args):
    return args.report or str(pathlib.Path(args.summaries).resolve().parent / REPORT_FILE)


def load_report(path):
    return load_json(path) if pathlib.Path(path).is_file() else {}


def main(argv=None):
    args = parse_args(argv)
    summaries = load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"no summary.json files under {args.summaries}")
    dashboards = load_json(args.dashboards) if pathlib.Path(args.dashboards).is_file() else {}
    written = build(summaries, load_rows(args.comparisons), load_rows(args.engines), load_families(args.families), dashboards, pathlib.Path(args.out),
                    load_report(report_path(args)))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} wrote {json.dumps(written)} and hub.json under {args.out}")


if __name__ == "__main__":
    main()
