#!/usr/bin/env python3
"""Build the Speech Bench highlights page: a few GPU charts of the hub as cards.

usage: build-highlights.py <dashboard-data folder> [--config highlights.json] -o highlights.html
"""
import argparse
import json
import pathlib
import sys

import task_views as views

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "highlights.json"
RUNS_FILE = "runs.json"
MACHINE_FILE = "machine.json"
HUB_FILE = "hub.json"
CARD_KEYS = ("task", "model", "scenario", "timing_class")
QVAC_LABEL = "QVAC"
PERCENT = 100.0
MS_PER_S = 1000.0
TASK_CLASSES = {"ASR": "t-asr", "TTS": "t-tts", "TTS cloning": "t-tts", "Music": "t-music"}
TASK_SHORT = {"ASR": "ASR", "TTS": "TTS", "TTS cloning": "Voice cloning", "Music": "Music"}
SET_SOURCES = {views.WEEKLY: "the Sunday weekly run of the engine master", views.REPORT: "the comparison report of 2026-11-04 (its own builds and inputs)",
               views.CAMPAIGN: "the dated comparison campaign"}
RTF_KEY = "RTF (real-time factor), lower is better:"
RTF_TEXT = "processing time divided by the duration of the audio. RTF 0.01 means one second of audio takes 10 ms."
BARS_TEXT = "Bars show each engine's median RTF. The number under each other engine is its RTF divided by QVAC's."
STYLE = """
:root {
  --bg: #f3f5f7; --surface: #fcfdfd; --ink: #13161b; --ink-2: #4b525c; --muted: #6f7782; --rule: #d8dde3; --track: #e7ebef; --focus: #2a6fd6;
  --cuda: #4d8a12; --vulkan: #b9303b; --cpu: #4f52c9; --metal: #c45a22; --bad: #b9303b;
  --t-asr: #2f6fdb; --t-tts: #1c9e8a; --t-music: #b04fc7;
  --e-qvac: #2f6fdb; --e-audiocpp: #e0663a; --e-acestep: #1c9e8a; --e-fw: #8b5cf6; --e-vllm: #d18a14; --e-sglang: #d14b8f;
  --e-moss: #7d8f1a; --e-cosy: #a8733a; --e-ttscpp: #4f9a3c; --e-minimax: #b04fc7; --e-localai: #2aa5c9; --e-transcribe: #b5951a; --e-nemo: #4a5d70; --e-whispercpp: #c93d3d; --e-mlx: #0f8f7a; --e-fluid: #8c5a3c; --e-other: #6f7782;
  --font-display: "Barlow Semi Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --font-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #101317; --surface: #171b21; --ink: #e8ebef; --ink-2: #b4bbc4; --muted: #8b939d; --rule: #2a3038; --track: #232931; --focus: #6ea4f5;
    --cuda: #8cc455; --vulkan: #ef6b74; --cpu: #9b9ef2; --metal: #ec8a52; --bad: #ef6b74;
    --t-asr: #5b8ff0; --t-tts: #33b8a2; --t-music: #c879dc;
    --e-qvac: #5b8ff0; --e-audiocpp: #ef7f55; --e-acestep: #33b8a2; --e-fw: #a283f8; --e-vllm: #e8a73a; --e-sglang: #e46aa6;
    --e-moss: #a3b83a; --e-cosy: #c99257; --e-ttscpp: #6fbb5b; --e-minimax: #c879dc; --e-localai: #4cc0e2; --e-transcribe: #d6b53c; --e-nemo: #8a9db2; --e-whispercpp: #e86a6a; --e-mlx: #3cc2a9; --e-fluid: #c08a62; --e-other: #8b939d;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #101317; --surface: #171b21; --ink: #e8ebef; --ink-2: #b4bbc4; --muted: #8b939d; --rule: #2a3038; --track: #232931; --focus: #6ea4f5;
  --cuda: #8cc455; --vulkan: #ef6b74; --cpu: #9b9ef2; --metal: #ec8a52; --bad: #ef6b74;
  --t-asr: #5b8ff0; --t-tts: #33b8a2; --t-music: #c879dc;
  --e-qvac: #5b8ff0; --e-audiocpp: #ef7f55; --e-acestep: #33b8a2; --e-fw: #a283f8; --e-vllm: #e8a73a; --e-sglang: #e46aa6;
  --e-moss: #a3b83a; --e-cosy: #c99257; --e-ttscpp: #6fbb5b; --e-minimax: #c879dc; --e-localai: #4cc0e2; --e-transcribe: #d6b53c; --e-nemo: #8a9db2; --e-whispercpp: #e86a6a; --e-mlx: #3cc2a9; --e-fluid: #c08a62; --e-other: #8b939d;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font: 400 15px/1.55 var(--font-body); }
.wrap { max-width: 1180px; margin-inline: auto; padding-block: 40px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 44px; }
header { display: flex; flex-direction: column; gap: 10px; max-width: 74ch; }
.eyebrow { font: 500 0.72rem/1.3 var(--font-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.eyebrow a { color: inherit; }
h1 { margin: 0; font: 600 clamp(2rem, 4.4vw, 2.9rem)/1.05 var(--font-display); letter-spacing: -0.01em; text-wrap: balance; }
h2 { margin: 0; font: 600 clamp(1.5rem, 3vw, 2rem)/1.1 var(--font-display); text-wrap: balance; }
h3 { margin: 0; font: 600 1.25rem/1.15 var(--font-display); text-wrap: balance; }
p { margin: 0; }
.lede { color: var(--ink-2); font-size: 1.02rem; }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--ink-2); }
.machine { display: flex; flex-direction: column; gap: 16px; }
.mh { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; border-top: 2px solid var(--ink); padding-top: 14px; }
.mh p { color: var(--ink-2); font-size: 0.92rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-flex; padding: 2px 9px; border-radius: 999px; font: 500 0.7rem/1.4 var(--font-mono); letter-spacing: 0.06em; text-transform: uppercase; border: 1px solid currentColor; }
.b-cuda { color: var(--cuda); } .b-vulkan { color: var(--vulkan); } .b-metal { color: var(--metal); } .b-cpu { color: var(--cpu); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.ch { display: flex; flex-direction: column; gap: 4px; }
.task { align-self: flex-start; font: 500 0.68rem/1.4 var(--font-mono); letter-spacing: 0.08em; text-transform: uppercase; }
.t-asr { color: var(--t-asr); } .t-tts { color: var(--t-tts); } .t-music { color: var(--t-music); }
.cap { color: var(--muted); font-size: 0.82rem; }
.win { font-size: 0.92rem; color: var(--ink-2); }
.win b { font: 600 1.9rem/1 var(--font-display); color: var(--e-qvac); margin-right: 4px; font-variant-numeric: tabular-nums; }
.win.loss b { color: var(--bad); }
.sub { font: 400 0.8rem/1.3 var(--font-mono); color: var(--muted); }
.bars { display: flex; flex-direction: column; gap: 5px; }
.row { display: grid; grid-template-columns: minmax(0, 9.5em) minmax(0, 1fr) auto; gap: 8px; align-items: center; font-size: 0.82rem; }
.row .lab { color: var(--ink-2); overflow-wrap: anywhere; line-height: 1.2; }
.row.q .lab { color: var(--ink); font-weight: 600; }
.track { height: 12px; background: var(--track); border-radius: 3px; overflow: hidden; }
.bar { display: block; height: 100%; border-radius: 3px; }
.val { font: 400 0.78rem/1.2 var(--font-mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; color: var(--ink-2); }
.val small { display: block; color: var(--muted); font-size: 0.7rem; }
.row.q .val { color: var(--ink); font-weight: 500; }
.e-qvac { background: var(--e-qvac); } .e-audiocpp { background: var(--e-audiocpp); } .e-acestep { background: var(--e-acestep); } .e-minimax { background: var(--e-minimax); }
.e-localai { background: var(--e-localai); } .e-transcribe { background: var(--e-transcribe); } .e-nemo { background: var(--e-nemo); } .e-whispercpp { background: var(--e-whispercpp); }
.e-fw { background: var(--e-fw); } .e-vllm { background: var(--e-vllm); } .e-sglang { background: var(--e-sglang); } .e-moss { background: var(--e-moss); } .e-cosy { background: var(--e-cosy); }
.e-ttscpp { background: var(--e-ttscpp); } .e-mlx { background: var(--e-mlx); } .e-fluid { background: var(--e-fluid); } .e-other { background: var(--muted); }
.note { font-size: 0.8rem; color: var(--muted); }
.more { margin-top: auto; font-size: 0.85rem; color: var(--focus); text-decoration: none; }
.more:hover { text-decoration: underline; }
a:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 3px; }
footer { display: flex; flex-direction: column; gap: 8px; max-width: 80ch; color: var(--ink-2); font-size: 0.88rem; border-top: 1px solid var(--rule); padding-top: 16px; }
footer ul { margin: 0; padding-left: 1.1em; display: flex; flex-direction: column; gap: 4px; }
"""
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')

esc = views.esc


def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_machine(folder, host):
    machines = load_json(pathlib.Path(folder) / host / MACHINE_FILE)
    return machines[0] if machines else {"host": host, "machine": host, "hardware": ""}


def rtf_of(row):
    if row.get("latency_ms") and row.get("audio_s"):
        return row["latency_ms"] / MS_PER_S / row["audio_s"]
    return row.get("rtf") or 0.0


def chartable(row):
    return views.is_accepted(row) and not row.get("no_chart") and rtf_of(row) > 0


def matches(row, machine, card):
    return (row.get("host") == machine["host"] and row.get("backend") == machine["backend"] and row.get("comparison_set") == card["set"]
            and all(row.get(key) == card[key] for key in CARD_KEYS))


def card_rows(rows, machine, card):
    return [row for row in rows if chartable(row) and matches(row, machine, card)]


def groups_of(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    return groups


def qvac_rows(rows):
    return [row for row in rows if row.get("is_qvac")]


def group_has_variant(rows, variant):
    return any(row.get("variant", "") == variant for row in qvac_rows(rows))


def group_with_variant(groups, variant):
    return next((rows for rows in groups.values() if group_has_variant(rows, variant)), [])


def pick_group(groups, card):
    if "variant" in card:
        return group_with_variant(groups, card["variant"])
    return group_with_variant(groups, "") or next(iter(groups.values()), [])


def select_rows(rows, machine, card):
    return pick_group(groups_of(card_rows(rows, machine, card)), card)


def fastest(rows):
    return min(rows, key=rtf_of) if rows else None


def competitors(rows):
    return sorted((row for row in rows if not row.get("is_qvac")), key=rtf_of)


def fmt_rtf(value):
    return f"{value:.3g}"


def fmt_factor(value):
    return f"{value:.1f}×"


def fmt_seconds(ms):
    return f"{ms:.0f} ms" if ms < MS_PER_S else f"{ms / MS_PER_S:.3g} s"


def factor_of(qvac, rival):
    return rtf_of(rival) / rtf_of(qvac)


def qvac_label(row, weekly):
    return f"{QVAC_LABEL} {row['variant']}" if weekly and row.get("variant") else QVAC_LABEL


def win_html(qvac, rivals):
    factor = factor_of(qvac, rivals[0])
    if factor >= 1:
        return f'<p class="win"><b>{esc(fmt_factor(factor))}</b> lower RTF than the fastest other engine</p>'
    return f'<p class="win loss"><b>{esc(fmt_factor(1 / factor))}</b> higher RTF than {esc(views.engine_name(rivals[0]))}, which leads here</p>'


def sub_html(qvac):
    return (f'<p class="sub">{QVAC_LABEL} RTF {esc(fmt_rtf(rtf_of(qvac)))} ({esc(fmt_seconds(qvac["latency_ms"]))} for '
            f'{esc(f"{qvac["audio_s"]:.1f}")} s of audio) · lower is better</p>')


def bar_width(row, top):
    return rtf_of(row) / top * PERCENT


def bar_html(row, top, qvac):
    own = bool(row.get("is_qvac"))
    ratio = "" if own else f'<small>{esc(fmt_factor(factor_of(qvac, row)))} {QVAC_LABEL}</small>'
    label = qvac_label(row, row.get("comparison_set") == views.WEEKLY) if own else views.engine_name(row)
    return (f'<div class="row{" q" if own else ""}"><span class="lab">{esc(label)}</span><span class="track">'
            f'<span class="bar {views.engine_class(row["engine"])}" style="width:{bar_width(row, top):.2f}%"></span></span>'
            f'<span class="val">RTF {esc(fmt_rtf(rtf_of(row)))}{ratio}</span></div>')


def own_rows(rows):
    return sorted(qvac_rows(rows), key=rtf_of)


def bars_html(qvac, rivals, rows):
    drawn = own_rows(rows) + rivals
    top = max(rtf_of(row) for row in drawn)
    bars = "".join(bar_html(row, top, qvac) for row in drawn)
    return f'<div class="bars" role="img" aria-label="Real-time factor per engine, lower is better">{bars}</div>'


def note_html(qvac):
    qualifier = qvac.get("ratio_qualifier", "")
    return f'<p class="note">{esc(qualifier)}</p>' if qualifier else ""


def caption(row):
    return f"{views.scenario_short(row)} · {views.timing_short(row)} · {views.set_label([row])}"


def link_html(machine, row, hub_url):
    token = views.view_token(machine["machine"], row.get("dashboard", "asr"))
    return (f'<a class="more" href="{esc(hub_url)}#{esc(token)}" target="_blank" rel="noopener">Full {esc(machine["machine"])} '
            f'{esc(TASK_SHORT.get(row["task"], row["task"]))} results</a>')


def card_html(machine, rows, hub_url):
    qvac, rivals = fastest(qvac_rows(rows)), competitors(rows)
    return (f'<article class="card"><div class="ch"><span class="task {TASK_CLASSES.get(qvac["task"], "t-asr")}">{esc(TASK_SHORT.get(qvac["task"], qvac["task"]))}</span>'
            f'<h3>{esc(qvac["model"])}</h3><p class="cap">{esc(caption(qvac))}</p></div>{win_html(qvac, rivals)}{sub_html(qvac)}'
            f'{bars_html(qvac, rivals, rows)}{note_html(qvac)}{link_html(machine, qvac, hub_url)}</article>')


def card_name(machine, card):
    return f'{machine["host"]} {machine["backend"]} {card["task"]} {card["model"]} {card["scenario"]} {card["timing_class"]} {card["set"]}'


def usable(rows):
    return bool(qvac_rows(rows)) and bool(competitors(rows))


def machine_cards(machine, rows, hub_url, skipped):
    cards = []
    for card in machine["cards"]:
        selected = select_rows(rows, machine, card)
        if usable(selected):
            cards.append(card_html(machine, selected, hub_url))
        else:
            skipped.append(card_name(machine, card))
    return cards


def machine_header(machine):
    backend = machine["backend"]
    return (f'<div class="mh"><h2>{esc(machine["machine"])}</h2><p><span class="chip b-{esc(backend)}">{esc(views.backend_label(backend))}</span> '
            f'{esc(machine.get("hardware", ""))}</p></div>')


def machine_html(machine, cards):
    return f'<section class="machine">{machine_header(machine)}<div class="cards">{"".join(cards)}</div></section>'


def timing_bullets(rows):
    seen = {}
    for row in rows:
        seen.setdefault(row["task"], views.timing_short(row))
    return "".join(f"<li>{esc(TASK_SHORT.get(task, task))} is timed as {esc(text)}.</li>" for task, text in seen.items())


def set_bullets(rows):
    names = views.ordered(views.SET_ORDER, {row["comparison_set"] for row in rows})
    return "".join(f"<li>Charts labelled {esc(name)} come from {esc(SET_SOURCES.get(name, name))}.</li>" for name in names)


def footer_html(rows, hub_url):
    return (f'<footer><p><b>How these were measured</b></p><ul><li>RTF is processing seconds per second of audio, from the median of repeated runs, so lower is better.</li>'
            f'{timing_bullets(rows)}{set_bullets(rows)}<li>This page is a selection. The <a href="{esc(hub_url)}" target="_blank" rel="noopener">full Speech Bench hub</a> '
            f'covers the CPU backends, every model and the cases where another engine is faster.</li></ul></footer>')


def header_html(config, hub):
    return (f'<header><div class="eyebrow"><a href="{esc(hub.get("hub_url", ""))}" target="_blank" rel="noopener">Speech Bench</a> · highlights · measured {esc(hub.get("measured", ""))}</div>'
            f'<h1>{esc(config["title"])}</h1><p class="lede">{esc(config["lede"])}</p>'
            f'<div class="key"><span><b>{RTF_KEY}</b> {RTF_TEXT}</span><span>{BARS_TEXT}</span></div></header>')


def page_html(config, hub, sections, rows):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(config["title"])}</title>{FONTS}<style>{STYLE}</style></head><body><div class="wrap">{header_html(config, hub)}'
            f'{"".join(sections)}{footer_html(rows, hub.get("hub_url", ""))}</div></body></html>\n')


def machine_rows(folder, host):
    return load_json(pathlib.Path(folder) / host / RUNS_FILE)


def build_machine(folder, entry, hub_url, skipped):
    machine = {**load_machine(folder, entry["host"]), **entry}
    rows = machine_rows(folder, entry["host"])
    cards = machine_cards(machine, rows, hub_url, skipped)
    drawn = [row for card in machine["cards"] for row in select_rows(rows, machine, card)]
    return (machine_html(machine, cards) if cards else "", drawn)


def machine_has_data(folder, host):
    return (pathlib.Path(folder) / host / MACHINE_FILE).is_file()


def build(folder, config):
    hub = load_json(pathlib.Path(folder) / HUB_FILE)
    sections, drawn, skipped = [], [], []
    for entry in config["machines"]:
        if not machine_has_data(folder, entry["host"]):
            skipped.append(f"{entry['host']} (no data for this machine)")
            continue
        section, rows = build_machine(folder, entry, hub.get("hub_url", ""), skipped)
        sections.append(section)
        drawn.extend(rows)
    return page_html(config, hub, sections, drawn), skipped


def report_skipped(skipped):
    for name in skipped:
        print(f"no chart with QVAC and another engine for: {name}", file=sys.stderr)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="dashboard-data folder written by build-dashboard-data.py")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("-o", "--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    page, skipped = build(args.data, load_json(args.config))
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    report_skipped(skipped)


if __name__ == "__main__":
    main()
