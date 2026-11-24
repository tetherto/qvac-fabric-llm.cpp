#!/usr/bin/env python3
"""Write the Speech Bench hub page of the weekly speech bench from hub.json and the per-machine data files.

usage: build-landing.py <hub.json> -o landing.html [--data <dashboard data dir>]

The hub lists, per task, every machine with a link to its task page, then the quality flags, the engines with where
they were measured, what each page shows and how the bench runs. The task pages (task_views.py) are views of the same
page, reached at the hub's link plus #<machine>-<task> (for example #dgx-spark-tts), so they need no artifact of their
own. --data is the folder build-dashboard-data.py wrote, with <host>/runs.json and <host>/coverage.json; it defaults to
the folder of hub.json.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
STYLE_FILES = (HERE / "dashboard" / "hub.css", HERE / "dashboard" / "task.css")
MACHINE_ORDER = ["DGX Spark", "RTX 5090", "Strix Halo", "Jetson Orin", "Mac Studio M3 Ultra", "Mac mini M4"]
TASKS = [("tts", "TTS and voice cloning", "Speech synthesis from a fixed English prompt and voice cloning from the JFK clip: Supertonic 3, the Audio8 LM and codec cuts, MOSS-TTS."),
         ("asr", "ASR", "Speech to text on the 11 s JFK clip with its reference transcript: Parakeet TDT and Unified, TranscribePsy, Nemotron 3.5 streaming."),
         ("music", "Music generation", "A 30 s render as a fresh process including model load: ACE-Step 1.5 Turbo, SFT and Base.")]
QUALITY_FLAG_PCT = 10
NO_ROWS = "no accepted row this week"
HUB_VIEW = "hub"
EXAMPLE_PAGE = ("DGX Spark", "tts")
ROUTER = """<script>
(function () {
  const views = Array.from(document.querySelectorAll('[data-view]'));
  const hub = views[0];
  function tokenOf(hash) { try { return decodeURIComponent((hash || '').replace(/^#/, '')); } catch (error) { return ''; } }
  function viewFor(token) { return views.find((view) => view.dataset.view === token) || hub; }
  function reveal(shown) { views.forEach((view) => { view.hidden = view !== shown; }); }
  function place(shown, token) {
    const anchor = shown === hub && token ? document.getElementById(token) : null;
    if (anchor) { anchor.scrollIntoView(); } else { window.scrollTo(0, 0); }
  }
  function show(token) { const shown = viewFor(token); reveal(shown); place(shown, token); }
  function remember(token) { try { history.pushState(null, '', '#' + token); } catch (error) { return; } }
  function follow(event) {
    const link = event.target.closest('a[href^="#"]');
    if (!link) { return; }
    event.preventDefault();
    const token = tokenOf(link.getAttribute('href'));
    remember(token);
    show(token);
  }
  document.addEventListener('click', follow);
  window.addEventListener('popstate', () => show(tokenOf(location.hash)));
  window.addEventListener('hashchange', () => show(tokenOf(location.hash)));
  show(tokenOf(location.hash));
})();
</script>"""


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


views = load_sibling("task_views")
esc = views.esc


def fmt_speed(value):
    return "n/a" if value is None else views.fmt_speed(value)


def machine_rank(machine):
    return MACHINE_ORDER.index(machine) if machine in MACHINE_ORDER else len(MACHINE_ORDER)


def sorted_machines(hub):
    return sorted(hub["machines"], key=lambda m: machine_rank(m["machine"]))


def backends_text(backends):
    return ", ".join(views.backend_label(b) for b in backends)


def task_detail(counts):
    if not counts.get("rows"):
        return NO_ROWS
    others = counts.get("others", 0)
    return f'{views.count_text(counts["models"], "model")} · {backends_text(counts["backends"])} · {views.count_text(others, "other engine")}'


def machine_link(machine, task_key):
    counts = machine["tasks"].get(task_key, {})
    return (f'<li><a href="#{views.view_token(machine["machine"], task_key)}"><b>{esc(machine["machine"])}</b>'
            f'<small>{esc(task_detail(counts))}</small></a></li>')


def task_box(hub, task_key, title, blurb):
    return (f'<section class="tbox" id="{task_key}"><h2>{esc(title)}</h2><p>{esc(blurb)}</p><ul>'
            + "".join(machine_link(m, task_key) for m in sorted_machines(hub)) + "</ul></section>")


def task_boxes(hub):
    return '<div class="boxes">' + "".join(task_box(hub, *task) for task in TASKS) + "</div>"


def engine_span(engine):
    return f'<span class="eng {views.engine_class(engine)}"><i></i>{esc(engine)}</span>'


def flag_item(flag):
    cloning = " (cloning)" if flag["task"] == "TTS cloning" else ""
    return (f'<li>{engine_span(flag["engine"])} <b>{esc(flag["model"])}</b> ({esc(flag["variant"])}) on {esc(flag["machine"])} {esc(flag["backend"])}{cloning}: '
            f'intelligibility WER <b>{flag["wer_pct"]:.1f}%</b> at {esc(fmt_speed(flag["speed_x"]))} ({esc(flag["wer_kind"])}).</li>')


def quality_section(hub):
    flags = hub.get("quality_flags", [])
    body = '<ul class="flags">' + "".join(flag_item(f) for f in flags) + "</ul>" if flags else f'<p class="muted small">No TTS row reaches the {QUALITY_FLAG_PCT}% intelligibility WER flag this week.</p>'
    return (f'<section class="block" id="quality"><h2>Quality flags</h2><p class="muted small">Fast results that come with a quality problem: TTS rows with '
            f'Whisper-tiny intelligibility WER at or above {QUALITY_FLAG_PCT}%. Intelligibility WER does not measure naturalness or speaker similarity.</p>{body}</section>')


def engine_row(engine):
    return (f'<tr><td>{engine_span(engine["engine"])}</td><td>{esc(engine["status"])}</td><td>{esc(engine["measured_on"] or "none")}</td>'
            f'<td>{esc(engine["details"])}<br><span class="muted">Source: {esc(engine["source"])}</span></td></tr>')


def engines_table(title, rows):
    if not rows:
        return ""
    return (f'<h3>{esc(title)}</h3><div class="tbl"><table><thead><tr><th>Engine</th><th>This week</th><th>Measured on</th><th>Scope and caveats</th></tr></thead>'
            f'<tbody>{"".join(engine_row(e) for e in rows)}</tbody></table></div>')


def engines_section(hub):
    engines = hub.get("engines", [])
    qvac = [e for e in engines if e["role"] == "QVAC engine"]
    prior = [e for e in engines if e["role"] == "prior-report competitor"]
    new = [e for e in engines if e["role"] not in ("QVAC engine", "prior-report competitor")]
    return f'<section class="block" id="engines"><h2>Engines</h2>{engines_table("Weekly engine", qvac)}{engines_table("From the previous comparison", prior)}{engines_table("New competitors", new)}</section>'


def dashboard_links(hub):
    machines = [m for m in sorted_machines(hub) if m.get("url")]
    return ", ".join(f'<a href="{esc(m["url"])}" target="_blank" rel="noopener">{esc(m["machine"])}</a>' for m in machines)


def example_link(hub):
    machine, task = EXAMPLE_PAGE
    return f'{esc(hub.get("hub_url", ""))}#{views.view_token(machine, task)}'


def pins_lines(hub):
    return "".join(f'<li><b>{esc(m["machine"])}</b>: run {esc(m["newest_run"])} ({esc(m["status"])}, {m["ok"]} of {m["expected"]} cells) at {esc(m["pins"])}.</li>' for m in sorted_machines(hub))


def method_section(hub):
    return f'''<section class="method" id="method">
    <div>
      <h3>What each page shows</h3>
      <ul>
        <li>Pick a task, then a machine: its task page has one section per model with a speed chart per backend and input, every engine in its color. Weekly charts hold this week's QVAC variants; report and campaign charts hold QVAC and its competitors as that comparison measured them.</li>
        <li>Each section folds away the details of every chart: exact values, precision per row, timed runs and warmups, generated duration, the competitor ÷ QVAC time ratio with its qualifier, and a Sources panel with source file, engine commits, model hashes, command, configuration and caveats. Every task page ends with what was not measured there, and why.</li>
        <li>Each task page has its own link, the hub's link plus the machine and task, for example {example_link(hub)}.</li>
        <li>{dashboard_links(hub)}: the machine dashboards, with an overview of the machine, ASR, TTS and music tabs with a week-over-week line per backend (buttons for speed, latency, RTF, host memory, device memory and WER), and a Runs and coverage tab listing every row and every skipped or failed cell.</li>
      </ul>
    </div>
    <div>
      <h3>How it runs</h3>
      <ul>
        <li>Every Sunday from 11:00 CET each shared machine updates the engine master head and the ggml commit it pins (the ggml of qvac-fabric-llm.cpp), builds every backend it has, provisions the pinned models and runs the matrix of <code>scripts/benchmarks/desktop-weekly/families.json</code>; a Claude task collects the summaries on Monday morning and refreshes these pages.</li>
        <li>Weekly rows, report rows and campaign rows are never in one chart: the report rows are copied from the comparison report of 2026-11-04 (QVAC and twelve other engines on the same machines, with that report's builds, inputs and timing scopes) and the campaign rows keep the build, inputs and dates of the QVAC speech engines report (October 2026 edition); each set is compared only among itself.</li>
        <li>ASR, Supertonic and Nemotron rows time inference only; Audio8, MOSS-TTS and ACE-Step rows are whole fresh processes including model load from a warm filesystem cache.</li>
        <li>Quality checks are narrow: ASR WER on one English clip, TTS intelligibility on one prompt. Music quality and speaker similarity are unmeasured.</li>
        <li>Deviations from the ticket: ACE-Step Base runs at Q4_K_M because the registry has no Q8_0 Base cut; MOSS-TTS Q8_0 is the F16 backbone with its transformer blocks requantized on the machine.</li>
        <li>Newest runs and pins: <ul>{pins_lines(hub)}</ul></li>
      </ul>
    </div>
  </section>'''


def styles():
    return "\n".join(path.read_text() for path in STYLE_FILES)


def hub_view(hub, built_at):
    return f'''<div class="view" data-view="{HUB_VIEW}"><div class="wrap">
  <header>
    <div class="eyebrow">QVAC speech · weekly bench · newest run {esc(hub["measured"])} · {len(hub["machines"])} machines · {hub["accepted"]} accepted measurements · page built {esc(built_at)}</div>
    <h1>Speech Bench</h1>
    <p class="lede">QVAC (qvac-fabric-speech.cpp) every week on the shared machines, with the rows of the comparison report and of the campaign against other speech engines. Pick a task, then a machine: each task page shows, per model and backend, this week's numbers and how QVAC compared with the other engines on that machine.</p>
  </header>
  {task_boxes(hub)}
  <details class="more">
    <summary>Quality flags, engines and how it runs</summary>
  {quality_section(hub)}
  {engines_section(hub)}
  {method_section(hub)}
  </details>
</div></div>'''


def render(hub, built_at, rows_by_host=None, coverage_by_host=None):
    task_pages = views.render_views(sorted_machines(hub), rows_by_host or {}, coverage_by_host or {})
    return f'''<title>Speech Bench</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
{styles()}
</style>
{hub_view(hub, built_at)}
{task_pages}
{ROUTER}
'''


def load_list(path):
    if not path.is_file():
        return []
    with open(path) as handle:
        return json.load(handle)


def load_host_files(data_dir, machines, name):
    return {m["host"]: load_list(data_dir / m["host"] / f"{name}.json") for m in machines}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hub")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--data")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.hub) as handle:
        hub = json.load(handle)
    data_dir = pathlib.Path(args.data) if args.data else pathlib.Path(args.hub).resolve().parent
    rows = load_host_files(data_dir, hub["machines"], "runs")
    coverage = load_host_files(data_dir, hub["machines"], "coverage")
    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pathlib.Path(args.out).write_text(render(hub, built_at, rows, coverage))
    print(f"wrote {args.out} for {len(hub['machines'])} machines and {len(hub['machines']) * len(TASKS)} task pages")


if __name__ == "__main__":
    main()
