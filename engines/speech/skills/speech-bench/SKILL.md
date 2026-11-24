---
name: speech-bench
description: Run, collect and publish the weekly desktop benchmark of the qvac-fabric-speech.cpp engines (Parakeet, TranscribePsy, Nemotron, Supertonic, Audio8, MOSS-TTS, ACE-Step) on the team's shared Tailscale machines, the RTX 5090 Linux box and the Mac Studio M3 Ultra, and keep the weekly results artifact current. Use when asked to benchmark the speech engines, check this week's speech numbers, rerun a disturbed cell, deploy the bench on a machine, or update the speech benchmark page.
argument-hint: collect | publish | run <linux|mac> [quick] | deploy <linux|mac>
---

# Speech weekly benchmark on the shared Tailscale machines

Every Sunday from 11:00 CET, cron on each shared machine runs `scripts/benchmarks/desktop-weekly/weekly.sh`
under the installing user's own account: it updates the engine `master` head and the ggml commit that head pins (the ggml of qvac-fabric-llm.cpp), provisions the
pinned models, builds every backend the machine has, drives `scripts/benchmarks/run-family.sh` over the matrix
in `families.json`, and writes `~/speech-bench/runs/<date>/summary.json` plus `report.md`. No developer
machine is involved in the measurement. The Speech Bench hub and its per-machine dashboards, linked from the team channel's canvas,
are where everyone reads the numbers. This skill is the Claude side: on-demand runs, collecting the summaries, building the
dashboards' data and the hub from them and publishing them, and the rules that keep two automated benches and twenty-odd people from
disturbing each other on the same hardware.

| path | what it is |
| --- | --- |
| `scripts/benchmarks/desktop-weekly/README.md` | the full description of the bench, its layout on a machine and its configuration |
| `scripts/benchmarks/desktop-weekly/weekly.sh` | the entry point cron runs; `--quick`, `--families`, `--backends`, `--no-update`, `--no-mail`, `--label` |
| `scripts/benchmarks/desktop-weekly/families.json` | the matrix: 59 families with their driver spec, iteration counts, timeouts and model provisioning; on the Mac Studio Metal lane eleven run the MLX engines (parakeet-mlx, mlx-whisper, mlx-audio), four FluidAudio (Core ML on the Apple Neural Engine, built from `fluidaudio-bench/`), ten the QVAC Core ML sidecars (`-coreml` twins of the four Parakeet-family ASR models on both clips, Supertonic 3 and Audio8; `COREML_PYTHON` in config.env, `-DPARAKEET_COREML=ON -DTTS_CPP_COREML=ON` on Darwin) and one MiniMax Music3 (`prepared` GGUF pair in the store); the models no MLX engine covers are the `not_measured` gaps of `engines.json` |
| `scripts/benchmarks/desktop-weekly/report.py` | `summary.json` (+ previous run) -> Markdown report with the changes over 5% and whether a pin moved |
| `scripts/benchmarks/desktop-weekly/build-dashboard-data.py` | a folder of `summary.json` files (plus `comparisons.json` and the `report-rows.json` next to the summaries folder) -> per machine the six JSON files its dashboard reads, and `hub.json` |
| `scripts/benchmarks/desktop-weekly/mlx-bench.py` | the bench the MLX families run (`run-family.sh` calls it through the venv at `~/speech-bench/mlx/venv` on the Mac Studio; README, Install) |
| `scripts/benchmarks/desktop-weekly/build-highlights.py`, `highlights.json` | the dashboard-data folder -> `speech-bench-highlights.html`, a short page of cards (QVAC against the other engines of a few GPU charts) for blog posts and updates |
| `scripts/benchmarks/desktop-weekly/import-report.py` | the saved task pages of a comparison report -> `report-rows.json`; the pages of the 2026-11-04 report and the rows are kept in the history folder (`report-pages/`, `report-rows.json`) |
| `scripts/benchmarks/desktop-weekly/build-landing.py`, `task_views.py` | `hub.json` and the machines' `runs.json` and `coverage.json` -> the Speech Bench hub (per task every machine with a link to its task page, quality flags, engines, how it runs) with the 18 task pages as views of the same page (`<hub link>#<machine>-<task>`) |
| `scripts/benchmarks/desktop-weekly/dashboard/`, `dashboards.json` | the page every machine dashboard runs (`index.html`, `app.js`, `app.css`) and the hub's stylesheets (`hub.css`, `task.css`); the links of the hub and of every machine's dashboard |
| `scripts/benchmarks/desktop-weekly/install.sh` | deploys the tools to `~/speech-bench/tools` and installs the crontab line |

## Machines, schedule and coordination

| machine | host | backends | clock | config.env | notes |
| --- | --- | --- | --- | --- | --- |
| DGX Spark (GB10, 128 GB unified) | `spark-d9ca` | cpu, cuda | UTC | `BACKENDS=cpu,cuda`, `CUDA_ARCH=121a-real`, `CUDA_HOME=/usr/local/cuda-13.3`, `MEMORY_SAMPLER=meminfo` | nvidia-smi reports no memory.used; weights load at about 0.5 GB/s |
| RTX 5090 box (2x 32 GB, GPU 0 used) | `qvac-dev-linux-x64` | cpu, cuda, vulkan | UTC | `MODELS_DIR=/home/shared/models/qvac-speech/bench` | 22-account `dev` group can write the shared model folder |
| Strix Halo (Radeon 8060S, 128 GB) | `qvac-dev-strix-1` | cpu, vulkan | UTC | `BACKENDS=cpu,vulkan` | no cron: `install.sh` starts `scheduler.sh`, which does not survive a reboot; no ninja (the bench falls back to make) |
| Jetson Orin (8 GB) | `qvac-dev-linux-arm64` | cpu, cuda, vulkan | UTC | `BACKENDS=cpu,cuda,vulkan`, `CUDA_ARCH=87-real`, `MEMORY_SAMPLER=meminfo`, `EVICT_PAGE_CACHE=1`, `SKIP_FAMILIES` of the MOSS families and of every ACE-Step family but `acestep-turbo-q8_0`, `CMAKE_PREFIX_PATH=$HOME/speech-bench/deps` | no Vulkan SDK and no root: SPIRV-Headers and Vulkan-Headers are installed under `deps` by hand; no weekly schedule installed yet |
| Mac Studio M3 Ultra (96 GB) | `qvac-dev-mac-arm64` | cpu, metal | CET | `MODELS_DIR=/Users/Shared/models/qvac-speech/bench`, `COREML_PYTHON=~/speech-bench/coreml-venv/bin/python` | shares Saturday with the fabric-llm bench; the only host of the MLX families, through the venv at `~/speech-bench/mlx/venv` (README, Install) |
| Mac mini M4 (16 GB) | `mac-mini-m4-gpu` | cpu, metal | CET | `SKIP_FAMILIES=moss-tts-f16,moss-tts-q8_0,moss-tts-clone-f16` | macOS 15: `/bin/date` has no %N, the bench routes `run-family.sh` through `nsdate` |

`qvac-dev-strix-0` has the same hardware as `qvac-dev-strix-1` and ran the bench until 2026-11-04 (`build-dashboard-data.py` lists it in `RETIRED_HOSTS` and skips its summaries); the results show one
Strix Halo, so it is not part of the weekly set.

Every machine runs the speech bench on Sunday from 11:00 CET (10:00 on the UTC clocks: `install.sh --hour 10`
there, `--hour 11` on the Macs, `--hour 15 --minute 30` on the Jetson's India time), about 6 to 7 hours. The fabric-llm bench (Aman's `engine-bench`) owns Saturday
from 18:00 CET on the Mac Studio, the Strix Halo and the DGX Spark. The bench folder is `~/speech-bench` everywhere,
with runs in `runs/<date>` and models in `models/` unless `MODELS_DIR` says otherwise.

Two automated benches share three of the machines. They are kept apart by **day** (LLM on Saturday, speech on Sunday, both
from 18:00 CET) and by the **lock** `/tmp/gpu-bench.lock`: each bench waits while the file exists or another
user's GPU job runs, creates the file atomically when it starts measuring, and removes it when it is done. A
contaminated run is one where something else ran on the machine during the measurement; the report flags it as a
wall-clock move with unchanged pins. The rules for every Claude session and every person:

- **Never start a run on either machine from Saturday 17:00 CET to Monday 02:00 CET**, which covers both
  weekly benches with margin. Rerun a disturbed cell on a weekday instead.
- **Check before any on-demand run:** `who`, `ps -eo user,pcpu,etime,comm --sort=-pcpu | head` (macOS:
  `ps -Ao user,pcpu,etime,comm -r | head`), `nvidia-smi` on linux, and `ls -l /tmp/gpu-bench.lock`. If someone
  else is running GPU work, wait or ask them in the channel. Never delete someone else's lock and never kill
  another user's process.
- **Ask your user before a run longer than about 30 minutes during European working hours.** `--quick` is
  about 3 minutes once models and builds are in place; a full `weekly.sh` is 6 to 7 hours.
- **One weekly schedule per bench per machine.** Do not add a second crontab line or a second scheduled task
  that runs the bench; change the hour with `install.sh --hour` if the day split ever moves.
- **Do not change the other bench's files or lock.** If the LLM bench's schedule moves, agree the new split with
  its owner first and update this file and the README together.
- **No root, no `sudo`, nothing installed outside `~/speech-bench` and the shared model folder.**

## Procedure

Arguments: `collect`, `publish`, `run <linux|mac> [quick]` or `deploy <linux|mac>`. Without an argument, ask
which one. SSH as yourself with the host names above.

### collect

Copy every finished run's summary into the local history folder, one file per run, named by host and run:

```sh
H=qvac-dev-linux-x64   # and spark-d9ca, qvac-dev-strix-1, qvac-dev-linux-arm64, qvac-dev-mac-arm64, mac-mini-m4-gpu
HISTORY=${SPEECH_BENCH_HISTORY:-$HOME/speech-bench-history}
mkdir -p $HISTORY/summaries/$H
for run in $(ssh $H 'cd ~/speech-bench/runs && for d in */; do [ -e "$d/DONE" ] && echo "${d%/}"; done'); do
  scp -q $H:speech-bench/runs/$run/summary.json $HISTORY/summaries/$H/$run.json
done
```

The history folder is the input of `build-dashboard-data.py`. Keep it; a run pruned on the machine (the bench keeps
12) stays in the page's history only through this folder. Also copy `report.md` of the newest run when the user
wants to read it.

### publish

Build the data files and the hub from the history folder:

```sh
HISTORY=${SPEECH_BENCH_HISTORY:-$HOME/speech-bench-history}
D=scripts/benchmarks/desktop-weekly
python3 $D/build-dashboard-data.py $HISTORY/summaries -o $HISTORY/dashboard-data
python3 $D/build-landing.py $HISTORY/dashboard-data/hub.json -o $HISTORY/speech-weekly-landing.html
```

The links are in `$D/dashboards.json`: `landing` (the Speech Bench hub, the page linked from the canvas) and one
dashboard per host. Each dashboard is a Dashboard-type artifact whose page (`files/index.html`, `files/app.js`,
`files/app.css`) reads six attached files from `$HISTORY/dashboard-data/<host>/`: the datasets `runs`, `history`,
`coverage`, `engines`, `machine` and `commentary`. Refresh them in place, never as new artifacts:

1. For each host with a new run, upload the files of `$HISTORY/dashboard-data/<host>/` that changed (usually `runs`,
   `history`, `coverage`, `machine` and `commentary`; `engines` changes when an engine gains or loses a machine) with
   the Artifact tool: `url` = that host's dashboard, `asset: true`, `file_path` = the file. Each upload answers with a
   `/_blob/<id>` url.
2. Point the datasets at the new files with the ArtifactData tool, one batch per dashboard: read `datasets/<id>` for
   its `version`, then `update` it with `if_version`, `source` = `{kind: "file", url: <the blob url>, name: <the file
   name>, format: "json"}`, `updated: {at: <UTC ISO>, by: "Claude (weekly run <date>)"}` and
   `edit: {via: "claude", at: <UTC ISO>}`.
3. Only when `$D/dashboard/index.html`, `app.js` or `app.css` changed: read `files/<name>` and `set` it to
   `{"text": <the file>, "edit": {...}}` with `if_version`, on every dashboard.
4. Read the hub (`landing`) with the Artifact tool, then publish `$HISTORY/speech-weekly-landing.html` with the same
   `url` (no icon); it carries the task pages of every machine, so there is nothing else to publish for them. If the publish is refused because a newer version is live, publish again: the page is rebuilt from
   the history folder and contains everything.

A machine new to the bench gets its dashboard once: create an artifact from the Dashboard type
(`https://claude.ai/artifact/GZLZthZ2sQjn7T16skfdeF`, title `Speech Bench · <machine>`), write `dash/meta`, the six
datasets and the three page files in one ArtifactData batch, and add the host to `dashboards.json`, to `MACHINE_LABELS`
in `build-dashboard-data.py` and, when the campaign measured that machine, to `CAMPAIGN_HOSTS`. Only its owner can share
it with the organization, from its Share menu.

The pages follow the Speech Bench report the team approved. The hub lists, per task (TTS and voice cloning, ASR,
music), every machine with its models, backends and other engines, linking to that machine's task page, then the
quality flags (TTS rows with intelligibility WER at or above 10%), the engines and how the bench runs. A task page is a
view of the hub page, reached at the hub's link plus `#<machine>-<task>` (for example `#dgx-spark-tts`), so it is
shared with the hub and needs no artifact of its own: one section per model with a horizontal speed chart per backend
and input (the weekly charts hold QVAC's variants, the campaign charts hold QVAC and its competitors in engine colors),
a details panel per chart (exact values, precision per row, the competitor-over-QVAC time ratio, notes and sources)
and, at the end, what was not measured there and why. A machine dashboard has an overview (status, a summary of the week generated from the summaries, quality flags, which models ran
on which backend, gaps, engines, comparison rules), one tab per task with a section per model (a speed chart per backend
and input: the weekly charts hold QVAC's variants, the campaign charts hold QVAC and transcribe.cpp, audio.cpp, LocalAI
parakeet.cpp or acestep.cpp in engine colors; details with exact values, precision, time ratios and sources) and a
week-over-week line per backend, and a Runs and coverage tab. Tell the user what changed since the previous week: failed
cells, moves over 5% and which pins moved, as each dashboard's summary states them. To refresh a comparison, re-run the
campaign's measurement for that model and device and replace the row (`qvac_rtf`, `ratio`, versions, `measured`,
`source`) in `comparisons.json`; TranscribePsy, Nemotron and MOSS-TTS have no campaign rows yet. `build-artifact.py`
still writes a single self-contained HTML report from the same folder for reading offline.

### run

On demand, outside the protected window and after the checks above. Start it detached so it survives the SSH
session, and wait for `DONE` or `FAILED` without polling in the foreground:

```sh
ssh <host> 'nohup /bin/bash ~/speech-bench/tools/weekly.sh --quick --no-mail --label manual < /dev/null > /dev/null 2>&1 &'
```

`--quick` runs three families with one timed iteration into `runs/<date>-quick-manual`; leave `--quick` out for
the full matrix, or narrow it with `--families a,b` and `--backends a,b`. The log is `runs/<run>/weekly.log`; a
family's own output is `results/<backend>/<family>.log`. A labelled or quick run never becomes the baseline of
the next weekly report, and the page lists it under its own name. To rerun a disturbed cell of a weekly run,
delete that cell's `results/<backend>/<family>.json` on the machine and rerun `weekly.sh --no-update` with the
same date (it resumes, keeping the ok cells and rebuilding nothing when the commits did not move).

### deploy

```sh
scp -r scripts/benchmarks/desktop-weekly <host>:speech-bench-src
ssh <host> 'bash speech-bench-src/install.sh --hour 11 && rm -rf speech-bench-src'   # --hour 10 on linux (UTC)
```

Then set the machine's keys in `~/speech-bench/config.env` (see the machines table) and try
`weekly.sh --quick --no-mail` once. Results reach people through the artifact page, not by mail; `MAIL_TO` and
`ANTHROPIC_API_KEY_FILE` stay unset unless someone asks for mail or the narrative. The README has the prerequisites
and the layout.

## Reading a run

`summary.json` (schema 1): `run` (name, host, started, finished, status ok/partial/failed), `machine`,
`pins` (engine and ggml commits, CUDA arch, registry client, venv packages, every model file with bytes, sha256
and origin), `backends`, `families`, `results` (one row per backend and family: status, observed backend,
wall median/min/max in ms, rtf, audio seconds, speed_x, peak RSS MiB, GPU memory peak and rise MiB, quality
kind and value, runs, notes), `counts`, `build_failures`, `problems`.

- Speed is seconds of audio per second of wall clock. ASR and Supertonic rows time inference only; Audio8,
  MOSS and ACE-Step rows are process wall clock including model load, so their speed is a throughput figure, not
  a latency.
- A slower cell with unchanged pins points at load from another user; a changed pin names the engine, ggml or
  model cut to look at. Report every number as measured and say what was disturbed or not like for like.
- Two rows are not what the ticket asked for because the registry has no such cut: ACE-Step Base runs at
  Q4_K_M, and MOSS-TTS Q8_0 is the F16 backbone with its transformer blocks requantized on the machine.

## Rules

- Benchmarks only: do not commit, push or open pull requests from a benchmark session.
- Publish only to the hub and the dashboards named in `dashboards.json`, and only when the user asks or the scheduled
  task runs.
- Never start a run inside the weekend window, never touch other users' files, locks or processes, and never run
  anything as root.
