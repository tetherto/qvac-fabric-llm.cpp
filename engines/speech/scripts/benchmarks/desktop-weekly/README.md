# Weekly desktop benchmark on the shared Tailscale machines

`weekly.sh` benchmarks the speech engines every Sunday evening on the team's shared machines, without any
developer laptop in the loop: cron on each machine builds the current engine head, with the ggml commit that head pins (the ggml of qvac-fabric-llm.cpp), for every backend the
machine has, provisions the pinned models, runs the matrix in `families.json` through
`scripts/benchmarks/run-family.sh`, and writes a Markdown report with the numbers, the memory peaks, the exact
machine and toolchain versions and the sha256 of every model file. The results everyone reads are the
Speech Bench hub, with a task page per machine and task, and one dashboard per machine, refreshed from the machines'
summaries every Monday (see
`skills/speech-bench/SKILL.md`); e-mail delivery exists but is off unless `MAIL_TO` is set.

| machine | host | backends | schedule |
| --- | --- | --- | --- |
| DGX Spark GB10 | `spark-d9ca` | cpu, cuda | Sunday 11:00 CET (10:00 on the UTC clock) |
| Linux x64, 2x RTX 5090 | `qvac-dev-linux-x64` | cpu, cuda, vulkan | Sunday 11:00 CET (10:00 on the UTC clock) |
| Strix Halo | `qvac-dev-strix-1` | cpu, vulkan | Sunday 11:00 CET (10:00 on the UTC clock) |
| Mac Studio M3 Ultra | `qvac-dev-mac-arm64` | cpu, metal | Sunday 11:00 CET |
| Mac mini M4 | `mac-mini-m4-gpu` | cpu, metal | Sunday 11:00 CET |

The two weekly benches on the shared machines run on different days so that neither disturbs the other: the
fabric-llm engine bench on Saturday from 18:00 CET, the speech bench on Sunday from 11:00 CET (a full run takes
about 6 to 7 hours). Both also hold the same `/tmp/gpu-bench.lock` on the Mac; this run waits for the lock (up to
`LOCK_WAIT_HOURS`, 12 by default) and for every other GPU job to end before it starts, then holds the lock while it
benchmarks. See [`skills/speech-bench/SKILL.md`](../../../skills/speech-bench/SKILL.md) for the coordination rules
and the weekly artifact.

## What is measured

`families.json` lists one family per model and quantization requested for the desktop matrix:

| task | family | weights |
| --- | --- | --- |
| ASR | `parakeet-tdt` | Parakeet TDT 0.6B v3 Q8_0 |
| ASR | `parakeet-unified` | Parakeet Unified English 0.6B Q8_0 |
| ASR | `transcribepsy` | TranscribePsy-250M-EU-sub Q8_0 (HuggingFace, pinned revision) |
| ASR | `nemotron` | Nemotron 3.5 ASR Streaming 0.6B F16 |
| ASR | `parakeet-eou` | Parakeet Realtime EOU 120M F16 |
| ASR | `whisper-small` | Upstream whisper.cpp `ggml-small.bin` through `whisper-cli` (HuggingFace, pinned revision) |
| ASR | `{parakeet-tdt,parakeet-unified,transcribepsy,nemotron,parakeet-eou,whisper-small}-abcba` | the same models on the 160.6 s multi-speaker ABCBA clip, which has no reference transcript (no WER) |
| ASR | `parakeet-mlx`, `parakeet-mlx-abcba` | another engine beside QVAC, Mac Studio Metal only: parakeet-mlx on the mlx-community Parakeet TDT 0.6B v3 conversion (BF16), timed in process after one model load like the QVAC native rows, so it shares their charts |
| ASR | `nemotron-mlx-audio`, `nemotron-mlx-audio-abcba` | mlx-audio on the mlx-community Nemotron 3.5 ASR Streaming 0.6B conversion (BF16), Mac Studio Metal only, offline decoding timed in process like the QVAC `nemotron` rows |
| TTS | `chatterbox-turbo-mlx-audio` | mlx-audio Chatterbox Turbo (F16 conversion, built-in voice, seed 42), Mac Studio Metal only, a fresh process including load like the QVAC `chatterbox-turbo` rows |
| TTS, TTS cloning | `audio8-mlx-audio`, `audio8-clone-mlx-audio` | mlx-audio Audio8 TTS Preview 0.6B (BF16 conversion) with the sampling of the Audio8 families, synthesis and JFK cloning, Mac Studio Metal only, fresh processes like the `audio8-lm-*` and `audio8-clone-enc-*` rows |
| Music | `minimax-music3-mlx-audio` | mlx-audio on the 8-bit MiniMax Music3 conversion (the bf16 one needs 27 GB more disk), the 30 s instrumental request of the ACE-Step rows, Mac Studio Metal only, beside the QVAC `minimax-music3-q8_0` row |
| TTS | `moss-tts-mlx-audio` | mlx-audio on the original MOSS-TTS v1.5 checkpoint (BF16, seed 1234), Mac Studio Metal only, a fresh process like the `moss-tts-f16` rows |
| ASR | `{parakeet-tdt,parakeet-unified,parakeet-eou,nemotron}-coreml` and their `-abcba` twins | the same QVAC models with the engine's Core ML encoder sidecar on the Apple Neural Engine (exported once from the GGUF, 6-bit palettized; Parakeet TDT also gets the batched TDT decoder sidecar), Mac Studio only; the driver also runs the forced-ggml baseline in the same process and the row keeps its time; the ABCBA twins have no reference transcript, so they keep both timings and score nothing; the EOU and Nemotron ABCBA twins export their encoders at the clip's exact 16060 mel frames, because the engine takes the ggml path for any input longer than an EOU or Nemotron sidecar |
| TTS | `supertonic3-coreml`, `audio8-lm-q8_0-dec-q8_0-coreml` | Supertonic 3 with its Core ML vocoder sidecar and Audio8 with its Core ML codec decoder sidecar on the Apple Neural Engine (exported once by the engine's exporters through `COREML_PYTHON`, a `sidecar` provision entry), Mac Studio only |
| Music | `minimax-music3-q8_0` | MiniMax Music3 through `mm3-replay`, the 30 s instrumental request at seed 42, Mac Studio only; its GGUF pair is prepared by `scripts/benchmarks/prepare-minimax.py` elsewhere and placed in `models/store/` by hand |
| ASR | `parakeet-fluidaudio`, `parakeet-fluidaudio-abcba`, `parakeet-unified-fluidaudio`, `parakeet-unified-fluidaudio-abcba` | FluidAudio (Swift, Core ML, encoder on the Apple Neural Engine, int8 encoder) on its Parakeet TDT 0.6B v3 and Parakeet Unified English 0.6B conversions, Mac Studio only, timed in process like the QVAC native rows through `fluidaudio-bench/` |
| ASR | `mlx-whisper`, `mlx-whisper-abcba` | mlx-whisper on the mlx-community Whisper small conversion (F16), Mac Studio Metal only, timed in process; the QVAC `whisper-small` rows are fresh-process timings, so these stay in their own charts |
| TTS | `supertonic3` | Supertonic 3 Q8_0, every component and voice in one GGUF |
| TTS | `chatterbox-turbo` | Chatterbox Turbo: T3 Turbo Q8_0 with S3Gen Q8_0, built-in voice |
| TTS | `parler-mini`, `parler-indic` | Parler-TTS mini v1 and Indic Parler-TTS Q8_0, sampled decoding at seed 42 |
| TTS | `audio8-lm-{f16,q8_0,q4_0}-dec-{f16,q8_0}` | Audio8 TTS Preview 0.6B, LM x codec decoder grid |
| TTS cloning | `audio8-clone-enc-{f16,q8_0}` | Audio8 with the codec encoder, cloning the JFK clip |
| TTS | `moss-tts-f16`, `moss-tts-q8_0` | MOSS-TTS v1.5 8.5B: F16 backbone, and transformer blocks requantized to Q8_0; decoder F16 |
| TTS cloning | `moss-tts-clone-f16` | MOSS-TTS with the F16 tokenizer encoder and the JFK clip at 24 kHz |
| TTS cloning | `chatterbox-mtl-clone` | Chatterbox Multilingual 0.5B (T3 MTL Q8_0, S3Gen MTL Q8_0) cloning the JFK clip |
| TTS cloning | `cosyvoice3-clone` | CosyVoice3 0.5B (LM Q8_0, flow and HiFT F32, speech tokenizer Q8_0, CAM++ F32) cloning the JFK clip with its transcript as the prompt text |
| Music | `acestep-turbo-q8_0`, `acestep-turbo-q4_k_m`, `acestep-sft-q8_0`, `acestep-base-q4_k_m` | ACE-Step 1.5 DiT variants with Qwen3-Embedding-0.6B Q8_0, 5Hz LM 0.6B Q8_0 and the BF16 VAE |

Every family is a `run-family.sh` spec (same fields as `scripts/benchmarks/families.json`) plus `runs`, `warmup`,
`cpu_runs`, `cpu_warmup`, `timeout_minutes` and a `provision` list that says where each model file comes from.
ASR rows report the native bench's inference time, RTF and WER on the 11 s JFK clip, and the `-abcba` families the
same on the 160.6 s ABCBA clip (`parakeet/test/samples/abcba.wav`, no reference transcript, so no WER); the
whisper rows are the whole `whisper-cli` process including model load. TTS rows report the process wall clock
including model load (the native Supertonic, Parler and CosyVoice benches time synthesis only), the generated duration
and the TTS intelligibility WER (pinned Whisper tiny on CPU); music rows report the process wall clock of a 30 s render
at seed 42. A family with its own input names it in `scenario` (`abcba`); the report keeps every input in its own chart.
A provision entry with `as` links the file into the family directory under another name (CosyVoice3 requires
`voice.gguf` even when cloning). Memory is the peak RSS from
`/usr/bin/time` and, on NVIDIA backends, the rise of `nvidia-smi` memory.used over the run.

Next to the weekly rows the dashboards and task pages show two copied sets, never mixed with the weekly charts: the
campaign rows of `comparisons.json` (ratios from the October 2026 report) and the report rows of `report-rows.json`,
which `import-report.py` copies from the saved task pages of the comparison report of 2026-11-04 (QVAC and twelve
other engines on the same machines, with that report's own builds, inputs and timing scopes: resident API, native
stage sum, fresh process, HTTP). `build-dashboard-data.py` reads `report-rows.json` from the folder above the
summaries folder (or `--report`); the file and the saved pages live in the history folder, not in the repo. Summaries of a
retired host (`RETIRED_HOSTS`, today `qvac-dev-strix-0`, replaced by `qvac-dev-strix-1`) are ignored even when they are still
stored. Cloning rows of a time-wrapped family include model load and reference encoding; those of a native bench
(CosyVoice3) time synthesis only, and the pages say which.

A family may name another engine (`engine`, `platform`, `backends` in `families.json`): `weekly.sh` skips it on the other
platforms, the driver skips it on the other backends, and `build-dashboard-data.py` puts its rows in the chart of the QVAC
family with the same model, input and timing scope, so the weekly charts compare QVAC with that engine every week. The
MLX families run `mlx-bench.py` with the venv described under Install; their models are pinned Hugging Face snapshots
that the script fetches once into `models/mlx/<conversion name>/` on the first run, outside the timed region, shared by the families of one model.

The models of the matrix that no MLX engine can run on the Mac Studio (Supertonic 3, CosyVoice3, both Parler models, Chatterbox
Multilingual, Parakeet Unified and EOU, TranscribePsy and ACE-Step) are listed as `not_measured` of the mlx-audio
entry in `engines.json`, with the reason, and appear as coverage gaps on that machine's task pages.

`build-highlights.py` turns the charts named in `highlights.json` (a few GPU results per machine: Parakeet, Nemotron,
Supertonic 3, Audio8, MiniMax Music3, ACE-Step, and the MLX engines on the Mac Studio) into one short page of cards for a
blog post or a status update, QVAC against every other engine of that chart; a card whose chart has no other engine yet is
left out and named on stderr.

Two requested cuts are not in the registry: MOSS-TTS Q8_0 is produced on the machine by requantizing the `blk.*`
tensors of the F16 backbone with `tts/scripts/requantize-gguf.py`, and the ACE-Step Base DiT ships only as
Q4_K_M, so that row is Q4_K_M and says so.

## Files

| file | what it does |
| --- | --- |
| `weekly.sh` | entry point: lock, update sources, fetch models, build and bench every backend, summarize, report, e-mail |
| `import-report.py` | the saved task pages of a comparison report -> `report-rows.json` (rows in the shape of `runs.json`, comparison set `report`, plus its not-measured lists as coverage cells) |
| `fluidaudio-bench/` | Swift package: native-style bench of FluidAudio (Parakeet TDT v3 and Unified on Core ML, ANE encoder) for `run-family.sh`, same JSON as `mlx-bench.py` |
| `mlx-bench.py` | native-style bench of the MLX engines (parakeet-mlx, mlx-whisper, mlx-audio Nemotron) for `run-family.sh`: pinned snapshot, model loaded once, warm-ups, timed transcriptions, the driver's bench JSON; `--once` for a single transcription, or for the mlx-audio Chatterbox Turbo synthesis of a time-wrapped family |
| `build-highlights.py`, `highlights.json` | the dashboard-data folder -> `speech-bench-highlights.html`, the cards of the charts listed in `highlights.json` |
| `lib.sh` | shared functions: config, lock, source update, per-backend builds, watchdog, GPU sampler, run selection |
| `families.json` | the matrix (see above) |
| `fetch-models.sh`, `fetch-registry.js`, `prepare-derived.py` | model provisioning: registry client, HuggingFace, repo fixtures, requantized and resampled files |
| `machine-info.sh`, `collect-pins.sh` | `machine.json` (hardware, OS, driver, toolchain) and `pins.json` (engine and ggml commits, model sha256 and origins) |
| `summarize.py` | run directory -> `summary.json` |
| `report.py` | `summary.json` (+ previous run) -> `report.md` with the changes over 5% and whether a pin moved |
| `build-dashboard-data.py`, `comparisons.json`, `engines.json` | a folder of `summary.json` files -> per machine the JSON files its dashboard reads (`<host>/runs.json`, `history.json`, `coverage.json`, `engines.json`, `machine.json`, `commentary.json`) and `hub.json` for the hub; `comparisons.json` holds the dated competitor rows of the campaign (qvac RTF and competitor-over-qvac time ratio per device and backend), `engines.json` the engines with their scope |
| `dashboard/index.html`, `dashboard/app.js`, `dashboard/app.css`, `dashboards.json` | the page every machine dashboard runs (overview, a tab per task with a speed chart per model, backend and input and a week-over-week line per backend, runs and coverage) and the links of the hub and the dashboards |
| `build-landing.py`, `task_views.py`, `dashboard/hub.css`, `dashboard/task.css` | `hub.json` and the machines' `runs.json` and `coverage.json` -> the Speech Bench hub: per task every machine with a link to its task page, the quality flags, the engines and how the bench runs. The task pages (one section per model with a speed chart per backend and input, details with exact values, precision, time ratios and sources, and what was not measured) are views of the hub page, each at the hub's link plus `#<machine>-<task>`, for example `#dgx-spark-tts` |
| `build-artifact.py` | a folder of `summary.json` files -> one self-contained HTML report (tables per task, history and pins per machine) for reading offline, without the dashboards |
| `nsdate`, `scheduler.sh` | nanosecond clock shim for macOS before 26, and the user-space scheduler for hosts without cron |
| `narrate.py` | optional: asks Claude for a ten-line narrative placed above the report in the e-mail |
| `send-report.py` | SMTP delivery with `report.md` and `summary.json` attached |
| `install.sh`, `config.example.env` | deployment and the per-machine configuration |
| `test-desktop-weekly.sh`, `test_*.py` | the test suite (runs in the desktop benchmark workflow's self-check) |

## Layout on a machine

Everything lives under `BENCH_HOME` (`~/speech-bench` by default):

```
tools/                 this directory, copied by install.sh
config.env             per-machine settings (see config.example.env)
src/speech, src/ggml   shallow checkouts of the engine and ggml branches
builds/<backend>/      ggml, ggml-install and speech build trees, plus BUILT with the commit pair they came from
registry-client/       the npm registry client, private to this account
venv/                  python venv with gguf and numpy for the requantized cut, private to this account
models/store/          one copy of every model file, with .sha256 and .source sidecars
models/families/<f>/   symlinks the driver reads as --models-root
runs/<date>/           weekly.log, machine.json, pins.json, results/<backend>/<family>.{json,log,gpu.csv,load.json},
                       summary.json, report.md, narrative.md, mail-body.md, DONE or FAILED
```

`MODELS_DIR` defaults to the private `models/` above. It can point at the shared model folders instead
(`/home/shared/models/qvac-speech/bench` on the Linux box, `/Users/Shared/models/qvac-speech/bench` on the Mac) so
that every account reuses one copy, at a price: every account that can write there can replace a model file or its
`.sha256` sidecar, so the report's pins are only as trustworthy as that folder's writers. Only data goes to
`MODELS_DIR`; the registry client, the venv, the checkouts and the builds always stay under `BENCH_HOME`.

## Install

On each machine, as your own user (nothing needs root):

```sh
scp -r scripts/benchmarks/desktop-weekly <host>:speech-bench-src
ssh <host> 'bash speech-bench-src/install.sh --hour 11'   # --hour 10 on a UTC machine, --hour 15 --minute 30 on the Jetson (India time)
```

`install.sh` copies the tools to `~/speech-bench/tools`, seeds `~/speech-bench/config.env` and installs the cron
line (Sunday 18:00 machine time by default; `--weekday`, `--hour`, `--minute` change it, `--no-cron` skips it).
Then edit `config.env` for the machine's settings (`MODELS_DIR`, `BACKENDS`, `CUDA_ARCH` and the other keys in
`config.example.env`); mail and the Claude narrative are optional and stay off without their keys. Any secret
lives in a file with mode 0600 that the config points at, never in `config.env` itself.

Prerequisites already present on the machines: git, cmake, ninja, jq, python3, node, curl; CUDA 13.3 and the
Vulkan SDK on the Linux boxes; Xcode command line tools and Homebrew on the Macs. The registry client is installed
by `fetch-models.sh` into `registry-client/` under the bench home from npm. Per-machine settings (`BACKENDS`,
`CUDA_ARCH`, `CUDA_HOME`, `MEMORY_SAMPLER=meminfo` for unified-memory NVIDIA hosts, `SKIP_FAMILIES` for the 16 GB
Mac mini, where a `family:backend` entry skips one cell that does not fit on one lane, as on the 8 GB Jetson) are
listed in `config.example.env` and in the skill. A host without cron (the Strix Halo) gets
`scheduler.sh` from `install.sh` instead of a crontab line; it does not survive a reboot. A macOS host whose
`/bin/date` lacks `%N` (macOS 15) gets `run-family.sh` routed through the `nsdate` shim after every source update.

The MLX families need, on the Mac Studio only, a Python virtual environment at `~/speech-bench/mlx/venv` with `mlx`,
`parakeet-mlx`, `mlx-whisper`, `mlx-audio`, `huggingface_hub`, `soundfile` and `numpy` (Python 3.11 or newer; the machine has no
Homebrew, so use a uv-managed interpreter: `python3.11 -m venv ~/speech-bench/mlx/venv && ~/speech-bench/mlx/venv/bin/pip
install mlx parakeet-mlx mlx-whisper mlx-audio huggingface_hub soundfile numpy`). Without it those cells report `build-failed`
(binary missing) and the rest of the run is unaffected.

On macOS the engine is configured with `-DPARAKEET_COREML=ON -DTTS_CPP_COREML=ON` (part of the build signature, so the first run
after the change rebuilds), which the `-coreml` families need. They also need, on the Mac Studio only, a Python with `parakeet/scripts/requirements-coreml.txt`
(coremltools, torch) named by `COREML_PYTHON` in `config.env`; the first run of each family exports and compiles the encoder
sidecar into the family folder (minutes per model), later runs reuse it. The MiniMax family needs `mm3-lm-q8_0.gguf` and
`mm3-synth-q8_0.gguf` in `models/store/`: run `scripts/benchmarks/prepare-minimax.py --models-root <dir> --quant q8_0` on a
machine with 64 GiB of RAM and 80 GB of free disk (an x86_64 Linux box or a Mac; its pinned requirements carry no aarch64
hashes) and copy the pair over.

The FluidAudio families need, on the Mac Studio only, the Swift bench package built once: copy `fluidaudio-bench/` to
`~/speech-bench/fluidaudio/bench` and run `swift build -c release` there (Xcode's toolchain; SwiftPM fetches FluidAudio at the
commit pinned in `Package.swift`). Its Core ML models download into `models/fluidaudio/` on the first run, outside the timed
region. Without the binary those cells report `build-failed`.

Try the pipeline before the first Sunday:

```sh
ssh <host> 'nohup ~/speech-bench/tools/weekly.sh --quick --no-mail < /dev/null > /dev/null 2>&1 &'
```

`--quick` runs `QUICK_FAMILIES` (Parakeet TDT, Supertonic 3 and one Audio8 pair) with one timed iteration into
`runs/<date>-quick`; `--families a,b` and `--backends a,b` narrow a run further, `--no-update` keeps the current
checkouts, `--label NAME` keeps several runs of one day apart. Watch `runs/<date>-quick/weekly.log` and wait for
`DONE` or `FAILED`; a family's own output is in `results/<backend>/<family>.log`.

## Where the results go

Each run leaves `report.md` and `summary.json` under `runs/<date>/` on the machine. The speech-bench skill (a
scheduled Monday task) copies every machine's summaries into a history folder, turns them into the dashboards' data
files with `build-dashboard-data.py`, uploads each machine's files to that machine's dashboard and republishes the
hub, with its task pages, from `build-landing.py`, so the hub and its dashboards are the place to look. E-mail is optional and off by default: with `MAIL_TO`, `MAIL_FROM` and the SMTP keys set in
`config.env` (tether.io mail is Google Workspace, so `smtp.gmail.com:465` with an app password in a mode 0600 file),
`send-report.py` also mails the report with `report.md` and `summary.json` attached, and a failed run mails a
one-paragraph notice.

## Narrative by Claude

With `ANTHROPIC_API_KEY_FILE` set, `narrate.py` sends `report.md` to the Messages API (model
`NARRATIVE_MODEL`, `claude-opus-5-5` by default, with the server-side refusal fallback enabled) and asks for at
most twelve plain lines: the overall state, every failed cell, every metric that moved by more than the threshold
with the most likely cause given the pins that changed, and what to look at this week. The call uses the standard
library over HTTPS because the bench machines carry no SDK; any error or a missing key skips the narrative and the
report still goes out.

## Reading a report

- **Pins** names the engine and ggml commits, the CUDA architecture, the registry client, the TTS scorer pin and
  every model file with its size, sha256 and origin (registry key, HuggingFace URL, repo fixture or derived cut).
- **Results on &lt;backend&gt;** has one row per family: status (`ok (under load)` for a cell that started on a busy
  machine, see below), wall median with its min and max, speed (seconds of audio per second of wall clock), peak RSS,
  GPU memory rise and peak, quality, the backend the driver observed in the logs, and the iteration count.
- **Speed across backends** is the same matrix transposed for a quick look.
- **Changes** compares every ok cell with the previous finished run: wall median and peak RSS moves over 5%, and
  whether the engine, ggml or model pins changed in between. A slower cell with unchanged pins usually means
  another user loaded the machine, which the under-load flag confirms when that load was there as the cell started;
  a changed pin points at a commit or a model cut.
- **Problems** lists build failures, timeouts, every non-ok cell with the driver's note, and every cell measured
  under load with its busy share.
- **Under load** marks a cell that started while other work kept the machine busy. Just before each cell the driver
  measures, over one second (`HOST_LOAD_WINDOW_SECONDS` in `lib.sh`), the share of the CPUs that were busy (from
  `/proc/stat` on Linux, from `top` on macOS) and the one-minute load average, and stores both in
  `results/<backend>/<family>.load.json`. Nothing of the bench runs during that second, so the share is other work
  on the machine. At 10% or more (`CONTENDED_BUSY_PERCENT` in `summarize.py`) the cell's status reads
  `ok (under load)`, Problems lists it with the share, and the machine's dashboard and the hub's task pages mark it.
  The measurement stays in the results, but its time can be slower than the machine's own. The sample is taken once,
  so load that begins during a long cell is not caught. A resumed run keeps the sample of every cell it reuses.

## Adding or changing a model

Add a family to `families.json` with the driver fields, the weekly fields and its `provision` list, keep registry
keys exact (`qvac_models_compiled/ggml/<key>`), pin HuggingFace files to a revision, and list a derived file after
the file it is made from. `test-desktop-weekly.sh` checks that every `${MODEL_DIR}` file in `args` is provisioned
and that requantized cuts follow their source. Families run in the order of the file.

## Shared-machine rules

- Both machines are shared. The run waits for the lock and for other users' GPU work; never delete someone else's
  lock or kill their processes.
- Keep `BENCH_HOME` small; a shared `MODELS_DIR` saves disk but trusts every writer of that folder.
- No root: everything installs under `BENCH_HOME` and the shared model folder.
- One weekly run per machine is enough; run on demand with `--quick` or `--families` during working hours only
  after checking `who` and `nvidia-smi`.

## Mobile

The matrix and the report are host-agnostic: a mobile lane would keep `families.json` and `report.py`, replace the
build and run steps with the Android or iOS harnesses, and feed `summary.json` rows from the device. AWS Device
Farm can host that lane later; nothing here assumes a particular device host beyond `run-family.sh`.

## Tests

```sh
scripts/benchmarks/desktop-weekly/test-desktop-weekly.sh
```

Checks that every script parses, that `families.json` is complete and self-consistent, exercises the lock, the
watchdog, the build markers, run selection and the offline model provisioning path against temporary
directories, and runs the Python unit tests for `summarize.py`, `report.py`, `narrate.py`, `send-report.py` and
`prepare-derived.py`. The desktop benchmark workflow runs the same script in its plan job.
