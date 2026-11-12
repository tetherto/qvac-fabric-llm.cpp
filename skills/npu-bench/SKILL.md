---
name: npu-bench
description: Benchmark Qwen3.5 0.8B, 2B, 4B and 9B on the Strix Halo NPU with qvac-fabric's xdna backend against FastFlowLM (flm), speed and package energy from 1k to 16k prompt tokens, and build the comparison page (Summary, Per model). Use when asked to compare xdna with FastFlowLM, rerun or schedule the weekly NPU bench, or update the NPU dashboard.
argument-hint: quick | full
---

# xdna vs FastFlowLM on the Strix Halo NPU

This skill runs one fixed method on the shared Strix Halo box, under your own account: the latest qvac-fabric xdna build and the latest FastFlowLM release serve the same prompts, one after the other, while the package power is sampled. The result is a `summary.json` and an HTML page in the format of the NPU dashboard.

What you get, for each model and prompt length (1k, 2k, 4k, 8k, 16k tokens), for xdna and flm:
- prefill and decode t/s, and time to first token;
- energy per request, per 1k prefill tokens and per generated token, and tokens per joule;
- a drift check (xdna at 1k repeated after flm), a log of other jobs seen during the run, and failed sessions;
- the page: xdna against flm, all models at once (Summary) and one model at a time (Per model).

## Files

| path | what it is |
| --- | --- |
| `scripts/bench_npu.py` | the client: streaming chat requests, PPT and Tctl sampling, one JSON line per energy window; `sensors` and `prep` subcommands |
| `scripts/summarize.py` | run directory -> `summary.json`; when two sessions measured the same cell, the later one wins |
| `scripts/build_page.py`, `scripts/page.html` | `summary.json` -> the page |
| `machines/strix-halo/weekly.sh` | entry point: waits for the GPU lock and a quiet machine, updates, runs the matrix, writes `runs/<date>/summary.json` and `DONE` or `FAILED` |
| `machines/strix-halo/update.sh` | builds qvac-fabric with the XDNA backend and its NPU kernels, installs the newest flm portable release and its models; a failed update keeps the previous install |
| `machines/strix-halo/setup.sh` | once: rebuilds the frozen prompt set and checks it against `bench.conf` |
| `machines/strix-halo/session.sh` | one server (xdna or flm) for one model and one client session |
| `machines/strix-halo/bench.conf`, `models.conf` | matrix, ports, server flags, prompt hashes; GGUF files, their sha256, flm tags and the xdna server environment |

Everything runs from one folder on the machine, `~/npu-bench` by default (`BENCH_DIR`). `MODELS_DIR` points at the GGUFs, `AIE_PYTHON` at the IRON interpreter, `GPU_LOCK` at the lock file shared with the engine bench (`/tmp/gpu-bench.lock`), `XDNA_BRANCH` at the qvac-fabric branch (`temp-11739`).

## Machine

`qvac-dev-strix-1` on the team tailnet: Ryzen AI MAX+ 395, XDNA 2 NPU (`/dev/accel/accel0`), XRT 2.25 under `/opt/xilinx/xrt`. Ask the machine owners for an account; nothing here needs root.

- **IRON:** the kernel build needs a Python with mlir-aie 1.4.3 and llvm-aie (see `ggml/src/ggml-xdna/README.md`). Without it the backend builds with no kernels and runs on the host; `update.sh` refuses that build.
- **flm:** `update.sh` installs `fastflowlm_<version>_linux.tar.gz` from the newest `ROCm/FastFlowLM` release into `engines/flm-<version>` and keeps its models in `flm-models` (`FLM_MODEL_PATH`), apart from any system flm.
- **Models,** in `MODELS_DIR` (default `~/xdna-port/models`), with `ssm_out` requantized to Q5_K so the fused decode layer takes them: `llama-quantize --tensor-type ssm_out=q5_K <bf16> Qwen3.5-<size>-Q4_K_M-so5.gguf Q4_K_M`, from `ggml-org/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B-BF16.gguf`) and `bartowski/Qwen_Qwen3.5-<size>-GGUF` (`Qwen_Qwen3.5-<size>-bf16.gguf`). `models.conf` holds their sha256; `weekly.sh` checks every file.
- **Schedule:** the box has no cron daemon. The weekly run uses the engine bench's `machines/strix-halo/scheduler.sh` and `status.sh` with `BENCH_DIR=~/npu-bench`: they run `weekly.sh` at 00:00 every Saturday, Europe/Zurich. Set `TV_DIR` to a folder without `turbovec.sh` too, or the scheduler also runs the TurboVec bench after `weekly.sh`. Only the owner of the weekly run starts that scheduler; everyone else runs on demand.

## Shared-machine rules

The engine bench's rules apply (`skills/engine-bench/SKILL.md`, Shared-machine rules). On top of them:
- **One NPU job at a time.** Never start a session beside another NPU process. Two processes on the NPU have broken it until a reboot.
- **Energy needs an idle machine.** `weekly.sh` waits (up to 6 h) for the lock, for no other inference or build job of any user, fewer than one busy core and an idle GPU, and checks again before every session. It never kills another user's process.
- **Disk:** the root disk is nearly full; check `df -h ~` before a first install. The engines take about 0.5 GB; the four flm models took 17.6 GB in flm 0.9.43's format (1.0.12's 0.8B is 0.74 GB).

## Method

Both runtimes get the same prompts: token-exact slices of qvac-fabric's `README.md` and `docs/*.md` at the commit in `bench.conf`, each request with a unique first line, no prompt cache and `ignore_eos`. Requests are streaming chat completions at temperature 0.

| runtime | server |
| --- | --- |
| xdna | `llama-server -ngl 99 --device XDNA0 -fa on -np 4 -b 4096 -ub 4096 -t 16 -c 147456`; `OMP_WAIT_POLICY=PASSIVE` for the 0.8B, unset for the larger models (the backend README's advice) |
| flm | `flm serve <tag>` with its defaults (performance power mode) |

Per model and prompt length (`bench.conf`): wait for Tctl to fall to 47 C, 2 warm-up requests, 8 timed requests of 512 generated tokens (each its own energy window), one burst of one-token requests over at least 8192 prompt tokens, one tiny request. Each window starts 1 s before its first request and ends 10 s after its last, because the NPU stays powered about 5 s after a job and PPT lags about 1 s. After the flm session, a short xdna session at 1k checks for drift.

- Prefill t/s = prompt tokens / TTFT, decode t/s = completion tokens / (total - TTFT), both over the timed requests; TTFT is measured at the client.
- Power is the package power PPT (amdgpu hwmon, which includes the NPU) every 0.1 s, minus the mean of the 1 s before a window and the last 1 s of its tail.
- Prefill J per 1k tokens = (burst energy - tiny request) / prompt tokens. Decode J per token = (timed request - tiny request - its prompt at the prefill energy rate) / (completion tokens - 1). Tokens per joule = 1 / decode J per token.

`--quick` runs the 0.8B at 1k with 2 short timed requests and no updates: a pipeline check, not a result.

## Procedure

1. **Pick the scope.** Ask whether the user wants a `--quick` check or a full run (about 6 h), unless they said so. Survey the machine as in the rules above.
2. **Deploy the scripts** from this repository:
   ```sh
   H=qvac-dev-strix-1; S=skills/npu-bench   # from the repository root
   ssh $H 'mkdir -p ~/npu-bench'
   scp $S/scripts/bench_npu.py $S/scripts/summarize.py $S/machines/strix-halo/* $H:npu-bench/
   ```
3. **Models:** check that `MODELS_DIR` holds the four GGUFs of `models.conf`; quantize missing ones as above.
4. **Engines,** once by hand on a new account, in the background (the fabric build takes about 2 minutes; the first run also downloads the flm models):
   ```sh
   ssh $H 'cd ~/npu-bench && mkdir -p runs/setup && nohup ./update.sh runs/setup > runs/setup/update.log 2>&1 &'
   ```
   Read `runs/setup/updates.txt`: both engines need a version line, not `FAILED`.
5. **Prompts,** once: `ssh $H '~/npu-bench/setup.sh'` must end with `SETUP_OK`.
6. **Run** detached and wait for `runs/<date>/DONE` or `FAILED` in the background, for example with a loop of `test -e` over SSH every 5 minutes:
   ```sh
   ssh $H 'nohup /bin/bash ~/npu-bench/weekly.sh --quick < /dev/null > /dev/null 2>&1 &'
   ```
   The log is `runs/<date>/weekly.log`; each session logs one line per energy window.
7. **Page.** Copy `summary.json` back and build the page:
   ```sh
   scp $H:npu-bench/runs/<date>/summary.json ./npu-<date>.json
   python3 skills/npu-bench/scripts/build_page.py npu-<date>.json -o npu-<date>.html
   ```
8. **Check for disturbance before trusting a run.** Read `errors`, `missing`, `drift` and `foreign` in `summary.json` (the page's Method section repeats them). A drift above 2%, or another job in `foreign.log` during a session, marks those cells as disturbed. Re-run only the disturbed sessions on the quiet machine with `session.sh <run dir> <model> <xdna|flm> rerun <the same arguments>` under the lock, then run `summarize.py <run dir>` again: the later session replaces the earlier one cell by cell.
9. **Deliver.** Give the user the headline (cells ahead of flm per metric, the largest gaps), the page, and what was missing, disturbed or not like for like. Publish or update the dashboard Artifact only where the user asks.

## Data layout

A run directory `runs/<date>` holds `weekly.log`, `run.json` (machine, versions, models, flags, matrix), `results.jsonl`, `foreign.log`, `errors.log`, `logs/` (servers, updates), copies of `engines.env`, `bench.conf` and `models.conf`, and `summary.json`.

- `results.jsonl`: one energy window per line: `label` (`<model>|<runtime>|<session>`, sessions `main`, `drift` or a re-run name), `ctx` (0 for the tiny request), `kind` (`full`, `burst`, `tiny`), `reqs` (start, ttft, total, end, prompt and completion tokens), `idle_w`, `energy_j`, `tctl`.
- `summary.json` (schema 1): the `run.json` fields plus `status` (`ok`, `partial` when a cell is missing, `failed`), `reason`, `contexts`, `models_order`, `runtimes`, `cells` (`[model][metric][runtime][context] = [mean, sd]` or null; metrics `prefill`, `decode`, `ttft`, `req`, `pj`, `dj`, `tpj`), `tokens` (mean completion tokens of the timed requests), `drift`, `missing`, `foreign`, `errors`.

## Reading the results

- flm loads its own quantised weights from its model store, not the GGUF, so every xdna/flm cell compares runtimes and models at once.
- flm may end a request before 512 tokens. Its decode rate is still valid, but energy cells whose requests averaged fewer than 256 tokens carry an asterisk, and a decode energy that computes to zero or below is n/a.
- Where decode runs depends on the build: on `temp-11739` the fused decode layer runs all four sizes on the NPU; with `GGML_XDNA_FUSED_WIDE=0` the 2B to 9B decode on the host CPU. State it with every decode number.
- Energy is package power with the idle baseline removed. Compare energy only between runs on the same machine and sensor.
- Prefill and TTFT come from timed requests that each follow 10 s of idle, so they include the NPU's resume; back-to-back requests prefill faster (0.8B at 1k on xdna, 2026-11-02: 1928 against 1514 tokens/s).

## Rules

- Benchmarks only: do not commit, push or open pull requests from a benchmark session.
- Publish only where the user asks.
- Never start a run on a machine someone else is using, never touch other users' files or processes, and never run anything as root.
