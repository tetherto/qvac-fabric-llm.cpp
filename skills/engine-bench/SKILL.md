---
name: engine-bench
description: Benchmark Qwen3.8-27B across inference engines (llama.cpp, qvac-fabric, mlx-lm, oMLX, SGLang, Splash, vLLM, Strix Halo forks, gufo) on the team's shared Tailscale machines - the Mac Studio M3 Ultra, the Strix Halo box and the DGX Spark - and write a report. Covers one request, 1-16 concurrent requests, speculative decoding (MTP, DFlash2) and prefill/decode by prompt depth. Use when asked to benchmark or compare engines, check how qvac-fabric does against other engines, or reproduce the weekly engine bench on one of those machines.
argument-hint: mac | strix | spark [quick | full]
---

# Engine benchmark on the shared Tailscale machines

This skill runs one fixed benchmark method for Qwen3.8-27B on a shared machine, under your own account, and turns the result into a Markdown report. The same method runs weekly on all three machines (Saturday 00:00 Zurich), so your numbers are comparable with those runs.

What you get, per machine:
- llama-bench prefill and decode for the llama.cpp-family engines;
- decode t/s on one request in each speculative mode (`ar` = no speculation, `mtp`, `dflash2`), with draft acceptance;
- total t/s and time to first token at 1-16 concurrent requests, with and without speculation;
- prefill and decode t/s by prompt depth (about 1k to 128k tokens on the Mac);
- a problems list (failed steps, short outputs, notes) and, against an earlier run, every metric that moved by more than 5%.

## Files

| path | what it is |
| --- | --- |
| `scripts/bench_client.py` | the load generator (Python stdlib only); every number in the report comes from it, never from an engine's own benchmark |
| `scripts/client_check.py` | records every failed client call, depth step, request or warm-up in `<run>/errors.jsonl` |
| `scripts/report.py` | `summary.json` (one or more machines) -> Markdown report, optionally compared with earlier runs |
| `machines/<id>/weekly.sh` | entry point: waits for the GPU lock, updates the engines, runs the matrix, writes `runs/<date>/summary.json` and `DONE` or `FAILED` |
| `machines/<id>/update.sh` | installs or updates every engine into its own folder, keeps the previous one if the new one fails, writes `engines.env` |
| `machines/<id>/run_engine.sh` | starts one engine's server in one mode, sends one unmeasured warm-up, runs the client, stops the server |
| `machines/<id>/summarize.py` | one run directory -> `summary.json` (schema 3, the same layout on every machine) |
| `machines/strix-halo/*.conf`, `scheduler.sh`, `status.sh`, `download.sh`, `gpu-check.sh` | Strix extras: build backends, per-engine speculative flags, a user-space scheduler (the box has no cron), resumable downloads |
| `machines/dgx-spark/memguard.sh`, `weights.json` | Spark extras: stops your own engines if free memory drops below 6 GiB; weight provenance |

Everything runs from one folder on the machine, `~/bench-weekly` by default. Three environment variables move things:
- `BENCH_DIR`: the bench folder;
- `MODELS_DIR`: where the weights are;
- `GPU_LOCK`: the lock file, `/tmp/gpu-bench.lock` by default.

## Machines

All three are on the team tailnet. Ask the machine owners for an account; nothing here needs root.

| | mac (`mac-m3-ultra`) | strix (`strix-halo`) | spark (`dgx-spark`) |
| --- | --- | --- | --- |
| host | `qvac-dev-mac-arm64` | `qvac-dev-strix-0` | `spark-d9ca` |
| hardware | Apple M3 Ultra, 60-core GPU, 96 GB | Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB (121 GiB visible) | NVIDIA GB10 (sm_121), 128 GB unified |
| backends | Metal, MLX | Vulkan (RADV), ROCm 7.2.4 | CUDA 13 |
| engines | llama.cpp master, qvac-fabric, mlx-lm, oMLX, SGLang (MLX), Splash | llama.cpp master, strix-llama.cpp, qvac-fabric, Nathanw1014 fork, paoai engine, gufo | llama.cpp master, qvac-fabric, vLLM, SGLang |
| target weights | pure Q4_0 GGUF vs MLX 4-bit (same bytes per weight) | UD-Q4_K_XL GGUF for every engine | NVFP4: the ModelOpt checkpoint for vLLM/SGLang, a GGUF converted from it for llama.cpp |
| full run | about 4 h 10 min (1.5 h of it the ~128k depth level) | about 2 h 45 min | about 2 h 30 min |
| `--quick` | about 30 min | about 30 min | about 40 min |

SSH with your own user name. If the host name fails host-key verification, connect with the machine's Tailscale IP (`tailscale ip -4 <host>`) and accept the key under the IP.

- **mac:** cmake lives in `/opt/homebrew/bin`, which is not on the non-interactive PATH (the scripts add it). The disk is about 90% full: keep your bench folder small. The weights are already shared under `/Users/shared/models/mlx-vs-llama` (the scripts' default `MODELS_DIR`):
  - `mlx/Qwen3.8-27B-4bit` (mlx-community 4-bit), `mlx/Qwen3.8-27B-MTP-4bit` (oMLX's MTP head), `hf/Qwen3.8-27B-DFlash2` (z-lab draft, bf16);
  - `gguf/Qwen3.8-27B-Q4_0-pure.gguf` (every tensor Q4_0, `llama-quantize --pure`, same size as MLX 4-bit) and `gguf/Qwen3.8-27B-DFlash2-Q4_0.gguf`.
  - Splash needs macOS 26 and runs part of its prefill on the Neural Engine.
- **strix:** no cron daemon and no lingering systemd user manager, so `scheduler.sh` stands in for cron (`setsid nohup`; it does not survive a reboot). Downloads run at about 2 MB/s: use `download.sh` (resumable `curl -C -`), a 17.6 GB file takes about 2.5 h. ROCm builds need some headers that are not installed; `update.sh` unpacks them from `apt-get download` into `$BENCH_DIR/deps/root`. Weights, in `MODELS_DIR` (default `~/models/qwen38-27b`):
  - `Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` (17.6 GB, with the MTP layer);
  - `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f8ce46e151f61c6499c99dee6079e1d610`.
  - For ROCm builds of every llama.cpp fork, the scripts set `HIP_LAUNCH_BLOCKING=1` (gfx1151 batched-output bug); the report says so.
- **spark:** the clock is UTC and cron ignores `CRON_TZ`, so cron calls `weekly.sh --cron` at 22:00 and 23:00 UTC on Friday and only the call that is 00:00 Saturday in Zurich runs. The first start of vLLM or SGLang JIT-compiles FlashInfer kernels: always keep `MAX_JOBS=4` (the scripts set it), or parallel `cicc` processes run the box out of memory and stall it for everyone. Weights, in `MODELS_DIR` (default `~/models/qwen38-27b`), as `weights.json` records:
  - `nvfp4-hf/`: `nvidia/Qwen3.8-27B-NVFP4` @ `482ca0f` (ModelOpt: NVFP4 MLP and lm_head, FP8 attention, bf16 MTP head);
  - `gguf/Qwen3.8-27B-NVFP4-modelopt.gguf`: that checkpoint through `convert_hf_to_gguf.py --outtype q8_0` (same NVFP4 bits; FP8 attention dequantized to Q8_0);
  - `dflash2-hf/` (`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4`, bf16) and `dflash2-gguf/` (`z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f`, Q8_0).
  - `weekly.sh --heavy` adds a throughput-tuned profile (CUDA graphs, up to 64 requests, 8k-token prompts, depths to 128k, about 2.5 h).

## Shared-machine rules

These machines are shared by several people, and the benchmark is only meaningful when nothing else runs.

- **Check before you start:** `who`, `ps -eo user,pcpu,etime,comm --sort=-pcpu | head` (on macOS `ps -Ao user,pcpu,etime,comm -r`), and `nvidia-smi` or `amd-smi`/`rocm-smi`. If someone else is running GPU work, wait or ask.
- **One run at a time:** `weekly.sh` waits (up to 6 hours) while the lock file exists or any user's inference process is running, then holds the lock for the whole run. Never delete someone else's lock and never kill another user's process.
- **Ask your user before a long run** (anything over about 30 minutes) during European working hours.
- **Do not schedule your own weekly run:** one already runs on every machine on Saturday from 00:00 Zurich. Run on demand instead.
- **No root:** no sudo, no docker, no changes to drivers, groups, kernel parameters or other users' files. Install only under your bench folder and `~/.local`.
- **Check disk** before installing or downloading (`df -h ~`).

## Method

Every engine gets the same prompts, greedy decoding and one unmeasured warm-up per server start. Prefix and prompt caching are off everywhere, so no engine reuses work between requests:

| engine | flags |
| --- | --- |
| llama.cpp and every fork | `--no-cache-prompt -cram 0 -ctxcp 0`: no prompt reuse, no host-RAM prompt cache (which also saves idle slots), no context checkpoints (which split prompt batches and copy recurrent state; they cost up to 29% prefill at 1k tokens on the Spark) |
| vLLM | `--no-enable-prefix-caching` |
| SGLang | `--disable-radix-cache` (and `--mamba-radix-cache-strategy no_buffer` on MLX) |
| oMLX | `--no-cache` (its in-memory cache is off by default) |
| Splash | `--max-cache-disk 0 --idle-release off` |
| mlx-lm | no switch: its server always splits a chat prompt before the last few template tokens to save a cache there (one small extra forward pass per prompt) |

The prompt corpus is llama.cpp's `docs/**/*.md` at commit `0efe8e4c6`, frozen so prompts stay identical across runs and machines. Prompts longer than the corpus repeat it.

The matrix (`weekly.sh`; the machine scripts trim it to their time budget):

1. **llama-bench:** `-fa 1 -p 512 -n 128 -r 3` for every llama.cpp-family engine.
2. **Concurrency without speculation:** `bench_client.py --mode conc --long-output --concurrency 1,2,4,8,16 --rounds 2 --prompt-tokens 128 --gen 512` (essay prompts of about 250 real tokens, 512 out). llama.cpp: `-np 32 -kvu -c 65536`; vLLM and SGLang: default batching. Splash cannot switch its draft off, so it runs here with DFlash2.
3. **One request per mode:** `--mode spec --fixed-prefix --gen 512`, four prompts (code, rewrite, story, explain), identical prefix for every engine. MTP drafts 3 tokens; DFlash2 uses block 8 (`--spec-draft-n-max 7` in llama.cpp). llama.cpp: `-np 1 -c 16384`; MTP is `--spec-type draft-mtp --spec-draft-n-max 3`, DFlash2 is `-md <draft> -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7`.
4. **Speculation under concurrency:** the client of step 2 at `--concurrency 1,4,8 --rounds 1` for each engine's MTP and DFlash2.
5. **Depth:** one request per prompt length through one server start (`DEPTHS="1024 4096 16384 32768 ..."`, `--concurrency 1 --gen 128`). Prefill t/s is prompt tokens / time to first token; decode t/s is that request's decode rate. Real prompts run 0-28% longer than requested (104857 requested is about 123k real), so size the context for that.
6. **Depth with DFlash2** (Mac) for the engines that can switch it on.

qvac-fabric runs the PR stack branch `dflash-deferred-commit-temp-11259` with `LLAMA_GDN_DEFERRED=1` (deferred recurrent-state commit for speculative decoding).

## Procedure

1. **Pick the machine and scope.** Ask the user which machine (`mac`, `strix` or `spark`) and whether they want a `--quick` pipeline check or a full run, unless they said so. Survey the machine as in the rules above.
2. **Deploy the scripts** from this repository:
   ```sh
   H=<host>; M=<mac-m3-ultra|strix-halo|dgx-spark>; S=skills/engine-bench   # from the repository root
   C=0efe8e4c6bb946fb71327b068b26c17d5d15121e   # llama.cpp commit of the frozen prompt corpus (upstream only)
   ssh $H 'mkdir -p ~/bench-weekly/corpus'
   scp $S/scripts/*.py $S/machines/$M/* $H:bench-weekly/
   ssh $H "curl -sfL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$C | tar xz -C ~/bench-weekly/corpus --strip-components=1 llama.cpp-$C/docs"
   ```
   The corpus is 53 Markdown files (about 520k characters) under `~/bench-weekly/corpus/docs`.
3. **Weights:** check they are where `MODELS_DIR` points (see Machines). Download missing files at the pinned revisions; on strix use `download.sh`.
4. **Engines:** run `update.sh` once by hand on a new account, in the background (it builds llama.cpp and qvac-fabric and installs the Python engines; 30-60 minutes):
   ```sh
   ssh $H 'cd ~/bench-weekly && mkdir -p runs/setup && nohup ./update.sh runs/setup > runs/setup/update.log 2>&1 &'
   ```
   Read `runs/setup/updates.txt`: every engine needs a version line, not `FAILED`. Fix failures from `runs/setup/logs/` before you go on; an engine you cannot install is a missing cell in the report, so say which and why.
5. **Run.** Start the suite detached, so it survives the SSH session, and wait for its marker without polling in the foreground:
   ```sh
   ssh $H 'nohup /bin/bash ~/bench-weekly/weekly.sh --quick < /dev/null > /dev/null 2>&1 &'
   ```
   `--quick` skips the engine updates and runs a smaller matrix into `runs/<date>-quick`. Without it, `weekly.sh` updates every engine first and writes `runs/<date>`. The log is `runs/<date>/weekly.log`; each job logs `== <label> server up HH:MM:SS`. Wait in the background for `runs/<date>/DONE` or `FAILED`, for example with a loop of `test -e` over SSH every 5 minutes.
6. **Report.** Copy `summary.json` back and run `report.py`, comparing with an earlier run of the same machine when there is one:
   ```sh
   scp $H:bench-weekly/runs/<date>/summary.json ./<machine>-<date>.json
   python3 skills/engine-bench/scripts/report.py <machine>-<date>.json [--previous <older>.json] -o report.md
   ```
   Several machines' summaries go into one report, one section each. Sections from different machines are not directly comparable: the weights formats differ.
7. **Check for disturbance before trusting a run.** Read the report's Problems and Changes sections:
   - errors such as `[METAL] ... kIOGPUCommandBufferCallbackErrorInnocentVictim` (another process caused a GPU reset), `Insufficient Memory` or `RemoteDisconnected` mean another user's load hit the run;
   - drops that cluster in one time window (the `job started` column) or hit engines whose version did not change are interference, not code.
   - Re-run only the disturbed jobs on the idle machine with the same builds: move their rows out of `runs/<date>/results.jsonl`, run `run_engine.sh` for each label with `ENGINES_ENV=runs/<date>/engines.env` and the same environment `weekly.sh` uses for that step (`NP`, `CTX`, `DEPTHS`), take the lock as `weekly.sh` does, then run `summarize.py runs/<date>` again.
8. **Deliver.** Give the user the report's headline numbers (fastest engine per mode on one request, best total at 16 requests, the depth curve) and the report file, and say what was missing, disturbed or not like for like. Offer to publish the report as an Artifact page.

## Data layout

`summary.json` (schema 3) is the same on every machine:
- `machine` (`mac-m3-ultra`, `strix-halo`, `dgx-spark`), `date`, `status` (`ok`, `partial` when any step failed, `failed`), `reason`, `missing` (expected cells without a result), `started`, `finished`, `hardware`, `versions`, `updates`;
- `single`: engine, backend, tg128, pp512;
- `spec`: engine, backend, mode, profile, draft, mean, code, rewrite, story, explain, runs, acc (accepted draft tokens in %, null when the engine does not report it);
- `conc`: engine, backend, mode, profile, draft, prompt (mean prompt tokens per request), c, gen (total t/s), req (mean per-request t/s), ttft, p90 (seconds);
- `depth`: engine, backend, mode, profile, depth (real prompt tokens), pp, tg, ttft;
- `errors` (label, step, error, at), `short_outputs` and `short_by_label` (requests that stopped before `--gen` tokens), `jobs` (label -> time its server came up).

Labels in `results.jsonl` are `<kind>-<engine>-<mode>[-<profile>]` (Strix adds the backend) with kinds `conc`, `concspec`, `spec` and `depth`. Engine keys: `up` (llama.cpp master), `fab` (qvac-fabric), `strix`, `nath`, `paoai`, `gufo`, `vllm`, `sglang`, `mlx`, `omlx`, `splash`.

## Reading the results

- Splash always speculates (DFlash2), so leave it out of decode-by-depth comparisons without speculation.
- With DFlash2 under concurrency, llama.cpp drafts and verifies the full block for every sequence, so each step grows with the batch and gains shrink past a few requests; MTP holds up better. oMLX verifies adaptively.
- Unified KV (`-kvu`, used in the concurrency steps) makes flash attention scan every sequence's cells. That costs little with short prompts but a lot with long prompts at 16 requests; compare with `-no-kvu` before reading llama.cpp's long-prompt concurrency as an engine limit.
- Report every result as measured and mark what is not like for like: weights format, draft precision, backend, `HIP_LAUNCH_BLOCKING`, flags a fork lacks.

## Rules

- Benchmarks only: do not commit, push or open pull requests from a benchmark session.
- Publish only where the user asks.
- Never start a run on a machine someone else is using, never touch other users' files or processes, and never run anything as root.
