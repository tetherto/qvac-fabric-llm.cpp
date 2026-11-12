---
name: engine-bench
description: Benchmark Qwen3.8-27B across inference engines (llama.cpp, qvac-fabric, mlx-lm, oMLX, SGLang, Splash, Uzu, TensorFold, vLLM, Strix Halo forks, gufo) on the team's shared machines - the Mac Studio M3 Ultra, the Strix Halo box, the DGX Spark, a pair of DGX Sparks and one H100 - and write a report. Covers one request, 1-16 concurrent requests, speculative decoding (MTP, DFlash2) and prefill/decode by prompt depth; also the TurboVec CPU bench of qvac-fabric's vector index. Use when asked to benchmark or compare engines, check how qvac-fabric does against other engines, or reproduce the weekly engine bench on one of those machines.
argument-hint: mac | strix | spark | spark-x2 | h100 | turbovec [quick | full]
---

# Engine benchmark on the shared machines

This skill runs one fixed benchmark method for Qwen3.8-27B on a shared machine, under your own account, and turns the result into a Markdown report. The same method runs weekly on every machine below (Saturday, from 00:00 Zurich), so your numbers are comparable with those runs.

What you get, per machine:
- prefill and decode on one sequence for the llama.cpp-family engines (llama-bench; on the Mac, llama-server single requests);
- decode t/s on one request in each speculative mode (`ar` = no speculation, `mtp`, `dflash2`), with draft acceptance;
- total t/s and time to first token at 1-16 concurrent requests, with and without speculation;
- prefill and decode t/s by prompt depth (about 1k to 128k tokens on the Mac and the H100);
- a problems list (failed steps, short outputs, notes) and, against an earlier run, every metric that moved by more than 5%.

`turbovec/` is a separate, smaller weekly CPU bench of qvac-fabric's vector index against the `turbovec` package; see its README.

## Files

| path | what it is |
| --- | --- |
| `scripts/bench_client.py` | the load generator (Python stdlib only); every number in the report comes from it, never from an engine's own benchmark. `--chat` sends chat requests for servers without `/v1/completions` |
| `scripts/client_check.py` | records every failed client call, depth step, request or warm-up in `<run>/errors.jsonl` |
| `scripts/report.py` | `summary.json` (one or more machines) -> Markdown report, optionally compared with earlier runs |
| `machines/<id>/weekly.sh` | entry point: waits for the GPU lock, updates the engines, runs the matrix, writes `runs/<date>/summary.json` and `DONE` or `FAILED` |
| `machines/<id>/update.sh` | installs or updates every engine into its own folder, keeps the previous one if the new one fails, writes `engines.env` |
| `machines/<id>/run_engine.sh` | starts one engine's server in one mode, sends one unmeasured warm-up, runs the client, stops the server |
| `machines/<id>/summarize.py` | one run directory -> `summary.json` (schema 3, the same layout on every machine) |
| `machines/mac-m3-ultra/single_server.py`, `uzu_server.py`, `ssh-entry.sh` | Mac extras: the llama-server single-request step, an OpenAI-style server for Uzu, the forced command for the cron-over-ssh trigger |
| `machines/strix-halo/*.conf`, `scheduler.sh`, `status.sh`, `download.sh`, `gpu-check.sh` | Strix extras: build backends, per-engine speculative flags, a user-space scheduler (the box has no cron), resumable downloads |
| `machines/dgx-spark/memguard.sh`, `weights.json` | Spark extras: stops your own engines if free memory drops below 6 GiB; weight provenance |
| `machines/dgx-spark-x2/pair.conf`, `build.sh`, `rpc.sh`, `gpu-busy.sh`, `install-vllm.sh` | 2x Spark extras: link settings, builds with the RPC backend, the rpc-server on both nodes, the busy check and vLLM install on the second node |
| `machines/h100/scheduler.sh`, `status.sh`, `stop-bench.sh`, `weights.json` | H100 extras: a user-space scheduler (the container has no cron), status, a safe stop |
| `turbovec/` | the TurboVec CPU bench (its own README) |

Everything runs from one folder on the machine. Environment variables move things:
- `BENCH_DIR`: the bench folder (`~/bench-weekly`; `~/bench-pair` for the Spark pair);
- `MODELS_DIR`: where the weights are;
- `GPU_LOCK`: the lock file, `/tmp/gpu-bench.lock` by default (see the rules below);
- per machine: `TF_HF_HOME` and `FAB_BRANCH` (Mac), `BENCH_GPU` (H100, default 7), the link settings in `pair.conf` (Spark pair), `TV_DIR` (TurboVec, default `~/bench-turbovec`).

## Machines

Ask the machine owners for an account; nothing here needs root. SSH with your own user name; if a host name fails host-key verification, connect with its Tailscale IP (`tailscale ip -4 <host>`) and accept the key under the IP.

| id | host | hardware, backend | engines (keys) | target weights | full run |
| --- | --- | --- | --- | --- | --- |
| `mac-m3-ultra` (mac) | `qvac-dev-mac-arm64` | Apple M3 Ultra, 60-core GPU, 96 GB; Metal, MLX | llama.cpp (`up`), qvac-fabric (`fab`), mlx-lm (`mlx`), oMLX (`omlx`), SGLang on MLX (`sglang`), Splash (`splash`), Uzu (`uzu`), TensorFold (`tf`) | pure Q4_0 GGUF vs MLX 4-bit (same bytes per weight) | about 5 h 20 min (depth to ~123k is 3 h 40 min of it) |
| `strix-halo` (strix) | `qvac-dev-strix-0` | Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 121 GiB visible; Vulkan, ROCm 7.2.4 | llama.cpp (`up`), strix-llama.cpp (`strix`), qvac-fabric (`fab`), Nathanw1014 fork (`nath`), paoai engine (`paoai`), gufo (`gufo`, ROCm) | UD-Q4_K_XL GGUF for every engine | about 2 h 10 min |
| `dgx-spark` (spark) | `spark-d9ca` | NVIDIA GB10 (sm_121), 128 GB unified; CUDA 13 | llama.cpp (`up`), qvac-fabric (`fab`), vLLM (`vllm`), SGLang (`sglang`) | NVFP4: the ModelOpt checkpoint for vLLM/SGLang, a GGUF converted from it for llama.cpp | about 2 h 20 min |
| `dgx-spark-x2` (spark-x2) | `spark-1ff6` + `spark-d9ca` | two GB10 on a direct ConnectX-7 cable (RoCE) | llama.cpp and qvac-fabric over RPC `-sm tensor` (`up`, `fab`), vLLM TP2 (`vllm`) | the same NVFP4 files as the single Spark | about 1 h 15 min |
| `h100` | `cosmicac-a26cbb09` | one H100 80 GB (GPU 7 of a shared 8x H100 container); CUDA | llama.cpp (`up`), qvac-fabric (`fab`, `fab0` = deferred commit off), vLLM (`vllm`), SGLang (`sglang`) | Q8_0 GGUF vs FP8 (`Qwen/Qwen3.8-27B-FP8`) | about 1 h 15 min |

`weekly.sh --quick` (no updates, a smaller matrix) takes 30-40 minutes on each.

- **mac:** cmake lives in `/opt/homebrew/bin`, which is not on the non-interactive PATH (the scripts add it). The disk is about 94% full: keep your bench folder small. The weights are shared under `/Users/shared/models/mlx-vs-llama` (the default `MODELS_DIR`):
  - `mlx/Qwen3.8-27B-4bit` (mlx-community 4-bit), `mlx/Qwen3.8-27B-MTP-4bit` (oMLX's MTP head), `hf/Qwen3.8-27B-DFlash2` (z-lab draft, bf16);
  - `gguf/Qwen3.8-27B-Q4_0-pure.gguf` (every tensor Q4_0, `llama-quantize --pure`, same size as MLX 4-bit) and `gguf/Qwen3.8-27B-DFlash2-Q4_0.gguf`;
  - `uzu/Qwen3.8-27B-M` (`trymirai/Qwen3.8-27B-M` @ `3d1fe988a0ee35770a5051f3c89912f383adc46e`, with a hand-written `encoding.json` of `{"name":"qwen3.8","type":"hanashi"}`, which the HF repo lacks) and `uzu/Qwen3.8-27B-M-nospec` (the same without `speculator/`, for `ar`);
  - TensorFold reads `TensorFold/Qwen3.8-27B-MLX-4bit` @ `22d8d538` and `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` from an HF cache, `TF_HF_HOME` (default `/Users/Shared/huggingface`, a shared cache).
  - qvac-fabric builds `temp-11739` with `-DLLAMA_COREML=ON` and runs with the Core ML FFN sidecar (`--coreml-ffn`, part of each prefill FFN on the Neural Engines); its first start exports the programs into an 8.4 GB cache (1-2 min). llama-bench never applies `--coreml-ffn`, which is why step 1 runs llama-server here.
  - Splash needs macOS 26 and runs part of its prefill on the Neural Engine. Uzu runs `ar` only (its drafter has no M3 Ultra tree shapes) and queues concurrent requests.
  - TensorFold's draft acceptance is only in its server log. Never run `tensorfold benchmark`: it uploads results to tensorfold.dev.
  - Jobs started directly by cron run at a lower QoS on macOS and decode 25-30% slower. The weekly run is therefore started from cron through `ssh localhost`: `0 0 * * 6 ssh -i ~/.ssh/bench_localhost -o BatchMode=yes localhost weekly`, with that key's `authorized_keys` entry limited to `from="127.0.0.1,::1",command="/bin/bash <bench>/ssh-entry.sh"` and no forwarding.
- **strix:** no cron daemon and no lingering systemd user manager, so `scheduler.sh` stands in for cron (`setsid nohup`; it does not survive a reboot; it also runs the TurboVec bench after `weekly.sh`). Downloads run at about 2 MB/s: use `download.sh` (resumable `curl -C -`), a 17.6 GB file takes about 2.5 h. ROCm builds need some headers that are not installed; `update.sh` unpacks them from `apt-get download` into `$BENCH_DIR/deps/root`. The disk is about 92% full. Weights, in `MODELS_DIR` (default `~/models/qwen38-27b`):
  - `Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` (17.6 GB, with the MTP layer);
  - `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f8ce46e151f61c6499c99dee6079e1d610`.
  - For ROCm builds of every llama.cpp fork, the scripts set `HIP_LAUNCH_BLOCKING=1` (gfx1151 batched-output bug); the report says so.
  - gufo has stayed at 0.22.0 since 2026-10-30: newer releases fail cmake for a missing `sndfile` pkg-config file.
- **spark:** the clock is UTC and cron ignores `CRON_TZ`, so cron calls `weekly.sh --cron` at 22:00 and 23:00 UTC on Friday and only the call that is 00:00 Saturday in Zurich runs. The first start of vLLM or SGLang JIT-compiles FlashInfer kernels: always keep `MAX_JOBS=4` (the scripts set it), or parallel `cicc` processes run the box out of memory and stall it for everyone. SGLang DFlash2 runs with `--max-running-requests 8`: it reserves recurrent state per running request and draft token, and the default leaves almost no KV pool. Weights, in `MODELS_DIR` (default `~/models/qwen38-27b`), as `weights.json` records:
  - `nvfp4-hf/`: `nvidia/Qwen3.8-27B-NVFP4` @ `482ca0f` (ModelOpt: NVFP4 MLP and lm_head, FP8 attention, bf16 MTP head);
  - `gguf/Qwen3.8-27B-NVFP4-modelopt.gguf`: that checkpoint through `convert_hf_to_gguf.py --outtype q8_0` (same NVFP4 bits; FP8 attention dequantized to Q8_0);
  - `dflash2-hf/` (`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4`, bf16) and `dflash2-gguf/` (`z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f`, Q8_0).
  - `weekly.sh --heavy` adds a throughput-tuned profile (CUDA graphs, up to 64 requests, 8k-token prompts, depths to 128k, about 2.5 h; SGLang DFlash2 needs `SGL_MAXREQ=32` there).
- **spark-x2:** the head node runs the scripts, llama-server, vLLM rank 0 and the client; the second node runs a `ggml-rpc-server` and vLLM rank 1 (`--headless`). vLLM runs TP2 without Ray (`--distributed-executor-backend mp --nnodes 2 --node-rank 0|1 --master-addr <head>`), NCCL over both RoCE functions. Requirements, all in `pair.conf`:
  - a direct cable on the second ConnectX-7 port with addresses on both nodes (defaults 10.10.30.1/.2 on `enp1s0f1np1` and 10.10.40.x on `enP2p1s0f1np1`);
  - passwordless ssh from the head to the second node over that link (`BatchMode`; restrict the key with `from=`);
  - the bench folder, models, builds and venvs at the same absolute paths on both nodes (`update.sh` builds on the head and rsyncs to the peer; `gpu-busy.sh` and `install-vllm.sh` must be on the peer too).
  - DFlash2 is left out for llama.cpp and qvac-fabric: its draft shares the lm_head, which `-sm tensor` splits. qvac-fabric runs with `LLAMA_GDN_DEFERRED=0`.
  - vLLM can fail its memory-profiling check while the previous engine still frees memory; `run_engine.sh` waits for free memory to settle. Ports: 8097 (server), 50070 (RPC), 29517 (vLLM master). Cron on the head: `0 2,3 * * 6 weekly.sh --cron` (proceeds at 04:00 Zurich, after the single Spark's 00:00 run).
- **h100:** a shared 8x H100 container; the bench uses one GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${BENCH_GPU:-7}`) and checks before it starts that the GPU has no compute process, no load and less than 2 GB in use. Other users can still take its memory mid-run (both runs of 10-31 and 11-07 ended `partial`); check the errors. The container has no cron and no time-zone data: start `scheduler.sh` detached (`setsid nohup $BENCH_DIR/scheduler.sh >> $BENCH_DIR/cron.log 2>&1 < /dev/null &`, again after a container restart); it computes Zurich time with a POSIX TZ string. vLLM and SGLang compile their FP8 kernels on first use and need a CUDA toolkit matching torch's CUDA, newer than the box's nvcc; `update.sh` installs NVIDIA's `cuda-toolkit` wheels into `$BENCH_DIR/cuda/<ver>`. First starts compile for 5-14 min (cached in `~/.cache`). Prerequisites no script creates: `uv` in `~/.local/bin` and `$BENCH_DIR/venvs/tools` with cmake, ninja and `huggingface_hub`. Weights, in `MODELS_DIR` (`weights.json`): `Qwen/Qwen3.8-27B-FP8` @ `017b9c7`, unsloth `Qwen3.8-27B-Q8_0.gguf` @ `4ca7207` (with the MTP layer), `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4` and its Q8_0 GGUF @ `2d9571f`. Stop a run with `stop-bench.sh`, never with `pkill -f` typed into ssh.

## Shared-machine rules

These machines are shared by several people, and the benchmark is only meaningful when nothing else runs.

- **Check before you start:** `who`, `ps -eo user,pcpu,etime,comm --sort=-pcpu | head` (on macOS `ps -Ao user,pcpu,etime,comm -r`), and `nvidia-smi` or `amd-smi`/`rocm-smi`. If someone else is running GPU work, wait or ask.
- **One run at a time:** `weekly.sh` waits (up to 6 hours) while the lock file exists or any user's inference process is running, then holds the lock for the whole run. The weekly runs already deployed use their owner's lock file (on the Mac `~/bench-1019/GPU_BUSY`, elsewhere `~/bench-weekly/GPU_BUSY` in the owner's home); look at it before you start, and set `GPU_LOCK` to a path everyone can see if you coordinate with others. Never delete someone else's lock and never kill another user's process.
- **Ask your user before a long run** (anything over about 30 minutes) during European working hours.
- **Do not schedule your own weekly run:** one already runs on every machine on Saturday. Run on demand instead.
- **No root:** no sudo, no docker, no changes to drivers, groups, kernel parameters or other users' files. Install only under your bench folder and `~/.local`.
- **Check disk** before installing or downloading (`df -h ~`).
- **Shell patterns:** `pkill -f`/`pgrep -f` inside `ssh host '...'` match the remote shell's own command line (and can kill your session); use a bracket pattern such as `[l]lama-server`.

## Method

Every engine gets the same prompts, greedy decoding and one unmeasured warm-up per server start (the H100 adds unmeasured bursts of 2-16 requests, since SGLang's first batched prefill costs seconds once). Prefix and prompt caching are off everywhere, so no engine reuses work between requests:

| engine | flags |
| --- | --- |
| llama.cpp and every fork | `--no-cache-prompt -cram 0 -ctxcp 0`: no prompt reuse, no host-RAM prompt cache (which also saves idle slots), no context checkpoints (which split prompt batches and copy recurrent state; they cost up to 29% prefill at 1k tokens on the Spark) |
| vLLM | `--no-enable-prefix-caching` |
| SGLang | `--disable-radix-cache` (and `--mamba-radix-cache-strategy no_buffer` on MLX) |
| oMLX | `--no-cache` (its in-memory cache is off by default) |
| Splash | `--max-cache-disk 0 --idle-release off` |
| TensorFold | `--prompt-cache-gib 0 --no-thinking` |
| Uzu | its server resets the session for every request |
| mlx-lm | no switch: its server always splits a chat prompt before the last few template tokens to save a cache there (one small extra forward pass per prompt) |

The prompt corpus is llama.cpp's `docs/**/*.md` at commit `0efe8e4c6`, frozen so prompts stay identical across runs and machines. Prompts longer than the corpus repeat it. Uzu has no `/v1/completions`, so its client calls use `--chat`.

The matrix (`weekly.sh`):

1. **One sequence:** llama-bench `-fa 1 -p 512 -n 128 -r 3` for every llama.cpp-family engine. On the Mac: `single_server.py` against llama-server, prompts of 512 and 4096 tokens and 128 out, written in llama-bench's shape with `"tool": "llama-server"`.
2. **Concurrency without speculation:** `bench_client.py --mode conc --long-output --concurrency <levels> --rounds <n> --prompt-tokens 128 --gen 512` (essay prompts of about 250 real tokens, 512 out). llama.cpp: `-np 32 -kvu -c 65536`; vLLM and SGLang: default batching. Splash cannot switch its draft off, so it runs here with DFlash2.
3. **One request per mode:** `--mode spec --fixed-prefix --gen 512`, four prompts (code, rewrite, story, explain), identical prefix for every engine. MTP drafts 3 tokens; DFlash2 uses block 8 (`--spec-draft-n-max 7` in llama.cpp). llama.cpp: `-np 1 -c 16384`; MTP is `--spec-type draft-mtp --spec-draft-n-max 3`, DFlash2 is `-md <draft> -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7`.
4. **Speculation under concurrency:** the client of step 2 for each engine's MTP and DFlash2 (llama.cpp `-np 8 -c 32768`).
5. **Depth:** one request per prompt length through one server start (`DEPTHS="1024 4096 16384 32768 ..."`, `--concurrency 1 --gen 128`). Prefill t/s is prompt tokens / time to first token; decode t/s is that request's decode rate. Real prompts run 0-28% longer than requested (104857 requested is about 123k real), so size the context for that.
6. **Depth with DFlash2** (Mac) for the engines that can switch it on.

How far each machine runs it:

| machine | one request (runs per prompt) | concurrency (rounds) | speculation under concurrency | depth (requested) |
| --- | --- | --- | --- | --- |
| mac | 2 | 1, 2, 4, 8, 16 (2) | 1, 4, 8 | 1k-104857, also with DFlash2 |
| strix | 1 | 1, 4, 16 (1) | 4 | 1k-32k |
| spark | 2 | 1, 4, 16 (2) | 4, 8 | 1k-32k |
| spark-x2 | 2 | 1, 4, 16 (2) | 4 (and 8 while the run is under 1 h 45 min) | 1k-32k |
| h100 | 2 | 1, 2, 4, 8, 16 (2) | 1, 4, 8 | 1k-100000 |

qvac-fabric runs with `LLAMA_GDN_DEFERRED=1` (deferred recurrent-state commit for speculative decoding; `0` on the Spark pair and for `fab0`). It builds `temp-11739` on the Mac (with Core ML) and `temp-11259` elsewhere; `FAB_BRANCH` overrides the Mac's branch.

## Procedure

1. **Pick the machine and scope.** Ask the user which machine and whether they want a `--quick` pipeline check or a full run, unless they said so. Survey the machine as in the rules above.
2. **Deploy the scripts** from this repository:
   ```sh
   H=<host>; M=<mac-m3-ultra|strix-halo|dgx-spark|dgx-spark-x2|h100>; S=skills/engine-bench   # from the repository root
   B='~/bench-weekly'   # '~/bench-pair' for dgx-spark-x2
   C=0efe8e4c6bb946fb71327b068b26c17d5d15121e   # llama.cpp commit of the frozen prompt corpus (upstream only)
   ssh $H "mkdir -p $B/corpus"
   scp $S/scripts/*.py $S/machines/$M/* $H:${B#\~/}/
   ssh $H "curl -sfL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$C | tar xz -C $B/corpus --strip-components=1 llama.cpp-$C/docs"
   ```
   The corpus is 53 Markdown files (about 520k characters) under `corpus/docs`. For `dgx-spark-x2`, deploy to both nodes at the same path, check `pair.conf`, and use `~/bench-pair` in the commands below.
3. **Weights:** check they are where `MODELS_DIR` points (see Machines). Download missing files at the pinned revisions; on strix use `download.sh`.
4. **Engines:** run `update.sh` once by hand on a new account, in the background (it builds llama.cpp and qvac-fabric and installs the Python engines; 30-60 minutes, longer on the first vLLM/SGLang start):
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
   - errors such as `[METAL] ... kIOGPUCommandBufferCallbackErrorInnocentVictim` (another process caused a GPU reset), `Insufficient Memory`, CUDA out of memory or `RemoteDisconnected` mean another user's load hit the run;
   - drops that cluster in one time window (the `job started` column) or hit engines whose version did not change are interference, not code.
   - Re-run only the disturbed jobs on the idle machine with the same builds: move their rows out of `runs/<date>/results.jsonl`, run `run_engine.sh` for each label with `ENGINES_ENV=runs/<date>/engines.env` and the same environment `weekly.sh` uses for that step (`NP`, `CTX`, `DEPTHS`), take the lock as `weekly.sh` does, then run `summarize.py runs/<date>` again.
8. **Deliver.** Give the user the report's headline numbers (fastest engine per mode on one request, best total at 16 requests, the depth curve) and the report file, and say what was missing, disturbed or not like for like. Offer to publish the report as an Artifact page.

## Data layout

`summary.json` (schema 3) is the same on every machine:
- `machine` (`mac-m3-ultra`, `strix-halo`, `dgx-spark`, `dgx-spark-x2`, `h100`), `date`, `status` (`ok`, `partial` when any step failed, `failed`), `reason`, `missing` (expected cells without a result), `started`, `finished`, `hardware`, `host`, `versions`, `updates`, and where the machine has them `provenance` (weights), `checks`, `nodes` and `link` (the pair);
- `single`: engine, backend, tool (`llama-bench` or `llama-server`), tg128, pp512 (and pp4096 on the Mac);
- `spec`: engine, backend, mode, profile, draft, mean, code, rewrite, story, explain, runs, acc (accepted draft tokens in %, null when the engine does not report it);
- `conc`: engine, backend, mode, profile, draft, prompt (mean prompt tokens per request), c, gen (total t/s), req (mean per-request t/s), ttft, p90 (seconds);
- `depth`: engine, backend, mode, profile, depth (real prompt tokens), pp, tg, ttft;
- `errors` (label, step, error, at), `short_outputs` and `short_by_label` (requests that stopped before `--gen` tokens), `jobs` (label -> time its server came up).

Labels in `results.jsonl` are `<kind>-<engine>-<mode>[-<profile>]` (Strix adds the backend) with kinds `single`, `conc`, `concspec`, `spec` and `depth`. Engine keys: `up` (llama.cpp master), `fab` (qvac-fabric), `fab0` (qvac-fabric, deferred commit off), `strix`, `nath`, `paoai`, `gufo`, `vllm`, `sglang`, `mlx`, `omlx`, `splash`, `uzu`, `tf`.

## TurboVec

`turbovec/` benchmarks qvac-fabric's vector index (`ggml-vector-index`, 2- and 4-bit) against the PyPI `turbovec` package on the CPU: 100k random unit vectors at dim 1024 and 1,000 queries; batch add and search on 1 and 8 threads, single queries at k 10 and 64 (p50/p95), and recall@1/10/64 against an exact search. It builds only the vector-index target from the newest `temp-<n>` branch. `turbovec.sh --after-llm` runs right after each machine's Saturday LLM bench (Mac, Spark, Strix), once the LLM run has `DONE` or `FAILED` and the load is low; it takes 3-11 minutes and about 0.7 GB. Deploy and run it as `turbovec/README.md` says.

## Reading the results

- Splash always speculates (DFlash2), so leave it out of decode-by-depth comparisons without speculation.
- With DFlash2 under concurrency, llama.cpp drafts and verifies the full block for every sequence, so each step grows with the batch and gains shrink past a few requests; MTP holds up better. oMLX and TensorFold shorten or stop drafting under load (TensorFold drafts almost nothing at 8 requests), so their DFlash2 numbers at 8 requests are close to their plain decode.
- Unified KV (`-kvu`, used in the concurrency steps) makes flash attention scan every sequence's cells. That costs little with short prompts but a lot with long prompts at 16 requests; compare with `-no-kvu` before reading llama.cpp's long-prompt concurrency as an engine limit.
- Uzu queues concurrent requests, so its totals stay at one request's rate.
- Report every result as measured and mark what is not like for like: weights format, draft precision, backend, `HIP_LAUNCH_BLOCKING`, flags a fork lacks, the Core ML sidecar on the Mac.

## Rules

- Benchmarks only: do not commit, push or open pull requests from a benchmark session.
- Publish only where the user asks.
- Never start a run on a machine someone else is using, never touch other users' files or processes, and never run anything as root.
