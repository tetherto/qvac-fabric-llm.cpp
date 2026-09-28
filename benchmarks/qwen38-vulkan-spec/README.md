# Qwen3.8-27B Vulkan speculative benchmark

Run `run_campaign.py` from a checkout of commit `d21a547da25d50bee71c2ea941df56414030d0c5`. The driver requires a Linux Vulkan build with `llama-server` and `llama-batched-bench`, the two pinned Unsloth target GGUFs, paired MTP and DFlash draft GGUFs, converted DSpark GGUFs, a checked LongBench v2 data file, and a Python environment with `gguf`, `numpy`, and `matplotlib`. Keep the source/build/models/venv/results/fixtures tree in one isolated campaign root. The driver checks the exact source commit before each phase.

```bash
ROOT=/home/pratik/qwen38-vulkan-spec-20260924T073141Z
PY=$ROOT/venv/bin/python
DRIVER=$ROOT/source/benchmarks/qwen38-vulkan-spec/run_campaign.py
MPLBACKEND=Agg "$PY" "$DRIVER" --root "$ROOT" prepare-fixture
MPLBACKEND=Agg "$PY" "$DRIVER" --root "$ROOT" smoke
MPLBACKEND=Agg "$PY" "$DRIVER" --root "$ROOT" unprofiled
MPLBACKEND=Agg "$PY" "$DRIVER" --root "$ROOT" profiles
MPLBACKEND=Agg "$PY" "$DRIVER" --root "$ROOT" verify
```

The fixture uses LongBench v2 row 350 (`66fa6702bb02136c067c6abb`, Code Repository Understanding). The script compares both target GGUF vocabularies and full token arrays, then retains exactly 110000 tokens. Native `/completion` takes numeric token arrays; the first 16 positions form a per-request cache-isolation tag. The 100000-token warm prefix is byte-for-byte the prefix of the 110000-token prompt for each slot and repetition. No implicit BOS is added to a numeric prompt.

Each target/mode pair first gates a 1024-token greedy request on Vulkan. Unsupported pairs get withheld rows with the gate failure and raw log, not performance numbers. Supported pairs run `off`, `draft-mtp`, `draft-dspark`, or `draft-dflash` at concurrency 1, 2, and 4. Context is 131072 tokens per slot; target and draft weights are offloaded to `Vulkan0`; F16 KV, flash attention, batch 2048, and microbatch 512 are fixed. Idle-slot prompt cache stays enabled so the warm request can reuse the preceding cold 100000-token request. Cold requests set `cache_prompt=false`; warm requests set it to `true`. Per-request server timings must show `(cache_n, prompt_n)` of `(0,10000)`, `(0,100000)`, or `(100000,10000)` and exactly 1024 output tokens.

For this hybrid model, the server checkpoints a cold 100000-token prompt at token 99996 before its final prompt batch. A direct warm request would therefore report 99996 cached and 10004 processed tokens. The harness sends an unmeasured, one-token 100004-token primer with the same first 100000 tokens and four deliberately different tail tokens after cold 100k. This advances the checkpoint while keeping the next request's common prefix at 100000. The measured 110000-token request must report exactly 100000 cached and 10000 processed tokens. Primer timings and token IDs remain in the raw protocol record; primer work is never included in a measured row.

The pinned DFlash Q4_0 and Q8_0 GGUFs contain 81 tensors, including selector and convolution tensors. This commit's `llama_model_dflash::load_arch_tensors` consumes only 58 of them and rejects both drafts at load time (`wrong number of tensors; expected 81, got 58`). The driver withholds their 18 shape/concurrency rows and retains both full gate logs. It does not substitute DSpark or MTP measurements for DFlash.

For each launch, discard a complete three-shape warmup protocol, collect two full repetitions, and run a third if either prefill or decode speed varies more than 3% for any shape. The summary uses medians. Prefill throughput is total fresh prompt tokens divided by the interval from the release barrier to the last first token. Decode throughput counts only streamed tokens after all slots produced their first token, divided by the time until the last response ends. TTFT uses the same barrier-to-last-first-token interval; average ITL averages per-slot intervals across the 1023 gaps. Acceptance length is `1 + accepted_draft_tokens / verification_steps` from `/metrics`. Baseline off-mode 10k results are cross-checked with `llama-batched-bench` as a separate measurement, not merged into the HTTP throughput rows.

Every published row has separate prefill and decode Vulkan profiling logs, reports from the checked-in weighted `vulkan_profiling_analyzer.py`, the original raw logs compressed as `.raw.gz`, command/env evidence, and a row Markdown report. The prefill replay requests one token; the decode replay extracts profiling sections only after every slot has produced its first token. Profiling does not alter the unprofiled throughput numbers. `verify` recomputes every aggregate and the summary from raw events, checks phase/slot/cache/output invariants, verifies all row reports, and checks new text artifacts for ASCII. Results and logs live in `$ROOT/results/`.
