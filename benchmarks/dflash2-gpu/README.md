# DFlash2 GPU campaign tools

## Generate a report

```sh
python3 benchmarks/dflash2-gpu/make_table.py /absolute/path/session.json
python3 benchmarks/dflash2-gpu/render_report.py /absolute/path/REPORT.md /absolute/path/session.json
python3 -m unittest discover -s benchmarks/dflash2-gpu -p test_report.py -v
```

`make_table.py` writes Markdown to stdout. `render_report.py` updates only `<!-- dflash2-gpu-next -->` through `<!-- /dflash2-gpu-next -->` in an existing report. If absent, it appends that section. It preserves historical content outside those markers, including line endings. Duplicate, incomplete or reversed markers are errors; legacy `table`, `upstream` and `crosscheck` sections are never replaced.

Both commands accept `--require-complete`: write useful partial evidence but exit 1 when the aggregate is withheld. Malformed top-level indexes exit 2. Without that option, a valid partial index exits 0; `UNMEASURED` is not a passed campaign. All raw links are absolute, URL-escaped controller paths, so they remain exact when the report is copied elsewhere. Publish those artifacts at accessible paths before distributing a report to another machine.

## Session index: version 1

Paths are explicit, absolute or relative to the index file. No filename discovery, retry preference, fastest-session selection or implicit fallback occurs. Empty or omitted lanes remain in the six-lane denominator and withhold the aggregate. Unknown lane names, duplicate selected paths and unknown schema versions are errors.

This is a complete, valid *partial-baseline* index example; only the selected result and its configuration evidence need to exist. Its single-prompt canary cannot support a confidence claim:

```json
{
  "schema_version": 1,
  "campaign": "DFlash2 GPU next: baseline canaries",
  "baseline_sha": "20e89b5a0b87f9f79e640e0bc3e209b9e192bc52",
  "candidate_sha": "20e89b5a0b87f9f79e640e0bc3e209b9e192bc52",
  "prompt_indices": [0],
  "bootstrap": {"seed": 20261003, "resamples": 10000},
  "lanes": {
    "metal": {"draft_width": 3, "base": ["results/mac/base-metal-n3-canary.json"]},
    "rtx5090-cuda": {"draft_width": 3},
    "rtx5090-vulkan": {"draft_width": 3},
    "spark-cuda": {"draft_width": 3},
    "spark-vulkan": {"draft_width": 3},
    "strix-vulkan": {"draft_width": 3}
  }
}
```

Required top-level fields: `schema_version`, nonempty `campaign`, pinned `baseline_sha`, full 40-hex `candidate_sha`, ordered unique `prompt_indices` in 0..11, `bootstrap` with integer `seed` and `resamples >= 1000`, and `lanes`. The six fixed IDs map to manifest `(device, backend)` as follows:

| ID | device | backend |
|---|---|---|
| metal | mac | metal |
| rtx5090-cuda | rtx5090 | cuda |
| rtx5090-vulkan | rtx5090 | vulkan |
| spark-cuda | spark | cuda |
| spark-vulkan | spark | vulkan |
| strix-vulkan | strix | vulkan |

Each populated lane requires `draft_width` (3, 5 or 7, selected from baseline characterization). Optional arrays `base`, `candidate`, `serial_base`, `serial_candidate`, `tf` contain exact result paths. Optional `blocks` and `serial_blocks` contain `{ "id": "unique-block-name", "launches": [A1, B1, B2, A2] }` objects. `A` is base and `B` is candidate; serial uses its corresponding arms. List blocks in chronological order. Every selected path for that comparison must appear exactly once across blocks, without reusing server launches, labels or timestamps. All launches must contain the same selected prompt indices, with at least three fixtures per launch for paired comparison. At least three independent blocks per lane are needed for confidence intervals and the aggregate.

For example, replace one lane in an acceptance index (`prompt_indices: [0,1,2]`) with:

```json
{
  "draft_width": 7,
  "base": ["b1-a1.json", "b1-a2.json", "b2-a1.json", "b2-a2.json", "b3-a1.json", "b3-a2.json"],
  "candidate": ["b1-b1.json", "b1-b2.json", "b2-b1.json", "b2-b2.json", "b3-b1.json", "b3-b2.json"],
  "blocks": [
    {"id": "b1", "launches": ["b1-a1.json", "b1-b1.json", "b1-b2.json", "b1-a2.json"]},
    {"id": "b2", "launches": ["b2-a1.json", "b2-b1.json", "b2-b2.json", "b2-a2.json"]},
    {"id": "b3", "launches": ["b3-a1.json", "b3-b1.json", "b3-b2.json", "b3-a2.json"]}
  ]
}
```

Use separate indexes for intentionally changed configurations or tuning searches, never mix them into the fixed headline comparison. This generator qualifies only the approved 10k/1024 workload and frozen baseline-selected width; it does not publish a flags-only tuning result as a kernel gain.

## Result and manifest proof

The tools consume the existing `bench_pp_tg.py` result: `label`, `engine`, `complete`, `error`, `summary`, `runs`, `provenance`. Required summary values are `all_ok`, `server_prefill_tps`, `decode_tps`, `prefill_tps`. Optional engine decode remains `unreported` when absent.

Each run must have the exact indexed `prompt_index`, 10000 prompt tokens, 1024 completion tokens, `ok: true`, all five client `checks: true`, valid request settings, positive client timings, engine prefill rate, and full-output `text_sha256`. Row rates must match the client formulas and summaries must match row medians. Partial, cached, error and unreported-cache Fabric rows are invalid even if `summary.all_ok` says true. Fabric requires `cached_tokens: 0`, `cache_status: "reported"`, `request.cache_prompt: false`. TensorFold may instead use `cache_status: "evidenced"`, `cached_tokens: null`, and `cache_evidence: {"uncached": true, "source_revision": "6ea5ade26c4335491c50275af0be32b81f75f525", "artifact": "path/to/source-or-log-evidence"}`; that artifact must exist. Paths in embedded configuration/cache evidence are relative to the session index, not the remote host.

The complete `text` must be present and its UTF-8 SHA256 must equal `text_sha256`. A hash without the saved output, or a truncated/modified output with a stale hash, is invalid evidence.

`provenance.manifest` must contain matching `label`, `device`, `backend`, `draft_width`, a full `binary_sha256`, exact `command` array, `environment` object, and positive `created_unix_s` recording the distinct server launch. The client and canonical fixture hashes must be present in `provenance.client_sha256` and `provenance.fixture_sha256`. Different client revisions cannot be paired. A launch timestamp is recorded evidence of orchestration, not a substitute for actually restarting the server.

The parent normalizes verified startup/build logs into `provenance.manifest.lane.report_config`. This explicit schema avoids guessing among staging-host manifest variants. Missing proof withholds the comparison; it is never silently inferred from a model name or requested flag. Preserve the original manifest and logs. Example Fabric speculative config:

```json
{
  "source_sha": "20e89b5a0b87f9f79e640e0bc3e209b9e192bc52",
  "model_sha256": "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d",
  "draft_sha256": "ef6a1340ed018cc58f53efd3613312c719b9ebf72b5586a046214c7ba366b297",
  "fixture_sha256": "cf991cd976528946657d122b6315ef73acd1f800ac68f07221b64e45833d5a83",
  "context": 16384,
  "batch": 2048,
  "ubatch": 512,
  "cache_k": "f16",
  "cache_v": "f16",
  "cache_k_draft": "f16",
  "cache_v_draft": "f16",
  "threads": 8,
  "threads_batch": 8,
  "flash_attn": "on",
  "offload_layers": 99,
  "parallel": 1,
  "greedy": true,
  "gpu_id": "verified physical GPU UUID or Metal registry identity",
  "backend_config": {"precision": "stock", "shader_compiler_version": "verified pin"},
  "toolchain": {"compiler_version": "verified version", "architecture": "verified target", "flags": ["verified meaningful options"]},
  "evidence": ["logs/startup.log", "logs/build-manifest.json"]
}
```

The thread counts, actual offload/snapshot values and backend/toolchain values above are illustrative, **not measured defaults**. Use each lane's observed values. `backend_config` is a nonempty object containing normalized inference feature/precision settings, meaningful backend environment flags and actual offload/state details. `toolchain` contains compiler versions, target architecture and meaningful build flags, not source/artifact paths or raw CMakeCache hashes. `evidence` is a nonempty list of existing controller artifact paths. `source_sha` must equal the indexed base/candidate pin for that arm. Required normalized config fields must match across the comparison except `source_sha` and `evidence`. Raw manifest environments, source/binary/cache hashes and artifact paths remain provenance, not cross-arm equality fields. Each arm must use one binary hash throughout. A candidate source may differ from baseline but cannot vary between lanes or blocks.

Serial manifests use `draft_width: 0`; draft model/speculation flags must be absent from the exact command. Draft-only keys `draft_sha256`, `cache_k_draft`, `cache_v_draft` may be null or omitted and are ignored in serial comparisons. Serial comparisons are separate controls, never speculative baselines. TensorFold uses `backend: "mlx"` for the Metal lane, `"cuda"` for CUDA, pinned `source_sha: "6ea5ade26c4335491c50275af0be32b81f75f525"`, its own target/draft hashes and `drafting_active: true`. Required TF config fields are `source_sha`, `model_sha256`, `draft_sha256`, `fixture_sha256`, `context`, `parallel`, `greedy`, `gpu_id`, nonempty `backend_config`, `evidence`, and `drafting_active`. The TF comparison matches physical device, rendered prompts, client revision and workload, not Fabric weight bytes or backend-specific arithmetic. It is explicitly **engine plus format** and descriptive only, not a paired confidence claim. Vulkan never gets a TensorFold ratio, even if a TF path is supplied.

Fabric also requires `provenance.manifest.lane.runtime_evidence` from actual startup/graph diagnostics:

```json
{
  "target_gpu_layers": 66,
  "target_layers": 66,
  "draft_gpu_layers": 6,
  "draft_layers": 6,
  "target_n_rs_seq": 7,
  "draft_n_rs_seq": 0,
  "evidence": ["logs/actual-offload-and-state.log"]
}
```

These are illustrative numbers, not fabricated measurements. All six numeric fields must be nonnegative integers. Full actual target and drafter offload is required. Serial has both draft layer counts zero, `target_n_rs_seq: 0`, `draft_n_rs_seq: 0`. Speculative has `target_n_rs_seq` equal to frozen `draft_width`, `draft_n_rs_seq: 0`, and positive actual draft layer counts. Do not confuse `n_rs_seq` with GDN snapshot slots: `K = 1 + n_rs_seq`, so serial target K is 1, not 0. Runtime values must match across a paired comparison. Runtime `evidence` paths are nonempty and must exist on the controller, relative to the session index.

Historical raw files are never rewritten. Measurement-valid rows without compatible normalized configuration/runtime proof remain visible as **PROVISIONAL**, with exact rates and missing-proof reasons. They cannot enter paired ratios or the aggregate. New accepted runs must include the frozen proof schema; this tool does not accept proof sidecars or silently synthesize missing metadata.

## Statistics and acceptance

For each matched prompt in an ordered block, compute `log_ratio = (log(B1) + log(B2) - log(A1) - log(A2)) / 2`. Average these log ratios over prompts to obtain **one** observation per independent launch block. The lane estimate is `exp(median(block_log_ratios))`. Report every block ratio. The deterministic seeded bootstrap resamples whole blocks with replacement within each lane and recomputes its median; its 2.5/97.5 percentile interval requires at least three independent blocks. Prompts, chunks and tokens are never independent bootstrap samples.

Aggregate estimate: `exp(mean(median(block_log_ratios_for_lane)))` over **all six lanes with equal weights**. Its bootstrap independently resamples blocks within each lane before recomputing that expression. Missing/invalid lanes, mismatched configurations or fewer than three blocks withhold the aggregate. Lane baselines and one/two-block point estimates remain visible without confidence claims. An aggregate target is met only if its CI lower bound reaches 1.50x prefill / 1.35x decode and no lane's median regresses for that metric. A per-lane gain requires CI lower bound >1; median >=1 with CI including 1 is labeled unchanged. Median <1 is a regression requiring rejection or more evidence, even if aggregate improves. Performance-only target labels are not a quality acceptance or campaign completion claim.

## Separate quality and operation evidence

Optional `quality_index` points to a JSON object with `entries`, each `{ "lane": "metal", "schedule": "width8/rs7/R1", "reference_sha": "full source pin", "candidate_sha": "full source pin", "artifact": "quality-output.json" }`. Artifact paths are relative to this evidence index. The report links these as **INDEXED ONLY**, not a verified quality pass. Timing, text hashes, index labels and absent evidence cannot establish top-token agreement, KL, perplexity or rollback fidelity. A missing artifact is `UNMEASURED`.

Optional `profile_index` points to `{ "entries": [...] }`; each entry has:

```json
{
  "lane": "metal",
  "stage": "prefill",
  "kind": "proxy",
  "artifact": "raw-op-profile.txt",
  "ops": [ {"family": "MUL_MAT", "shape": "captured exact dimensions", "dtype": "Q4_0", "time_ms": 0.25} ]
}
```

`stage` is `prefill` or `decode`; `kind` is `production` or `proxy`. Entries sort by `time_ms` and display at most ten families, with all exact shapes/dtypes and raw links retained in the index. Production entries require positive measured `denominator_ms` for that stage; only those entries show stage share. Proxy times are never summed or divided by a fabricated production denominator. Optional `residual_cpu_sync_ms` is displayed as supplied; absent residuals are `UNMEASURED`.

## Collect selected runs (parent-owned tools)

Freeze `THREADS` and `BATCH_THREADS` from baseline startup logs before accepted runs. Use the same values in each candidate. The launcher prints a canary identifier and records its command/manifest and raw artifacts under `$ROOT/results/$DEVICE`.

```sh
export ROOT="$HOME/fabric-dflash2-gpu-next"
export LANE_MANIFEST="$ROOT/cuda-lane.json"
export REPS=3 PROMPT_START=0
export THREADS=8 BATCH_THREADS=8 # replace with observed lane counts
bash "$ROOT/scripts/run.sh" rtx5090 "$ROOT/base/build-cuda" cuda 7 base-cuda-n7-block1-a1
```

`run.sh DEVICE BUILD_DIR BACKEND NMAX LABEL` requires `LANE_MANIFEST`; use explicit new labels for each server launch. For Vulkan, set `GGML_VK_VISIBLE_DEVICES` to the ordinal verified against the intended physical GPU, not an assumed default. `NMAX=0` is the separately selected serial control. Do not run concurrent campaign GPU jobs on the same host.

The benchmark client requires a manifest:

```sh
python3 "$ROOT/scripts/bench_pp_tg.py" http://127.0.0.1:8093 --engine fabric --prompts "$ROOT/prompt.json" --manifest run.manifest.json --reps 3 --prompt-start 0 --label explicit-run-label --output selected-result.json
```

Collect baseline native token IDs for the separate teacher-forced tool, not by retokenizing streamed text:

```sh
python3 benchmarks/dflash2-gpu/collect_quality_tokens.py http://127.0.0.1:8093 --prompts prompt.json --manifest run.manifest.json --output-dir quality-tokens --prompt-start 0 --count 6
```

The collector and launcher are parent-owned; their runtime verification is separate from the report generator's synthetic CLI and behavioral tests.
