# Qwen3.8-27B Q4_K_XL on H100: Hopper CUDA dispatch optimization

Historical report (2026-09-14): describes the state before the prefill/decode campaign. The campaign log `progress-log.md` is the authority for the current code, config (ubatch 4096, f16 KV, cache-ram 32768) and numbers; this file is kept for the engine comparison tables and the dispatch-crossover measurements it records.

Setup: 1x H100 80GB, Qwen3.8-27B-UD-Q4_K_XL (unsloth GGUF), 262144 context, all layers on GPU, KV cache q4_0, flash-attn on, batch 2048 / ubatch 512, concurrency 1 unless stated.

Branch base: temp-10549 (54db4c109). Nothing committed; changes are uncommitted working-tree edits.

## Retained changes (7 lines, 3 files)

1. `ggml/src/ggml-cuda/mmq.cu:311` + `ggml/src/ggml-cuda/mmq.cuh:9` - on Hopper, dense quantized MUL_MAT with `ne11 >= 512` skips MMQ and uses cuBLAS. Gate is `n_experts == 0`, so MoE and all batches below 512 keep existing MMQ behavior. Disabled under `GGML_CUDA_FORCE_MMQ`.
2. `ggml/src/ggml-cuda/ggml-cuda.cu:1512` - FP16 cuBLAS compute keeps FP32 output on Hopper (adds `cc == GGML_CUDA_CC_HOPPER` to the existing Volta/RDNA4/CDNA list).

## Per-change attribution, 10k prompt + 1k output

| Step | Prefill tok/s | Wall |
|---|---|---|
| Baseline | 2340.6 | 18.06 s |
| + change 1 (cuBLAS for large batches) | 2709.7 (+15.8%) | 17.53 s (-3.0%) |
| + change 2 (FP32 output), ungated attribution run | 2891.3 (+6.7% more) | 17.16 s (-2.1% more) |
| Current working tree (both changes + batch gate) | 2948.4 (+25.9%) | 17.23 s (-4.6%) |

Gated and ungated builds are equivalent at ubatch 512 (ne11 = 512 goes to cuBLAS in both), so 17.16 vs 17.23 s is run-to-run noise, not a cost of the gate.

## Current numbers, all measured workloads

| Workload | Wall before -> after | Prefill tok/s before -> after | Decode tok/s before -> after |
|---|---|---|---|
| 10k + 1k | 18.06 -> 17.23 s (-4.6%) | 2340.6 -> 2948.4 (+25.9%) | 72.57 -> 72.33 |
| Cold 110k + 1k | 81.11 -> 73.47 s (-9.4%) | 1979.0 -> 2301.8 (+16.3%) | 39.34 -> 39.35 |
| Cached 100k prefix + 10k + 1k | 31.83 -> 31.16 s (-2.1%) | 1618.0 -> 1791.3 (+10.7%) | 39.26 -> 39.33 |

Decode is unchanged within noise (under 0.5% either direction). Batch-1 decode uses MMVQ, which neither change touches. All gains are prefill.

The 110k and cached rows were measured on the ungated build; the gate cannot affect them at ubatch 512.

## Cross-format engine comparison: fabric vs vLLM vs SGLang

This section compares different engines running different weight formats. It is not a measurement of the CUDA changes above, and it is not apples-to-apples on numerics: fabric runs Q4_K_XL (about 4.5 bits per weight), vLLM and SGLang run FP8 or NVFP4. No accuracy gate was run across engines.

Identical request contract for every engine: 32 prompts from `ryanmarten/OpenThoughts-1k-sample` (user turns only, one fixed exact-token prompt set, 17339 input tokens total, mean 542 tokens per prompt), concurrency 8, 2048 output tokens per request with `ignore_eos`, 262144 advertised context. Client is `vllm bench serve --backend openai` in all cases.

Every row is one single repetition: the one with median output throughput among the three cold runs (r1-r3). Wall, TTFT, TPOT and ITL therefore come from that same run and stay mutually consistent; they are not per-column medians. Warm-cache repetitions are excluded and reported separately below. Output throughput counts generated tokens over total run wall time.

### Single H100 (TP1)

| Engine and config | Output tok/s | Total tok/s | Wall | Mean TTFT | Mean TPOT | Mean ITL | vs fabric |
|---|---|---|---|---|---|---|---|
| fabric-llm Q4_K_XL (pre-optimization) | 181.0 | 228.9 | 362.0 s | 5656 ms | 41.44 ms | 41.43 ms | 1.00x |
| vLLM FP8 | 456.5 | 577.2 | 143.6 s | 1006 ms | 17.04 ms | 17.04 ms | 2.52x |
| SGLang FP8 | 581.7 | 735.6 | 112.7 s | 267 ms | 13.63 ms | 13.63 ms | 3.21x |
| vLLM NVFP4 (Marlin fallback, no FP4 tensor cores on SM90) | 653.0 | 825.8 | 100.4 s | 641 ms | 11.94 ms | 11.96 ms | 3.61x |
| SGLang FP8 + DFlash2 speculative decode | 768.3 | 971.5 | 85.3 s | 6563 ms | 5.83 ms | 20.47 ms | 4.25x |

### Two H100 (TP2)

Fabric was not measured at TP2, so these rows carry no fabric ratio and must not be compared against the TP1 fabric row.

| Engine and config | Output tok/s | Total tok/s | Wall | Mean TTFT | Mean TPOT | Mean ITL |
|---|---|---|---|---|---|---|
| vLLM FP8 | 699.8 | 884.9 | 93.7 s | 565 ms | 11.16 ms | 11.16 ms |
| vLLM NVFP4 (Marlin fallback) | 823.3 | 1041.1 | 79.6 s | 291 ms | 9.58 ms | 9.58 ms |
| SGLang FP8 | 917.6 | 1160.4 | 71.4 s | 208 ms | 8.62 ms | 8.62 ms |

Run-to-run spread within each engine and config was under 2% on output throughput for vLLM and SGLang, and 173.6 to 182.1 tok/s (4.9%) for fabric. All 32 requests completed in every run.

Warm-cache repetitions (same prompts, server cache retained) changed nothing material: fabric 179.3, vLLM FP8 TP1 460.5, SGLang FP8 TP1 571.8, SGLang DFlash2 772.8 tok/s. Whether these prompts share cacheable prefixes was not measured, so this is an observation, not evidence about prefix caching.

On the DFlash2 row, TPOT and ITL diverge (5.83 ms vs 20.47 ms) because speculative decoding emits several tokens per server step. For non-speculative rows the two are equal. Compare speculative against non-speculative on output throughput, not on TPOT.

### What the gap is made of

1. This workload is decode-dominated, not decode-only: 65536 output tokens against 17339 input tokens. Fabric mean ITL is 41.4 ms against 13.6 ms for SGLang FP8, a 3.0x ratio close to the 3.2x throughput ratio, so per-token decode cost accounts for most of the gap at this shape.
2. Speculative decoding is the largest same-format lever measured: SGLang FP8 TP1 goes 581.7 -> 768.3 tok/s (+32.1%) with DFlash2 on the same GPU and the same weight format. Changing format goes further still, see the next point.
3. NVFP4 gives vLLM +43% over its own FP8 at TP1 (456.5 -> 653.0) while running the Marlin W4A16 fallback, since SM90 has no FP4 tensor cores. That is a 4-bit weight path reaching 653.0 tok/s on this hardware.
4. TTFT differs by more than an order of magnitude at equal concurrency and prompt set: fabric 5656 ms, SGLang FP8 267 ms, vLLM FP8 1006 ms. The cause was not isolated. SGLang with DFlash2 also shows high TTFT (6563 ms) while leading on throughput, so TTFT does not rank engines here.
5. The CUDA changes in this report do not move this table. They are prefill-only, and this workload runs a 3.8:1 output-to-input ratio, so the expected effect on aggregate serving throughput is under 2%, inside the 4.9% fabric run-to-run spread.

## Prefill vs decode split at 10k, 110k and 100k cached, concurrency 1

Single request at a time, 1024 output tokens with `ignore_eos`, temperature 0, one H100 each. Prefill is client TTFT, decode is the remaining stream time over `n_out - 1` tokens, both from the same harness (`benchmarks/qwen38-h100/pd_bench.py`). Every request carries a unique tag, so a cold shape cannot reuse an earlier run's cache; `/slots/{id}?action=erase` does not clear llama-server's host-side prompt cache. Cached-row hits are server-reported and never assumed: 100043 of 110044 prompt tokens on fabric (`timings.cache_n`), 100032 of 110043 on sglang (`usage.prompt_tokens_details.cached_tokens`, needs `--enable-cache-report`). Fabric cold rows reported 0 hits. SGLang cold rows reported no hit field at all, so their coldness rests on the successful `/flush_cache` plus the unique tag, not on a server-verified zero. Means of 2 repetitions.

| Shape | Engine | Fresh in tok | Prefill s | Prefill tok/s | Decode s | Decode tok/s | Total s |
|---|---|---|---|---|---|---|---|
| 10k + 1k | fabric Q8_K_XL | 10039 | 3.52 | 2849 | 16.73 | 61.16 | 20.25 |
| 10k + 1k | SGLang FP8 | 10039 | 0.58 | 17414 | 12.46 | 82.08 | 13.04 |
| cold 110k + 1k | fabric Q8_K_XL | 110041 | 38.45 | 2862 | 28.53 | 35.86 | 66.98 |
| cold 110k + 1k | SGLang FP8 | 110039 | 8.31 | 13245 | 13.75 | 74.38 | 22.06 |
| 100k cached + 10k + 1k | fabric Q8_K_XL | 10000 | 4.75 | 2105 | 28.60 | 35.77 | 33.35 |
| 100k cached + 10k + 1k | SGLang FP8 | 10009 | 1.24 | 8106 | 13.74 | 74.47 | 14.97 |
| 10k + 1k | SGLang NVFP4 | N/A | N/A | N/A | N/A | N/A | N/A |
| cold 110k + 1k | SGLang NVFP4 | N/A | N/A | N/A | N/A | N/A | N/A |
| 100k cached + 10k + 1k | SGLang NVFP4 | N/A | N/A | N/A | N/A | N/A | N/A |

The fabric 10k prefill row is noisy: the two repetitions gave 2.78 s and 4.27 s client TTFT, a 54% spread. The 110k and cached rows repeated within 0.6%. Fabric server-side prefill time for the same requests was 2.59 s, 37.93 s and 4.63 s (3883, 2901 and 2162 tok/s); those are diagnostics only and are not used in any ratio here, because sglang exposes no equivalent server-side prefill time and mixing the two would drop fabric's request and first-token overhead.

SGLang NVFP4 was attempted on the same host and is not available: with flashinfer enabled the scheduler segfaults in `flashinfer_bmm_fp8` during fp8_gemm autotuning, and with `SGLANG_IS_FLASHINFER_AVAILABLE=false` the FP4 Marlin fallback raises `sglang::apply_fp4_marlin_linear() Expected a value of type 'Tensor' for argument 'input' but instead found type 'tuple'`. SM90 has no FP4 tensor cores. No framework patching was attempted.

### Fabric Q4_K_XL, prior run, different method

The Q4 rows come from the earlier single-request runs in this report, measured from server `prompt_ms` and `predicted_ms` with 1000 output tokens, one repetition. They are not from the harness above, so sglang rates are normalized to the same 1000 output tokens before any comparison, and fabric's times exclude the client and first-token overhead the Q8 rows include. Every Q4 comparison below is therefore a mixed-method estimate; the bias direction is not established.

| Shape | Fabric Q4 prefill s | Fabric Q4 prefill tok/s | Fabric Q4 decode s | Fabric Q4 decode tok/s | Fabric Q4 total s | SGLang FP8 total s at 1000 tok |
|---|---|---|---|---|---|---|
| 10k + 1k | 3.39 | 2948 | 13.83 | 72.33 | 17.22 | 12.75 |
| cold 110k + 1k | 47.79 | 2302 | 25.41 | 39.35 | 73.20 | 21.74 |
| 100k cached + 10k + 1k | 5.58 | 1791 | 25.43 | 39.33 | 31.01 | 14.65 |

### Gap per phase

Values divide the sglang rate by the fabric rate, so they state how many times slower fabric is in that phase. The Q4 columns divide a client-timed sglang rate by a server-timed fabric rate: mixed-method estimates, bias direction not established.

| Shape | Q8 prefill | Q8 decode | Q4 prefill | Q4 decode |
|---|---|---|---|---|
| 10k + 1k | 6.11x | 1.34x | 5.91x | 1.13x |
| cold 110k + 1k | 4.63x | 2.07x | 5.75x | 1.89x |
| 100k cached + 10k + 1k | 3.85x | 2.08x | 4.53x | 1.89x |

### Phase contribution to the total gap

Contribution is excess time, `(T_phase_fabric - T_phase_sglang) / (T_total_fabric - T_total_sglang)`, so the two shares sum to 100% per row. Only the Q8 rows qualify: both engines were measured in one harness with the same 1024-token output. Q4 is N/A here because its phase times come from server counters at 1000 output tokens and exclude the request and first-token overhead the sglang side includes; borrowing the Q8 overhead to patch that would be synthetic, and that overhead itself varied from 0.21 to 1.67 s across the two Q8 repetitions.

| Shape | Engine | Total excess s | Prefill excess s | Prefill share | Decode excess s | Decode share |
|---|---|---|---|---|---|---|
| 10k + 1k | fabric Q8_K_XL | 7.21 | 2.95 | 41% (34 to 46% across the 2 reps) | 4.26 | 59% (54 to 66%) |
| cold 110k + 1k | fabric Q8_K_XL | 44.92 | 30.15 | 67.1% | 14.77 | 32.9% |
| 100k cached + 10k + 1k | fabric Q8_K_XL | 18.37 | 3.52 | 19.1% | 14.86 | 80.9% |
| all three shapes | fabric Q4_K_XL | N/A | N/A | N/A | N/A | N/A |

Prefill is the worse relative phase in every shape (fabric is 3.9x to 6.1x slower) while decode trails by 1.1x to 2.1x, but which phase dominates the wall-clock gap depends on shape: the cold 110k prompt is prefill-bound (67.1% of the excess), and once the 100k prefix is cached decode owns 80.9% of it. Fabric's decode gap widens with context, from 1.34x at 10k to 2.07x at 110k on Q8, which is consistent with growing attention and KV cost, but the cause was not isolated. Q8 prefills faster than Q4 on the same server timings (3883 vs 2948 tok/s at 10k) and decodes slower (61.2 vs 72.3 tok/s), as expected from 31 GB against 17 GB of weights.

Raw results: `benchmarks/qwen38-h100/results/fabric/q8-phase-split-tp1.json` and `benchmarks/qwen38-h100/results/sglang/fp8-phase-split-tp1.json`.


## Why the batch gate exists

MMQ vs cuBLAS on H100, m=4096, k=14336, from `test-backend-ops perf -o MUL_MAT` console output. Values are MMQ speedup over cuBLAS; below 1.0 means cuBLAS wins.

| n | q4_K | q5_K | q6_K | iq4_xs | q8_0 |
|---|---|---|---|---|---|
| 16 | 5.66 | 4.64 | 3.49 | 4.43 | 2.33 |
| 64 | 2.93 | 2.65 | 2.20 | 3.41 | 1.59 |
| 128 | 1.78 | 1.73 | 1.46 | 2.14 | 1.18 |
| 256 | 0.99 | 0.98 | 0.85 | 1.22 | 0.77 |
| 512 | 0.74 | 0.72 | 0.63 | 0.90 | 0.60 |

End-to-end confirmation at ubatch 128 (10k prompt, 64 output): gated 6.812 s / 1694.3 tok/s prefill, ungated 9.064 s / 1226.9 tok/s. The ungated build is 33% slower there, so the crossover gate is required.

## Rejected candidates

| Candidate | Result | Verdict |
|---|---|---|
| Unconditional cuBLAS on Hopper | ubatch 128 prefill 1694 -> 1227 tok/s | replaced by batch >= 512 gate |
| CUDA graph optimizer (`GGML_CUDA_GRAPH_OPT=1`) | 17.45 -> 17.87 s (+2.4%) | rejected |
| BF16 cuBLAS compute | 0.27% faster, lower input precision | rejected |
| Upstream `73ab7599b` (MMVQ Q4_K/Q5_K) | provisional +10.5 to +11.5% per-request decode at concurrency 8, one sample per cell | not applied, needs interleaved A/B |

## Correctness

- `test-backend-ops test -b CUDA0 -o MUL_MAT`: 1203/1203 cases pass on the retained build.
- Deterministic 10k prompt, 64 tokens: retained build and original cuBLAS path select the same first token.
- Mean top-logprob error against a full-context CPU reference improved 1.397 -> 1.120 with the FP32-output change.

## Where remaining GPU time goes

Nsight `cuda_gpu_kern_sum`, production-shaped load (8 requests x 2048 output tokens, concurrency 8):

| Kernel group | Share of GPU time |
|---|---|
| MMVQ (batch-1 decode matmul) | 40.1% |
| MMQ (batch matmul) | 32.9% |
| Gated Delta Net | 6.2% |
| Quantize | 5.8% |
| Flash attention | 2.0% |

## Next steps, ranked

1. Speculative decoding: the largest measured same-format serving lever (SGLang FP8 TP1 581.7 -> 768.3 tok/s, +32.1% with DFlash2, same GPU and format). Format changes measured larger, vLLM FP8 -> NVFP4 at +43%, but they alter the weights rather than the serving path. Needs draft-model plumbing on the fabric path. Multi-day.
2. MMVQ tuning for SM90: the largest single share in the fabric profile at 40.1% of GPU time, and fabric mean ITL is 41.4 ms against 13.6 ms for SGLang FP8. The profile share makes this the best-evidenced internal target; it does not by itself prove MMVQ explains the whole cross-engine gap. Half a day to measure, gain unknown.
3. CUTLASS / Marlin-style W4A16 kernels for Q4_K: cross-engine precedent only, vLLM's Marlin fallback reaches 653.0 tok/s at TP1 on this hardware. Whether a Q4_K variant reaches comparable numbers inside fabric is unmeasured. New kernel subsystem. Multi-day.
4. Settle upstream `73ab7599b`: interleaved A/B/A/B with discarded warmups, judged on server per-request decode rate (within-build spread there was 1.4-2.3%, vs 35.6% on aggregate wall time). About 30 minutes.
5. Gated Delta Net fusion or tuning: 6.2% of GPU time. About a day.
6. Measure the TTFT gap (fabric 5656 ms vs SGLang FP8 267 ms on the same prompt set) and isolate its cause before acting on it. Half a day.

## Measurement notes

- `aggregate_output_tok_s` in the bench client is end-to-end output throughput (generated tokens over total wall time, prefill included). It is not decode throughput. Per-request decode rates come from server timings.
- Aggregate wall time at concurrency 8 showed 35.6% within-build spread across repetitions, so it cannot resolve effects below roughly 10%. Use per-request decode rates for small effects.
- Width sweep numbers come from console output of `test-backend-ops perf`; this branch's CSV printer omits `time_us`, `flops` and `n_runs`.
