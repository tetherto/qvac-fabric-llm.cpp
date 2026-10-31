# Qwen3.8-27B Core ML prefill: experiment log

Handover record of everything tried on `feat/coreml-qwen-3.8`, 2026-10-29 to 2026-10-31. It covers the path to the committed state (64 layers, FFN split across Metal and both Neural Engines), and what was tried after it. Raw evidence is in [results/2026-10-30/](results/2026-10-30/); earlier evidence is in [results/2026-10-29/](results/2026-10-29/).

Sections 0-13 measurements: Apple M3 Ultra (96 GB, two Neural Engines), macOS 26.7.1, `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 512-token microbatches, prompts of 2048 and 8192 tokens followed by 128 generated tokens, 3 timed repetitions after one warmup unless noted. "Prefill gain" is throughput relative to plain Metal.

## Current state (committed)

Commit `fbb340d` on top of `33123d5`: every one of the 64 FFNs keeps channels 0-8191 on Metal and sends 8192-12799 and 12800-17407 to two Core ML packages that run at the same time on the two ANEs. Core ML writes both outputs straight into Metal shared buffers and the Metal join adds the three parts (on by default for the resident path; `FFN_COREML_OUTPUT_BACKING=0` restores the CPU copy). Preset: [configs/qwen3.8-27b-64-dual.json](configs/qwen3.8-27b-64-dual.json). Exporter: `export_sidecar.py`.

| Measurement (same binary, same session) | Metal | Current state | Gain |
| --- | ---: | ---: | ---: |
| Prefill, 2048 tokens | 6.797 s | 4.776 s | +42.3% |
| Prefill, 8192 tokens | 28.075 s | 20.067 s | +39.9% |
| Request incl. 128 decoded tokens, 2048 / 8192 | 11.250 s / 32.615 s | 9.222 s / 24.603 s | -18.0% / -24.6% |
| Held-out quality (65,536 tokens) | | ΔNLL +0.00026 nats/token, KL 0.00042 | passes all checks |

Against the fastest plain-Metal setting (2048-token microbatches, 6.509 s / 26.889 s, section 13) the gain is about +37% at 2048-token prompts and +34% at 8192. Decode speed is unchanged and decode never calls Core ML. The committed code was rebuilt and checked on the benchmark Mac before committing: default settings gave 4.763 s at 2048 tokens with 0 output-backing fallbacks and the same quality on a held-out window as the gated run. Only the FFN split and output backing are in the branch; the DeltaNet projection experiment (section 8) is kept only as `results/2026-10-30/deltanet-projection.patch`. The 64-layer dual split without output backing measured +29.9% / +30.2% on the same build (section 7); the first team report quoted +23.8% / +23.2% from libraries built in an older remote tree.

### Since `fbb340d` (2026-10-31)

- `7bad9c2` on this branch pads partial prefill microbatches of at least 256 tokens to 512 so they also use the ANEs (section 17).
- Branch `feat/metal-fa-prefill-d256` (`a3ec7e2`, on `temp-11259`) adds a faster Metal prefill flash-attention kernel for head size 256 (section 16). It is independent of Core ML; the Core ML numbers in sections 15-17 were measured with both changes.
- Sections 14-17 use the pure Q4_0 GGUF (`Qwen3.8-27B-Q4_0-pure.gguf`) and the engine-bench skill (PR #424), not the UD-Q4_K_XL file and harness above.

## Before the committed state

### 0. Starting point (commit `150e600`, see RESULTS.md)

24 layers (31, 35, 39, 43-63), one ANE, Metal 11264 channels / Core ML 6144 channels. Prefill gain +6.8% (2K) / +7.3% (8K). Qualified with the repeated-prompt KL gate (mean KL <= 0.001); a 28-layer set failed it (code, KL 0.00113).

### 1. Profiling: where prefill time goes

Isolated FFN of layer 47, one 512-token chunk ([microbench/layer47-split.txt](results/2026-10-30/microbench/layer47-split.txt)):

| Path | Time |
| --- | ---: |
| Full FFN on Metal | 17.0 ms |
| Metal prefix (11264 channels) | 11.1 ms |
| Core ML suffix (6144 channels, 8-bit palette) | 9.85 ms |
| Split, in parallel | 11.9 ms |

The FFN is about 64% of Metal prefill (17 of 26.5 ms per layer per chunk). Layer 31 with FP16 weights gained nothing: its suffix took 16.0 ms, the split 17.1 ms, against 17.0 ms on Metal.

### 2. ANE weight formats (rejected)

Layer 47 suffix (layer 31 for the linear quantization rows), random input, relative L2 against Metal:

| Format | Ops on ANE | Suffix | Split | Rel. L2 |
| --- | ---: | ---: | ---: | ---: |
| 8-bit palette, per tensor (used) | 29/29 | 9.85 ms | 11.9 ms | 0.0091 |
| 8-bit palette, group 32 | 29/29 | 12.3 ms | 13.3 ms | 0.0086 |
| 8-bit palette, group 16 | 29/29 | 12.3 ms | 13.1 ms | 0.0085 |
| int8 per channel | 29/29 | 9.84 ms | 11.7 ms | 0.0089 |
| int8 per block 32 | 20/29 | 15.7 ms | 24.0 ms | 0.0103 |
| int4 per block 32 | 20/29 | 15.0 ms | 24.3 ms | 0.076 |
| FP16, tanh SiLU | 29/29 | 16.0 ms | 17.4 ms | 0.0059 |
| FP16, exact sigmoid | 23/23 | 15.6 ms | 16.6 ms | 0.0088 |

Per-block formats are not native on the M3 Ultra ANE. ANE time follows weight bytes. Much of the remaining error is FP16 arithmetic, so no weight format removes it. Scripts: `scripts/quant.py`, `scripts/probe-random-input.patch`.

### 3. Choosing channels by sensitivity (not pursued)

`llama-imatrix` statistics on calibration text not used anywhere else gave per-channel activation energy. The trailing 6144 channels carry 32-34% of FFN output energy in most layers; the best possible 256-channel blocks carry 29-32%. Exceptions (layers 1, 22-24, 54, 55, 59, 63) already used little of the error budget. Expected gain was about one layer, not worth a runtime gather. Data: `microbench/block-energy.json`, script `scripts/energy.py`.

### 4. Layer re-selection (superseded)

Layer 35 used about a third of the KL budget and layer 31 (FP16) added no speed. Swapping them gave "sel-27b" (layers 37-63, all 8-bit), which passed the old 12-case suite (worst KL 0.00094) at +8.6% (2K) / +9.1% (8K). Code KL at 2048 tokens for the tried sets: 25 layers 0.00065, 26 layers 0.00080, 27 layers (+37, +38) 0.00094, 27 layers (+36, +38) 0.00104 (fail), 28 layers 0.00101 (fail). The qualification logs were deleted during a disk cleanup; these numbers are from the session record.

### 5. Both Neural Engines

Two concurrent Core ML predictions run on the two ANEs through the public API (`scripts/dual.mm`):

| Run | Median |
| --- | ---: |
| One 6144-channel package | 9.75 ms |
| Two 3072-channel packages, one after the other | 10.73 ms |
| Two 3072-channel packages, concurrently | 5.48 ms |
| Two 6144-channel packages, concurrently | 9.82 ms |

Balanced split for layer 47: Metal 8192 channels (8.19 ms) with 4608 + 4608 channels on the two ANEs (8.01 ms), 9.29 ms in parallel against 16.94 ms on Metal. Rel. L2 rose from 0.0091 to 0.0111, in line with more channels in FP16.

### 6. Quality gate redesign

The old gate (mean KL <= 0.001 against Metal, on short prompts repeated to length) rejected layers 0-26. All-Metal controls through the offload boundary also failed there: KL 0.01-0.03 at 2048 tokens, with perplexity moving in both directions (-2% to +0.7%). The repeated text made the model copy, so near-tied tokens flipped, and KL measured that divergence rather than quality loss. On real text the same control reads about 0.000001-0.000004.

Agreed gate, on 65,536 held-out tokens of real, non-repeated text (C++ and Python from the repo, unused repo docs, a public-domain novel, Wikipedia mathematics; 20 windows of 2048 tokens, 3 of 8192, none used for calibration or selection):

- mean ΔNLL <= +0.002 nats/token and bootstrap 95% upper bound <= +0.005
- each domain (code, docs, prose, math) mean ΔNLL <= +0.003
- no single window above +0.005
- mean KL <= max(0.002, 2x all-Metal control), and <= max(0.003, 2x control) per domain
- exact offload counts, finite outputs, zero decode offloads

Scripts: `scripts/run_gate.sh`, `scripts/analyze_gate.py`, `scripts/build_windows.py` (paths are those of the benchmark Mac). dual27 was first judged with a draft (overall checks only, KL <= 0.001); it also passes the final gate.

### 7. Scaling the dual split (layers 37-63 down to all 64)

| Selection | Layers | Gate | ΔNLL | KL | Prose ΔNLL / KL | Prefill 2K / 8K vs Metal |
| --- | --- | --- | ---: | ---: | --- | --- |
| dual27 | 37-63 | full, pass | +0.00022 | 0.00020 | +0.00047 / 0.00061 | +11.7% / +12.4% |
| d36 | 28-63 | full, pass | +0.00026 | 0.00026 | +0.00069 / 0.00082 | +14.8% / +15.9% |
| d44 | 20-63 | quick (6 windows), pass | +0.00051 | 0.00066 | +0.00070 / 0.00094 | not timed |
| d64 | 0-63 | full, pass | +0.00024 | 0.00042 | +0.00101 / 0.00108 | +23.8% / +23.2% (old libraries), +29.9% / +30.2% (fresh build) |

d52 was skipped because d64 passed. Early layers turned out cheap: quality cost barely grew from 27 to 64 layers. Gate data: `gate/*.json`; timing: `timing/t2-*`, `t3-*`, `t4-*`, `t5-d64`, `t5-metal`.

## After the committed state

### 8. DeltaNet input projections on the ANEs (shelved)

Metal time per 512-token chunk, summed over layers (`microbench/projection-shares.txt`):

| Work | Time | Share of Metal prefill |
| --- | ---: | ---: |
| FFN, 64 layers | 1130 ms | 66.5% |
| DeltaNet `attn_qkv` + `attn_gate`, 48 layers | 273 ms | 16.1% |
| DeltaNet `ssm_out`, 48 layers | 119 ms | 7.0% |
| Attention q/k/v + o, 16 layers | 120 ms | 7.1% |
| DeltaNet alpha/beta, 48 layers | 19 ms | 1.1% |

Splitting `[attn_qkv; attn_gate]` (16384 rows) as Metal 8192 rows plus 4096 + 4096 rows on the two ANEs took 3.20 ms against 5.95 ms on Metal for layer 0, rel. L2 0.007 (`scripts/proj-split.mm`).

Integration (not committed; full change in [results/2026-10-30/deltanet-projection.patch](results/2026-10-30/deltanet-projection.patch)): `GGML_OP_CUSTOM` registration in ggml-metal, a hook in `build_qkvz()` returning contiguous `qkv`/`z` views of one buffer, and a `decoder_proj` runtime. The all-Metal control through the new boundary was bit-identical to Metal (KL 0). Packages came from `scripts/export_proj.py` / `scripts/make_proj.sh`.

Result for d64 plus 48 projection splits (d64p):

- Gate: fail on prose KL only. Overall ΔNLL +0.00046, KL 0.00110 (pass); prose ΔNLL +0.00187 (pass), KL 0.00301 against 0.003 (fail). KL roughly tripled against d64, probably because q/k/v errors enter the DeltaNet recurrent state.
- Speed: 8.997 s (2K) / 37.058 s (8K), 24% slower than Metal. FFN Core ML calls slowed from 9.56 to 24.91 ms because of the program limit below.

### 9. Core ML program limit (blocks further ANE offload)

One process keeps about 128 ANE programs resident (`microbench/residency.txt`, `scripts/residency.mm`, `scripts/make_models.py`):

| Loaded programs | Separate models, median / p90 | Multifunction package |
| ---: | --- | --- |
| 64 | 1.14 / 1.29 ms | 1.16 ms |
| 128 | 1.15 / 1.17 ms | 1.16 ms |
| 160 | 1.24 / 7.35 ms | function 129 fails to load |
| 192 | 1.32 / 7.36 ms | fails to load |
| 256 | 6.18 / 7.40 ms | fails to load |

Past the limit, calls evict and reload programs ("Must re-compile the E5 bundle"). Packing functions into one package does not help. The committed d64 already uses 128 programs (64 layers x 2), so more offload through public Core ML has to take programs away from the FFN, which saves about three times more per program. oMLX reaches the same model through Apple's private ANE API, packing everything into two programs.

### 10. Moving the FFN split point

Same binary, same session, back to back (`timing/t6-*`):

| Split (Metal / ANE + ANE) | Prefill 2K | Prefill 8K |
| --- | ---: | ---: |
| 8192 / 4608 + 4608 (committed) | 5.080 s | 21.299 s |
| 9216 / 4096 + 4096 | 5.269 s (-3.7%) | 21.848 s (-2.6%) |

A second session compared the other direction (`timing/t7-*`):

| Split (Metal / ANE + ANE) | Prefill 2K | Prefill 8K |
| --- | ---: | ---: |
| 8192 / 4608 + 4608 (committed) | 5.093 s | 21.210 s |
| 7168 / 5120 + 5120 | 5.749 s (-11.4%) | 23.719 s (-10.6%) |

Both directions are slower; moving work to the ANEs costs about three times more than moving it to Metal. The committed split is the best of the three, and finer steps would sit inside the session-to-session drift, so the split point was not tuned further. Neither alternative was run through the quality gate. Profiling had suggested the ANE side was the bottleneck (9.5 ms per call against 8.2 ms for the Metal part in isolation), but the isolated Metal time does not hold while both ANEs share memory bandwidth.

### 11. Core ML writes its outputs straight into the Metal join

Before this change, every FFN call copied both ANE outputs on the CPU (summing them into one buffer, about 1.1-1.3 ms) and then uploaded the sum (about 0.2 ms) while Metal waited. With `FFN_COREML_OUTPUT_BACKING=1`, each prediction gets `MLPredictionOptions.outputBackings` pointing at its own Metal shared buffer, and the Metal join adds three inputs (Metal part, ANE 1, ANE 2). If Core ML ignores a backing, the runtime copies instead and counts it; every run recorded 0 fallbacks. Same Core ML programs, so the 128-program budget is unchanged.

Same binary, same session (`timing/t8-*`, then `timing/t9-*` with Metal):

| | Metal | d64 | d64 + output backing |
| --- | ---: | ---: | ---: |
| Prefill 2048 (t8) | | 5.155 s | 4.772 s (+8.0%) |
| Prefill 8192 (t8) | | 21.588 s | 19.982 s (+8.0%) |
| Prefill 2048 (t9) | 6.797 s | | 4.776 s (+42.3% vs Metal) |
| Prefill 8192 (t9) | 28.075 s | | 20.067 s (+39.9% vs Metal) |
| Request incl. 128 decoded tokens, 2048 (t9) | 11.250 s | | 9.222 s (-18.0%) |
| Request incl. 128 decoded tokens, 8192 (t9) | 32.615 s | | 24.603 s (-24.6%) |

Full gate (`gate/d64bfull.json`), passes every check: ΔNLL +0.00026 (upper bound +0.00082), KL 0.00042; code -0.00014 / 0.00021, docs +0.00040 / 0.00030, math -0.00009 / 0.00013, prose +0.00109 / 0.00108; worst window doc-cli +0.00225. Only the order of the final additions changed, so the numbers match d64. Committed as `fbb340d` and enabled by default.

### 12. FP16 inputs and outputs for the ANE packages (measured, not pursued)

Core ML converts the FP32 activations to FP16 for the ANE and the FP16 results back to FP32. A layer-47 slice exported with FP16 input and output (`scripts/export-fp16-io.patch`; the tanh-SiLU constants must also be FP16) took 7.17 ms per call against 7.56 ms with FP32 I/O (`microbench/fp16-io.txt`), about 0.4 ms. Both ANEs run at once, so the end-to-end gain is at most about 0.4 ms per layer and chunk (roughly 2%), and less where Metal is the critical path. Numerics would not change. It needs an FP16 copy of the FFN input from Metal (a third custom-op input), FP16 join inputs, and exporter changes, so it was not built.

Side finding: loading the same compiled model twice and predicting concurrently serializes the two calls (14.7 ms for the pair); two different models run on the two ANEs in parallel (section 5).

### 13. 2048-token microbatches (measured, not kept)

Larger microbatches mean 4x fewer Core ML calls and Metal/ANE handoffs per prompt. Scratch changes made the token count configurable in the harness and exporter (`results/2026-10-30/ubatch-2048.patch`), and all 64 layers were re-exported at 2048 tokens with the same 8192 / 4608 + 4608 split.

Plain Metal also gets faster with larger microbatches (`microbench/ubatch-2048.txt`, llama-bench): 301.4 / 309.9 / 313.9 tok/s at 2048-token prompts for microbatches of 512 / 1024 / 2048 (+4.2% at 2048), and +4.4% at 8192-token prompts. In isolation, one 2048-token call per ANE was 8% more efficient than four 512-token calls (7.09 vs 7.69 ms per 512 tokens).

Same session (`timing/t10-*`), prefill per run:

| Configuration | 2048 tokens | 8192 tokens |
| --- | --- | --- |
| Metal, 2048-token microbatches | 6.517 / 6.509 / 6.508 s | 26.890 / 26.857 / 26.889 s |
| Split, 512 (committed) | 4.804 / 4.757 / 4.786 s | 20.028 / 20.046 / 20.071 s |
| Split, 2048 | 6.407 / 5.242 / 4.533 s | 25.527 / 23.635 / 25.395 s |

The 2048-token split was correct (exact offload counts, 0 output-backing fallbacks) but unstable and slower on average. It was not memory: peak footprint 52.8 GB, no swap-outs during the run (`microbench/u2048-memory.txt`). Inside the model, ANE calls took 47-61 ms on average with spikes up to 475 ms, against 28.4 ms in isolation (`microbench/u2048-profile.txt`); at 512 tokens the in-model penalty is about 1.2x. The likely cause is contention for memory bandwidth, or ANE working sets that no longer fit on chip, while the GPU runs at the same time; ANE-level profiling would be needed to confirm. The scratch code was reverted. 1024-token microbatches were not tried.

### 14. Engine-bench comparison (2026-10-30 to 2026-10-31)

The engine-bench skill (PR #424) ran its full matrix on the benchmark Mac, then the Core ML engine through the same client and flags as qvac-fabric (`-fa on -np 32 -kvu`, prefix caching off). The Core ML engine is plain `llama-server` with the sidecar registered at startup (`server-coreml.mm`, built with `FFN_COREML_SERVER=ON`, not committed). The sidecar was re-exported from the Q4_0 GGUF with the same preset; its full held-out gate passes (ΔNLL +0.00056, upper bound +0.00089, KL 0.00034, worst window prose +0.00210). Run directories on the Mac: `~/bench-weekly/runs/2026-10-30` (all engines), `2026-10-31-x` (Splash with and without its Neural Engine split, Core ML engine with DFlash2), `2026-10-31-fa` (new kernel), `2026-10-31-pad*` (padding), `2026-10-31-final` (Core ML engine built from both pushed branches, new kernel and padding: 398.7 / 434.0 / 400.0 / 366.0 / 316.5 / 255.3 t/s from 977 to 123k tokens).

Single-request prefill t/s by prompt length (old attention kernel, no padding):

| Prompt tokens | 977 | 4.5k | 21k | 38k | 70k | 123k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qvac-fabric | 326.3 | 326.4 | 298.7 | 272.2 | 232.9 | 183.8 |
| qvac-fabric + Core ML | 356.8 | 417.1 | 386.3 | 345.0 | 284.1 | 213.2 |
| Splash 1.5.0 | 379.6 | 431.9 | 413.8 | 382.9 | 332.2 | 270.9 |
| Splash, `--disable-ane` | 325.4 | 339.8 | 322.7 | 302.3 | 269.8 | 227.9 |

- Decode is unchanged by the split: 38.4 t/s without speculation, 105.2 with DFlash2 (same draft acceptance as qvac-fabric, 44.2%), against Splash's 109.7.
- Concurrency with short prompts (about 240 tokens) is within 1% of qvac-fabric.

### 15. Splash already splits the FFN onto the Neural Engine

Splash's log reports "Neural Engine FFN split at share 0.38 for chunks of 512 rows or more". Measured against itself with `--disable-ane`, its Neural Engine gain (+27% to +28% at 4.5k-21k) matches ours (+28% to +29%). The difference was the GPU path: with the Neural Engine off, Splash matched qvac-fabric at 1k tokens and was 24% faster at 123k. The part of prefill that grows with the prompt is attention in the 16 full-attention layers; the rest of each 512-token microbatch costs about 1.57 s at any depth. Splash's metallib names GQA-grouped prefill attention kernels (`prefill_attention_gate_kv2_g12` and similar).

### 16. Metal prefill attention kernel (branch `feat/metal-fa-prefill-d256`)

Microbench of the model's attention shape (24 query heads, 4 KV heads, head size 256, F16 KV, causal mask, 512 queries): the existing kernel ran at 10-12 TFLOPs at every KV length, against 18.6 TFLOPs for a plain F16 matmul (`mul_mm`). This model accounts for the end-to-end numbers: at 123k tokens attention took about 300 s of qvac-fabric's 670 s.

The new kernel `kernel_flash_attn_ext_tb` takes 32 query rows per threadgroup with 8 simdgroups. Per 64-row KV tile each simdgroup computes a 16x16 block of S (Q staged as half in threadgroup memory, K from device memory), the online softmax runs on S in threadgroup memory, and each simdgroup accumulates P*V into its own 32 x 32 slice of O in registers. About 14 TFLOPs; outputs match the old kernel within 1e-7. It is used for prefill with head size 256, F16 K/V, KV length a multiple of 64, a mask and no sinks/ALiBi/softcap; `GGML_METAL_FA_TB=0` falls back.

| KV length | Old kernel | New kernel |
| --- | ---: | ---: |
| 4,096 | 4.58 ms | 3.82 ms |
| 65,536 | 78.1 ms | 57.1 ms |
| 122,880 | 145.6 ms | 106.7 ms |

`llama-bench` pp512 on qvac-fabric: 334 t/s at depth 0 with both kernels, 179.7 to 206.4 t/s at depth 65,536 (+14.8%). Server sweep (same client as section 14):

| Prompt tokens | 977 | 4.5k | 21k | 38k | 70k | 123k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qvac-fabric | 326.3 | 327.3 | 306.3 | 285.3 | 253.8 | 213.1 |
| qvac-fabric + Core ML | 354.9 | 424.6 | 398.9 | 366.6 | 316.5 | 255.6 |
| Core ML vs Splash | -6.5% | -1.7% | -3.6% | -4.3% | -4.7% | -5.6% |

Decode is unchanged (38.44 t/s). Layouts tried and dropped, all correct but no faster than the old kernel: 32 queries with K/V staged in threadgroup memory (5-6 TFLOPs), Q and O in registers with K/V from device memory (10.8), GQA-grouped 6 heads per K/V tile (9.1-9.7), head split across simdgroup pairs (11.3). On the new kernel, splitting Q*K across the head, an exp2 softmax, half-precision P, half accumulators and freeing threadgroup memory for occupancy each changed speed by under 4%. Removing the Q*K or P*V multiplies (timing only) put the phases at about 14.7 and 16.6 TFLOPs; the softmax and barriers take about 11%.

### 17. Padding partial microbatches (`7bad9c2`)

The Core ML programs take exactly 512 rows, so the last partial microbatch of a prompt ran on Metal alone. `decoder_build` now pads a partial microbatch of at least `FFN_COREML_PAD_MIN` (default 256) rows with zero rows to 512 and returns a view of the real rows. A greedy 48-token completion of a 1,513-token prompt gave the same text with padding on and off (top-token log-probabilities within 0.003). Prefill t/s with the new kernel, mean of two alternating rounds:

| Prompt tokens | Last microbatch | Off | On | Change |
| --- | ---: | ---: | ---: | ---: |
| ~710 | 192-204 | 377-383 | 351-354 | -7% to -8% |
| ~770 | 258 | 374.3 | 379.4 | +1.4% |
| ~980 | 471 | 357.8 | 402.8 | +12.6% |
| ~1,980 | 447 | 406.0 | 432.6 | +6.6% |
| ~4,540 | 444 | 425.7 | 436.6 | +2.6% |

At 1k tokens this is 6% ahead of Splash (379.6), and level at 4.5k (431.9).

### 18. Quality with the new kernel

The full held-out gate (23 windows, 65,536 tokens) was rerun on the Q4_0 sidecar with the new kernel; split and plain Metal both use it. It passes every check: ΔNLL +0.00060 (upper bound +0.00092), KL 0.00035, worst window prose +0.00251, exact offload counts, no failed runs. With the old kernel the same gate gave +0.00056 / +0.00089 / 0.00034 / +0.00210. Plain Metal's own NLL per window with the new kernel against the old differs by -0.00001 on average and at most 0.00008, so the kernel does not change quality. The gate's prompts are multiples of 512 tokens, so it does not exercise padding (section 17 checks that). Summaries: `results/2026-10-31/gate-q40-*-summary.txt`.

### 19. Overlapping attention with the ANE FFN (measured, not built)

The ANEs only work during the FFN step, so they are idle while Metal runs attention, DeltaNet and the projections. A profiled 8,192-token prefill with the split (new kernel, `FFN_COREML_PROFILE=1`, 18.97 s, 1,024 Core ML calls) shows the per-layer sequence: Metal works 9.1 ms before each FFN (the CPU waits on it, ANEs idle), then the ANE call takes 9.25 ms while Metal runs its 8,192-channel share (about 7 ms) and then waits. The ANEs are idle 9.36 s of the 18.97 s. On long prompts the Metal step grows with attention: at 123k tokens the average layer takes about 32.6 ms, matching the measured 2.0 s per microbatch.

Contention, measured with the attention microbench, a 17408 x 512 x 5120 F16 matmul and both ANE partitions of layer 40 running in a loop (`overlap/ane_loop.swift`): none in either direction. Attention 57.8 / 111.4 ms alone and 57.8 / 110.7 ms with the ANEs busy (64k / 123k KV); matmul 5.34 ms alone, 5.20-5.24 ms alongside; ANE calls 7.33 ms alone, 7.25-7.30 ms alongside. The slowdown seen with 2048-token microbatches (section 13) does not occur with this work.

If a second 512-token chunk's Metal work ran during the first chunk's ANE call (two chunks interleaved layer by layer; chunk B's attention at layer L only needs chunk A's KV at layer L), the step per layer becomes the larger of the Metal and ANE work. Estimated from the profile: about -15% per layer at short prompts and about -9% at 123k tokens with the current split; moving more FFN channels to the ANEs (they would no longer wait for Metal) gives about -20% and -15%. The gain is per layer, so the 16 attention layers stay Metal-bound on long prompts. Splash is 5.6% ahead of us at 123k, so this would roughly close or reverse that.

What it needs: two 512-token chunks per graph interleaved per layer in the Qwen graph (a 1,024-token microbatch), asynchronous Core ML predictions joined later through a CPU-signalled `MTLSharedEvent` the Metal command stream waits on, and changes to ggml's Metal encoding so other work can be encoded between the submit and the wait. Several days of work plus validation. Evidence: `results/2026-10-31/overlap/`.

### Unattributed runs

`unattributed/` holds logs found on the benchmark Mac whose launch is not part of this session's record: a 32-layer profile (`prof-d32`), and a 9728 / 3840 + 3840 export and timing (`export-d64r`, `t6-d64r`: 5.142 s at 2K, 21.362 s at 8K). The latter has no same-session d64 baseline, so it does not settle the split question.

## Measurement notes

- Compare configurations only within one session. d64 at 2048 tokens measured 5.491 s (old libraries), 5.235 s and 5.080 s (fresh build) on different runs; session-to-session drift is about 3%.
- Metal itself is stable: medians at 2048 tokens of 6795.4-6799.9 ms across five sessions (512-token microbatches). Its best setting is 2048-token microbatches, about 4% faster.
- Before committing `results/2026-10-30/`, exclude `results/**/scripts/*.py` from `ty` (they import coremltools, like the exporter override in `ty.toml`).
- The benchmark Mac is shared. The disk filled twice during the work (other accounts); Core ML fails to write its compile cache when free space is low, which can push layers off the ANE. Keep several GB free.
- The quality numbers for each configuration are from `quality` mode (full-vocabulary teacher-forced logits against Metal); timing numbers are from `split` mode with greedy decoding.

## Suggested next steps

1. Overlap a second chunk's Metal work with the current chunk's ANE FFN (section 19): no contention, about 9-15% expected at long prompts and 15-20% at short ones. Needs two chunks per graph and asynchronous Core ML calls.
2. The split point (8,192 Metal channels) should not depend on prompt length, since the ANEs only work during the FFN step, whose cost does not change with depth; not measured at long prompts.
3. For `feat/metal-fa-prefill-d256`: run `test-backend-ops -o FLASH_ATTN_EXT`, measure on other Apple GPUs, and consider the same layout for head size 128.
4. Run a task-level check (for example HumanEval and math word problems, greedy) on d64 against Metal.
5. Output copies are removed (section 11). FP16 inputs/outputs would save at most about 2% more (section 12).
6. Production work before any server use: lifecycle, concurrency, cancellation, cache validation, memory and power measurement.
7. Single-ANE Macs need the single-package layout (`ane_partitions: 1`) and their own balance point; the dual preset would likely give little or no gain there.
8. Further ANE offload (DeltaNet projections, attention projections) is blocked by the 128-program limit through public Core ML.
9. Larger microbatches were unstable at 2048 tokens (section 13); 1024 is untested.
10. Not yet measured: the GPU timeline around each handoff, and runs without `FFN_COREML_SINGLE_ENCODER`.

## Where things are

- Branch `feat/coreml-qwen-3.8`: runtime, exporter, presets, this log, evidence.
- Branch `feat/metal-fa-prefill-d256` (on `temp-11259`): the prefill attention kernel (section 16).
- Benchmark Mac `~/ffn-probe-20261029/`: the held-out text and gate scripts (`heldout/`), full logs, and the projection-aware build (`tree/build-proj/`). The 64-layer UD-Q4_K_XL sidecar (`dual27/`) was deleted on 2026-10-30 to free disk space; it and all other assets can be regenerated with `export_sidecar.py` and the scripts above.
- Benchmark Mac `~/bench-weekly/`: the engine-bench checkout with `run_engine.sh` patched for the Core ML engine (`fabcml`) and `SPLASH_EXTRA` (originals in `*.orig`), the Q4_0 sidecar selection (`cml/sel-q40`, compiled caches in `~/ffn-probe-20261029/q40-a`, `q40-b`), gate scripts (`cml/run_gate_q40*.sh`) and the run directories listed in section 14.
- Evidence for sections 14-19: [results/2026-10-31/](results/2026-10-31/).
- The new-kernel builds used in sections 16-19 (`~/fa-work`, `~/bench-weekly/src/fabric-coreml-fa`, `~/ovl`) were removed after the runs. To rebuild: check out `feat/coreml-qwen-3.8`, merge `feat/metal-fa-prefill-d256`, and build `examples/ffn-coreml` with `FFN_COREML_DECODER_EXPERIMENT=ON FFN_COREML_SERVER=ON` (as in `results/2026-10-31/scripts/fa_run.sh`).
