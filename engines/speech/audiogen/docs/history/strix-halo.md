# AceSTEP on AMD Strix Halo — optimization campaign ledger

Archived development record. Commands, layouts, status statements and open
items describe the revisions recorded here and are not current usage guidance.
For supported behavior, see the [engine README](../../README.md).

Goal: 2x faster end-to-end generation vs baseline on the Strix Halo Vulkan backend
(turbo-Q4_K_M and turbo-Q8_0, 30 s, warm), quality >= 98% (bit-correct preferred),
no regressions on other backends/models. sft-Q8_0 tracked at milestones.

Machine: AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU (RADV GFX1151, RDNA3.5, 40 CU),
Mesa RADV 25.2.8, 2 GiB VRAM carve + 112 GiB GTT (UMA), 121 GB RAM, 32 threads.
Runtime device line: `uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | int dot: 0 |
matrix cores: KHR_coopmat`.

Method: every iteration = hypothesis -> cheapest verification first -> change ->
gate -> regression net -> verdict. Verdict has two axes (observation reproduced /
mechanism confirmed); only mechanism-confirmed counts as CORRECT. Three WRONG in a
row => step back + independent review.

Benchmark protocol: the campaign workspace's `bin/bench.sh` (kept outside the
repo) — pinned prompts P1
(pop, clean English lyrics) / P2 (rock), seeds 42/1337, 30 s, 1 discarded cold run +
3 warm runs, median warm wall = the metric. Results: benchmarks/results.jsonl.

## Status

CAMPAIGN COMPLETE. Final milestone m3 (branch HEAD, all changes committed):

| Config | Baseline warm | Final warm | Speedup | Cold speedup |
|---|---|---|---|---|
| turbo-q4 / P1 | 15.8 s | 7.6 s | 2.07x | 2.09x |
| turbo-q4 / P2 | 16.9 s | 7.6 s | 2.22x | 2.24x |
| turbo-q8 / P1 | 19.3 s | 7.9 s | 2.45x | 2.51x |
| turbo-q8 / P2 | 17.5 s | 7.8 s | 2.24x | 2.43x |
| sft-q8 / P1 (not gated) | 22.5 s | 12.5 s | 1.81x | 1.78x |

All m3 output hashes identical to milestone m2 (bit-exactness held through the
DiT graph cache and tiled-im2col changes). Remaining known headroom, awaiting
direction: Vulkan per-submit encode (LM ~2.5 s host; HIP proves the ceiling),
CUDA/HIP ports of the tiled kernels, ROCm allowlist validation. The IM2COL_3D
workgroup-overflow crash, pre-existing at baseline, was fixed on the ggml
branch after this table was recorded (y grid-stride + host clamp; the full
single-process Vulkan op sweep now completes with zero failures).

Campaign history below.

## Earlier status (Phase 0)

Phase 0 (infrastructure) complete; Phase 1 baseline matrix running.

Phase 0 verified facts:
- Builds: ggml@speech Vulkan (ninja+ccache, install ~/ggml-install/vk) and audiogen
  (build/audiogen) green; all 5 unit tests pass (LD_LIBRARY_PATH must include the
  ggml install lib dir).
- Regression reference: per-op test-backend-ops sweep on Vulkan0 — 137/137 ops
  covered, all pass except pre-existing IM2COL_3D crash
  (GGML_ASSERT ggml-vulkan.cpp:7123 workgroup-count overflow; not in AceSTEP graphs;
  candidate generic fix). Full-sweep-single-process log truncates at that crash, so
  per-op sweeps (campaign bin/sweep_ops.sh) are the reference method.
- Coopmat ground truth (probe): 16x16x16 f16xf16 with f16 AND f32 accumulators
  (FA_COOPMAT1 with GGML_PREC_F32 is possible); full int8 coopmat set present but
  unused by current shaders. No bf16 coopmat. VK_EXT_external_memory_host present
  (H3c viable). System glslc lacks GL_EXT_integer_dot_product -> int dot: 0 (H11).
- Smoke run observation (turbo-q4, requested --dur 5, seed 42, GPU): LM 8460 ms
  (56%), VAE 4687 ms (31%), DiT 1014 ms (7%); total 15.3 s. LM runs on CPU by
  default policy. NOTE: --dur forces duration metadata but the LM generated codes
  to the max_tokens cap (dur*5+100 = 125 codes -> 25.2 s actual audio). Benchmarks
  record actual WAV seconds; RTF computed against actual.

## Baseline (Phase 1) — measured 2026-08-24, engine e0e08fc4 / ggml cd3b4159

Protocol: 30 s requested (actual audio 29.4-30.2 s), 1 cold + 3 warm, median warm.
All runs bit-deterministic (identical WAV sha256 across cold+warm of each config).

| Config | Cold (s) | Median warm wall (s) | RTF | 2x target (s) |
|---|---|---|---|---|
| turbo-q4 / P1 / s42 | 15.4 | 15.8 | 2.08 | 7.9 |
| turbo-q4 / P2 / s42 | 16.6 | 16.9 (CV 11%) | 1.75 | 8.5 |
| turbo-q8 / P1 / s42 | 19.7 | 19.3 | 1.46 | 9.7 |
| turbo-q8 / P2 / s42 | 18.9 | 17.5 | 1.69 | 8.8 |
| turbo-q4 / P1 / s1337 | 16.9 | 18.4 | 1.62 | seed-variance probe |
| turbo-q8 / P1 / s1337 | 17.8 | 16.6 | 1.63 | seed-variance probe |
| sft-q8 / P1 / s42 | 22.2 | 22.5 | 1.33 | tracking only |

Stage shares (median warm): turbo-q4 = LM 53.5% (8.3 s, CPU) / VAE 37.8% (5.9 s) /
DiT 6.4% (1.0 s); turbo-q8 = LM 58.9% (11.2 s) / VAE 33.2% / DiT 5.7%; sft = LM 46% /
VAE 26.7% / DiT 25.3% (5.7 s, 50 steps ~113 ms/step). wall - stages_sum ~ 0.3 s only.

Observations with verdicts:
- LM-on-CPU stage time is HIGH-VARIANCE (7.4-13.1 s across all baseline runs,
  same 677 MB Q8_0 LM). The apparent q4-vs-q8 LM delta was noise, NOT a mechanism
  (governor=powersave, shared power envelope). VERDICT: noise — do not chase.
  GPU stage times are stable (DiT ~990/1080 ms, VAE 5.4-6.3 s).
- GPU temps 27-36 C before/after every run — no thermal confound at these loads.
- --dur is a soft target: LM generates codes to its own EOS/cap; requested 30 s
  produced 29.4-30.2 s. RTF computed against actual WAV seconds.

## Phase 2 — env A/B matrix (turbo-q4/P1/s42, 1 warmup + 2 timed each)

Caveat applied to all rows: LM-on-CPU noise (7.4-13.1 s) pollutes wall deltas of
rows that do not touch the LM; verdicts below use the stable VAE/DiT stage columns.

| Row | Result | Verdict |
|---|---|---|
| A1 ACESTEP_LM_GPU=1 | wall 11.1/11.3 s (base 15.8), LM 4638/4066 ms (was ~8300) | CONFIRMED observation: LM-on-Vulkan ~2x faster than CPU on Strix. Output deterministic but differs from CPU-LM (expected); quality gates pending. |
| A2 ACESTEP_KEEP_STAGES=1 | WAV bit-identical to baseline; VAE stage 4700 ms (was ~5900), DiT 860 ms (was ~990) | CONFIRMED: ~1.2 s VAE + 0.13 s DiT of per-generate weight reload eliminated. H3a mechanism proven (load cost is inside stage timings). |
| A3 DISABLE_HOST_VISIBLE_VIDMEM | VAE/DiT unchanged | no effect — killed |
| A4 ALLOW_GRAPHICS_QUEUE | VAE slightly worse | no win — killed |
| A5 DISABLE_FUSION (control) | VAE/DiT unchanged | expected no-op confirmed — methodology control passed |
| A6 DISABLE_COOPMAT | DiT 1421 ms (+43%), VAE +8% | coopmat active and valuable in baseline (negative control) |
| A7 ASYNC_USE_TRANSFER_QUEUE | unchanged | no effect — killed |
| A7b DISABLE_ASYNC | unchanged | no effect — killed |
| A9 ALLOW_SYSMEM_FALLBACK | unchanged | no effect — killed |

H4 pre-verification: PASSED (A1). H3a pre-verification: PASSED (A2).

Combined LM_GPU+KEEP_STAGES (single-shot CLI): q4 12.2 s, q8 12.2 s — WORSE than
A1 alone (11.2 s) because KEEP_STAGES moves weight loads upfront; for a single
generation it only relocates the ~1.5 s load cost. KEEP_STAGES pays off only for
multi-generation processes (addon). Single-shot load cost must be attacked via
faster loads (H3c zero-copy), not via residency. Learning recorded.

H4 quality gates:
- lm-smoke --gpu --quantized-batch-cfg-regression (Q8 LM): PASS, cosine=1.000000000,
  max_abs=0, argmax 189417/189417 — GPU LM logit-exact vs the gate's tolerance.
- WER (large-v3-turbo, 30 s clips; absolute WER dominated by deletions since only
  part of the lyrics fits in 30 s; seed band ~0.10): baselines q4 0.575 (s42) /
  0.479 (s1337), q8 0.603 (s42); GPU-LM q4 0.657 (within band, PASS);
  GPU-LM q8 0.781 (single sample, outside band) -> extra seeds being scored
  before verdict. Sampled tokens legitimately differ from CPU-LM (different song,
  not degraded logits — parity gate above is the mechanism evidence).
- OBSERVATION (not our target): base-sft P1/s42 output has WER 1.0 (zero lyric
  hits in first 30 s) — sft path parity is documented as unverified upstream;
  flagged for user listening, not investigated further in this campaign.

## Milestone m2 (H4 + col2im + LM graph cache + parallel VAE load + large transpose)

| Config | Cold | Median warm | Speedup (warm) | Speedup (cold) |
|---|---|---|---|---|
| q4 / P1 | 7.5 | 7.5 | 2.09x | 2.04x |
| q4 / P2 | 7.5 | 7.5 | 2.25x | 2.21x |
| q8 / P1 | 7.7 | 8.1 | 2.40x | 2.57x |
| q8 / P2 | 8.0 | 8.0 | 2.18x | 2.38x |
| sft / P1 | 12.6 | 12.6 | 1.79x (not gated) | 1.76x |

All outputs bit-deterministic. 2x bar CLEARED on all gated configs.

Milestone verification:
- CPU lane: lm-smoke CPU logit dumps bit-identical to pre-campaign build
  (graph cache + parallel load proven neutral off-GPU).
- Gate C WER: P1 within band. P2 s42 single sample exceeded band; 3-seed
  study (7/1337/9001) shows GPU-LM WER 0.53/0.42/0.78 vs CPU-LM
  0.82/0.84/0.84 — GPU-LM is at least as lyrical; s42 was an unlucky draw.
  PASS. sft WER 1.0 both before and after (pre-existing upstream sft parity
  gap; not a regression).
- Full per-op Vulkan sweep vs Phase-0 reference: IDENTICAL — 133 ops OK, only
  the pre-existing IM2COL_3D workgroup-overflow crash (present at baseline).
  Zero new failures.
- Shared-path guard: whisper-cli (large-v3-turbo) ran dozens of transcriptions
  against the same modified ggml install throughout the WER batteries —
  the shared Vulkan paths (matmul, FA, transposes) behave correctly for
  whisper as well.
- Addon RTF bench: NOT run — the npm addon consumes released vcpkg ports and
  cannot link this campaign's local trees without a registry release; the
  music-cli end-to-end benchmarks stand in for the production entry point.

## HIP/ROCm data point (user-approved scope; no pivot)

Build: -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 (ROCm 7.2), audiogen linked against
it; registry falls through to ROCm0 (unvalidated pass), default placement
keeps LM+detok on CPU as designed. 30 s, P1/s42, 1 warmup + 1 timed:

| Config | Wall | LM | VAE | DiT |
|---|---|---|---|---|
| HIP default (LM CPU), q4 | 15.8 s | 10.7 s | 2.9 s | 1.1 s |
| HIP + ACESTEP_LM_GPU, q4 | 7.8 s | 2.8 s | 2.8 s | 1.1 s |
| HIP + ACESTEP_LM_GPU, q8 | 7.8 s | 2.8 s | 2.9 s | 1.0 s |

Findings: overall parity with optimized Vulkan (7.5-8.1 s) with opposite
strengths — the HIP/CUDA LM decode is ~1.6x faster than Vulkan's (better
per-token submit path), while the tiled Vulkan VAE is ~1.8x faster than HIP's
(our col2im/transpose kernels are Vulkan-only; CUDA inherits its own). ROCm
stays off the allowlist (would need the F32-parity protocol + addon backend
map work); Vulkan remains the product path. Future headroom: LM submit-side
work on Vulkan, or CUDA ports of the tiled kernels.

### H1 — WRONG magnitude, kept — DiT graph cache (stretch item)

- Hypothesis: per-step DiT graph+gallocr rebuild costs ~16-24 ms/step; caching
  it should cut ~1 s from sft (50 steps).
- Change: DitGraphCache mirrors LMGraphCache (key: T/N/enc_S/H_enc/mask
  presence); inputs still re-uploaded every step.
- Outcome: bit-identical WAVs (q4 and sft), unit tests pass, but DiT stage
  time UNCHANGED (sft 5713 vs 5683 ms). The stage-wall-minus-GPU-op gap is
  command-buffer encode + input upload/readback, not graph construction.
  Kept: removes per-step allocator churn and is the prerequisite for any
  future device-resident-latent step; cost-neutral.
- STEP-BACK (two magnitude-misses in a row, H5 then H1, same class): "host
  overhead" on this Vulkan backend is dominated by per-submit command
  encoding, which neither graph caching nor allocation reuse touches.
  Further wins in that class need backend-level command replay — out of this
  campaign's scope. The class is closed in this ledger; remaining in-plan
  levers are memory-pattern kernels (im2col tiling) only.

### H10c — CORRECT — tiled 1D im2col Vulkan pipeline (stretch item, in-plan)

- Same strided-read disease as col2im (KW-wide gathers at input-row stride):
  im2col_1d_tiled.comp stages the contiguous input window for a 128x16 output
  tile in LDS; writes stay contiguous along CHW. Gated: 1D, window fits LDS,
  dst addressable without BDA, SIGNAL >= 32 MiB (crossover measured — the
  cache-resident-input shape prefers the generic shader and stays on it).
- TRAP found: on BDA-capable devices ggml_vk_op_f32 binds the im2col dst as a
  1-byte dummy (stock shaders write via device address). A non-BDA pipeline
  behind GGML_OP_IM2COL silently writes into the dummy -> garbage without any
  error. Fixed by binding the real dst for the tiled pipelines. Recorded as a
  structural fact.
- Results: VAE-shaped 1D im2col 74-87 -> 157-186 GB/s on DRAM-resident
  shapes (per-window im2col 10.6 -> 6.9 ms); engine q4 7.7 s / sft 12.4 s,
  WAVs bit-identical. Correctness suite extended (ragged large 1D cases,
  k=1 and dilated k=7); IM2COL/CONV_TRANSPOSE_1D suites green.

## Hypothesis ledger

(entries appended below; format: Hn — status — hypothesis — why — test — changes —
outcome — verdict (observation / mechanism) — learning — next)

### H4 — CORRECT (mechanism confirmed) — LM belongs on the GPU on Strix Vulkan

- Hypothesis: the LM-on-CPU-under-Vulkan policy is a Mali carry-over; on Strix
  Halo RADV the GPU LM is faster with no quality loss.
- Why: policy comment cites only Mali-G715; Strix has coopmat+fp16 and the LM is
  54-59% of baseline wall.
- Test before change: A1 env A/B (ACESTEP_LM_GPU=1) -> wall 15.8->11.2 s, LM
  8.3->4.3 s. Quality: lm-smoke batched-CFG PASS (GPU-internal consistency);
  then F32-dequant reference protocol (README "Backends"): on 3 seeds of
  300-token prefills GPU-Q8 matches F32 argmax 3/3 with cosine >= 0.99999995,
  top-50 overlap 50/50; CPU-Q8 matches argmax 1/3, cosine ~0.9998, max_abs up
  to 5.4. GPU LM is MORE faithful than the CPU path (CPU quantizes activations
  to Q8_1). WER across seeds statistically inconclusive (seed band ~0.1-0.3)
  and superseded by the mechanism evidence; sampled songs legitimately differ.
- Changes: stage_placement.h vulkan_device_lm_validated (device-desc "RADV"
  substring, per-device allowlist), resolve_stage_placement takes device_desc;
  backend_registry.h backend_dev_description helper; engine.cpp passes it;
  placement unit tests extended (RADV lifts LM, Mali/Xclipse/NVIDIA/proprietary
  AMD/non-Vulkan names do not, hatches still override); engine README updated.
- Outcome: default run places lm=Vulkan0 on this box; WAV bit-identical to the
  A1 env run (sha b6e2839802d5...); wall 10.9 s (1.45x vs q4 baseline on that
  run). All 5 unit suites pass. HOW confirmed: placement log line + hash match.
- Learning: same-backend smoke gates are NOT cross-backend parity evidence;
  the F32-dequant protocol is the decisive instrument. Mali default unchanged.
- Next: remeasure milestone matrix on new HEAD; then attack VAE (~5 s) and
  LM-on-GPU internals (H5 gallocr churn, MMVQ), DiT H1/H2, load cost H3c.

### Milestone m1 (H4 only): q4 12.0 s (1.32-1.41x), q8 12.2 s (1.46-1.58x),
sft 16.8 s (1.34x); all deterministic. Perf-logger decomposition: VAE GPU op
time dominated by COL2IM_1D (2391 ms of 3684 ms); LM stage wall 4.8 s vs 1.8 s
GPU op time => ~3 s host-side churn; DiT ~0.87 s GPU-bound.

### H10a — CORRECT (mechanism confirmed) — tiled col2im_1d Vulkan kernel

- Hypothesis chain (2 refuted, 1 confirmed — kept for the record):
  (1) REFUTED "uncoalesced per-thread reads": first tiled kernel (TT=128,
  OCB=8, 256 thr) only gave 2-2.5x; (2) REFUTED "UMA prefer_host memory type":
  prefer-host OFF rebuild changed nothing (H3b answered on Strix: keep ON);
  (3) CONFIRMED by microprobes: disabling global loads made the kernel 30-60x
  faster while disabled gather changed nothing => the entire cost is DRAM reads
  of small (256 B) segments at large (8 KiB) stride; a 393 MB linear ADD
  streams at 208 GB/s on the same box, so it is pattern, not bandwidth.
- Fix: col2im_1d_tiled.comp — 64 t_out x 64 oc tile staged in 48 KiB LDS,
  1024 threads, contiguous 1-5 KiB row-slab loads, coalesced writes, identical
  t_in summation order (bit-identical results). Host gates: pipeline created
  only when device limits allow (>=1024 invocations, >=48 KiB shared); tiled
  selected only when smem fits, grid fits, and columns+signal >= 32 MiB
  (cache-resident shapes stay on the untiled pipeline, which wins there).
- Results (test-backend-ops perf, ACE-Step VAE shapes): 4.8-22 GB/s -> 88-146
  GB/s; per-generation col2im ~2391 ms -> ~40 ms. Engine: q4 15.8 -> 9.2-9.3 s
  (1.70x), q8 19.3 -> 8.8 s (2.19x, past the bar). Gate A: WAVs bit-identical
  pre/post for both quants. Correctness suite extended with tiled-path shapes
  (NOTE: the gate later changed from an element-count to a 32 MiB byte
  threshold; the eval shapes were re-enlarged to cross it in the review pass)
  (ragged tails, k != 2*s0) — all pass; SNAKE/ZERO_UPSAMPLE/CHANNEL_SHUFFLE/
  AFFINE_PRELU/CPY unaffected.
- Learning: on Strix Halo, small-segment strided DRAM reads are ~20x slower
  than linear streaming; occupancy (threads in flight) doubles throughput
  twice (256->512->1024 threads). Microprobe-by-deletion beats access-pattern
  theorizing. Also: perf-mode batching of a slow kernel can trip the GPU
  watchdog (context lost) — not an engine bug.
- Next: LM host-side churn (H5), then loads (H3c), DiT (H1/H2).

### H5 — PARTIAL (observation smaller than hypothesis) — LM graph cache

- Hypothesis: per-token graph+gallocr rebuild is most of the LM stage's ~3 s
  host gap. Change: LMGraphCache in lm_ggml.cpp — the decode graph and its
  allocation are keyed on (batch, S, n_kv_pad, kv set, head tensor) and reused
  until the key moves (n_kv_pad steps by 256), for both the single and
  batched-CFG paths; debug layer_states path stays uncached.
- Gates: per-step logit dumps BIT-IDENTICAL to pre-change (prefill 300 +
  decode 100), batched-CFG regression PASS, unit tests pass, engine WAV hash
  unchanged.
- Outcome: LM stage only 4.7 -> 4.55 s. Perf-logger split shows the remaining
  host cost is per-submit command encoding + fence (~4.5 ms/token vs 5.3 ms
  GPU), comparable to upstream Vulkan norms. Verdict: mechanism partially
  right, magnitude wrong; deeper submit-side work deprioritized (cheapest
  remaining levers first).
- Learning: graph BUILD cost is small vs command-buffer encode on this
  backend; measure the split before attacking "host overhead" as one number.

### H3-load — CORRECT — parallel VAE weight-norm resolution

- The default per-generation VAE load spent ~1.2 s in single-threaded
  weight-norm resolution (bf16 g/v -> per-channel norm/scale/transpose) +
  f32->f16 conversion (vae_gguf.cpp). Parallelized per-channel (rows
  independent -> bit-identical). Engine: q4 9.2 -> 8.0 s, hash unchanged.

### H10b — CORRECT — large-tile transpose copy + switch-routing bug fix

- VAE decode does 20 big [T,C]->[C,T] transposes (op_conv_t1d cont) = 334 ms;
  test-backend-ops showed the upstream 32x32 cpy_transpose collapsing to
  2.6-5 GB/s on DRAM-resident skinny matrices and HANGING the GPU (watchdog)
  on large f16 cases. Root cause same as col2im: 128 B strided segments.
- SELF-INFLICTED BUG found en route: the earlier col2im elements-switch split
  had attached the shared elementwise case-label list (CPY/UNARY/GLU/...) to
  the new COL2IM_1D case, silently dropping the cpy_transpose and conv2d_dw
  element overrides — correctness survived only because the tiled shaders
  grid-stride over the full range. LESSON: when splitting a case out of a
  shared label list, re-check which labels the ORIGINAL body retains; verify
  with a print that the intended branch actually executes before trusting a
  contradictory measurement.
- Fix: copy_transpose_large.comp (64x128 LDS tile, 512-B reads / 256-B
  writes, >=32 MiB + device-limit gated) + restored case routing. Transposes:
  5 -> ~190 GB/s (f32), f16 hang gone. Correctness: full CPY/DUP/CONT +
  affected-op suite green, ragged-dim large cases added.
- Engine result: q4 7.9 s (2.00x), q8 8.1 s (2.38x), WAVs bit-identical.

### Pre-registered backlog (from plan, ordered; re-rank after Phase 2 data)

- H4 LM placement: LM-on-Vulkan may beat LM-on-CPU on Strix (policy is a Mali
  carry-over). Pre-verify: ACESTEP_LM_GPU=1 A/B.
- H1 DiT graph+gallocr reuse across Euler steps. Pre-verify: perf-logger gap
  (stage wall - GPU op time).
- H2 Euler+DCW in-graph, latent resident on device. Pre-verify: transfer ops in
  perf-logger.
- H3 weight residency: (a) KEEP_STAGES policy, (b) UMA prefer_host_memory A/B
  (rebuild), (c) zero-copy weights via VK_EXT_external_memory_host.
- H5 LM per-token gallocr churn.
- H7 RDNA3 subgroup pinning entry + RDNA3.5 shmem occupancy tuning.
- H8 FA path: confirm f32acc coopmat / FA actually running; tune or re-precision.
- H6 shared GPU backend instance (engine + VAE).
- H10 VAE kernel/fusion work (only if VAE share stays large).
- H11 integer-dot shaders: system glslc lacks GL_EXT_integer_dot_product so `int
  dot: 0` at runtime; newer glslc may unlock int8-dot paths for q8/q4 on RDNA3.
