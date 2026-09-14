# Qwen4Exp optimization worklog

Last updated: 2026-09-14 (Asia/Singapore)

## Objective

Get Qwen4Exp FP8 performance close to SGLang on Hopper while keeping the work on
`qwen4-exp-opt`. Use only GPUs 6 and 7 on `aman@cosmicac-b4c09bd2`.

The next comparison must be apples-to-apples: same prompt length, request/batch
shape, generated-token count, GPU count, model, warmup state, and CUDA-graph
mode. The current SGLang decode number appears batched and must not be compared
directly with single-stream `llama-bench` token generation.

## Git checkpoint

Current branch: `qwen4-exp-opt`

Recent commits, newest first:

```text
4e1f24d24 cuda: fuse MoE weighted expert reduction (#25952)
9d77e2b65 ggml: allow passing alloc dependencies in graph_optimize (#27301)
d907668f2 docs: record qwen4exp pp2048 profile
0b5c41852 docs: record qwen4exp optimization worklog
6d2008c5d cuda: prototype DeepGEMM fused MoE paths
677d4d80d tests: add SM90 DeepGEMM MegaMoE prototype
e94017aff qwen4exp: align FP8 tensor split with MoE graph
060b4dcbf qwen4exp: add FP8 DeepGEMM checkpoint
9db850853 cuda: enable sparse fa for qwen4exp
```

`6d2008c5d` is a useful correctness/feasibility checkpoint, but its fused path is
not the performance direction in its current form.

The original worklog checkpoint was clean after `6d2008c5d`. Subsequent
tracked work is represented by the commits above. Pre-existing untracked files
were intentionally left untouched.

## Machines and paths

- Remote: `ssh aman@cosmicac-b4c09bd2`
- Authorized GPUs: 6 and 7 only
- Remote source: `/home/aman/qwen4-exp-opt-bench-20260911/deepgemm-mmid-src`
- Existing DeepGEMM build: `/home/aman/qwen4-exp-opt-bench-20260911/build-deepgemm-mmid`
- MegaMoE build: `/home/aman/qwen4-exp-opt-bench-20260911/build-deepgemm-megamoe`
- Older DeepGEMM tree: `/home/aman/qwen4-exp-opt-bench-20260911/DeepGEMM-v2.1.1.post3`
- SGLang-matching DeepGEMM tree: `/home/aman/qwen4-exp-opt-bench-20260911/DeepGEMM-sgl-release`
  - branch/release: `release/v0.1.7`
  - commit: `40ffa395e0b5c86592d7de1557c46d5125f9d357`
- Staged MegaMoE headers: `/home/aman/qwen4-exp-opt-bench-20260911/DeepGEMM-megamoe`
- SGLang virtualenv: `/home/aman/sglang-qwen38-venv`
- FP8 GGUF shard:
  `/home/aman/qwen4-exp-opt-bench-20260911/fp8-gguf/Qwen3.8-Flash-Next-BF16-FP8-00001-of-00005.gguf`
- Shared models are under `/users/shared` and `/home/shared/models`; recover the
  exact SGLang Hugging Face model path from the saved launch command/log rather
  than guessing it.

SGLang is `0.5.20.dev494+g887c401e1`; its source commit is
`887c401e151e2d1b082d74ef8769a23eef5c465c`. Its installed package is
`sgl-deep-gemm 0.1.7+cu130`, reporting DeepGEMM `0.1.7`.

CUDA 13.3 is not a prerequisite for this experiment. The kernels built and ran
with CUDA 12.8. DeepGEMM warns that CUDA >= 12.9 gives best performance, while
the installed SGLang wheel uses a CUDA 13.0 runtime.

## Trusted performance checkpoint

Known-good non-fused FP8 run:

- `-sm tensor -lm none`
- PP2048: **6669.59 +/- 119.34 tok/s**

This is the checkpoint to reproduce before making another optimization. The
earlier approximately 5k tok/s PP2048 result was real, but the later tensor-split
fix improved it to the value above.

### 2026-09-14: warmed PP2048 Nsight profile

The baseline was reproduced before profiling with the whole-MoE and MegaMoE
paths explicitly disabled:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
GGML_CUDA_DEEPGEMM_MUL_MAT_ID=1 \
GGML_CUDA_DEEPGEMM_MOE_FFN=0 \
GGML_CUDA_DEEPGEMM_MEGA_MOE=0 \
/home/aman/qwen4-exp-opt-bench-20260911/build-deepgemm-mmid/bin/llama-bench \
  -m /home/aman/qwen4-exp-opt-bench-20260911/fp8-gguf/Qwen3.8-Flash-Next-BF16-FP8-00001-of-00005.gguf \
  -ngl 99 -sm tensor -ts 1/1 \
  -ot 'per_layer_token_embd\.weight=CPU' \
  -lm none -p 2048 -n 0 -b 2048 -ub 2048 -t 32 -r 5 -o jsonl
```

Result: **6674.29 +/- 98.27 tok/s**, with samples 6500.49, 6706.55,
6700.86, 6725.86, and 6737.66 tok/s. This independently reproduces the
previous 6669.59 tok/s checkpoint.

The steady profile uses the same command with `nsys profile`, CUDA graph node
tracing, and two measured repetitions. Nsight overhead reduced its samples to
6388.51 and 6579.27 tok/s. The analysis below isolates only the second graph
replay, after all 194 graph objects had been created.

Artifacts:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-mmid-tensor-pp2048-steady-20260914.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-mmid-tensor-pp2048-steady-20260914.sqlite`
- The earlier cold-graph capture is
  `fp8-mmid-tensor-pp2048-baseline-20260914.nsys-rep` in the same directory.

The warmed replay's GPU window is 277.154 ms. GPU 0 has 273.376 ms of summed
kernel work and 3.411 ms of D2D copies; GPU 1 has 273.259 ms of kernel work and
3.418 ms of D2D copies. After merging overlapping activity, both devices are
active for about **99.2%** of the window. There are no H2D or peer copies in the
warmed replay. Each GPU performs 272 D2D copies totaling 2.748 GB.

Kernel time is nearly symmetric across the two GPUs. The following percentages
are sums across both GPUs; the per-GPU percentages are effectively identical:

| Category | Time across both GPUs | Kernel-time share |
| --- | ---: | ---: |
| Elementwise | 170.332 ms | 31.16% |
| Recurrent core (`gated_delta_net`, SSM convolution) | 97.348 ms | 17.81% |
| Type conversions | 58.452 ms | 10.69% |
| Other dense GEMMs | 43.775 ms | 8.01% |
| MoE DeepGEMM FP8 GEMMs | 36.829 ms | 6.74% |
| Normalization | 31.283 ms | 5.72% |
| Top-k/sort, shared by MoE and attention indexer | 23.511 ms | 4.30% |
| Layout/gather/copy kernels | 19.749 ms | 3.61% |
| MoE activation pack/quantize | 19.425 ms | 3.55% |
| MoE output scatter | 17.708 ms | 3.24% |
| NCCL all-reduce | 17.555 ms | 3.21% |
| Attention/RoPE | 7.980 ms | 1.46% |
| Other | 2.688 ms | 0.49% |

The largest individual groups are `gated_delta_net` (93.925 ms across both
GPUs), elementwise multiply (66.149 ms), f32-to-bf16 conversion (54.495 ms),
elementwise add (41.576 ms), and gated sigmoid (22.404 ms). The four local FP8
DeepGEMM shapes together consume 36.829 ms. Pack/quantize and scatter add another
37.133 ms.

Conclusions from this trace:

- This PP2048 path is GPU-compute/memory-kernel bound, not CPU-launch bound.
  Nsight records 194 `cudaGraphLaunch` calls in the replay, but the devices stay
  busy. Most `cudaStreamSynchronize` API time is the CPU waiting for GPU work.
- NCCL is not the primary PP2048 problem. It costs about 8-9 ms per GPU, or 3.2%
  of kernel time. Even free all-reduce would improve throughput by only about
  3.3%.
- DeepGEMM plus its activation packing and output scattering is 13.5% of kernel
  time. Making that entire sequence free has a theoretical ceiling of about
  1.16x, roughly 7.7k tok/s from the unprofiled baseline. Including every
  top-k/sort kernel still cannot explain a multi-x gap to SGLang.
- Gate and up are already fused: the trace shows the expected N=512 and N=768
  DeepGEMM shapes, which are `2 * I` for the I256 and I384 tensor-parallel
  shards.
- The largest opportunity is the surrounding elementwise, conversion,
  normalization, and recurrent pipeline. Elementwise plus conversions alone is
  41.9% of kernel time. The next matched SGLang profile should determine which
  of these operations it fuses into its recurrent/DeltaNet kernels.

This supersedes graph setup and NCCL as the first PP2048 optimization targets.
The next comparison should focus on SGLang's fused recurrent block and on why
llama.cpp emits thousands of standalone elementwise/conversion kernels.

Do not use the experimental whole-`GGML_MOE_FFN` path as the baseline. Its real
end-to-end results were approximately:

- PP128: about 4.18 tok/s
- PP1: about 16 tok/s

That path is functionally selected but catastrophically slow.

## SGLang observations and comparison caveat

`/tmp/sg5.log` contains a warmed non-eager/graph SGLang run:

- Reported decode throughput settles around 131-134 tok/s.
- The log advances `#full token` by 64 and reports `mamba num: 4`, so this looks
  like batched decode, not a single-stream TG result.
- Stable 8192-token prefill chunks are around 25k-31k tok/s.
- The cold first prefill is only about 346 tok/s; small tail chunks produce very
  high and misleading instantaneous rates.
- The input appears to be roughly 100k tokens. We still need an exact comparable
  PP2048 SGLang run.

Before declaring a performance gap, rerun both engines with identical workload
semantics. In particular, distinguish aggregate batched decode throughput from
single-request token generation.

## Implemented and verified

### Sparse attention

Commit `9db850853` enables sparse flash attention for Qwen4Exp. The upstream
mechanism is activated by setting `set_n_kv_max`. This is present on the branch,
but it has not been established as the main current bottleneck.

### CUDA MoE weighted expert reduction

Commit `4e1f24d24` cherry-picks upstream #25952. It structurally matches the
MoE combine tail and replaces router-weight multiplication, expert views, and
the ordered ADD chain with one `moe_weighted_reduction_f32` CUDA kernel. It
supports scaled and unscaled graphs with k=2 through k=15, including Qwen4Exp's
top-k 10. Unsupported graphs retain the original per-op path.

The fusion requires allocator lifetime dependencies. Its prerequisite was
cherry-picked as `9d77e2b65` from upstream #27301. The scheduler conflict was
resolved by retaining this branch's prefetch/MoE-cache allocation terms and
adding `n_dep_nodes` to the graph-size calculation. The CUDA include conflict
was resolved by retaining both `mmid-back.cuh` and
`moe-weighted-reduction.cuh`.

Verification:

- The local CUDA build completed successfully.
- The SM90 build with DeepGEMM completed successfully on the remote host.
- All 6/6 upstream `MOE_WEIGHTED_REDUCTION` cases passed on physical GPU 6:
  scaled/unscaled, aligned/unaligned, k=2/8/12/15, and the k=16 fallback.
- Nsight trace `/tmp/qwen4-moe-weighted-reduction-test.nsys-rep` contains five
  `moe_weighted_reduction_f32` launches for the supported cases. The k=16 case
  emits the original MUL/ADD kernels, proving both dispatch and fallback.

The first post-pick PP2048 smoke was not stable enough to claim a performance
change: 6050.43, 5758.03, 5876.70, 5970.26, and 6625.13 tok/s, averaging
6056.11 tok/s. The final sample returned near the 6.67k checkpoint, while the
first four were unusually slow. Do not use the average as the new baseline.
Before judging this fusion, take a longer warmed run and confirm that the real
Qwen4Exp graph emits `moe_weighted_reduction_f32`; the targeted operator trace
only proves the synthetic matcher cases.

Despite the upstream commit message mentioning
`GGML_CUDA_MOE_WEIGHTED_REDUCTION=0`, this revision only implements the global
`GGML_CUDA_DISABLE_FUSION` guard. Do not use that global switch for an
apples-to-apples A/B because it disables other CUDA fusions too.

### FP8 GGUF and tensor parallel split

Commit `060b4dcbf` adds the FP8 GGUF/DeepGEMM checkpoint. Commit `e94017aff`
aligns the FP8 tensor split with the MoE graph:

- The meta backend supports tensor parallelism for `GGML_MOE_FFN`.
- Merged gate/up tensors use a segmented split so corresponding gate and up
  shards remain paired.
- The two local intermediate sizes are I=256 on GPU 0 and I=384 on GPU 1.

This fixed the earlier `-sm tensor` behavior and produced the 6669 tok/s PP2048
checkpoint.

### `GGML_MOE_FFN` / hand-built DeepGEMM path

The upstream-style whole MoE operator was copied/prototyped. The CUDA launch was
fixed to use 384 total threads while limiting math to 256 threads. A CPU-reference
matrix passed 9/9 cases for token counts 1, 2, and 128 and intermediate sizes
256, 384, and 640.

The operator selection check was conclusive:

```text
moe_ffn_supported: accept
ggml_cuda_deepgemm_moe_ffn: CUDA execute
```

This appeared for all 48 layers on both devices. Therefore the severe slowdown
is not caused by `supports_op` rejection or CPU fallback; it is inside the
experimental implementation/data path.

### MegaMoE feasibility prototype

Commit `677d4d80d` added `tests/test-deepgemm-mega-moe.py` against DeepGEMM's:

- `mega_moe_pre_dispatch_sm90`
- `fp8_mega_moe`
- `get_symm_buffer_for_sm90_mega_moe`
- `transform_weights_for_mega_moe_sm90`

MegaMoE fuses routing pre-dispatch, L1 gate/up FP8 GEMM, SwiGLU plus FP8
requantization, L2 down GEMM, scatter, and top-k combine. Gate/up FP8 rows are
interleaved in groups of 8; their scales remain unchanged.

Standalone results on GPU 6:

| Shape | Result |
| --- | --- |
| E512, H2560, I256, top-k 10, M1 | correct; symmetric relative diff `1.893039e-04`, max abs `1.5625e-02`; about 0.225 ms |
| E512, H2560, I256, M128 | about 0.497 ms, about 257k token/s |
| E512, H2560, physical I512, M128 | about 0.786 ms, about 163k token/s |
| logical I384 padded to physical I512, M1 | correct; symmetric relative diff `1.078495e-05`, max abs `3.90625e-03`; about 0.250 ms |
| E16, H2560, I256, M1 | about 0.073-0.079 ms standalone |

I384 cannot be passed directly because its activation scale-factor row is 24
bytes and violates the required 16-byte workspace alignment. It must be padded
to physical I512. The E16/M128 package heuristic also generates an invalid
combine-chunk configuration; the initial integration was decode-only.

Commit `6d2008c5d` adds an opt-in C++ bridge for exactly E16/H2560/I256/top-k10/M1:

```bash
CUDA_VISIBLE_DEVICES=6 \
GGML_TEST_DEEPGEMM_MOE_FFN=1 \
GGML_CUDA_DEEPGEMM_MOE_FFN=1 \
GGML_CUDA_DEEPGEMM_MEGA_MOE=1 \
./bin/test-backend-ops test -o MOE_FFN_DEEPGEMM -b CUDA0 \
  -p "n_tokens=1,n_ff=256"
```

It passes the CPU reference. In the standard backend perf harness, repeated
5088 times under CUDA graph:

- Mega bridge: 281.15 us/run, 139.86 GFLOPS
- Existing hand-built fused path: 133.01 us/run, 295.64 GFLOPS

Nsight report `/tmp/ggml-megamoe-one.nsys-rep` attributes one bridge launch to:

| Work | GPU time |
| --- | ---: |
| BF16 -> FP8 weight quantization (2 kernels) | 100.448 us |
| Gate/up interleave | 92.352 us |
| MegaMoE kernel | 32.992 us |
| Predispatch | 2.112 us |
| F32 -> BF16 | 1.280 us |
| BF16 -> F32 | 1.088 us |
| Workspace memset | 15.201 us |

Roughly 80% of GPU work is transient weight preparation. This bridge proves
that a static C++ launch can be correct, but it is not performance-useful and is
not a route toward SGLang speed as currently implemented. Do not spend more time
optimizing the E16 synthetic test except as a correctness guard.

The DeepGEMM CUDA translation unit needs a trailing source-specific
`--std=c++20` for the newer headers. Both the new MegaMoE and old DeepGEMM object
builds were verified. A finite activation clamp of `1e30f` avoids an NVCC host
stub issue that emitted invalid `inf`.

## Weight residency and likely integration point

PLE/expert weights are resident in host memory (pinned when configured), not on
disk during inference. `llama_moe_cache` copies selected expert slices into GPU
cache slots.

In `src/llama-moe-cache.cpp`:

- `get_layer_projections` includes merged GATE_UP, or separate GATE and UP, plus
  DOWN weights. It does not include their scale tensors.
- Complete host layers are detected and GPU banks are created with
  `n_slots + guard` experts; views expose `ne[2] = n_slots`.
- Cache fills call `ggml_backend_tensor_set_async` for selected expert slices
  and remap expert IDs to cache slots.

If profiling says MegaMoE is the right optimization, the probable integration
point is persistent transformation during MoE cache fill/cache representation,
not quantization and interleaving on every operator call. Open questions:

- The I384 tensor-parallel shard needs physical I512 padding for MegaMoE.
- Gate/up cached rows need granularity-8 interleaving; the second down shard also
  needs I512 padding.
- Scale tensors do not need gate/up interleaving, but I384 -> I512 requires a
  padded scale representation.
- The specialization must use the real MoE cache slot count, not the synthetic
  E16 shape. Cache slots depend on the configured budget.
- A full persistent copy of all 512 experts per layer on GPU would defeat the
  current host-offload/cache design and may not fit the intended memory budget.

Measure the actual `n_groups`/slot count from a real run before implementing
this. `GGML_CUDA_DEEPGEMM_DEBUG=1` and the MoE-cache logs are available for that.

## Saved profiling artifacts

SGLang:

- `/tmp/sg5.log`
- `/tmp/sglang-qwen38-fp8-standard-ep.nsys-rep` and `.sqlite`
- `/tmp/sglang-qwen38-fp8-graph-decode.nsys-rep` and `.sqlite`

llama.cpp:

- `/tmp/ggml-fp8-graph-decode.nsys-rep`
- `/tmp/qwen-fp8-tg8-tensor.nsys-rep` and `.sqlite`
- `/tmp/qwen-q4-tg8-tpgraph.nsys-rep` and `.sqlite`
- `/tmp/qwen-q4-tg8-onegpu.nsys-rep`
- `/tmp/qwen-q4-tg8-tensor.nsys-rep`
- `/tmp/ggml-megamoe-one.nsys-rep` and `.sqlite`

Fused-path reports/results:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-layer-pp128-steady.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-pp64-direct/`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-pp64-current/`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-readback-debug/`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-ep-pp8-nographs/`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-fused-ep-pp64-nowarm/`

Useful debug logs:

- `/tmp/qwen4-get-rows-p128.log`
- `/tmp/qwen4-sched-debug.log`
- `/tmp/qwen4-moe-exec.log`
- `/tmp/qwen4-dg-support.log`
- `/tmp/qwen4-moe-build.log`
- `/tmp/qwen4-fp8-sm-tensor-meta-debug.log`
- `/tmp/qwen4-fp8-pp2048-sm-tensor-smoke.log`
- `/tmp/qwen4-fp8-pp2048-ub2048-smoke.log`

`/tmp/qwen-tpgraph-direct.log` and its captured counterpart have zeroed perf
timers under profiling. Their approximately 501 ms versus 544 ms total for 20
tokens is not a valid throughput benchmark. Ignore `/tmp/qwen4-perf-record.log`;
it only records that the attempted perf command was unavailable.

## Prioritized next steps

1. **Pending:** reconstruct an apples-to-apples SGLang target on GPUs 6/7.
   - Run an exact PP2048 single request after warmup.
   - Run decode with the same concurrency/batch semantics and output length as
     the llama.cpp test.
   - Record complete commands, model path, memory settings, graph/eager mode,
     and raw output in a dated results directory.
2. **Complete (2026-09-14):** reproduce and profile the known-good non-fused
   FP8 llama.cpp checkpoint. The exact command, results, and traces are in the
   warmed PP2048 section above.
3. **Partially complete:** calculate the matched per-layer critical-path delta.
   The llama.cpp side is profiled; the comparable SGLang PP2048 capture remains.
   Compare MoE kernels, data movement, collectives, graph gaps, and especially
   recurrent/elementwise fusion.
4. Choose the next implementation only from the measured delta:
   - First inspect whether SGLang fuses the elementwise/conversion work around
     `gated_delta_net`; these categories dominate the llama.cpp trace.
   - Only if resident/transformed MegaMoE weights explain SGLang's advantage,
     transform and pad at MoE cache fill and specialize for the actual slot
     count. Never requantize/interleave weights per token.
   - If SGLang keeps all experts resident while llama.cpp host-caches them, first
     match the memory/residency configuration. That may be the real ceiling.
   - Do not prioritize PP2048 graph-launch or all-reduce work without new
     evidence; the warmed llama.cpp trace shows saturated GPUs and only 3.2%
     NCCL kernel time.
5. After every change, rerun warmed PP2048 and matched TG. Reject changes that
   do not improve end-to-end numbers, even if a synthetic operator benchmark is
   faster.

The immediate next action is a matched warmed SGLang PP2048 trace, followed by
an operator-by-operator comparison with the saved llama.cpp trace. The success
criterion is closing the matched end-to-end gap to SGLang.
