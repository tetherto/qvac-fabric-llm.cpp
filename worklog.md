# Qwen4Exp optimization worklog

Last updated: 2026-09-16 (Asia/Singapore)

## Objective

Get Qwen4Exp FP8 performance close to SGLang on Hopper while keeping the work on
`qwen4-exp-opt`. Use only GPUs 6 and 7 on `aman@cosmicac-b4c09bd2`.

The matched SGLang comparison is now complete for batch-1 decode and PP2048 at
a resident 32K context. Future comparisons must retain the same request shape,
GPU count, model, warmup state, and CUDA-graph mode.

## Git checkpoint

Current branch: `qwen4-exp-opt`

Recent commits, newest first:

```text
642df9f48 cuda: fuse batch-1 DeepGEMM MoE decode
d429fee9f cuda: remove failed fused MoE prototypes
ad6c3e5e8 cuda: add opt-in FlashInfer SM90 GDN path
556468da6 qwen4exp: add HC inject and separate-state SSM paths
9aae78437 qwen4exp: share BF16 prefill activation casts
69098ed5c cuda: add chunked gated delta net prefill path
f1e3ab835 docs: record qwen4exp HC fusion profile
eef7dd05f qwen4exp: add hc ops
0611b6329 use TENSOR_ALLOW_RESHAPE
88bdd21d1 qwen4exp: enable rms_norm + mul fusion
a9c97c109 docs: record MoE reduction fusion checkpoint
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

Current optimized FP8 run:

- `-sm tensor -lm none`
- DeepGEMM `MUL_MAT_ID` enabled; failed whole-MoE and MegaMoE prototypes removed
- Qwen4Exp HC, separate-state SSM convolution, CUDA MoE-reduction, and external
  FlashInfer SM90 GDN paths enabled
- PP2048 post-cleanup warmed mean: **9764.91 tok/s** (last four samples)
- TG128 post-cleanup warmed mean: **62.83 tok/s** (last four samples)

This is the checkpoint to reproduce before making another optimization. The
first repetition includes model/kernel cold start and is excluded from the
warmed figures. Historical checkpoints below remain useful for attributing
individual optimizations.

### 2026-09-14: warmed PP2048 Nsight profile

The pre-HC-fusion baseline was reproduced before profiling with the whole-MoE
and MegaMoE paths explicitly disabled:

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
change: 6050.43, 5758.03, 5876.70, 5970.26, and 6625.13 tok/s. A later warmed
run after the HC PRs supersedes this result; see the next section. The real
Qwen4Exp graph is now confirmed to emit `moe_weighted_reduction_f32`.

Despite the upstream commit message mentioning
`GGML_CUDA_MOE_WEIGHTED_REDUCTION=0`, this revision only implements the global
`GGML_CUDA_DISABLE_FUSION` guard. Do not use that global switch for an
apples-to-apples A/B because it disables other CUDA fusions too.

### Qwen4Exp HC graph fusions

PR #28896 was cherry-picked as:

- `88bdd21d1`: arrange Qwen4Exp RMS norm and gamma multiplication for backend
  fusion.
- `0611b6329`: load HC gamma tensors as `[n_embd, hc]` with
  `TENSOR_ALLOW_RESHAPE`, removing graph reshapes.

PR #28901 was cherry-picked as `eef7dd05f`. It maps the explicit Qwen4Exp
sigmoid/multiply/stream-reduction chain to gated `DSV4_HC_PRE`, and maps its
repeat/multiply/add residual chain to identity-combine `DSV4_HC_POST`.

The PR #28901 cherry-pick conflicted with newer CPU and Vulkan HC code already
on this branch. The resolution retained the vectorized CPU HC-post
implementation and added its identity-combine behavior. The existing Vulkan
support checks now reject gated-pre and null-combine post graphs, which its
current shaders do not implement.

Verification:

- Local CUDA build completed.
- All 12 focused CPU HC tests passed.
- The SM90/DeepGEMM build completed on the remote host.
- On physical GPU 6, all 18 focused CUDA cases passed: six HC-pre, six HC-post,
  and six MoE weighted-reduction cases.

The 15-repetition FP8 PP2048 run used the same command as the earlier baseline
with `-sm tensor -lm none`, `-b 2048 -ub 2048`, and only GPUs 6 and 7.
It produced **7687.85 +/- 73.05 tok/s**:

```text
7451.16, 7630.02, 7682.45, 7694.23, 7692.94,
7665.82, 7664.95, 7724.96, 7735.18, 7737.62,
7726.11, 7739.74, 7720.40, 7723.74, 7728.38
```

The last eight samples average **7729.52 tok/s**. The full mean is 15.19%
above the reproduced pre-HC-fusion 6674.29 tok/s checkpoint.

The real-model graph dispatch was verified with CUDA graph node tracing:

- Report:
  `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-hc-moe-fusions-pp2048-20260914.nsys-rep`
- SQLite:
  `/home/aman/qwen4-exp-opt-bench-20260911/results/fp8-hc-moe-fusions-pp2048-20260914.sqlite`
- The complete two-repetition capture contains 576 gated
  `dsv4_hc_pre_f32<true>` launches, 576 identity
  `dsv4_hc_post_f32<false>` launches, and 72
  `moe_weighted_reduction_f32` launches.
- `rms_norm_f32<..., true, false>` is present, confirming RMS norm plus
  multiplication is using the fused CUDA kernel.

Across the complete graph warmup plus two profiled repetitions, summed CUDA
kernel time is 1430.275 ms across both GPUs. The largest groups are:

| Kernel group | Time | Share |
| --- | ---: | ---: |
| Gated DeltaNet core | 282.051 ms | 19.72% |
| F32 to BF16 conversion | 164.315 ms | 11.49% |
| Four FP8 DeepGEMM shapes | 111.267 ms | 7.78% |
| Identity HC-post | 81.857 ms | 5.72% |
| NCCL all-reduce | 82.336 ms | 5.76% |
| Elementwise multiply | 63.713 ms | 4.45% |
| MoE activation pack/quantize | 58.237 ms | 4.07% |
| MoE output scatter | 53.189 ms | 3.72% |
| Gated HC-pre | 41.076 ms | 2.87% |
| MoE weighted reduction | 6.398 ms | 0.45% |

The remaining largest target is still the recurrent core and surrounding type
conversion work. The weighted MoE reduction is now small; further MoE gains
need to reduce DeepGEMM packing/scattering or fuse a larger region.

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

## Chunked GDN and shared BF16 activation casts (2026-09-15)

The warmed two-H100 FP8 PP2048 baseline was `7754.61 tok/s` (last four of
five runs). Nsight showed repeated F32 -> BF16 activation staging before BF16
cuBLAS GEMMs. Normalized to one captured execution, these conversions used
1550 launches and 54.772 ms summed across both GPUs. The main redundant
fan-outs were:

- the `[10240, 2048]` normalized HC activation, consumed by both the down and
  injection projections;
- the `[2560, 2048]` recurrent-layer input, consumed by QKV, gate, beta, and
  alpha projections.

`src/models/qwen4exp.cpp` now materializes one BF16 graph tensor for each
fan-out and shares it between those projections. The change is guarded on BF16
weights and no active LoRA, so quantized weights, other model variants, and
adapter behavior retain their previous paths.

Using the generic contiguous `GGML_OP_CPY` cast initially regressed warmed
PP2048 to `7690.72 tok/s`: its four-dimensional scalar copy kernel was about
2.3x slower than the flat converter used by cuBLAS staging. The CUDA copy
dispatcher now sends contiguous F32 -> BF16 casts through that lower-overhead
flat converter and keeps the existing copy kernel for non-contiguous tensors.
Focused contiguous and non-contiguous CPY tests both pass.

With the fast explicit casts and recurrent GDN (`GGML_CUDA_GDN_CHUNKED=0`):

```text
samples: 5217.24, 8056.31, 8042.09, 8079.89, 8085.15 tok/s
warmed last-four mean: 8065.86 tok/s
gain over baseline: +4.01%
```

The first sample includes new CUDA graph capture and is excluded from the
steady-state comparison. A confirming warmed Nsight sample measured
`8092.4 tok/s`. Conversion work fell to 1526 launches and 41.157 ms per
captured execution, a 24.9% time reduction from the normalized baseline. The
old `cpy_scalar_contiguous<float, bf16>` kernel is absent from the winning
trace.

An opt-in FLA-style chunked GDN prefill specialization was also added for the
Qwen4Exp scalar-gate shape (`K=1`, state size 128, at least 32 tokens) on
Ampere-or-newer NVIDIA GPUs. It uses 32-token chunks, a block triangular
inverse, WMMA for the lower-left solve, precomputed W/U, a register-resident
FP32 state, and fused state/output production. Other shapes use the existing
recurrent kernel. Enable it with:

```bash
GGML_CUDA_GDN_CHUNKED=2
```

Four focused GPU-6 correctness cases pass against the CPU reference: T64,
T65, T256, and T127 with two sequences and repeated values. For the standalone
H32/S128/T1024 shape, the recurrent kernel took 681.71 us and the chunked path
took 661.20 us, about 3% faster.

The chunked path is additive with the shared casts in the full model:

```text
samples: 7999.32, 8207.61, 8196.82, 8221.12, 8249.96 tok/s
warmed last-four mean: 8218.88 tok/s
gain over casts alone: +1.90%
gain over original baseline: +5.99%
```

Exact end-to-end configuration: GPUs 6/7, FP8 GGUF, tensor split `1/1`,
`-sm tensor`, `-p 2048 -n 0 -b 2048 -ub 2048`, PLE token-embedding weights
on CPU, DeepGEMM `MUL_MAT_ID` enabled, and the experimental fused/MegaMoE
paths disabled.

## HC injection and separate-state SSM convolution (2026-09-15)

Two additional Qwen4Exp prefill experiments are now available behind opt-in
environment flags.

`GGML_CUDA_HC_INJECT=1` folds the four-output injection projection into
`GGML_OP_DSV4_HC_PRE`. The extended op returns contiguous mixed-HC and
injection regions from one CUDA launch, with BF16 and F32 injection-weight
support. Two focused Hopper cases pass against the CPU implementation.

This path is correct but neutral at PP2048:

```text
samples: 7740.98, 8029.11, 8123.16, 8132.62,
         8178.19, 8215.24, 8225.53, 8213.89 tok/s
warmed last-four mean: 8208.21 tok/s
chunked-GDN baseline: 8218.88 tok/s
delta: -0.13%
```

The profile explains why the projected 3% did not materialize. The fused
scalar-reduction kernel used 20.419 ms summed across both GPUs. The original
HC-pre kernel plus the small tensor-core injection GEMM used about 21.111 ms.
That is only about 0.35 ms of critical-path saving per PP2048 execution. Keep
this path disabled; a useful successor would need a tensor-core/cooperative
projection rather than scalar dot products.

`GGML_CUDA_SSM_CONV_SEPARATE_STATE=1` keeps recurrent history separate and
teaches `GGML_OP_SSM_CONV` to consume channel-major `[channels, T, sequences]`
activations directly. Qwen4Exp uses this only when the ubatch is long enough;
decode and short-token graphs retain the old concat path. The cache update now
materializes only the last three token columns. CPU, CUDA, tensor-split
metadata, and non-CUDA support checks were updated. Focused short-token,
long-token, and real 3328-channel fused-SiLU cases all pass on GPU 6 against
the CPU reference, and the full `-sm tensor` model graph runs on GPUs 6/7.

The first CUDA prototype removed concat but loaded the channel-major tensor in
an uncoalesced order: convolution grew from 3.430 ms to 9.076 ms summed across
both GPUs and warmed PP2048 was effectively neutral. Mapping one thread to one
channel made each timestep load coalesced. The final profile is:

```text
                                      old path    final path
concat_non_cont (both GPUs)            5.801 ms     0.275 ms
ssm_conv (both GPUs)                   3.430 ms     2.136 ms
remaining cache-tail transpose             --      0.359 ms
concat launches                            74           2
```

The removed concat plus faster convolution saves about 3.4 ms of PP2048
critical-path time, approximately 1.4% at the 249 ms baseline. End-to-end
samples on the shared machine drifted between runs, but the repeatable warmed
range with the final path was `8168-8340 tok/s`; one final run stabilized at
`8339.58 tok/s`. The adjacent flag-off runs stabilized at `8048.35` and
`7934.53 tok/s`, while the earlier clean baseline was `8218.88 tok/s`.
Because the paired throughput is noisy, treat the profile-derived ~1.4% as the
credible gain rather than the larger adjacent-run deltas.

Exact benchmark configuration remained two H100s (physical GPUs 6/7), FP8
GGUF, `-sm tensor -ts 1/1`, PP2048/ubatch2048, CPU-resident PLE embeddings,
DeepGEMM `MUL_MAT_ID`, chunked GDN mode 2, and HC injection disabled.

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

GDN and BF16-cast reports (durable remote copy):

- `/home/aman/qwen4-exp-opt-bench-20260911/results/gdn-bf16-casts-20260915/`
- `qwen4-gdn-chunked-fused-state-out.nsys-rep` and `.sqlite`
- `qwen4-fp8-shared-bf16-casts.nsys-rep` and `.sqlite` (slow generic CPY)
- `qwen4-fp8-shared-fast-bf16-casts.nsys-rep` and `.sqlite` (winning path)

HC injection and separate-state SSM reports/results (durable remote copy):

- `/home/aman/qwen4-exp-opt-bench-20260911/results/hc-ssm-fusions-20260915/`
- `qwen4-hc-inject-fused.nsys-rep` and `.sqlite`
- `ssm-separate-pp2048-profile.nsys-rep` and `.sqlite` (uncoalesced prototype)
- `ssm-separate-coalesced-pp2048-profile.nsys-rep` and `.sqlite` (final kernel)
- raw paired `*-pp2048-r8.jsonl` benchmark outputs

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

## FlashInfer SM90 GDN AOT experiment (2026-09-15)

The FlashInfer Hopper fully fused delta-rule kernel is now callable from the
GGML CUDA GDN operator through an opt-in external-library adapter. Set
`GGML_CUDA_GDN_AOT_LIB=/path/to/libflashinfer_gdn_sm90_aot.so`; without that
variable, unsupported shapes, or a failed library load, the existing GGML
chunked/recurrent paths remain unchanged. The current dispatch is intentionally
narrow: Linux CUDA, SM90, `S=128`, at least 64 tokens, scalar decay, one final
state, and at most 1023 sequences.

Sources evaluated:

- FlashInfer SM90 GDN implementation:
  https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py
- FLA chunked gated delta rule reference:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
- SGLang's GDN benchmark:
  https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_gdn_prefill.py

FlashInfer 0.6.18 generates this kernel with CuTe DSL. Directly calling the
device entry point is impractical because its ABI contains four 128-byte TMA
descriptors. Instead, the CuTe wrapper was AOT-exported as a host-callable
object and linked into a small C ABI bridge. The GGML adapter dynamically loads
that bridge and runs three stages on the current CUDA stream:

1. Pack F32 Q/K/V to BF16, exponentiate the log decay to F32 alpha, and pack
   beta.
2. Launch FlashInfer's fully fused SM90 delta-rule kernel.
3. Convert the BF16 output back to the F32 GGML destination.

FlashInfer groups adjacent value heads for GVA, while the existing GGML GDN
kernel maps value heads with `h_v % H_q`. The adapter therefore expands Q/K to
the value-head count in transient BF16 buffers using GGML's current modulo
mapping and invokes FlashInfer in equal-head mode. This preserves the branch's
existing model behavior rather than introducing a hidden head-order change.

Standalone validation against an independent CPU scalar reference at
`T=64, H=8, H_v=24, S=128` gave output max-abs `7.60e-4`, output NRMSE
`0.00260`, state max-abs `4.36e-4`, and state NRMSE `0.00164`. The bridge output
and final state were bit-identical to FlashInfer's Python API at both
`T=2048, H=8, H_v=24` and multi-sequence `B=2, T=127, H=4, H_v=8`. The GGML
backend cases are within BF16 accuracy, although two randomized cases can land
just above the test's strict `1e-5` NMSE threshold (`1.13e-5` to `1.40e-5`)
because FlashInfer stores its output as BF16 whereas GGML's native path returns
F32. The large mismatch seen before the head-order adapter is gone.

The end-to-end FP8 PP2048 comparison used physical GPUs 6/7, `-sm tensor
-ts 1/1`, ubatch 2048, CPU-resident PLE embeddings, DeepGEMM `MUL_MAT_ID`, the
separate-state SSM path, and six repetitions. Excluding the first cold sample:

```text
path                         warmed samples (tok/s)                     mean
GGML chunked GDN             8227.72 8296.18 8300.69 8322.91 8356.64  8300.03
FlashInfer SM90 GDN AOT      9574.34 9568.00 9633.58 9640.28 9668.54  9616.95
```

This is a `15.9%` warmed end-to-end PP2048 throughput improvement. Nsight
Systems counted 216 calls of each GDN stage (36 GDN modules x two GPUs x three
executions). Mean time per module/device was:

```text
FlashInfer fully fused kernel       97.11 us
F32/BF16 + gate preparation         81.97 us
BF16/F32 output unpack              18.56 us
cu_seqlens setup                      0.89 us
total                               198.53 us
```

The prior GGML chunked path was approximately `1.187 ms` per module/device, so
the adapter removes about `83%` of GDN time. Input packing is now almost as
expensive as the fused kernel itself; the next GDN-specific optimization is to
keep its inputs/outputs in BF16 or fuse the surrounding projection/layout work
so the 100.5 us of conversion overhead disappears.

Durable remote artifacts are in:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/flashinfer-gdn-aot-20260915/`
- `qwen4-gdn-aot-pp2048.nsys-rep`
- `libflashinfer_gdn_sm90_aot.so`, generated object/header, export script,
  C bridge, standalone C++ harness, and Python comparison scripts

The generated CuTe object/runtime has NVIDIA packaging/licensing and Hopper
portability constraints, so it is kept as an external experiment artifact and
is not vendored into the repository. A production version needs an explicit
dependency/build policy before enabling this path by default.

## Decode profile and prototype cleanup (2026-09-15)

The failed `GGML_OP_MOE_FFN` vertical slice and MegaMoE experiment have been
removed from the branch. This includes their graph/backend dispatch, CPU and
CUDA implementations, test registrations, environment flags, and standalone
prototype scripts. The working DeepGEMM `MUL_MAT_ID` path and FlashInfer GDN
adapter remain. Removing the serialized GGML op shifts later op IDs, so the RPC
protocol major was intentionally advanced from 108 to 109 and its op-count
guard updated from 114 to 113.

Post-cleanup validation on physical GPU 6:

```text
DeepGEMM MUL_MAT_ID focused correctness     4/4 passed
standard CUDA GATED_DELTA_NET              47/47 passed
```

Post-cleanup end-to-end results on physical GPUs 6/7 used the FP8 GGUF,
`-sm tensor -ts 1/1`, ubatch 2048, CPU-resident PLE embeddings, DeepGEMM
`MUL_MAT_ID`, the separate-state SSM path, and the FlashInfer GDN AOT adapter:

```text
TG128 samples (tok/s)      57.2654 62.8450 62.8241 62.8419 62.8238
TG128 warmed mean                                           62.8337
PP2048 samples (tok/s)    4651.76 9776.50 9721.49 9749.41 9812.22
PP2048 warmed mean                                         9764.91
```

The first sample is consistently cold. The warmed decode result reproduces the
pre-cleanup 62.9-63.1 tok/s range, and prefill reproduces the approximately
9.6k tok/s FlashInfer-GDN checkpoint.

The current two-GPU tensor-parallel TG profile covers 130 token executions.
The main GPU-time buckets are:

```text
DeepGEMM FP8 expert GEMMs                                  26.0%
dense BF16 GEMV (two largest shapes alone)                 17.6%
NCCL all-reduce                                             6.7%
MoE pack/quantize, scatter, top-k, and weighted reduction    7.5%
recurrent GDN                                               0.6%
SSM convolution                                             0.7%
```

There were 12,222 `cudaGraphLaunch` API calls, approximately 94 per generated
token, with a 53.4 us mean host API duration. That is about 5.0 ms of host-side
launch activity per token in aggregate, although overlap means it is not all
necessarily on the critical path. The result changes the decode priority:
GDN is no longer a meaningful decode target; MoE data movement/GEMMs, dense
GEMV, tensor-parallel collectives, and graph fragmentation are the next areas
to measure and optimize.

Keeping GDN Q/K/V in BF16 is still a useful *prefill* follow-up. Today the
Qwen4Exp graph produces F32 after SSM convolution, SiLU, and Q/K normalization,
and the GDN contract requires F32 inputs and output. A useful implementation
must extend that contract and adjust or fuse the surrounding producers; adding
isolated casts would not save the approximately 100.5 us per module/device of
FlashInfer adapter packing and output conversion.

Durable decode profile artifacts:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-cleanup-20260915/qwen4-fp8-decode-current.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-cleanup-20260915/qwen4-fp8-decode-current.sqlite`

## Matched SGLang comparison at 32K context (2026-09-15)

The comparison uses physical H100 GPUs 6/7 and batch size one. "At 32K"
means that 32,768 tokens and all recurrent/KV state are resident before either
the timed PP2048 extension or the timed decode begins. Prefix construction is
excluded from the reported measurements.

SGLang uses the Hugging Face FP8 checkpoint at
`/home/shared/models/Qwen3.8-Flash-Next-FP8`, TP=2, EP=2, FA3, full decode CUDA
graphs, Triton fused GDN decode, FlashInfer GDN prefill, and the default pinned
CPU PLE offload. Custom all-reduce and FlashInfer all-reduce fusion were
disabled. CUDA 12.8 was selected explicitly because `/usr/local/cuda` points
at an incomplete CUDA 13.0 toolkit on this host. A small wrapper around the
stock SGLang one-batch benchmark constructs the prefix in four 8192-token
extensions while preserving KV and recurrent state.

The llama.cpp run uses the FP8 GGUF, `-sm tensor -ts 1/1`, ubatch 2048,
CPU-resident PLE embeddings, DeepGEMM `MUL_MAT_ID`, the separate-state SSM
convolution, and the external FlashInfer GDN adapter. Its essential benchmark
arguments are:

```text
-d 32768 -p 2048 -n 128 -b 2048 -ub 2048 -t 32 -r 5
-ngl 99 -sm tensor -ts 1/1 -lm none
-ot 'per_layer_token_embd\.weight=CPU'
```

Environment:

```text
CUDA_VISIBLE_DEVICES=6,7
GGML_CUDA_DEEPGEMM_MUL_MAT_ID=1
GGML_CUDA_GDN_CHUNKED=2
GGML_CUDA_SSM_CONV_SEPARATE_STATE=1
GGML_CUDA_GDN_AOT_LIB=/home/aman/qwen4-exp-opt-bench-20260911/results/flashinfer-gdn-aot-20260915/libflashinfer_gdn_sm90_aot.so
```

Raw matched results:

| Workload | llama.cpp FP8 | SGLang FP8 | SGLang advantage |
| --- | ---: | ---: | ---: |
| PP2048 after 32K | 5698.81 tok/s, 359.374 ms | 10906.31 tok/s, 187.781 ms | 1.914x |
| TG128 after 32K | 49.5286 tok/s, 20.190 ms/token | 101.4722 tok/s, 9.855 ms/token | 2.049x |

The llama.cpp figures are the mean of the final four repetitions after its
first cold repetition. The SGLang PP number is the median of five timed
extensions; the decode number is the median of 640 timed decode steps across
five repetitions. Relative to llama.cpp's depth-zero checkpoint, the 32K
context reduces PP2048 throughput by **41.6%** and decode by **21.2%**.

### PP2048 profile

The dominant long-context difference is sparse full attention. On each GPU,
llama.cpp spends about 88 ms under tracing in the sparse-index/sort/attention
stack: five CUB segmented-sort kernels per full-attention layer, additional
argsort/index setup, and a separate `flash_attn_ext_f16` call. Segmented sort
alone is 19.4% of its aggregate GPU kernel time. SGLang uses one
`fast_topk_kernel` and one `_sparse_gqa_chunk_prefill` per full-attention layer;
its corresponding large kernels take about 10.4 ms per GPU under tracing.
Absolute Nsight durations include tracing overhead, but the approximately 8x
operator delta and launch structure are unambiguous.

The next visible PP delta is the MoE path. llama.cpp's DeepGEMM gate-up/down,
activation pack/quantize, and scatter consume about 37.8 ms per GPU under
tracing before smaller top-k and reduction kernels. SGLang's fused MoE,
activation/quantization, and reduction sequence is roughly 25-27 ms per GPU.
llama.cpp also has a prominent F32-to-BF16 conversion bucket (5.8%); no
equivalent conversion bucket is prominent in SGLang.

SGLang's aggregate NCCL percentage is misleading. Rank 0 records 82.0 ms over
98 all-reduces (median 705 us), while rank 1 records only 4.8 ms over 90
all-reduces (median 53 us). The long rank-0 NCCL kernels mostly expose TP/EP
rank imbalance and waiting, rather than 82 ms of data-transfer work. llama.cpp
is symmetric at about 8-9 ms of NCCL per GPU. SGLang wins despite this
imbalance, so all-reduce is not the first PP target. The FlashInfer GDN path is
also small on both sides and is no longer a primary PP target.

### Decode profile

The most actionable decode difference is graph granularity. SGLang issues one
whole-model `cudaGraphLaunch` per GPU per generated token. llama.cpp records
24,638 graph launches for 128 generated tokens, or **192.5 launches per token
across both GPUs (96.2 per GPU)**. This matches the many scheduler/backend graph
fragments around the 48-layer MoE path. Nsight inflates CUDA API latency, so its
absolute API time is not an end-to-end saving estimate; the earlier low-overhead
trace put host graph-launch activity near 5 ms/token. The launch-count delta is
nevertheless exact.

Normalized device-side costs under the decode traces are approximately:

| Area | llama.cpp per GPU/token | SGLang per GPU/token | Observation |
| --- | ---: | ---: | --- |
| MoE GEMMs + pack/scatter/top-k/reduce | >=5.5 ms | ~2.6 ms | SGLang's fused MoE path is much cheaper |
| Sparse-attention gather/sort/attention | ~2.4 ms | ~0.5 ms plus small helpers | llama.cpp still sorts/gathers at every full-attention layer |
| Dense BF16 projections | ~3.5 ms | ~2.7-3.0 ms | SGLang uses tuned NVJet kernels |
| NCCL | ~0.85-1.12 ms | ~0.76-0.82 ms | secondary |
| GDN/SSM core | ~0.2 ms | ~0.2 ms | already competitive; not a target |

The SGLang decode report named `tg16` actually contains the final 64 steady
steps (steps 64-127). `start_profile()` returns `None` for CUDA-profiler mode,
so the wrapper originally used a false stop guard and capture continued to the
end. The raw timing is unaffected, and 64 steps give a stable kernel mix. The
durable wrapper has been fixed to track profiling activity separately, so
future bounded captures stop correctly.

### Revised order of attack

1. A/B the existing tensor-parallel whole-graph path on the current FP8 model
   at depth 32K (`GGML_CUDA_TP_GRAPHS=1`, with the known NCCL graph launch
   setting), then fix it until decode approaches one replay per GPU/token.
2. Replace the QSA preselection pipeline with a fused top-k/indexer and sparse
   prefill/decode kernel, eliminating CUB segmented/radix sorts and large
   `get_rows` materialization. This is the largest PP2048 target and a material
   decode target.
3. Build a practical fused MoE execution path around the already fused gate-up
   weights: persistent packing/routing plus gate-up, activation, down, and
   reduction with fewer intermediate launches. The removed whole-`MOE_FFN`
   prototype is not a base for this work.
4. Route the remaining batch-1 BF16 projection shapes to tuned GEMV/GEMM
   kernels or fuse adjacent projections.

Durable raw results, logs, Nsight reports, SQLite exports, and CSV summaries:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/sglang-vs-ggml-d32k-20260915/`
- `sglang-d32768-pp2048-r5.jsonl`
- `sglang-d32768-tg128-r5.jsonl`
- `ggml-d32768-pp2048-tg128-r5.jsonl`
- `sglang-d32768-pp2048.nsys-rep`
- `sglang-d32768-tg16.nsys-rep` (64 steady steps, as noted above)
- `ggml-d32768-pp2048.nsys-rep`
- `ggml-d32768-tg128.nsys-rep`

The temporary llama-bench CUDA-profiler marker was removed after capture and
the normal remote binary was rebuilt. No profiler-only source changes remain
in the branch or remote benchmark tree.

## 2026-09-15: direct sparse-index FlashAttention checkpoint

The QSA-selected token indices can now be passed directly into CUDA
`FLASH_ATTN_EXT`, avoiding the old union mask and KV compaction across all
queries in the ubatch. A focused H100 backend-op test passed for an 8192-token
cache with 512 direct sparse indices. Vulkan and other backends retain the
existing mask path.

At 32K context on GPUs 6 and 7 with tensor split, FP8 weights, PP2048/TG128,
and five repetitions:

| Workload | warmed mean | Previous comparable mean | Delta |
| --- | ---: | ---: | ---: |
| PP2048 | 5725.06 tok/s | 5698.81 tok/s | +0.46% |
| TG128 | 50.0909 tok/s | 49.5286 tok/s | +1.14% |

The previous union represented about 7718 unique cache rows per layer for the
eight-query PP2048 ubatch, versus 2051 selected rows per query. Direct indexing
removes that amplification, but the QSA score and CUB segmented/radix selection
remain. The warmed Nsight capture is:

`/home/aman/qwen4-exp-opt-bench-20260911/results/qsa-direct-20260915/ggml-d32768-pp2048-direct.nsys-rep`

## 2026-09-15: incremental QSA pooled-key cache

Ported the incremental pooled-key-cache design from llama.cpp PR #28699 onto
the branch's newer multimodal-safe QSA path. Each complete
`compress_ratio` position block now stores one F32 mean-pooled, normalized,
and roped indexer key per QSA layer. Steady decode gathers and transforms only
the newly completed block, then writes it with `set_rows`; it no longer
regathers and repools every cached raw indexer key at every full-attention
layer.

The pooled buffers are grouped by the raw indexer cache's backend buffer type,
so tensor-split layers keep their summaries on the owning GPU. The existing
full recompute remains available for multi-stream memories and through
`LLAMA_QSA_NO_POOLED_CACHE=1`. Image-bearing sequences still take the dense
fallback. Per-sequence validity watermarks are reset or clamped by cache clear,
tail removal, sequence copy/keep, position shifts, and full state restores.

Controlled FP8 A/B on H100 GPUs 6 and 7, `-sm tensor -ts 1/1`:

| Workload | pooled ON | pooled OFF | Delta |
| --- | ---: | ---: | ---: |
| PP2048 after 32K, steady plateau | 5863.30 tok/s | 5810.70 tok/s | +0.91% |
| TG128 after 32K, warmed final four | 51.5393 tok/s | 50.3049 tok/s | +2.45% |
| TG128 after 64K, warmed final four | 51.8596 tok/s | 47.6044 tok/s | +8.94% |

The pooled PP graph needed two to three extra one-time warmup repetitions; its
last five samples were 5850.95, 5874.21, 5860.09, 5855.67, and 5875.57 tok/s.
This is finite graph/kernel warmup rather than a steady regression.

Validation:

- full local CUDA build passed
- `test-memory-hybrid-idx` now instantiates the pooled cache and checks
  allocation plus clear/copy watermark behavior; it passes
- `test-llama-archs --arch qwen4exp` passes CUDA and CPU numerical checks
- the Meta-backend split-axis assertion also reproduces with the pooled-cache
  kill switch and belongs to the preceding direct-sparse-index metadata path

## 2026-09-15: batch-1 tensor-parallel whole-graph experiment

The existing opt-in `GGML_CUDA_TP_GRAPHS=1` path was tested with the current
FP8 checkpoint at 64K context on physical H100 GPUs 6 and 7. Verbose depth-zero
tests proved that the native-FP8 DeepGEMM graph passes compatibility checking,
captures successfully on both devices, and replays a stable batch-1 graph.
Nsight at 32K recorded 252 `cudaGraphLaunch` calls for 128 generated tokens,
approximately one launch per GPU per token after the initial direct/capture
steps, versus 24,638 calls (192.5 per token across both GPUs) without the
whole-model graph.

The launch-count reduction did not translate into a material throughput gain:

| 64K TG128 mode | samples (tok/s) | mean |
| --- | --- | ---: |
| TP graph, default concurrent launcher | 51.3589, 50.6579, 48.8907, 51.2357, 51.1583 | 50.6603 |
| TP graph, concurrent launcher plus NCCL mixing disabled | 51.6392, 50.8139, 47.8513, 51.5675, 51.4967 | 50.6737 |
| Adjacent TP graph-off control | 49.5332, 50.0497, 49.9025, 51.4438, 51.5429 | 50.4944 |

The last-two stable means are 51.5321 tok/s for the best graph mode and
51.4934 tok/s for graph-off, only +0.08%. The earlier pooled-cache result of
51.8596 tok/s is consistent with ordinary run-to-run drift. Treat whole-model
TP capture as performance-neutral for this FP8 batch-1 workload.

`NCCL_GRAPH_MIXING_SUPPORT=0` with the current sequential rank launcher hung
at the first collective graph launch. An experimental concurrent-launch
override made that configuration complete, but it was also neutral, so the
override was removed. The normal concurrent launcher remains the only working
configuration tested here.

The 32K graph trace is `/tmp/qwen4-fp8-tpgraph-d32768-20260915.nsys-rep` on the
benchmark host. CUDA tracing shows about 5.18 ms average API time per collective
graph launch, but this is substantially inflated by graph-node tracing. More
importantly, summed GPU kernel time is about 4.55 seconds across the two GPUs
for 128 tokens, or about 17.8 ms per GPU/token. The device-side MoE GEMMs and
pack/scatter path, dense BF16 projections, QSA selection/attention, and 96
all-reduces per GPU/token remain. Removing host graph fragmentation alone
therefore cannot close the SGLang gap.

The temporary llama-bench CUDA-profiler marker and concurrent-launch override
were removed after measurement. No experiment-only source changes remain.

## 2026-09-15: batch-1 hybrid tensor/expert-parallel prototype

An opt-in Qwen4Exp path now keeps the dense, attention, and recurrent graph in
the existing tensor-parallel layout while splitting the 512 routed experts
256/256 across two equally weighted GPUs. Enable it with
`LLAMA_QWEN4EXP_EXPERT_PARALLEL=1`; the normal `-sm tensor -ts 1/1` behavior is
unchanged when the variable is absent.

The Meta backend carries a full-shaped `EXPERT` intermediate whose nonzero
expert slots are disjoint across ranks. This lets gate/up, its views, and
SwiGLU stay local. The down `MUL_MAT_ID` produces a normal partial hidden-state
tensor, so the existing delayed MoE reduction performs one all-reduce after
weighting and summing expert slots rather than between the two projections.
Each simple `MUL_MAT_ID` receives its global expert offset; the DeepGEMM pack
kernel translates global IDs to the local 256-expert table and its scatter
writes zero for nonlocal assignments.

This is an A/B experiment, not an assumed win. The average MoE FLOP count and
collective count are close to ordinary tensor parallelism. Potential gains are
the native full-I=640 DeepGEMM shapes and the smaller per-rank expert table;
the main risk is stochastic 5/5 routing imbalance at batch size one.

Validation completed on physical H100 GPU 6 or GPUs 6/7:

- local CPU build plus `test-llama-archs --arch qwen4exp` passed
- the remote SM90 build of `llama-bench` and `test-backend-ops` passed
- focused DeepGEMM gate/up and down correctness passed 4/4 for one- and
  two-token shapes
- a deterministic 24-token greedy completion matched the ordinary tensor path
  token for token

Warmed TG128 A/B results:

| context | ordinary tensor final four | hybrid TP/EP final four | gain |
| --- | ---: | ---: | ---: |
| depth zero | 58.2616 tok/s | 65.8878 tok/s | +13.09% |
| after 32K | 51.4206 tok/s | 59.6422 tok/s | +15.99% |

The last three 32K samples were especially stable: 51.6313, 51.6359, and
51.6364 tok/s for ordinary tensor parallelism versus 60.1901, 60.4585, and
60.4834 tok/s for hybrid TP/EP. Their means are 51.6345 and 60.3773 tok/s,
respectively, a 16.93% improvement. Stable PP2048 changed only modestly, from
5737.34 to 5803.66 tok/s (+1.16%), as expected for an optimization aimed at
small-token expert GEMMs.

## 2026-09-15: compressed QSA block selection experiment

`LLAMA_QSA_BLOCK_TOPK=1` is an opt-in exact-block selector for the current
ratio-4 QSA layout. It ranks 512 complete blocks instead of sorting 2051 token
cells, expands the winning blocks through a physical-cell map, and appends the
three-cell incomplete tail. Negative fixed-width padding is supported by the
CPU and CUDA `SET_ROWS` implementations. The block map, incomplete tail, bias,
and graph composition have focused CPU/CUDA tests, and the 4K model graph now
runs end to end under the Meta backend.

The first 32K A/B shows that this graph is not yet a decode optimization:

| block selector | ordinary tensor TG128 | hybrid TP/EP TG128 | stable PP2048 |
| --- | ---: | ---: | ---: |
| disabled | 51.6345 tok/s | 60.3773 tok/s | 5737 / 5804 tok/s |
| enabled | 43.5151 tok/s | 49.3197 tok/s | 6284 / 6575 tok/s |

These are last-three stable means. The compressed selector improves prefill
by about 9.5% without EP and 13.3% with EP relative to the corresponding
disabled run, but regresses decode by roughly 16-18%. It therefore remains off by
default. The next step is an Nsight A/B of the selector subgraph; likely targets
are the `GET_ROWS` expansion and `CONCAT`/layout chain rather than the reduced
top-k width itself.

## 2026-09-16: compressed QSA batch-1 tensor-split recovery

Nsight showed that the first compressed-selector graph paid for a generic I32
`GET_ROWS` launch whose 256-thread block copied only one four-I32 row. A compact
`k_get_rows_i32_x4` path now assigns one aligned `int4` row to each thread. The
focused `TOPK_QSA_BLOCKS` CUDA test passes on physical H100 GPU 6. This change
mainly recovers prefill: the gather is only 32.5 ms across 14,100 QSA-layer
invocations in the final profile.

The remaining batch-1 regression came from the separately uploaded
three-cell live-tail input. Across 24 QSA layers it repeatedly split the Meta
graph, producing extra host copies and stream synchronizations. Batch-1 decode
now appends one sentinel row to the existing block map, stores the live tail in
that row with `-1` padding, gives its pooled score a finite forced bias, and
selects 513 blocks. The resulting 2052 cell indices contain the same 512
complete blocks and at most three live tail cells; sparse attention ignores the
one padded index. Prefill retains the separate per-query tail representation.

Adjacent five-repetition FP8 measurements at 32K context on physical H100 GPUs
6/7 used `-sm tensor -ts 1/1`, hybrid expert parallelism, ubatch 2048, CPU PLE,
DeepGEMM `MUL_MAT_ID`, separate-state SSM convolution, and the FlashInfer GDN
adapter:

| mode | PP2048 warmed last two | TG128 warmed last three |
| --- | ---: | ---: |
| QSA token selector control | 5821.73 tok/s | 56.0721 tok/s |
| compressed block selector + sentinel | 7001.81 tok/s | 61.3449 tok/s |

This is +20.27% PP2048 and +9.40% batch-1 decode in the adjacent A/B. Relative
to the pre-sentinel compressed result (49.9258 tok/s), decode recovered 22.87%.
The block selector remains opt-in through `LLAMA_QSA_BLOCK_TOPK=1`.

A complete steady TG128 repetition in the old and new Nsight captures had:

| compressed path | `cudaMemcpyAsync` | `cudaStreamSynchronize` |
| --- | ---: | ---: |
| separate tail | 10405 (81.3/token) | 26660 (208.3/token) |
| sentinel tail | 6656 (52.0/token) | 17413 (136.0/token) |

The sentinel therefore removes 36.0% of the async-copy calls and 34.7% of the
stream synchronizations while leaving the 3072 expected compact gathers (24
QSA layers x 128 tokens). The winning trace and SQLite export are:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/qsa-block-20260915/qsa-block-sentinel-d32768.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/qsa-block-20260915/qsa-block-sentinel-d32768.sqlite`

### Mirrored block-map handoff

The sentinel trace still uploaded its 132112-byte physical block map once per
QSA layer: 3072 H2D copies per TG128. Directly allocating the graph input on
the Meta backend worked but used synchronous `ggml_backend_tensor_set` calls
before replay. The retained form keeps the writable input host-backed, inserts
one `DUP` explicitly assigned to the tensor-split Meta backend, and shares that
mirrored graph-local result across all QSA layers. Layer split keeps the host
map because its QSA layers may live on different backends.

A host/device-DUP/host bracket at 32K context produced these warmed final-three
TG128 means:

| run | TG128 |
| --- | ---: |
| device-map first | 59.7170 tok/s |
| host-map control | 58.2753 tok/s |
| device-map repeat | 59.5385 tok/s |

The two device-map runs average 59.6278 tok/s, +2.32% over the intervening
control. Absolute throughput drifted from the earlier 61.3449 tok/s checkpoint,
so this adjacent bracket—not cross-hour absolute values—is the attribution.

One complete profiled TG128 repetition confirms the intended mechanism:

| path | 132112-byte H2D | `cudaMemcpyAsync` | `cudaStreamSynchronize` |
| --- | ---: | ---: | ---: |
| host map | 3072 | 6656 | 17413 |
| Meta `DUP` map | 258 | 5120 | 10119 |

That is 91.6% fewer full-map uploads, 23.1% fewer async-copy API calls, and
41.9% fewer stream synchronizations, with the same 3072 compact QSA gathers.
The device-map artifacts are:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/qsa-block-20260915/qsa-block-device-map-d32768.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/qsa-block-20260915/qsa-block-device-map-d32768.sqlite`

## 2026-09-16: batch-1 DeepGEMM broadcast packing

The merged gate/up `MUL_MAT_ID` receives one activation row broadcast over the
ten selected experts during batch-1 decode. The generic DeepGEMM pack kernel
previously loaded and FP8-quantized that same 2,560-element row independently
for every selected expert. A batch-1 specialization now quantizes each
128-element K block once and fans the result out to the selected expert rows.
The down projection keeps the generic path because its ten expert activations
are distinct. Set `GGML_CUDA_DEEPGEMM_B1_PACK=0` only to disable the automatic
specialization for A/B testing.

The existing H100 CPU-reference tests passed 4/4 on physical GPU 6, including
one- and two-token gate/up and down shapes. Full 32K-context tensor/EP runs on
physical GPUs 6/7 also completed. End-to-end samples were too affected by host
drift to assign a reliable throughput percentage: candidate warmed final-three
means were 60.69 and 53.39 tok/s, while the adjacent disabled controls were
54.75 and 51.15 tok/s. Do not cite the one fast candidate as a 10% gain.

The exact final-128-token CUDA trace gives a stable device-side attribution:

| pack path, summed over GPUs 6/7 | launches | GPU time |
| --- | ---: | ---: |
| prior generic gate/up + down | 24,576 | 184.597 ms |
| new broadcast gate/up | 12,288 | 42.795 ms |
| unchanged generic down | 12,288 | 50.285 ms |
| new combined total | 24,576 | 93.080 ms |

Combined activation packing falls by 49.6%, saving 91.517 ms of summed GPU
kernel time over 128 tokens. The saving is symmetric across ranks and is about
0.357 ms/token after dividing by two concurrently executing GPUs. Against the
roughly 16.8 ms/token healthy baseline this bounds the expected end-to-end gain
near 2%, rather than the noisy bracket's larger apparent change.

Artifacts are under:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-b1-pack-20260916/`
- `candidate-nsys.nsys-rep` and `candidate-nsys.sqlite`
- `control-1.jsonl`, `candidate.jsonl`, `control-2.jsonl`, and `candidate-2.jsonl`

## 2026-09-16: full batch-1 DeepGEMM MoE fusion

Batch-1 tensor-split decode now recognizes the existing 25-node local MoE
subgraph and executes it as one internal CUDA fusion without introducing a new
public GGML operation or moving the tensor-split all-reduce boundary. The fast
path stages the activation, expert IDs, and route weights; runs the merged
gate/up DeepGEMM; computes SwiGLU while quantizing directly into the packed FP8
layout required by the down projection; runs the down DeepGEMM; and reduces the
packed BF16 expert results directly with the staged router weights. The
original graph remains the fallback. Set `GGML_CUDA_DEEPGEMM_B1_FFN=0` to
disable the full fusion for A/B testing.

The focused `MOE_FFN_DEEPGEMM_B1` H100 test passes against the CPU reference,
and a short real-model trace confirms the fast path on all 48 MoE layers on
both physical GPUs 6/7. Adjacent five-repetition FP8 measurements at 32K
context used `-sm tensor -ts 1/1`, hybrid expert parallelism, QSA block top-k,
ubatch 2048, CPU PLE, and the FlashInfer GDN adapter:

| full batch-1 fusion | TG128 samples | stable last-three mean |
| --- | --- | ---: |
| disabled | 58.6813, 59.5333, 60.7367, 60.6770, 60.6849 | 60.6995 tok/s |
| enabled | 60.2996, 61.6302, 62.7103, 62.7352, 62.7384 | 62.7280 tok/s |

This is a reproducible **+3.34%** batch-1 decode improvement. The warmed final
two PP2048 samples were 6971.17 tok/s disabled and 7115.04 tok/s enabled
(+2.06%), although prefill is not the target of this batch-1-only path.

A matching graph-disabled Nsight pair was used only to expose individual
steady-state kernels. Relative to the disabled run, the fusion removed 6,328
kernel launches from the complete captured workload. The directly attributable
MoE changes removed 24.382 ms and added 9.266 ms, a net 15.116 ms reduction in
summed GPU kernel time. Normalizing the 1,664 per-device-layer fusion calls and
dividing for two concurrently executing GPUs gives about 0.436 ms saved per
token, consistent with the graph-enabled end-to-end result. With graphs
disabled, throughput rose from 8.1268 to 15.1872 tok/s; that larger number is
diagnostic launch-overhead removal and is not the production result.

Artifacts are under:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-b1-pack-20260916/`
- `full-ffn-b1-staged-d32768.nsys-rep` and `.sqlite`
- `full-ffn-b1-staged-nographs-d32768.nsys-rep` and `.sqlite`
- `full-ffn-b1-disabled-nographs-d32768.nsys-rep` and `.sqlite`

## 2026-09-16: Hopper radix QSA block selection

The generic CUDA `TOP_K` path used CUB `DeviceTopK` for every QSA layer. At a
32K resident context, selecting the 513 block IDs needed by the batch-1
sentinel graph expanded into eight kernels: index and offset initialization,
a histogram, four radix onesweep passes, and an exclusive scan. The new Hopper
specialization handles the QSA widths 512 and 513 in one 1024-thread kernel.
It finds the exact F32 threshold through four in-kernel radix passes and emits
all greater keys plus the required number of exact-threshold ties. Rows larger
than 65,536 elements, other top-k widths, and non-Hopper devices retain CUB.
Set `GGML_CUDA_RADIX_TOP_K=0` for the CUB A/B fallback.

Focused physical-H100 validation passed 6/6: distinct and tied one-/two-row
TOP_K cases for both widths, plus the full sentinel and non-sentinel QSA block
graphs. Five synchronous stress repetitions also passed. A focused Nsight
microtrace measured 11.232 us per 8193-to-513 radix kernel versus about
52.4 us per CUB selection, a 4.7x operator speedup.

An adjacent radix/CUB/radix bracket used the same 32K FP8 tensor/EP workload as
the preceding MoE result. Warmed means were:

| QSA selector | PP2048 final-two mean | TG128 final-three mean |
| --- | ---: | ---: |
| radix candidate 1 | 7622.25 tok/s | 65.4140 tok/s |
| CUB control | 7038.62 tok/s | 62.9557 tok/s |
| radix candidate 2 | 7666.27 tok/s | 65.1687 tok/s |

The two radix runs average **7644.26 PP2048 tok/s (+8.60%)** and
**65.2914 TG128 tok/s (+3.71%)** relative to the intervening control.

The graph-disabled 16-token trace gives exact steady-window attribution. Across
24 QSA layers, the selector launch count falls from 3072 to 384. Selector GPU
time falls from 17.1557 ms to 6.7456 ms, saving 10.4101 ms over the trace, or
0.6506 ms/token. Graph-disabled throughput rises from 15.1872 to 19.2312 tok/s;
that larger percentage includes CPU launch overhead and is diagnostic only.

Artifacts:

- `/tmp/qsa-topk-radix.nsys-rep` and `/tmp/qsa-topk-cub.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-b1-pack-20260916/radix-topk-nographs-d32768.nsys-rep`
- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-b1-pack-20260916/radix-topk-nographs-d32768.sqlite`

## 2026-09-16: batch-1 recurrent projection inputs

The explicit F32-to-BF16 activation cast shared by the four recurrent GDN
projection groups is beneficial for prefill, but not for batch-1 decode. The
decode graph now keeps the recurrent input in F32 and uses the CUDA
BF16-weight/F32-vector kernels directly. Multi-token batches retain the shared
BF16 cast, so the existing prefill path is unchanged.

A profiler-bounded 16-token trace at 32K context on physical H100 GPUs 6/7
measured 471.161 ms of summed GPU kernel work for the shared-BF16 control and
462.930 ms for the F32-vector path, a **1.75% reduction**. Qwen4Exp CUDA
validation passed with NMSE 8.38e-08 and a successful serialization roundtrip;
the test subsequently encountered the already-known unrelated Meta-backend
split-axis assertion.

An unprofiled five-repeat tensor/EP comparison confirmed the gain:

| recurrent projection input | TG64 samples | stable last-three mean |
| --- | --- | ---: |
| shared BF16 cast | 59.4030, 59.2673, 63.3876, 63.2930, 63.3823 | 63.3543 tok/s |
| direct F32 vector | 59.8645, 61.1251, 64.7424, 64.6765, 64.7325 | 64.7171 tok/s |

That is a **+2.15%** stable batch-1 decode improvement. Applying the same idea
to the 96 HC projection inputs was rejected: enabling both paths increased the
recurrent-only trace from 462.930 ms to 468.573 ms. HC therefore keeps its
shared BF16 activation cast.

Artifacts are under:

- `/home/aman/qwen4-exp-opt-bench-20260911/results/decode-b1-proj-20260916/`
- `bf16-control.nsys-rep`, `recurrent-only.nsys-rep`, and `f32-candidate.nsys-rep`
