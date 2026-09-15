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
- DeepGEMM `MUL_MAT_ID` enabled; experimental whole-MoE and MegaMoE paths disabled
- Qwen4Exp HC and CUDA MoE-reduction fusions enabled
- PP2048: **7687.85 +/- 73.05 tok/s** over 15 repetitions
- Stable last-eight mean: **7729.52 tok/s**

This is the checkpoint to reproduce before making another optimization. The
prior non-fused checkpoint was 6674.29 +/- 98.27 tok/s. The current full mean
is 15.19% faster.

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
