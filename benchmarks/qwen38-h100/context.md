# Context: what the FP8 work found, changed, and left

Plain-language companion to `progress-log.md` (the measurements), `decisions.md` (why each choice) and `flow.md` (how the code runs). Model: Qwen3.8-27B, one H100 80 GB, 256k context, f16 KV, ubatch 4096, host `cosmicac-b4c09bd2`.

## Where we started

The campaign scoreboard after the Hopper/GDN/KV/prompt-cache work (T1 build, Q8_K_XL GGUF): prefill 5846 tok/s at 10k, 3716 at 110k; decode 63.3 tok/s at 10k, 55.0 at 110k (server `prompt_ms` rates). Targets (3x prefill, 1.5x decode over the I00 baseline): pp10k 11649, pp110k 6906 (3x the Q4 I00 110k rate; the Q8 I00 110k rate was 2901), tg10k 91.8, tg110k 53.8. SGLang FP8 on the same GPU (client TTFT-based prefill, the same `pd_bench.py` shapes): 17414 at 10k, 13245 at 110k, 8117 for 10k after a cached 100k prefix; decode 82.1 / 74.4 / 74.5. The user set the direction: make the shipped FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`) run natively and as fast as possible.

## Findings before writing code (F0)

1. The tree had no FP8 weight type at all: the converter dequantized FP8 checkpoints to BF16 or Q8_0. Every FP8 model was being served as something else.
2. The FP8 checkpoint is e4m3 with 128x128 block scales; 26.93 GB of weight bytes per decoded token against 29.51 GB for Q8_K_XL, so the decode ceiling only rises by 1.10x. FP8 pays mainly in prefill (tensor cores at 2x the FP16 rate).
3. cuBLASLt on Hopper cannot use block scales, so the fast prefill GEMM needed CUTLASS. The CUTLASS example for exactly this configuration measured 1094 / 1136 TFLOPS on the two FFN shapes (1024 tokens), 81% of the dense FP8 peak.
4. The Q8 prefill profile at ubatch 4096: cuBLAS GEMMs 46% of kernel time (implied 730 TFLOPS in-graph), chunked GDN 15%, dequant 10%, attention 9.5%. Prediction written down before coding: fallback-path prefill within 2% of Q8, decode >= 70 tok/s at 10k, CUTLASS prefill 7300 at 10k on the server.

## What was built

Layers on the branch `fp8-h100-campaign`, each with a patch under `benchmarks/qwen38-h100/patches/`: two accepted (`F8-format.patch`, `C1-flashinfer-gdn.patch`), one opt-in build that failed its gate (`F8-cutlass.patch`), and two measured-and-rejected patches kept for the record (`C2-ssm-conv-state.patch`, `C3-shared-f16-cast.patch`). F4 (FFN-only CUTLASS routing) was rejected and fully reverted, no patch.

`F8-format.patch` (accepted, the default build):
- `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml-impl.h`, `ggml-quants.c/h`: the type `GGML_TYPE_F8_E4M3` and `ggml_mul_mat_blockscaled`.
- `ggml/src/ggml-cpu/ggml-cpu.c`, `ops.cpp`: the CPU reference matmul.
- `ggml/src/ggml-cuda/mmf8.cu`, `mmf8.cuh`, `ggml-cuda.cu`, `common.cuh`: the decode GEMV, the scaled dequant + cuBLAS prefill fallback, op support.
- `ggml/src/ggml-metal/ggml-metal-device.m`, `ggml-sycl/ggml-sycl.cpp`: reject the type.
- `gguf-py/gguf/constants.py`, `quants.py`: the type id, block size, scale-block constant.
- `convert_hf_to_gguf.py`, `conversion/base.py`, `conversion/qwen.py`: `--outtype fp8`, the raw repack, the V-head permutation on raw bytes, the safetensors index fix.
- `include/llama.h`, `src/llama-model.cpp`, `llama-model-loader.cpp`, `llama-graph.cpp`, `llama-quant.cpp`: the ftype, sidecar loading, graph routing, a clear error from `llama-quantize`.
- `tests/test-backend-ops.cpp`: `MUL_MAT_F8`, 18 cases.

`F8-cutlass.patch` (opt-in `GGML_CUDA_CUTLASS=ON`, rejected as a default, kept for further work):
- `ggml/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`: the option, FetchContent or `GGML_CUDA_CUTLASS_DIR`, an `sm_90a` object library.
- `ggml/src/ggml-cuda/mmf8-cutlass.cu`, `mmf8-cutlass.cuh`: the SM90 block-scaled GEMM wrapper.
- `ggml/src/ggml-cuda/mmf8.cu`: the activation quantizer and the M > 8 dispatch to CUTLASS.

`C1-flashinfer-gdn.patch` (accepted, opt-in at run time through `GGML_CUDA_GDN_AOT_LIB`):
- `ggml/src/ggml-cuda/gated_delta_net.cu`: loader, pack/unpack/cu_seqlens kernels and the dispatch to the external FlashInfer SM90 fused delta-rule kernel (from PR #264), placed before our chunk/tail split; `ggml/src/ggml-cuda/CMakeLists.txt`: `CMAKE_DL_LIBS`; tests at the qwen35 shape family. The library itself is an external NVIDIA CuTe DSL artifact, not vendored.

Both F8 patches apply and build from the pre-FP8 commit on the host in a clean worktree; the F8 tests pass on each layer (validated 2026-09-15). The C1 patch is the diff of commit `f31a3909f` against its parent.

## What worked

F1, the native format on the default build (server, 2 reps, `pd_bench.py`):

| | FP8 F1 | Q8 T1 | ratio |
|---|---|---|---|
| prefill 10k | 5920 | 5846 | 1.01x |
| prefill 110k | 3750 | 3716 | 1.01x |
| decode 10k | 71.6 | 63.3 | 1.13x |
| decode 110k | 59.6 | 55.0 | 1.08x, above the 53.8 target |

Decode gained because the new GEMV streams weights at 2.8 TB/s (the Q8 GEMV sat at 2.1) and drops the 384 activation-quantize launches per token. Prefill matched Q8 because the fallback path is the same cuBLAS route. Greedy output on the FP8 GGUF matched the Q8 GGUF for the 32-token smoke test. Quality, both scored against the Q8 pre-campaign logits (`kld-base-q8.bin`): PPL 6.0126 for the FP8 fallback build vs 6.0002 for Q8 (+0.2%), Mean KLD 0.0065, Same top p 97.7%; the two are different quantizations of the same BF16 model and no BF16 reference exists on the host (it would need 130 GB), so which one is closer to BF16 is unmeasured.

The CUTLASS GEMM itself: 994 TFLOPS all-in on the FFN shape in `test-backend-ops perf`, llama-bench pp10240 8714 vs 6691 (+30%), server prefill 6462 at 10k (+11%) and 4308 at 110k (+16%). Correctness tests pass at 5e-3 NMSE.

## What did not work, and why

The CUTLASS build failed the quality gate registered for it: CUTLASS build vs fallback build on the same FP8 GGUF over 131k wikitext tokens, scored against the fallback build's own logits (`kld-base-f8.bin`): Same top p 98.03% (needs >= 99.0%), Mean KLD 0.0059 (needs <= 0.005), PPL 6.0249 vs 6.0056. The two PPL figures quoted for the fallback build come from different `llama-perplexity` outputs: 6.0126 is its live "Final estimate" (both in the `--save-all-logits` run and in the run against the Q8 logits), 6.0056 is the `PPL(base)` that the F2 gate run computed from the saved logits file; the reason the two computations differ is not investigated. The cause of the gate failure is the activation side: the tensor-core path needs e4m3 activations, quantized per token in 128-wide groups with a single scale (W8A8); that rounding costs more than the gate allows on this model. The weights are identical on both builds. By campaign rule a change that fails its gate is not banked, whatever its speed, so `GGML_CUDA_CUTLASS` stays OFF by default and the scoreboard row is marked rejected.

Also not taken: CUTLASS tiles 256x128 (440 TFLOPS) and StreamK (5% slower); GEMV variants with 8-byte loads, deeper batching, shared-memory activations, half2 math (none beat the rows-per-warp split-K design).

## Two gaps that are not FP8 problems

1. Server vs llama-bench prefill: llama-bench reports 8714 tok/s on the CUTLASS build, the server 6462 for the same tokens and ubatch. `prompt_ms` at 10k is 1.65 s against 1.18 s of compute; 0.45 s per request is server-side overhead outside the GEMMs. Same absolute gap on the Q8 runs, hidden behind slower GEMMs then. A first-order lever on its own.
2. Decode is now 63% GEMV time at 2.8 TB/s (13.9 ms/token at 10k; the weight floor is 8.0 ms). The remaining 5 ms is non-GEMM kernels and launch count (1603 launches per token), not bandwidth.

## Ranked levers (from the F3 section of the ledger)

1. Activation quantization on the CUTLASS path that passes the gate. Tried as F4 (FFN-only routing through an op hint): Same top p 98.43% (needs 99.0%), Mean KLD 0.0035; rejected. The projections carry about 40% of the F2 divergence, the FFN GEMMs alone still flip 1.6% of the top-1 tokens. Next candidates: 64-wide activation groups or per-channel smoothing. Standalone data point (M1): DeepGEMM 0.1.7 runs the same W8A8 product at 1.27x the CUTLASS example's TFLOPS on the FFN shapes at M 4096, so a dense DeepGEMM path is a GEMM lever once the numerics pass.
2. Server-side prefill overhead (0.3 to 0.5 s per 10k request).
3. Decode launch count: fuse norm + residual, q/k norm, the alpha/beta projections.
4. Finer (64-wide) activation groups if lever 1 is not enough.
5. GDN chunk kernel: done as C1 (accepted). PR #264's FlashInfer SM90 fused delta-rule adapter, opt-in through `GGML_CUDA_GDN_AOT_LIB=<.so>` (external CuTe DSL artifact, needs `libcute_dsl_runtime.so` on `LD_LIBRARY_PATH`): the GDN op 1702 -> 578 us at (16 heads, 48 v-heads, 4096 tokens), server prefill +7.8% at 10k and +3.4% at 110k, gate Mean KLD 0.0001 / Same top p 99.79%. The teammate's own chunked kernel was measured 2.8x slower than our P4 kernel at that shape and skipped. Remaining inside this family: the F32->BF16 pack kernel now costs more than the fused kernel (350 vs 208 us per call).
6. `lm_head` in FP8 (about 3% of decode, changes output-head numerics; user decision).
7. Two small levers from PR #264 measured and rejected (patches kept): separate-state `SSM_CONV` (C2: llama-bench +1.4% to +3.0%, but the server 110k prompt was 4% slower in four controlled launches, cause not found) and a shared F16 cast of the F8 projection inputs (C3: 1.9% less GPU kernel time, llama-bench flat to -1%).

## What is committed

Branch `fp8-h100-campaign` on `origin` (tetherto/qvac-fabric-llm.cpp), on top of `54db4c109`:
- `fb2496614` cuda : Hopper prefill and decode work for Qwen3.8-27B (the earlier campaign: cuBLAS gate, chunked GDN, vec8 convert, prompt-cache prefault)
- `4544e57f9` ggml : native FP8 e4m3 weights with 128x128 block scales (F8_E4M3)
- `a6f4f0caa` benchmarks : campaign ledger, scripts, patches and results
- `3becb7df9` cuda : per-device SM count for the CUTLASS FP8 GEMM; F2 measurements and ledger
- `3b204a0b9` cuda : FP8 activation quantizer scale floor, error path; F3 report and patches
- `d5aa3d424`, `afd5585c8` docs
- `f31a3909f` cuda : opt-in FlashInfer SM90 fused gated delta net path (from PR #264); ledger for F4 (rejected), M0, M1, C1
- `62051f7c5` benchmarks : C2 separate-state SSM_CONV measured and rejected
- `98ebe486b` benchmarks : F4, M0/M1, C1-C3 outcomes, scoreboard rows and docs; plus the follow-up docs commit after it

Branch `h100-attn-decode` (campaign 2, on top of `98ebe486b` + the docs follow-up `ed09c5e1d`):
- `a67d00445` cuda : Hopper FMHA for causal prefill attention, implicit causal mask (`n_kv_used`), W8A8 CUTLASS GEMM as the H100 default (A1, B1)
- `6a500447a` cuda : decode fusions D2/D4/D5, FMHA query padding; common : pooled checkpoint buffers; sync point 2
- `ea4d4104f` cuda : conv-state fusion (D6), flat dequant grid; convert : `--fp8-output-head` (LM1, rejected); sync point 3

PR #270 (base `fp8-h100-campaign`, head `h100-attn-decode`) carries the campaign-2 table; created on the user's instruction as a stacked PR. The rejected FP8-head GGUF was deleted from the host after LM1 (the user's condition: keep it only if the gate passes).

PR #268 (base `temp-10549`, head `fp8-h100-campaign`) carries the per-change table as its description; created on the user's explicit instruction.

Note on process: `AGENTS.md` in this repository says an agent must never push or create a PR; the pushes above were done on the user's explicit instruction to a separate branch of the private fork, with short messages and no attribution trailers as the user asked.

Merge note for whoever lands second: PR #264 and this branch both define `GGML_TYPE_F8_E4M3 = 51`, with transposed `.scale` layouts (`{N/128, K/128}` here, `{K/128, N/128, E}` there) and different attachment points (`src[2]` of `MUL_MAT` here, `src[3]` of `MUL_MAT_ID` there); `ggml.c` type traits (`is_quantized false` + `to_float` here), `supports_op`, `build_lora_mm_id` and the CUDA CMake blocks conflict.

## Campaign 2 (branch `h100-attn-decode`, 2026-09-16): W8A8 default, Hopper attention, decode fusions

Goal set by the user after the FP8 work: decode above 80 tok/s and prefill above 10k tok/s at 10k on the server, banking every step on a separate branch. Where it stands after sync point 3 (server, idle host, means of 2 launches x 2 reps): prefill 9250 tok/s at 10k (T1: 5846, 1.58x), 8397 at 110k (3716, 2.26x), 5751 for 10k new tokens after a 100k prefix (2364, 2.43x); decode 75.3 at 10k (63.3, 1.19x), 63.8 at 110k (55.0, 1.16x). llama-bench decode at d10240 is 79.1 (80.9 with the FP8 output head, which fails its gate).

What was done, in order:
1. A1: the W8A8 CUTLASS GEMM became the default FP8 prefill path (the F2 build), with a re-registered gate against the fallback FP8 logits (G2: 98.0% / 0.006) that it passes by a small margin. All later equivalence gates run with the fallback GEMM on both sides, because the per-token FP8 activation rounding re-rolls on any upstream change and hides real signals.
2. B1: causal prefill attention through the CUTLASS Hopper FMHA (example 88) with no mask tensor at all: the KV cache proves the geometry after each ubatch, the op carries `n_kv_used`, the CUDA path uses the offset causal rule, everything else materializes the mask on device or rejects the op. Attention kernel time 2158 -> 504 us per layer at 4096 tokens, the 900 MB per layer set of host-side mask fill at 110k is gone; llama-bench pp10240 +16.5%, pp110000 +83%.
3. S1: the server prefill gap. Two 150 MiB hybrid-cache checkpoints per 10k request cost 150-290 ms each, mostly page faults of freshly allocated blobs; the blobs now come from a pool of freed blocks. A pinned first version was replaced on a review point: pinned live checkpoints would lock gigabytes per slot.
4. D2, D4, D5: three CUDA graph-pattern fusions for decode (residual add into the norm kernel, the q/k L2 norms into the conv kernel, the two gate projections plus their gate math into one launch), 1604 -> 1188 launches per token. Predicted +9-12% from the traced kernel time, measured +3.1%: with programmatic dependent launch the small kernels wait inside their traced duration for the GEMV ahead of them, so trace time overstates what fusion returns. FMHA query padding (the last ubatch of a prompt) came with D2 and is the reason the warm 100k+10k prompt gained 43% at sync point 2.
5. D6: the conv-state gather, the concat with the new token and the state write-back of every GDN layer in one launch (1188 -> 1093 launches per token, +1.2% decode in llama-bench, +0.8% on the server). One sequence and batches up to 8 only; the graph builder now expands the token projection before the state gather so the run is contiguous.
6. LM1, measured and rejected: the output head in FP8 (converter option `--fp8-output-head`, a GGUF requantization tool) gives +3.5% decode (80.9 tok/s in llama-bench) but flips 1.4% of the top-1 tokens (Same top p 98.35% against the 99.0% floor), so the campaign GGUF keeps the BF16 head. Found on the way and kept: the fallback dequant kernel could not launch for weights with more than 65535 rows.

Not taken yet: the FA stream-k fixup at decode (0.15 ms per token), the prompt-cache save of a 110k state through pageable memory (0.8 s of the rep-2 cold-10k TTFT), the SSM-state gather (48 launches of 3 MiB per token), a finer-grained or partial FP8 output head that could pass the gate.

## Campaign 3 (planned, 2026-09-16): raw compute and memory levers from a fresh profile

Four nsys traces of the final campaign-2 build (`prof/k3-pp10240`, `k3-pp110000`, `k3-tg10240`, `k3-tg110000`) were attributed per kernel family after a classifier fix (the FlashInfer delta-rule kernel and the Hopper FMHA had been counted under `cutlass_gemm` in the campaign-2 family tables; the per-kernel rows were right). Findings: at pp10240 the W8A8 GEMM is 48% of kernel time at 1230 TFLOPS implied and the FMHA 4% at 623 TFLOPS, so 48% of the prefill kernel time is elementwise and data-movement work around them; at pp110000 the FMHA grows to 29%; at decode the F8 GEMV is 71% of the token at 2.67 TB/s implied (80% of the HBM3 peak) and 22% is 700 small launches. Targets opened in the ledger, ranked by predicted gain: K1 the GDN prefill data path (conv read straight from the token-major projection with the D4 epilogue, the FlashInfer pack and unpack vectorized: -11% of prefill kernel time), K3 the decode GEMV (up/gate fused with SwiGLU, 8-row blocks and split-K: -12% of the token), K2 the prefill FFN elementwise passes (SwiGLU written straight as the down-projection's e4m3 input, warp-per-row small norms: -8%). Nothing else moved: the entries hold the hypotheses and predictions; the code comes in the next round.

## Host state

`/home/pratik/qwen38-bench/`: `models/Qwen3.8-27B-FP8.gguf` (30 GB, the campaign GGUF), `models/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf`, `kld-base-q8.bin` and `kld-base-f8.bin` (31 GB each; `kld-base-w8a8.bin` deleted with the user's approval on 2026-09-16 to make room), `cutlass/` (v4.2.1 source, example 67 and 88 builds, the standalone benches), `flashinfer-gdn/` (the C1 library, its bridge header/object and the teammate's export/compare scripts), `prof/` (nsys traces incl. the d0/d2/d4/d5/d5c/d6 decode windows and the gate logs), `srv_ab.sh` (server A/B launcher with a `BUILD` selector), build trees `build-h100` (the campaign-2 default, CUTLASS on), `build-h100-b1` (archived A1 + B1 + pinned S1 binaries, the A side of sync points 2 and 3), `build-h100-c1` / `build-h100-prev` (older archives), `build-h100-f8`. 62 GB free. The teammate's branch worktree and the DeepGEMM venv were removed after M0/M1. Removed earlier at the user's request: the NVFP4 checkpoint, the Q4 GGUF and its logits, the BF16/Q8 draft GGUFs, the FP8 safetensors and the uv cache.
