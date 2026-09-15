# Context: what the FP8 work found, changed, and left

Plain-language companion to `progress-log.md` (the measurements), `decisions.md` (why each choice) and `flow.md` (how the code runs). Model: Qwen3.8-27B, one H100 80 GB, 256k context, f16 KV, ubatch 4096, host `cosmicac-b4c09bd2`.

## Where we started

The campaign scoreboard after the Hopper/GDN/KV/prompt-cache work (T1 build, Q8_K_XL GGUF): prefill 5846 tok/s at 10k, 3716 at 110k; decode 63.3 tok/s at 10k, 55.0 at 110k. Targets: 11649 / - / 91.8 / 53.8. SGLang FP8 on the same GPU: 17414 prefill, 82 decode. The user set the direction: make the shipped FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`) run natively and as fast as possible.

## Findings before writing code (F0)

1. The tree had no FP8 weight type at all: the converter dequantized FP8 checkpoints to BF16 or Q8_0. Every FP8 model was being served as something else.
2. The FP8 checkpoint is e4m3 with 128x128 block scales; 26.93 GB of weight bytes per decoded token against 29.51 GB for Q8_K_XL, so the decode ceiling only rises by 1.10x. FP8 pays mainly in prefill (tensor cores at 2x the FP16 rate).
3. cuBLASLt on Hopper cannot use block scales, so the fast prefill GEMM needed CUTLASS. The CUTLASS example for exactly this configuration measured 1094 / 1136 TFLOPS on the two FFN shapes (1024 tokens), 81% of the dense FP8 peak.
4. The Q8 prefill profile at ubatch 4096: cuBLAS GEMMs 46% of kernel time (implied 730 TFLOPS in-graph), chunked GDN 15%, dequant 10%, attention 9.5%. Prediction written down before coding: fallback-path prefill within 2% of Q8, decode >= 70 tok/s at 10k, CUTLASS prefill 7300 at 10k on the server.

## What was built

Two layers, delivered as two patches on the branch `fp8-h100-campaign` (`benchmarks/qwen38-h100/patches/`):

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

Both patches apply and build from the pre-FP8 commit on the host in a clean worktree; the F8 tests pass on each layer (validated 2026-09-15).

## What worked

F1, the native format on the default build (server, 2 reps, `pd_bench.py`):

| | FP8 F1 | Q8 T1 | ratio |
|---|---|---|---|
| prefill 10k | 5920 | 5846 | 1.01x |
| prefill 110k | 3750 | 3716 | 1.01x |
| decode 10k | 71.6 | 63.3 | 1.13x |
| decode 110k | 59.6 | 55.0 | 1.08x, above the 53.8 target |

Decode gained because the new GEMV streams weights at 2.8 TB/s (the Q8 GEMV sat at 2.1) and drops the 384 activation-quantize launches per token. Prefill matched Q8 because the fallback path is the same cuBLAS route. Greedy output on the FP8 GGUF matched the Q8 GGUF for the 32-token smoke test; PPL 6.0056 vs 6.0002 (Q8), Mean KLD 0.0065 against the Q8 logits (the two are different quantizations of the same BF16 model; no BF16 reference exists on the host, it would need 130 GB).

The CUTLASS GEMM itself: 994 TFLOPS all-in on the FFN shape in `test-backend-ops perf`, llama-bench pp10240 8714 vs 6691 (+30%), server prefill 6462 at 10k (+11%) and 4308 at 110k (+16%). Correctness tests pass at 5e-3 NMSE.

## What did not work, and why

The CUTLASS build failed the quality gate registered for it: CUTLASS build vs fallback build on the same FP8 GGUF over 131k wikitext tokens: Same top p 98.03% (needs >= 99.0%), Mean KLD 0.0059 (needs <= 0.005), PPL 6.0249 vs 6.0056. The cause is the activation side: the tensor-core path needs e4m3 activations, quantized per token in 128-wide groups with a single scale (W8A8); that rounding costs more than the gate allows on this model. The weights are identical on both builds. By campaign rule a change that fails its gate is not banked, whatever its speed, so `GGML_CUDA_CUTLASS` stays OFF by default and the scoreboard row is marked rejected.

Also not taken: CUTLASS tiles 256x128 (440 TFLOPS) and StreamK (5% slower); GEMV variants with 8-byte loads, deeper batching, shared-memory activations, half2 math (none beat the rows-per-warp split-K design).

## Two gaps that are not FP8 problems

1. Server vs llama-bench prefill: llama-bench reports 8714 tok/s on the CUTLASS build, the server 6462 for the same tokens and ubatch. `prompt_ms` at 10k is 1.65 s against 1.18 s of compute; 0.45 s per request is server-side overhead outside the GEMMs. Same absolute gap on the Q8 runs, hidden behind slower GEMMs then. A first-order lever on its own.
2. Decode is now 63% GEMV time at 2.8 TB/s (13.9 ms/token at 10k; the weight floor is 8.0 ms). The remaining 5 ms is non-GEMM kernels and launch count (1603 launches per token), not bandwidth.

## Ranked levers (from the F3 section of the ledger)

1. Activation quantization on the CUTLASS path that passes the gate: route only the FFN GEMMs through CUTLASS and keep the attention / linear-attention projections on the fallback, then re-gate (chosen next).
2. Server-side prefill overhead (0.3 to 0.5 s per 10k request).
3. Decode launch count: fuse norm + residual, q/k norm, the alpha/beta projections.
4. Finer (64-wide) activation groups if lever 1 is not enough.
5. GDN chunk kernel (21 us/token of prefill, second largest family).
6. `lm_head` in FP8 (about 3% of decode, changes output-head numerics; user decision).

## What is committed

Branch `fp8-h100-campaign` on `origin` (tetherto/qvac-fabric-llm.cpp), five commits on top of `54db4c109`:
- `fb2496614` cuda : Hopper prefill and decode work for Qwen3.8-27B (the earlier campaign: cuBLAS gate, chunked GDN, vec8 convert, prompt-cache prefault)
- `4544e57f9` ggml : native FP8 e4m3 weights with 128x128 block scales (F8_E4M3)
- `a6f4f0caa` benchmarks : campaign ledger, scripts, patches and results
- `3becb7df9` cuda : per-device SM count for the CUTLASS FP8 GEMM; F2 measurements and ledger
- `3b204a0b9` cuda : FP8 activation quantizer scale floor, error path; F3 report and patches

The working tree is clean. Nothing was merged to the default branch; no PR exists. Note on process: `AGENTS.md` in this repository says an agent must never push or create a PR; the pushes above were done on the user's explicit instruction to a separate branch of the private fork. Later commits and pushes follow the user's rule: short messages, no attribution trailers.

## Host state

`/home/pratik/qwen38-bench/`: `models/Qwen3.8-27B-FP8.gguf` (30 GB), `models/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf`, `kld-base-q8.bin` and `kld-base-f8.bin` (31 GB each), `cutlass/` (v4.2.1 source, example 67 build, the standalone benches), `prof/` (nsys traces for f8-pp10k, f8-tg10240, f8-tg110000), two build trees `build-h100` (default) and `build-h100-f8` (CUTLASS on). 71 GB free. Removed earlier at the user's request: the NVFP4 checkpoint, the Q4 GGUF and its logits, the BF16/Q8 draft GGUFs, the FP8 safetensors and the uv cache.
