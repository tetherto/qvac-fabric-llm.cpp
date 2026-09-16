# Design decisions: native FP8 for Qwen3.8-27B on H100

Each entry: the question, the options that were on the table, the option taken, and why. Numbers come from `progress-log.md` (sections F0 to F3).

## 1. Which FP8 checkpoint semantics to support

Options:
- (a) Requantize the weights to per-tensor or per-channel scales so cuBLASLt's FP8 GEMM can run them.
- (b) Keep the checkpoint as shipped: e4m3 weights with 128x128 block scales (`weight_scale_inv`, `W = fp8 * scale_inv`).

Taken: (b). It is exactly what vLLM and SGLang run on this GPU (SGLang FP8 reached 17414 tok/s prefill here), it keeps the numerics of the checkpoint untouched, and (a) changes the weights and needs its own quality gate before any speed work. Cost of (b): cuBLASLt cannot consume block scales on SM90, so the fast GEMM had to come from CUTLASS.

## 2. How to store FP8 in GGUF

Options:
- (a) A new block-quant type with the scale inside the block (like Q8_0: scale + values interleaved).
- (b) A plain 1-byte type `GGML_TYPE_F8_E4M3` (block size 1, no scale inside) plus a separate F32 sidecar tensor `<name>.scale` per weight.

Taken: (b). The tensor-core GEMM (TMA + WGMMA) wants a plain K-contiguous e4m3 matrix; an interleaved block type would need a repack copy of every weight (+24 GB) at load time. The sidecar pattern already existed for NVFP4 (`.scale`, `.input_scale`), so the loader and graph had a precedent. `llama-quantize` cannot produce the type (`from_float_ref` is NULL on purpose); only the converter writes it.

## 3. Sidecar scale layout

Options:
- (a) Qwen's row-major `[N/128][K/128]` (K-block index contiguous).
- (b) Transposed `[K/128][N/128]` (N-block index contiguous), i.e. ggml shape `{N/128, K/128}`.

Taken: (b), decided in F0 before the converter was written: CUTLASS `Sm90BlockwiseScaleConfig` only supports MN-major scale factors on SM90 (`blockwise_scale_layout.hpp:273`), verified with a standalone bench against a double-precision reference (0/512 samples off). The GEMV and the CPU reference index `scale[kb * (N/128) + n/128]`; the converter does one transpose of a 3 MB tensor per model.

## 4. How the scale reaches the matmul

Options:
- (a) Multiply the matmul output by the scale afterwards (the NVFP4 way, `ggml_mul(res, scale)`), which only works for per-tensor scales.
- (b) Pass the scale tensor into the matmul as `src[2]` of `GGML_OP_MUL_MAT` through a new `ggml_mul_mat_blockscaled(a, a_scale, b)`.

Taken: (b). Block scales apply per 128x128 block inside the dot product, so they must be visible to the kernel. `MUL_MAT` never used `src[2]` on CPU or CUDA (checked before the change), so no existing path reads it. The NVFP4 post-multiply is kept for every non-F8 type.

## 5. Decode kernel design (batch 1 to 8)

Options:
- (a) Reuse `mmvq` (needs `ggml_is_quantized`, quantizes activations to q8_1: 384 extra launches per token).
- (b) Reuse `mmvf` (F32/F16/BF16 only; no scale slot).
- (c) A new GEMV: F32 activations, e4m3 weights converted in registers with the hardware `cvt` instruction, fp32 accumulation, the block scale applied once per 512-column trip.

Taken: (c). First version (one warp per row, 16-byte loads) stalled at 1.66 TB/s on both 89 MB FFN weights; a standalone bench showed the pure-streaming ceiling of that access pattern at 2.77 TB/s and the same kernel without activation loads at 2.53, i.e. activation L1 traffic (4 float4 loads per 16 weight bytes) was the limiter. Final geometry: a block of 4 warps owns 4 consecutive rows and splits the k range across its warps, one activation slice per trip reused for the 4 rows, shared-memory reduce: 2.79 / 2.83 TB/s in `test-backend-ops perf` (1.7x). Rejected on the way: 8-byte loads (2.1 TB/s), deeper load batching (no effect), staging activations in shared memory (same bandwidth as L1), half2 FMAs (no gain, lower precision).

## 6. Prefill GEMM

Options:
- (a) Scaled dequant to F16 into a pool buffer, then cuBLAS (F16 in, F32 out): the numerics of today's Q8 path, runs on every CUDA arch.
- (b) CUTLASS SM90 `KernelTmaWarpSpecializedCooperativeFP8Blockwise`: e4m3 weights as stored, activations quantized per token per 128-group to e4m3 (W8A8), fp32 accumulation with block-scale promotion.

Taken: both. (a) is the default path and the accepted build (`build-h100`). (b) is behind the CMake option `GGML_CUDA_CUTLASS` (default OFF), compiled in its own object library for `sm_90a`. F0 measured (b) at 1094 / 1136 TFLOPS on the FFN shapes against 734 for (a) in-graph, so it was worth integrating; F2 measured +30% prefill in llama-bench and +11% / +16% on the server, but the registered quality gate (CUTLASS build vs fallback build of the same GGUF, Same top p >= 99.0% and Mean KLD <= 0.005) failed: 98.03% / 0.0059. The campaign rule is that a change failing its gate is rejected regardless of speed, so (b) stays opt-in and unbanked. Tile choice inside (b): 128x128x128 with cluster 1x1x1 and the persistent scheduler (1094 to 1136 TFLOPS); 256x128 ran at 440, 128x256 does not compile with 128-wide scale groups, StreamK was 5% slower.

## 7. Activation quantization scale (inside option 6b)

Options:
- (a) `scale = amax/448`, `inv = 448/amax` with a branch for `amax == 0`.
- (b) `scale = max(amax/448, FLT_MIN)`, `inv = 1/scale`.

Taken: (b). (a) overflows `inv` for subnormal `amax` and makes `0 * inf = NaN` in the quantized activations; (b) keeps the quantized values and the stored scale consistent and needs no branch. Covered by two new `test-backend-ops` cases with activations of magnitude 1e-37 on the GEMM paths.

## 8. Token count padding for the CUTLASS path

The activation scale tensor is `[K/128][M]` and TMA needs a 16-byte row stride, so M must be a multiple of 4 (CUTLASS `can_implement` rejected M=9). Taken: pad M up to a multiple of 4, zero the padding rows, and copy the M rows of the padded output back only when padding was needed. Alternative (rejecting non-multiple-of-4 ubatches) was not acceptable: the last ubatch of a prompt has an arbitrary length.

## 9. What stays BF16

`lm_head` (2.54 GB), `token_embd`, the linear-attention `in_proj_a/b`, `conv1d`, `A_log`, `dt_bias`, the MTP `eh_proj`: all as shipped in the checkpoint. Moving `lm_head` to FP8 would save about 0.4 ms/token of decode (3%) but changes the output-head numerics; left as a user decision (lever 6 in F3).

## 10. Rejected and parked along the way

- Q8_0 GEMV geometry retune (Q8-D2): parked when the campaign moved to FP8; its per-node table is the design input the F8 GEMV used.
- BF16 reference logits for an absolute quality number: needs about 130 GB of disk; not run. All quality rows are relative to the Q8_K_XL reference logits or to the FP8 fallback logits.
- A relaxed gate for W8A8 prefill: not taken; thresholds are set before results, not after.

## 11. FFN-only CUTLASS routing (F4)

Options:
- (a) Route only the FFN up/gate/down matmuls through the W8A8 GEMM and keep the attention / linear-attention projections on the F16 fallback, selected by an op hint set in `build_ffn` (no tensor-name or shape heuristics in the backend).
- (b) Finer activation groups (64-wide) for every F8 matmul.

Taken: (a) first (the user's choice among the F3 levers), then rejected by the gate: Same top p 98.43% against the 99.0% floor, Mean KLD 0.0035 (F2 with everything routed: 98.03%, 0.0059). The hint mechanism and the tests were reverted; the CUTLASS build stays opt-in and rejected. (b) is the next candidate, together with per-channel smoothing.

## 12. Which PR #264 pieces to take, and on what basis

The user's rule: nothing is skipped on reasoning alone; every candidate is measured at our shape before a decision. Measured on his own branch built on the host (M0): his BF16 wmma chunked GDN kernel 4695 us vs our P4 kernel 1702 us per (16 heads, 48 v-heads, 4096 tokens) op, skipped; his FlashInfer SM90 adapter 578 us, taken as C1. Measured standalone (M1): DeepGEMM 0.1.7 dense FP8 at 1.27x the CUTLASS example's TFLOPS on the FFN shapes, recorded as a future GEMM lever, nothing merged (his DeepGEMM integration is MoE-only with hard-coded qwen4exp shapes and BF16 output). His rms_norm + mul arrangement: already active in our graph. His shared BF16 casts: no standalone effect on our graph (no F32->BF16 CPY), so the analogue was built as C3.

## 13. FlashInfer GDN adapter as an opt-in external library (C1)

Options:
- (a) Vendor the CuTe DSL kernel source or the generated object into the tree.
- (b) Keep it external: `dlopen` a `.so` named by `GGML_CUDA_GDN_AOT_LIB`, fall through to our kernels when unset or when the shape is outside the adapter's conditions.

Taken: (b), as in PR #264. The artifact is an NVIDIA CuTe DSL AOT export (needs `libcute_dsl_runtime.so`), not something the build can produce; keeping it external keeps the default build and the default numerics unchanged (without the variable the trace shows zero FlashInfer launches and the P4 kernel as before). Accepted on the campaign rules: gate Mean KLD 0.0001 / Same top p 99.79% (BF16 q/k/v/out rounding, fp32 state), server prefill +7.8% at 10k and +3.4% at 110k on both reps, decode unchanged. The test tolerance for the external kernel is 5e-5 NMSE (BF16 class; measured 1.0e-5 to 1.3e-5), and the TF32 chunked path got 5e-7 after it was found sitting on the generic 1e-7 line with run-to-run noise.

## 14. Separate-state SSM_CONV (C2) and the shared F16 cast (C3): rejected on the acceptance rules

C2 removes the per-layer concat of conv state and tokens (his op extension `ggml_ssm_conv_ext`, CUDA and CPU only, opt-in by env var because the other backends reject `src[2]`). Tests and a 4-chunk PPL equality passed, llama-bench gained 1.4 to 3.0%, but the server's 110k prompt was 4.0% slower in four alternating launches while the 10k prompt was 1.8% faster; the rule (server not worse on both shapes) rejects it. The cause was not found in the time box (ruled out: run order, CPU fallback on partial ubatches, server checkpoints); the patch is kept.

C3 replaces the per-matmul F32->F16 activation convert inside the F8 fallback with one graph-level cast shared by the projections of a layer (GDN wqkv+gate, attention q/k/v, FFN up+gate: 144 of 400 converts per ubatch). It needed the F8 matmul, the CUDA `supports_op` and the CPU reference to accept F16 activations, and it sent contiguous F32->F16/BF16 copies through the vectorized converter. Tests and equality passed and the trace showed 1.9% less GPU kernel time, but llama-bench was flat to -1%; rejected on the +1% rule, patch kept.

## 15. Server acceptance runs are A/B on one idle GPU

After C2's first server run (measured while the other GPU ran an A/B) disagreed with its controlled repeat, server acceptance numbers are taken with the host otherwise idle, the server relaunched per configuration, and the two configurations alternated (`srv_ab.sh` on the host). The C1 acceptance was measured before this rule; its 10k gain (+9.2% and +6.4% per rep, +7.8% on the mean) is well above the noise seen since (about 1.5% at 10k, 0.2% at 110k).

## 16. W8A8 CUTLASS GEMM as the default FP8 prefill path (A1, campaign 2)

Options:
- (a) Keep the F16 fallback GEMM as the default and the W8A8 GEMM opt-in (the F2/F4 state: the W8A8 build failed the equivalence gate against the fallback build's logits).
- (b) Make the W8A8 GEMM the default (`GGML_CUDA_CUTLASS=ON` in the campaign build) under a user-authorized relaxed gate against the fallback FP8 logits (G2: Same top p >= 98.0%, Mean KLD <= 0.006), with the limits set around F2's measured 98.03% / 0.005892. Whether SGLang/vLLM serve this checkpoint with an identical activation quantizer was not verified (the checkpoint config is no longer on the host), so G2 is an F2-vs-fallback tolerance, not a claim of vendor equivalence.

Taken: (b), the user's decision for campaign 2 (prefill is the larger gap and the fallback GEMM already runs at 76% of the H100 F16 peak, so the W8A8 tensor-core rate is the only way to a 10k prompt above 10k tok/s). G2 measured 98.018% / 0.005949, inside the registered limits by a small margin. Every later equivalence gate (G1) is measured with `GGML_CUDA_DISABLE_MMF8_CUTLASS=1` on both sides, because the W8A8 rounding re-rolls on any upstream change of about 1e-4 and a W8A8-vs-W8A8 KLD near 0.01 is not a signal (decision recorded in the B1 ledger entry).

## 17. Causal attention without a mask tensor (B1)

Options:
- (a) Keep the mask tensor and give the CUTLASS FMHA the mask as an extra input (the example has no mask input; it would need a custom fusion and the mask fill and upload would stay).
- (b) Describe the common case (one sequence, cells [0, n_kv_used) in position order, the ubatch at the tail) with one integer on the op (`ggml_flash_attn_ext_set_kv_used`), let the KV cache prove the geometry after `apply_ubatch`, and let backends that cannot run it materialize the mask themselves.

Taken: (b). The mask for a 4096-token ubatch at 110k is 900 MB of f16 per layer set filled on the host every ubatch; with (b) it is gone on the CUDA path (the FMHA uses the offset causal rule, the ggml kernels get a device-side fill when the FMHA does not apply) and the other backends reject the op in `supports_op` so the scheduler never hands it to them. The CPU reference implements the same rule so `test-backend-ops` compares against it. Batches below 256 tokens keep the mask (the cell scan is not worth a launch there). M-RoPE stays admitted because its only special case fails the contiguity check.

## 18. Checkpoint blobs: pooled pageable memory, not pinned (S1)

Options:
- (a) Pinned host buffers (`ggml_backend_dev_host_buffer_type`) for every checkpoint blob, pooled after free: fastest device copies (measured mid-prompt checkpoint 148 -> 63 ms).
- (b) A bounded pinned staging buffer plus a memcpy into pageable blobs.
- (c) Pageable blobs from a pool of freed blocks (1 GiB idle cap), so the pages are faulted once and reused.

Taken: (c). (a) keeps every live checkpoint page-locked: 32 checkpoints per slot at 150 MiB each is 4.7 GiB of pinned memory per slot, and the prompt cache copies them, so a server could lock 10+ GiB; a pool cap does not bound live allocations. (b) adds a 150 MiB memcpy per checkpoint. The page faults were the dominant cost (about 57 ms of the 148, 38k faults), the pageable copy itself is 20 ms slower than pinned; (c) removes the faults with no lock footprint and keeps the code to one small class (`common_state_buffer`). Measured under host load only (72-96 ms mid-prompt); the sync-point-2 server A/B carries the accepted number.

## 19. Decode launch fusions are graph-pattern matches with explicit alias rules (D2, D4, D5)

Options:
- (a) Change the graph builders (`build_norm`, the delta-net builder) to emit new fused ops.
- (b) Match the existing op runs in the CUDA backend and replace them with one launch, as the existing RMS_NORM+MUL and SSM_CONV+SILU fusions do.

Taken: (b): no new ops, no change for other backends, the CPU reference stays the unfused graph. Three patterns: residual `ADD -> RMS_NORM -> MUL` (the add stays an output, exact-alias rules listed in `ggml_cuda_should_fuse_add_rms_norm_mul`: in-place add and dst on a dead add input are allowed because each element is touched by one thread, dst on the add result is not, partial overlaps never); `ssm_conv -> silu -> per-head L2_NORM slices (+ scale)` scanned as a run of view/norm/scale nodes because the fused and the ggml-op delta-net builders emit different view chains; the gate projections `alpha/beta mul_mat -> add/softplus/mul, sigmoid` at batch <= 8 (above that the GEMM path keeps the tensor cores). Every match runs `ggml_can_fuse_subgraph` with the surviving nodes as outputs and `ggml_cuda_check_fusion_memory_ranges`, and each pattern has a whole-graph `test-backend-ops` case that checks all of its outputs plus an nsys count proving the fused kernel ran.

## 20. FMHA query padding

The CUTLASS FMHA needs the query count to be a multiple of 8; the last ubatch of a prompt has an arbitrary length (1841 tokens at 10k). Taken: pad with phantom queries that attend the same number of phantom cells past `n_kv_used` (they exist in the K/V views, which the support check verifies), zero the phantom Q rows, discard their outputs; no host-side change. Alternative (fall back to the mask fill for those ubatches) was the B1 state and cost the mask path on 18% of the tokens of a 10k prompt.

## 21. Output head in FP8: measured and rejected (LM1)

The user chose to try it behind the weight-only gate (Same top p >= 99.0%, Mean KLD <= 0.002 against the fallback FP8 logits). Options for producing it: (a) re-convert from the safetensors with a new converter flag; (b) requantize the BF16 head inside the existing GGUF with the same function. Both were built (`--fp8-output-head` in the converter, `gguf_fp8_output_head.py` for the GGUF, sharing `fp8_block_quantize`), (b) was used because the safetensors were no longer on the host. Result: decode +3.5% (80.9 tok/s at d10240, the campaign's decode target in llama-bench) but Same top p 98.35%: the head's rounding flips 1.4% of the top-1 tokens against 0.24% for the whole block-scaled trunk, so the gate rejects it and the campaign GGUF keeps the BF16 head. The option stays available for models where it passes. A side effect that is kept: the fallback dequant kernel's grid could not address more than 65535 rows (the head has 248320), fixed with a flat grid and covered by two op cases.

## 22. Conv-state fusion limited to one sequence and small batches (D6)

Options:
- (a) A general kernel over (sequence, channel, column) elements, with the three launches replaced for every shape.
- (b) One thread per channel for one sequence at batch <= 8: every thread reads its channel's state and token values into registers before writing the conv input row and the next state row.

Taken: (b). The state cache is both read (the gathered row) and written (the updated row) by the fused op; with one sequence the rows are the same or disjoint per channel and the read-before-write order inside a thread makes the in-place update safe, while with several sequences a thread could overwrite a row another thread is still gathering (a forked sequence reading the slot another one updates), and a per-channel loop over thousands of prefill tokens is uncoalesced. Multi-sequence and prefill batches keep the get_rows / concat / cpy launches; the existing `ggml_cuda_check_fusion_memory_ranges` would reject every match here (the cache view is outside the run), so the conv input's disjointness from the inputs is checked directly and the cache aliasing is what the thread design covers.
