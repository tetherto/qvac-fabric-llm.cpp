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
