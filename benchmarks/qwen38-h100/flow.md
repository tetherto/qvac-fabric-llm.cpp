# Code flow: an FP8 checkpoint through fabric, before and after

## Before (HEAD `54db4c109` plus the campaign patches I00 to T1)

Conversion (`convert_hf_to_gguf.py` -> `conversion/base.py`):
1. `index_tensors` lists the safetensors parts. Only parts named `model*.safetensors` were recognized; a checkpoint with `layers-N.safetensors` plus an index produced an empty tensor map.
2. `dequant_model` (`quant_method == "fp8"`) replaced every `weight` that has a `_scale_inv` with a lazy `dequant_simple(w, s, block_size)`: F32 = fp8 * scale expanded over 128x128 blocks.
3. The main loop cast to the requested `--outtype` (bf16 or q8_0). The FP8 information was gone; the GGUF held BF16 or Q8_0 weights.

Inference (Q8_0 weights):
- Decode (1 token): `ggml_cuda_mul_mat` -> `ggml_cuda_should_use_mmvq` -> `mul_mat_vec_q<q8_0>` after `quantize_q8_1` turns the F32 activation into q8_1 (one extra launch per matmul, 384 per token).
- Prefill (4096 tokens per ubatch): `ggml_cuda_should_use_mmq` is false above the Hopper gate (ne11 >= 256), so `ggml_cuda_mul_mat_cublas`: `dequantize_block_q8_0_f16` on the whole weight into a pool buffer, `convert_unary_vec8` F32->F16 on the activations, `cublasGemmEx` F16 in / F32 out.

## After (branch `fp8-h100-campaign`)

### Conversion, `--outtype fp8`

1. `index_tensors`: a checkpoint is treated as safetensors when `model.safetensors.index.json` exists even if no part starts with `model` (fix for this checkpoint's `layers-N.safetensors`).
2. `prepare_tensors` calls `_repack_fp8()` before `dequant_model` when `ftype == MOSTLY_F8_E4M3`:
   - requires `quant_method == "fp8"` and `weight_block_size == [128, 128]`;
   - for every `weight` with a `_scale_inv` whose shape is 128-aligned: `weight.view(torch.uint8)` (the raw e4m3 bytes) and `scale.float()` go through the arch hook `modify_fp8_raw(name, weight, scale)`;
   - `conversion/qwen.py::modify_fp8_raw` permutes the V head blocks of `in_proj_qkv` rows, `in_proj_z` rows and `out_proj` columns exactly like `modify_tensors` does for dequantized weights, and permutes the scale grid the same way (the permutation moves whole 128-row/column blocks, so the 128x128 scale grid moves block-wise);
   - the weight is written as `GGML_TYPE_F8_E4M3` with the GGUF name from `map_tensor_name`, the scale as F32 `<name>.scale` with ggml shape `{N/128, K/128}` (numpy `scale.T`, C-contiguous);
   - both are removed from `model_tensors`, so `dequant_model` and the main loop never see them. Tensors that fail the 128-alignment rule (none in this checkpoint) keep the old dequant path and end up BF16.
3. The remaining 2-D tensors (lm_head, token_embd, in_proj_a/b, MTP eh_proj) become BF16 under this ftype; 1-D tensors stay F32.

Result for Qwen3.8-27B: 407 `F8_E4M3` tensors, 407 F32 `.scale` tensors, `general.file_type = 42`, 29.96 GB. A cross-check (`f8check.py`, kept on the host) showed `F8 * scale` equal bit-for-bit to the old dequant path after `modify_tensors` for the permuted linear-attention tensors, an FFN tensor and an attention tensor.

### Loading (`src/llama-model.cpp`, `src/llama-model-loader.cpp`)

- `llama_model_ftype_name` and the ftype guess know `LLAMA_FTYPE_MOSTLY_F8_E4M3`.
- The generic sidecar pass after `load_arch_tensors` uses `create_scale(tn, weight)`: for an F8 weight it loads `{N/128, K/128}` and throws if the sidecar is missing; for other types it loads the `{1}` per-tensor scale as before; for a skipped weight (the unused NextN layer when MTP is off) it consumes the sidecar with `TENSOR_SKIP` so the loader's tensor count still matches.
- The output head condition accepts F8 as well as NVFP4 (not used by this checkpoint, lm_head is BF16).

### Graph (`src/llama-graph.cpp`)

- `build_lora_mm(w, cur, w_s)`: F8 weight -> `ggml_mul_mat_blockscaled(w, w_s, cur)` (scale as `src[2]`); every other type -> `ggml_mul_mat` plus the old post-multiply when a scale exists.
- `build_ffn`: passes `up_s/gate_s/down_s` into `build_lora_mm` only for F8 weights and skips the post-multiply for them; NVFP4 behavior unchanged.
- `build_lora_mm_id` asserts no F8 (MoE is not supported).

### ggml core

- `GGML_TYPE_F8_E4M3` (51): `type_size 1`, `blck_size 1`, `is_quantized false`, `to_float` returns unscaled e4m3 values, `from_float_ref` NULL.
- `ggml_mul_mat_blockscaled` asserts the type, the 128-alignment and the sidecar shape, then builds a normal `MUL_MAT` node with `src[2] = scale`.
- `ggml_e4m3_to_fp32` / `ggml_fp32_to_e4m3` (`ggml-impl.h`), `ggml_validate_row_data` accepts every byte.
- CPU: `ggml_compute_forward_mul_mat` branches to `ggml_compute_forward_mul_mat_f8_e4m3` first (threads split rows; each 128-block decoded once per 64-token chunk; scale applied per block). This is the reference `test-backend-ops` compares CUDA against.
- Metal and SYCL reject the type in `supports_op`; Vulkan and WebGPU are allowlists and never accept it.

### CUDA (`ggml/src/ggml-cuda/mmf8.cu`, first branch of `ggml_cuda_mul_mat`)

```
ggml_cuda_mul_mat_f8(src0 = F8 weight [K][N], src1 = F32 activations [K][M], dst = F32 [N][M], src[2] = scale [N/128][K/128])
  M <= 8   -> mul_mat_vec_f8_e4m3<ncols_dst, nrows_w>   (decode GEMV)
  M > 8    -> (GGML_CUDA_CUTLASS build on Hopper) quantize_f8_e4m3_group128 + ggml_cuda_mmf8_cutlass   (W8A8 GEMM, opt-in, rejected by the gate)
           -> otherwise dequant_f8_e4m3_blockscaled_f16 + cublasGemmEx (F16 in, F32 out)   (default)
```

- GEMV: block = 4 warps, owns 4 consecutive rows; warps split k in 512-column trips (16 bytes per lane); per trip each lane converts 4 x 16 weights with the hardware e4m3->f16 cvt, reuses one F32 activation slice for the 4 rows, applies the block scale once; warp reduce, then shared-memory reduce across warps. 1 to 8 activation columns supported (rows per warp drop to 2 and 1 as columns grow).
- Fallback GEMM: identical structure to the old Q8 path (whole-weight dequant into F16, activation convert, cuBLAS), with the block scale applied inside the dequant.
- CUTLASS path (`mmf8-cutlass.cu`, `sm_90a` object library): activations quantized per token per 128-group (scale `max(amax/448, FLT_MIN)`, scales stored `[K/128][M_pad]`), M padded to a multiple of 4 for TMA, `KernelTmaWarpSpecializedCooperativeFP8Blockwise` tile 128x128x128, cluster 1x1x1, persistent scheduler, output F32 row-major straight into `dst` (or a padded buffer copied back).
- `supports_op` accepts F8 `MUL_MAT` with F32 activations (the loader probes the op without `src[2]`).

### Tests

`tests/test-backend-ops.cpp`: `test_mul_mat_f8(k, n, m, b_max)` builds `ggml_mul_mat_blockscaled`, fills the weights with random e4m3 bytes (no NaN encodings), scales in [0.5, 2], activations in [-b_max, b_max]; 18 cases (model shapes, odd M, tiny activations), tolerance 5e-4 NMSE at M <= 8 and 5e-3 above; run with `-o MUL_MAT_F8`.
