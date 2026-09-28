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

### Gated delta net on CUDA (`ggml/src/ggml-cuda/gated_delta_net.cu`, C1)

`ggml_cuda_op_gated_delta_net_impl` order for a prefill ubatch (q/k [S, H_k, T, S_n], v [S, H_v, T, S_n], gate, beta, state):
1. Opt-in external kernel: when `GGML_CUDA_GDN_AOT_LIB` names a library that exports `flashinfer_gdn_sm90_aot_launch` (loaded once with `dlopen`, Linux only), and the op is not KDA, keeps one state slot (K == 1), runs on cc 900 with S_v 128, T >= 64, H_v a multiple of H_k and at most 1023 sequences: a pack kernel writes BF16 q/k (expanded to one head per v-head with ggml's `h_v % H_k` mapping), BF16 v, `alpha = exp(g)` and beta into pool buffers, a tiny kernel writes `cu_seqlens`, the external fused delta-rule kernel writes BF16 output and the fp32 state, an unpack kernel converts the output to F32 into `dst`. Any failed condition or a nonzero launch status falls through.
2. Otherwise the P4 path: whole 64-token chunks through `gated_delta_net_chunked_cuda<S_v>` (TF32 tensor cores), then the serial kernel on the tail from the chunk state. Decode (T = 1) always takes the serial kernel.

Without the variable nothing changes: the loader returns null once and every call falls through (verified by a trace with zero FlashInfer launches and the P4 kernel present). Tests: `test_gated_delta_net` cases at the qwen35 shape family ((16, 128, 64/4096/4033, v_repeat 3), (4, 128, 127, 2 seqs, v_repeat 2)); tolerance 5e-5 NMSE with the library (BF16 I/O), 5e-7 for the TF32 chunked path (T >= 64, not KDA), the generic 1e-7 elsewhere.

## Campaign 2 (branch `h100-attn-decode`, on top of the above)

### Prefill attention without a mask (`ggml_flash_attn_ext_set_kv_used`, B1)

1. `llama_kv_cache::get_n_kv_used(sinfo, ubatch)` after `apply_ubatch`: for a ubatch of at least 256 tokens (`n_tokens_min_implicit_mask`), one stream, one sequence, no SWA/ALiBi/2D positions, it scans the cells and returns n when cells [0, n) are used, in position order, with the ubatch at the tail; 0 otherwise. `llama_context` probes once per context whether the attention backend accepts the op without a mask (`cparams.attn_implicit_mask`).
2. `build_attn_inp_kv_impl` stores `n_kv_used` on the input instead of filling the mask; `build_attn` passes it to `build_attn_mha`, which calls `ggml_flash_attn_ext` with a null mask and sets op param 4. `can_reuse` (plain and hybrid inputs) compares it.
3. CUDA `ggml_cuda_flash_attn_ext`: on cc 900 with D 256, f16 or f32 K/V, n_q >= 256 and K/V head strides that TMA accepts, `ggml_cuda_flash_attn_ext_cutlass_run` converts Q to f16, pads n_q to a multiple of 8 with zero phantom rows (and the same number of phantom cells past `n_kv_used`), converts f32 K/V cells to f16 when needed, launches the example-88 kernel (`fattn-cutlass.cu`, one launch per layer, B = kv heads, H = 6, `CausalOffsetFusion` with offset `n_kv_used - n_q`), and converts O back to f32. Otherwise `fattn_causal_mask_f16` materializes the [n_kv, n_q] mask on device and the existing kernels run. Every other backend returns false from `supports_op` for a mask-less op; the CPU reference applies the rule inside `ggml_compute_forward_flash_attn_ext_f16_one_chunk`.

### Decode fusions in `ggml_cuda_try_fuse` (D2, D4, D5, D6)

- `ADD -> RMS_NORM -> MUL` (the residual add feeding a norm): `ggml_cuda_should_fuse_add_rms_norm_mul` checks F32, same shapes, contiguity, width a multiple of 4 up to 16384, 16-byte alignment and the alias rules; `rms_norm_add_mul_f32<256|1024, vals>` keeps the row in registers, writes the sum and the normalized product. The subgraph check lists the add and the mul as outputs (the residual stream is read by the next layer).
- `SSM_CONV -> [ADD bias] -> UNARY silu` followed by the run of views / `L2_NORM` / `SCALE` nodes the delta-net builder emits (`ggml_cuda_try_ssm_conv_l2_fusion`): each `L2_NORM` must read a `[128, n_heads, n_t, n_s]` view of the silu output at a 128-aligned channel; the conv kernels (`with_l2` template) block-reduce the squares per token and write the normalized (scaled) rows; up to two slices (`ggml_cuda_ssm_conv_l2`). Head dims other than 128 keep the separate norm launches.
- alpha/beta gate projections (`ggml_cuda_try_gdn_gates_fusion`, `gdn-gates.cu`): `MUL_MAT -> [views] -> ADD dt_bias -> SOFTPLUS -> MUL a -> [views], MUL_MAT (same input) -> [views] -> SIGMOID -> [views]` at batch <= 8 with F16/BF16 weights; one block of 256 threads per output row and token, fp32 products like `mul_mat_vec_f`, the gate math in the epilogue.
- recurrent conv input (`ggml_cuda_try_conv_state_fusion`, `conv-state.cu`, D6): `GET_ROWS` of the state row -> [views, zero-sized extra-state nodes] -> `CONCAT` with the transposed tokens -> [views] -> `CPY` of the last d_conv-1 columns into the cache; one sequence, batch <= 8, d_conv 3/4/5/9; `conv_state_pack<d_conv>` reads a channel's values into registers and writes the conv input row and the next state row.

Each fusion is proven by a whole-graph `test-backend-ops` case (`ADD_RMS_NORM_MUL`, `SSM_CONV_L2`, `GDN_GATES`, `CONV_STATE`) and an nsys count of the fused kernel; `GGML_CUDA_DISABLE_FUSION=1` turns all of them off.

### Server checkpoints (`common_state_buffer`, S1)

`common_prompt_checkpoint::data_tgt/data_dft` are `common_state_buffer`s: `resize()` takes the smallest idle block that fits from a process-wide pool (freed blocks are kept up to 1 GiB), so a steady stream of 150 MiB checkpoints stops faulting fresh pages on every save; contents after `resize()` are undefined (the checkpoint fills the whole blob).

### Converter: `--fp8-output-head` (LM1, available but not used for the campaign GGUF)

With `--outtype fp8`, `_quantize_fp8_output_head` runs after `_repack_fp8` and quantizes the BF16 output head with `fp8_block_quantize` (per 128x128 block `scale = amax/448`, e4m3 round-to-nearest, sidecar `{n/128, k/128}`); `benchmarks/qwen38-h100/gguf_fp8_output_head.py` applies the same function to an existing F8 GGUF. Rejected for Qwen3.8-27B by the gate (decision 21).
