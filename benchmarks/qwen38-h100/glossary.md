# Glossary for the Qwen3.8-27B / H100 campaign

Terms as they are used in `progress-log.md`, `decisions.md`, `flow.md` and `context.md`. Each entry says what the thing is, what it physically holds or does, and how fabric (llama.cpp) and vLLM/SGLang use it.

## Model files

**safetensors**: the HuggingFace checkpoint format. One file (or several `*.safetensors` parts plus a `model.safetensors.index.json` that maps tensor names to parts) holding a JSON header (tensor name, dtype, shape, byte offset) followed by the raw tensor bytes. Nothing else: the tokenizer, `config.json` (architecture, layer count, `quantization_config`) and the chat template are separate files in the same directory. The Qwen FP8 checkpoint stores each linear weight as `model.layers.N.mlp.up_proj.weight` (dtype `F8_E4M3`, shape [N, K]) and its scales as `...up_proj.weight_scale_inv` (dtype BF16, shape [N/128, K/128]); `config.json` says `quant_method: fp8`, `weight_block_size: [128, 128]`, `activation_scheme: dynamic`. vLLM and SGLang read the safetensors directly: the loader maps the names onto their model classes, keeps the e4m3 bytes and the scales on the GPU, and quantizes activations on the fly ("dynamic") in the FP8 GEMM.

**GGUF**: llama.cpp's single-file format. A header with key/value metadata (`general.architecture`, `qwen35.block_count`, tokenizer vocabulary and merges, chat template, `general.file_type`) then a tensor directory (name, ggml type, shape, offset) and the tensor data, aligned so the file can be memory-mapped and used in place. Tensor names are llama.cpp's own (`blk.N.ffn_up.weight`, `blk.N.attn_qkv.weight`, `token_embd.weight`, `output.weight`); the converter (`convert_hf_to_gguf.py`) renames, merges (q/k/v into one `attn_qkv`) and permutes (the linear-attention V heads) while writing. The type of each tensor is a ggml type id: F32 (0), F16 (1), Q8_0 (8), Q4_K (12), BF16 (30), NVFP4 (47), and now F8_E4M3 (51). Our FP8 GGUF stores each F8 weight as raw e4m3 bytes plus a separate F32 tensor `blk.N.ffn_up.scale` of shape {N/128, K/128}; the weight and its scale are tied together by name in the loader.

**Q8_0, Q4_K (ggml block types)**: llama.cpp quantizations with the scale stored inside every block, so one tensor is self-contained. Q8_0: blocks of 32 values, 1 int8 per value + 1 F16 scale = 34 bytes per 32 weights (8.5 bits/weight). Q4_K: super-blocks of 256 values, 4-bit values with 6-bit sub-scales and a super-block scale, about 4.5 bits/weight. Both run decode through `mmvq` (activations quantized to q8_1 first) and prefill through MMQ or, on Hopper above 256 tokens, cuBLAS after a dequant to F16.

**Q8_K_XL / Q4_K_XL (unsloth "UD" GGUFs)**: not a type but a per-tensor mix: unsloth's recipe chooses a ggml type per tensor (the trunk in Q8_0 or Q4_K, the token embedding, output head and some attention tensors in a wider type such as BF16 or Q8_0) and "XL" marks the more generous mix. The Q8_K_XL file used here is 31.46 GB and needs 29.51 GB of weight reads per decoded token (24.62 GB Q8_0 + 4.89 GB BF16); the FP8 GGUF 26.93 GB.

**FP8 e4m3 (`float8_e4m3fn`)**: 1 byte per value: sign, 4 exponent bits, 3 mantissa bits, range +-448, no infinities, one NaN encoding (0x7F / 0xFF). Too coarse to hold a whole weight matrix at one scale, so it is always paired with scales.

**BF16**: 2 bytes per value, the training dtype; same exponent range as F32 with 8 bits of mantissa. In the FP8 checkpoint `lm_head`, the token embedding and a few small linear-attention tensors are BF16 and we keep them so.

**Block scale / per-channel scale / per-tensor scale**: how many values share one scale factor. Per-tensor: one float for the whole matrix (what cuBLASLt's FP8 GEMM accepts on Hopper). Per-channel: one per output row. Block (128x128): one per 128x128 tile of the weight, so 3 MB of scales for a 24 GB model; this is what Qwen shipped and what DeepSeek-style FP8 uses. The dequantized weight is `W[n][k] = fp8[n][k] * scale[n/128][k/128]`.

**Sidecar tensor**: a separate tensor in the GGUF that belongs to another tensor (`.scale` next to `.weight`). The loader finds it by name and the graph passes it into the matmul as a third input (`src[2]`). NVFP4 already used this pattern for its per-tensor scales; F8 uses it for the block grid.

**W8A16 / W8A8**: weight and activation bit widths in a GEMM. Our fallback prefill and decode GEMV are W8A16/W8A32 (weights 8-bit, activations F16 or F32, so the weight is the only approximation). The CUTLASS path is W8A8: the activations are also rounded to e4m3 (per token, per 128-column group, one scale each) so the FP8 tensor cores can multiply them. That extra rounding is what failed the quality gate.

## Hardware and kernels

**Kernel**: one GPU program launched with a grid of thread blocks. A forward pass of this model launches about 1600 kernels per decoded token and about 20000 for a 10k-token prefill. Each launch costs a few microseconds of CPU/GPU overhead even when it does no work, which is why "launch count" is a decode lever.

**Thread, warp, block, SM**: a warp is 32 threads executing in lockstep; a block is a group of warps (we use 4 warps = 128 threads) sharing fast on-chip shared memory; an SM (streaming multiprocessor) runs several blocks at once; the H100 has 132 SMs. "One warp per row" means each 32-thread group computes one output row of a matrix-vector product.

**HBM bandwidth**: the H100's memory delivers 3.35 TB/s peak; about 2.8 to 2.9 TB/s is reachable by a real kernel. Decode is bandwidth-bound: every token must stream all 27 GB of weights, so the floor is 27 GB / 3.35 TB/s = 8 ms per token (124 tok/s). A GEMV at 2.8 TB/s takes 9.6 ms for the same bytes.

**Tensor cores, mma / WGMMA**: matrix multiply units. `mma` is the per-warp instruction (Ampere and later; our chunked GDN kernel uses it in TF32); `WGMMA` is the Hopper warp-group instruction that four warps issue together, fed from shared memory, and the only way to reach the H100's FP8 rate (1979 TFLOPS dense; FP16 is 989). **TMA** is the Hopper copy engine that moves tiles from HBM to shared memory asynchronously; CUTLASS's "TmaWarpSpecialized" kernels dedicate some warps to TMA loads and others to WGMMA.

**GEMM**: general matrix multiply, C[M][N] = A[M][K] x B[K][N]. In an LLM every linear layer is a GEMM with the tokens as M, the weight as B. FLOPs = 2 x M x N x K. Prefill runs GEMMs with M = 4096 tokens per micro-batch and is compute-bound; a 10k prompt is 48.7 GFLOP per token x 10240 tokens.

**GEMV**: the M = 1 (or up to 8) case: matrix times a vector. Decode is a sequence of GEMVs over every weight, bandwidth-bound, so the design goal is bytes in flight, not FLOPs. llama.cpp's GEMV kernels: `mmvq` (quantized weights, activations quantized to q8_1 first), `mmvf` (F16/BF16/F32 weights), and now `mul_mat_vec_f8_e4m3` (`mmf8.cu`).

**cuBLAS**: NVIDIA's closed GEMM library. llama.cpp calls `cublasGemmEx` with F16 inputs and F32 output for prefill of quantized models: it first dequantizes the weight to F16 into a scratch buffer (`dequantize_block_q8_0_f16`, or our `dequant_f8_e4m3_blockscaled_f16`), converts the activations to F16, then runs the library kernel. Measured in-graph rate: about 730 TFLOPS on this model (of 989 peak). **cuBLASLt** is its lower-level sibling with an FP8 GEMM, but on Hopper it takes only per-tensor scales, so it cannot run the 128x128 block-scaled checkpoint as shipped.

**CUTLASS**: NVIDIA's open-source C++ template library for writing GEMMs from tiles, copy atoms and MMA atoms. Version 4.2.1 ships `KernelTmaWarpSpecializedCooperativeFP8Blockwise`, the SM90 kernel that multiplies e4m3 A and B with per-block scales promoted into the fp32 accumulator every 128 K; DeepGEMM, vLLM and SGLang all run this kernel family for DeepSeek/Qwen FP8 on H100. Our `mmf8-cutlass.cu` instantiates it once (tile 128x128x128, cluster 1x1x1, persistent scheduler) and is compiled for `sm_90a` in a separate object library behind `GGML_CUDA_CUTLASS`. Measured: 1094 to 1136 TFLOPS standalone on the FFN shapes, 994 all-in with the activation quantizer.

**MMQ**: llama.cpp's own tensor-core GEMM for quantized weights (int8 mma). On H100 it saturates at 150 to 225 TFLOPS, so the campaign routes dense prefill above 256 tokens to cuBLAS instead.

**Tile / cluster / scheduler (CUTLASS terms)**: the tile is the output block one thread block computes (128x128 outputs over K in 128 steps); a cluster is a group of blocks that can share data through distributed shared memory (1x1x1 = off); the persistent scheduler keeps one block per SM alive and hands it tiles in turn, StreamK splits K across blocks for load balance (5% slower here).

**Swizzle / raster order**: the order in which tiles are assigned to SMs, chosen so concurrent tiles reuse the same rows of A or B from L2. `max_swizzle_size = 4` measured best.

**CUDA graph**: llama.cpp records the decode kernel sequence once and replays it, cutting per-launch CPU cost. It only helps when the graph does not change shape between tokens.

**Occupancy / bytes in flight**: how many warps (and outstanding loads) an SM holds at once. A bandwidth-bound kernel needs about 64 KB of loads in flight per SM to saturate HBM; our first GEMV had too few and sat at 1.66 TB/s, the final one splits K across the 4 warps of a block and reaches 2.8.

## llama.cpp / fabric internals

**ggml**: the tensor library under llama.cpp. A model forward pass is a **graph** of **ops** (`MUL_MAT`, `RMS_NORM`, `ROPE`, `FLASH_ATTN_EXT`, ...) over **tensors** (`ne[]` shape, `nb[]` byte strides, `type`, `src[]` inputs). Backends (CUDA, CPU, Metal, Vulkan, SYCL) implement the ops; `supports_op` says which ops a backend accepts, and the scheduler splits the graph between backends. `ggml_mul_mat_blockscaled` is the new graph builder that creates a `MUL_MAT` node with the scale as `src[2]`.

**Backend / device / buffer type**: the CUDA backend owns the GPU; weights live in a CUDA buffer; a tensor whose op the backend rejects falls back to the CPU (slowly). The loader probes `supports_op` for every weight with a MUL_MAT before placing it on the GPU.

**Loader (`llama-model-loader`)**: opens the GGUF, checks every expected tensor name and shape for the architecture (`create_tensor`), memory-maps the data, uploads to the backend buffers. The tensor count must match exactly; unused tensors (the NextN/MTP layer when speculative decoding is off) are marked `TENSOR_SKIP`.

**Graph builder (`llama-graph.cpp`, `models/qwen35.cpp`)**: builds the ggml graph per micro-batch: `build_lora_mm` wraps every weight matmul (and applies LoRA if loaded), `build_ffn` builds gate/up/activation/down, `build_attn` the attention, the Qwen3.5 model file the alternation of 48 gated-delta-net layers and 16 attention layers.

**ubatch / batch / n_ctx**: `--ubatch-size` (4096 here) is the number of tokens per graph evaluation, i.e. the M of the prefill GEMMs; `--batch-size` the logical batch the server assembles; `--ctx-size 262144` the KV cache capacity in tokens (256k).

**KV cache**: the attention keys and values of every past token, stored per layer; `-ctk f16 -ctv f16` stores them in F16 (q4_0 KV was 4x smaller but failed the quality floor). At 110k tokens the 16 attention layers hold 7.2 GB of KV, read once per decoded token.

**Gated delta net (GDN)**: the linear-attention block of Qwen3.5 (48 of 64 layers). It keeps a fixed-size recurrent state instead of a KV cache; prefill runs it in chunks on tensor cores (our P4 kernel, TF32 mma), decode runs one recurrent step per token.

**Prompt cache (`--cache-ram`)**: the server saves the KV/recurrent state of finished requests in host RAM so a later request with the same prefix skips the prefill (110k restored in 2.6 s instead of a 30 s re-prefill). The T1 fix made the save 3.7x faster by prefaulting the buffer pages in parallel.

**MTP / DFlash / speculative decoding**: extra small heads (`nextn` tensors) predict several tokens per step which the main model verifies in one batched pass. Reported in its own column, never counted as raw decode.

**Prefill vs decode, TTFT, tok/s**: prefill processes the prompt (compute-bound, measured in prompt tokens per second: pp10k = 10240-token prompt); decode generates one token per forward pass (bandwidth-bound, tg10k = generation rate with 10k tokens of context). TTFT is the client-side time to the first streamed token; `prompt_ms` the server's own prefill timer.

**llama-bench / pd_bench.py / llama-server**: `llama-bench` times the raw model at fixed shapes without HTTP or tokenization (`-p 10240 -n 0` = prefill only; `-p 0 -n 128 -d 10240` = decode at depth 10240); `llama-server` is the OpenAI-compatible server; `pd_bench.py` drives the server with the campaign's prompts (cold 10k, cold 110k, warm 100k+10k) and reports server timings and client TTFT. The server is what users see and its numbers are 11 to 16% below llama-bench for the same tokens.

**test-backend-ops**: ggml's op test: every op is run on the CUDA backend and on the CPU backend with the same random inputs and compared (`test` mode, tolerance in NMSE = normalized mean squared error) or timed (`perf` mode). `-o MUL_MAT_F8` selects our 18 F8 cases. Passing means the CUDA kernel matches the CPU reference; it says nothing about model quality.

## Measurement

**nsys / ncu**: NVIDIA's timeline profiler (`nsys profile --trace=cuda`, exported to sqlite; `kern_families.py` and `kern_window.py` sum kernel time per family and per token) and kernel profiler (`ncu`: per-kernel achieved bandwidth, occupancy; needs GPU counter permission, which this host does not grant).

**TFLOPS / TB/s / roofline**: a GEMM is scored in trillions of floating-point operations per second against the peak (H100: 989 FP16, 1979 FP8 dense); a GEMV in terabytes per second of weight bytes against 3.35. "Bound" means which of the two is the limit for a kernel; prefill is FLOP-bound, decode byte-bound.

**Perplexity (PPL)**: exp of the average per-token negative log-likelihood on a text (wikitext-2 test, 32 chunks of 4096 tokens here). Lower is better. Numbers in the ledger's F3 table: Q8_K_XL 6.0002 (its own logits run), FP8 fallback build 6.0126 and FP8 CUTLASS build 6.0249 (both scored against the Q8 logits); in the F2 gate run (CUTLASS scored against the fallback build's saved logits) the fallback build's `PPL(base)` came out as 6.0056, a different figure for the same build that is not investigated.

**KL divergence (KLD) and Same top p**: `llama-perplexity --kl-divergence` compares the full next-token distribution of a build against saved reference logits (`--save-all-logits`, 31 GB for 131k tokens). Mean KLD is the average divergence per token (0 = identical); Same top p is the share of positions where the argmax token agrees. Campaign gates: weight-only changes must keep Same top p >= 99.0% and Mean KLD <= 0.002 against the pre-campaign logits; the CUTLASS activation quantization had its own gate (<= 0.005 vs the fallback build) and failed it at 0.0059 / 98.03%.

**Quality gate / regression gate**: the pass/fail rule written into the ledger before a change is measured. A change that fails its gate is rejected regardless of speed; `test-backend-ops` must stay green for touched ops.

**A/B interleaved**: running build A and build B alternately (A B A B) on the same GPU so clock drift and neighbours affect both equally; the campaign's per-change signal for small effects.
