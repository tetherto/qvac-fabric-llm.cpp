# Changelog

## [b10069] - 2026-07-31

Rebase onto upstream `ggml-org/llama.cpp` tag `b10069`
(previous base: `b9840`).

- Upstream window: `b9840` -> `b10069` (229 commits)
- Downstream commits: 364 (was 379 at `tether/temp-9840`)
- New downstream subjects since `tether/temp-9840`: 25 (~22 genuinely new work;
  3 are reworded/squashed carry-overs). Excludes 339 carry-over commits already
  present at the previous rebase point.
- 40 downstream commit *subjects* disappeared, but almost none of the code did.
  Each was checked by content, not by subject: 13 landed upstream in this window
  (see Downstream > Upstreamed), several were squashed into neighbouring commits
  (see Downstream > Changed), 4 were merge commits carrying no changes of their
  own, and 1 was already a no-op. Exactly one functional change is gone: the
  security-baseline workflow (see Downstream > Removed).

## Upstream (b9840 -> b10069)

Summary of upstream changes pulled in by this rebase.

#### DeepSeek-V4 / DFlash / Lightning Indexer

The dominant theme of this window. Much of what the fork carried downstream at
`b9840` is now upstream:

- `ggml : add GGML_OP_LIGHTNING_INDEXER that implements DeepSeek V3.2/V4 lightning indexer (#24231)`
- `cuda : CUDA GGML_OP_LIGHTNING_INDEXER implementation (generic vector kernel + wmma kernel) (#25545)`
- `DeepseekV4: Add fused hyper-connection ops (#25585)`
- `DeepseekV4: reduce graph splits (#25702)`
- `DeepseekV4: fix seq_rm (#25588)`
- `DeepseekV4: clear cache only for seq rather than full (#25521)`
- `llama : make all KQ masks f16 if FA is used, remove zero attention bias, remove raw_k repeats in DeepSeek V4 (#25370)`
- `llama: fix quantized kv-cache for dsv4 (#25202)`
- `tests : initialize all tensors in test_dsv4_hc to avoid NaNs in sentinel tensors (#25822)`
- `model: rotate injected K/V cache for DFlash (#25823)`
- `convert : fix dflash target tokenizer mismatch during conversion (#25733)`
- `common : auto-download dflash- and eagle3- HF sidecars (#25811)`
- `spec: support spec-draft-p-min in DFlash (#25246)`

#### New Model / Convert / Vocab / Chat

- `model: add Hy3 (hy_v3) support with MTP speculative decoding (#25395)`
- `conversion: accept BitNetForCausalLM architecture name (#25769)`
- `model : register t_layer_inp for qwen3next (#25141)`
- `chat : fix reasoning leak with force-opened bare <think> templates (#24674)`
- `chat: trim messages sent to StepFun parser (fixes long reasoning loops) (#25238)`
- `fix: OOB reads in UGM tokenizer (precompiled_charsmap handling) (#18750)`
- `TP: fix Phi3, Bert, Plamo2/3, ChatGLM (#25536)`

#### Quantization

- `Add Q2_0 quantization: type definition and CPU backend (#24448)`, with backend
  support following in `vulkan: Support Q2_0 (#25430)` and
  `metal : add Q2_0 support (#25419)`
- `quant : allow using manual tensor types with --pure (#25716)`
- `llama-quant : exclude i32 ffn_gate_tid2eid routing table from quantization (#25787)`
- `llama : add llama_model_ftype_name() (#25134)`

#### Multimodal (mtmd)

- `mtmd: deepseek-ocr v1 multi-tile (#24717)`
- `mtmd: fix silent prompt truncation on embedded NUL (#25548)`
- `server : allow text-only slot save/restore with mtmd (#25076)`
- `hexagon: add VISION RoPE support (#25216)`

#### Server / WebUI

- `server : refactor prompt cache state ownership (#25649)`
- `server: refactor server_stream (#25541)` and
  `server-stream: follow-up on SSE Replay Buffer (#23226) (#25047)`
- `server: enforce prompt cache RAM limit (#25070)`
- `server : evict checkpoints within min-step of each other (#25472)` and
  `server : respect min-step when splitting prompt batches (#25420)`
- `server: honour per-request reasoning_budget_tokens in chat completions (#23116)`
- `server: add --cors-* options (#25655)` and
  `server: Ignore empty / non-existing Origin headers (#25756)`
- `server: improve tools, remove apply_diff (#25498)`,
  `server: allow stream for exec_shell_command (#25526)`,
  `server: fix read_file append_loc space breaking edit_file match (#25705)`
- `server : add timings and progress to /responses API stream (#25348)`
- `server : fix image blocks in tool_result being dropped during Anthropic OpenAI conversion (#22536)`
- `server: fix deadlock in load_models() when erasing a finished download (#25358)`
- `server : fix draft model fit vs load inconsistency (#25056)` and
  `Fix stale tensor-split params for draft models (#24814)`
- `server: Don't consider models with --no-mmproj-auto as multimodal (#25590)`
- `server : move chat-template thinking probe inside the init try/catch (#24093)`
- `server: accept null sampling params (#25538)`, `server: remove loading.html (#25500)`
- `server + ui: ping silent SSE streams every 1s and kick only after 3s so slow prefill never drops healthy connections (#25241)`
- `ui: Context usage gauge and panel (#25340)`, `ui: Agentic Content UX improvements (#25450)`
- `ui: Improve performance when streaming (#25225)`,
  `ui: export full message tree instead of active path only (#25501)`
- MCP UI work: `#25535`, `#25631`, `#25239`
- `feat: pre-select models in the webui using alias (#25492)`
- Numerous smaller UI fixes: `#25637`, `#25634`, `#25529`, `#25539`, `#25503`,
  `#25307`, `#25298`, `#25132`, `#25137`, `#25242`, `#25177`, `#25174`, `#24874`

#### CLI / Common / API

- `cli : move to HTTP-based implementation (#24948)`, `cli: add --output option (#25484)`,
  `cli: fix crash on wrong server base url (#25497)`
- `common : use hf primary split as model path (#25194)` and
  `common : dedup preset and cached model entries in /v1/models (#25131)`
- `common,server: handle bracketed IPv6 literals in URL authority (#25140)`
- `common: auto-create prompts-log-dir at argument parsing, so all tools using the flag benefit (#25322)`
- `common: Set optimal default thread count for ppc ( linux as well as AIX) (#25237)`
- `arg: prevent duplicate spec model downloads (#25527)`,
  `arg: Flush log before exiting after usage() (#25504)`
- `gguf : add tensor shape accessor (#24405)`, `gguf : reject empty metadata keys (#24917)`
- `tokenize : drop --stdin mutual-exclusion check (#25672)`,
  `tokenize : align usage by using common args (#25516)`

#### llama core / KV cache / batching / speculative

- `llama: refactor fused ops (#24646)`
- `llama-batch: fix allowed decreasing pos in a seq (#25449)`,
  `llama-batch: add n_keep_tail in split_equal for recurrent models (#25278)`,
  `llama-batch: add unit test (#25471)`
- `llama : add guard for K/V rotation input when buffer is unallocated (#25215)`
- `speculative : fix out-of-bounds read in ngram-map on prompt shrink (#23936)`
- `Fix crash with draft-simple (#25720)`
- `llama : make tensor-split regex patterns static (#24710)`
- `Revert "sched : reintroduce less synchronizations during split compute (#20793)" (#25138)`

#### Backends

- Vulkan: Q2_0 (#25430); native e2m1/e4m3 conversions for mxfp4/nvfp4 (#25338);
  f16 as SET_ROWS src (#25432) and src0 type check (#25351); f16 out_prod +
  out_prod op (#23997); route large matmuls to medium tile on Adreno (#24877);
  disable FA mask_opt on GCN (#24362); reduce submission threshold by CU count
  on small AMD GPUs (#25240); flops-based submission heuristic (#25005); roll bk
  loop in matmul for Asahi Linux (#24663); fix 32-bit integer overflow in
  CEIL_DIV (#25245); sync on event_wait for transfer-queue async copies (#25229).
- Metal: Q2_0 (#25419); CONV_2D_DW depthwise convolution (#21565); col2im_1d
  f32/f16/bf16 (#25176); set_rows with src0 f16 (#25434); fuse snake activation
  (#25459).
- CUDA: Support CUDA Virtual Devices (#25228); remove -sm row, refactor cuBLAS
  (#24216); refactor MMQ kernel configuration (#24127); enable CUDA graphs on
  Volta+Turing (#25749); dedup MoE gate/up activation quantization (#25441);
  topk-moe fusion for 288 experts (#25267); Fuse MMVQ post-scale for NVFP4
  (#24481); quantized concat (#25303) and relaxed contiguity (#25678); smaller
  chunks in top_k/argsort to cut temp memory (#24776); fix get_rows_back for
  >65535 rows (#25103); fix Gemma E4B MTP FlashAttention (#25148); KQ mask
  stride overflow fix (#24945); VMM pool / Turing P2P fix (#24491).
- SYCL: Flash Attention with XMX engine via oneDNN (#25222); fused top-k MoE
  (#25217); xielu op (#25550); col2im_1d (#25264); cross_entropy_loss(+back)
  (#25236); Q2_K in DMMV reorder path (#25064); get_rows Q2_K/Q4_K/Q5_K fix
  (#25656); conv2d_dw fp16 (#25653); rename env vars from "disable" to "enable"
  (#25042).
- OpenCL (heavy Adreno focus this window): allow loading precompiled binary
  kernels from library (#23042) and use `kernel_gemm_moe_q6_k_f32_ns` from it
  (#25797); initial q1_0 support (#25160); int8 dp4 dense + MoE prefill
  optimization (#25537); cluster-parallel decode FA (#25473); general FA decode
  optimizations (#25366); ragged-tile MoE prefill FP16 GEMM (#25433); 128-bit
  vectorized LD/ST for MoE dp4a tiles (#25810); q4_K noshuffle scale transpose
  (#25805); q4_K/q5_K quants as uint for A7x (#25780); broadcast + `view_offs`
  for Adreno MUL_MAT (#25910); ABS op (#25115); several A7x/850 FA and MoE
  correctness fixes and workarounds (#25697, #25698, #25745, #25640, #25671,
  #25639, #25673, #25464, #25383).
- Hexagon: flash attention rework (#25085); new VTCM layouts and improved
  pipelines for MUL_MAT, MUL_MAT_ID and FLASH_ATTN_EXT (#25425); L2 cache
  handling rework with dirty-bit tracking and lazy flushing (#25762); tiling,
  tracing and unary-op optimizations (#25474); ARGSORT for small tensors
  (#25512); VISION RoPE (#25216); hmx-queue enum-narrowing fix (#25677).
- WebGPU: NVFP4 support (#25143); tune subgroup split in flash_attn_vec (#25418).
- HIP/ROCm: use hipBLAS for dense prefill on gfx900, keep MMQ for MoE (#24588);
  enable -ffast-math (#23862) / -funsafe-math-optimizations (#24668) and add
  -fno-finite-math-only (#25373); restore prop.integrated on HIP builds (#24233).
- ggml-cpu / KleidiAI: AVX2 nvfp4 dot product + UE4M3 LUT (#23961) and ARM
  UE4M3 LUT (#25331); tiled matmul on AIX (#25199); SME vs SME2 kernel dispatch
  distinction (#25478); SME2 f32 kernel (#24414).
- `ggml-et: Initial ET backend (#24179)`

#### ggml core / ops / tests

- `ggml : bump version to 0.17.0 (ggml/1568)`
- `ggml : add a set of functions for checking contiguity of inner tensor dimensions (#25650)`
- `ggml : uniformize im2col dst_type for all conv ops (#23660)` and `ggml : fix conv 2d dw (#25490)`
- `ggml : fix broken CPU concat implementation for quantized types (#25247)`
- `ggml : add support for CPU f16->f16 GGML_OP_SET_ROWS (#25344)`
- `ggml : fix tensor-parallel + -ncmoe crash on MoE models (#25028)`
- `ggml : fix A indexing in simd_gemm scalar tail-column path (#25390)`
- `ggml : make ggml_time_init idempotent (#24422)`
- `tests: actually exercise test-recurrent-state-rollback (#25758)`
- `Refactor: Consistently use smart pointers in test-backend-ops (#25440)`

#### Build / CI / Vendor

- `vendor : update cpp-httplib to 0.50.1 (#25576)` and `0.49.0 (#25218)`
- `vendor: update BoringSSL to 0.20260713.0 (#25624)`
- `ci : add HF_TOKEN to self-hosted workflows (#25706)` and
  `scripts : use HF_TOKEN when downloading UI assets (#25280)`
- `Make hip quality check run on all changes (#25403)`
- `ci : add official website link to release notes (#25728)`
- `meta: add hard emphasis on agents not writing descriptions/comments (#25480)`

## Downstream

### Added

#### Backend op-coverage CI guard

- `5587e68af ci: add backend op-coverage manifest guard + SVE variant tripwire`
  (+ `e8fb282d9` fixup)

A `supports_op()` regression never fails `test-backend-ops`: the case silently
falls back to CPU and is reported as `not supported [backend]`, i.e. skipped.
Adds `scripts/check-backend-op-coverage.sh` plus a checked-in
`ci/op-coverage/opencl-pocl.txt` manifest so a dropped op type turns the build
red instead of vanishing into the skip list.

#### Flash-attention shape coverage

- `4d6e1f72a tests: add partial-tile FLASH_ATTN_EXT cases across KV-type kernel variants`,
 `n_q=33` / `n_kv=513` leaves out-of-range query lanes for any power-of-two
  query tile and a 1-valid-row final KV tile, exercising the barrier-crossing
  race class. Covers each KV type separately because backends specialize
  kernels per KV type.
- `2a29c4901 tests: add no-mask n_q == n_kv FLASH_ATTN_EXT cases (vision-tower shape)`,
  the existing FA sweep never sets `nb == kv`, so the
  `mask == NULL && n_q == n_kv` shape a ViT self-attention layer produces was
  untested; a backend inferring causality from that shape silently computed
  causal attention over the whole vision tower with no test going red.

#### Misc

- `8759dd4d5 common: extract SFT dataset builder into common/finetune.{h,cpp}`,
  moves `common_opt_sft_dataset_init` out of `common.cpp`, dropping the
  nlohmann include, and unifies the per-template assistant-span scanners into
  one tag-driven loop.

### Fixed

#### Vulkan

- `b4f1b6f75 vulkan: fix count_equal_masked device-lost hang on RADV/GFX1151`,
  reduce in shared memory and accumulate with a 32-bit atomic instead of a
  64-bit one.
- `88378f29b vulkan: compile pipelines on a big-stack thread on macOS`, the
  512 KiB secondary-thread stack overflows in `ggml_vk_load_shaders`. Squashes
  the two separate lazy/eager commits carried at `b9840`.
- `6e360c810 vulkan: bounds-check the padded row index in norm/sum-family shaders`.
  so >512-row dispatches don't write past the destination tensor.
- `ae13198ce fixup! vulkan : keep SPIRV-Headers a build-only dep so ggml export stays clean (QVAC-21361)`
- `fixup! tests: add ops tests for cross_entropy_loss_masked`, read the
  cross-entropy upstream gradient on the host again. Reading it from `data_a[0]`
  in the shader made MoE LoRA finetuning diverge: `gemma-4-26B-A4B-it-Q8_0`
  reached a final loss of 6.04 instead of 3.24, learning ~4x less per step.
  Only MoE archs are affected, since `llama_context::opt_init` gives them a
  non-unit `loss_scale`, which makes `dst->src[0]` an in-graph `ggml_scale` node
  rather than the `grad_acc` leaf.

#### OpenCL / Adreno

- `02618863d Serialize lazy OpenCL Flash Attention compilation`, guards the
  per-device FA maps and shared kernel-argument state at both entry points.
- `97a1ecd12 fixup! ggml-opencl: add trailing barrier in f32/f16 flash-attn tile loop + guard upscale zero dims`
- `a68b35970 fixup! fix: QVAC-21914 review fixes, work-budget flush, memory-clamp rework, tests`

#### ROCm / HIP

- `64706d829 hip: allow-list rwkv_wkv_f32<128> VGPRs`, the VGPR check fails any
  HIP kernel using more than 256 fast registers unless allow-listed.

#### Multimodal

- `aeaf0a663 fixup! feat(mtmd/qwen3vl): multi-tile batching with --image-tile-mode`
- `c23bc1e18 fixup! fix(mtmd/qwen3vl): collapse batched attention loop + sequential default`

#### Mali

- `163b38750 fixup! feat: QVAC-21320 Mali GPU projector optimizations, disable FA, warptile, layernorm fusion`

#### Server

- `ee4d94bc2 fixup! server: save and clear idle slots on new task (--clear-idle) (#20993)`

#### Build

- `396bd1616 ui: pin HF asset version to the upstream base tag, not the commit count`,
 `LLAMA_BUILD_NUMBER` comes from `git rev-list --count HEAD`, which in this
  fork also counts every downstream commit. The prebuilt-UI download then asked
  the `ggml-org/llama-ui` bucket for a never-published tag, 404'd, and fell back
  to `latest`, whose bundle no longer ships `loading.html`. Since
  `llama-ui-embed` requires that asset, every networked build reaching the
  `llama-ui-assets` target died at configure time, even with
  `-DLLAMA_BUILD_UI=OFF`. Now resolved from the nearest reachable upstream
  release tag.
- `801aec4b4 Load the exported OpenSSL dependency`

#### Tests / CI

- `a4ad5dc6b ci: make nproc portable and verify it in self-hosted deps`, macOS
  runners often lack GNU coreutils, so `$(nproc)` silently expanded to nothing
  (unbounded `-j`, empty thread args).
- `9f1c8a457 ci: fail fast when jinja2 is missing from the CI venv`, the venv
  pip installs run without `set -e`, so a failed install surfaced much later as
  a confusing `ModuleNotFoundError` inside `test-jinja-py`.
- `288a01368 ci: install jinja2 explicitly in the venv so test-jinja-py doesn't depend on the torch pin surviving pip install`
- `c4e7d431c tests: skip test-recurrent-state-rollback on WebGPU builds`

### Changed

- `70554697f model: drop unused params from the model-load path (post-#22004 cleanup)`,
  removes the unused `metadata`, `set_tensor_data`, `set_tensor_data_ud` and
  `file` parameters from `llama_model_load`. Carried as a standalone downstream
  patch rather than a `fixup!`, because its target is an upstream commit and so
  can never autosquash.
- `b5333f5db vulkan: Add debug info`, reworded carry-over of `d71ef548b`; adds
  `vulkan_profiling_analyzer.py` and the backend instrumentation it consumes.
- 9 `fixup!` / `squash!` commits remain unsquashed on the branch and should be
  autosquashed before this becomes a release point.
- Several `b9840` commit subjects disappeared without the code being dropped:- The four GitHub merge commits from the `b9840` line (`#176`, `#186`, `#187`,
  `#194`). `qvac-b10069` has no merge commits at all, as expected for a linear
  rebase; nothing of substance was carried in them.
  the patches were squashed or folded into neighbouring commits during the
  rebase. Verified still present on `qvac-b10069` by content, not by subject:
  - `vulkan: guard HC comb dispatch limits` — `DSV4_HC_COMB_WG_SIZE` and the
    `maxComputeWorkGroupCount[0]` guard (`ggml-vulkan.cpp:1748`)
  - `vulkan: fix strided concat addressing` — the `src0_nb0`/`src1_nb0`/`dst_nb0`
    dimension-0 view-stride fix (`ggml-vulkan.cpp:14931`); still downstream-only,
    not upstream
  - `vulkan: require wave64 for Lightning Indexer CM1` — Lightning Indexer
    Vulkan code retained; upstream `b10069` has none of it
  - `ggml-vulkan: remove sc_carry scratch region (S_v * S_v) ... GDN-back op`
    (`ggml-vulkan.cpp:14724`)
  - `CPU: add support for fp16_fp32 OUT_PROD op` — superseded upstream by
    `ggml: add f16 out_prod support for CPU and out_prod op for Vulkan (#23997)`,
    which carries `ggml_compute_forward_out_prod_f16_f32` in this window

### Upstreamed

These downstream patches were dropped because upstream now carries them:

- `ggml : add GGML_OP_LIGHTNING_INDEXER ... (#24231)`
- `cuda : CUDA GGML_OP_LIGHTNING_INDEXER implementation (#25545)`
- `DeepseekV4: Add fused hyper-connection ops (#25585)`
- `llama : make all KQ masks f16 if FA is used ... DeepSeek V4 (#25370)`
- `llama: fix quantized kv-cache for dsv4 (#25202)`
- `llama : add guard for K/V rotation input when buffer is unallocated (#25215)`
- `llama: refactor fused ops (#24646)`
- `ggml : add support for CPU f16->f16 GGML_OP_SET_ROWS (#25344)`
- `cuda : add support for f16->f16 GGML_OP_SET_ROWS (#25367)`
- `vulkan/cpu: Support f16 as SET_ROWS src. (#25432)`
- `vulkan : check src0 type in GGML_OP_SET_ROWS ... (#25351)`
- `metal : add set_rows with src0 f16 (#25434)`
- `ggml : fix broken CPU concat implementation for quantized types (#25247)`

The downstream Metal `CONV_2D_DW` work (kernel, f32-only weights + CWHN, kargs
comment trim) is likewise superseded by upstream
`metal : add CONV_2D_DW (depthwise convolution) support (#21565)`.
