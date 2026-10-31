# Qwen3.8-27B Core ML prefill sidecar

This preserves the validated 24-layer experiment from `temp-11259` (`be6e387b9f2fda9a6fa0ca59f9607f2b6bb21db2`). It is a standalone, opt-in benchmark; `llama-server-coreml` (below) runs it inside `llama-server` for benchmarking, not as production integration. Normal builds do not enable the hooks.

Start with [RESULTS.md](RESULTS.md), the [24-layer preset](configs/qwen3.8-27b-24.json), and the [recorded measurements](results/2026-10-29/summary.json). Run `python3 examples/ffn-coreml/verify_results.py` to verify the committed evidence without loading a model.

## Validated configuration

- Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, architecture `qwen35`, 64 layers; model SHA256 is in the preset.
- Selected zero-based layers: 31, 35, 39, and 43 through 63.
- Each selected FFN keeps its first 11264 channels on Metal and sends the last 6144 of 17408 channels to Core ML CPU+Neural Engine. Hidden size is 5120; prefill microbatches contain 512 tokens.
- Layer 31 uses FP16 weights. The other 23 sidecars use 8-bit per-tensor palettes, tanh SiLU, channel chunks of 2048, and output scale 1.
- Attention, recurrent-state operations, other FFN channels, and single-token decode remain on Metal. This is about 13.24% of uniform-width FFN channel work, not 24 complete layers or 13.24% of total prefill time.

## Dual Neural Engine configuration

The [64-layer dual preset](configs/qwen3.8-27b-64-dual.json) uses both Neural Engines on M3 Ultra. Each of the 64 FFNs keeps its first 8192 channels on Metal. Core ML gets two 4608-channel packages per layer, `ffn-512.mlpackage` and `ffn-512-1.mlpackage`. The runtime loads the second package when it exists and runs both predictions at the same time. Two concurrent Core ML predictions ran on separate ANEs in our tests; one 6144-channel package took 9.75 ms, two concurrent 3072-channel packages took 5.48 ms.

By default, each prediction writes its output straight into its own Metal shared buffer (`MLPredictionOptions.outputBackings`) and the Metal join adds the three parts. This removes the CPU copy and sum of the Core ML outputs (about 1.3 ms per layer call) from the critical path. If Core ML ignores a backing, the runtime copies instead and reports it as `output_backing_fallbacks`; all recorded runs reported 0. Set `FFN_COREML_OUTPUT_BACKING=0` to keep the CPU copy, for example for A/B timing.

Measured on M3 Ultra (96 GB), 2026-10-30, same binary and session, 3 repetitions each:

| Prompt | Metal prefill | Dual prefill, output backing | Throughput |
| --- | ---: | ---: | ---: |
| 2048 tokens | 6.797 s | 4.776 s | +42.3% |
| 8192 tokens | 28.075 s | 20.067 s | +39.9% |

With `FFN_COREML_OUTPUT_BACKING=0`, the same build measured 5.155 s / 21.588 s (+8.0% for backing). Decode speed is unchanged, with zero decode offloads. Accuracy was qualified on 65536 held-out tokens of real, non-repeated text (code, documentation, prose, mathematics) instead of repeated prompts. With output backing, mean NLL change is +0.00026 nats/token (95% bootstrap upper bound +0.00082), mean KL 0.00042, and the worst domain (prose) +0.00109 nats/token with KL 0.00108. An all-Metal control through the same boundary has mean KL 0.0000014. The repeated-prompt KL suite above overstates divergence for this model; on held-out text it reads 5-10x lower. The held-out gate scripts and raw logs are not committed. Task benchmarks, memory and power were not measured.

Moving the split point was slower in both directions (9216 / 4096 + 4096: -3.7%; 7168 / 5120 + 5120: -11.4% at 2048 tokens). Core ML keeps about 128 ANE programs resident per process, and this preset uses 128, so offloading more operations through public Core ML slows every call.

## Build and run

Use Apple Silicon/macOS with Core ML and Metal. Exports target macOS 15; conversion used coremltools 9.0.

```sh
cmake -S examples/ffn-coreml -B build-ffn-coreml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DFFN_COREML_DECODER_EXPERIMENT=ON -DGGML_METAL=ON
cmake --build build-ffn-coreml --target llama-coreml-decoder-bench llama-ffn-coreml -j 8

export FFN_COREML_GPU_RESIDENT=1
export FFN_COREML_THREADS_BATCH=1
export FFN_COREML_SINGLE_ENCODER=1
export FFN_COREML_DECODE_PLAN=512
export FFN_COREML_CACHE_COMPILED=1

MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
ASSETS=/path/to/sidecar24
PROMPTS=examples/ffn-coreml/results/2026-10-29/prompts
build-ffn-coreml/llama-coreml-decoder-bench "$MODEL" "$ASSETS" \
  quality "$PROMPTS/code.txt" 512,2048 16 1 /tmp/qwen-quality
build-ffn-coreml/llama-coreml-decoder-bench "$MODEL" "$ASSETS" \
  split "$PROMPTS/prose.txt" 2048,8192 128 5 /tmp/qwen-split
```

Use fresh output directories. Mode `metal` is the ordinary-Metal timing reference; `quality-metal` checks the custom boundary with all FFN arithmetic on Metal. `quality` compares full-vocabulary teacher-forced logits against ordinary Metal. A completed quality command can return zero with `screen_pass=0`: inspect every quality row. Require finite metrics, mean KL <= 0.001, perplexity ratio <= 1.01, and exactly `selected_layers * prompt_tokens / 512` offload calls. Decode must have zero offloads.

A full qualification checks code/prose/math at 512 and 2048 tokens, plus build-guide/schema-code at 512, 2048 and 8192 tokens. The recorded suite contains 12 Core ML cases per configuration, plus GPU-only controls. Passing individual layers does not qualify their combination.

## Server, padding and the attention kernel

`-DFFN_COREML_SERVER=ON` (with `-DFFN_COREML_DECODER_EXPERIMENT=ON`) also builds `llama-server-coreml`: plain `llama-server` with the sidecar registered at startup. Set `FFN_COREML_ASSETS` to the sidecar directory and `FFN_COREML_GPU_RESIDENT=1`, then pass the usual `llama-server` flags; the Metal part reads its weights from `-m`. Without `FFN_COREML_ASSETS` it is a plain `llama-server`. The engine-bench comparison ran it with `FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_CACHE_COMPILED=1 LLAMA_GDN_DEFERRED=1` and `-ngl 99 -fa on --no-cache-prompt -cram 0 -ctxcp 0 -kvu`.

The Core ML programs take 512 rows. A partial last microbatch of at least `FFN_COREML_PAD_MIN` rows (default 256) is padded with zero rows to 512 and split as usual; smaller ones stay on Metal. Below about 250 rows padding was slower than Metal alone; at 465 rows (a 977-token prompt) it gave +12.6% prefill.

Long-prompt numbers in the engine-bench report were measured with the Metal prefill attention kernel from branch `feat/metal-fa-prefill-d256` merged in. That kernel is independent of Core ML and benefits plain Metal as well.

## Overlapping attention with the Neural Engines

The Neural Engines only work during each layer's FFN, so on their own they sit idle while Metal runs attention and DeltaNet. With `LLAMA_QWEN35_HALVES=1`, `FFN_COREML_ASYNC=1` and 1024-token microbatches (`-ub 1024` for the server, `FFN_COREML_UBATCH=1024` for the harness), each prefill microbatch of 513 to 1024 tokens runs as two interleaved halves: half A is the first 512 tokens, half B the rest. Per layer the GPU runs B's attention while the Neural Engines run A's FFN, then A's next attention while they run B's FFN. B's attention reads A's K/V of the same layer from the cache, and B's DeltaNet continues from A's state. Core ML calls are handed to a serial worker; the GPU signals a shared Metal event when a half's FFN input is written and waits on a second one before the join. A second half shorter than 512 rows is padded, or runs its FFN on Metal below `FFN_COREML_PAD_MIN`. Decode, multi-sequence batches and the deferred gated delta net (speculative decoding) keep the normal graph.

Because the Neural Engines no longer hold the GPU up, the overlap pairs with a larger Neural Engine share: [qwen3.8-27b-64-dual-s5120.json](configs/qwen3.8-27b-64-dual-s5120.json) keeps 5120 channels on Metal and gives 6144 to each Neural Engine. On M3 Ultra with the Q4_0 GGUF and the prefill attention kernel from `feat/metal-fa-prefill-d256`, single-request server prefill (engine-bench client) measured 519.4 / 537.6 / 507.4 / 454.9 / 379.1 / 294.4 tok/s at about 1k / 4.5k / 21k / 38k / 70k / 123k tokens, against 395.8 / 435.9 / 400.0 / 365.9 / 316.5 / 255.6 for the 8192-channel split without the overlap. The held-out gate passes: mean NLL change +0.00044 nats/token (upper bound +0.00083), KL 0.00046, worst window +0.00211. Without the overlap this split is slower than the 8192-channel one, so keep the two together.

## Held-out quality gate

The gate compares the split against ordinary Metal on 23 windows of real text that were never used for calibration or layer selection: 20 of 2048 tokens and 3 of 8192 (C++, Python, documentation, a novel and mathematics, 65,536 tokens). [gate/windows.json](gate/windows.json) lists every window as a slice of a source with its SHA256. Repository sources are read at the recorded commit; the five external texts (Project Gutenberg and Wikipedia wikitext, not committed) must be downloaded into one directory first.

```sh
python3 examples/ffn-coreml/gate/build_windows.py /path/to/external /tmp/heldout
examples/ffn-coreml/gate/run_gate.sh build-ffn-coreml/llama-coreml-decoder-bench "$MODEL" "$ASSETS" /tmp/heldout /tmp/gate-out
```

`run_gate.sh` runs each window in `quality` mode and in `quality-metal` mode (the all-Metal control through the same boundary), then `analyze_gate.py` writes `gate.json`. It passes when the mean NLL change is at most +0.002 nats/token with a 95% bootstrap upper bound at most +0.005, every domain is at most +0.003 with KL at most max(0.003, 2x control), no window exceeds +0.005, the overall KL is at most max(0.002, 2x control), and every run has exact offload counts and finite outputs.

## Sidecar assets

`ASSETS/layers.txt` lists selected IDs. Each `ASSETS/layerN/ffn-512.mlpackage` contains that layer's suffix. Weights and compiled models are not committed. On the original benchmark Mac, the retained 24-layer selection is `$HOME/qwen38-overall-20261029/combined/expand24/selection`. Connection details remain in the task conversation. Recorded paths replace the original user home with `${HOME}`.

`export_sidecar.py` builds a complete asset directory from a preset: it exports and palettizes every selected layer, writes one or two packages per layer, `layers.txt`, and an `export.json` manifest with the GGUF SHA256, tool hashes and channel ranges. The destination must not exist; failures remove temporary output. Use a separate Python environment with the pinned requirements:

```sh
python3 -m venv .venv-ffn-coreml
.venv-ffn-coreml/bin/python -m pip install -r examples/ffn-coreml/requirements.txt
.venv-ffn-coreml/bin/python examples/ffn-coreml/export_sidecar.py "$MODEL" \
  examples/ffn-coreml/configs/qwen3.8-27b-64-dual.json "$ASSETS" --workers 8
```

The 64-layer dual export needs about 8.4 GB for compressed packages, plus the same again for compiled caches unless they are APFS clones. The equivalent manual steps for the 24-layer preset, with the low-level tools:

```sh
mkdir -p "$ASSETS"
cp examples/ffn-coreml/configs/qwen3.8-27b-24.layers.txt "$ASSETS/layers.txt"
for layer in $(cat "$ASSETS/layers.txt"); do
  source="$ASSETS/fp16/layer$layer"
  if [ "$layer" = 31 ]; then source="$ASSETS/layer31"; fi
  python3 examples/ffn-coreml/export_ffn.py "$MODEL" "$source" \
    --layer "$layer" --tokens 512 --chunk 2048 --activation tanh \
    --output-scale 1 --channel-start 11264 --channel-count 6144 --no-reference-files
  if [ "$layer" != 31 ]; then
    python3 examples/ffn-coreml/palettize_ffn.py "$source" "$ASSETS/layer$layer" \
      --bits 8 --tokens 512 --workers 4 --no-reference-files
  fi
done
```

Re-exported packages require fresh qualification; do not assume identical compression across tool versions. The palette tool refuses an existing destination. Intermediate FP16 packages can be archived or removed after recording their manifests and verifying the compressed packages. Keep layer 31 FP16.

Compiled-cache reuse follows the embedding-Gemma pattern. It creates `ffn-512.mlmodelc` next to each package, resolving selection symlinks so different selections reuse the same cache. Remove that compiled directory before changing its package. There is no automatic weight-identity invalidation or concurrent-writer support. Core ML compilation and model loading occur before timed inference.

## Runtime changes and handoff

- `src/models/qwen35.cpp`: optional FFN builder; only matching 512-token F32 inputs at selected layers are replaced. The normal builder handles all other cases.
- `ggml-metal.cpp` / `ggml-metal.h`: experiment-only custom-op registration and GPU-resident graph boundaries. The Metal prefix overlaps Core ML prediction; Metal joins the suffix result.
- `FFN_COREML_SINGLE_ENCODER`: scopes single-encoder execution to registered custom graphs. This avoids an observed fusion-boundary numerical change. It restores the configured default encoder count; future general-purpose support must preserve any runtime override instead.
- `FFN_COREML_DECODE_PLAN=512`: restores the full Metal allocation plan before decode, including output capacity. Omitting this previously caused a decode regression. Transition cost is included in timing.
- `FFN_COREML_PROFILE=1`: per-layer callback stages. Disable profiling and graph tracing for final timing.
- `decoder-coreml.mm`: an optional `ffn-512-1` package (package or compiled cache) must start where `ffn-512` ends and share its output scale. Its prediction runs on a GCD queue while the first runs on the calling thread.
- Output backing: on by default when `FFN_COREML_GPU_RESIDENT` is set and the output scale is 1; `FFN_COREML_OUTPUT_BACKING=0` disables it. Core ML outputs go to the Metal join inputs directly; `ffn-coreml-common.h` adds an optional third join input for the second package.

Callbacks and builder state are process-global. The harness drives contexts serially. Do not enable this as a multi-request/server backend without lifecycle, concurrency, cancellation, cache validation, and fallback work. This branch intentionally preserves the tested experiment rather than introducing production integration.

All remaining layers were screened: 25 new individual candidates were found; the combined 28-layer selection failed (mean KL 0.001127598). Both 22 and 24 passed. Layers 25-27 in *selection count* were not tested; 24 is the largest passing configuration tested, not a proven maximum. See `layer-screen.json` and `rejected28/` in the results directory for evidence.

For the next prefill investigation, keep this 24-layer configuration and ordinary Metal as baselines. Profile the remaining attention/state-update work and synchronization before choosing another offload boundary. Keep the existing accuracy gates and zero-decode-offload check; do not infer correctness or speedup from an isolated operator benchmark.
