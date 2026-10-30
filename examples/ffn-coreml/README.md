# Qwen3.8-27B Core ML prefill sidecar

This preserves the validated 24-layer experiment from `temp-11259` (`be6e387b9f2fda9a6fa0ca59f9607f2b6bb21db2`). It is a standalone, opt-in, single-request benchmark, not server or application integration. Normal builds do not enable the hooks.

Start with [RESULTS.md](RESULTS.md), the [24-layer preset](configs/qwen3.8-27b-24.json), and the [recorded measurements](results/2026-10-29/summary.json). Run `python3 examples/ffn-coreml/verify_results.py` to verify the committed evidence without loading a model.

## Validated configuration

- Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, architecture `qwen35`, 64 layers; model SHA256 is in the preset.
- Selected zero-based layers: 31, 35, 39, and 43 through 63.
- Each selected FFN keeps its first 11264 channels on Metal and sends the last 6144 of 17408 channels to Core ML CPU+Neural Engine. Hidden size is 5120; prefill microbatches contain 512 tokens.
- Layer 31 uses FP16 weights. The other 23 sidecars use 8-bit per-tensor palettes, tanh SiLU, channel chunks of 2048, and output scale 1.
- Attention, recurrent-state operations, other FFN channels, and single-token decode remain on Metal. This is about 13.24% of uniform-width FFN channel work, not 24 complete layers or 13.24% of total prefill time.

## Dual Neural Engine configuration

The [64-layer dual preset](configs/qwen3.8-27b-64-dual.json) uses both Neural Engines on M3 Ultra. Each of the 64 FFNs keeps its first 8192 channels on Metal. Core ML gets two 4608-channel packages per layer, `ffn-512.mlpackage` and `ffn-512-1.mlpackage`. The runtime loads the second package when it exists, runs both predictions at the same time, and sums them in the existing output conversion. Two concurrent Core ML predictions ran on separate ANEs in our tests; one 6144-channel package took 9.75 ms, two concurrent 3072-channel packages took 5.48 ms.

Measured on M3 Ultra (96 GB), 2026-10-30, 3 split repetitions against 6 Metal repetitions:

| Prompt | Metal prefill | Dual prefill | Throughput |
| --- | ---: | ---: | ---: |
| 2048 tokens | 6.796 s | 5.491 s | +23.8% |
| 8192 tokens | 28.062 s | 22.776 s | +23.2% |

Decode speed is unchanged, with zero decode offloads. Accuracy was qualified on 65536 held-out tokens of real, non-repeated text (code, documentation, prose, mathematics) instead of repeated prompts. Mean NLL change is +0.00024 nats/token (95% bootstrap upper bound +0.00079), mean KL 0.00042, and the worst domain (prose) +0.00101 nats/token with KL 0.00108. An all-Metal control through the same boundary has mean KL 0.0000014. The repeated-prompt KL suite above overstates divergence for this model; on held-out text it reads 5-10x lower. The held-out gate scripts and raw logs are not committed. Task benchmarks, memory and power were not measured.

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

Callbacks and builder state are process-global. The harness drives contexts serially. Do not enable this as a multi-request/server backend without lifecycle, concurrency, cancellation, cache validation, and fallback work. This branch intentionally preserves the tested experiment rather than introducing production integration.

All remaining layers were screened: 25 new individual candidates were found; the combined 28-layer selection failed (mean KL 0.001127598). Both 22 and 24 passed. Layers 25-27 in *selection count* were not tested; 24 is the largest passing configuration tested, not a proven maximum. See `layer-screen.json` and `rejected28/` in the results directory for evidence.

For the next prefill investigation, keep this 24-layer configuration and ordinary Metal as baselines. Profile the remaining attention/state-update work and synchronization before choosing another offload boundary. Keep the existing accuracy gates and zero-decode-offload check; do not infer correctness or speedup from an isolated operator benchmark.
