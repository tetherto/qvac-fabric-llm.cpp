# Qwen3.8-27B Core ML prefill sidecar

## Command line

The sidecar is off by default. Build with `-DLLAMA_COREML_FFN=ON -DGGML_METAL=ON` (macOS 15 or later) and add `--coreml-ffn` to `llama-server`, `llama-cli` or any other tool that uses `common`:

```sh
cmake -S . -B build -DLLAMA_COREML_FFN=ON -DGGML_METAL=ON
cmake --build build --target llama-server -j 8
build/bin/llama-server -m Qwen3.8-27B-Q4_0.gguf --coreml-ffn
```

The first start compiles 4.5-9 GB of Core ML programs into `--coreml-ffn-cache` (default `~/Library/Caches/llama.cpp/coreml-ffn`, or `$LLAMA_CACHE/coreml-ffn`) in about 1-2 minutes; later starts load them in about 3 s. Only prompt batches of 256 or more tokens use the Neural Engine; decode stays on Metal. Other models, other builds and unsupported contexts log a warning and run on Metal. The default split is `8192x2` on chips with two Neural Engines (Ultra) and `11264x1` otherwise; `--coreml-ffn-split` overrides it.

## Library integration

Build `libllama` with `-DLLAMA_COREML_FFN=ON -DGGML_METAL=ON` to use the in-graph FFN sidecar from a native application or addon. This is separate from the EmbeddingGemma `LLAMA_COREML` option. It works with examples, tools and common utilities disabled. Each context must own its scheduler. The installed header is `llama-coreml-ffn.h`; the ordinary build keeps the API and reports the feature unavailable.

The runtime, exporter and templates now live in `src/`. CMake embeds both templates in the Apple library. Applications provide a writable cache directory, with no Python exporter, template directory or FFN environment variables. The disk model source is recorded by `llama_model_load_from_file`; buffer-backed and split-file models are rejected. The source GGUF must remain unchanged and available while configuring the sidecar.

```cpp
#include "llama-coreml-ffn.h"

// ctx is a normal Metal context for a Qwen3.8-27B GGUF.
auto params = llama_coreml_ffn_default_params();
params.cache_dir = "/application/writable/cache/coreml-ffn";
char error[512];
if (!llama_set_coreml_ffn(ctx, &params, error, sizeof(error))) {
    // Report error; ctx retains its previous configuration.
}
// llama_decode(ctx, batch) now builds the sidecar into eligible FFNs.
// Disable after synchronizing through the configuration API:
llama_set_coreml_ffn(ctx, nullptr, error, sizeof(error));
```

The initial implementation supports the 64-layer, 5120-hidden, 17408-FFN Qwen3.8-27B shape with all FFN weights on `MTL0`, no external weight scales, Hadamard transforms or per-weight precision overrides. Defaults are 8192 Metal channels, two 4608-channel Core ML programs and a minimum of 256 rows. The other bundled width is 6144 channels; each package must use one of these widths. Single-token decode and smaller batches keep the normal FFN. LoRA adapters, training and eval callbacks also keep the normal FFN; configuration rejects training and eval callbacks. This is a model-specific sidecar, not a general ggml Core ML backend.

Each context owns its models, serial worker, events and pending calls. Two contexts may be used independently; calls on one context must follow the existing caller-side synchronization contract. Configuration waits for outstanding graph work, invalidates graphs containing old runtime pointers, and replaces the old sidecar only after successful setup. Cache entries are shared on disk, and concurrent compilation uses separate staging paths. Compiled weights are keyed by the source path, size and modification time, the template program and layout, and the channel range. Do not modify the GGUF while configuring it.

Each chunk starts a prediction with shared Metal input/output buffers. An identity custom op holds the input alive and inserts the GPU wait, then ordinary graph nodes add the Core ML partials. There are no hidden join graphs or cached bindings to scheduler allocations. Prediction errors are checked before encode/decode returns success; an error fails the request, whose sidecar must be disabled and KV state discarded before retrying.

A small application using only the public library API is built as `llama-coreml-ffn` when examples are enabled:

```sh
cmake -S . -B build-coreml -DLLAMA_COREML_FFN=ON -DGGML_METAL=ON
cmake --build build-coreml --target llama-coreml-ffn -j 8
build-coreml/bin/llama-coreml-ffn model.gguf /application/writable/cache "Your prompt"
```

`test-coreml-ffn`, `test-coreml-ffn-export` and `test-metal-custom` cover the public error contracts, template and quantization layout, cache invalidation, and asynchronous Metal custom ops with independent workers. The library refactor still needs a full-model quality and throughput run against the experiment on the target Mac before release. The measurements below describe the original experiment, not new measurements of the library integration.

## Original experiment


Opt-in Core ML sidecar for Qwen3.8-27B prefill on Apple Silicon: part of each prefill FFN runs on the Neural Engines while Metal runs the rest. Normal builds do not enable the hooks; build from `examples/ffn-coreml` (below).

The recommended setup is the in-graph sidecar with load-time export, see [Load-time export and the in-graph sidecar](#load-time-export-and-the-in-graph-sidecar): no conversion step, any microbatch size, multi-sequence batches and speculative decoding. The other sections document the asset-based runtime it grew from, a 24-layer experiment from `temp-11259` (`be6e387b9`). The recorded runs and experiment notes are on branch `feat/coreml-qwen-3.8` (`examples/ffn-coreml/results`, `RESULTS.md`, `EXPERIMENTS.md`).

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
PROMPTS=/path/to/prompts  # code.txt, prose.txt: plain-text prompts
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

The 8192-channel split on the Q4_0 GGUF uses [qwen3.8-27b-64-dual-q40.json](configs/qwen3.8-27b-64-dual-q40.json) (the exporter checks the model SHA256).

Because the Neural Engines no longer hold the GPU up, the overlap pairs with a larger Neural Engine share: [qwen3.8-27b-64-dual-s5120.json](configs/qwen3.8-27b-64-dual-s5120.json) keeps 5120 channels on Metal and gives 6144 to each Neural Engine. On M3 Ultra with the Q4_0 GGUF and the prefill attention kernel from `feat/metal-fa-prefill-d256`, single-request server prefill (engine-bench client) measured 519.4 / 537.6 / 507.4 / 454.9 / 379.1 / 294.4 tok/s at about 1k / 4.5k / 21k / 38k / 70k / 123k tokens, against 395.8 / 435.9 / 400.0 / 365.9 / 316.5 / 255.6 for the 8192-channel split without the overlap. The held-out gate passes: mean NLL change +0.00044 nats/token (upper bound +0.00083), KL 0.00046, worst window +0.00211. Without the overlap this split is slower than the 8192-channel one, so keep the two together.

## Load-time export and the in-graph sidecar

`FFN_COREML_OTF=<Metal channels>x<packages>` (for example `8192x2`) builds the Core ML side from the GGUF when the model loads, so there is no conversion step and no assets folder. A weight-free int8 template per channel count ([templates](../../src/llama-coreml-ffn-templates), made by `make_template.py` with coremltools) holds the program. For each layer and package the runtime dequantizes the FFN weights from the GGUF, quantizes them per channel to int8 the way coremltools does (the `weight.bin` is byte-identical to a coremltools export), writes the package, compiles it and caches the compiled model under `FFN_COREML_CACHE` (default `~/Library/Caches/ffn-coreml`). The cache key is the GGUF path, size and modification time and the template. Set `FFN_COREML_TEMPLATES` to the templates folder. On M3 Ultra with the Q4_0 GGUF, a cold start of the 128 programs takes 92 s (export 24.5 s, compile 2.6 s, load 60 s); a cached start takes 3-7 s. `llama-server-coreml` takes `FFN_COREML_OTF` in place of `FFN_COREML_ASSETS`.

`FFN_COREML_INGRAPH=1` (with `FFN_COREML_GPU_RESIDENT=1 FFN_COREML_ASYNC=1`) puts the sidecar into the model graph. The Metal share of each FFN runs as ordinary graph nodes on views of the model's own weights, so there is no weight copy and there are no per-layer prefix and join graphs. Two custom nodes, run by the Metal backend (`ggml_backend_metal_set_custom`), bracket it: start signals the serial worker to run the Neural Engine share on the node's input, finish waits for it and adds the partial outputs. The GPU runs the Metal share while the Neural Engines work. A microbatch of more than 512 rows goes to the Neural Engines in 512-row calls; a last chunk of at least `FFN_COREML_PAD_MIN` rows is padded, a smaller one has its Neural Engine share computed on Metal. Any microbatch size, multi-sequence batches and speculative decoding take this path, which the half-batch overlap does not.

Harness prefill (`split`, Q4_0, M3 Ultra, 8192 Metal channels and 2 x 4608, without the attention kernel, 3 repetitions):

| Microbatch | Metal 2048 / 8192 tokens | In-graph 2048 / 8192 tokens |
| --- | ---: | ---: |
| 512 | 331.7 / 320.4 | 447.0 / 427.8 (+35% / +34%) |
| 1024 | 340.4 / 328.7 | 480.1 / 455.6 (+41% / +39%) |
| 2048 | 344.8 / 333.3 | 496.1 / 473.4 (+44% / +42%) |

Programs with 1024 rows were slower than 512-row chunks (485.1 / 459.0 at microbatch 2048) and are not kept. Peak process footprint was 6.2-6.8 GB, against 18.0 GB for the runtime with its own Metal weight copies and 1.4-1.8 GB for Metal alone. Quality on held-out text: over 13 repository windows mean NLL change -0.00006 nats/token and KL 0.00017; at microbatch 2048 the logits are identical to microbatch 512; partial chunks (a padded 300-row microbatch, a 100-row Metal tail, a padded chunk inside a 1024-row microbatch) had mean KL 0.00009-0.00010.

```sh
cmake -S examples/ffn-coreml -B build-ffn-coreml -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DFFN_COREML_DECODER_EXPERIMENT=ON -DFFN_COREML_SERVER=ON
cmake --build build-ffn-coreml --target llama-server-coreml -j 8

FFN_COREML_OTF=8192x2 FFN_COREML_TEMPLATES=src/llama-coreml-ffn-templates FFN_COREML_INGRAPH=1 \
FFN_COREML_GPU_RESIDENT=1 FFN_COREML_ASYNC=1 FFN_COREML_SINGLE_ENCODER=1 \
  build-ffn-coreml/llama-server-coreml -m Qwen3.8-27B-Q4_0.gguf -ngl 99 -fa on -ub 2048
```

Through `llama-server-coreml` with the attention kernel from `feat/metal-fa-prefill-d256` and `-ub 2048`, single-request prefill (engine-bench client) measured 420 / 448 / 429 tok/s at about 1k / 4.4k / 21k tokens, against about 331-335 for Metal alone; 431-449 tok/s with MTP and DFlash2, where the half-batch overlap stays off (about 365); and 148 tok/s total at 16 concurrent requests. In no-speculation, MTP and DFlash2 modes the first 64 generated tokens were identical to Metal alone on three prompts of 784 to 2322 tokens, with the same draft acceptance.

## Held-out quality gate

The gate compares the split against ordinary Metal on 23 windows of real text that were never used for calibration or layer selection: 20 of 2048 tokens and 3 of 8192 (C++, Python, documentation, a novel and mathematics, 65,536 tokens). [gate/windows.json](gate/windows.json) lists every window as a slice of a source with its SHA256. Repository sources are read at the recorded commit; the five external texts (Project Gutenberg and Wikipedia wikitext, not committed) must be downloaded into one directory first.

```sh
python3 examples/ffn-coreml/gate/build_windows.py /path/to/external /tmp/heldout
examples/ffn-coreml/gate/run_gate.sh build-ffn-coreml/llama-coreml-decoder-bench "$MODEL" "$ASSETS" /tmp/heldout /tmp/gate-out
```

`run_gate.sh` runs each window in `quality` mode and in `quality-metal` mode (the all-Metal control through the same boundary), then `analyze_gate.py` writes `gate.json`. It passes when the mean NLL change is at most +0.002 nats/token with a 95% bootstrap upper bound at most +0.005, every domain is at most +0.003 with KL at most max(0.003, 2x control), no window exceeds +0.005, the overall KL is at most max(0.002, 2x control), and every run has exact offload counts and finite outputs.

## Sidecar assets

`ASSETS/layers.txt` lists selected IDs. Each `ASSETS/layerN/ffn-512.mlpackage` contains that layer's suffix. Weights and compiled models are not committed.

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

Callbacks and builder state are process-global. The harness drives contexts serially. Do not enable this as a multi-request/server backend without lifecycle, concurrency, cancellation, cache validation, and fallback work.

All remaining layers were screened: 25 new individual candidates were found; the combined 28-layer selection failed (mean KL 0.001127598). Both 22 and 24 passed. Layers 25-27 in *selection count* were not tested; 24 is the largest passing configuration tested, not a proven maximum. The evidence (`layer-screen.json`, `rejected28/`) is in the results directory on `feat/coreml-qwen-3.8`.

For the next prefill investigation, keep this 24-layer configuration and ordinary Metal as baselines. Profile the remaining attention/state-update work and synchronization before choosing another offload boundary. Keep the existing accuracy gates and zero-decode-offload check; do not infer correctness or speedup from an isolated operator benchmark.
