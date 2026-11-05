# Qwen3.8-27B Core ML prefill sidecar

Opt-in Core ML sidecar for Qwen3.8-27B prefill on Apple Silicon: part of each prefill FFN runs on the Neural Engines while Metal runs the rest.

## Command line

The sidecar is off by default. Build with `-DLLAMA_COREML=ON` (macOS 15 or later, Metal backend linked) and add `--coreml-ffn` (or set `LLAMA_COREML=1`) to `llama-server`, `llama-cli` or any other tool that uses `common`:

```sh
cmake -S . -B build -DLLAMA_COREML=ON
cmake --build build --target llama-server -j 8
build/bin/llama-server -m Qwen3.8-27B-Q4_0.gguf --coreml-ffn
```

The first start compiles 4.5-9 GB of Core ML programs into `--coreml-ffn-cache` (default `~/Library/Caches/llama.cpp/coreml-ffn`, or `$LLAMA_CACHE/coreml-ffn`) in about 1-2 minutes; later starts load them in about 3 s. Only prompt batches of 256 or more tokens use the Neural Engine; decode stays on Metal. Other models, other builds and unsupported contexts log a warning and run on Metal. The default split is `8192x2` on chips with two Neural Engines (Ultra) and `11264x1` otherwise; `--coreml-ffn-split` overrides it.

On an M3 Ultra with `Qwen3.8-27B-Q4_0`, `llama-server` prefill of a 4372-token prompt went from 341.7 tok/s on Metal to 470.5 tok/s with the default split. The KL divergence from Metal's logits was 1e-5 to 7e-5.

## Library integration

Build `libllama` with `-DLLAMA_COREML=ON` to use the in-graph FFN sidecar from a native application or addon. That option also builds the EmbeddingGemma sidecar; `-DLLAMA_COREML_FFN=ON` alone builds only the FFN sidecar, and `-DLLAMA_COREML_FFN=OFF` keeps only the embedding one. It works with examples, tools and common utilities disabled. Each context must own its scheduler. The installed header is `llama-coreml-ffn.h`; the ordinary build keeps the API and reports the feature unavailable.

The runtime, exporter and templates live in `src/`, and CMake embeds both templates in the Apple library. Applications provide a writable cache directory, with no Python exporter, template directory or FFN environment variables. `src/llama-coreml-ffn-templates/make_template.py` regenerates the templates with coremltools when the program changes. The disk model source is recorded by `llama_model_load_from_file`; buffer-backed and split-file models are rejected. The source GGUF must remain unchanged and available while configuring the sidecar.

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

The initial implementation supports the 64-layer, 5120-hidden, 17408-FFN Qwen3.8-27B shape with all FFN weights on `MTL0`, no external weight scales, Hadamard transforms or per-weight precision overrides. `llama_coreml_ffn_default_params` picks 8192 Metal channels and two 4608-channel Core ML programs on chips with more than 16 Neural Engine cores, and 11264 Metal channels and one 6144-channel program otherwise, with a minimum of 256 rows. Each package must use one of these two widths. Single-token decode and smaller batches keep the normal FFN. LoRA adapters, training and eval callbacks also keep the normal FFN; configuration rejects training and eval callbacks. This is a model-specific sidecar, not a general ggml Core ML backend.

Each context owns its models, serial worker, events and pending calls. Two contexts may be used independently; calls on one context must follow the existing caller-side synchronization contract. Configuration waits for outstanding graph work, invalidates graphs containing old runtime pointers, and replaces the old sidecar only after successful setup. Cache entries are shared on disk, and concurrent compilation uses separate staging paths. Compiled weights are keyed by the source path, size and modification time, the template program and layout, and the channel range. Do not modify the GGUF while configuring it.

Each chunk starts a prediction with shared Metal input/output buffers. An identity custom op holds the input alive and inserts the GPU wait, then ordinary graph nodes add the Core ML partials. There are no hidden join graphs or cached bindings to scheduler allocations. Prediction errors are checked before encode/decode returns success; an error fails only that request, whose KV state must be discarded, and later requests use the sidecar again.

`test-coreml-ffn`, `test-coreml-ffn-export` and `test-metal-custom` cover the public error contracts, template and quantization layout, cache invalidation, and asynchronous Metal custom ops with independent workers.
