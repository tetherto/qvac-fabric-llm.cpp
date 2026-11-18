# Qwen Core ML prefill sidecar

Opt-in Core ML sidecar for Qwen3.5-0.8B, 2B, 4B and 9B and Qwen3.8-27B prefill on Apple Silicon: part of each prefill FFN runs through Core ML while Metal runs the rest. The runtime requests CPU and Neural Engine compute units; Core ML selects operation placement.

## Command line

The sidecar is off by default. Build with `-DLLAMA_COREML=ON` (macOS 15 or later, Metal backend linked) and add `--coreml-ffn` (or set `LLAMA_COREML=1`) to `llama-server`, `llama-cli` or any other tool that uses `common`:

```sh
cmake -S . -B build -DLLAMA_COREML=ON
cmake --build build --target llama-server -j 8
build/bin/llama-server -m Qwen3.8-27B-Q4_0.gguf --coreml-ffn
```

The same `--coreml-ffn` flag enables all these models and selects the split automatically. For the supported Qwen3.5 shapes (0.8B and 2B with 24 layers, 4B and 9B with 32 layers), the Metal FFN channel count equals the model's hidden dimension, with one Core ML program per layer. Qwen3.8-27B keeps its existing hardware-dependent defaults:

| Model | Automatic split |
| --- | --- |
| Qwen3.5-0.8B | `1024x1` |
| Qwen3.5-2B | `2048x1` |
| Qwen3.5-4B | `2560x1` |
| Qwen3.5-9B | `4096x1` |
| Qwen3.8-27B | `8192x2` on chips with two Neural Engines (Ultra), or `11264x1` otherwise |

`--coreml-ffn-split METALxN` is an optional override: `METAL` is the number of intermediate FFN channels kept on Metal and `N` is the number of Core ML programs per layer. An explicit split takes precedence over automatic selection. The equivalent environment variables are `LLAMA_COREML=1` and the optional `LLAMA_ARG_COREML_FFN_SPLIT`. Unsupported models and splits log a configuration warning and keep inference on Metal.

Only prompt batches of 256 or more tokens use the sidecar; decode stays on Metal. Other models, other builds and unsupported contexts log a warning and run on Metal. The first start exports and compiles programs into `--coreml-ffn-cache` (default `~/Library/Caches/llama.cpp/coreml-ffn`, or `$LLAMA_CACHE/coreml-ffn`); size and startup cost depend on the model and split. For Qwen3.8-27B, this takes about 1-2 minutes and 4.5-9 GB; later starts load them in about 3 s.

For Qwen3.5-0.8B, 1024 intermediate FFN channels stay on Metal and the remaining 2560 run through one Core ML program per layer:

```sh
build/bin/llama-server -m Qwen3.5-0.8B-Q4_K_M.gguf --coreml-ffn
```

For Qwen3.5-2B, 2048 FFN channels stay on Metal and the remaining 4096 run through one Core ML program per layer:

```sh
build/bin/llama-server -m Qwen3.5-2B-Q4_K_M.gguf --coreml-ffn
```

For 2B, `--coreml-ffn-split 3072x1` and `--coreml-ffn-split 4096x1` remain supported overrides with smaller Core ML partitions. The hidden-dimension rule is specific to the supported Qwen3.5 models, not a general split formula for other model sizes.

For Qwen3.5-4B, 2560 FFN channels stay on Metal and the remaining 6656 run through one Core ML program per layer. `--coreml-ffn-split 3072x1` is a supported override with 6144 Core ML channels. For Qwen3.5-9B, 4096 FFN channels stay on Metal and the remaining 8192 run through one Core ML program per layer. The compiled cache is about 1.6 GB for 4B and 3.2 GB for 9B, and the same amount of memory is resident in addition to the GGUF. 9B is intended for macOS.

On a MacBook Air M2 (16 GB) with Q4_K_M models, `llama-perplexity` prefill over four 2048-token passes (mean of three alternating runs) went from 208.3 to 315.5 tok/s for 4B with `2560x1` (307.6 tok/s with `3072x1`), and from 114.7 to 180.4 tok/s for 9B. The mean KL divergence from Metal's logits was 4e-4 for 4B and 3e-4 for 9B.

On an M3 Ultra with `Qwen3.8-27B-Q4_0`, `llama-server` prefill of a 4372-token prompt went from 341.7 tok/s on Metal to 470.5 tok/s with the default split. The KL divergence from Metal's logits was 1e-5 to 7e-5.

## Library integration

Build `libllama` with `-DLLAMA_COREML=ON` to use the in-graph FFN sidecar from a native application or addon. That option also builds the EmbeddingGemma sidecar; `-DLLAMA_COREML_FFN=ON` alone builds only the FFN sidecar, and `-DLLAMA_COREML_FFN=OFF` keeps only the embedding one. It works with examples, tools and common utilities disabled. Each context must own its scheduler. The installed header is `llama-coreml-ffn.h`; the ordinary build keeps the API and reports the feature unavailable.

The runtime, exporter and templates live in `src/`, and CMake embeds the templates in the Apple library. Applications provide a writable cache directory, with no Python exporter, template directory or FFN environment variables. `src/llama-coreml-ffn-templates/make_template.py` regenerates the templates with coremltools when the program changes. The disk model source is recorded by `llama_model_load_from_file`; buffer-backed and split-file models are rejected. The source GGUF must remain unchanged and available while configuring the sidecar.

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

For a Qwen3.5-0.8B context, set `params.metal_channels = 1024` and `params.n_packages = 1` before calling `llama_set_coreml_ffn`. Its supported shape is 24 layers, 1024 hidden dimensions and 3584 intermediate FFN channels. The fixed Core ML program takes 512 rows and stores INT8 weights with FP16 computation. Applications must retain the GGUF weights because decode and ineligible batches use Metal. Apple frameworks built with `build-xcframework.sh` expose the configuration API through `llama-coreml-ffn.h`.

For a Qwen3.5-2B context, set `params.metal_channels = 2048` and `params.n_packages = 1`. Its supported shape is 24 layers, 2048 hidden dimensions and 6144 intermediate FFN channels. Metal partitions of 3072 or 4096 channels also use one package. These templates take 512 rows and store INT8 weights with FP16 computation. For a Qwen3.5-4B context (32 layers, 2560 hidden dimensions, 9216 FFN channels), set `params.metal_channels = 2560` or `3072`; for Qwen3.5-9B (32 layers, 4096 hidden dimensions, 12288 FFN channels), set `params.metal_channels = 4096`. Both use `params.n_packages = 1`. Automatic model-based selection is performed by the shared CLI initialization code; applications calling this API directly must set these smaller-model parameters explicitly because `llama_coreml_ffn_default_params` retains the 27B hardware defaults.

Keep Metal available if sidecar setup fails or its extra memory does not fit the application's budget. Check the configuration API's return value; a failed setup leaves the context's previous configuration intact. Resident sidecar models add memory alongside the GGUF and Metal buffers. An operating-system memory termination cannot be recovered through this API. A prediction error fails the request; callers must discard that request's KV state before retrying.

Qwen3.8-27B supports the 64-layer, 5120-hidden, 17408-FFN shape. `llama_coreml_ffn_default_params` picks 8192 Metal channels and two 4608-channel Core ML programs on chips with more than 16 Neural Engine cores, and 11264 Metal channels and one 6144-channel program otherwise, with a minimum of 256 rows. Each package must use one of these two widths. All supported models require FFN weights on `MTL0`, with no external weight scales, Hadamard transforms or per-weight precision overrides. Single-token decode and smaller batches keep the normal FFN. LoRA adapters, training and eval callbacks also keep the normal FFN; configuration rejects training and eval callbacks. This is a model-specific sidecar, not a general ggml Core ML backend.

To regenerate the weight-free 0.8B template, use coremltools 9.0 and NumPy 1.26.4 in a separate Python environment. The output directory must not exist:

```sh
python src/llama-coreml-ffn-templates/make_template.py /tmp/int8-h1024-c2560 --hidden 1024 --channels 2560 --chunk 2048
```

Copy its `Manifest.json`, `model.mlmodel` and `layout.txt` into `src/llama-coreml-ffn-templates/int8-h1024-c2560/`. The runtime supplies weights from the GGUF and compiles each layer on first use; later loads reuse the compiled cache.

For the 2B `2048x1` split, generate the `int8-h2048-c4096` template with `--hidden 2048 --channels 4096 --chunk 2048`. The `3072x1` and `4096x1` splits use `int8-h2048-c3072` and `int8-h2048-c2048`, respectively, with the matching `--channels` value. The 4B `2560x1` and `3072x1` splits use `int8-h2560-c6656` and `int8-h2560-c6144` (`--hidden 2560`), and the 9B `4096x1` split uses `int8-h4096-c8192` (`--hidden 4096 --channels 8192`).

Each context owns its models, serial worker, events and pending calls. Two contexts may be used independently; calls on one context must follow the existing caller-side synchronization contract. Configuration waits for outstanding graph work, invalidates graphs containing old runtime pointers, and replaces the old sidecar only after successful setup. Cache entries are shared on disk, and concurrent compilation uses separate staging paths. Compiled weights are keyed by the source path, size and modification time, the template program and layout, and the channel range. Do not modify the GGUF while configuring it.

Each chunk starts a prediction with shared Metal input/output buffers. An identity custom op holds the input alive and inserts the GPU wait, then ordinary graph nodes add the Core ML partials. There are no hidden join graphs or cached bindings to scheduler allocations. Prediction errors are checked before encode/decode returns success; an error fails only that request, whose KV state must be discarded, and later requests use the sidecar again.

`test-coreml-ffn`, `test-coreml-ffn-export` and `test-metal-custom` cover the public error contracts, template and quantization layout, cache invalidation, and asynchronous Metal custom ops with independent workers.
