# Core ML EmbeddingGemma 2 support

Optional Core ML acceleration for the existing EmbeddingGemma 2 embedding API, CLI, and server. The runtime is built into `llama`; the exporter, benchmark, and compute-plan tools live in this directory. Requires macOS 15 / iOS 18 and Apple silicon for ANE acceleration.

The package already contains the fixed-shape graph and overflow-safe FP16 RMSNorm. The adapter reuses model objects and input buffers, memory maps the original BF16 token table, chooses the smallest fixed function that fits, and packs adjacent short texts into 256 tokens, using the package's eight or sixteen output slots. Attention blocks and positions are independent for each packed text. A single-text bin uses its smaller fixed function. Output order matches input order. Calls on one context are serialized. Set `llama_coreml_embedding_set_max_inflight(ctx, 8)` to pipeline predictions asynchronously on shared models within an encode call; the default is one synchronous prediction at a time. Each pending request owns reusable input buffers. Encode waits for all callbacks before returning, including on error. Separate contexts remain available for independent callers. Core ML selects CPU/ANE placement; selecting CPU + ANE does not prove that every operation runs on ANE.

## Build on Apple silicon

```sh
cmake -S . -B build-ane -DCMAKE_BUILD_TYPE=Release -DLLAMA_COREML=ON
cmake --build build-ane --target llama llama-embedding llama-server llama-coreml-embedding-bench -j 8
```

The option is off by default. Core ML support remains in the installed `llama` library when examples, common utilities, and tools are disabled. Use `llama.h` for the existing API integration or the installed `llama-coreml.h` for the lower-level adapter. Feature-off builds retain the configuration API and report Core ML unavailable.

## Existing embedding API, CLI, and server

Load a matching GGUF normally, create an embedding context with mean pooling, then configure the sidecar before calling `llama_encode()` or `llama_decode()`:

```cpp
if (!llama_set_coreml_model(ctx, "../embeddinggemma-2-coreml", 8)) {
    // The existing ggml path remains available.
}
llama_encode(ctx, batch);
const float * embedding = llama_get_embeddings_seq(ctx, seq_id);
```

The sidecar must come from the same checkpoint as the GGUF. Returned vectors are raw, mean-pooled embeddings, preserving the existing API's normalization contract. Callers can apply their usual normalization or Matryoshka truncation. Pass a null directory to disable the sidecar; a failed configuration leaves the previous configuration unchanged. `llama_supports_coreml()` reports build-time availability. Context calls retain the usual caller-side synchronization requirements.

The CLI and server accept `--coreml-model DIR` (or `LLAMA_ARG_COREML_MODEL`). Set `--coreml-max-inflight N` (or `LLAMA_ARG_COREML_MAX_INFLIGHT`) to 1..8 predictions; the default is one. They retain their existing output formats, tokenization, and `--embd-normalize` behavior:

```sh
build-ane/bin/llama-embedding -m ../embeddinggemma-2-BF16.gguf \
    --coreml-model ../embeddinggemma-2-coreml --pooling mean \
    -p 'title: none | text: Example document' --embd-normalize 2
build-ane/bin/llama-server -m ../embeddinggemma-2-BF16.gguf \
    --embedding --pooling mean --coreml-model ../embeddinggemma-2-coreml --coreml-max-inflight 8 \
    -np 32 -c 16384 -b 1024 -ub 1024
```

For short-text throughput, tune `-np` together with `-b` and `-ub`. The server default of four parallel sequences often produces only one packed prediction per encode, leaving asynchronous slots idle. Larger parallelism allows multiple predictions to overlap when enough requests are queued; benchmark the memory and latency tradeoff on the target device.

Setup rejects non-mean pooling and RoPE/YaRN or attention settings that differ from the sidecar. Compatible batches contain independent token sequences starting at position zero, at most 512 tokens each, with mean pooling. Longer sequences, shared token rows, supplied embeddings, position offsets, adapters, callbacks, and feature extraction use ggml. A prediction failure disables that context's sidecar and retries the batch with ggml. Keep the GGUF weights available for fallback; this optimization does not eliminate their memory use.

This integration requires an `embedding_raw` output. Re-export older normalized-only packages with the exporter below. The pinned FluidInference package remains usable through the lower-level normalized adapter; the existing embedding API rejects it and the CLI/server continue with ggml.

## Model assets

Download from [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml), revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`:

- `EmbeddingGemma2Text.mlpackage/` (all files)
- `embeddings.bf16`

Pass their containing directory to `llama_coreml_embedding_load`. The first load compiles the package and copies the result to `EmbeddingGemma2Text.mlmodelc` in that directory. The directory must be writable for this initial step. When replacing the `.mlpackage` or switching checkpoint revisions, stop callers and delete the adjacent `EmbeddingGemma2Text.mlmodelc` before loading again. An existing compiled cache is reused without checking the package; the compatibility probe is not a cache invalidation mechanism. Precompiled assets can instead be supplied in a read-only directory. The integrated API loads and validates all seven functions at setup and compares a short fixed text against ggml with a minimum cosine of 0.998. Setup can take minutes on first use. This numerical probe detects incompatible output but does not prove weight identity; matching checkpoints remain the caller's responsibility. The lower-level normalized adapter still loads functions lazily. Warm the workload before measuring throughput.

The token table must have exactly 262144 rows of 512 little-endian BF16 values. It is converted to FP16 and scaled by sqrt(512) during input preparation. This is the original token table, not a GGUF quantized table. Model weights and tokenizer must come from matching checkpoints. The adapter checks file size and model input/output contracts; it does not authenticate downloaded files. Pin revisions and verify checksums when distributing assets.

## Export from a local checkpoint

`export_coreml.py` converts the text encoder from the original Hugging Face safetensors checkpoint. Each function provides `embedding` (L2-normalized FP16) and `embedding_raw` (unnormalized FP32 mean pooling). It writes `EmbeddingGemma2Text.mlpackage`, the unscaled original `embeddings.bf16`, `tokenizer.json`, `source-config.json`, and an `export.json` manifest with source hashes, tool versions, and function shapes. Audio and vision weights are not loaded. The destination must not exist; conversion failures remove temporary output.

Use a separate Python 3.11 environment on an Apple silicon Mac. The requirements pin Core ML Tools 9, its supported PyTorch 2.7 release, and a Transformers revision with EmbeddingGemma 2 support. Do not combine these requirements with the repository's GGUF conversion environment.

```sh
python3.11 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install -r examples/embedding-coreml/requirements.txt

# Accept the source model's access terms and authenticate with Hugging Face first.
.venv-coreml/bin/hf download google/embeddinggemma-2 \
    --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
    --include config.json tokenizer.json '*.safetensors' '*.safetensors.index.json' \
    --local-dir ../embeddinggemma-2-hf

.venv-coreml/bin/python examples/embedding-coreml/export_coreml.py \
    ../embeddinggemma-2-hf ../embeddinggemma-2-coreml --pack-slots 8
```

The exporter also accepts a compatible local fine-tuned checkpoint with the same text architecture. It uses no remote Python code and supports single-file or sharded safetensors. The six single-text functions use lengths 32/48/64/128/256/512; `pack_256` has eight output slots by default. Pass `--pack-slots 16` to export sixteen slots directly. RMSNorm and final L2 normalization rescale before squaring to avoid FP16 overflow. Rotary tables are computed in FP32 before conversion; the packed function looks up positions that restart for each text.

Export traces the text encoder on CPU and converts the graphs; it does not execute Core ML predictions or prove Neural Engine placement. Allow substantial memory, time, and temporary disk space for seven intermediate packages. Core ML Tools merges them with shared weights into the final package. This is an independent export, not a byte-identical reproduction of FluidInference's package.

Validate a new export on the target Mac before using it: build the benchmark above, convert the same source checkpoint to a BF16 GGUF in the normal GGUF conversion environment, then compare the two paths. Include short texts, lengths crossing every bucket, and enough short texts to fill all packing slots. The benchmark checks cosine agreement and normalization. For raw-output exports, it also verifies the existing encode/decode API, raw vector magnitudes, sequence ID mapping, disabling the sidecar, and ggml fallback. Run the compute-plan tool to inspect anticipated device placement; ANE placement is not guaranteed by export alone.

```sh
# Use the existing GGUF conversion environment, not .venv-coreml.
python convert_hf_to_gguf.py ../embeddinggemma-2-hf \
    --outtype bf16 --outfile ../embeddinggemma-2-BF16.gguf
build-ane/bin/llama-coreml-embedding-bench \
    ../embeddinggemma-2-BF16.gguf ../embeddinggemma-2-coreml \
    examples/embedding-coreml/benchmark-texts.txt 5 metal 1 32 1024 8

# These checks use metadata, temporary files, and synthetic normalization tensors.
# They do not load model weights or run the encoder.
python3 -m unittest discover -s examples/embedding-coreml -p 'test_export_coreml.py'

# C++ packing, grouping, and context contract checks need no model assets.
ctest --test-dir build-ane -R '^coreml-contract$' --output-on-failure
```

BF16 GGUF provides a reference close to the original checkpoint. Also validate the official Q8_0 GGUF with the same corpus and batching arguments when deploying that quantization. Core ML uses the original checkpoint through its FP16 sidecar, independently of GGUF quantization; the raw output is exposed as FP32. The selected GGUF controls ggml execution and fallback.

The adapter compiles `.mlpackage` to `.mlmodelc` on first load. Use `--pack-slots 16` for a sixteen-slot local export.

## Lower-level adapter and tokenization

Reuse llama.cpp's GGUF vocabulary with `llama_model_params.vocab_only = true`. Apply the task prefix before tokenization, for example `title: none | text: ` for documents or `task: search result | query: ` for queries. Tokenize with `add_special=false` and `parse_special=false`, then explicitly prepend BOS (2) and append EOS (1). Do not rely on the GGUF's generation defaults to insert EOS.

Pass an array of `llama_coreml_embedding_sequence` to `llama_coreml_embedding_encode` with packing enabled. Each sequence includes its BOS/EOS and must contain 1..512 tokens. The adapter rejects overlength inputs; the caller owns any truncation policy and must preserve EOS. The default `llama_coreml_embedding_load` produces 768 floats per text, already mean-pooled and L2-normalized by the model. `llama_coreml_embedding_load_raw` instead selects unnormalized mean-pooled output from a compatible local export. For Matryoshka output, keep the first 512/256/128 components and renormalize. Errors return through a caller-owned buffer; no C++ exceptions cross the C API. A failure may leave partial output, which the caller must discard.

## Accuracy and throughput

Use the BF16 model from [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), revision `bfcd298762cc34d0357ece5ebdd31791a3a374d8`. The benchmark reads one already-prefixed text per line, uses the same llama.cpp token IDs for both paths, warms the models, compares packed versus single embeddings and Core ML versus GGUF cosine similarity, tests rejected token IDs/lengths/buffer capacities, then measures repeated inference. Model loading and first-use compilation are excluded. Tokenization is excluded from throughput for all paths. CPU/Metal sequence count and token batch capacity are configurable.

```sh
build-ane/bin/llama-coreml-embedding-bench \
    ../llama-coreml-assets/gguf/embeddinggemma-2-BF16.gguf \
    ../llama-coreml-assets/coreml examples/embedding-coreml/benchmark-texts.txt 5 metal 1 32 1024 8
# Repeat with 64 sequences / 2048 tokens to compare a larger Metal batch.
# Replace metal with cpu for the CPU comparison.
```

The optional arguments after the execution path are Core ML workers (1..8, default 1), reference parallel sequences (1..64, default 8), reference token batch capacity (32..2048, default 512), and Core ML calls in flight (1..8, default 1). Values above one worker add a concurrent Core ML throughput measurement. Each context owns its input buffers; the model package and mapped token table remain the same. The benchmark exits with an error when any cosine similarity is below 0.998 or normalized outputs are non-finite or not L2 normalized. With a raw-output package it additionally checks raw relative L2 error below 3% and reports `coreml_integrated` throughput through the normal API. A normalized-only package reports `integrated_api=unavailable` and tests only the lower-level adapter. `benchmark-texts.txt` supplies a small multilingual/code smoke corpus. Include lengths crossing 32/48/64/128/256/512 token boundaries when validating a deployment. This is a smoke/accuracy check, not an embedding quality evaluation. Measure your real corpus and target hardware before claiming an application speedup.

Build `llama-coreml-embedding-plan` and run it with the compiled `.mlmodelc` directory to report Core ML's preferred device for every non-constant operation of each function. Constants are excluded because they do not execute and have no device usage entry. A compute plan describes anticipated placement; actual execution can be inspected with Core ML profiling in Instruments.

The runtime input contract was checked against [FluidUse's EmbeddingGemma2Manager](https://github.com/FluidInference/FluidUse/blob/main/Sources/FluidUse/EmbeddingGemma2/EmbeddingGemma2Manager.swift) (Apache-2.0). Audio and vision encoders are outside this adapter's scope.
