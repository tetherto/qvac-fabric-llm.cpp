# Core ML text embedding adapter

Optional C API for EmbeddingGemma 2 text models exported below or published by FluidInference. Intended for Fabric integration after its llama.cpp rebase. It is a separate full-model execution path, not a ggml backend or a change to `llama_decode` or the server. Requires macOS 15 / iOS 18 and Apple silicon for ANE acceleration.

The package already contains the fixed-shape graph and overflow-safe FP16 RMSNorm. The adapter reuses model objects and input buffers, memory maps the original BF16 token table, chooses the smallest fixed function that fits, and packs adjacent short texts into 256 tokens, using the package's eight or sixteen output slots. Attention blocks and positions are independent for each packed text. A single-text bin uses its smaller fixed function. Output order matches input order. Calls on one context are serialized. Set `llama_coreml_embedding_set_max_inflight(ctx, 8)` to pipeline predictions asynchronously on shared models within an encode call; the default is one synchronous prediction at a time. Each pending request owns reusable input buffers. Encode waits for all callbacks before returning, including on error. Separate contexts remain available for independent callers. Core ML selects CPU/ANE placement; selecting CPU + ANE does not prove that every operation runs on ANE.

## Build on Mac Studio

```sh
cmake -S . -B build-ane -DCMAKE_BUILD_TYPE=Release -DLLAMA_COREML_EMBEDDING=ON
cmake --build build-ane --target llama-coreml-embedding-bench -j 12
```

The option is off by default. `llama-embedding-coreml` is a static target that also builds when `LLAMA_BUILD_EXAMPLES=OFF` and `LLAMA_BUILD_COMMON=OFF`. Link it into Fabric and include `embedding-coreml.h`; no changes to llama's public ABI are needed.

## Model assets

Download from [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml), revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`:

- `EmbeddingGemma2Text.mlpackage/` (all files)
- `embeddings.bf16`

Pass their containing directory to `llama_coreml_embedding_load`. The first load compiles the package and copies the result to `EmbeddingGemma2Text.mlmodelc` in that directory. The directory must be writable for this initial step. Precompiled assets can instead be supplied in a read-only directory. Functions load lazily; the first inference using a function may spend minutes compiling for the device. Warm all functions required by the workload before timing.

The token table must have exactly 262144 rows of 512 little-endian BF16 values. It is converted to FP16 and scaled by sqrt(512) during input preparation. This is the original token table, not a GGUF quantized table. Model weights and tokenizer must come from matching checkpoints. The adapter checks file size and model input/output contracts; it does not authenticate downloaded files. Pin revisions and verify checksums when distributing assets.

## Export from a local checkpoint

`export_coreml.py` converts the text encoder from the original Hugging Face safetensors checkpoint. It writes `EmbeddingGemma2Text.mlpackage`, the unscaled original `embeddings.bf16`, `tokenizer.json`, `source-config.json`, and an `export.json` manifest with source hashes, tool versions, and function shapes. Audio and vision weights are not loaded. The destination must not exist; conversion failures remove temporary output.

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

Export traces the text encoder on CPU and converts the graphs; it does not execute Core ML predictions or prove Neural Engine placement. Allow substantial memory, time, and temporary disk space for seven intermediate packages. Core ML Tools merges them with shared weights into the final package. This is an independent export, not a byte-identical reproduction of FluidInference's package. The historical benchmarks below apply to the downloaded package, not to freshly exported graphs.

Validate a new export on the target Mac before using it: build the benchmark above, convert the same source checkpoint to a BF16 GGUF in the normal GGUF conversion environment, then compare the two paths. Include short texts, lengths crossing every bucket, and enough short texts to fill all packing slots. The benchmark checks cosine agreement and normalization. Run the compute-plan tool to inspect anticipated device placement; ANE placement is not guaranteed by export alone.

```sh
# Use the existing GGUF conversion environment, not .venv-coreml.
python convert_hf_to_gguf.py ../embeddinggemma-2-hf \
    --outtype bf16 --outfile ../embeddinggemma-2-BF16.gguf
build-ane/bin/llama-coreml-embedding-bench \
    ../embeddinggemma-2-BF16.gguf ../embeddinggemma-2-coreml \
    examples/embedding-coreml/benchmark-texts.txt 5 metal 1 8 512 8

# These checks use metadata, temporary files, and synthetic normalization tensors.
# They do not load model weights or run the encoder.
python3 -m unittest discover -s examples/embedding-coreml -p 'test_export_coreml.py'
```

The adapter compiles `.mlpackage` to `.mlmodelc` on first load. `expand_pack16.py` is only for the pinned downloaded FluidInference graph and rejects independently exported graphs; use `--pack-slots 16` for a local export. `palettize_pack.py` can produce a separate compressed variant, which requires its own accuracy validation.

## Tokenization and integration

Reuse llama.cpp's GGUF vocabulary with `llama_model_params.vocab_only = true`. Apply the task prefix before tokenization, for example `title: none | text: ` for documents or `task: search result | query: ` for queries. Tokenize with `add_special=false` and `parse_special=false`, then explicitly prepend BOS (2) and append EOS (1). Do not rely on the GGUF's generation defaults to insert EOS.

Pass an array of `llama_coreml_embedding_sequence` to `llama_coreml_embedding_encode` with packing enabled. Each sequence includes its BOS/EOS and must contain 1..512 tokens. The adapter rejects overlength inputs; the caller owns any truncation policy and must preserve EOS. It produces 768 floats per text, already mean-pooled and L2-normalized by the model. For Matryoshka output, keep the first 512/256/128 components and renormalize. Errors return through a caller-owned buffer; no C++ exceptions cross the C API. A failure may leave partial output, which the caller must discard.

## Accuracy and throughput

Use the BF16 model from [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), revision `bfcd298762cc34d0357ece5ebdd31791a3a374d8`. The benchmark reads one already-prefixed text per line, uses the same llama.cpp token IDs for both paths, warms the models, compares packed versus single embeddings and Core ML versus GGUF cosine similarity, tests rejected token IDs/lengths/buffer capacities, then measures repeated inference. Model loading and first-use compilation are excluded. Tokenization is excluded from throughput for all paths. CPU/Metal sequence count and token batch capacity are configurable.

```sh
build-ane/bin/llama-coreml-embedding-bench \
    ../llama-coreml-assets/gguf/embeddinggemma-2-BF16.gguf \
    ../llama-coreml-assets/coreml examples/embedding-coreml/benchmark-texts.txt 10 metal
# Repeat with cpu as the last argument for the CPU comparison.
```

The optional arguments after the execution path are Core ML workers (1..8, default 1), reference parallel sequences (1..64, default 8), reference token batch capacity (32..2048, default 512), and Core ML calls in flight (1..8, default 1). Values above one worker add a concurrent Core ML throughput measurement. Each context owns its input buffers; the model package and mapped token table remain the same. The benchmark exits with an error when any cosine similarity is below 0.998 or outputs are non-finite or not L2 normalized. `benchmark-texts.txt` supplies a small multilingual/code smoke corpus. Include lengths crossing 32/48/64/128/256/512 token boundaries when validating a deployment. This is a smoke/accuracy check, not an embedding quality evaluation. Measure your real corpus and target hardware before claiming an application speedup.

Build `llama-coreml-embedding-plan` and run it with the compiled `.mlmodelc` directory to report Core ML's preferred device for every non-constant operation of each function. Constants are excluded because they do not execute and have no device usage entry. A compute plan describes anticipated placement; actual execution can be inspected with Core ML profiling in Instruments.

The runtime input contract was checked against [FluidUse's EmbeddingGemma2Manager](https://github.com/FluidInference/FluidUse/blob/main/Sources/FluidUse/EmbeddingGemma2/EmbeddingGemma2Manager.swift) (Apache-2.0). Audio and vision encoders are outside this adapter's scope.

## Validation on Mac Studio

Measured on Apple M3 Ultra, macOS 15.7.9, Release build, llama.cpp `30f7df1`, October 13-14, 2026. Model revisions are pinned above. Throughput excludes loading, first-use compilation, and tokenization.

960 texts (the 12-line `benchmark-texts.txt` repeated 80 times), 18400 tokens, five repetitions:

| Path | Texts/second |
| --- | ---: |
| Core ML, one text per call | 334.48 |
| Core ML, packed, one context | 568.68 |
| Core ML, packed, four contexts | 572.74 |
| llama.cpp Metal, BF16, batched | 607.74 |

Packing improved Core ML throughput by 70%. Four contexts added less than 1% for this workload, so one context is the default. Metal was faster here; this result does not establish an ANE speedup over Metal. Core ML versus BF16 minimum cosine was 0.99994323, mean 0.99997082; packed versus single minimum cosine was 0.99999323.

A separate 12-text, 2014-token test crossing all six length buckets achieved 84.39 texts/second packed versus 13.70 on CPU (three repetitions). Minimum reference cosine was 0.99995606. L2 norms, output ordering, concurrent isolation, and rejected token IDs, overlength sequences and undersized output buffers passed.

Core ML's compute plan preferred ANE for all 3862 executable operations in each single-text function and all 3863 in `pack_256`, with no unknown device assignments. Constant nodes are excluded. This verifies the anticipated plan, not an Instruments trace of actual execution. The standalone adapter build with common utilities/examples/tools disabled and the default feature-off configuration also passed.

## Initial follow-up experiments before shared-model async support

All inference and conversion experiments ran on the same Mac Studio. Core ML was configured with CPU + ANE and excluded GPU. ggml Metal offloaded all transformer layers using the BF16 GGUF. ggml CPU was a separate reference backend; Core ML CPU-only and CPU + GPU were not benchmarked.

`expand_pack16.py` expands only the pooling head from eight to sixteen rows, preserving transformer weights and the six single-text functions. The runtime detects the slot count from model metadata. `palettize_pack.py` applies six- or eight-bit LUT weight compression to 218 matrix tensors in the packed function only. This is weight compression, not integer activation inference. Both scripts require coremltools 9 and write separate asset directories.

On the same 960-text corpus, five repetitions:

| Path or experiment | Texts/second | Reference sequences / token capacity |
| --- | ---: | --- |
| Core ML, sixteen slots, FP16 | 850.15 | Metal comparison: 16 / 256 |
| ggml Metal, BF16 | 874.75 | 16 / 256 |
| Core ML, sixteen slots, FP16 | 850.09 | Metal comparison: 16 / 512 |
| ggml Metal, BF16 | 1015.74 | 16 / 512 |
| Core ML, sixteen slots, FP16 | 850.00 | Metal comparison: 32 / 1024 |
| ggml Metal, BF16 | 1356.63 | 32 / 1024 |
| Core ML, sixteen slots, eight-bit palettes | 860.41 | Metal comparison: 32 / 1024 |
| ggml Metal, BF16, palette experiment reference | 1353.02 | 32 / 1024 |

The earlier 70% gain compares eight-slot Core ML packing (568.68) with single-text Core ML calls (334.48). Expanding to sixteen slots increased packed throughput by about 49.5%. Eight-bit palettes added about 1.2%, which is too small to establish a reliable speedup without repeated independent trials. Tuned Metal remained faster. Six-bit palettes failed the 0.998 cosine threshold and have no accepted throughput result.

For the eight-bit packed run, 400 calls took 63.694 ms in host input preparation, 5490.157 ms in Core ML prediction, and 24.727 ms in output extraction. Explicit host preparation and extraction were about 1.6% of measured time. Prediction includes Core ML overhead and device execution; this is not a direct ANE hardware timing. The sixteen-slot FP16 compute plan preferred ANE for all nonconstant executable operations, with no unknown assignments. Instruments hardware tracing and power measurements were not performed.

The sixteen-slot boundary test (12 texts, 2014 tokens, two repetitions) measured 84.20 texts/s on Core ML and 13.43 texts/s on ggml CPU. It is a different workload from the short-text Metal comparison.

Final direct packed-versus-BF16 validation gave minimum cosine 0.99972944 for eight-bit palettes on the 960-text corpus. Six-bit palettes gave 0.99623307 versus BF16 and 0.99615811 versus the original single-text model, failing the 0.998 threshold. A separate 128-text corpus exercised all sixteen output slots: packed-versus-BF16 minimum cosine was 0.99997141 for FP16 and 0.99985177 for eight-bit palettes. These are numerical parity checks, not downstream retrieval-quality benchmarks.

## Matched original Swift runtime comparison before adapter async support

On October 14, the pinned FluidUse runtime (`d278e17df85e39e8c21bb64a467da81c39576fcd`) and adapter ran in the same process on the M3 Ultra using the original eight-slot model and identical token IDs. Two process runs each rotated six modes over five passes on the 960-text corpus. Loading, compilation and warm-up were excluded. Tokenization was excluded except in the separately labeled public API mode.

| Mode | Run 1 texts/s | Run 2 texts/s |
| --- | ---: | ---: |
| Adapter, single | 324.83 | 325.06 |
| Swift, single | 322.10 | 321.74 |
| Adapter, packed eight slots | 569.29 | 568.79 |
| Swift, packed, one in-flight job | 566.80 | 566.17 |
| Swift, packed, eight in-flight jobs | 1144.62 | 1144.86 |
| Swift public API, including tokenization | 1162.54 | 1162.06 |

The sequential implementations differ by less than 0.5%; the original Swift runtime's shared-model asynchronous prediction doubled throughput. Eight independent synchronous adapter contexts reached only 575.56 texts/s in a fresh adapter benchmark; tuned Metal reached 1352.26 texts/s. Packed calls per pass were 120 for both implementations. Tokenization matched exactly; minimum packed-versus-single cosine was 0.99999323. The same-process wrapper copies adapter output into Swift arrays, adding a small wrapper cost.

The public 644 texts/s result uses M5 Pro/macOS 27 and another corpus. Its original benchmark harness was not located. These results use their published runtime with benchmark-only pretokenized entry points and one/eight selectable in-flight jobs; they do not reproduce the exact published experiment. Model input preparation, BF16 lookup, prediction and output extraction remain unchanged. The unused download entry point is disabled. Raw data, provenance and harness are under `reports/swift-coreml-comparison/`.

## Shared-model asynchronous prediction

`llama_coreml_embedding_set_max_inflight(ctx, n)` selects one synchronous prediction at a time (`n=1`, default), or 2..8 pending asynchronous predictions on each function's shared model. Each request slot owns its input buffers. Output rows remain in input order. The existing encode call remains blocking and drains all pending callbacks before return, including on an error. Timing statistics sum per-request times, so asynchronous prediction times overlap and must not be treated as a wall-time breakdown.

The benchmark's final optional argument is `max-inflight` (default 1). For eight asynchronous requests, use `... 5 metal 1 32 1024 8`; the `workers` argument controls independent contexts and is separate from predictions in flight.

A matched five-pass run of the original eight-slot model reached 1178.24 texts/s with the adapter and 1177.94 with the pinned Swift runtime, both using eight asynchronous calls in flight. A second run reached 1178.16 and 1178.00 respectively. This is throughput parity; the small difference is not a meaningful speed advantage. With sixteen-slot packing, the adapter reached 1766.31 texts/s in a separate five-pass run; that changes the packing capacity and is not the identical-model comparison. Per-pass cosine and normalization checks ran outside the timing interval.

Final-code matched eight-slot results were 1178.50 texts/s for the adapter and 1178.12 for Swift, confirming parity. Fresh five-pass sixteen-slot adapter runs reached 1766.68 / 1766.81 texts/s; Metal reached 1354.86 at 32 sequences / 1024 tokens and 1562.29 at 64 / 2048. The optimized adapter was about 13.1% faster than the largest tested Metal batch on the short-text corpus. Metal remained faster on the boundary corpus (156.71 versus 128.85 texts/s, two repetitions), so this is workload-dependent. Boundary and all-sixteen-slot async checks passed, as did two independent asynchronous contexts and rejected-input checks. Final eight-slot comparisons and validation logs are under `reports/swift-coreml-comparison/async*`.

The additional final comparison in Swift 6 language mode (matching the original package's tools version) reached 1178.16 texts/s for our eight-slot async adapter and 1178.01 for Swift. Initial process runs used Swift 5 compatibility mode with the same Swift 6.1.2 compiler; this final check confirms parity in Swift 6 mode. Data is in `async8-swift6-results.json`.
