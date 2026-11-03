# EmbeddingGemma 2 text validation

Use `llama-embedding` and `scripts/snapdragon/embeddinggemma2_eval.py` to check text embeddings before selecting a Hexagon configuration. These commands do not imply device validation has passed. Store actual results with the source revision and model revisions used.

## Build

Build a host CPU reference from the repository root:

```bash
cmake -S . -B build-eg2-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=OFF \
  -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_MTMD=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-eg2-host --parallel 8 \
  --target llama-embedding test-backend-ops test-export-graph-ops
```

For Android, use the [Snapdragon build instructions](README.md), which build the CPU, OpenCL and Hexagon backends and the v73/v75/v79/v81 HTP libraries. The standard Docker build is:

```bash
python3 scripts/snapdragon/build.py --target android
```

With existing local SDK installations, this equivalent native configuration avoids downloading a container. Replace the SDK paths with the installations on your host:

```bash
TASK_NDK=/path/to/android-ndk-r28b
TASK_HEXAGON=/path/to/Hexagon_SDK/6.6.0.0
TASK_OPENCL=/path/to/opencl-toolchain
cmake -S . -B build-eg2-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TASK_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DPREBUILT_LIB_DIR=android_aarch64 \
  -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
  -DGGML_HEXAGON=ON -DGGML_OPENCL=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DHEXAGON_SDK_ROOT="$TASK_HEXAGON" \
  -DHEXAGON_TOOLS_ROOT="$TASK_HEXAGON/tools/HEXAGON_Tools/19.0.07" \
  -DOpenCL_INCLUDE_DIR="$TASK_OPENCL/OpenCL-Headers" \
  -DOpenCL_LIBRARY="$TASK_OPENCL/opencl-device/libOpenCL.so" \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_TESTS_INSTALL=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_MTMD=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-eg2-android --parallel 8
cmake --install build-eg2-android --prefix pkg-eg2-android/llama.cpp
```

## Validate

Use pinned GGUF checkpoints, for example the text BF16 and Q8_0 files from `ggml-org/embeddinggemma-2-GGUF`. Supply their exact repository revisions. The driver hashes each model, binary and corpus. It uses the model tokenizer and preserves the task prefixes in the input fixture. It does not convert, quantize or download models.

The Q8_0 checkpoint retains `per_layer_model_proj.weight` in BF16. Hexagon caches an F16 copy of supported BF16 matrix weights on first use and runs their matrix multiplication through the existing F16 kernels. The GGUF and original BF16 bytes remain unchanged, including backend readback. Tensor uploads and buffer clears invalidate the cached copy. This avoids the CPU fallback for the shared per-layer projection without an offline model conversion.

This path accepts contiguous, non-view, two-dimensional BF16 leaf weights with an inner dimension divisible by 32, stored in a Hexagon device buffer, with F32 activations and output. Other BF16 layouts and operations retain their fallback behavior. Computation uses F16 range and precision. The cache adds one F16 copy of each accepted BF16 matrix plus alignment; the EmbeddingGemma 2 projection adds 12 MiB. Check numerical parity for the target model rather than assuming general BF16 arithmetic equivalence.

The path is enabled by default (`GGML_HEXAGON_BF16=1`). Set `GGML_HEXAGON_BF16=0` before starting the process to restore BF16 matrix multiplication on CPU without allocating F16 mirrors. With the Snapdragon runner, use `--hex-bf16 0`. The switch applies to eligible BF16 weights in any model; models with many such weights can require substantially more mirror memory than this Q8_0 checkpoint.

The existing backend test executable covers chunked and strided uploads, exact BF16 readback, cached-weight updates and graph replay for both small and packed inputs:

```bash
test-backend-ops -b HTP0 -o MUL_MAT_BF16_UPLOAD
```

Example GGUF artifacts are available at repository revision `58faf90940886df2d62be555c8765d00dac362c5`:

| Precision | File bytes | SHA-256 |
| --- | ---: | --- |
| BF16 | 557950176 | `68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216` |
| Q8_0 | 309855456 | `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135` |

```bash
python3 scripts/snapdragon/embeddinggemma2_eval.py \
  --binary build-eg2-host/bin/llama-embedding \
  --model /path/to/embeddinggemma-2-Q8_0.gguf \
  --model-revision 58faf90940886df2d62be555c8765d00dac362c5 \
  --reference-model /path/to/embeddinggemma-2-BF16.gguf \
  --reference-model-revision 58faf90940886df2d62be555c8765d00dac362c5 \
  --device none --output /tmp/eg2-host-results
```

Each run creates a new results directory containing `results.json` and stdout/stderr for each invocation. An existing output directory is rejected so earlier measurements are preserved. A failed execution or failed acceptance check returns a nonzero status. High-precision differences are reported separately; use `--reference-min-cosine` to apply an explicitly chosen accuracy threshold. The packing threshold is fixed at a minimum cosine of `0.998`; the separate same-quantization CPU parity threshold defaults to `0.998`.

The small fixture exercises independent texts, changed neighbors, reordered outputs, mixed lengths, a long input, repeated loads and partial batches. Every invocation checks finite, nonempty, 768-dimensional vectors and unit-norm error at most `1e-4`. Repeated passes compare their vectors with the first pass outside the timed region. Two simple retrieval rankings are sanity checks, not a retrieval-quality evaluation. Token counts are recorded; the fixture does not guarantee an exact token count at any particular window boundary. Add a pinned task corpus with `--corpus` for a representative evaluation. `--expected-dimensions` allows explicit use with another embedding model.

The driver rounds requested context/batch/ubatch capacities up to multiples of 256 and records the requested and effective sizes. Partial-batch validation requires multiple actual decode batches and a final batch below the effective capacity.

For a provisioned Android device, push the package and models before running the driver. Use a task-specific device directory and the intended ADB server and serial. The driver does not connect, root, configure, install or push to a device; it checks that the device binary and model SHA-256 hashes match the local files:

```bash
python3 scripts/snapdragon/embeddinggemma2_eval.py \
  --binary pkg-eg2-android/llama.cpp/bin/llama-embedding \
  --model /path/to/embeddinggemma-2-Q8_0.gguf \
  --reference-model /path/to/embeddinggemma-2-BF16.gguf \
  --adb DEVICE_SERIAL \
  --remote-dir /data/local/tmp/embeddinggemma2 \
  --device HTP0 --output /tmp/eg2-htp-results
```

ADB uses server port `5037` by default; override it with `--adb-port` for a custom server. Default remote paths are `REMOTE_DIR/bin/llama-embedding`, `REMOTE_DIR/lib`, and `REMOTE_DIR/models/LOCAL_FILENAME`. Override model or executable locations with `--remote-model`, `--remote-reference-model` and `--remote-binary`. The libraries must match the binary build. The driver uses `-p` to pass its corpus, without writing prompt files on the device.

Choose `--device GPUOpenCL` for the Adreno comparison, or the exact name shown by `llama-embedding --list-devices`. CPU reference invocations always use `--device none -ngl 0 --no-op-offload --no-kv-offload -fa off`. These are full-model CPU baselines. For per-op reference-kernel comparisons, use the existing `test-backend-ops` harness, which enables CPU reference kernels internally:

```bash
./bin/test-backend-ops test -b HTP0 -o GELU,GELU_QUICK,GEGLU,GEGLU_ERF,GEGLU_QUICK
./bin/test-backend-ops test -b HTP0 -o FLASH_ATTN_EXT
```

Run those commands from the device package directory with `LD_LIBRARY_PATH=./lib`, `ADSP_LIBRARY_PATH=./lib` and `GGML_HEXAGON_DEVICES=HTP0`. Filter attention shapes with `-p` when investigating a specific failure. Model-shaped cases can also be exported with `test-export-graph-ops` and supplied with `test-backend-ops --test-file`.

## Timing and evidence

The embedding example adds opt-in environment variables. For example, run `LLAMA_EMBED_BENCH_REPETITIONS=5 LLAMA_EMBED_BENCH_WARMUPS=1 llama-embedding -m MODEL ...`. The original command-line arguments are passed to the common parser unchanged, including its Windows UTF-8 handling:

```text
LLAMA_EMBED_BENCH_REPETITIONS  measured corpus passes, 1..1000
LLAMA_EMBED_BENCH_WARMUPS      prior corpus passes, 1..1000, default 1
```

The repetition variable enables benchmarking; the warmup variable requires it. Values must contain only decimal digits in the stated range. It prints vectors once in the requested format and emits one `EMBEDDING_BENCH {JSON}` line on stderr. Default execution and output formats are unchanged. All benchmark passes reuse the loaded model, context and tokenized input. Each timed pass includes batch assembly, memory clearing, decode, pooled-output retrieval and normalization; getters synchronize the backend before timing stops. Tokenization, vector printing and vector validation are outside the timer.

`model_context_init_ms` measures model/context initialization, after backend initialization. The benchmark disables the common empty-input warmup. `first_pass_ms` measures the first corpus pass on that loaded context; it does not establish a cold OS filesystem cache or cold device. `pass_ms` contains the warmed corpus passes. Throughput is total measured texts divided by total measured pass time. Corpus-pass latency is reported separately from amortized throughput; it is not single-query latency when more than one text is packed. The Python `process_wall_ms` includes startup, load, tokenization, all passes, output and shutdown.

The driver uses sequential passes with one context. It does not measure shared-model asynchronous prediction, multiple concurrent contexts or multiple-process throughput. The fixture reloads the model between invocations to exercise the lifecycle.

HTP runs include a separate untimed diagnostic invocation with scheduling, verbose and per-op profiling enabled. The common empty-input warmup is disabled so the log reflects the explicit corpus. Acceptance requires at least one HTP model-compute profile record, such as `MUL_MAT`, `FLASH_ATTN_EXT` or `GELU`, with strictly positive cycles, saved in `htp-diagnostic.stderr`. OPBATCH/control-only records and zero-cycle records fail this gate. Initialization or offloaded-layer messages alone are insufficient evidence of accelerator execution. Summarize the diagnostic log with:

```bash
python3 scripts/snapdragon/ggml-hexagon-profile.py /tmp/eg2-htp-results/htp-diagnostic.stderr --top 30
```

Timing invocations disable those diagnostic settings. Record device build, SoC, toolchain, source changes and runtime settings with the results. Use multiple independent driver runs and a representative sustained workload before making performance claims. Peak memory, thermal behavior and power/energy require separate device telemetry; this driver does not report them. Record any telemetry method, sampling interval, idle subtraction and measurement limitations with the task results.

The existing Snapdragon workflow builds these tools and can run QDC device tests. Its current Llama 3.2 smoke tests do not establish EmbeddingGemma 2 correctness. A model-dependent embedding check needs an explicit pinned model/corpus and runtime budget; a green build without a device run is not runtime evidence.
