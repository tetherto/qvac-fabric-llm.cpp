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

The existing backend test executable covers chunked and strided uploads, exact BF16 readback, cached-weight updates and graph replay for both small and packed inputs:

```bash
test-backend-ops -b HTP0 -o MUL_MAT_BF16_UPLOAD
```

The task's selected artifacts are pinned to repository revision `58faf90940886df2d62be555c8765d00dac362c5`:

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

The driver rounds requested context/batch/ubatch capacities up to multiples of 256 and records both requested and effective batch sizes. This respects the current embedding example's default 256-sequence graph reservation. An unaligned capacity of 1011 was observed to abort during CPU context initialization: graph reservation rounded its synthetic token count to 1024, retained 1011 output rows, and then constructed incompatible mean-pooling matrices. This is an existing reserve-graph limitation, separate from Hexagon execution. Alignment avoids that unsupported configuration; it does not fix the engine limitation. The original failing logs should be retained with the task evidence. Partial validation still requires multiple actual decode batches and a final batch below the effective capacity.

For a provisioned Android device, push the package and models before running the driver. Use a task-specific device directory and the intended ADB server and serial. The driver does not connect, root, configure, install or push to a device; it checks that the device binary and model SHA-256 hashes match the local files:

```bash
python3 scripts/snapdragon/embeddinggemma2_eval.py \
  --binary pkg-eg2-android/llama.cpp/bin/llama-embedding \
  --model /path/to/embeddinggemma-2-Q8_0.gguf \
  --reference-model /path/to/embeddinggemma-2-BF16.gguf \
  --adb DEVICE_SERIAL --adb-port 15045 \
  --remote-dir /data/local/tmp/embeddinggemma2 \
  --device HTP0 --output /tmp/eg2-htp-results
```

Default remote paths are `REMOTE_DIR/bin/llama-embedding`, `REMOTE_DIR/lib`, and `REMOTE_DIR/models/LOCAL_FILENAME`. Override model or executable locations with `--remote-model`, `--remote-reference-model` and `--remote-binary`. The libraries must match the binary build. The driver uses `-p` to pass its corpus, without writing prompt files on the device.

Choose `--device GPUOpenCL` for the Adreno comparison, or the exact name shown by `llama-embedding --list-devices`. CPU reference invocations always use `--device none -ngl 0 --no-op-offload --no-kv-offload -fa off`. These are full-model CPU baselines. For per-op reference-kernel comparisons, use the existing `test-backend-ops` harness, which enables CPU reference kernels internally:

```bash
./bin/test-backend-ops test -b HTP0 -o GELU,GELU_QUICK,GEGLU,GEGLU_ERF,GEGLU_QUICK
./bin/test-backend-ops test -b HTP0 -o FLASH_ATTN_EXT
```

Run those commands from the device package directory with `LD_LIBRARY_PATH=./lib`, `ADSP_LIBRARY_PATH=./lib` and `GGML_HEXAGON_DEVICES=HTP0`. Filter attention shapes with `-p` when investigating a specific failure. Model-shaped cases can also be exported with `test-export-graph-ops` and supplied with `test-backend-ops --test-file`.

The F32 CPU reference for QUICK uses the scalar formula without lookup-table rounding. The optimized unary F32 path retains FP16 input indices but stores F32 output values, adding a 256 KiB table. FP16 unary and gated QUICK paths retain their original table and output behavior. Operator accuracy thresholds are unchanged.

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

## Measured Snapdragon configuration

Local development validation on 20 October 2026 used a CPH2723 phone with SM8750, Android 16, Adreno 830 and runtime-detected Hexagon v79. The build used engine base `370265fedc9c69a4ca2ebd80516bde34892e9f7d` with the GELU and CPU reference changes described above, NDK r28b, Hexagon SDK 6.6.0.0 and Hexagon tools 19.0.07. The pinned Q8_0 checkpoint above produced 768-dimensional, finite, normalized embeddings. This validates this target and development source; it does not establish other devices or a published Fabric release.

HTP packing, output order, changed neighbors and partial batches all exceeded the `0.998` gate; the minimum across these checks was `0.99999950`. Same-quantization CPU parity reached a minimum cosine of `0.99981295`, and the separate BF16 comparison reached `0.99983903`. A diagnostic run recorded 484 positive-cycle model-compute operations. The Adreno suite also passed, with a minimum packing/isolation/order/partial-batch cosine of `0.99990480` and CPU parity of `0.99976531`. The small retrieval checks passed on both backends. These are numerical and retrieval sanity checks, not a retrieval-quality benchmark.

The following throughput and latency tables measure the standalone raw `llama-embedding` executable, separately from the installed Java/JNI app probe, Fabric carrier and embedding addon. Throughput ranges cover two independent process runs in forward and reverse backend order. Each process measured five passes after one warmup pass; that warmup is the separately recorded first pass. All runs used six CPU threads, mean pooling and L2 normalization. CPU used no operation/KV offload and flash attention off; HTP and Adreno used flash attention on and 99 offloaded layers. The CPU column is this explicit reference configuration, not a claim of optimal CPU tuning.

| Workload and token capacity | CPU texts/s | HTP texts/s | Adreno texts/s |
| --- | ---: | ---: | ---: |
| Short texts, capacity 256 | 24.38-24.63 | 64.01-64.56 | 60.71-60.82 |
| Short texts, capacity 2048 | 6.22-10.23 | 62.93-64.97 | 79.62-80.32 |
| Mixed lengths, capacity 2048 | 1.75-1.84 | 8.61-11.29 | 4.82-5.19 |

The short workload contains seven unique multilingual/code entries repeated sixteen times: 112 texts and 2,832 tokens per pass. The mixed workload contains eight entries and 1,188 tokens, including one 1,011-token input. These small synthetic fixtures are not representative production corpora. All final benchmark parity and repeatability checks passed. Earlier full correctness-suite timings used a different CPU library and are excluded from this comparison.

For one 18-token text at capacity 256, two independent processes each measured ten warm passes:

| Measurement | CPU | HTP | Adreno |
| --- | ---: | ---: | ---: |
| Warm single-query median, ms | 19.49-19.79 | 17.94-17.96 | 45.38-50.24 |
| First query on loaded context, ms | 27.08-37.07 | 31.94-32.18 | 3910.69-3935.00 |
| Model/context initialization, ms | 755.33-786.47 | 894.14-923.09 | 1585.20-1600.95 |

The first-query values include work deferred until first use, such as backend compilation or repacking; their individual components were not isolated. Initialization excludes backend initialization. No run establishes a cold OS filesystem cache. The timed query excludes tokenization and startup. Query and throughput timings use the device's C++ steady clock. The separately recorded host `process_wall_ms` does not consistently enclose the device intervals: the sustained run reports 165.49 seconds of device computation versus 162.82 seconds of host process time, before initialization is added. The clock disagreement's cause was not isolated. Keep host wall values as diagnostics; do not combine these clocks or use the host values as corroborating end-to-end latency.

A finite sustained HTP run at capacity 256 measured 90 passes of the short workload: 10,080 texts in 165.49 seconds, or 60.91 texts/s. The median corpus pass rose from 1,568.30 ms for the first ten passes to 2,027.95 ms for the last ten, about 29% longer. Numerical parity and every pass's repeatability check still passed. This is sustained variation on this phone, not a thermal-equilibrium result or proof of its cause.

Read-only HAL temperature and task-process memory snapshots were collected approximately every ten seconds. During that HTP run, the vendor-reported skin sensor ranged from 38.61 to 58.34 C and battery temperature from 36.2 to 38.7 C. Sampled process VmHWM peaked at 624,436 KiB (about 610 MiB); it does not account for all DSP/GPU memory. A separate 31-minute mixed-workload session sampled process peaks up to 2,895,664 KiB and a skin-sensor maximum of 57.66 C, but those peaks were not attributed to an individual backend. Device-reported HAL severity remained zero; that does not establish the absence of clock throttling. No root, governor or thermal overrides were used.

Reliable integrated power/energy telemetry was unavailable: direct battery sysfs current/voltage reads were denied, and the exposed HAL/battery values were not a calibrated power measurement. No energy-per-text estimate or energy comparison is claimed.

For this short fixture, capacity 256 retained HTP throughput while increasing CPU throughput; larger packing favored Adreno throughput. Inputs longer than 256 tokens need a larger aligned capacity. Select settings again with a representative corpus. Multiple loaded addon instances sharing an HTP device require serialized, synchronized execution because the backend shares mutable session state; do not assume independent raw backend handles are thread-safe or claim asynchronous scaling from these sequential measurements. The embedding addon documents its bounded operation gate and initialization/destruction requirements separately.

## Hexagon BF16 weight acceleration

The original Q8_0 GGUF retains its shared per-layer projection in BF16. Cache an F16 mirror in the Hexagon device buffer and use the existing F16 matrix kernels, while preserving the original BF16 bytes for readback and CPU copies. Uploads and clears invalidate the cache. The path is limited to contiguous 2D BF16 leaf matrices with F32 activations/output; it adds 12 MiB for this projection and uses F16 arithmetic range/precision.

Measurements on 21 October 2026 used the same SM8750 phone and unchanged pinned GGUF, six CPU threads, capacity 256, and 112 texts / 2,832 tokens per pass. HTP/OpenCL used flash attention; the CPU reference disabled offload and flash attention. Each short process used one warmup and five timed passes, with two before/after HTP processes in opposite orders. Times exclude model loading, warmup, tokenization and printing.

| Configuration | Mean seconds per 112 texts | Texts/s |
| --- | ---: | ---: |
| CPU | 3.549 | 31.56 |
| HTP before this fix | 1.545 | 72.51 |
| HTP with BF16 weight cache | 0.562 | 199.28 |
| OpenCL GPU | 1.118 | 100.21 |

The warm short comparison reduced HTP encode time by 63.6% (2.75x throughput).

Separate 180-second runs after one warmup:

| Backend | Mean seconds per 112 texts | Encode texts/s | Completed passes |
| --- | ---: | ---: | ---: |
| HTP with BF16 weight cache | 0.571 | 196.09 | 315 |
| OpenCL GPU | 1.841 | 60.82 | 98 |

HTP: first ten passes 198.11 texts/s; last ten 194.23 texts/s (98.0% retained).

OpenCL: first ten passes 62.82 texts/s; last ten 60.18 texts/s (95.8% retained).

All accepted runs kept the screen off with a bounded Android partial wake lock preventing CPU suspend for every configuration. The lock was verified before and after each run; paired BOOTTIME/MONOTONIC checks showed no detectable phone suspend. No governor or thermal controls were changed. Earlier sleep-affected or display-state-changing attempts were excluded. Initial temperatures were unmatched, so these are descriptive throughput measurements and do not establish a thermal or energy advantage.

Validation: all 11 model checks passed; minimum packing/order/isolation/partial-batch cosine 0.99999863, and same-Q8 CPU cosine 0.99981335 (gate 0.998). The matching 112-text trace records 24 positive-cycle HMX projection operations and no projection CPU rejection.

Operator checks passed: 4 cache/upload/readback regressions, 55 BF16 matrix cases and 100 F16/Q8_0 controls. Unsupported layouts were skipped rather than counted as executed checks. Host and Android builds passed. Existing addon probes passed 20 sequential checks, 6 concurrency/recovery checks and 3 cancellation/recovery checks.
