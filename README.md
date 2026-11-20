<div align="center">

<h1>qvac-fabric-llm.cpp</h1>

<p><strong>AI inference and training engine for desktop and mobile platforms.</strong></p>

<p>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
<a href="https://github.com/ggml-org/llama.cpp"><img src="https://img.shields.io/badge/based%20on-llama.cpp%20b10796-orange.svg" alt="Based on llama.cpp"></a>
</p>

</div>

`qvac-fabric-llm.cpp` is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for embedded systems, mobile devices, and enterprise deployment scenarios. It extends the excellent foundation of llama.cpp with additional capabilities focused on memory-based model loading, mobile GPU optimization, and flexible integration patterns.

---

## Key Features

The following capabilities are developed and maintained as part of qvac-fabric-llm.cpp.

### Cluster Inference

Run inference across GPUs on multiple machines using RPC. TCP is supported by default; **RDMA is Linux-only**, with compatible RoCEv2 adapters and `libibverbs`, and is negotiated automatically.

Build with `-DGGML_RPC=ON` and the GPU backend for each host. For RDMA, also set `-DGGML_RPC_RDMA=ON` on the Linux client and servers, with `libibverbs` installed (see the [RPC Guide](tools/rpc/README.md)). Start one server per GPU on a private network:

```bash
# Run on GPU host 1 and GPU host 2, respectively
./build/bin/ggml-rpc-server -H 192.168.88.10 -p 50052 --device CUDA0
./build/bin/ggml-rpc-server -H 192.168.88.11 -p 50052 --device CUDA0
```

#### Pipeline Parallelism

Split layers across GPUs and overlap prompt microbatches. Use one endpoint per stage and a batch larger than the microbatch. On the main host:

```bash
./build/bin/llama-cli -m model.gguf --rpc 192.168.88.10:50052,192.168.88.11:50052 \
    --device RPC0,RPC1 --split-mode layer -ngl all -b 2048 -ub 256
```

#### Tensor Parallelism

Split work within each layer across GPUs to accelerate decoding for supported models. Fast network links help because the GPUs exchange results frequently. Using the same servers:

```bash
./build/bin/llama-cli -m model.gguf --rpc 192.168.88.10:50052,192.168.88.11:50052 \
    --device RPC0,RPC1 --split-mode tensor -ngl all
```

#### Direct All-Reduce and RDMA Transport

Tensor split across a power-of-two number of RPC devices (2, 4, 8, 16) uses a direct recursive-doubling (butterfly) all-reduce between the servers, so partial results no longer pass through the main host. Asynchronous RPC transfers and events (protocol 113) let layer split overlap microbatches across stages. Shared builds load `libibverbs` at runtime, so hosts without RDMA fall back to TCP automatically. All servers and the client must run the same protocol version; the direct communicator is for trusted private networks only. See the [RPC Guide](tools/rpc/README.md).

### LoRA Fine-Tuning

`qvac-fabric-llm.cpp` provides native [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) fine-tuning across CPU, Vulkan, and Metal backends. The training pipeline runs directly on consumer hardware, including mobile phones and integrated GPUs.

- Multi-backend GPU training on CPU, Vulkan (NVIDIA, AMD, Intel, Mali, Adreno), Metal (Apple) and CUDA
- Backward kernels for GELU/GEGLU/SIGLU, L2-norm, MoE `mul_mat_id`, flash attention, SSM-conv and gated-delta-net, plus quantized `out_prod` on Vulkan
- On-device fine-tuning from iOS via the `llama.swiftui` example (`FinetuneBridge`)
- FP32, FP16, Q8, and Q4 training paths
- Supervised instruction-tuning via assistant-only masked loss (SFT)
- Checkpoint saving, optimizer-state save/load, resumable training and stop requests (`llama_opt_*` / `llama_lora_*` C API)
- Learning rate schedulers (constant, cosine, linear) with warmup support
- LoRA adapter merging into base models via **llama-export-lora**
- Verified compatibility with Qwen3, Qwen3.5 and Qwen3.6 (dense and MoE), Gemma3, and Gemma4 (dense and MoE) model architectures

For usage details and CLI reference, see the [Finetuning Guide](examples/training/README.md).

### TurboVec / Local Vector Search

`ggml-vector-index` provides a standalone C API for local vector search, including compressed CPU indexes for applications that need on-device retrieval.

- **Storage**: full-precision f32, generic q8 and packed q4, plus TurboVec 2-, 3- and 4-bit modes (a byte-exact port of [turbovec](https://github.com/ryancodrai/turbovec) 1.1.2, dimensions up to 16384) with scalar, NEON, and AVX2 scoring paths.
- **Search**: exact top-k scans, filtering by caller-provided IDs, reusable prepared filters, and optional IVF search for approximate candidate selection.
- **Persistence**: index snapshots, with read-only mmap loading and incremental mutation logs for the generic storage modes. TurboVec writes turbovec's own v8 files, in memory or on disk, and saves incrementally with crash-safe `sync`.

The library is disabled by default. Enable it with `-DGGML_VECTOR_INDEX=ON` and link `ggml::vector-index` explicitly; it is not integrated into the llama runtime or server. See the [Vector Index Guide](docs/vector-index.md) for supported dimensions, API usage, persistence contracts, and benchmarks.

### Apple Neural Engine via Core ML *(exclusive)*

Opt-in Core ML sidecars run part of the workload on the Apple Neural Engine (ANE) while Metal runs the rest, freeing GPU time on Apple silicon. Build with `-DLLAMA_COREML=ON` (macOS 15 / iOS 18 or later, Metal backend). If a model, build or context is unsupported, the sidecar logs a warning and the model runs on Metal as usual.

- **Qwen3.8-27B prefill** (`--coreml-ffn`): part of each prefill FFN runs on the Neural Engines, and decode stays on Metal. On an M3 Ultra with `Qwen3.8-27B-Q4_0`, prefill of a 4372-token prompt rose from 341.7 to 470.5 tok/s, with KL divergence from Metal's logits of 1e-5 to 7e-5.
  - Only prompt batches of 256 or more tokens use the Neural Engine.
  - The split between Metal and Core ML adapts to the chip (two Neural Engines on Ultra parts, one elsewhere) and can be overridden with `--coreml-ffn-split`.
  - The Core ML programs (4.5-9 GB) compile on first start in about 1-2 minutes into `--coreml-ffn-cache`; later starts load them in about 3 s.
  - It currently supports the Qwen3.8-27B shape (64 layers, 5120 hidden, 17408 FFN).
- **EmbeddingGemma 2 embeddings** (`--coreml-model DIR`): ANE acceleration for the embedding API, CLI and server. Short texts are packed into fixed 256-token windows, predictions can be pipelined asynchronously (`--coreml-max-inflight 1..8`), and output order and normalization match the regular path. An exporter, a compute-plan tool and a benchmark are in [examples/embedding-coreml](examples/embedding-coreml/README.md).
- **Library integration**: `llama_supports_coreml()`, `llama_set_coreml_model()` and `llama_set_coreml_ffn()` configure the sidecars from native apps and addons. Each context owns its Core ML models and worker, and a failed configuration leaves the previous one unchanged. Applications only provide a writable cache directory; no Python exporter is needed at run time. `-DLLAMA_COREML_FFN=ON/OFF` selects only the FFN or only the embedding sidecar.

```bash
cmake -B build -DLLAMA_COREML=ON
cmake --build build --target llama-server
build/bin/llama-server -m Qwen3.8-27B-Q4_0.gguf --coreml-ffn
```

See [docs/coreml-ffn.md](docs/coreml-ffn.md) for the full guide.

### AMD XDNA (NPU) Backend *(exclusive)*

A ggml backend that runs part of a graph on the AMD XDNA2 NPU (Ryzen AI: Strix Point, Strix Halo, Krackan) through XRT. It is built around Qwen3.5-style models with gated-delta-net layers, where single-token decode runs almost entirely on the NPU array. The EmbeddingGemma 2 embedding model also runs, checked against the CPU by `test-xdna-embedding`. See [ggml/src/ggml-xdna/README.md](ggml/src/ggml-xdna/README.md) for build, environment variables and troubleshooting.

- **Matrix multiplication**: BF16, F16, Q4_K, Q5_K and Q6_K weights with F32 activations. Decode GEMV keeps weights quantized in memory and applies group parameters on the accumulator. Small batches use a decode GEMM and larger prefill batches use a prefill GEMM, with an int8 path for K-quants.
- **Fused decode layer**: one design holds the whole recurrent layer of a decode token (conv, norm, gated delta net, gated epilogue, projections and the FFN transition), with the recurrent state kept on the device between tokens.
- **Prefill ops**: gated-delta-net and flash-attention kernels run on the array above size thresholds. Below them the host is faster and cheaper per token, so the backend leaves them there.
- **Wider models**: besides the 0.8B, the fused decode layers run Qwen3.5 2B, 4B and 9B (residual widths up to 4096), by default for the quantizations they cover. On a Ryzen AI MAX+ 395 their Q4_K_M decode takes 36-44% less package energy per token above idle than the host decode, at 41-47% of its speed; `GGML_XDNA_FUSED_WIDE=0` restores the faster host decode. A model wider than 1024 whose layers are not Qwen3.5's keeps the host decode.
- **llama-server**: with the default four slots, requests that run one at a time decode their attention on the NPU. While two requests overlap, their attention runs on the host, and the later request's decode attention stays there until it ends. `-np 1` keeps every request's decode attention on the NPU.
- **Packaging**: it can be built as a loadable module (`GGML_BACKEND_DL`), with the kernel artifacts installed next to the library.

Requires Linux with an NPU2 device (`/dev/accel/accel0`), XRT 2.25.x under `/opt/xilinx/xrt`, `uuid-dev`, and a Python environment with mlir-aie (IRON) 1.4.3 to compile the kernels at build time. Enable with `-DGGML_XDNA=ON`. All llama contexts in a process share one backend context and take turns on the array.

### BitNet Inference and Fine-Tuning

Native support for [BitNet](https://arxiv.org/abs/2402.17764) ternary quantized models via the TQ2_0 data type, enabling efficient inference and LoRA fine-tuning of models such as [bitnet_b1_58-xl](https://huggingface.co/gianni-cor/bitnet_b1_58-xl-TQ2_0) on resource-constrained devices.

The official [microsoft/BitNet](https://github.com/microsoft/BitNet) inference framework provides optimized CPU kernels and GPU support limited to CUDA. `qvac-fabric-llm.cpp` **extends** BitNet to all major GPU backends -- Vulkan, Metal, and CPU -- bringing cross-platform GPU-accelerated BitNet inference and on-device fine-tuning to hardware not covered by the upstream framework, including Apple Silicon, mobile GPUs (Adreno, Mali), and AMD/Intel discrete GPUs. Compatible with models such as [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).

- **Backends**: Vulkan, Metal and CPU TQ2_0 quantization support
- **Training**: LoRA fine-tuning of BitNet models on Vulkan, Metal, and CPU backends
- **Conversion**: HuggingFace-to-GGUF conversion for BitNet model architectures
- Cooperative matrix (coopmat) support for Vulkan devices that expose the extension

### TurboQuant KV Cache Quantization

TurboQuant adds low-bit KV-cache quantization formats for long-context inference while preserving token-generation quality close to higher-bit caches. It supports:

- **TBQ3_0 / TBQ4_0**: TurboQuant 3-bit and 4-bit formats with QJL Stage 2 correction.
- **PQ3_0 / PQ4_0**: PolarQuant 3-bit and 4-bit Stage 1-only formats.
- **64-wide variants**: `tbq3_0_64`, `tbq4_0_64`, `pq3_0_64`, and `pq4_0_64` for models with smaller attention head dimensions.
- **Backend support**: CPU quantization/dequantization and Vulkan inference kernels, including attention paths and mixed K/V cache configurations. CUDA and Metal do not include TurboQuant kernels in this release.
- **Test coverage**: performance coverage via [test-kv-cache-quantization-perf.sh](tests/test-kv-cache-quantization-perf.sh), perplexity coverage via [test-kv-cache-quantization-perp.sh](tests/test-kv-cache-quantization-perp.sh), and eval coverage via [test-kv-cache-ruler.py](tests/test-kv-cache-ruler.py), [test-kv-cache-niah.py](tests/test-kv-cache-niah.py), [test-kv-cache-longbench.py](tests/test-kv-cache-longbench.py), [test-kv-cache-zeroscrolls.py](tests/test-kv-cache-zeroscrolls.py), and [test-kv-cache-leval.py](tests/test-kv-cache-leval.py).

**Qwen3.5-4B Q8_0 benchmark highlights:**

| Context | Cache config | BPW | RTX 5090 pp | RTX 5090 tg | Strix Halo pp | Strix Halo tg |
| :-- | :-- | --: | --: | --: | --: | --: |
| 2k | `f16/f16` | 16.00 | 14,538 t/s (baseline) | 236.02 t/s (baseline) | 1,750 t/s (baseline) | 42.30 t/s (baseline) |
| 2k | `pq4_0/pq4_0` | 4.25 | 0.96x | 0.96x | 0.94x | 0.99x |
| 2k | `pq3_0/pq3_0` | 3.25 | 0.96x | 0.96x | 0.95x | 0.99x |
| 2k | `tbq4_0/pq4_0` | 4.75 | 0.71x | 0.94x | 0.60x | 0.97x |
| 2k | `tbq3_0/pq3_0` | 3.75 | 0.71x | 0.94x | 0.60x | 0.98x |
| 8k | `f16/f16` | 16.00 | 13,981 t/s (baseline) | 236.67 t/s (baseline) | 1,585 t/s (baseline) | 42.46 t/s (baseline) |
| 8k | `pq4_0/pq4_0` | 4.25 | 0.89x | 0.96x | 0.82x | 0.99x |
| 8k | `pq3_0/pq3_0` | 3.25 | 0.89x | 0.96x | 0.85x | 0.99x |
| 8k | `tbq4_0/pq4_0` | 4.75 | 0.42x | 0.94x | 0.32x | 0.97x |
| 8k | `tbq3_0/pq3_0` | 3.75 | 0.43x | 0.94x | 0.32x | 0.97x |

Quality checks on Qwen3.5-4B Q8_0 show `tbq4_0/pq4_0` at -0.03% perplexity delta versus `f16/f16`, with 94.8% RULER main score and 37.04 LongBench average versus 37.52 for `f16/f16`. See the full [TurboQuant benchmark report](docs/turboquant-benchmarks.md) for all measured models, contexts, and quality results.

### Weight Prefetch for Offloaded Models

`-pw/--prefetch-weights 0|1|auto` overlaps CPU-to-GPU weight transfers with compute when only part of a model fits in VRAM. `auto` (the default) enables it for fitted dense models. The scheduler also exposes prefetch, reserve-size and shared-compute-buffer APIs (`ggml_backend_sched_*`) for embedding applications.

### Mobile GPU Optimization

Enhanced GPU support with targeted optimizations for Qualcomm Adreno GPUs.

- **Vulkan on Adreno 800+**: quantized inference (Q4_0, Q8) and LoRA fine-tuning for Gemma3, Qwen3, and BitNet (TQ2_0) model architectures.
- **OpenCL on Adreno**: inference support for all other model architectures.
- Adreno-specific Vulkan shader variants for improved throughput.
- Vulkan Memory Allocator (VMA) integration for efficient GPU memory management.

### VisionPsy Nano / Flash Support

Run VisionPsy Nano and Flash vision-language models through the multimodal subsystem using compatible GGUF models and projectors.

- **Image sizing**: `--image-no-upscale on` enables the Flash preprocessing rule, rounding image sizes to the slice grid without always stretching smaller images to the maximum size. Use it for Flash projectors that do not declare this rule in GGUF metadata.
- **Memory-aware vision attention**: automatic flash-attention selection accounts for image size and device memory on backends without efficient cooperative-matrix flash attention.

---

## Quick Start

### Building from Source

```bash
git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git
cd qvac-fabric-llm.cpp

# Standard build
cmake -B build
cmake --build build --config Release

# With Vulkan support (Android, Windows, Linux)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release

# With Metal support (macOS, iOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# With Core ML / Apple Neural Engine sidecars (macOS 15+, requires Metal)
cmake -B build -DGGML_METAL=ON -DLLAMA_COREML=ON
cmake --build build --config Release

# With AMD XDNA NPU support (Linux, Strix-class APUs)
cmake -B build -DGGML_XDNA=ON -DGGML_OPENMP=ON \
      -DGGML_XDNA_GEMM_PYTHON=$HOME/aie-env/bin/python
cmake --build build --config Release
```

For more detailed build instructions, see [docs/build.md](docs/build.md).

### Running a Model

```bash
# Run a model with Vulkan GPU acceleration, 4096 context length, and a prompt
./build/bin/llama-cli -m model.gguf -ngl 99 -c 4096 -p "Explain quantum computing in simple terms"
```

---

## Supported Platforms

| Platform | Backend | Status |
| :-- | :-- | :--: |
| Linux (x86_64, ARM64) | CPU, Vulkan, CUDA, OpenCL | ✅ Full support |
| Linux (Ryzen AI / Strix) | XDNA NPU (XDNA2) | ✅ Inference backend |
| macOS (Intel, Apple Silicon) | CPU, Metal, Core ML (Apple silicon) | ✅ Full support |
| Windows (x86_64) | CPU, Vulkan, CUDA | ✅ Full support |
| Android (ARM64) | CPU, Vulkan, OpenCL | ✅ Full support |
| iOS | CPU, Metal | ✅ Full support |

---

## Relationship with llama.cpp

qvac-fabric-llm.cpp is a maintained fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The project regularly synchronizes with upstream releases to incorporate improvements, bug fixes, and new model support, while extending the engine with capabilities not present in the upstream project.

**Current upstream baseline:** llama.cpp b10796

### Upstream Compatibility

All standard llama.cpp functionality, models, and APIs remain fully compatible.

- Any GGUF model supported by llama.cpp is supported by qvac-fabric-llm.cpp
- Existing llama.cpp documentation applies to shared functionality

---

## Contributing

We welcome contributions! Please see our development workflow:

1. Fork the repository
2. Create a feature branch from `master`
3. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

qvac-fabric-llm.cpp is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.

---

For additional documentation, refer to the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs).

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [minja](https://github.com/google/minja) - Minimal Jinja parser in C++, used by various tools/examples - MIT License
- [linenoise.cpp](./tools/run/linenoise.cpp/linenoise.cpp) - C++ library that provides readline-like line editing capabilities, used by `llama-run` - BSD 2-Clause License
- [curl](https://curl.se/) - Client-side URL transfer library, used by various tools/examples - [CURL License](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
