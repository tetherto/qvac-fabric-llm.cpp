<div align="center">

<h1>qvac-fabric-llm.cpp</h1>

<p><strong>AI inference and training engine for desktop and mobile platforms.</strong></p>

<p>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
<a href="https://github.com/ggml-org/llama.cpp"><img src="https://img.shields.io/badge/based%20on-llama.cpp%20b10549-orange.svg" alt="Based on llama.cpp"></a>
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

### MoE Cache for Faster Decoding *(experimental)*

Improve decoding speed on hybrid systems with limited GPU VRAM and ample system RAM by keeping MoE expert weights in system RAM and caching recently used experts on the GPU. Cache hits reuse GPU-resident weights, reducing repeated CPU-to-GPU transfers during token generation. The benefit depends on expert reuse, cache size, and transfer bandwidth.

- **Configurable VRAM budget**: `--moe-cache-mib N` sets the persistent GPU expert cache budget in MiB; `0` disables it (the default).
- **Hybrid placement**: combine GPU layer offload with `--cpu-moe` to keep all expert weights in system RAM, or `--n-cpu-moe N` to keep the first N layers' expert weights there.
- **On-demand caching**: upload missing experts and evict least-recently-used entries when the cache fills. The cache supports single-GPU inference; OpenCL, tensor parallelism, multi-GPU execution, and training are not supported.

For example, keep expert weights in system RAM and allocate a 1 GiB GPU cache:

```bash
./build/bin/llama-cli -m moe-model.gguf \
    --n-gpu-layers all --cpu-moe --moe-cache-mib 1024 \
    -p "Explain how mixture-of-experts models work"
```

Choose a cache budget that leaves VRAM for other model weights, the KV cache, and compute buffers. It must be large enough to hold the active experts for at least one routed layer.

### TurboVec / Local Vector Search

`ggml-vector-index` provides a standalone C API for local vector search, including compressed CPU indexes for applications that need on-device retrieval.

- **Storage**: full-precision f32, generic q8 and packed q4, plus TurboVec 2-bit and 4-bit modes with scalar, NEON, and AVX2 scoring paths.
- **Search**: exact top-k scans, filtering by caller-provided IDs, reusable prepared filters, and optional IVF search for approximate candidate selection.
- **Persistence**: index snapshots, with read-only mmap loading and incremental mutation logs for the generic storage modes. TurboVec supports snapshot save/load; mmap and logged mutations are not yet supported for TurboVec.

The library is disabled by default. Enable it with `-DGGML_VECTOR_INDEX=ON` and link `ggml::vector-index` explicitly; it is not integrated into the llama runtime or server. See the [Vector Index Guide](docs/vector-index.md) for supported dimensions, API usage, persistence contracts, and benchmarks.

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

### LoRA Fine-Tuning

`qvac-fabric-llm.cpp` provides native [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) fine-tuning across CPU, Vulkan, and Metal backends. The training pipeline runs directly on consumer hardware, including mobile phones and integrated GPUs.

- Multi-backend GPU training (NVIDIA, AMD, Intel, Apple, Mali, Adreno)
- FP32, FP16, Q8, and Q4 training paths
- Supervised instruction-tuning via assistant-only masked loss (SFT)
- Checkpoint saving and resumable training
- Learning rate schedulers (constant, cosine, linear) with warmup support
- LoRA adapter merging into base models via **llama-export-lora**
- Verified compatibility with Qwen3, Qwen3.5 and Qwen3.6 (dense and MoE), Gemma3, and Gemma4 (dense and MoE) model architectures

For usage details and CLI reference, see the [Finetuning Guide](examples/training/README.md).

### BitNet Inference and Fine-Tuning

Native support for [BitNet](https://arxiv.org/abs/2402.17764) ternary quantized models via the TQ2_0 data type, enabling efficient inference and LoRA fine-tuning of models such as [bitnet_b1_58-xl](https://huggingface.co/gianni-cor/bitnet_b1_58-xl-TQ2_0) on resource-constrained devices.

The official [microsoft/BitNet](https://github.com/microsoft/BitNet) inference framework provides optimized CPU kernels and GPU support limited to CUDA. `qvac-fabric-llm.cpp` **extends** BitNet to all major GPU backends -- Vulkan, Metal, and CPU -- bringing cross-platform GPU-accelerated BitNet inference and on-device fine-tuning to hardware not covered by the upstream framework, including Apple Silicon, mobile GPUs (Adreno, Mali), and AMD/Intel discrete GPUs. Compatible with models such as [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).

- **Backends**: Vulkan, Metal and CPU TQ2_0 quantization support
- **Training**: LoRA fine-tuning of BitNet models on Vulkan, Metal, and CPU backends
- **Conversion**: HuggingFace-to-GGUF conversion for BitNet model architectures
- Cooperative matrix (coopmat) support for Vulkan devices that expose the extension

### VisionPsy Nano / Flash Support

Run VisionPsy Nano and Flash vision-language models through the multimodal subsystem using compatible GGUF models and projectors.

- **Image sizing**: `--image-no-upscale on` enables the Flash preprocessing rule, rounding image sizes to the slice grid without always stretching smaller images to the maximum size. Use it for Flash projectors that do not declare this rule in GGUF metadata.
- **Memory-aware vision attention**: automatic flash-attention selection accounts for image size and device memory on backends without efficient cooperative-matrix flash attention.

### Mobile GPU Optimization

Enhanced GPU support with targeted optimizations for Qualcomm Adreno GPUs.

- **Vulkan on Adreno 800+**: quantized inference (Q4_0, Q8) and LoRA fine-tuning for Gemma3, Qwen3, and BitNet (TQ2_0) model architectures.
- **OpenCL on Adreno**: inference support for all other model architectures.
- Adreno-specific Vulkan shader variants for improved throughput.
- Vulkan Memory Allocator (VMA) integration for efficient GPU memory management.

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
| Linux (x86_64, ARM64) | CPU, Vulkan, CUDA | ✅ Full support |
| macOS (Intel, Apple Silicon) | CPU, Metal | ✅ Full support |
| Windows (x86_64) | CPU, Vulkan, CUDA | ✅ Full support |
| Android (ARM64) | CPU, Vulkan, OpenCL | ✅ Full support |
| iOS | CPU, Metal | ✅ Full support |

---

## Relationship with llama.cpp

qvac-fabric-llm.cpp is a maintained fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The project regularly synchronizes with upstream releases to incorporate improvements, bug fixes, and new model support, while extending the engine with capabilities not present in the upstream project.

**Current upstream baseline:** llama.cpp b10549

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
