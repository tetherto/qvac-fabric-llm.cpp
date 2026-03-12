# qvac-fabric-llm.cpp

**AI inference and training engine for desktop and mobile platforms.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp%20b7248-orange.svg)](https://github.com/ggml-org/llama.cpp)

`qvac-fabric-llm.cpp` is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for embedded systems, mobile devices, and enterprise deployment scenarios. It extends the excellent foundation of llama.cpp with additional capabilities focused on memory-based model loading, mobile GPU optimization, and flexible integration patterns.


## Key Features

The following capabilities are developed and maintained as part of qvac-fabric-llm.cpp. Features marked as *exclusive* are not available in upstream llama.cpp.

### LoRA Fine-Tuning *(exclusive)*

`qvac-fabric-llm.cpp` provides native [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) fine-tuning across CPU, Vulkan, and Metal backends. The training pipeline runs directly on consumer hardware, including mobile phones and integrated GPUs.

- Multi-backend GPU training (NVIDIA, AMD, Intel, Apple, Mali, Adreno)
- FP32, FP16, Q8, and Q4 training paths
- Supervised instruction-tuning via assistant-only masked loss (SFT)
- Checkpoint saving and resumable training
- Learning rate schedulers (constant, cosine, linear) with warmup support
- LoRA adapter merging into base models via **llama-export-lora**
- Verified compatibility with Qwen3 and Gemma3 model architectures

For usage details and CLI reference, see the [Finetuning Guide](examples/training/README.md).

### BitNet Inference and Fine-Tuning *(exclusive)*

Native support for [BitNet](https://arxiv.org/abs/2402.17764) ternary quantized models via the TQ2_0 data type, enabling efficient inference and LoRA fine-tuning of models such as [bitnet_b1_58-xl](https://huggingface.co/gianni-cor/bitnet_b1_58-xl-TQ2_0) on resource-constrained devices.

The official [microsoft/BitNet](https://github.com/microsoft/BitNet) inference framework provides optimized CPU kernels and GPU support limited to CUDA. `qvac-fabric-llm.cpp` **extends** BitNet to all major GPU backends -- Vulkan, Metal, and CPU -- bringing cross-platform GPU-accelerated BitNet inference and on-device fine-tuning to hardware not covered by the upstream framework, including Apple Silicon, mobile GPUs (Adreno, Mali), and AMD/Intel discrete GPUs. Compatible with models such as [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).

- **Backends**: Vulkan, Metal and CPU TQ2_0 quantization support
- **Training**: LoRA fine-tuning of BitNet models on Vulkan, Metal, and CPU backends
- **Conversion**: HuggingFace-to-GGUF conversion for BitNet model architectures
- Cooperative matrix (coopmat) support for Vulkan devices that expose the extension

### Memory-Based Model Loading *(exclusive)*

Load models directly from memory buffers instead of the filesystem, enabling deployment in environments where disk access is restricted or unavailable.

- Embedded systems with limited or no filesystem access
- WebAssembly deployments where models are fetched over the network
- Encrypted model storage where models are decrypted in memory
- Streaming scenarios where models arrive over network connections

```cpp
#include "llama-cpp.h"

std::vector<uint8_t> model_data = /* load from network, decrypt, etc. */;
auto model = llama_model_load_from_buffer(std::move(model_data), params);

auto model = llama_model_load_from_split_futures(paths, n_paths, context, tensor_list, params);
llama_model_load_fulfill_split_future(path, context, std::move(streambuf));
```

### Mobile GPU Optimization *(exclusive)*

Enhanced GPU support with targeted optimizations for Qualcomm Adreno GPUs.

- **Vulkan on Adreno 800+**: quantized inference (Q4_0, Q8) and LoRA fine-tuning for Gemma3, Qwen3, and BitNet (TQ2_0) model architectures.
- **OpenCL on Adreno**: inference support for all other model architectures.
- Adreno-specific Vulkan shader variants for improved throughput.
- Vulkan Memory Allocator (VMA) integration for efficient GPU memory management.


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


## Supported Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| Linux (x86_64, ARM64) | CPU, Vulkan, CUDA | ✅ Full support |
| macOS (Intel, Apple Silicon) | CPU, Metal | ✅ Full support |
| Windows (x86_64) | CPU, Vulkan, CUDA | ✅ Full support |
| Android (ARM64) | CPU, Vulkan, OpenCL | ✅ Full support |
| iOS | CPU, Metal | ✅ Full support |


## Relationship with llama.cpp

qvac-fabric-llm.cpp is a maintained fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The project regularly synchronizes with upstream releases to incorporate improvements, bug fixes, and new model support, while extending the engine with capabilities not present in the upstream project.

**Current upstream baseline:** llama.cpp b7248

### Exclusive Features

The following features are developed in qvac-fabric-llm.cpp and are not available in upstream llama.cpp:

| Feature | Description |
|---------|-------------|
| LoRA fine-tuning | On-device training across CPU, Vulkan, and Metal with SFT, checkpointing, and LR scheduling |
| BitNet inference and training | TQ2_0 quantization on Vulkan, Metal, and CPU for inference and LoRA fine-tuning; extends [microsoft/BitNet](https://github.com/microsoft/BitNet) beyond its CUDA-only GPU support |
| Memory-based model loading | Load models from in-memory buffers with split-model and async fulfillment support |
| Mobile GPU optimization | Adreno 800+ quantized inference (Q4_0, Q8), Adreno-specific Vulkan shader variants, VMA integration |

### Upstream Compatibility

All standard llama.cpp functionality, models, and APIs remain fully compatible.

- Any GGUF model supported by llama.cpp is supported by qvac-fabric-llm.cpp
- Existing llama.cpp documentation applies to all non-exclusive features


## Contributing

We welcome contributions! Please see our development workflow:

1. Fork the repository
2. Create a feature branch from `master`
3. Submit a pull request


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
