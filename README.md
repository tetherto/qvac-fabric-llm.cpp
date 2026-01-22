# qvac-fabric-llm.cpp

AI inference engine for embedded and mobile platforms**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp%20b7248-orange.svg)](https://github.com/ggml-org/llama.cpp)

qvac-fabric-llm.cpp is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for embedded systems, mobile devices, and enterprise deployment scenarios. It extends the excellent foundation of llama.cpp with additional capabilities focused on memory-based model loading, mobile GPU optimization, and flexible integration patterns.


## Key Features

### Memory-Based Model Loading

Load models directly from memory buffers instead of files - essential for:
- **Embedded systems** where filesystem access is limited or unavailable
- **WebAssembly** deployments where models are fetched over network
- **Encrypted model storage** where models are decrypted in memory
- **Streaming scenarios** where models are received over network connections

```cpp
#include "llama-cpp.h"

// Load model from memory buffer
std::vector<uint8_t> model_data = /* load from network, decrypt, etc. */;
auto model = llama_model_load_from_buffer(std::move(model_data), params);

// Or use split model loading with async fulfillment
auto model = llama_model_load_from_split_futures(paths, n_paths, context, tensor_list, params);

// ... later, fulfill splits as they become available
llama_model_load_fulfill_split_future(path, context, std::move(streambuf));
```

### Optimized Vulkan & OpenCL Backend for Mobile GPUs

Enhanced GPU support with specific optimizations for **Qualcomm Adreno GPUs**:
- **Support for running quantized LLMs (Q4_0, Q8) on Adreno 700+ GPUs**
- **Both Vulkan and OpenCL backends supported on Adreno GPUs**
- Adreno-specific shader variants for improved performance
- Q4_K optimized `mul_mat_vec` operations
- Vulkan Memory Allocator (VMA) integration for efficient memory management, with specific improvements for **Google Pixel** devices


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

qvac-fabric-llm.cpp is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp), an excellent open-source LLM inference engine. We regularly synchronize with upstream llama.cpp releases to incorporate the latest improvements, bug fixes, and model support.

**Current upstream version:** b7248

### What We Add

- Memory-based model loading API
- Vulkan Memory Allocator (VMA) integration (optimized for Pixel 9 devices)
- Adreno 700+ GPU support for quantized LLMs (Q4_0, Q8)
- Vulkan and OpenCL backends for Adreno GPUs
- Adreno GPU-specific shader optimizations
- Performance profiling tools

### Upstream Compatibility

All standard llama.cpp functionality, models, and APIs remain fully compatible. You can:
- Use any GGUF model supported by llama.cpp
- Use all standard CLI tools (`llama-cli`, `llama-server`, etc.)
- Follow llama.cpp documentation for general usage

---

## Model Support

qvac-fabric-llm.cpp supports all models compatible with llama.cpp. Models must be in GGUF format. Convert from other formats using the provided Python scripts or use pre-converted models from [Hugging Face](https://huggingface.co/models?library=gguf).

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
