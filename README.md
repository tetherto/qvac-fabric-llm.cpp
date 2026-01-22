# fabric-llm-finetune

![qvac fabric llm](https://raw.githubusercontent.com/tetherto/qvac-fabric-llm.cpp/db93fc7425e72b1bb375c5b9dd33f5fcc072dac6/media/qvac-fabric-llm.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## News
**January 22, 2026:** BitNet fine tuning is now officially supported!  
Read our full announcement and technical deep dive in these blog posts:  
- [Fabric-LLM-Finetune: General Overview](https://huggingface.co/blog/qvac/fabric-llm-finetune)
- [Fabric-LLM-Finetune with BitNet Support](https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet)



## LoRA Fine-Tuning

llama.cpp includes native [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) fine-tuning across CPU, Vulkan, Metal and CUDA backends.

LoRA fine-tuning represents the weight updates with two smaller matrices through low-rank decomposition while keeping the base model frozen. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. This makes training possible on devices with very limited memory, including phones and integrated GPUs. Key capabilities include:

- Train LoRA adapters on any GPU (NVIDIA, AMD, Intel, Apple, Mali, Adreno)
- Full support for FP32/FP16/Q8/Q4 training paths
- Instruction-tuning via assistant-only masked loss
- Checkpointing + resumable training
- Merge LoRA adapters back into a base model `model.gguf`
- Compatible with Qwen3, Gemma, LLaMA, TinyLlama, and other GGUF models

The [Finetuning Guide](examples/training//README.md) has more details.

### BitNet Support 
BitNet introduced an extreme quantization scheme that represents weights using only 1.58 bits. BitNet's quantization technique greatly reduces memory constraint and makes fine tuning possible on edge devices. This branch presents the world's first framework to enable fine-tuning of BitNet models using Low-Rank Adaptation (LoRA) on GPUs. Built on llama.cpp with a Vulkan backend, the framework supports BitNet model fine-tuning with LoRA across heterogeneous consumer GPUs. By bringing BitNet to GPUs, the framework enables fine-tuning on edge devices and achieves substantial performance improvements over CPU-based implementations.


## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [main (cli)](tools/main/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [minja](https://github.com/google/minja) - Minimal Jinja parser in C++, used by various tools/examples - MIT License
- [linenoise.cpp](./tools/run/linenoise.cpp/linenoise.cpp) - C++ library that provides readline-like line editing capabilities, used by `llama-run` - BSD 2-Clause License
- [curl](https://curl.se/) - Client-side URL transfer library, used by various tools/examples - [CURL License](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
