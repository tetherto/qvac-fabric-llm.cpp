# tts engine: backend selection

Part of the [tts documentation](../README.md).

Backends are compiled into, or loaded by, the shared speech-branch ggml.
See the [capability table](../README.md#capabilities) for model validation;
the ability to compile a ggml backend does not establish model parity.

## Runtime selection

Where a build carries both CUDA and Vulkan, selection prefers CUDA on NVIDIA:
measured on one binary on an RTX 3090, Chatterbox Turbo is 8.6x faster on CUDA
than on that card's Vulkan adapter. WebGPU (Dawn) ranks with Vulkan and Metal,
so CUDA still wins on the same NVIDIA card. Set `TTS_CPP_GPU_BACKEND` to `cuda`,
`vulkan`, `metal`, `opencl` or `webgpu` to pin one for a test arm or a
comparison; an unrecognised value is rejected rather than silently dropping to
the CPU. Where an engine takes an explicit backend request
(`EngineOptions::backend`, CLI `--backend`), `webgpu` names the WebGPU device
the same way `opencl` and `hexagon` name theirs; an unavailable explicit
request is an error, not a CPU fallback. Each engine admits WebGPU to its
automatic GPU selection only once validated; the model table in the
[README](../README.md#supported-models-and-backends) records that status.
Among several Vulkan adapters, Audio8 takes the one with the most free memory
and never an integrated adapter while a discrete one is visible; Supertonic and
CosyVoice3 take `EngineOptions::vulkan_device` (default: the first adapter).


Explicit Hexagon placement is supported for
[Chatterbox Turbo and Multilingual](chatterbox.md#snapdragon-hexagon-npu),
[Supertonic 3](supertonic.md#snapdragon-hexagon-npu),
[Parler mini](parler.md#snapdragon-hexagon-npu),
[CosyVoice3](cosyvoice3.md#snapdragon-hexagon-npu),
[Audio8](audio8.md#hexagon-npu-snapdragon),
[MOSS-TTS / MOSS-TTSD](moss.md#hexagon-npu),
[MOSS-SoundEffect](moss.md#hexagon-npu-generation),
[MOSS-Speech](moss.md#hexagon-npu-speech-to-speech) and
[PocketTTS](pocket-tts.md#hexagon-execution); it is not an automatic GPU fallback.

## Apple Core ML sidecars

`TTS_CPP_COREML=ON` enables optional Chatterbox S3Gen, Pocket TTS FlowLM step and Mimi decoder, Supertonic vocoder,
Audio8 codec, Parler-TTS DAC and MOSS sidecars on Apple. Their guides own naming, export commands, accepted
shapes, low-bit restrictions, placement and runtime status:

- [Chatterbox S3Gen encoder, CFM and HiFT](chatterbox.md#core-ml-sidecars)
- [Pocket TTS FlowLM generation step and Mimi decoder](pocket-tts.md#core-ml-sidecars)
- [Supertonic vocoder](supertonic.md#core-ml-vocoder-sidecar)
- [Audio8 codec](audio8.md#core-ml-codec-sidecar)
- [Parler-TTS DAC decoder](parler.md#core-ml-dac-sidecar-apple-neural-engine)
- [MOSS-TTS / MOSS-TTSD codec decoder](moss.md#core-ml-codec-decoder)
- [MOSS-SoundEffect DiT and VAE](moss.md#core-ml-dit-and-vae)
- [MOSS-Speech speech tokenizer](moss.md#core-ml-speech-tokenizer)

Other TTS model families run entirely through their ggml paths.

Sidecars can also be generated on the device instead of downloaded. With a
cache directory (`CoremlOptions::cache_dir` in `tts-cpp/coreml_options.h`, or
`QVAC_COREML_CACHE_DIR`), macOS 15 or later and `TTS_CPP_COREML_GENERATE=ON`
(the macOS default), an engine builds a stage's sidecar from the GGUF it
loads, using a weight-free template embedded in the library, compiles it on
the device and keeps it in the cache; the stage runs on ggml until the sidecar
is ready. A prebuilt sidecar next to the GGUF still takes precedence. Today
the [Parler-TTS DAC decoder](parler.md#generated-sidecar-and-background-loading)
is generated; the other families load only prebuilt sidecars.
