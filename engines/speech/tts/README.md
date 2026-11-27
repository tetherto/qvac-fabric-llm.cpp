# tts-cpp

Native C++17/ggml speech synthesis, voice cloning, sound-effect generation,
and post-synthesis enhancement for the QVAC speech stack. There is no Python, PyTorch, or ONNX
Runtime dependency after models have been converted to GGUF.

This package exposes seven public synthesis engine APIs covering eight synthesis
families and ten model lines: Chatterbox Turbo, Chatterbox Multilingual,
Supertonic 1, 2, and 3, Parler-TTS, CosyVoice3, Audio8, Pocket TTS, and MOSS
Delay. MOSS-SoundEffect adds a separate text-to-sound-effects API
(`moss/sound_effect.h`), MOSS-Speech a speech-to-speech API
(`moss/speech.h`), and LavaSR is a speech-enhancement pipeline; none of the
three is a TTS synthesizer. The API
count follows the installed public headers under `include/tts-cpp/`; the two
Chatterbox families share one engine API, while the Supertonic generations
share another.

Pocket TTS runs on CPU or opt-in Metal, CUDA, Vulkan or OpenCL; cloning requires encoder-enabled weights. MOSS Delay
and MOSS-TTSD are validated end to end on CPU/Metal and on the
[Snapdragon Hexagon NPU](docs/moss.md#hexagon-npu), with numerical reference
parity pending. MOSS-SoundEffect and MOSS-Speech have CPU/Metal stage validation,
and MOSS-SoundEffect also matches the reference pipeline on the
[Hexagon NPU](docs/moss.md#hexagon-npu-generation), where MOSS-Speech
[runs its LM and speech tokenizer](docs/moss.md#hexagon-npu-speech-to-speech);
CUDA and Vulkan add generated-model CPU parity and full-checkpoint smoke coverage.
MOSS Delay on Metal projects only the text-control tokens needed during audio
generation and omits text projection during the forced drain. Embedding tables
stay on the GPU, and the control-token head preserves the model's weight type.
On Vulkan, MOSS Delay batches audio-head logits into one readback per forward
pass while retaining the individual projections and sampling behavior.
MOSS-TTSD reuses the encoded reference for dialogue continuation when only one
recording is supplied. Multiple recordings retain the joined-audio encoding
needed for their continuation; all prepared codes stay cached per engine.
TTS streaming, cancellation/reuse and ASR intelligibility gates are described
in [MOSS GPU validation](docs/moss.md#gpu-validation-cuda-and-vulkan); smoke coverage does
not establish general speech quality. OpenCL remains untested here. See the [API matrix](docs/api.md) for
streaming support and the [model guide](docs/models.md) for storage tiers.

Supertonic cancellation applies to the in-flight synthesis request. Starting a
subsequent batch or streaming `Engine::synthesize()` clears the prior cancellation
state, so voice applications can interrupt playback and reuse the loaded engine
for the next reply.
Cancellation is checked between synthesis stages and chunks; an active vector
estimator can finish its entire loop before cancellation is observed.

Supertonic also exposes `Engine::synthesize_segment()` for callers that already
select sentence or clause boundaries. It synthesizes the entire supplied segment
without applying the internal text chunker, preserves continuation intonation
for unfinished clauses, and optionally delivers one PCM callback. The existing
`synthesize()` streaming API retains its configured internal text chunking.

Hosts that synthesize on worker threads should reuse one thread for the session
and call `Engine::release_thread_resources()` on that thread before it exits.
This releases its inference graph caches while the model backend is still
alive. Keep the engine alive until the worker has completed this cleanup; the
same engine can synthesize again after its caches are released.

The `test-supertonic-request-lifecycle` and `-v3` fixture tests cover cancellation
and reuse, whole-segment continuation/resampling parity, optional callbacks, and
worker cache cleanup. They run when the corresponding model exists under
`TTS_CPP_TEST_MODEL_DIR`; `test-supertonic-api` checks the public API without models.

MOSS generation starts at the reference assistant header. ISO language codes
accepted by the API/CLI are converted to the model's language names (`en` to
`English`, `zh` to `Chinese`); spelled-out names pass through.

MOSS reference cloning applies the upstream loudness adjustment, and TTSD uses
its dialogue-specific prompt template. Use the explicit TTSD sampling settings
in the [dialogue example](docs/moss.md); the CLI's shared defaults are for
TTS-v1.5, and sampled dialogue can stop before all requested turns are spoken.
The opt-in [validated cloning runner](docs/moss.md#validated-cloning) checks
English dialogue and each requested turn with local MOSS transcription, retrying rejected outputs,
including ASR results that reach the decoding token limit. It supports CPU and
CUDA, plus Vulkan through `--gpu --backend vulkan`.

This in-tree package consumes the shared fabric ggml (`qvac-fabric`) dependency. Build and
vcpkg consumption instructions are in [docs/build.md](docs/build.md).

MOSS-SoundEffect native diffusion prepares the fixed cross-attention keys and
values once per prompt and reuses them across steps. The request owns the cache;
Metal also uses DiT flash attention when the backend supports both attention
shapes. Metadata-only memory fit includes the cache alongside the scheduler's
retained peak across text encoding, diffusion and VAE decoding, including repeated
requests. On the CPU, quantized text-encoder and DiT matrices load into
ggml's repacked layout where one exists, which halves the diffusion time on a
Galaxy S25 (`q8_0` and `q4_0`) and on x86 (`q4_0`).
Besides `f16` and `q8_0`, a 2.8 GB `q8_0` GGUF with `q4_0` DiT blocks
is validated on CPU, Metal, CUDA and the Hexagon NPU; quantizing the text
encoder to `q4_0` as well is not recommended
([4-bit weights](docs/moss.md#4-bit-weights)). See the
[SoundEffect guide](docs/moss.md#moss-soundeffect) for details.

## Capabilities

### Supported models and backends

Backends below are the paths validated or explicitly implemented by each
engine, not every backend ggml can compile.

| Model line | Languages | Voice source | Native rate | CPU | Metal | Vulkan | OpenCL | CUDA | WebGPU |
|---|---|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Chatterbox Turbo | English | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes (+ optional Core ML sidecars) | yes | yes | yes | untested |
| Chatterbox Multilingual | 23 | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes (+ optional Core ML sidecars) | yes | yes | yes | untested |
| Supertonic 1 | English | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes | untested |
| Supertonic 2 | `en`, `ko`, `es`, `pt`, `fr` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes | untested |
| Supertonic 3 | 31 languages plus `na` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes | yes² |
| Parler-TTS mini/large/Indic | English or 21 Indic languages | natural-language description | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes | yes² |
| Fun-CosyVoice3-0.5B | model-advertised multilingual text | baked voice or zero-shot/cross-lingual reference WAV; instruct controls | 24 kHz | yes | yes (+ optional Core ML sidecars) | yes | yes | yes | yes² |
| Audio8-TTS-Preview-0.6B | multilingual checkpoint vocabulary | model voice or zero-shot reference WAV + transcript | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes | yes² |
| Pocket TTS | English | prepared voice; cloning requires encoder-enabled weights | 24 kHz | yes | yes (+ optional Core ML sidecars) | yes | yes | yes | untested |
| MOSS Delay (MOSS-TTS-v1.5 / MOSS-TTSD) | model-advertised multilingual text | model voice, zero-shot reference WAV, or two-speaker dialogue references; pause/duration/pronunciation controls | 24 kHz | yes | yes (+ optional Core ML sidecar) | scoped¹ | untested | scoped¹ | untested |
| MOSS-SoundEffect-v2 (text to sound effects) | text prompt | none | 48 kHz | yes | yes (+ optional Core ML sidecars) | scoped¹ | untested | scoped¹ | untested |
| MOSS-Speech (speech to speech) | spoken English and Chinese | built-in default voice or reference WAV | 24 kHz | yes | yes (+ optional Core ML sidecar) | scoped¹ | untested | scoped¹ | untested |
| LavaSR denoiser | language agnostic | input PCM | rate preserving | yes | yes (+ optional Core ML model) | yes | yes | yes | untested |
| LavaSR enhancer | language agnostic | input PCM | 48 kHz | yes | yes (+ optional Core ML model) | yes | yes | yes | untested |

¹ MOSS CUDA/Vulkan: generated-model CPU parity; quantized TTS/TTSD,
SoundEffect and Speech checkpoint smoke cases. Vulkan also covers the
F16 TTS/TTSD/SoundEffect and BF16 Speech cases. The full-checkpoint TTS quality
gate requires every expected CPU/GPU batch, streaming and reuse WAV to pass
WER <= 0.25. Wrong text has no baseline-audio exemption; missing outputs,
ASR errors and non-finite scores fail the gate. See the linked MOSS guide.

² WebGPU (Dawn on an RTX 5090): Supertonic 3 renders with a 0.997 waveform
correlation to the CPU run and the same transcript (2026-10-20). Parler mini
(f16, q8_0), CosyVoice3 (q8_0 LM) and Audio8 (q8_0, f16 LM) give the same
Whisper transcripts as their CPU renders over four prompts (2026-10-26);
their sampled tokens diverge between backends, so waveforms are not compared.
These are the TTS models whose automatic GPU selection admits WebGPU
(CosyVoice3 on Windows and desktop Linux, like Vulkan and CUDA). The
LM-based ones need a fabric ggml whose WebGPU kernels honour
`GGML_PREC_F32` (qvac-fabric-llm.cpp `be6e387b` or later).

Process exit with Core ML: a process that exits while a sidecar is still compiling or loading in the
background waits for that load (by default at most 15 s on macOS, 2 s on iOS) when its main thread is armed.
The main thread is armed automatically when it makes the engine; a host that makes engines on another thread
(as the qvac addons do) calls `coreml_arm_exit_guard()` on its main thread, or the terminal
`coreml_shutdown_for_exit()` right before the process exits (`tts-cpp/coreml_exit.h`). See
[the runtime's README](../coreml/README.md#process-exit).

LavaSR denoiser and enhancer GPU selection includes ARM Mali/Immortalis
through Vulkan when GPU execution is requested, and excludes WebGPU: the
graphs run without a per-op CPU fallback and ggml-webgpu lacks `MEAN`.
Both LavaSR stages also accept explicit Core ML models and one of `CPU_ONLY`,
`CPU_AND_GPU`, `CPU_AND_NE`, or `ALL`, with `TTS_CPP_COREML=ON` on Apple.
See [LavaSR Core ML](docs/lavasr.md#core-ml) for export and C++ examples.
The Core ML denoiser submits independent chunks through batch prediction,
reusing the same loaded model.

Chatterbox Turbo and Multilingual, Supertonic 3, Parler-TTS mini, CosyVoice3, Audio8, MOSS-TTS-v1.5 /
MOSS-TTSD, MOSS-SoundEffect-v2 and MOSS-Speech (LM and speech tokenizer) also support explicit Hexagon placement on Snapdragon; see their
model guides for setup and validated tiers. Audio8's corrected baseline uses `q8_0` quantisation with
`--greedy` sampling under `--backend hexagon`; see
[Audio8](docs/audio8.md#hexagon-npu-snapdragon) for the full-compute tuning
recipe (`OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`), the corrected single-prompt
baseline, and the remaining cross-backend correctness limits. Optional
Apple Core ML sidecars accelerate the Chatterbox Turbo and Multilingual S3Gen
encoder, CFM and HiFT, the Pocket TTS FlowLM generation step and Mimi decoder, the CosyVoice3
speech LM, flow and HiFT neural
stages, the Supertonic vocoder, the Audio8 codec, the Parler-TTS DAC decoder
(on the Neural Engine), and
the MOSS-TTS codec decoder, MOSS-SoundEffect DiT and VAE, and MOSS-Speech
speech tokenizer. Chatterbox loads its sidecar functions in the background and runs
each sentence on Metal until they are ready (`EngineOptions::coreml`,
`tts-cli --coreml-use|wait|prewarm|compute-units|cache-dir|status-json`).
Export, routing, fallback, and per-call status are documented in
[Chatterbox](docs/chatterbox.md#core-ml-sidecars),
[Pocket TTS](docs/pocket-tts.md#core-ml-sidecars),
[CosyVoice3](docs/cosyvoice3.md#core-ml-sidecars),
[Supertonic](docs/supertonic.md#core-ml-vocoder-sidecar),
[Audio8](docs/audio8.md#core-ml-codec-sidecar),
[Parler-TTS](docs/parler.md#core-ml-dac-sidecar-apple-neural-engine), and
[MOSS](docs/backends.md#apple-core-ml-sidecars).

The Parler-TTS DAC sidecar can also be generated on the device from the GGUF
itself: with `EngineOptions::coreml.cache_dir` (or `QVAC_COREML_CACHE_DIR`)
set, macOS 15 or later and no prebuilt sidecar next to the GGUF, the engine
builds it from an embedded template, compiles it and keeps it in the cache;
synthesis runs on the ggml DAC until it is ready. Its accuracy against the
ggml DAC depends on how Core ML partitions it on the chip: 56 dB on an M2,
where `all` runs it on the Neural Engine; 51 dB on an M4 Max, where `all`
runs most of it on the GPU and the output stage on the Neural Engine. CI
runs on an M4 Max: it holds `all` to that chip's recorded 50 dB floor and
the same sidecar under `cpu_and_ane` to 54 dB. The M2's 54 dB gate under
`all` is checked by the device run, not by CI. See
[Parler-TTS generated sidecar](docs/parler.md#generated-sidecar-and-background-loading).

CosyVoice3 HiFT sidecars use FP32 for waveform parity. With enabled Chatterbox,
Pocket TTS, CosyVoice3 or prebuilt Parler-TTS sidecars present, `fit_params` returns `Error` with
`coreml-memory-estimate-unavailable`; it cannot project Core ML resident memory.
A generated Parler-TTS DAC is projected: fit adds its template's Core ML memory envelope,
963 MiB under `all` (1.25 times the largest values measured on an M2 and on the M4 Max CI
runner, where `all` runs on the GPU, with no further margin). Phones are not affected today, since sidecars are generated
on macOS only; see [Parler-TTS memory validation](docs/parler.md#generated-sidecar-and-background-loading).
Re-export CosyVoice3 LM sidecars with the current exporter for correct cache
feedback on macOS 15 and 26; see [CosyVoice3 Core ML](docs/cosyvoice3.md#core-ml-sidecars).

Chatterbox Multilingual has native tokenization for 18 languages; Japanese,
Hebrew, Russian, Chinese and Hindi need external preprocessing. See
[CLI setup](docs/cli.md#run--end-to-end-text--wav) for the supported paths.

Backend selection and model-specific validation are documented in
[docs/backends.md](docs/backends.md) and [docs/testing.md](docs/testing.md).
Cross-platform CI selects affected engines and caches Windows compiler outputs;
see the [CI guide](../docs/CI.md).
The test guide includes a model-free matched WAV/mel fixture generator,
reference-generation commands for CAMPPlus and S3TokenizerV2, and truncated-NPY
rejection checks, plus the failure criteria for voice-cloning and streaming
parity. Model-free checks run without downloading checkpoints.
Metadata-only memory preflight covers MOSS Delay/TTSD and MOSS-SoundEffect,
including text/diffusion/decode workloads for sound effects. Entry points are indexed in
[docs/api.md](docs/api.md#memory-fit-preflight). Every fit result names its device
by the ggml device name, and a device that shares host memory and reports none
of its own carries the CPU's figures; see
[memory fit results](../README.md#memory-fit-results).

## Performance

`audio8-codec-bench` can decode captured codes on CPU, Hexagon or OpenCL and
export WAV/raw F32 PCM while measuring the codec separately from generation.
See the [fixed-code decoder benchmark](docs/audio8.md#fixed-code-decoder-benchmark).

`RTF = generation_time / audio_duration`; lower is faster. Dated CI snapshots,
Supertonic 3 and Audio8 campaigns, Apple-silicon measurements, streaming
latency, build pins and reproduction steps live in
[docs/performance.md](docs/performance.md).

CosyVoice3 uses capability-checked flash attention for single-token LM decoding
on CUDA and Metal. Supported CUDA backends reuse decode graphs within masked
128-token cache windows, allowing CUDA graph capture. Unsupported backends or
head dimensions retain their existing attention path.

CosyVoice3 flow uses F16 key/value operands with F32 accumulation for
capability-checked Vulkan flash attention, falling back to F32 operands or
regular attention when required.

CosyVoice3 expands bf16 flow weights to f32 when loading on ARM CPUs. This
avoids scalar bf16 matmuls at twice the resident storage for those weights;
memory preflight includes the expansion. Prefer an f16 flow bundle on ARM.
Use `cosyvoice-bench --llm-gguf`, `--flow-gguf`, and `--hift-gguf` to vary
one stage at a time; see [stage profiling](docs/cosyvoice3.md#stage-profiling).

## Documentation

| Topic | Where |
|---|---|
| Model formats and quantization | [docs/models.md](docs/models.md) |
| APIs, pipelines, per-surface defaults | [docs/api.md](docs/api.md) |
| Voice conditioning (emotion, pace) | [docs/voice-conditioning.md](docs/voice-conditioning.md) |
| Chatterbox | [docs/chatterbox.md](docs/chatterbox.md) |
| Parler-TTS | [docs/parler.md](docs/parler.md) |
| Supertonic | [docs/supertonic.md](docs/supertonic.md) |
| CosyVoice3 | [docs/cosyvoice3.md](docs/cosyvoice3.md) |
| Audio8 | [docs/audio8.md](docs/audio8.md) |
| Pocket TTS | [docs/pocket-tts.md](docs/pocket-tts.md) |
| MOSS Delay, MOSS-SoundEffect, and MOSS-Speech | [docs/moss.md](docs/moss.md) |
| LavaSR enhancement | [docs/lavasr.md](docs/lavasr.md) |
| Build paths and repository layout | [docs/build.md](docs/build.md) |
| CLIs, weight conversion, end-to-end runs | [docs/cli.md](docs/cli.md) |
| Performance deep dive (Apple silicon, CI, streaming) | [docs/performance.md](docs/performance.md) |
| Desktop benchmark coverage (Chatterbox Turbo and Multilingual, Supertonic 1/2/3, Parler mini and Indic, CosyVoice3, Audio8, LavaSR, MOSS), metrics and runner limits; the MOSS-TTS, MOSS-SoundEffect and MOSS-Speech families are macOS-only opt-ins | [Benchmark coverage](../docs/PERFORMANCE.md#benchmark-coverage) |
| Memory lifecycle (Supertonic and Chatterbox) | [MEMORY.md](MEMORY.md) |
| Backend selection | [docs/backends.md](docs/backends.md) |
| Fixtures and static validation | [docs/testing.md](docs/testing.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Archived port and integration reports | [Chatterbox](docs/history/chatterbox-port.md), [Supertonic](docs/history/supertonic-port.md), [Pocket](docs/history/pocket-integration.md) |
| Voice-clone backward passes | [docs/voiceclone-backward-gap-matrix.md](docs/voiceclone-backward-gap-matrix.md) |

## License

The package code is released under the [MIT License](LICENSE). Models and
conversion-time dependencies retain their own terms: Chatterbox MIT, Pocket
CC-BY-4.0, Supertonic OpenRAIL-M, and Parler/CosyVoice3/Audio8/LavaSR/MOSS
Apache-2.0; see [NOTICE](NOTICE) for
canonical upstream sources and license identities. This in-tree package does
not bundle ggml.

Audio8 supports codec-window PCM callbacks through `Engine::synthesize_stream()`.
The CLI `--stream` mode measures callback first-audio latency and real-time factor;
see [incremental PCM streaming](docs/audio8.md#incremental-pcm-streaming).

Audio8 preserves separate CUDA F32 projection shapes during multi-token prefill
for fused/separate numerical consistency. Decode and other weight types retain
projection fusion; shared CUDA matmul routing is unchanged.
