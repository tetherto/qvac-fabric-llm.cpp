# qvac-fabric-speech.cpp

On-device speech and audio AI in C++17 on [ggml](https://github.com/tetherto/qvac-fabric-llm.cpp/tree/master/ggml): speech-to-text, speaker diarization, end-of-utterance detection, text-to-speech, voice cloning, sound effects, speech-to-speech, enhancement, and music generation.

| Property | Value |
|---|---|
| Build | Feature-gated CMake superbuild over Whisper and three native engines |
| Runtime | No Python, PyTorch, NeMo, or ONNX Runtime at inference time |
| Model formats | Whisper/Silero GGML `.bin`; Parakeet, TTS, and AudioGen GGUF |
| Desktop | Linux, macOS, Windows |
| Mobile | Android arm64-v8a, iOS arm64; model and backend coverage varies by engine |
| Backends | CPU, Metal, Vulkan, OpenCL, CUDA, WebGPU (Dawn, opt-in; see [WebGPU](#webgpu-dawn)); selected models also support explicit Hexagon placement or optional Apple Core ML sidecars |
| Shared ggml | One `qvac-fabric` dependency: the `ggml/` tree of [qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp), the same ggml the LLM stack ships |

## Supported models

Parakeet mel preprocessing keeps FFT twiddle tables in worker or stream scope,
avoiding a reproduced MinGW heap crash during dynamic thread-local cleanup while
preserving parallel execution and serial/threaded numerical parity.

Supertonic engines can be reused after cancellation without reloading their
models. `Engine::synthesize_segment()` preserves caller-selected sentence or
clause boundaries, and `Engine::release_thread_resources()` releases a worker's
inference caches before that thread exits. See the
[TTS guide](tts/README.md) for the lifecycle contract.
Cancellation can wait for an active vector-estimator loop to finish.

The engine guides own model formats, quantization, backend validation, and input
requirements. An available ggml backend does not imply every model has been
validated on it. AudioGen automatic GPU selection requires F32 matrix
multiplication support and otherwise continues to another device or CPU.

Eligible offline Parakeet TDT and RNN-T calls keep encoder features on the GPU
through decoding, including Nemotron's locale-prompt projection. Encoder tensors
with extra padding frames retain host routing. See the
[device encoder handoff](parakeet/docs/backends.md#device-encoder-handoff)
for routing, fallback and validation coverage.

Parakeet TDT uses the native encoder beyond a standard Core ML sidecar's input
capacity; short Core ML windows require an export with a validated seam policy.
These standard TDT sidecars retain native weights even with
`PARAKEET_COREML_ONLY=1`; incompatible encoder-less GGUFs fail at load time.
Standard TDT push streams default to 30-second chunks and reject smaller explicit
chunks to prevent phrase loss. Small PCM writes and final tail flushing remain
supported; full TranscribePsy EU retains its separate streaming policy. See the
[Parakeet streaming contract](parakeet/docs/api.md).

The standard TDT ggml encoder masks right-padding in attention and convolution,
so padding cannot alter valid encoder frames. Offline TDT still re-encodes
windows in push streams; accuracy can vary with the available context.

| Engine | Model families | Capabilities and limitations |
|---|---|---|
| [Whisper](docs/WHISPER.md) | tiny, base, small, medium, large v1/v2/v3, large-v3-turbo; Silero v5.1.2/v6.2.0 | ASR, multilingual-to-English translation, VAD, tinydiarize speaker turns; metadata-only memory fit from a model or its weightless description; `.en` checkpoints are English-only and turbo is intended for transcription; tiny/base/small/large-v3-turbo validated on WebGPU |
| [Parakeet](parakeet/README.md) | CTC, Unified RNN-T, TDT, TranscribePsy, IndicConformer, realtime EOU, Nemotron 3.5 ASR | Offline, streaming, long-form ASR and end-of-turn detection; IndicConformer CTC requires a language selection; TranscribePsy EU-sub ships as ready-made GGUF and Core ML files for eight European languages and runs on the Hexagon NPU; the full 25-language EU model supports CPU, Metal, Vulkan/Android and macOS/iOS Core ML, including CAFE adapters and automatic language gating; on every backend, full-EU streaming defaults to 2 s chunks and auto push streams buffer the first 20 s for routing (shorter streams route on finalize; known `en`/`other` routes emit immediately; [streaming API](parakeet/docs/api.md), [Core ML guide](parakeet/docs/transcribepsy-eu-coreml.md)) |
| Parakeet | Sortformer v1/v2/v2.1, Nemotron 3 Diarization | Speaker diarization and attributed ASR; Sortformer supports up to four speakers, Nemotron up to eight |
| Parakeet | MOSS-Transcribe-Diarize | One-pass transcription, speakers, timestamps, hotwords and metadata-only memory fit including scheduler storage and CPU scratch; fused CPU encoder attention; CPU/Metal validated on Spanish/Chinese; Metal decode reuses one graph per cache window (29-42% faster decode on M3 Ultra, identical transcripts); explicit CUDA/Vulkan/OpenCL/Metal/Hexagon selection with generated-model CPU parity and quantized checkpoint smoke coverage; English long-form skips spans in both native and reference runs |
| [TTS](tts/README.md) | Chatterbox Turbo/Multilingual, Supertonic 1/2/3, Parler mini/large/Indic, CosyVoice3, Audio8 | Synthesis with preset, described, or cloned voices, depending on the model; streaming support varies by API and CLI; Supertonic 3, Parler mini, CosyVoice3 and Audio8 also validated on WebGPU |
| TTS | Pocket TTS | English synthesis and incremental CPU/Metal/CUDA/Vulkan/OpenCL streaming; cloning requires encoder-enabled weights |
| TTS | MOSS-TTS-v1.5 / MOSS-TTSD | Multilingual synthesis, cloning, dialogue and streaming; single-reference dialogue avoids duplicate reference encoding; selective text-control projection on Metal; batched audio-logit readback on Vulkan; CPU/Metal validated; CUDA/Vulkan generated-model parity and synthesis/streaming/cancellation checks (see scoped validation in the TTS guide); explicit Hexagon NPU placement with a `q4_0` transformer; numerical reference parity pending |
| TTS | MOSS-SoundEffect-v2, MOSS-Speech | Sound-effect generation and spoken replies respectively; SoundEffect reuses fixed cross-attention K/V, selects supported Metal DiT flash attention, and runs quantized matrices from ggml's repacked CPU layout, with the cache and retained scheduler peak included in metadata-only memory fit for repeated requests; CPU/Metal generation validated, including a `q8_0` GGUF with `q4_0` DiT blocks; CUDA/Vulkan generated-model parity and quantized SoundEffect/Speech checkpoint smoke coverage; explicit Hexagon NPU placement for SoundEffect, with reference parity, and for the Speech LM and speech tokenizer |
| TTS | LavaSR denoiser/enhancer | Speech denoising and bandwidth extension; GPU selection includes ARM Mali for both stages and excludes WebGPU, whose kernels lack an op the graphs need |
| [AudioGen](audiogen/README.md) | ACE-Step v15 turbo/sft/base | Music generation, source-conditioned covers, and editing; base additionally supports lego stems ([CLI](audiogen/docs/cli.md), [C++ usage](audiogen/docs/build.md)) |
| AudioGen | MiniMax-Music3 | Desktop music generation on CPU or GPU, optional [Core ML conditioner projection](audiogen/docs/backends.md#minimax-core-ml-conditioner), and metadata-only [memory-fit preflight](audiogen/docs/memory-fit.md); unavailable on Android/iOS |

Parakeet also accepts explicit [AMD XDNA](parakeet/docs/backends.md#amd-xdna-npu)
selection through Fabric LLM's native GGML backend. It uses the existing GGUF,
requires matching XDNA runtime artifacts, and schedules unsupported operations
on CPU. Q8_0 matrices stay on the host; use F16 for NPU validation.

ACE-Step also supports explicit [AMD XDNA DiT execution](audiogen/docs/backends.md#amd-xdna-npu)
through `backend = "xdna"` or `XDNA0`. Unsupported DiT operations use a CPU
scheduler fallback; the encoders, LM, detokenizer and VAE retain CPU placement.

MOSS cloning matches upstream reference loudness preprocessing and uses the
TTSD-specific dialogue prompt. The [TTSD example](tts/docs/moss.md)
sets its model-specific sampling parameters explicitly; check generated speech
for complete dialogue rather than relying on a successful exit alone.
An optional [validated cloning runner](tts/docs/moss.md#validated-cloning)
checks English dialogue, including each requested turn, with local ASR and retries
rejected generations before
publishing an output WAV. It supports CPU, CUDA, and Vulkan (`--gpu --backend vulkan`).
ASR results that reach the decoding token limit are rejected.

MOSS generation starts at the reference assistant header and converts ISO language
codes such as `en` to the model's language names (`English`). The full-checkpoint
TTS CI gate requires every expected CPU/GPU batch, streaming and reuse output to
pass WER <= 0.25, with no baseline-audio exemption for wrong text. See
[MOSS validation](tts/docs/moss.md#gpu-validation-cuda-and-vulkan).

CosyVoice3 uses capability-checked flash attention for single-token LM decoding
on CUDA and Metal. Supported CUDA backends reuse decode graphs within masked
128-token cache windows, allowing CUDA graph capture. Unsupported backends or
head dimensions retain their existing attention path.

CosyVoice3 flow uses F16 key/value operands with F32 accumulation for
capability-checked Vulkan flash attention, falling back to F32 operands or
regular attention when required.

CosyVoice3 expands bf16 flow weights to f32 on ARM CPUs to avoid scalar bf16
matmuls; f16 remains the recommended ARM flow bundle. Its benchmark supports
independent LM, flow and HiFT model overrides for
[stage profiling](tts/docs/cosyvoice3.md#stage-profiling).

Nemotron 3 Diarization preparation supports additional weight precisions via
`convert_nemotron_diarization.py --outtype`. CPU/CUDA offline validation covers
F16, BF16, Q6_K, Q5_K, Q4_K, and NVFP4. Q4_K/Q6_K also pass offline
Metal and Vulkan validation; NVFP4 runs on Vulkan but falls back to CPU
matmuls on Metal. Streaming quality varies by precision and backend.
See the [quantization measurements](parakeet/docs/performance.md#nemotron-3-diarization-quantizations)
before selecting a format. MXFP4 failed CPU quality checks.

### Experimental Parakeet-Mosaic

[Mosaic](parakeet/docs/mosaic.md) uses a frozen mixed-precision
recipe with ordinary GGML kernels. The shared 486.7 MB Mosaic and 339.6 MB
[Mosaic Compact](parakeet/docs/mosaic.md#shared-mosaic-and-mosaic-compact-models)
also support separate, matching optional Core ML encoder sidecars. The earlier experimental
[`parakeet-affine-v1` format](parakeet/docs/backends.md#experimental-packed-affine-tdt-recipe)
uses integer carriers, with reconstruction by default and opt-in Metal fusion
via `PARAKEET_AFFINE_FUSED=1`. Fusion requires the matching
[GGML dependency](parakeet/docs/fused-affine-metal.md#integration-and-pr-handoff);
older dependencies and unsupported fused backends fail explicitly. Offline TDT
is the evaluated scope; packed-affine substage profiling is rejected. Recipe
receipts and measured limitations are documented in the linked guides.

Fixed TDT Core ML exports now require both `mel` and `mel_valid_mask`, and need
the updated runtime. Legacy single-input sidecars still load but must be
re-exported for padding invariance. See the
[sidecar compatibility contract](parakeet/docs/backends.md#core-ml-encoder-sidecar).

### Apple Core ML sidecars

Whisper exposes `whisper_encoder_on_coreml(ctx)` for the loaded encoder state,
independently of the ggml primary device.

Optional sidecars accelerate a stage while the rest of the pipeline stays on
ggml. They default to disabled and have model-specific input and fallback
contracts; see the engine guides before exporting or deploying one.

| Engine | Accelerated stage | Guide |
|---|---|---|
| Whisper | Encoder | [Whisper Core ML](third_party/whisper.cpp/README.md#core-ml-support) |
| Parakeet | Full TranscribePsy EU English/adapted encoders and language-gate prefix, macOS/iOS Core ML streaming with the backend-independent full-EU 2 s default, optional offline batched TDT decode | [Full-EU Core ML guide](parakeet/docs/transcribepsy-eu-coreml.md) |
| Parakeet | Eligible ASR and Sortformer v2.1 encoder paths; TDT models also run a fixed-window Neural Engine encoder and a batched TDT decoder | [Backend and routing contracts](parakeet/docs/backends.md#core-ml-encoder-sidecar), [Neural Engine sidecars](parakeet/docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars) |
| Parakeet | Nemotron 3 Diarization and Sortformer v2.1 whole networks | [Diarizer sidecars](parakeet/docs/backends.md#core-ml-diarizer-sidecars) |
| Parakeet | MOSS-Transcribe-Diarize audio encoder | [MOSS-Transcribe-Diarize](parakeet/docs/moss-transcribe.md#core-ml-encoder-sidecar) |
| TTS | Chatterbox Turbo and Multilingual S3Gen encoder, CFM and HiFT (overlapping the next sentence's T3) | [Chatterbox Core ML](tts/docs/chatterbox.md#core-ml-sidecars) |
| TTS | Pocket TTS FlowLM generation step (KV cache in Core ML state) and Mimi decoder (Neural Engine) | [Pocket TTS Core ML](tts/docs/pocket-tts.md#core-ml-sidecars) |
| TTS | CosyVoice3 speech LM, flow and HiFT neural stages | [CosyVoice3 Core ML](tts/docs/cosyvoice3.md#core-ml-sidecars) |
| TTS | Supertonic vocoder, Audio8 codec synthesis | [Supertonic](tts/docs/supertonic.md#core-ml-vocoder-sidecar), [Audio8](tts/docs/audio8.md#core-ml-codec-sidecar) |
| TTS | LavaSR denoiser (batch prediction) and enhancer neural cores, with explicit model paths and compute policies | [LavaSR Core ML](tts/docs/lavasr.md#core-ml) |
| TTS | MOSS-TTS/TTSD codec decoder, MOSS-SoundEffect DiT and VAE, MOSS-Speech speech tokenizer | [MOSS](tts/docs/moss.md#core-ml-codec-decoder), [SoundEffect](tts/docs/moss.md#core-ml-dit-and-vae), [Speech](tts/docs/moss.md#core-ml-speech-tokenizer) |
| AudioGen | ACE-Step VAE decoder | [AudioGen backends](audiogen/docs/backends.md#core-ml-vae-decoder-sidecar) |
| AudioGen | MiniMax conditioner projection, or explicit full-network Core ML (`coreml_profile_dir` + `coreml_compute_units`, macOS 15+) | [MiniMax Core ML](audiogen/docs/backends.md#minimax-core-ml-conditioner) |

Nemotron 3 Diarization and Sortformer v2.1 diarizers need no exported bundle
on macOS: a build with
`PARAKEET_COREML=ON` (the generator, `PARAKEET_COREML_GENERATE`, is on by default
on macOS) generates each diarizer capacity from the GGUF on the device in the
background, compiles it and keeps it in a cache directory, and diarization runs
on ggml until a capacity is ready. Generation needs that cache directory: set
`CoremlOptions::cache_dir` or `QVAC_COREML_CACHE_DIR` for the library (it has
none by default, and then nothing is generated), or `--coreml-cache-dir` for
the `parakeet` CLI (which defaults to
`~/Library/Application Support/qvac-speech/coreml`). See
[Generated sidecars](parakeet/README.md#generated-sidecars) for the
setup, options and memory.
The generated Sortformer v2.1 capacity is charged at 259 MiB after calibration
against the latest overlapped Core ML preparation measurements, retaining the
required 25% safety margin.

TranscribePsy conversion uses 60-second windows with 25% overlap for full EU
and records its Apache-2.0 provenance. For EU-sub Core ML deployment, the
repository provides an encoder-stripping tool and decoder output-projection
palettization with `--nbits-out`; see the
[conversion guide](parakeet/docs/models.md#transcribepsy).
Full-EU Core ML export requires an F16 GGUF and macOS with Xcode for compilation.
The [full-EU guide](parakeet/docs/transcribepsy-eu-coreml.md#export-and-deployment)
provides the conversion command and recommends explicit FP16 encoders/decoder
as the baseline, with six-bit encoders for smaller deployments. The Engine also
loads a [shared-weight multifunction encoder](parakeet/docs/transcribepsy-eu-coreml.md#shared-weight-runtime-deployment)
for English, adapted and gate routes, including encoder-less GGUF deployments.
It replaces the three fixed bundles without changing the public API or Core ML build option.
An [experimental route-sharing harness](parakeet/docs/transcribepsy-eu-coreml.md#trying-route-sharing-exports)
can measure those alternatives separately from the deployed runtime.
Its [automated regression tests](parakeet/examples/ios-coreml-bench/README.md#route-harness-regression-tests)
cover fixture/report serialization and native prediction/loading with generated models,
including rejection of malformed manifests and duplicate WAV stems.

Chatterbox loads its sidecar functions in the background through the shared
Core ML runtime (`coreml`): construction and synthesis never wait for a
Neural Engine compile, each sentence runs on Metal until the functions it needs
are ready, and `tts_cpp::CoremlOptions` (`EngineOptions::coreml`, `tts-cli
--coreml-*`) selects what loads up front, whether to wait, and where warm-load
hints live; see [Chatterbox loading, fallback and options](tts/docs/chatterbox.md#loading-fallback-and-options).

MiniMax's full Core ML route is also exposed through its public C++ engine.
Build with `AUDIOGEN_MINIMAX_COREML=ON` and supply both `coreml_profile_dir` and
`coreml_compute_units`; the separate flag does not enable automatic ACE-Step
sidecars. It supports `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`, and `ALL` on
macOS 15+. See the [full Core ML guide](audiogen/docs/minimax-coreml-validation.md)
for the matching F16 compiled profile, measurements, and remaining numerical
validation.

CosyVoice3 exports HiFT in FP32 to preserve waveform parity. Its memory-fit
preflight returns `Error` with `coreml-memory-estimate-unavailable` when
enabled sidecars are present, rather than undercounting their resident memory.
Re-export CosyVoice3 LM sidecars with the current exporter to preserve cache
feedback on macOS 15 and 26; see the [Core ML guide](tts/docs/cosyvoice3.md#core-ml-sidecars).

`scripts/coreml/bench-load-time.py` measures Core ML load time for the
Chatterbox sidecars: first-load compile on a fresh host identity, warm load,
time until every function is ready with and without the admission gate, and
whether loaded functions share weight memory. The
[Core ML load time measurements](docs/COREML-LOAD-TIME.md#setup-and-usage)
list its requirements, the commands to run it and the recorded results.

### Hexagon NPU (Snapdragon)

Selected models run on the Hexagon HTP accelerator via `--backend hexagon`
(CLI) or the engine's `backend = "hexagon"` option (API), end to end except
for MOSS-Speech, whose reply decoder stays on the CPU or the GPU.
Each guide names the validated quantisation tier, the runtime env flags
that recover the measured speedup, and how the backend compares to CPU
and OpenCL on the same device.

| Engine | Models | Guide |
|---|---|---|
| ASR | Parakeet CTC and TDT 0.6B (`q8_0`), TranscribePsy-250M-EU-sub (`q8_0`), MOSS-Transcribe-Diarize (`q8_0`, `f16`) | [Parakeet Hexagon](parakeet/docs/backends.md#hexagon-npu), [MOSS Hexagon](parakeet/docs/moss-transcribe.md#hexagon-npu) |
| TTS | Chatterbox Turbo and Multilingual (`q4_0`, `q8_0`, `f16` or `f32`), Supertonic 3, Parler-TTS mini, CosyVoice3, Audio8, MOSS-TTS-v1.5 / MOSS-TTSD (`q4_0` transformer), MOSS-SoundEffect-v2 (`f16`, `q8_0`, or `q8_0` with `q4_0` DiT blocks), MOSS-Speech LM (`q4_0` transformer) and speech tokenizer (`f16` codec) | [Chatterbox Hexagon](tts/docs/chatterbox.md#snapdragon-hexagon-npu), [Audio8 Hexagon](tts/docs/audio8.md#hexagon-npu-snapdragon), [MOSS Hexagon](tts/docs/moss.md#hexagon-npu), [MOSS-SoundEffect Hexagon](tts/docs/moss.md#hexagon-npu-generation), [MOSS-Speech Hexagon](tts/docs/moss.md#hexagon-npu-speech-to-speech), and the matching sections in the other model guides |
| AudioGen | ACE-Step v15 turbo (`q8_0` DiT; K-quant GGUFs are refused at load) | [AudioGen Hexagon](audiogen/docs/backends.md#hexagon-npu) |

### WebGPU (Dawn)

`GGML_WEBGPU=ON` (the `webgpu` feature of the `speech-cpp` port) adds ggml's WebGPU backend on Google Dawn, driven natively inside the
process; no browser is involved. It registers as `WebGPU`, ranks with Vulkan
and Metal in every engine's automatic GPU selection (CUDA still wins on the
same NVIDIA card), and is selected explicitly with `--backend webgpu` or
`backend = "webgpu"`. Build and host requirements are in
[docs/BUILD.md](docs/BUILD.md#webgpu-dawn). Validation on an RTX 5090
(Dawn on Vulkan; 2026-10-20, the LM-based TTS models and large-v3-turbo
2026-10-26), each model against the same CPU run:

| Engine | Model | WebGPU result |
|---|---|---|
| Whisper | tiny.en, base, small | Validated: same transcripts; encoder 8x (base) to 33x (small) faster than CPU, a 2-minute clip 3.1x faster end to end |
| Whisper | large-v3-turbo | Validated: same transcripts; a 2-minute clip encodes in 0.27 s against 23.4 s on the CPU, 11x faster end to end |
| Parakeet | TranscribePsy 250M TDT | Validated: identical transcript, 2.8x faster end to end |
| TTS | Supertonic 3 | Validated: waveform correlation 0.997 with the CPU render, identical transcript |
| TTS | Parler-TTS mini (f16, q8_0) | Validated: the same Whisper transcripts as the CPU render over four prompts; 3.5x faster end to end |
| TTS | CosyVoice3 (q8_0 LM, f16 flow and HiFT) | Validated: the same Whisper transcripts as the CPU render over four prompts; 12x faster end to end |
| TTS | Audio8 (q8_0 and f16 LM, f16 codec) | Validated: the same Whisper transcripts as the CPU render over four prompts; 5x faster end to end |
| TTS | Pocket, Chatterbox, MOSS, LavaSR | Not tested |
| AudioGen | ACE-Step, MiniMax-Music3 | Not tested; the ACE-Step LM and detokenizer keep their CPU placement on an unmeasured backend |

Parler, CosyVoice3 and Audio8 sample their tokens, so their WebGPU and CPU
renders diverge after the first differing token the way any two backends do;
the transcripts are the check. They rely on ggml-webgpu computing the matmul
and flash-attention ops they mark `GGML_PREC_F32` with f32 activations and
f32 accumulation, which the fabric ggml provides since qvac-fabric-llm.cpp
`be6e387b` (ported from ggml-speech 2026-10-27) and on the temp-11739 line
since its base `ee1ccf4e3`; with an older ggml their
decoders overflow f16 and produce NaN logits, silence or
noise.

## Getting started

Install a shared speech-branch ggml using [docs/BUILD.md](docs/BUILD.md), then
configure from this checkout with its install prefix:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build --parallel
```

For a first transcription on a single-config build:

```sh
bash third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
  -f third_party/whisper.cpp/samples/jfk.wav
```

The [build guide](docs/BUILD.md#windows-and-multi-config-generators) provides
Windows and multi-config commands. Use [docs/CLI.md](docs/CLI.md) to choose a
tool, and the engine's model guide to acquire its required weights.

## Performance

`audio8-codec-bench` can decode captured codes on CPU, Hexagon or OpenCL and
export WAV/raw F32 PCM while measuring the codec separately from generation.
See the [fixed-code decoder benchmark](tts/docs/audio8.md#fixed-code-decoder-benchmark).

`RTF = inference_time / audio_duration`; lower is faster. These are dated
measurements with different workloads and timing boundaries, not universal
speed guarantees. Full methods, build pins and limitations live in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and the linked engine reports;
[benchmark coverage](docs/PERFORMANCE.md#benchmark-coverage) lists which
model each desktop benchmark family measures and scores.

| Task | Model | Recorded measurements |
|---|---|---|
| ASR | Parakeet TDT 0.6b v3 | September 2026 GPU lanes: short-clip RTF 0.0007–0.0055; [method and full table](parakeet/docs/performance.md#multi-machine-benchmark-2026-09) |
| TTS | Supertonic 3 | September 2026 GPU lanes: process wall 0.61–0.82 s, including model load; [full table](tts/docs/performance.md#supertonic-3-multi-machine-benchmark-2026-09) |
| TTS | Audio8 | September 2026 GPU campaign: RTF 0.20–0.65, with later RTX 5090 measurements down to 0.116; [full table and follow-up](tts/docs/performance.md#audio8-multi-machine-benchmark-2026-09) |
| Music | ACE-Step 1.5 | September 2026 CUDA/Vulkan/Metal lanes: generation 1,338–9,016 ms, RTF 0.139–0.957; [full table](audiogen/docs/performance.md#ace-step-15-multi-machine-benchmark-2026-09) |
| Music | MiniMax-Music3 f16 | September 2026 RTX 5090: two-minute song in 73.9 s CUDA / 84.2 s Vulkan, excluding model load; [full table](audiogen/docs/performance.md#minimax-music3-f16-on-rtx-5090-2026-09) |

## Memory fit results

Every engine's memory-fit preflight reports the same device fields, so a caller
can add up several models that share one device:

- `device_name` is the ggml device name, as `ggml_backend_dev_by_name` finds it:
  `CPU`, `CUDA0`, `Vulkan0`, `MTL0`, `GPUOpenCL`, `HTP0`. Two engines on the
  same device report the same name.
- `device_shares_host_memory` is true for a device that draws from host RAM: the
  CPU, an integrated GPU, Apple Metal, Hexagon and Android's OpenCL GPUs.
- `device_free_bytes` and `device_total_bytes` are the device's own figures. A
  device that shares host memory and reports none of its own, such as Hexagon,
  carries the CPU's figures.

## Use in QVAC

These engines ship inside [QVAC](https://github.com/tetherto/qvac) as SDK
addons consuming the `speech-cpp` vcpkg port. For JavaScript/TypeScript APIs,
Bare runtime integration, and desktop/mobile applications, see QVAC.

| QVAC addon | Wraps | `speech-cpp` features |
|---|---|---|
| `@qvac/asr-ggml` | ASR, diarization, end-of-utterance | `whisper`, `parakeet` |
| `@qvac/tts-ggml` | Synthesis, voice cloning, enhancement | `tts` |
| `@qvac/audiogen-ggml` | Music generation | `audiogen` |
| `@qvac/bci-whispercpp` | Brain-computer interface transcription | `whisper` |

## Licenses

Code and model weights have separate terms. No model weights are shipped in
this source tree. Engine license sections and `NOTICE` files identify sources
and model-specific exceptions.

| Component | Code | Model weights |
|---|---|---|
| [Whisper](third_party/whisper.cpp/LICENSE) | MIT | OpenAI Whisper MIT; Silero under its model terms |
| [Parakeet](parakeet/README.md#license) | Apache-2.0 | NVIDIA Parakeet/Sortformer CC-BY-4.0; EOU NVIDIA Open Model License; Nemotron OpenMDW-1.1; IndicConformer MIT; MOSS-Transcribe-Diarize and TranscribePsy Apache-2.0 |
| [TTS](tts/NOTICE) | MIT | Chatterbox MIT; Pocket CC-BY-4.0; Supertonic OpenRAIL-M; Parler, CosyVoice3, Audio8, LavaSR and MOSS models Apache-2.0 |
| [AudioGen](audiogen/NOTICE) | MIT | ACE-Step 1.5 MIT; Qwen3-Embedding Apache-2.0; MiniMax-Music3 Community License |

## Documentation

The [CI guide](docs/CI.md) describes engine-specific cross-platform job selection,
Windows compiler caching, and the VAD contract tests.

Parakeet's [test guide](parakeet/docs/testing.md) separates model-free
unit tests from fixture coverage. Its fixture runner rejects runs with no executed
tests and can require specific model-family tests to pass; the opt-in GPU CI lane
uses this gate.

TTS's [test guide](tts/docs/testing.md) separates model-free checks
from optional numerical fixtures and documents how to generate independent
WAV/mel, CAMPPlus and speech-tokenizer references. WAV/mel fixtures need no model
checkpoint, and NPY loading rejects truncated files. Chatterbox preprocessing and streaming
parity harnesses reject numerical mismatches, non-finite values and incomplete
output.

AudioGen's [test guide](audiogen/docs/testing.md) documents independent
model-free suites and required model-backed CI coverage. The macOS lane stages
pinned ACE-Step and MiniMax fixtures and fails if generation tests skip.

| Topic | Where |
|---|---|
| Architecture and repository layout | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Building and consuming packages | [docs/BUILD.md](docs/BUILD.md) |
| CLI selection and examples | [docs/CLI.md](docs/CLI.md) |
| Performance overview and reports | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Whisper models and formats | [docs/WHISPER.md](docs/WHISPER.md) |
| Native Whisper Snapdragon build and validation | [docs/WHISPER-HEXAGON.md](docs/WHISPER-HEXAGON.md) |
| Whisper subtree deltas and synchronization | [PATCHES.md](third_party/whisper.cpp/PATCHES.md), [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [Parakeet](parakeet/README.md) |
| Synthesis, voice cloning, sound effects, speech-to-speech, enhancement | [TTS](tts/README.md) |
| Music generation and editing | [AudioGen](audiogen/README.md) |
| Benchmark quality diagnostics and model preparation | [Music alignment guide](scripts/benchmarks/music-alignment.md) |
| Weekly desktop benchmark on the shared Tailscale machines | [desktop-weekly](scripts/benchmarks/desktop-weekly/README.md) |
| Running, collecting and publishing that benchmark from Claude | [speech-bench skill](skills/speech-bench/SKILL.md) |
| Historical development and integration reports | [Chatterbox](tts/docs/history/chatterbox-port.md), [Supertonic](tts/docs/history/supertonic-port.md), [Pocket](tts/docs/history/pocket-integration.md), [Parakeet](parakeet/docs/history/parakeet-port.md), [Strix Halo](audiogen/docs/history/strix-halo.md) |

Audio8 exposes incremental PCM through `Engine::synthesize_stream()`, with codec-window
delivery on ggml and Core ML. See the [streaming API and measurement guide](tts/docs/audio8.md#incremental-pcm-streaming).

Audio8 preserves separate CUDA F32 projection shapes during multi-token prefill
for fused/separate numerical consistency. Decode and other weight types retain
projection fusion; shared CUDA matmul routing is unchanged.

### Sortformer Core ML reproduction preflight

The `Sortformer Core ML preflight` workflow runs on same-repository pull requests
that touch Parakeet or benchmark scripts, and supports manual dispatch. It uses
the existing self-hosted authorization gate and the Apple GPU runner. It downloads
NVIDIA's public v2.1 NeMo checkpoint at revision
`cd03eee90fbec18297ac31b8c21546e596b7f71c`, checks its pinned SHA-256, and converts
it to q8_0 with the repository converter before exporting Core ML. Keeping the
original checkpoint filename lets the converter emit the required AOSC variant
metadata. No AWS role or private model artifact is required. Download, checksum,
conversion and exporter validation failures fail the job.

`model-provenance.json` records the checkpoint URL, revision and checksum,
converter checksum, command and generated GGUF checksum. `conversion.log`
preserves converter diagnostics. Exporter compatibility validation is unchanged.

The preflight makes two complete `Engine::diarize` calls on the three-speaker
`abcba` recording per backend, reusing each engine between calls. It compares
forced native Metal with a freshly exported fp16 2048-frame whole-network
Core ML diarizer from the same q8_0 GGUF. The capacity covers the entire
recording; only the full diarizer is staged beside the accelerated model, so
`encoder_used_coreml` confirms whole-network execution. A native fallback or
empty output fails the preflight. The accelerated process sets
`PARAKEET_COREML_STRICT=1` so current runtimes wait for background sidecar
loading and throw if the requested capacity cannot serve a call. The native
process clears strict mode and disables Core ML. The audio frontend remains native.

Artifacts retain each call's speaker probabilities, segments, routing flag,
commands, logs, asset hashes and revisions. Probability comparisons use the
existing `test_sortformer_diarizer_coreml_parity.cpp` limits: maximum absolute
error <= 0.02 and relative L2 <= 0.003, with native probabilities as reference.
Both calls must pass. DER is reported per call with a 250 ms collar and optimal
speaker-label mapping; it has no new threshold. The separate AOSC speaker test
has a default 30% DER ceiling on streaming output without this collar, so that
ceiling is not applied to these offline calls.

This is a repository-fixture reproduction, not the original failed recording:
the original audio, model/bundle identity and gate invocation are unknown.
The parity limits above originate in a lower-level batch/slab test; applying
them to complete calls is an explicit diagnostic extension. The exported bundle
is a diagnostic candidate, not a replacement for distributed artifacts.

The generic benchmark driver now computes `der_median` from every measured run
and preserves `.der-runs.json` plus per-run hypothesis files. Any scoring failure
leaves the median unset. The legacy `.hypothesis.jsonl` remains the last run's
output, not a representative median hypothesis.
