# Architecture and repository layout

Part of the [speech-stack documentation](../README.md).

## Shared runtime

The umbrella resolves one installed fabric ggml (the `ggml/` tree of
[qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp), the
same ggml the LLM stack ships) before adding the engines. Every enabled engine
consumes that dependency; the ggml tree inside the Whisper subtree is not
compiled by the umbrella. Direct engine builds may instead stage their own
checkout of that tree, as their build guides describe.

```text
Whisper          Parakeet          TTS          AudioGen
   |                 |             |              |
   +-----------------+-------------+--------------+
                     |
       qvac-fabric ggml (qvac-fabric-llm.cpp/ggml)
                     |
       CPU / Metal / Vulkan / OpenCL / CUDA / Hexagon

Selected model stages may also use optional Apple Core ML sidecars.
```

Backend and sidecar support varies by model. The
[root capability overview](../README.md#supported-models) links to the engine
validation and routing contracts.

## Pipelines

| Engine | High-level flow | Details |
|---|---|---|
| Whisper | Audio → log-mel → encoder/decoder → text; optional VAD and encoder sidecar | [Whisper guide](../third_party/whisper.cpp/README.md) |
| Parakeet | Audio → features → model-specific encoder/head → text, speakers or turn boundary; MOSS uses a separate transcription engine | [Public APIs](../parakeet/docs/api.md), [MOSS](../parakeet/docs/moss-transcribe.md) |
| TTS | Text or speech → model-specific generation → acoustic decoder/codec → PCM; optional LavaSR enhancement | [Pipelines and APIs](../tts/docs/api.md#pipelines) |
| AudioGen | Caption/lyrics or source audio → model-specific conditioning/generation → stereo PCM | [ACE-Step and MiniMax pipelines](../audiogen/docs/pipeline.md) |

## Repository layout

| Path | Role |
|---|---|
| `CMakeLists.txt` | Feature-gated umbrella superbuild |
| `third_party/whisper.cpp/` | Upstream Whisper git subtree; release recorded in [UPSTREAM_PIN](../third_party/whisper.cpp/UPSTREAM_PIN), deliberate deltas in [PATCHES.md](../third_party/whisper.cpp/PATCHES.md) |
| `parakeet/` | Native ASR, diarization, EOU and MOSS transcription |
| `tts/` | Synthesis, cloning, sound effects, speech-to-speech and enhancement |
| `audiogen/` | ACE-Step and MiniMax music generation/editing |
| `engines/*/docs/` | Engine build, model, API, CLI, backend and validation guides |
| `engines/*/docs/history/` | Archived implementation and integration records |
| `scripts/benchmarks/` | Shared performance/quality tooling and operational guides |
| `docs/UPSTREAM-SYNC.md` | Whisper subtree synchronization procedure |
