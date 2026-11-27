# parakeet.cpp

Parakeet is a pure C++/ggml implementation of NVIDIA FastConformer ASR,
end-of-turn detection, and Sortformer speaker diarization. Inference requires no
Python, PyTorch, NeMo, or ONNX Runtime. A single `parakeet::Engine` loads CTC,
RNN-T, TDT, EOU, Nemotron, or Sortformer GGUFs and selects the implementation
from GGUF metadata. MOSS-Transcribe-Diarize, a Qwen3-based model that
transcribes and labels speakers in one pass (with optional per-request
hotwords), ships alongside with its own `parakeet::moss::TranscribeEngine`
and `moss-transcribe` CLI. `--backend cuda|vulkan|metal` (or
`TranscribeOptions::backend = "cuda"` / `"vulkan"` / `"metal"`) explicitly selects a GPU backend; the
request fails if unavailable. Generated-model CPU/CUDA/Vulkan/Metal parity and
F16/Q5/Q8 full-checkpoint smoke cases cover this path. On Metal the decoder reuses one
graph per cache window, which makes decoding 29-42% faster on an M3 Ultra with identical
transcripts ([Metal decoding](docs/moss-transcribe.md#metal-decoding)). Its `parakeet::moss::fit_params` preflight measures
weights, encoder/decoder graphs, KV cache and host memory, including scheduler storage
and CPU scratch for the configured threads, using GGUF metadata only; see
[docs/moss-transcribe.md](docs/moss-transcribe.md).

The MOSS CPU audio encoder uses fused attention to avoid materializing full
attention score matrices. The GPU encoder layouts are unchanged; the decoder
changes only its Metal single-token graph (see above).

Mel preprocessing keeps FFT twiddle tables in each worker's scope and in the
incremental stream state. This avoids concurrent destruction of dynamic
thread-local caches, which reproduced a heap-corruption crash with the local
MinGW runtime. FFT workers remain parallel. `test-mel-threads` exercises repeated
worker lifetimes across FFT sizes 256, 512, and 1024 with identical serial and
threaded results; `test-mel-fft-parity` also covers incremental and offline parity.

## Supported checkpoints

| HF repository | Decoder/task | Mel | `d_model × layers` | Vocab | Parameters | GGUF size | Recorded RTF | Languages and notes |
|---|---|---:|---|---:|---:|---|---|---|
| `nvidia/parakeet-ctc-0.6b` | CTC | 80 | 1024 × 24 | 1024 | 600 M | 697 MiB q8_0 / 1.3 GiB f16 | 0.014–0.046 Metal | English; optional Core ML offline encoder |
| `nvidia/parakeet-ctc-1.1b` | CTC | 80 | 1024 × 42 | 1024 | 1.1 B | 1217 MiB q8_0 | 0.026–0.074 Metal | English; optional Core ML offline encoder |
| `ai4bharat/indic-conformer-600m-multilingual` | CTC default; optional RNN-T export | 80 | 1024 × 24 | 5632 + blank | 600 M | ~701 MiB q8_0 / ~373 MiB q4_0 / 1.3 GiB f16 | 0.008 q8_0 Metal / 0.0019 q8_0 Vulkan | 22 Indic languages; optional Core ML offline encoder; requires `--language` or `EngineOptions::language` |
| `nvidia/parakeet-unified-en-0.6b` | RNN-T | 128 | 1024 x 24 | 1024 | 600 M | 707 MiB q8_0 | 0.004 q8_0 Vulkan / 0.028 q8_0 Metal | English; offline full-context encoder with an optional Core ML sidecar; cache-aware streaming at 80/160/560/1040 ms chunks with 0-1040 ms right context |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | 128 | 1024 × 24 | 8192 | 600 M | 715 MiB q8_0 / 1.34 GiB f16 | 0.006 q8_0 Metal | About 25 languages, with punctuation and capitalization; optional Core ML offline encoder |
| `nvidia/parakeet-tdt-1.1b` | TDT | 80 | 1024 × 42 | 1024 | 1.1 B | 1225 MiB q8_0 | 0.027–0.079 Metal | English only; no punctuation or capitalization; optional Core ML offline encoder |
| `qvac/TranscribePsy-250M-EU-sub` ([GGUF](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf), [Core ML](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml)) | TDT | 128 | 640 × 24 | 8192 | 254 M | 291 MiB q8_0 / 17 MiB encoder-less + 186 MiB Core ML | 0.016 q8_0 CPU (8 threads, x86-64) / 0.024 q8_0 Hexagon (Galaxy S25, 191 s) | Eight European languages (nl, en, fr, de, it, pl, pt, es) with punctuation and capitalization, distilled from TDT 0.6B-v3; published GGUF loads as a TDT model; Neural Engine encoder and batched TDT decoder sidecars, see [Neural Engine sidecars](docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars); Hexagon NPU, see [Hexagon NPU](docs/backends.md#hexagon-npu) |
| [`qvac/TranscribePsy-250M-EU`](https://huggingface.co/qvac/TranscribePsy-250M-EU) | TDT + CAFE adapters + language gate | 128 | 640 × 24 | 8192 | 254 M trunk | 293 MiB q8_0 / 546 MiB f16 | — | Full 25-language model with CPU, Metal, Vulkan/Android and macOS/iOS Core ML backends ([export, streaming and measurements](docs/transcribepsy-eu-coreml.md)); `--language en` selects the bare trunk, `other` the adapters, and `auto` (default) the gate; convert the checkpoint and both safetensors into one GGUF ([instructions](docs/models.md#transcribepsy)) |
| `nvidia/parakeet_realtime_eou_120m-v1` | RNN-T + `<EOU>` | 128 | 512 × 17 | 1027 | 120 M | 246 MiB f16 / 132 MiB q8_0 | 0.0052 Vulkan | English ASR and native end-of-turn token; Core ML exact-shape encoder |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | Prompt-conditioned RNN-T | 128 | 1024 × 24 | 13087 | 600 M | ~1.3 GiB f16 | 0.108 CPU | Locale-conditioned ASR; empty language selects `auto`; emitted locale tags such as `<en-US>` are dropped from transcript text but kept in token IDs; cache-aware streaming at 80/160/320/560/1120 ms; Core ML exact-shape offline encoder |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer | 80 | 512 × 18 | n/a | 123 M | 263 MiB f16 / 141 MiB q8_0 / 75 MiB q4_0 | 0.0020 Vulkan | Up to four speakers; offline and sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Streaming-trained; sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Sortformer + AOSC | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Audio-Online Speaker Cache preserves slots across long gaps; optional Core ML whole-network diarizer, plus the older exact-shape batch/AOSC encoder sidecars |
| `nvidia/Nemotron-3-Diarization` | Nemotron diarization + AOSC | 128 | 512 × 31 RoPE | n/a | — | 107 MB q8_0 GGUF | 0.0002 q8_0 CUDA / 0.0002 q8_0 Vulkan / 0.0006 q8_0 Metal / 0.0065 q8_0 OpenCL (Adreno 830) / 0.0005 Core ML (Apple M4) | Eight speakers, 10 ms probabilities, native offline and cached streaming; optional Core ML whole-network diarizer |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Whisper-shaped encoder + Qwen3 decoder (text, speakers, timestamps) | 80 | 1024 × 24 encoder, 1024 × 28 decoder | 151936 | 0.9 B | 1.8 GB f16 / 0.98 GB q8_0 / 0.64 GB q5_0 | 0.05–0.39 f16 Metal (2 to 30 min, before the Metal decode graph) / 0.13–0.20 q8_0 Hexagon / 0.26–0.36 q8_0 OpenCL (Galaxy S25, 27 s to 3 min) | Multilingual checkpoint, validated on Spanish and Chinese; English long-form skips spans, as in the reference model; per-request hotwords; separate `moss-transcribe` API and CLI; optional Core ML audio encoder; Hexagon NPU and Adreno OpenCL with q8_0 or f16 |

TDT 0.6B-v3 is multilingual with punctuation/capitalization; TDT 1.1B is
English-only without those features. TranscribePsy EU-sub is a 254M FastConformer-TDT
distilled from TDT 0.6B-v3 for eight European languages and published as
ready-made GGUF and Core ML files; the separate full EU model adds CAFE adapters
and an English/non-English gate for 25 languages. The engine treats both as TDT checkpoints, and
60-second windows with 25% overlap are the automatic long-form defaults
(explicitly selectable with `--window-sec 60 --overlap 0.25`)
(see [CLI](docs/cli.md#long-form-window-flags)). IndicConformer exports CTC by default,
with an optional RNN-T branch; CTC requires a language selection. Use the
[model acquisition and format guide](docs/models.md) for each family's
download and conversion path.

Full **TranscribePsy-250M-EU** streaming uses the same defaults on all backends,
including CPU, Metal, Vulkan and Android. It defaults to 2000 ms chunks
(`StreamingOptions::chunk_ms = 0` selects the model default); an explicit
1000 ms request is preserved. Other ASR models retain their previous 1000 ms
default. Auto push streams buffer the first 20 seconds for the model's gate,
or gate shorter streams on `finalize()`; known `en`/`other` streams emit without
that delay.
See the [streaming API](docs/api.md) for the backend-independent contract.

Nemotron 3 Diarization can also be converted to F16, BF16, Q6_K, Q5_K,
Q4_K, MXFP4, and NVFP4 with `scripts/convert_nemotron_diarization.py --outtype`.
F16/BF16 and the K-quants plus NVFP4 pass the CPU/CUDA offline fixture gates;
MXFP4 fails CPU quality checks. Q4_K cuts model size by 46% on the tested
checkpoint. Q4_K/Q6_K run as one GPU graph on Metal and Vulkan and pass
the offline fixtures; their live four-speaker fixture still fails the 15%
DER gate on both backends. NVFP4 passes offline Vulkan validation but uses
CPU matmuls on Metal, so it is not recommended for Metal acceleration.
See [precision-specific quality and timings](docs/performance.md#nemotron-3-diarization-quantizations),
including streaming limitations and reproduction commands.

Unified RNN-T and Nemotron 3.5 ASR have native cache-aware streaming.
[docs/api.md](docs/api.md) documents trained operating points and context
controls. Nemotron 3 Diarization supports eight speakers and uses cached
long-form processing for offline inputs over 90 seconds; see
[its model guide](docs/nemotron-3-diarization-port.md).

## Experimental Mosaic recipes

[Mosaic](docs/mosaic.md) uses the frozen standard Q4_0/Q8_0 recipe
and ordinary GGML kernels. The supplied 339.6 MB
[Mosaic Compact](docs/mosaic.md#shared-mosaic-and-mosaic-compact-models) also uses
stock Q3_K/IQ3_S blocks; each model supports its own matching optional Core ML
encoder sidecar. The earlier
[packed-affine TDT format](docs/backends.md#experimental-packed-affine-tdt-recipe)
preserves exact integer codes and FP32 minimum/step. Reconstruction is the
default; `PARAKEET_AFFINE_FUSED=1` requests Metal fusion with the
[matching GGML dependency](docs/fused-affine-metal.md#integration-and-pr-handoff).
Unsupported fused backends and older dependency headers are rejected.
Offline TDT is the evaluated scope; packed-affine substage profiling is not
supported. Preserve the recipe/tensor receipts and consult the linked evidence
before interpreting size, timing or accuracy comparisons as deployment results.


## Core ML sidecars

Full **TranscribePsy-250M-EU** supports the English trunk, non-English CAFE adapters,
and automatic language gate on macOS and physical iOS devices. Offline and precomputed-mel calls use fixed ANE sidecars. Callback/push streaming
and true unwindowed inference use additional FP32 programs that preserve native
context: static GPU on macOS where fitting, flexible Core ML CPU on iOS or larger
contexts. See the [full-EU guide](docs/transcribepsy-eu-coreml.md)
for export commands, actual dispatch reporting, startup cost and CPU/Metal/Core ML measurements.
Export from an F16 GGUF on macOS with Xcode; the guide includes its conversion
command. The recommended full-EU baseline uses explicit FP16 encoders and decoder;
six-bit encoders reduce deployment size with the documented quality trade-off.
The Engine supports a [shared-weight multifunction bundle](docs/transcribepsy-eu-coreml.md#shared-weight-runtime-deployment)
in place of the three fixed ANE encoder bundles, including Core ML-only loading.
Routes still load eagerly; disk sharing does not imply lower resident memory or startup time.
An [experimental export harness](docs/transcribepsy-eu-coreml.md#trying-route-sharing-exports)
compares loading subsets, shared-weight multifunction packages and a route-input
encoder; publish runtime bundles with the production merger described in the guide.
The [harness regression tests](examples/ios-coreml-bench/README.md#route-harness-regression-tests)
cover reporting fixtures and native prediction/loading without ASR checkpoints,
including malformed manifests and duplicate WAV stems.


Optional Apple sidecars support eligible CTC, IndicConformer, Unified RNN-T,
TDT, EOU, Nemotron 3.5 ASR and Sortformer v2.1 encoder paths, and the
MOSS-Transcribe-Diarize audio encoder. Input routing
is model-specific; native cache-aware ASR streaming stays on ggml. Missing or
incompatible sidecars fall back to ggml. Export, placement and routing
contracts are in [docs/backends.md](docs/backends.md#core-ml-encoder-sidecar)
and, for MOSS-Transcribe-Diarize, in
[docs/moss-transcribe.md](docs/moss-transcribe.md#core-ml-encoder-sidecar).

Standard TDT sidecars use ggml for audio beyond their compiled capacity (typically
15 seconds), preserving native encoder context. Only exports with a validated
seam policy use short Core ML windows. Standard TDT sidecars retain native
weights even with `PARAKEET_COREML_ONLY=1`, and encoder-less GGUFs paired with
these sidecars are rejected at load time. TDT push streams, except full TranscribePsy
EU, default to 30-second chunks and reject smaller explicit chunks to prevent
phrase loss. PCM writes can still be arbitrarily small; `finalize()` processes
the remaining audio. See the [streaming contract](docs/api.md).

For standard offline TDT models, the ggml encoder masks right-padding in
attention and zeros padded activations before depthwise convolution. Valid
encoder frames therefore do not depend on the padded tail. Push streaming
still re-encodes windows, so the model's accuracy can vary with window context.

Fixed TDT exports now accept `mel` plus `mel_valid_mask` and require this updated
runtime. Legacy single-input sidecars remain loadable but need re-exporting to
isolate padding; updating runtime code alone does not fix those old artifacts.
See the [fixed-TDT export contract](docs/backends.md#core-ml-encoder-sidecar).
TDT models can additionally run on the Apple Neural Engine end to end: a
fixed-window encoder export with padding masks and a batched TDT decoder
sidecar (`<model>-decoder.mlmodelc`). With the decoder sidecar the engine cuts
long recordings at pauses into independent segments and decodes 16 of them per
Neural Engine dispatch, and it allocates the sequential ggml decoder only when a
call needs it. A GGUF without encoder weights, such as the published
TranscribePsy Core ML distribution, loads only with its encoder sidecar. Create
that GGUF from the full EU-sub file with `scripts/strip-encoder-gguf.py`; metadata
and retained quantized weights are preserved. The decoder exporter supports
separate `--nbits-out` compression for the output projection, including its
8198-channel shape. See [conversion and stripping](docs/models.md#transcribepsy) and
[Neural Engine sidecars](docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).

Nemotron 3 Diarization and Sortformer v2.1 can also run their whole network,
encoder and speaker head, on fixed-capacity diarizer sidecars named
`<model>-diarizer-<frames>.mlmodelc`. Offline, long-form, and live calls use
the smallest capacity that fits and fall back to ggml above it. On an Apple M4
they run 1.4-2.7x faster than ggml Metal. On an M3 Ultra, whose GPU is faster,
they only tie with it (0.70-1.17x), so leave them out on Max and Ultra
machines; see [the diarizer guide](docs/backends.md#core-ml-diarizer-sidecars).

### Background loading, waiting and status

Encoder, bypass-encoder and diarizer sidecars load in the background. The
`Engine` constructor returns as soon as the GGUF is loaded. Until a sidecar
is ready, every call runs on ggml with the same plan as a ggml-only build, and
the result reports `coreml_pending = true`. The backend is chosen once per call
(one transcription, one diarization or one streaming window), so a sidecar
that becomes ready part-way through a call is used from the next call. A
prediction that fails retires the sidecar for the rest of the process, and the
call finishes on ggml (`encoder_backend` is `"mixed"` when some windows had
already run on Core ML). An input the sidecar cannot take, such as one longer
than a fixed capacity, runs that call on ggml and keeps the sidecar. Two loads
still happen inside the constructor: the TDT decoder sidecar, and an encoder
sidecar that owns the encoder (a GGUF without encoder weights, or
`PARAKEET_COREML_ONLY`). A standard TDT sidecar without a validated seam
policy never owns the encoder. With `PARAKEET_COREML_ONLY` and a full GGUF, the
GGUF keeps its encoder weights and the sidecar loads in the background like any
other; when the bundle's `metadata.json` shows it is not the fixed-window
Neural Engine export, the constructor does not load it first (a bundle without
metadata is loaded once in the constructor to read its seam context). An
encoder-less GGUF with such a sidecar is rejected. Full TranscribePsy-250M-EU keeps its own Core ML
programs outside this background runtime: its English, other-language and
gate programs load in the constructor, its precise programs on first use, and
a program that cannot serve a call falls back to ggml, also under
`use = require`. `use = off`, `PARAKEET_COREML_DISABLE` and `compute_units`
apply to it; `wait`, `prewarm` and the status methods do not.

`EngineOptions::coreml` (`parakeet::CoremlOptions`, in
`include/parakeet/coreml_options.h`) controls this. Every field is unset by
default and resolves, in order, from the field, the environment, then the
default. `PARAKEET_COREML_DISABLE` forces `use = off` and overrides
everything.

| Field | Values (default first) | Meaning |
|---|---|---|
| `use` | `automatic`, `off`, `require` | `require` (also `PARAKEET_COREML_STRICT`) waits like `wait = call`, and throws an exception derived from `std::runtime_error` instead of running on ggml when a sidecar cannot serve the call: it failed to load or timed out, the input is longer than a fixed-capacity encoder (for example with long-form windowing disabled, or a standard TDT input or push-stream window past the sidecar's capacity, which is never split into Core ML windows), the sidecar rejects the shape, or a prediction fails. `stream_start` refuses a push stream whose windows (left context, chunk and lookahead) are longer than a fixed-capacity encoder, before any audio; with a 15 s standard TDT sidecar that is every offline TDT push stream, since those use 30 s chunks. A diarization pass throws when no diarizer capacity that covers it can serve it: each one failed, or is a template capacity that is not generated (`no_template`, `unsupported(...)`, or `disabled` with `generate = never`), and when it is longer than every capacity (under `automatic` that pass runs on ggml). This changes `require` for a Nemotron 3 Diarization or Sortformer v2.1 model with no adjacent bundles in a build that embeds the templates (see [Generated sidecars](#generated-sidecars)): with no `cache_dir` (the library's default) every pass of up to 1125 frames (410 for Sortformer v2.1) now throws `unsupported(no_cache_dir)`, where it used to run on ggml; set `cache_dir` to generate the capacities, or use `automatic`. Builds without the generator (`PARAKEET_COREML_GENERATE=OFF`, iOS) have no template capacities and still run those passes on ggml. Calls that never use a sidecar, such as model families or shapes without one, still run on ggml. |
| `wait` | `none` on macOS and `call` on iOS; `construct` | `none` never blocks a call. `construct` makes the constructor wait for the construction group. `call` makes each call wait until its sidecar is ready or has failed. |
| `prewarm` | `predicted`, `none`, `minimal`, `all` | Sidecars requested at construction. `minimal` is the encoder (and bypass encoder), or for a diarization model the smallest diarizer capacity that can serve a pass (the encoders when every capacity is status-only, for example with no cache directory). `predicted` requests the same at construction, and each call also requests what it needs. `all` requests every diarizer capacity. |
| `compute_units` | the stage's own; `all`, `cpu_and_ane`, `cpu_and_gpu`, `cpu_only` | Also `PARAKEET_COREML_COMPUTE_UNITS`. Applies to the encoder, bypass-encoder, diarizer and TDT decoder sidecars. Unless the field is set, `PARAKEET_COREML_DECODER_COMPUTE_UNITS` overrides it for the TDT decoder. |
| `generate` | `background`, `never` | `background` generates a sidecar that has no adjacent bundle from the GGUF into `cache_dir`, on the runtime's background queue, when the library embeds a template that matches the GGUF (see [Generated sidecars](#generated-sidecars)). `never` turns generation off; adjacent bundles still load. |
| `cache_dir` | `QVAC_COREML_CACHE_DIR`, else none | Where generated sidecars and warm-load hints are kept. Without it nothing is generated and no hints are kept. |
| `strict_wait_ms` | unbounded | Upper bound on the wait under `use = require`. |
| `memory_budget_bytes` | none | The memory budget the caller checked with `parakeet-fit-params`. When it is set, and the ggml-only fit passes but the fit including the Core ML sidecars does not, the sidecars are disabled for this engine and it runs on ggml. Pass the totals of that check in `EngineOptions::coreml_fit` (`ggml_required_bytes` and `coreml.bytes` from `fit_params`) so the engine applies the same verdict; without them it projects fit itself with the default audio length, margin and streaming chunk. |

The engine reports progress through these methods:

- `Engine::coreml_status_json()` returns one JSON entry per sidecar: `stage`,
  `state` (`queued`, `loading`, `ready`, `failed`, `retired`, ...), `route`
  (`prebuilt`, `cached`, `generated` or `none`), `compute_units`, `label`,
  `path`, timings and the warm hint. It returns `{}` on builds without
  `PARAKEET_COREML`.
- `Engine::coreml_wait(timeout_ms)` blocks until the construction group has
  finished (no limit when `timeout_ms < 0`), and returns true when every
  sidecar in it is ready.
- `Engine::coreml_prewarm(scope, wait)` requests more sidecars (for example
  every diarizer capacity) and, with `wait`, blocks until they finish.
- `Engine::coreml_construction_json()` returns the construction group's result.
- `Engine::encoder_on_coreml()` reports whether a sidecar is ready now. It is
  false while a load is in flight.
- `EngineResult::encoder_used_coreml`, `encoder_backend` and `coreml_pending`
  report where each call ran. `DiarizationResult` has the same fields.

The `parakeet` CLI exposes the same controls:

| Flag | Values |
|---|---|
| `--coreml-use MODE` | `auto` (default), `off` or `require` |
| `--coreml-wait MODE` | `none` (default), `construct` or `call` |
| `--coreml-prewarm SCOPE` | `none`, `minimal`, `predicted` (default) or `all` |
| `--coreml-compute-units U` | `all`, `cpu_and_ane`, `cpu_and_gpu` or `cpu_only` |
| `--coreml-generate MODE` | `background` (default) or `never` |
| `--coreml-cache-dir DIR` | default `QVAC_COREML_CACHE_DIR`, else `~/Library/Application Support/qvac-speech/coreml` |
| `--coreml-status-json PATH` | writes the status JSON on exit; `-` writes to stdout |

```bash
parakeet --model tdt-0.6b-v3-q8_0.gguf --wav talk.wav \
    --coreml-wait construct --coreml-prewarm all --coreml-status-json -
```

`--require-coreml` (benchmark mode) implies `--coreml-wait construct`, so the
benchmark measures the sidecar and not the ggml calls made while it loads.

When `--coreml-use require` refuses a call, the CLI prints `error: <reason>`
to stderr and exits with status 1.

Process exit with Core ML: a process that exits while a sidecar is still compiling or loading in the
background waits for that load (by default at most 15 s on macOS, 2 s on iOS) when its main thread is armed.
The main thread is armed automatically when it makes the engine; a host that makes engines on another thread
(as the qvac addons do) calls `coreml_arm_exit_guard()` on its main thread, or the terminal
`coreml_shutdown_for_exit()` right before the process exits (`parakeet/coreml_exit.h`). See
[the runtime's README](../coreml/README.md#process-exit).

With `--verbose`, a diarization run also prints where its passes ran:

```text
[diarize] backend=coreml-ane pending=0
```

`backend` is the call's `encoder_backend`: `ggml`, the Core ML label of the
diarizer capacities that served it (`coreml-ane`, `coreml-all`,
`coreml-cpu-gpu` or `coreml-cpu`), or `mixed` when some passes ran on Core ML
and others on ggml. `pending=1` (`coreml_pending`) means at least one pass ran
on ggml because the capacity that covers it was still being generated or
loaded.

### Generated sidecars

macOS builds with `PARAKEET_COREML=ON` also compile the Core ML generator
(`PARAKEET_COREML_GENERATE`, default `ON` on macOS and `OFF` on iOS) and embed
these templates in the library:

| Template | Sidecar | Capacity |
|---|---|---:|
| `nemotron-diar-f358` | Nemotron 3 Diarization diarizer | 358 |
| `nemotron-diar-f745` | Nemotron 3 Diarization diarizer | 745 |
| `nemotron-diar-f1125` | Nemotron 3 Diarization diarizer | 1125 |
| `sortformer-diar-f410` | Sortformer v2.1 (`sortformer-streaming-v2.1-aosc`) diarizer | 410 |

When a capacity has no adjacent bundle, the engine generates it from the GGUF
alone: it writes the package's weights from the GGUF's tensors (dequantised by
ggml and cast to fp16), compiles the package on the device and publishes the
result in `cache_dir`. A later load of the same GGUF finds it there. This runs
on the runtime's background queue after the model has loaded; until the
capacity is ready, diarization runs on ggml and reports `coreml_pending`. The
generated bundle has the same weights, byte for byte, as the exporter's
bundle for that GGUF.

Generation needs macOS 15, a `cache_dir` (the CLI defaults to one) and
`generate = background`. The GGUF must match the template: for Nemotron 3
Diarization architecture `sortformer` and version `v3`, for Sortformer v2.1
architecture `parakeet-ctc`, model type `sortformer` and variant
`sortformer-streaming-v2.1-aosc`, with the encoder settings the exporter
accepts (symmetric convolution, `conv_context_style` `regular`, regular
full-context attention, non-causal 8x subsampling; any other value, such as a
causal convolution, gets `no_template` and runs on ggml); then the template's
hyperparameters and
every tensor's name and shape, with matrices in f32, f16, bf16, q8_0, q6_k,
q5_k or q4_k (Sortformer v2.1 also q4_0, and its depthwise convolutions in
f32, f16 or bf16). The
templates are validated for the `cpu_and_ane` compute units (the diarizers'
default), so another `compute_units` value turns generation off for them. An
adjacent bundle always wins. `Engine::coreml_status_json()` reports
`route: generated` for a bundle generated in this process, `cached` for one
found in the cache, and `generation.supported` with the reason when generation
is off. A capacity that is not generated keeps an entry with `route: none`,
which the construction group lists under `skipped_route_none`: state
`no_template` when the GGUF does not match the template,
`unsupported(unvalidated_units)` when the compute units are not validated, and
the reason generation is off for the whole instance otherwise (for example
`unsupported(no_cache_dir)`, or `disabled` with `generate = never`); every such
capacity is listed. When every capacity is suppressed for its units,
`generation.reason` is `unvalidated_units`. Such a capacity never serves a
pass; under `use = require` a pass it covers throws instead of running on ggml.
Each generated capacity takes about 200 MB in the cache (232 MB for
Sortformer v2.1). `parakeet-fit-params` charges the generated capacities'
Core ML memory in the Core ML-inclusive verdict (every generated capacity
under `prewarm = all`, otherwise those the workload's calls, and any shorter
single pass, can load, or fall back to when a capacity fails; for Sortformer
v2.1, every capacity up to the full AOSC step slab of a default stream, 409
frames, since the slab grows with the speaker cache and FIFO, together with
any adjacent prebuilt bundles on the way); see
[docs/backends.md](docs/backends.md#generated-diarizers).
The generated Sortformer v2.1 capacity's measured envelope is 259 MiB: 137 MiB
resident plus a 122 MiB preparation allowance. Its stage constants retain the
25% measurement margin, including the latest M4 Max CI runs.

## Backend selection

`parakeet --backend NAME` and `EngineOptions::backend` select the runtime device
(`auto`, `cpu`, `cuda`, `vulkan`, `opencl`, `webgpu`, `metal`, `hexagon`, `xdna` or an exact ggml device name);
`parakeet-fit-params --backend NAME` and `FitOptions::backend` project memory for
the same device. Long-form encoder graph projections conservatively count up to
three graphs, while windowed inference retains one; see
[memory-fit preflight](docs/cli.md#memory-fit-preflight-parakeet-fit-params).
The fit result names it by its ggml device name, and a device
that shares host memory and reports none of its own carries the CPU's figures;
see [memory fit results](../README.md#memory-fit-results). Explicit requests fail if unavailable. Hexagon is experimental and requires
a staged DSP library; see [docs/backends.md](docs/backends.md) for names,
automatic placement and validation coverage,
[Hexagon NPU](docs/backends.md#hexagon-npu) for the FastConformer models
(TranscribePsy measurements included), and
[docs/moss-transcribe.md](docs/moss-transcribe.md#hexagon-npu) for
MOSS-Transcribe-Diarize on the NPU.

AMD `--backend xdna` uses Fabric LLM's native GGML XDNA backend with the existing
GGUF model and CPU fallback. F16/BF16 matrix kernels can run on the NPU;
Q8_0 matrices use the host. This requires the compatible GGML dependency and
matching XRT/IRON artifacts; see [AMD XDNA](docs/backends.md#amd-xdna-npu).

## Performance

Eligible offline TDT, Unified RNN-T and Nemotron ASR calls consume encoder output
on the GPU, avoiding its download and re-upload for decoding. Nemotron's locale
projection also stays on the GPU. The handoff uses ordinary ggml backend APIs
and falls back to host output where needed, including encoder tensors with
extra padding frames; see
[device encoder handoff](docs/backends.md#device-encoder-handoff).

`RTF = inference_time / audio_duration`; lower is faster. Dated CI snapshots,
multi-machine campaigns, GPU diarization, accuracy, build pins and timing
methods live in [docs/performance.md](docs/performance.md).

## Documentation

Cross-platform CI selects affected engines and caches Windows compiler outputs;
see the [CI guide](../docs/CI.md).

Run model-free checks with `ctest --test-dir build-parakeet -L unit --output-on-failure`.
For model coverage, use `scripts/run-fixture-tests.py` and require the relevant test
names; missing fixtures and runtime skips cannot satisfy a required test. See
[the test guide](docs/testing.md) for fixture setup and CI inputs.

| Topic | Where |
|---|---|
| Build modes, CMake options, repository layout | [docs/build.md](docs/build.md) |
| Backends, runtime selection, Core ML encoder sidecar | [docs/backends.md](docs/backends.md) |
| Model download and GGUF conversion | [docs/models.md](docs/models.md) |
| Public C++ API and Sortformer AOSC | [docs/api.md](docs/api.md) |
| CLI, microphone examples, memory-fit preflight | [docs/cli.md](docs/cli.md) |
| Tests and NeMo parity | [docs/testing.md](docs/testing.md) |
| MOSS-Transcribe-Diarize (conversion, CLI, API, tests) | [docs/moss-transcribe.md](docs/moss-transcribe.md) |
| Benchmark method, CI snapshots, decode optimization history | [docs/performance.md](docs/performance.md) |
| Desktop benchmark coverage (CTC, IndicConformer, TDT, Unified, EOU, Nemotron ASR, Sortformer batch and streaming, Nemotron 3 Diarization, MOSS-Transcribe-Diarize), metrics and runner limits | [Benchmark coverage](../docs/PERFORMANCE.md#benchmark-coverage) |

## License

Code is Apache-2.0. NVIDIA Parakeet CTC, Unified RNN-T, TDT and Sortformer
weights are CC-BY-4.0. [IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
weights are MIT; [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
and [TranscribePsy EU-sub](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub) and
[full EU](https://huggingface.co/qvac/TranscribePsy-250M-EU) weights
are Apache-2.0. `parakeet_realtime_eou_120m-v1` uses the
NVIDIA Open Model License. Nemotron 3.5 ASR Streaming 0.6B and Nemotron 3
Diarization use OpenMDW-1.1; redistribution of their weights must retain the
license and applicable origin notices.
No weights are shipped by this repository. [NOTICE](NOTICE) lists the
reference implementations, dependencies and model-weight sources.

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
