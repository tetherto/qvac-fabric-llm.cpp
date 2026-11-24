# Pocket TTS integration and validation archive

Historical records from the September 2026 integration. Earlier timings,
rollout blockers and test counts are superseded where later sections say so;
they are not a current release-status claim. Current native usage and limits
are in the [Pocket guide](../pocket-tts.md).

The native implementation includes a native CPU text-to-waveform engine with streaming
24 kHz mono PCM. The public C++ API is `tts-cpp/pocket/engine.h`; `pocket-cli`
uses that API. Native inference and the addon/public SDK path work; package
rollout and broader platform validation remain in progress.

Implemented in `src/pocket/flow_lm.{h,cpp}`:

- Strict versioned GGUF loading, with metadata and tensor-shape/type validation
  before backend allocation. Uses Fabric's shared ggml and CPU backend registry.
- Text embedding lookup from supplied token IDs, conditioning prefill, a causal
  transformer with RoPE and bounded linear KV state, audio BOS, and EOS logits.
- Two-time-condition LSD flow sampling with caller-supplied noise, 1–64 steps,
  latent denormalization, and reset/reuse of the same resident weights.
- F32 arithmetic throughout. F16 GGUF is a **storage option**: matrices expand
  to F32 at load, so it does not reduce resident weight memory. This avoids the
  CPU F16 matmul path rounding activations before every projection.

The transformer uses PyTorch's tanh GELU formula directly. The CPU ggml GELU
lookup table is not sufficiently precise for the reference tier. Likewise,
Pocket's time-embedding class named `RMSNorm` divides by unbiased, centered
sample variance; substituting ordinary RMS normalization is incorrect.

Each instance is serialized by its caller. Invalid input does not advance
the cache. A failed backbone execution discards the current prefix. `reset()`
sets the logical cache position to zero; stale rows are never visible.
The engine owns prepared-voice conditioning, noise/temperature policy, EOS
thresholds, and text splitting. Model-load and graph exceptions currently
propagate as `std::runtime_error`.

Implemented in `src/pocket/mimi.{h,cpp}`: causal SEANet convolutions,
transposed convolutions, depthwise upsampling, the two-layer streaming
transformer with a 250-frame attention window, and the reference-audio encoder.
Decode accepts 1–16 latent frames per call and retains convolution/KV history.
Encoder calls use independent state and pad the reference to a codec frame.
The released without-voice-cloning checkpoint has zeroed SEANet encoder weights;
encoding with this checkpoint is explicitly rejected. Prepared voices do not
need the encoder. A separate random-weight upstream fixture tests non-zero
encoder math; this does not establish voice-cloning quality.

## Conversion

For a complete bundle, supply explicit local upstream assets. The output
directory must not already exist; it is published after all stages succeed.

```sh
python engines/tts/scripts/convert-pocket-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --tokenizer /path/to/tokenizer.model --voice /path/to/alba.safetensors \
  --output /path/to/pocket-bundle
cmake -S engines/tts -B build/pocket -DTTS_CPP_BUILD_EXECUTABLES=ON
cmake --build build/pocket --target pocket-cli
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --output /tmp/pocket.wav
```

The bundle contains `flow-lm.gguf`, `mimi.gguf`, `frontend.json`, `voice.gguf`
and a SHA-256 manifest. Use a prepared voice from the same upstream checkpoint
revision as the weights: legacy voice files do not contain a checkpoint hash,
so the converter binds the caller-selected assets, rather than proving the
provenance of arbitrary voice files. Native loading rejects mixed bundle
hashes and invalid cache lengths, types or payloads.

The identity SentencePiece unigram frontend includes byte fallback, Unicode
case/whitespace/digit metadata, punctuation repair and decimal-aware sentence
splitting. It matches 322 upstream corpus cases. Generate the fixture using
`dump-pocket-frontend-reference.py --tokenizer /path/to/tokenizer.model
--output /path/to/pocket-bundle/frontend-corpus.json`, then configure
`TTS_CPP_POCKET_MODEL_DIR` to enable frontend and public-engine CTest cases.

The persistent engine restores the chosen voice for each text chunk and runs
FlowLM/Mimi on separate CPU workers with at most 16 queued latent frames.
Callbacks run on the calling thread. Returning false or calling `cancel()`
ends streaming; concurrent/reentrant synthesis throws. Subsequent calls reset
the voice, codec and RNG. The Fabric seed uses a specified Box–Muller transform
and is not equivalent to PyTorch's seed. A voice-cloning-capable checkpoint can
use `--reference-audio` instead of the prepared voice; the public checkpoint
tested here explicitly rejects that path because its encoder is disabled.

`frames_after_eos` (CLI: `--frames-after-eos`) accepts an explicit 0–100-frame
tail, or -1 for the checkpoint/text default. Synthesis and memory preflight
reserve this tail in addition to the text-duration estimate. All chunks must
fit the context before any PCM is emitted; detecting EOS near the estimate's
end still permits the complete tail. If EOS never arrives within the original
text estimate, generation fails rather than spending the reserved tail.

Review regression coverage includes first/last-frame EOS with tails 0, 1, 3,
5 and 100; missing EOS; real streaming with a 100-frame tail; exact context
boundaries and rejection before the first callback; and memory pricing of
default and explicit tails. The nine Pocket CTests pass with the local model
and upstream FlowLM/Mimi fixtures, including F16 storage parity. After updating
to current `master`, all 124 enabled local CPU CTests pass; 100 tests without
the required fixtures/backends are disabled. CLI validation also covers argument
rejection, explicit tails, warmup/multiple runs, WAV/JSON output and memory fit.

`scripts/convert-pocket-flow-lm-to-gguf.py` takes a local Kyutai checkpoint and
its YAML configuration. It selects `flow_lm.*` tensors from the safetensors
bundle, leaving Mimi out. BF16 source weights expand to F32. Unknown/missing
FlowLM weights, unsupported flow types, and invalid dimensions fail explicitly.
The output is replaced only after a successful complete export.

GGUF architecture: `pocket-tts-flow-lm`; `pocket.schema_version = 1`.
Keys under `pocket.*` record dimensions, RoPE base, BOS policy, flow type,
source SHA-256, and the input configuration. Tensor names retain upstream
names with the `flow_lm.` prefix removed. PyTorch tensor dimensions are reversed
by GGUF without transposing data. `bos_before_voice` is flattened to a vector.
This format is specific to Fabric's FlowLM stage and is not interchangeable
with llama.cpp's full Pocket TTS model/mmproj files.

```sh
python3.12 -m venv /tmp/pocket-venv
/tmp/pocket-venv/bin/pip install -r engines/tts/scripts/pocket-flow-requirements.txt
/tmp/pocket-venv/bin/python engines/tts/scripts/convert-pocket-flow-lm-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /path/to/flow-lm-f32.gguf
```

Python 3.11+ and PyTorch are conversion/reference dependencies only. The C++
core does not load Python, safetensors, ONNX Runtime, or a second ggml copy.

## Reproduce numerical validation

The fixture generator calls Kyutai's actual neural modules at commit
`0c2db3bdea7c991c568989cc11b503f14483fabc`, pinned by the requirements file.
It verifies that revision before generating data. Its default small-weight
fixture needs no downloaded model or tokenizer. Its PyTorch oracle recomputes
full prefixes while the native implementation uses cached decoding.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --output /tmp/pocket-reference
/tmp/pocket-venv/bin/python engines/tts/test/pocket/test_converter.py /tmp/pocket-reference

# GGML_PREFIX must contain an installed qvac-ext-ggml speech build compatible
# with the rest of this checkout, including the CPU registry graph-planner export.
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_EXECUTABLES=OFF \
  -DTTS_CPP_BUILD_TESTS=ON -DTTS_CPP_POCKET_REFERENCE_DIR=/tmp/pocket-reference
cmake --build build/pocket --target test-pocket-flow-lm test-pocket-load
ctest --test-dir build/pocket -R test-pocket- --output-on-failure
```

The native tests check embedding lookup, prefill, four cached audio steps,
EOS, 1/2/4/16 flow steps, denormalization, chunked prefill, reset, invalid
inputs, and a trajectory that feeds native sampled latents back into the LM.
The loader test needs no model files. Numerical tests are visibly disabled
by CMake when fixtures are absent; no reference downloads occur during build.

For a real checkpoint, use the same generator with `--weights` and `--config`:

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /tmp/pocket-real-reference
build/pocket/test-pocket-flow-lm /tmp/pocket-real-reference/flow-lm-f32.gguf \
  /tmp/pocket-real-reference 0.0001
```

The initial implementation was checked on the released `english_2026-04`
without-voice-cloning bundle at Hugging Face revision
`d29db7978e464fb90cb3359ee0c69a273b9142cc`. Both storage variants compute in F32;
the F16 test permits storage-rounding error against the original checkpoint.
Fixtures deliberately supply conditioning embeddings and noise directly:
they establish stage parity, not tokenizer, voice-cloning, or speech quality.

Validation on 2026-09-10: macOS / Apple M2, PyTorch 2.14.0, CPU ggml from
`qvac-ext-ggml@speech` revision `157b299f16b2582f56ad5facc7e3ee13401e4c70`.
The complete `tts-cpp` static library built successfully, all three Pocket
CTest cases passed, and all 12 converter regression tests passed. Tests linked
the actual library objects, including their release optimization flags.
The real-checkpoint four-frame native trajectory and stage comparisons had
maximum absolute error `3.05e-5` for F32 storage and `3.03e-5` for F16 storage
expanded to F32. These short deterministic traces do not establish long-form
audio quality or real-time performance.

## Waveform validation and initial upstream baseline

The Mimi converter accepts the released English codec configuration and
rejects incompatible sample rates, strides and transformer settings before
publishing an artifact. All codec weights and computation are F32. Both codec
and FlowLM loaders validate every weight payload for finite values.

```sh
python engines/tts/scripts/convert-pocket-mimi-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /path/to/mimi.gguf
python engines/tts/scripts/dump-pocket-mimi-reference.py \
  --config /path/to/config.yaml --output /tmp/pocket-mimi-reference
python engines/tts/test/pocket/test_mimi_converter.py /path/to/config.yaml
cmake -S engines/tts -B build/pocket \
  -DTTS_CPP_POCKET_MIMI_REFERENCE_DIR=/tmp/pocket-mimi-reference
cmake --build build/pocket --target test-pocket-mimi
ctest --test-dir build/pocket -R test-pocket-mimi --output-on-failure
```

`scripts/benchmark-pocket-reference.py` takes explicit local `--weights`,
`--config`, `--tokenizer`, `--voice` (upstream prepared-voice safetensors), and
`--output` paths. It runs one warmup and three warm generations, writes an
upstream WAV and a JSON report, then separately captures latents and PCM for
native decoder validation. It checks the upstream commit and records source
hashes. Trace recording is excluded from measured runs. For real-waveform
parity, run `test-pocket-mimi /path/to/mimi.gguf /path/to/benchmark-output`.

Initial CPU results on Apple M2 / 24 GiB, English April 2026 checkpoint,
Alba prepared voice, temperature 0.3, one flow step, seed 1234:

| Measurement | Result |
| --- | --- |
| Upstream PyTorch, 5.52 s generated audio | 0.985–1.000 s total, 62–67 ms first audio |
| Upstream real-time factor (generation time / audio duration) | 0.178–0.181 |
| Native C++ engine, text to 5.52 s generated audio, three warm runs | 1.75–1.83 s total, about 187 ms first audio |
| Native real-time factor | 0.317–0.332 |
| Native Mimi on the exact upstream latents | max PCM error 1.42e-5, SNR 109.4 dB |
| Native variable chunks vs one frame per call | max PCM error 5.07e-7 |
| Non-zero random-weight encoder vs PyTorch | max latent error 2.50e-7 |
| Entire text-to-waveform path, zero temperature, same prepared voice | 132480 samples each, waveform SNR 35.8 dB, max sample difference 0.0396 |

Both engines use separate LM/decoder workers with one compute thread each.
The C++ native path currently generates about 3.0–3.2 times real time compared
with 5.5–5.6 times real time upstream. These initial measurements cover one
prompt and voice. The broader five-prompt measurements below supersede this initial timing sample.
The seed values match, but RNG algorithms differ, so they produce different
waveforms. Fixed-latent decoder replay independently validates numerical parity.
The zero-temperature comparison removes the RNG difference; its residual
waveform difference is substantially larger than decoder-only replay, so it
must not be described as bit-identical end-to-end output. Longer autoregressive
trajectory and multi-prompt quality checks are still needed.

An additional local faster-whisper 1.2.1 / tiny.en CPU transcription check
returned the intended words for the upstream sample, native-decoder replay,
and fully native text-to-speech sample: “Hello, this is Pocket TTS. We are bringing natural local speech
generation to fabric.” This single sample and automated transcription do not
replace listening tests across varied texts and voices.

Adversarial review found and fixed non-finite EOS handling and converter/native
architecture acceptance mismatch. Regression checks cover finite payload
overflow, exact LM context capacity, decoder chunk seams beyond the attention
window, encoder/decoder state independence, invalid input, and recovery after
failed execution.

## Historical rollout checklist

1. The initial public Bare-direct SDK client/server path was verified. Later
   registry-backed builds and desktop socket IPC validation are recorded below.
2. Android and physical-device coverage, plus mobile SDK transport. Native iOS
   Simulator synthesis and the packaged addon in an iOS Bare Kit worklet are
   verified below. Native fit is not yet exposed through the SDK model-fit package.
3. Broader listening evaluation across voices and texts. Deterministic follow-up
   below resolved the ASR contraction discrepancy for zero-temperature sampling;
   full autoregressive output still has measured numerical differences.
   The manual development build took about twice the upstream inference time;
   the final pinned-registry measurements below supersede that result.
4. Voice cloning requires a separate encoder-enabled checkpoint; the public
   checkpoint used here intentionally disables its encoder.

Native synthesis and lifecycle validation now pass. Training remains a separate task.

Reference: [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts/tree/0c2db3bdea7c991c568989cc11b503f14483fabc).


## Addon and SDK validation (2026-09-10)

The `@qvac/tts-ggml` adapter now selects `engine: "pocket"`, resolves the four
bundle files, streams Int16 PCM, reports first-audio latency, and supports
cancellation and reload. The SDK accepts `ttsEngine: "pocket"`, uses the main
model source for FlowLM, and resolves `mimiModelSrc`, `frontendSrc`, and `voiceSrc`
(or `referenceAudioSrc`). Native output resampling accepts 8000..192000 Hz;
8 kHz and 44.1 kHz streaming/batch duration and waveform tests pass.

Validation completed locally:

- 72 addon JS unit tests / 263 assertions, including invalid settings, reload
  activation/teardown failures, and delayed streaming input across reload/cancel.
- Real Bare addon integration covering chunk order, first PCM,
  cancellation, deterministic recovery, 44.1 kHz reload, empty-input failure,
  and recovery after failure.
- SDK plugin artifact resolution plus actual batch and streaming `textToSpeech`
  operations: 2 tests / 15 assertions.
- Public SDK client through the real Bare-direct RPC path: 20 assertions for
  model loading, short and >300-character batch/stream PCM parity, duplex output
  before input ends, cancellation of plain and open-input duplex streams,
  deterministic recovery, and unload. This does not validate desktop socket IPC.
- TTS request lifecycle: 10 tests / 46 assertions covering cancellation before
  admission, cancelled queued requests, native cancellation drain before successor
  admission, abandonment, delayed startup in all three TTS modes, cleanup failures,
  and failed lifecycle reporting. Public TTS currently exposes broad model
  cancellation; request-ID cases in these tests exercise the internal registry.
- Pocket SDK schema tests: 24 assertions; existing TTS schema tests: 112 assertions.
- Addon declaration typecheck and changed JS lint checks pass. Full SDK typecheck
  still reports missing OCR/BCI package declarations and decoder API errors in
  other files; it is not a green full-repository check.

Adversarial reviews found and drove fixes for failed replacement activation,
old-instance teardown errors, stale async text dispatch into replacement
instances, and cancellation completion ordering. New requests now wait until
both native cancellation and the old terminal JS event have completed.

The public client test also found and fixed undefined optional artifact paths
being rejected during model loading. TTS handlers now register with the request
registry, forward aborts through the wrapper's cancellation barrier, and serialize
same-model requests. They recheck cancellation after awaited model startup so an
abort cannot miss a later-dispatched job. Plain Pocket streaming uses native PCM
streaming; explicit sentence streaming retains the SDK's sentence orchestration.
Adversarial re-review found no remaining concrete issue in that lifecycle helper.

A WAV generated through the public client contains 78,720 samples at 24 kHz
(3.28 seconds). It is finite and non-silent, with peak 0.44144 and no clipping;
`tiny.en` recovers “Hello! Pocket TTS now speaks through the fabric client.”
The signal and transcription checks are automated, not a subjective listening score.

From `packages/sdk` in the SDK checkout, after building the local Pocket addon
and making it available as `@qvac/tts-ggml`, reproduce with:

```sh
npm run build:bare
QVAC_POCKET_MODEL_DIR=/absolute/path/to/pocket-bundle \
QVAC_POCKET_CLIENT_AUDIO=/absolute/path/to/pocket-sdk-client.wav \
  bare scripts/run-pocket-client-test.js '{"HOME_DIR":"/tmp/qvac-pocket-client-test"}'
bare test/dist/test/bare/tts-request-lifecycle.test.js
```

The harness gives generated `test/dist` its own package scope so the public
client and worker resolve the same compiled registries. Locally validated with
Bare 1.28.6 from the SDK dependencies. Global Bare 1.21.3 cannot load the SDK's
installed `bare-module` 6.2.0; a separately installed Bare 1.32.0 crashed in that
module's native loader before Pocket was loaded. The full SDK build remains
blocked by the unrelated OCR/decoder type errors noted above; local validation
used TypeScript's emitted test output followed by `tsc-alias`, with no Pocket
TypeScript diagnostics.

## Five-prompt warm CPU comparison

Apple M2, one compute thread per worker, two workers, Alba prepared voice,
24 kHz, one LSD step, temperature 0.3, seed 1234. Each implementation runs
sequentially, with one warmup and three measured runs. Numbers below are medians.
The output lengths differ slightly because Fabric and Torch use different RNGs.

| Prompt | Original audio / generation | Fabric audio / generation | Original RTF | Fabric RTF |
| --- | --- | --- | --- | --- |
| Short introduction | 5.52 s / 0.985 s | 5.52 s / 1.743 s | 0.179 | 0.316 |
| Spoken numbers | 7.36 s / 1.228 s | 7.44 s / 2.379 s | 0.167 | 0.320 |
| Dialogue/punctuation | 6.88 s / 1.123 s | 6.72 s / 2.135 s | 0.163 | 0.318 |
| Accented names/contraction | 7.20 s / 1.174 s | 7.36 s / 2.354 s | 0.163 | 0.320 |
| Long passage | 36.32 s / 5.923 s | 36.48 s / 11.677 s | 0.163 | 0.320 |

First-audio medians ranged from 53–66 ms upstream and 162–272 ms in Fabric.
All ten WAVs were finite, non-silent, and unclipped. `tiny.en` ASR recovered
identical intended words for the short and punctuation prompts. Both long
recordings transcribed “assistance” as “assistants” (1/106 words). Number-prompt
word error rates (24%/20%) reflect ASR formatting “twenty three point five” as
“23.5”, “tenth” as “10th”, and “nine” as “9”; the spoken content transcribes
consistently. The accented-name prompt has name/number spelling differences in
both versions; Fabric additionally transcribes “It’s” as “It,” requiring further
listening/quality evaluation. ASR scores alone do not measure naturalness.

Reproduce with `scripts/benchmark-pocket-matrix.py` from the pinned reference
venv, passing `--native`, `--bundle`, `--weights`, `--config`, `--tokenizer`,
`--voice`, and `--output`. It writes each prompt's WAVs, individual reports,
and a combined `comparison.json`. Then run `scripts/validate-pocket-audio.py`
on that output directory from a validation-only venv containing faster-whisper,
numpy and scipy. Keep CPU-heavy work idle during the timed comparison.


## Native memory preflight

`tts-cpp/pocket/fit.h` exposes `pocket::fit_params(FitOptions)`. It tokenizes the
requested text, validates voice/context capacity, and prices the actual FlowLM
and Mimi graphs with ggml's size-only allocator and CPU work-buffer planner.
The planner is resolved through the selected CPU backend's registry, so
preflight works when CPU variants are dynamically loaded (Linux/Android).
This requires the `ggml_graph_plan` registry export from
[qvac-ext-ggml#92](https://github.com/tetherto/qvac-ext-ggml/pull/92);
older backends report an explicit preflight error instead of omitting scratch memory.
GGUF tensor payloads are never read, allocated, or executed during preflight.
The estimate includes expanded F32 weights, state, overlapping worker compute,
host containers, thread stacks, load staging, and native batch PCM buffers.
It excludes application and SDK/JS copies. This is a memory projection, not a
validation of tensor contents, checkpoint encoder capability, or speech quality.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --fit --fit-margin-mib 256 \
  --fit-budget-mib 1024 --report /tmp/pocket-fit.json
```

The optional budget further limits available OS memory. On iOS, the projection
also respects `os_proc_available_memory()` when supported; a zero app allowance
returns does-not-fit. Linux uses MemAvailable; container/cgroup limits are not
automatically detected, so hosts must supply their application ceiling. The CLI
returns 0 for fit, 1 for no-fit, and 2 for a preflight error. Native callers receive
`FitStatus` and a reason/report. Fit remains a point-in-time estimate; another
process can consume memory between preflight and loading.

On the M2 with the short prompt above, preflight required 983,319,419 bytes
including a 256 MiB margin (about 682 MiB before margin). A measured native
synthesis process peaked at 558,465,024 bytes RSS (about 533 MiB). Preflight
itself peaked at 9,322,496 bytes RSS and took 0.20 seconds. These are single-run
resource checks, not a guarantee for other texts or host processes.

Eight Pocket native tests pass: bounded reference audio, frontend, fit, engine,
loader, FlowLM F32/F16, and Mimi. The fit test produces the same projection after
truncating all three GGUF files to metadata only, and covers insufficient budget,
margin overflow, context limits, missing files, and malformed WAV headers. The
shared-library public fit API also passes its fixture test.

Adversarial review found that optional preflight alone did not protect direct
model loading from a WAV header claiming an enormous sample count. Runtime and
preflight now use the same bounded reader. It validates before model allocation,
keeps the same file open through decoding, reads at most 4096 frames per block,
and rejects truncated or non-finite PCM. Accepted reference input is uncompressed
PCM8/16/24/32 or float32/64 WAV, 1–64 channels, 8–192 kHz, at most 30 seconds and
64 MiB on disk. Regression tests include the 82-byte RF64 allocation case, stereo
downmix, repeated reads, path replacement, invalid rate/channel count, and float
infinity. Independent adversarial re-review found no remaining concrete defect.

## iOS Simulator validation

Native arm64 builds and all eight Pocket tests pass under an iPhone 16 Pro
Simulator running iOS 18.6. GGML and tts-cpp were built with the iOS Simulator SDK,
minimum iOS 15, CPU/Accelerate enabled, and Metal/BLAS disabled. This exercises
native iOS binaries using `simctl spawn`; it does not establish installed-app,
Bare mobile addon, physical-device, or Android behavior.

Build/install the same pinned ggml source with `CMAKE_SYSTEM_NAME=iOS`,
`CMAKE_OSX_SYSROOT=iphonesimulator`, `CMAKE_OSX_ARCHITECTURES=arm64`,
`CMAKE_OSX_DEPLOYMENT_TARGET=15.0`, `GGML_NATIVE=OFF`, `GGML_METAL=OFF`,
`GGML_BLAS=OFF`, and `GGML_ACCELERATE=ON`. Configure tts-cpp with the same
platform settings, `TTS_CPP_USE_SYSTEM_GGML=ON`, its ggml install prefix, and
all three Pocket fixture-directory options. CMake's simulator executables are
inside `.app` bundles, so run tests and the CLI as follows:

```sh
xcrun simctl spawn "$SIMULATOR_UDID" \
  build/pocket-ios/test-pocket-engine.app/test-pocket-engine \
  /absolute/path/to/pocket-bundle /absolute/path/to/pocket-bundle/flow-lm.gguf
xcrun simctl spawn "$SIMULATOR_UDID" \
  build/pocket-ios/pocket-cli.app/pocket-cli \
  --model-dir /absolute/path/to/pocket-bundle \
  --text "Hello! Pocket TTS is speaking from the iOS simulator." \
  --output /absolute/path/to/pocket-ios-simulator.wav \
  --report /absolute/path/to/pocket-ios-simulator.json
```

The simulator sample has 82,560 float samples at 24 kHz (3.44 seconds), peak
0.63059, RMS 0.11084, and no clipping. `tiny.en` recovers the intended sentence
exactly. Its single un-warmed run took 1.464 seconds, with 214 ms to first audio;
these host-simulator timings are not a mobile-device benchmark. The non-app test
process receives zero from the iOS available-memory API, correctly producing a
no-fit result while still allowing the projection tests to verify byte counts.


## Deterministic quality follow-up

The benchmark scripts accept `--temperature` and `--steps`; the matrix also
accepts `--cases`. To investigate the contraction discrepancy without differing
random noise, rerun into a separate output directory with
`--temperature 0 --cases unicode long --runs 2`, then run the audio validator.

Both implementations now produce exactly 170,880 samples for the accented-name
prompt and 896,640 samples for the long passage (7.12 and 37.36 seconds at 24 kHz).
The ASR transcriptions match between implementations, including “It's a beautiful
morning.” The long passage has zero word errors for both; the accented-name
prompt's shared 9.52% raw WER is the `eight`/`8` and `René`/`Renee` formatting
already described. All four recordings are finite, non-silent, and unclipped.

Waveform comparison against the upstream PCM gives SNR 68.08 dB / maximum error
0.000799 for the accented-name prompt and 33.38 dB / maximum error 0.10281 for
the long passage. Small numerical differences can propagate through the
autoregressive pipeline; matching transcription does not establish bit parity
or equal naturalness. The earlier nonzero-temperature discrepancy is absent
under this controlled setting, so it does not demonstrate a systematic frontend
contraction bug. These checks remain automated signal/transcription evidence,
not a claim of subjective listening assessment.


## Packaged iOS addon worklet follow-up

The actual `@qvac/tts-ggml` iOS Simulator addon now builds against the iOS native
libraries. Its install includes shared tts-cpp and speech ggml libraries; the
standard `bare-link` tool packages them into the addon's framework. A standalone
host using the same Bare Kit C API as the installed React Native integration
runs the existing addon integration test inside a mobile worklet.

The reproducible runner is `packages/tts-ggml/scripts/run-pocket-ios-worklet.py`
in the addon checkout. It requires a booted simulator, model bundle, installed
`bare-pack`/`bare-link`, and the `react-native-bare-kit` package. Each run preserves
a new output directory containing frameworks, host, logs, TAP, JSON, and audio.
It does not change an app checkout. Local validation used iOS 18.6 and Bare Kit's
Bare 1.24.6 runtime. The runner passes streaming, cancellation, deterministic
recovery, resampling reload, empty-input failure, and cleanup checks. The latest
recorded run has 60 assertions; counts vary with the number of streaming chunks.
A separate missing-model run correctly returns failure with zero assertions.

Adversarial review found a false-green cancellation test: it previously accepted
both rejected and successfully completed synthesis after requesting cancellation.
The test now requires the cancellation promise to succeed and the synthesis to
reject with a cancellation error. The strengthened test passes on macOS and in
the mobile worklet, and re-review confirms a no-op cancellation would fail.

The worklet WAV contains 63,360 PCM16 samples at 24 kHz (2.64 seconds), peak
0.57874 and no clipping. `tiny.en` transcribes “Hello, we can generate speech
with fabric.” This adds packaged mobile-addon runtime evidence; installed-app,
physical-device, Android and mobile SDK transport validation remain separate.


## Public Node SDK transport and package checks

The addon checkout now includes `packages/sdk/test/pocket` and
`npm run test:pocket:node`. This compiles the production client APIs and TTS
server plugin, then uses the real Node RPC client, spawned Bare worker and
socket serialization. A focused custom worker registers TTS only. The build
passes with `noEmitOnError`; it does not bypass compiler errors from the broader
SDK's unrelated plugins.

Four tests pass on Node 23.9.0 / Bare 1.28.6: batch/stream PCM parity, duplex
output before input closes, cancellation with deterministic recovery, and
unload/worker lifecycle shutdown. Adversarial review found two harness issues:
relative model paths were resolved after changing directory, and a Node test
timeout did not unwind a hung await into `finally`. Paths now resolve before
changing directory, and abort/after hooks close the worker. A separate process
supervisor bounds native cleanup stalls by killing the test's owned process
group on macOS/Linux. A real-load injected caller hang exits on the test timeout;
a blocked child/grandchild fault test confirms both processes are terminated by
the supervisor. Windows has a taskkill fallback but is not validated here.

The installed CMake package advertises the `pocket` component. Consumers can
require it with `find_package(tts-cpp CONFIG REQUIRED COMPONENTS pocket)`;
the addon now does so. An installed-package consumer configured, linked the
public fit API and ran successfully. The existing older registry package is
rejected during configuration for lacking this component. The registry source
pin still needs a published revision containing this implementation; these
checks do not claim the release is available from that registry already.


## Fabric main validation (September 10, 2026)

The integration is also ported to Fabric main `59b80b7`, where the addon is
written in TypeScript and the in-process engine lives in `@qvac/inference`
0.19.0. `@qvac/sdk` 0.19.0 remains the public transport facade. The native
source includes speech master `8db5c437` and builds against registry
`ggml-speech` source `70179b2b` (2026-09-09#1). All eight Pocket native CTests
pass on that combination.

The rebuilt addon passes its real-model integration test and 282 addon unit
tests (878 assertions across all engines). The inference plugin passes batch
and streaming real-model tests, schema tests, and request lifecycle tests.
The current public Node SDK passes batch/stream PCM comparison, duplex output
before input closes, cancellation/recovery, and worker shutdown using its
production socket transport with a custom worker registering only TTS.
The SDK compiles successfully. A focused inference test build compiles with
`noEmitOnError`; the full inference build has existing errors in safe-fetch,
AudioGen and sdcpp, outside the Pocket changes.

Adversarial review identified and verified fixes for overlapping model loads,
use of disposed handles after unload, requests on destroyed instances,
AbortSignal cancellation that failed to stop native work, and unhandled chunk
completion rejections when native dispatch failed. Requests hold admission
until native terminal callbacks drain. Model lifecycle operations reject
overlap. The IPC test runs under a process supervisor and selects an explicit
test-owned config/cache directory, so inherited configuration cannot redirect
its cache writes.

The current addon recording contains 63,360 PCM16 samples at 24 kHz (2.64 s),
peak 0.57874, and no clipped samples. Automated transcription recovers
“Hello, we can generate speech with fabric.” This is a fresh current-main
runtime sample; the prior iOS worklet results above refer to the earlier
Fabric checkpoint until that port is rebuilt for mobile.


## Final registry build and release review

The native implementation is available in draft PR #240. Registry draft PR
`tetherto/qvac-registry-vcpkg#364` selects speech-cpp 2026-09-10#1 from source
`470e678f7c5ae8496bb5c11bd04229698379175e` with a verified archive checksum.
The Fabric addon pins registry baseline and reference
`7a70a0c301c155beb85e985c542f8b0ccd090e10`. Normal vcpkg-backed addon builds,
without development library overrides, succeed for macOS and iOS Simulator.

The current pinned macOS addon passes its real-model integration test and the
public Node SDK passes all four transport tests. The current iOS Simulator
worklet passes 58 assertions using Bare Kit 0.14.5 / Bare 1.29.4 on iOS 18.6.
Both produce valid 2.64-second PCM16 WAVs at 24 kHz. The mobile packager handles
pnpm's physical dependency paths and uses the actual linked addon framework.
Adversarial review found no remaining concrete issue in that harness.

A final CLI consumer linked against the exact pinned registry libraries gives
different performance from the manual development build reported earlier.
The final measurements supersede the earlier approximately 2x slowdown:

| Prompt | Upstream seconds | Pinned Fabric seconds | Fabric RTF |
| --- | ---: | ---: | ---: |
| Short | 0.941 | 0.984 | 0.178 |
| Numbers | 1.231 | 1.424 | 0.191 |
| Punctuation | 1.181 | 1.236 | 0.184 |
| Accented names | 1.230 | 1.379 | 0.187 |
| Long | 6.160 | 6.554 | 0.180 |

These are medians of three measured runs following one warmup, Apple M2 CPU,
one thread per worker and two workers. Fabric is about 5.2–5.6x faster than
real time; generation time is 5–16% above upstream. First audio is 74–116 ms
versus upstream 53–69 ms. Registry build flags differ from the manual build;
this comparison does not isolate which flag or compiler caused the change.
During the matrix, model loading took 0.23–0.31 s. An earlier first process
launch measured 9.61 s, so these warm-cache numbers are not a cold-start claim.
Public SDK startup and transport latency are additional.

All ten final recordings are finite, non-silent and unclipped. Automated ASR
results remain consistent with the earlier corpus: short and punctuation have
zero word errors, both long recordings share one assistance/assistants error,
and nonzero-temperature accented-name output differs in a contraction. The
zero-temperature follow-up above remains the controlled numerical comparison.

The earlier Linux/Windows Supertonic fit-test regression from
[PR #229](https://github.com/tetherto/qvac-fabric-speech.cpp/pull/229) was fixed
independently in [PR #242](https://github.com/tetherto/qvac-fabric-speech.cpp/pull/242),
now merged into this branch's base. The Pocket PR contains no Supertonic changes.
Full Fabric inference compilation still has the unrelated errors noted above.
No default-branch merge or package publication is claimed.
