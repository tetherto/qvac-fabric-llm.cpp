# Full TranscribePsy EU on Apple Core ML

The full 25-language `qvac/TranscribePsy-250M-EU` uses its 24-layer English
FastConformer trunk, six CAFE adapters for the `other` route, and a language gate
at layer 11. CPU and Metal already execute both routes. Core ML exports use
channel-first fixed windows and explicit padding masks so the English trunk,
adapted trunk, and 12-layer gate prefix run through the same validated encoder
contract on macOS and iOS. The small attentive-pooling language classifier stays
on CPU. `auto` uses that classifier's original threshold and cropped input.

## Export and deployment

Run the commands below from the repository root on **macOS with Xcode installed**
and its command-line tools selected. `--compile-dir` invokes `xcrun coremlc`;
these compilation commands cannot run on Linux or on the target iPhone.
Check the compiler and create an isolated Python environment:

```bash
xcrun --find coremlc
python3 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install -r parakeet/scripts/requirements-coreml.txt
.venv-coreml/bin/python -m pip install safetensors sentencepiece huggingface_hub
```

Download the checkpoint, adapters and gate, then convert them to an **F16 GGUF**.
See [models.md](models.md#transcribepsy) for the native inference conversion.
The Q8_0 conversion in that guide produces a different file; it does not create
the `.f16.gguf` used here. Convert from the original checkpoint, rather than
re-expanding Q8_0 weights:

```bash
MODEL_DIR=parakeet/models
PYTHON=.venv-coreml/bin/python
.venv-coreml/bin/hf download qvac/TranscribePsy-250M-EU \
  TranscribePsy-250M-EU.nemo adapters.safetensors language_gate.safetensors \
  --local-dir "$MODEL_DIR"
$PYTHON parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt "$MODEL_DIR/TranscribePsy-250M-EU.nemo" \
  --hf-repo qvac/TranscribePsy-250M-EU \
  --eu-adapters "$MODEL_DIR/adapters.safetensors" \
  --eu-gate "$MODEL_DIR/language_gate.safetensors" \
  --out "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" --quant f16 \
  --description "TranscribePsy-250M-EU, FastConformer-TDT speech recognition (Apache-2.0)" \
  --long-form-window-sec 60 --long-form-overlap 0.25
```

**Recommended baseline deployment:** FP16 fixed encoders/gate and an FP16 decoder,
as exported below. This configuration has the broadest full-EU validation on
Mac and iPhone. For size-constrained deployments, change `--nbits 0` to
`--nbits 6` on the three fixed encoder commands, keeping the decoder FP16.
Six-bit encoders/gate were also validated on both devices, with a small measured
long-English quality cost; see [quantization coverage](#quantization-coverage-and-refreshed-six-bit-results).
Both exporters default to `--nbits 6`, following the EU-sub deployment recipe,
so keep the explicit precision flags when reproducing this full-EU baseline.
In particular, omitting `--nbits 0` from the decoder command compresses its
eligible weights even though the output projection defaults to FP16.

```bash
$PYTHON parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route en --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route other --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder-other.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route gate --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder-gate.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON parakeet/scripts/export-decoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" --nbits 0 --nbits-out 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-decoder.mlpackage" --compile-dir "$MODEL_DIR"
```

`--check` checks the export graph against its reference before conversion; it
does not validate compiled low-precision predictions or deployment-corpus accuracy.
The compressed decoder has only the focused Mac checks described
[below](#conversion-and-decoder-regression-checks-2026-11-13), rather than the
FP16 decoder's broader Mac/iPhone coverage.

Main encoders default to 1,500 mel frames
(15 s). The gate export uses at least 2,001 frames, covering the model's first
20 seconds including preprocessing padding. `--nbits 0` retains FP16 weights;
`--nbits 6` uses grouped-channel palettization. `--palettize-workers` can speed up
k-means at the cost of RAM. Export from F16 when evaluating compression separately
from GGUF quantization. The generic exporter accepts full EU only with an explicit route and `--flexible`.

True unwindowed inference and compute policies excluding ANE additionally require
three flexible FP32 sidecars:

```bash
for ROUTE in en other gate; do
  SUFFIX=""; [ "$ROUTE" = en ] || SUFFIX="-$ROUTE"
  $PYTHON parakeet/scripts/export-encoder-coreml.py \
    --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
    --flexible --eu-route "$ROUTE" --precision fp32 \
    --n-mel-frames 1500 --min-frames 1 --max-frames 40000 \
    --out "$MODEL_DIR/transcribepsy-250m-eu-encoder$SUFFIX-flexible.mlpackage" \
    --compile-dir "$MODEL_DIR"
done
```

Flexible full-EU exports default to FP32 and runtime execution uses Core ML CPU
under every accepted compute policy. On the evaluated Apple runtimes, flexible
FP16 predictions contain non-finite features even on CPU; GPU execution of the
FP32 relative-position graph aborts inside MPS gather/reshape kernels. FP16
flexible full-EU export and incompatible runtime sidecars are rejected. Fixed
FP16/6-bit sidecars retain the fast Neural Engine path. The flexible range above
covers about 400 seconds; choose a deployment maximum appropriate to device
memory. Attention memory still grows quadratically for true unwindowed inputs.

Stateful callback/push streaming uses FP32 encoders to preserve words and decoder
continuity. On macOS, export two static capacities per route for GPU execution:

```bash
for FRAMES in 1500 6008; do
  for ROUTE in en other; do
    SUFFIX=""; [ "$ROUTE" = en ] || SUFFIX="-$ROUTE"
    $PYTHON parakeet/scripts/export-encoder-coreml-ane.py \
      --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
      --eu-route "$ROUTE" --precision fp32 --nbits 0 \
      --window-mel-frames "$FRAMES" \
      --out "$MODEL_DIR/transcribepsy-250m-eu-encoder$SUFFIX-fp32-$FRAMES.mlpackage" \
      --compile-dir "$MODEL_DIR"
  done
done
```

On macOS the smallest fitting static FP32 program runs with CPU/GPU permission.
Larger contexts, policies excluding GPU, and all iOS precise inference use the
flexible FP32 CPU program. iPhone deployments can omit the four static FP32
programs. Only one
precise main encoder (route and capacity combined) remains resident per Engine;
changing the route or required capacity evicts the previous one. The gate has
its own smaller CPU program. Optional static programs improve performance;
matching flexible programs cover every supported context without changing it.

On iOS 18.6.2, static FP32 GPU tensors initially match the native encoder, but
reusing an Engine across routes or running the CPU gate before the GPU encoder
can abort inside MPSGraph (`shape.count = 0 != strides.count = 4`). Allowing Core ML
to allocate its own output did not resolve the assertion. iOS therefore uses the
reliable masked FP32 **Core ML CPU** programs for stateful, unwindowed and precise
compute-policy execution. Every mode still executes Core ML; this path does not
claim Neural Engine or GPU acceleration. Offline fixed FP16/6-bit encoding keeps
its validated Neural Engine execution. macOS retains the validated static GPU
path. The table below reports these stages separately.

Precise programs preserve the native frontend's frame grid: for example, 400 valid
mel frames plus one preprocessing padding frame produce 51 encoder rows. The
extra row contains the subsampling projection bias and participates in native
attention. Masking only convolution inputs, while retaining that row, avoids
changing stateful decoder emissions. Flexible sidecars therefore require both
`mel` and `mel_valid_mask`, with native-padding metadata. Older single-input
full-EU flexible exports must be regenerated.

Build with `PARAKEET_COREML=ON`. Ship the compiled `.mlmodelc` directories
beside the GGUF, preserving the names above; the runtime strips trailing GGUF
quantization tags such as `.f16` or `.q8_0` when deriving names. English and adapted
exports must have matching capacities. Model variant, route and layer-count
metadata prevent EU-sub, exchanged-route, and wrong-depth sidecars from loading.
Missing/incompatible sidecars use native CPU/Metal fallback with a full GGUF.
`PARAKEET_COREML_ONLY=1` skips native encoder allocation when compatible fixed
routes are available, either in the shared bundle below or as three separate
sidecars. Once ownership succeeds, a failed or
missing required flexible encoder returns an error instead of reading unallocated
native weights. Leave `PARAKEET_COREML_ONLY` unset when CPU/Metal recovery is
required. Benchmark JSON exposes `encoder_owned_by_coreml` separately from
`encoder_coreml_all_runs`; availability or dispatch alone does not prove native
weights were omitted.

Accepted `PARAKEET_COREML_COMPUTE_UNITS` values are `cpu_and_ane`, `all`,
`cpu_and_gpu`, and `cpu_only`; the same setting selects decoder units. For full EU,
`all` and `cpu_and_ane` pin the fixed encoders to CPU and Neural Engine. Policies
excluding ANE select precise FP32 programs, because FP16 fixed graphs produce
non-finite predictions on the evaluated CPU/GPU kernels. `cpu_only` uses CPU;
`cpu_and_gpu` uses static GPU programs on macOS when they fit and CPU otherwise. Streaming
always uses precise programs: `cpu_and_ane` uses their reliable CPU execution.
Actual labels report `coreml-ane`, `coreml-cpu-gpu`, or `coreml-cpu`; allowing a processor
does not require dispatching work to it.
Set compute-policy environment variables before constructing the model/Engine;
changing environment variables is not a live reconfiguration API. Additive
CPU-only/CPU-GPU policies retain the English fixed handle for capacity/seam
metadata and availability, and skip unused fixed adapted/gate handles. Precise
programs load lazily as needed. Core ML-only ownership still validates all three
fixed programs before omitting native encoder weights.
`PARAKEET_COREML_DISABLE=1` disables both sidecar families. Core ML may perform
unsupported operations on CPU, so inspect the exporter's compute-plan report
before claiming ANE placement.

### Deployment size and possible optimizations

The three fixed programs embed separate weights: English contains the full
trunk, `other` contains the same trunk plus the CAFE adapters, and `gate`
contains its first 12 layers. Together they store roughly 2.5 trunk copies.
The [measured compiled sizes](#quantization-coverage-and-refreshed-six-bit-results)
sum to 1,129.35 MiB for FP16 or 450.18 MiB for six-bit encoders/gate, excluding
the decoder, GGUF and optional FP32 programs. EU-sub has one encoder and no
separate adapted/gate programs, so its package size is not a full-EU size target
with the current export layout. Compression reduces disk size without a measured
inference-speed advantage.

Fixed programs load eagerly under the default ANE policy, including for known
language requests; selecting `en` or `other` currently skips gate inference,
but does not skip loading the other fixed programs. Precise FP32 main programs
already load lazily with one route/capacity resident per Engine. First-install
specialization costs for the fixed programs are reported
[below](#startup-memory-and-validation-receipts).

Potential follow-up work, in increasing implementation scope:

- Load fixed routes on demand: known-language requests would load just their
  route, while `auto` would load the gate then the selected route. This could
  reduce startup work and resident memory, but would not reduce shipped size.
  It must preserve sidecar compatibility checks, Core ML-only ownership and
  fallback behavior, including when an Engine changes routes.
- The multifunction deployment below shares trunk weights on disk. Routes still
  load eagerly; specialization/cache reuse and resident memory savings are not promised.
- Evaluate one program with a route input scaling the adapter contribution,
  also returning layer-11 features for the gate. Automatic routing still needs
  a gate decision before the adapted route can be selected; returning gate
  features alone does not establish single-pass automatic routing. Validate
  route semantics, ANE placement, 20-second windows and FP16 edge cases.

Lazy loading and the single-program route input remain optimization candidates.
The runtime supports separate static graphs and shared-weight multifunction graphs.

### Shared-weight runtime deployment

The Engine discovers `transcribepsy-250m-eu-encoder-shared.mlmodelc` beside the
GGUF and selects its named `en`, `other`, and `gate` functions. This works through
the existing C/C++ APIs and `PARAKEET_COREML=ON`; consumers need no new flags or
dependencies. It requires macOS 15 or iOS 18. Older systems or incompatible shared
bundles try the separate sidecars, then the existing native fallback when the
GGUF contains encoder weights. Fixed ANE inference and Core ML-only ownership
validate all three shared routes before selecting them. Policies excluding ANE
retain only the English handle for shape metadata and use the precise programs.

Create the three source packages with the production encoder exporter. Use the
same capacity (at least 2,001 mel frames), precision, checkpoint and quantization
for all routes. The normal English export's smaller default capacity cannot be
merged with the 20-second gate. For example:

```bash
for ROUTE in en other gate; do
  $PYTHON parakeet/scripts/export-encoder-coreml-ane.py \
    --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
    --eu-route "$ROUTE" --window-mel-frames 2001 --nbits 0 --check \
    --out "$MODEL_DIR/$ROUTE.mlpackage"
done
$PYTHON parakeet/scripts/merge-transcribepsy-eu-coreml.py \
  --en "$MODEL_DIR/en.mlpackage" --other "$MODEL_DIR/other.mlpackage" \
  --gate "$MODEL_DIR/gate.mlpackage" \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder-shared.mlpackage" \
  --compile-dir "$MODEL_DIR"
```

For six-bit deployment use `--nbits 6` on **each source export**; compression
after merging is unsupported. The merger validates route identity, depth and
matching interfaces and writes versioned runtime metadata. Existing experimental
packages lack that identity and must be regenerated using these commands.

Ship the shared `.mlmodelc` instead of the three fixed encoder directories.
Keep the decoder and the FP32 flexible/static sidecars needed by your supported
inference modes: their precision and streaming policies are unchanged. In
particular, sharing the fixed ANE weights does not remove the precise encoders
needed for streaming and unwindowed inference. The roughly 60% reduction measured
below applies to the fixed encoder packages, not the entire deployment.

An encoder-less GGUF can retain the same basename so the shared bundle is found:

```bash
$PYTHON parakeet/scripts/strip-encoder-gguf.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --out "$RELEASE_DIR/transcribepsy-250m-eu.gguf"
```

Create `$RELEASE_DIR` first, then place the compiled sidecars beside that GGUF.
A missing valid encoder is a
load error for an encoder-less file. With a full GGUF, `PARAKEET_COREML_ONLY=1`
opts into skipping native encoder allocation after all shared routes validate.

For downstream packaging, `speech-cpp[coreml]` uses the same compiled engine and
public API. Publish the source revision and update its registry archive hash/tree
through the normal registry release process, then update the QVAC dependency.
Model sidecars are deployment assets, not files embedded in the vcpkg library.

Runtime regression coverage includes generated Core ML models exercising the
actual native wrapper (`test_eu_shared_coreml`), portable sidecar naming tests,
and a checkpoint-backed Engine test (`test-eu-shared-engine`). The Engine test
compares separate/shared transcripts for `en`, `other` and `auto`, checks native
encoder ownership with full and stripped GGUFs, tests compute policies excluding
ANE, and verifies corrupt-shared-bundle fallback. It requires the source separate
sidecars, shared sidecar and audio fixtures; absent fixtures skip it.

```bash
ctest --test-dir "$BUILD_DIR" \
  -R '^(test-coreml-paths|test_eu_shared_coreml|test-eu-shared-engine)$' \
  --output-on-failure
```

The 2026-11-17 runtime validation used the production exporter and merger with
the full-EU checkpoint. FP16 source packages totaled 1,191,797,800 bytes; the
shared package was 477,384,369 bytes (59.94% smaller). macOS passed separate/shared
transcript parity, encoder ownership/fallback and compute-policy checks. Both
macOS and a physical iPhone 16e passed the existing API suite for all three routes:
batch, precomputed mel, callbacks, fragmented push feeds, reset, cancellation,
empty/finalized streams and automatic language gating. The iPhone used an
encoder-less GGUF with no separate fixed encoder bundles. An installed static
library also linked and ran through the standalone `find_package` consumer.
Six-bit source packages totaled 479,603,954 bytes; the merged package was
192,909,223 bytes (59.78% smaller). Six-bit passed the same macOS Engine matrix
and four physical-iPhone offline checks (`en`, `other`, automatic English and
automatic French), with actual Core ML execution and encoder ownership confirmed
in benchmark JSON. These sizes cover fixed encoder packages; they exclude the
GGUF, decoder and precise streaming programs.

### Trying route-sharing exports

[`export-transcribepsy-eu-coreml-experiment.py`](../scripts/export-transcribepsy-eu-coreml-experiment.py)
provides isolated prototypes for the two export alternatives. These packages
are **not runtime sidecars**: do not replace a deployed encoder with them.
All experimental routes use the same 2,001-frame capacity, including the main
encoders, so the gate covers its first 20 seconds and shared positional constants
have matching shapes. Float32 graph checks run before every export.

```bash
EXPERIMENTS=/path/to/route-experiments
for BITS in 0 6; do
  DIR="$EXPERIMENTS/bits$BITS"
  for ROUTE in en other gate unified; do
    $PYTHON parakeet/scripts/export-transcribepsy-eu-coreml-experiment.py export \
      --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
      --route "$ROUTE" --nbits "$BITS" --palettize-workers 4 \
      --out "$DIR/$ROUTE.mlpackage"
  done
  $PYTHON parakeet/scripts/export-transcribepsy-eu-coreml-experiment.py merge \
    --source-dir "$DIR" --out "$DIR/shared.mlpackage"
done
```

The `shared` package exposes `en`, `other` and `gate` functions; select the
function when loading it. Core ML Tools deduplicates equal constants across
functions ([Apple's multifunction guide](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)).
Core ML Tools 9 does not support post-training palettization of an already
merged multifunction model, so the six-bit recipe compresses each source first.
Multifunction programs require iOS 18/macOS 15, matching these exporters' existing
deployment target. Shared disk weights do not establish shared specialization
or shared resident allocations.

The `unified` package takes the four ordinary mask/mel inputs plus an FP16
`adapter_scale` tensor of shape `[1, 1, 1, 1]`: supply exactly 0 for English or
1 for the adapted route. It returns `encoder_out` and `gate_features`.
Only classify `gate_features` from scale 0; scale 1 includes earlier adapters
and does not reproduce the unadapted gate. Automatic routing can reuse the
scale-0 main output when the gate chooses English, but needs a second full
prediction when it chooses `other`. Both adapter branches still compute at
scale 0, and collecting gate features computes the entire trunk.

[`route_experiments.py`](../examples/ios-coreml-bench/route_experiments.py) prepares
native mel fixtures and independent FP32 references, compares actual compiled
predictions, reports gate probabilities/decisions and compute plans, and measures
loading subsets in separate processes:

```bash
PROBE=parakeet/examples/ios-coreml-bench/route_experiments.py
$PYTHON "$PROBE" prepare --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --binary /path/to/parakeet --wav /path/to/english.wav --wav /path/to/french.wav \
  --output "$EXPERIMENTS/fixtures"
$PYTHON "$PROBE" compare --models "$EXPERIMENTS/bits0" \
  --fixtures "$EXPERIMENTS/fixtures" --output "$EXPERIMENTS/fp16-results"
$PYTHON "$PROBE" load --models "$EXPERIMENTS/bits0" --model-layout experiment \
  --routes en other gate --output "$EXPERIMENTS/eager"
$PYTHON "$PROBE" load --models "$EXPERIMENTS/bits0" --model-layout experiment \
  --routes en --output "$EXPERIMENTS/known-en"
$PYTHON "$PROBE" load --models "$EXPERIMENTS/bits0" --model-layout experiment \
  --routes gate other --output "$EXPERIMENTS/auto-other"
```

Repeat the comparison for `bits6`. Subset loading measures the cost of deferring
unused handles; it does not implement lazy selection in the current Engine or
its Core ML-only ownership checks. Timings retain OS specialization caches and
must not be presented as fresh-install cold-start measurements. Compiled
numerics and sample gate decisions do not replace transcription/corpus tests.
See the [iPhone harness](../examples/ios-coreml-bench/README.md#experimental-route-probe)
for running the same fixture comparisons on a physical device.

### Prototype measurements, 2026-11-16

The Apple M4 Mac and physical iPhone 16e checks use the same F16 GGUF for all
exports, Core ML Tools 9, and a common 2,001-frame capacity. Compiled encoder sizes below sum file lengths
in MiB, excluding the GGUF, decoder and precise FP32 programs. They differ from
the earlier `du` table, whose main encoders use 1,500 frames.

| Encoder layout | FP16 MiB | Six-bit MiB |
|---|---:|---:|
| Separate English + adapted + gate | 1,136.70 | 457.52 |
| Multifunction shared-weight package | 455.39 | 184.11 |
| Single route-input program | 454.21 | 182.86 |

Both alternatives reduce encoder storage by about 60%. Across six fixtures
(15-second English, a full 2,001-frame English crop, 800-millisecond English,
French, Hindi and silence), both prototypes reproduce the separate compiled
outputs **exactly** at each precision on both devices: 72 route/fixture
comparisons per device, 144 total. All 24 baseline gate decisions match the
independent FP32 reference. Native compute
plans assign every compute operation to ANE: 1,537/1,561/781 operations for the
separate and multifunction English/adapted/gate functions, or 1,567 for the
single route-input graph. Non-compute operations are reported separately.

In three cache-warm Mac loading-only processes per configuration, loading all
three fixed handles takes a median 186.8 ms (FP16) or 193.6 ms (six-bit).
Loading only a known route takes 73.4–73.6 ms or 78.9–81.2 ms; gate plus selected
route takes 110.8–112.7 ms or 118.6–120.4 ms. Current process physical footprint
decreases by only about 1 MiB when loading one route instead of all three.
This probe excludes the GGUF, decoder, inference allocations and Core ML service
processes; it supports a cached handle-load saving, not a large production RAM
claim or a fresh-install specialization bound.

The corresponding iPhone loading-only medians are 616.2 ms / 312.7 ms for all
three FP16/six-bit handles, 135.1–138.5 ms / 144.2–146.7 ms for one known route,
and 185.2–188.7 ms / 197.0–203.3 ms for gate plus selected route. The phone's
private physical-footprint difference is under 1 MiB. Mac runs include nominal
and fair thermal states; all repeated phone loading runs are fair. These are
indicative cached initialization measurements, not controlled performance
comparisons or inference-speed claims. We still observed substantial first-seen
load work for each multifunction entry point; shared weights alone did not
establish shared ANE specialization. The single graph's subsequent route loads
used the already specialized program.

Multifunction sharing preserves the existing per-route graphs and is the least
invasive disk-size candidate. The single program additionally shares one loaded
graph across route inputs, but automatic non-English selection needs two full
predictions. These historical prototype measurements predate the shared-weight
Engine integration above; the single-program prototype is not integrated into
Engine routing, fallback or
Core ML-only ownership. Raw receipts are retained under
`artifacts/transcribepsy-eu-coreml-experiments-20261116/` outside the source tree.
The durable [results JSON](transcribepsy-eu-coreml-results.json) includes sizes,
per-device/per-precision equality and gate checks, function-specific placement,
load measurements, thermal states and the scope limits under `route_sharing_experiments`.

## API, CLI, and streaming behavior

`EngineOptions::language` and CLI `--language` accept `en`, `other`, and `auto`
(also empty). Core ML supports `transcribe`, `transcribe_samples`,
`transcribe_mel`, full-input callback streaming, and `StreamSession` push PCM.
Default long recordings and large streaming contexts use bounded, overlapping
encoder windows. Explicit `EngineOptions::long_form_window_frames = -1` or CLI
`--window-sec 0` preserves true unwindowed attention using the flexible sidecar
when the static FP32 sidecar is too short. It does not silently tile the input.
Offline masked partial windows use fixed FP16/6-bit sidecars. Stateful calls use
FP32 programs and retain their native context and encoder frame counts.

Streaming defaults to two-second chunks for full EU, improving measured long
French push quality without changing explicit requests. API `chunk_ms = 0` means
auto; positive values, including 1000, remain explicit. CLI omission selects the
model default; `--stream-chunk-ms 1000` preserves the former one-second behavior.
Other ASR families retain their previous default. The effective value is exposed
by `StreamSession::options()` and CLI diagnostics.

Auto push streams buffer their initial 20 seconds, evaluate the same gate input
as offline transcription, and retain the chosen route for the session. Shorter
streams resolve the gate on `finalize()`. This delay follows the model's gate
contract; select known `en`/`other` to emit at the usual chunk-plus-lookahead
latency. Feed block size does not affect gate selection. New sessions reset the
route, including when reusing one Engine for different languages. The existing
push API had not applied full-EU language selection before this implementation.

The optional batched TDT decoder serves offline calls with automatic window and
context settings (`long_form_window_frames = 0`, `long_form_context_frames = 0`).
Language `auto` is independent of those two settings. Explicit window/context
requests use the ordinary overlapping encoder plan instead of pause-cut segments.
Full-EU automatic decoder
dispatch uses the measured platform crossover: at least four independent
segments on macOS with a Core ML CPU decoder, or eight on iOS with an ANE decoder.
Short inputs retain the faster native sequential decoder. Other decoder placements
require an explicit batch override, because no speed advantage was established.
The default full-EU decoder uses Core ML CPU on macOS and CPU/ANE on iOS.
`PARAKEET_COREML_DECODER_COMPUTE_UNITS` overrides just that stage; otherwise an
explicit global compute-unit setting takes precedence over the platform default. `PARAKEET_ANE_BATCH_DECODE=1`
forces Core ML decode when a compatible decoder is available: automatic window/context can use parallel independent
segments; explicit window/context and true unwindowed input use one continuous
Core ML lane over the context-preserving encoder output. A missing/incompatible optional decoder emits a warning under force1 and uses
native decode, with `decoder_used_coreml=false`; that case is not Core ML decoder
execution. `=0` forces native continuous decode, and
`PARAKEET_ANE_DECODE_HOST=1` compares sequential decode over exactly the same
independent segments when window/context are automatic. Independent windows reset decoder state, so long-form
wording can differ from the sequential path; use `=0` to retain that decoder
continuity. Callback and push streaming retain native stateful decoder
state while accelerating the encoder. This keeps callback timing and continuity.

`EngineResult::encoder_used_coreml` includes the gate prefix and every encoder
window. `decoder_used_coreml` reports actual batched predictions, rather than a
loaded decoder. `StreamingSegment::encoder_used_coreml` reports actual per-segment
encoder/gate dispatch. CLI benchmark logs/JSON report observed dispatch and
`--require-coreml` rejects per-invocation encoder or gate fallback.

## Validation and performance

Measurements use the full EU F16 checkpoint, four GGML threads (Core ML manages
its own scheduling), and matched audio/decoding
settings on an Apple M4 MacBook Air (16 GB, macOS 27.0.1) and iPhone 16e
(A18, iOS 18.6.2), built with Xcode 26.5 and GGML `6fe02a1`. Both builds use
`GGML_NATIVE=OFF`. Timed offline runs exclude model load and warmup; cold and
lazy specialization are reported separately. CPU/Metal baselines explicitly
disable Core ML. Final phone timing rows require nominal thermal state.

The device compute plans assign all fixed English/adapted/gate compute operations
(1537 / 1561 / 781 for FP16, 1500/1500/2001 mel frames) to the Neural Engine.
The masked FP32 CPU programs assign 1816 / 1864 / 940 compute operations to CPU.
Static FP32 macOS plans assign 1537 / 1561 encoder compute operations to GPU.
`None` entries are non-compute program operations. Runtime dispatch flags and
decoder prediction counters separately prove that inference executed Core ML.
These are compute-plan placement reports, not an Instruments hardware trace or
an ANE-only claim for preprocessing, classification and decoding.

### Exercised support matrix

| Mode or configuration | macOS | Physical iPhone | Actual encoder / decoder |
|---|---|---|---|
| Short offline English/French, `auto`/`en`/`other` | API + CLI | API + CLI | Fixed Core ML / native for short clips |
| Long windowed English/French | CLI, 300 s / 51 s | CLI, 300 s / 51 s | Fixed Core ML windows / measured decoder policy |
| True unwindowed input beyond ANE capacity | Static FP32 GPU, 60 s / 51 s | Flexible FP32 CPU, 60 s / 51 s | No tiling; native encoder frame grid |
| Precomputed mel and full-input callbacks | All routes; precise callbacks GPU/CPU | All routes; precise callbacks CPU | Precomputed/batch tokens match; callbacks match same-context native |
| Push PCM, 127- and 4,096-sample feeds | FP32 GPU/CPU, all routes | FP32 CPU, all routes | Core ML encoder / native stateful decoder |
| Large streaming context and API unwindowed context | 20 s context | 20 s context | Precise Core ML retains caller context and padding |
| Engine reuse, route reset, silence prefix, cancel/finalize | Passed | Passed | No stale route or duplicate final callback |
| Four compute policies, forced decoder stages | CLI | CLI | Observed per-stage labels; CPU fallback inside Core ML where documented |
| F16/Q8 GGUF and six-bit fixed sidecars | CLI; six-bit short + long | CLI; six-bit short + long | Same route metadata and decoder contract |
| Core ML-only native-weight omission | Direct ownership + full API/stream/unwindowed suite | Same full suite + direct ownership | Native encoder blocks absent; actual Core ML gate/encoder |
| Missing/exchanged gate or adapted sidecar; missing decoder | Negative CLI tests | Shared runtime, positive deployment tests | Native fallback or `--require-coreml` error; optional decoder remains native |
| Owned model with missing flexible sidecar | Explicit no-window rejection | Explicit no-window rejection | Fails safely with native weights absent |
| One through eight mel frames | Compiled tensors + native/CoreML CLI parity | Native/CoreML CLI parity | FP32 Core ML CPU, all three routes; empty PCM retains existing API error |

This matrix records exercised combinations, not every Cartesian combination of
options. The file-based Engine methods share the same PCM implementations.
The optional fixed sidecars accelerate default/windowed deployment; continuous
force0 with automatic window/context, true
unwindowed operation and compute policies excluding ANE require the additional
FP32 sidecars. CLI `--window-sec` controls offline operation; the CLI warns that
stream/profile modes ignore that offline option. Streaming Engine callers can
set `long_form_window_frames = -1` explicitly.

### Numerical and quality checks

The float32 PyTorch export graph agrees with the float32 reference at roughly
`1e-6` relative L2. This is a graph-equivalence check, not compiled FP16 accuracy.
The final masked/native-padding programs were checked on 110 actual compiled
predictions against the float32 reference: maximum relative L2 `1.12e-6` for CPU
and `1.23e-5` for static GPU, including 1–8 mel frames, aligned/nonaligned padding
boundaries, English 60 s and adapted French 51 s. Native runtime comparisons also
include its half-precision attention activation casts; measured boundary error
is below 0.004 with identical encoder row counts and stateful transcripts.

Actual ANE FP16 and six-bit predictions were compared against the fixed-mask float32 export reference
on a full 15-second input, an 800-millisecond prefix, and French speech. Across
nine route/input combinations each, maximum relative L2 was 0.0897 (FP16) and
0.1043 (six-bit). The largest FP16 difference occurred in the **intermediate
12-layer gate features** on English with a silence tail; the final English and
adapted encoder outputs had smaller errors. The six-bit maximum occurred in the
adapted 800-millisecond prefix. These are accumulated low-precision/compiler
errors, not a claim of bitwise equivalence. The FP32 compiled checks separate
these effects from an incorrect export graph; downstream transcription quality
is evaluated separately below.

Gate probabilities from the compiled ANE prefix differed by at most 0.00256
(FP16) / 0.00889 (six-bit). All six measured compiled gate decisions agreed with
the float32 reference: English probabilities 0.047–0.212 and French 0.938–0.940,
versus threshold 0.8582. No measured case was near that threshold. A recording
near the threshold could change route under quantization, so validate the gate
on the intended deployment corpus. The first-20-second gate contract also means
an unrepresentative or silent prefix can select a poor route; a known `en` or
`other` option avoids that model-level ambiguity.

The references here cover English/French routes and representative silence,
short prefixes, Hindi stability and repeated long speech. They establish backend
parity and route coverage, not accuracy across a 25-language evaluation corpus.
Forcing a wrong language route remains a model-level quality risk. In an explicit
one-second push configuration, forcing `other` for English JFK omits its final
phrase in both native and precise Core ML; `en`/`auto` preserve it. Exact native
comparisons cover that case, automatic/two-second and explicit one-second chunks, short/long French,
and reused Engines. Core ML does not change the caller's chunk size to conceal it.

The retained results include normalized WER/CER, reference words, transcript,
actual stage dispatch, repeat timings, and process memory. See the
[reproduction harness](../examples/ios-coreml-bench/README.md) for commands.

Explicit one-second push on the 51-second French fixture produces 42.86% WER in
both native and precise Core ML with identical text. Automatic/two-second chunks
reduce that to 7.14%, consisting of French homophone spelling substitutions,
with all twelve utterances present. The implementation changes the model's
automatic default and preserves caller-requested one-second context semantics.


### Measured offline latency

Median processing milliseconds, excluding model load and warmup. Short cases use
five timed runs after two warmups; long cases use three after one. All use
language `auto`. Long primary rows request **60 s windows and 25% overlap** on
every backend: native resolves 750/94 encoder frames; fixed Core ML is bounded
to 187/46 (95-frame advance), so the attention contexts differ as documented.
The decoder stays native for explicit window/context requests.

| Device / audio | CPU ms | Metal ms | Core ML ms | Metal / Core ML | Core ML WER |
|---|---:|---:|---:|---:|---:|
| Mac / English 11 s | 450.5 | 92.9 | 34.8 | 2.67x | 0.00% |
| Mac / French 3.26 s | 136.6 | 44.4 | 30.1 | 1.47x | 0.00% |
| Mac / English 300 s, explicit 60 s/.25 | 17003.9 | 2337.6 | 730.4 | 3.20x | 0.00% |
| Mac / French 51.10 s, explicit 60 s/.25 | 2331.4 | 341.2 | 154.3 | 2.21x | 9.52% |
| iPhone / English 11 s | 1421.8 | 182.7 | 48.7 | 3.75x | 0.00% |
| iPhone / French 3.26 s | 457.2 | 81.8 | 38.9 | 2.11x | 0.00% |
| iPhone / English 300 s, explicit 60 s/.25 | 66339.8 | 6937.7 | 996.9 | 6.96x | 0.00% |
| iPhone / French 51.10 s, explicit 60 s/.25 | 8943.1 | 1018.2 | 232.0 | 4.39x | 9.52% |

English and short French have 0 WER on all three backends. Long French native
WER is 13.10%; explicit Core ML is 9.52%. The strict French differences here are
`Ses`/`Ces` homophone spellings, not missing utterances. WER/CER and full case
metadata are retained in [the compact results](transcribepsy-eu-coreml-results.json).

### Automatic windows and decoder selection

Automatic **language** and automatic **window/context** are separate settings.
These rows leave both window/context at 0 and exercise the default independent
encoder policy. They are reported without a speedup ratio against unmatched
native explicit-window baselines. These historical measurements used a converted
GGUF with no optional long-form hints and a generic description. At that
revision, native auto used a 3750-frame ceiling with 256 context (the 300 s
example balanced to 2388-frame windows). Conversion now records the correct
TranscribePsy provenance and 750/94-frame hints (60 seconds / 25% overlap);
the runtime also recognizes older full-EU files through their model variant.
The historical auto rows below do not measure the corrected native default.

| Device / audio | Core ML ms | RTF | Actual decoder | WER |
|---|---:|---:|---|---:|
| Mac / English 300 s | 386.4 | 0.0013 | coreml-cpu | 0.00% |
| Mac / French 51.10 s | 115.5 | 0.0023 | coreml-cpu | 0.00% |
| iPhone / English 300 s | 557.7 | 0.0019 | coreml-ane | 0.00% |
| iPhone / French 51.10 s | 207.0 | 0.0041 | CPU | 0.00% |

Below the measured decoder crossover, or without the optional decoder, the same independent encoder segments use
the native host decoder. Encoder segmentation does not change at the crossover
or when the optional decoder is absent. Default French 51 s then has 0 WER on
both devices, including encoder-only/owned deployment. Explicit force0 retains
continuous decoding and uses precise Core ML over the normal native attention
context for oversized auto-window input; its French 13.10% WER matches native
homophone spelling and retains all twelve utterances.

### True unwindowed processing

| Device / audio | Core ML ms | Actual encoder | WER |
|---|---:|---|---:|
| Mac / English 60 s | 361.1 | coreml-cpu-gpu | 0.00% |
| Mac / French 51.10 s | 365.5 | coreml-cpu-gpu | 13.10% |
| iPhone / English 60 s | 1410.9 | coreml-cpu | 0.00% |
| iPhone / French 51.10 s | 1199.2 | coreml-cpu | 13.10% |

### First and warm streaming sessions

The 48-case streaming matrix uses separate processes: it excludes initial Engine
construction but includes lazy precise-program loading. It is **not** a
steady-state measurement. The focused 51.10 s French experiment below reuses one
Engine per backend, grouping callbacks then push sessions; first is followed by
three timed warm sessions with identical text. Phone starts/ends nominal.
All streaming decoders are native; Core ML encoders are FP32 GPU on Mac and
FP32 CPU on iPhone. Callback/push WER is 13.10%/7.14%, matching native.

| Device / mode | Metal first ms | Metal warm ms | Core ML first ms | Core ML warm ms | Core ML warm RTF |
|---|---:|---:|---:|---:|---:|
| Mac / callback | 443.7 | 376.4 | 5003.9 | 446.6 | 0.0087 |
| Mac / push | 1686.5 | 1703.5 | 3558.3 | 1714.8 | 0.0336 |
| iPhone / callback | 828.0 | 714.7 | 1724.7 | 1269.9 | 0.0249 |
| iPhone / push | 3160.4 | 3159.1 | 4525.5 | 4759.5 | 0.0931 |

Mac warm push is similar to Metal; callback is modestly slower. iPhone precise
Core ML CPU is slower than Metal in both modes but remains comfortably realtime.
The fixed FP16 ANE graph is not used for stateful decoding. First-use/lazy load
can dominate short sessions; initialization and cache state must be included in
a deployment latency budget. The first Mac callback above included a new precise
GPU program specialization, unlike the cached separate-process matrix.

### Startup, memory, and validation receipts

Fresh iPhone installs needed 26–43 s to initialize/specialize the three fixed
encoder programs, loaded eagerly under the default ANE policy. Shipping compiled
`.mlmodelc` avoids package conversion, but does not eliminate target/OS-specific
specialization. Repeat-process fixed initialization is usually about 0.5–0.7 s.
The final masked FP32 CPU programs first loaded in about 3.7–4.0 s per main route
and 1.6 s for the gate; cached loads were about 0.13–0.16 s. These are observed
artifact/runtime costs, not a guaranteed startup bound. Caches were warmed
separately from timed offline inference; app updates/model changes can invalidate
that specialization.

Owned short ANE deployment measured about 144–150 MiB peak RSS on Mac/iPhone,
versus roughly 640–666 MiB with the unused native encoder copy retained. Physical
iPhone owned mixed API/reset/no-window suites peaked around 673–724 MiB in kernel
physical footprint. The focused mixed Metal/Core ML streaming process peaked
at 2646 MiB RSS/1132MiB physical footprint; these are different OS measures and
include both sequentially exercised backends, not one encoder allocation.
The lazy runtime keeps only one precise main route/capacity handle per Engine.

Validation includes 14 distinct relevant CTests (12 initial plus 4 changed-area
passes with 2 overlap), 132 offline/config cases, 48 first-session stream cases,
98 tiny-input positives and 2 owned missing-flexible rejections, 16 default/explicit
chunk-quality cases, 18 explicit-context/load-policy CLI cases, 22 automatic/missing
decoder quality cases, 32 focused first/warm streaming sessions, 110 compiled FP32
predictions and 18 compiled FP16/six-bit predictions. Fair thermal receipts are
retained separately: final stream samples are nominal; one unchanged native CPU
French-push row uses its two stable nominal samples (0.2% spread), all other
stream rows use three. No model/build/cache data belongs in the git change.


### Quantization coverage and refreshed six-bit results

Fixed encoder/gate exports support grouped-channel palettization. Full-EU
validation covers FP16 (`--nbits 0`) and six-bit eligible weights (`--nbits 6`),
with FP16 computation and some FP16 constants retained when grouping is not
applicable. Exportable widths 2, 3, 4, and 8 have not been evaluated for this
model on these devices. Precise FP32 programs deliberately reject palettization;
stateful and continuous precise paths therefore use unquantized FP32 programs.

Compiled program sizes, measured with `du -sk`:

| Program | FP16 MiB | Six-bit MiB |
|---|---:|---:|
| English encoder | 449.56 | 178.76 |
| Adapted encoder | 450.52 | 179.18 |
| Gate prefix | 229.27 | 92.24 |

Together the three encoder programs shrink about 60.1%. The tested `models6`
**TDT decoder is entirely FP16**, including its joint output projection, and is
identical to the 17.87 MiB FP16 decoder in `models`. Decoder palettization is
exportable (`--nbits`, plus separate `--nbits-out`) but was not validated in that
2026-11-11 full-EU evaluation; the focused decoder checks below are separate.
A Q8_0 GGUF paired with FP16 Core ML sidecars is a hybrid:
it does not make the Core ML encoder Q8.

The four long six-bit rows were refreshed on 2026-11-11 against the final explicit
window/context dispatch guard, with three timed runs after one warmup. Both
phone cases finished at nominal thermal state. They use explicit 60 s/.25
windows, actual Core ML encoder/gate predictions, and **native decoding**.
Earlier long six-bit receipts that reported independent Core ML decoding under
those explicit settings are historical and superseded.

| Device / audio | Six-bit processing ms | WER | CER |
|---|---:|---:|---:|
| Mac / English 11 s | 35.724 | 0% | 0% |
| Mac / French 3.26 s | 30.781 | 0% | 0% |
| iPhone / English 11 s | 54.138 | 0% | 0% |
| iPhone / French 3.26 s | 40.308 | 0% | 0% |
| Mac / English 300 s | 752.684 | 1.136% | 1.286% |
| Mac / French 51.10 s | 156.110 | 9.524% | 1.094% |
| iPhone / English 300 s | 1058.619 | 1.136% | 1.286% |
| iPhone / French 51.10 s | 238.100 | 9.524% | 1.094% |

Short measurements retain the previously validated five-run/two-warmup receipts.
Both long English outputs omit one repeated five-word “and so my fellow
Americans” prefix (5/440 reference words); the matched FP16 explicit-window
results have 0 WER. French differences are eight `Ses`/`Ces` spelling substitutions
among 84 words, matching the FP16 explicit-window WER. The refreshed timings show no inference-speed advantage over FP16. Six-bit compression
has a small observed long-English quality cost; it is not transcript-identical
to FP16 on every case. Current raw receipts are in
`quant6-refresh-20261111/` under the retained artifact directory, and the four
records replace their superseded entries in the compact and durable final summaries.

### Conversion and decoder regression checks, 2026-11-13

These checks ran on the Mac only. The decoder exporter now chooses a group size
that divides the output width (2 for 8198 channels) and verifies the emitted
LUT indices and group size before reporting success. A real
`--nbits 6 --nbits-out 6` export produced six-bit output indices and compiled
successfully. The compiled decoder decreased from 18,724,224 to 7,622,918 bytes
(59.3%). JFK, short French and 51.10-second French transcripts matched the FP16
decoder with the same encoder/configuration and actual Core ML decoder dispatch.
Five random 16-lane prediction batches were finite, but quantization changes
numerical outputs: token argmax matched 78/80 slots and duration argmax 80/80.
These checks establish working compression and sample transcript parity, not
bit parity or a speedup. Artifacts are retained in
`artifacts/transcribepsy-eu-coreml-20261113/decoder-palette/`.

For an older full-EU Q8_0 GGUF with the generic NVIDIA description and no window
hints, native automatic windows, explicit `--window-sec 60 --overlap 0.25`, and
`--overlap 0.25` alone all selected 750-frame windows with 94-frame context and
produced identical 440-word output on 300-second English. Core ML automatic and
explicit requests also matched all 440 normalized words; automatic requests
retain independent segments, while explicit requests use the overlapping
encoder plan. Receipts are in
`artifacts/transcribepsy-eu-coreml-20261113/default-runtime/`.

The encoder-stripping tool also reproduced the published EU-sub Core ML GGUF
byte for byte: 948 encoder tensors removed, 13 retained, 17,313,056 output bytes.
The output and published file both have SHA256
`9b7a82f33891a8f55917333b58b920878fdcca3d29a1e92d4748c228ba6eb882`.
A stripped full-EU Q8_0 model additionally passed Mac inference with Core ML
required. Source revisions, comparison receipts and inference results are in
`artifacts/transcribepsy-eu-coreml-20261113/strip-validation/`.
