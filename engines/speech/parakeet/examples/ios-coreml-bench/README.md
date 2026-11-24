# Physical iPhone Core ML validation

This small UIKit host runs the ordinary Parakeet CLI on a worker thread, emits
process peak RSS, physical footprint and thermal state, and exits with the CLI status. It also runs
the full-EU API/streaming regression test (`--api-check`) and queries the actual
device's Core ML compute plans (`--compute-plan`). It is a development harness;
models, samples, build products and signing credentials stay outside the source tree.

For a desktop CLI using an explicit current GGML checkout, this small parent
project matches the validation build and avoids ignored stale local headers:

```bash
export SPEECH_SOURCE="$PWD/parakeet"
export GGML_SOURCE="/path/to/current/qvac-fabric-llm.cpp/ggml"
DRIVER_DIR=/tmp/eu-desktop-driver
mkdir -p "$DRIVER_DIR"
cat > "$DRIVER_DIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(eu-desktop LANGUAGES C CXX OBJC OBJCXX)
enable_testing()
set(PARAKEET_COREML ON CACHE BOOL "" FORCE)
set(PARAKEET_INSTALL OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_EXECUTABLES ON CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_TESTS ON CACHE BOOL "" FORCE)
add_subdirectory("$ENV{GGML_SOURCE}" ggml)
add_subdirectory("$ENV{SPEECH_SOURCE}" parakeet)
target_include_directories(qvac-parakeet BEFORE PRIVATE "$ENV{GGML_SOURCE}/include")
target_include_directories(parakeet.coreml BEFORE PRIVATE "$ENV{GGML_SOURCE}/include")
CMAKE
cmake -S "$DRIVER_DIR" -B /tmp/eu-desktop -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_METAL=ON \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" \
  -DPARAKEET_TEST_MODEL_DIR="$ASSETS/models"
cmake --build /tmp/eu-desktop --target parakeet -j4
PARAKEET_CLI=/tmp/eu-desktop/parakeet/parakeet
```

Use Xcode with a paired device, Developer Mode, and a development provisioning
profile for that device. The profile must permit the bundle ID you provide.

```bash
cmake -S parakeet/examples/ios-coreml-bench -B /tmp/eu-phone -G Ninja \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 \
  -DCMAKE_BUILD_TYPE=Release -DQVAC_GGML_SOURCE_DIR="$GGML_SOURCE"
cmake --build /tmp/eu-phone --target qvac-eu-bench -j4
python3 parakeet/examples/ios-coreml-bench/package.py \
  --binary /tmp/eu-phone/qvac-eu-bench.app/qvac-eu-bench \
  --profile "$PROVISIONING_PROFILE" --identity "$SIGN_IDENTITY" \
  --bundle-id "$BUNDLE_ID" --model-dir "models=$MODEL_DIR" \
  --samples "$ASSETS/inputs" --output /tmp/TranscribePsyBench.app
xcrun devicectl device install app --device "$DEVICE_ID" /tmp/TranscribePsyBench.app
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --model models/transcribepsy-250m-eu.f16.gguf --wav jfk.wav \
  --language auto --backend cpu --threads 4 \
  --bench --bench-runs 5 --bench-warmup 2 --require-coreml --bench-json bench.json
xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source Documents/bench.json --destination /tmp/bench.json
```

`GGML_SOURCE` is the `ggml/` tree of a current `qvac-fabric-llm.cpp` checkout.
`MODEL_DIR` contains the GGUF and exported `.mlmodelc` sidecars from the
[full-EU guide](../../docs/transcribepsy-eu-coreml.md). Packaging copies only
GGUFs and compiled models, excluding checkpoints and `.mlpackage` intermediates.
Pass another `--model-dir models6=PATH` to include a compressed comparison set.
Use a fresh output bundle when changing model sets so removed resources cannot
remain in a reused bundle.

For API and physical-device placement checks (append `--unwindowed` to the API
command when the flexible sidecars are packaged; append `--context-parity`
instead to compare precise callback/push tokens with native and test encoder
padding boundaries):

```bash
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --api-check models/transcribepsy-250m-eu.f16.gguf jfk.wav fr-multipiece-16k.wav
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --compute-plan models
```

Set runtime overrides through `devicectl --environment-variables`, for example
`'{"PARAKEET_COREML_DISABLE":"1"}'` for CPU/Metal baselines or
`'{"PARAKEET_COREML_ONLY":"1"}'` to measure encoder-owned memory. Set
`--backend metal` for a Metal baseline. For callbacks add `--stream`; for push PCM
add `--stream --stream-duplex`, adjusting context/feed options as needed. JSON output
filenames are redirected into the app's Documents directory. Keep the phone
connected and available for development while running the matrix.


To reproduce the complete serial measurement matrix, export the FP16 fixed and
FP32 flexible and static GPU sidecars into `$ASSETS/models`, and the six-bit fixed sidecars into
`$ASSETS/models6`. Place the same F16 GGUF and decoder sidecar in both directories;
place the Q8 GGUF in `models` as well. CPU-only and unwindowed deployments need
the FP32 sidecars beside whichever GGUF they use. Build the desktop CLI with
`PARAKEET_COREML=ON` and a current GGML dependency. Keep other conversion/build
and inference jobs stopped during measurements.

```bash
python3 parakeet/examples/ios-coreml-bench/prepare_inputs.py \
  --samples parakeet/test/samples --output "$ASSETS/inputs"
# Package and install the prepared inputs and both model directories as above.
python3 parakeet/examples/ios-coreml-bench/benchmark.py \
  --assets "$ASSETS" --output "$ASSETS/results" --binary "$PARAKEET_CLI" \
  --device "$DEVICE_ID" --bundle "$BUNDLE_ID"
python3 parakeet/examples/ios-coreml-bench/stream_benchmark.py \
  --assets "$ASSETS" --output "$ASSETS/stream-results" --binary "$PARAKEET_CLI" \
  --device "$DEVICE_ID" --bundle "$BUNDLE_ID"
$PYTHON parakeet/examples/ios-coreml-bench/compiled_precision.py \
  --assets "$ASSETS" --binary "$PARAKEET_CLI" --output "$ASSETS/compiled-precision.json"
$PYTHON parakeet/examples/ios-coreml-bench/precise_precision.py \
  --assets "$ASSETS" --binary "$PARAKEET_CLI" --output "$ASSETS/precise-precision.json"
```

`benchmark.py` runs Mac first, then iPhone, and resumes completed case names from
`results.json`. Use a new output directory after changing a model, binary or
configuration. Each case preserves its command/settings, per-run timings,
transcript, reference WER/CER, gate decision, actual Core ML dispatch, and process
peak RSS. Streaming measurements use three independent processes per case and
exclude load from the reported inference time; they do not claim an in-process
warm benchmark. API regression logs also show the first callback's input time
and processing time. Pass `--devices mac` or `--devices iphone` for one platform.

The precise sweep checks actual compiled CPU/GPU predictions against the float32
reference, including 1–8 mel frames, aligned/nonaligned frontend padding boundaries,
60-second English and 51-second adapted French. Streaming `--resume` reuses
completed rows, and `--backends coreml` reruns that backend separately when its
implementation changes. Keep the old logs and retain only unchanged baselines.
Phone cases wait for nominal thermal state before each process. Initial static
FP32 specialization is separate from cached inference; preserve its load logs.

For automatic/two-second and explicit one-second push, compare identical native/Core ML
contexts with `default_stream_parity.py`, using the same arguments as
`stream_benchmark.py`. It records exact native transcript equality and reference
WER/CER for English/French, long speech and explicit adapted routing. This check
records the model-specific default and preserves explicit one-second coverage.

Explicit encoder context and decoder/compute policy regressions use
`context_policy_benchmark.py` with the same arguments as `benchmark.py`.
`auto_quality_benchmark.py` checks automatic encoding below/above decoder
crossover, native continuous decode, and missing optional decoder/ownership.
For its encoder-only deployment, prepare a `models-nodecoder` directory with the
same GGUF/encoder/gate/precise sidecars and omit the decoder sidecar; package it
with `--model-dir models-nodecoder=PATH`. This deployment still uses Core ML
encoders and records native decoder dispatch truthfully.

For the focused first/warm French streaming experiment, append `--warm-stream`
to the `--api-check` command. It runs one first session plus three reused-Engine
sessions per callback/push mode, grouping modes to retain the precise capacity,
and compares native Metal with Core ML. The first session includes lazy precise
program loading; repeated sessions retain identical text. Use `--explicit-context`
instead to exercise API 750-frame windows/25-frame context with auto/force0/force1
decoder policies and actual prediction counters.

## Experimental route probe

`route_probe.mm` runs the isolated [route-sharing exports](../../docs/transcribepsy-eu-coreml.md#trying-route-sharing-exports)
directly through Core ML, including named multifunction entry points and the
runtime adapter-scale input. It does not link Parakeet or claim runtime support
for these packages. It compares compiled features against independent FP32
references, saves actual output tensors, records per-function compute placement,
load/first/warm timings, physical footprint and thermal state. The ordinary
benchmark above remains the transcription/streaming harness.

Export the prototypes and prepare fixtures with `route_experiments.py` first.
Build and package a **fresh** development bundle using a current provisioning
profile, matching bundle ID, and local signing identity:

If the development profile has expired, Xcode automatic signing can create a
fresh one using an existing signed-in developer account. Build the matching
development target with `xcodebuild -project "$SIGNING_PROJECT" -target "$SIGNING_TARGET" -sdk iphoneos -allowProvisioningUpdates build`,
then use the resulting app's `embedded.mobileprovision`. Building the target
directly also avoids scheme destination-selection errors when the SDK is
available but Xcode reports that the iOS platform is not installed. A newly
issued profile may need approval on the phone under Settings > General >
VPN & Device Management before launch.

```bash
PROBE=parakeet/examples/ios-coreml-bench
xcrun --sdk iphoneos clang++ -std=c++17 -fobjc-arc -arch arm64 \
  -miphoneos-version-min=18.0 -framework UIKit -framework CoreML -framework Foundation \
  "$PROBE/route_probe.mm" -o "$EXPERIMENTS/route-probe"
APP="$EXPERIMENTS/RouteProbe.app"
$PYTHON "$PROBE/route_experiments.py" phone-fixtures \
  --fixtures "$EXPERIMENTS/fixtures" --output "$APP/fixtures"
$PYTHON "$PROBE/package.py" --binary "$EXPERIMENTS/route-probe" \
  --profile "$PROVISIONING_PROFILE" --identity "$SIGN_IDENTITY" --bundle-id "$BUNDLE_ID" \
  --model-dir "bits0=$EXPERIMENTS/bits0" --model-dir "bits6=$EXPERIMENTS/bits6" \
  --samples "$EXPERIMENTS/fixtures" --output "$APP"
xcrun devicectl device install app --device "$DEVICE_ID" "$APP"
for BITS in 0 6; do
  for KIND in separate shared unified load-eager load-en load-other load-auto-en load-auto-other; do
    xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
      "bits$BITS" "$KIND" "bits$BITS-$KIND"
  done
done
```

The three positional arguments are the model directory inside the bundle, the
probe kind, and a simple output-directory name inside Documents. Loading probes
retain just the specified handles; `load-auto-*` models the gate plus selected
route. They measure handle initialization only, without inference or gate
classification. `separate`, `shared` and `unified` additionally run all fixtures,
with five warm predictions each; `unified` uses scale 0 for gate measurements.
Process-per-kind runs preserve existing OS specialization caches. Do not infer
first-install or ANE placement results from warmed timings alone. A compute plan
with no device assignments leaves placement unverified.

Copy each result directory from the app's Documents directory using
`devicectl device copy from`, keeping the output JSON and binary tensors together.
Summarize the copied directories and evaluate the original CPU gate classifier
on the phone's actual intermediate outputs:

```bash
$PYTHON "$PROBE/route_experiments.py" summarize-native \
  --native-results "$EXPERIMENTS/phone-results" --fixtures "$EXPERIMENTS/fixtures" \
  --output "$EXPERIMENTS/phone-summary"
```

The summary compares outputs with the separate compiled routes and reports gate
probabilities and decisions against the FP32 reference. These are numerical and
sample-routing checks, not language-corpus transcription accuracy measurements.

The same probe also builds on macOS by omitting UIKit and the iPhone target
flags. Pass an absolute model directory and the prepared phone-fixture directory
as a fourth argument; set `QVAC_ROUTE_OUTPUT_DIR` to keep desktop outputs in the
experiment directory. The native compute-plan query uses the selected function's
configuration, which also permits placement inspection of non-default functions.

### Route harness regression tests

Using the pinned Core ML Python environment, run these tests from the repository
root without downloading model checkpoints or connecting an iPhone:

```bash
$PYTHON parakeet/test/test_route_experiments.py
$PYTHON parakeet/test/test_route_probe_runtime.py
```

The Python suite checks binary fixture serialization, precision-specific baseline
matching, positive and negative gate-decision reports, valid-row cropping,
adapter-scale selection, non-finite output rejection, loading reports and
duplicate WAV stem rejection before preparation. Input WAVs must have distinct
filename stems, even when they are in different directories. The
native smoke suite builds `route_probe.mm` and generates tiny FP16/FP32 Core ML
models, including a multifunction package and a route-input program. It checks
saved prediction tensors, all five loading-only modes without fixture files,
and rejection of truncated inputs and malformed manifests. Prediction modes
require a JSON object with a nonempty `fixtures` array; invalid manifests fail
before any model is loaded. Native tests require macOS 15+ and Xcode
tools; unsupported hosts exit with skip code 77. Both suites are registered in
CTest and the existing Apple Core ML workflow lane.
