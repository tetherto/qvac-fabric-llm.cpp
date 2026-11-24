# tts engine: LavaSR enhancement

Part of the [tts engine documentation](../README.md).

## LavaSR enhancement

LavaSR is post-processing for synthesized or captured speech, not another
text-to-speech engine. The denoiser uses a 16 kHz internal STFT and preserves
the requested output rate; the bandwidth-extension enhancer emits 48 kHz.
Public APIs are installed under `<tts-cpp/lavasr/>`, and `lavasr-bench`
exercises the denoiser, enhancer, or two-stage pipeline. Backend support differs
between the two stages, so use the capability matrix rather than assuming every
compiled ggml backend applies to both.

### Convert

`scripts/download-lavasr-onnx.sh` fetches the LavaSRcpp ONNX release assets
the two converters read (the original torch weights live at
[YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)); both converters
accept `--ftype {f32,f16}`:

```bash
scripts/download-lavasr-onnx.sh --dir checkpoints/lavasr

python3 scripts/convert-lavasr-denoiser-to-gguf.py \
    --denoiser checkpoints/lavasr/denoiser_core_legacy_fixed63.onnx \
    --out models/lavasr-denoiser-f32.gguf
python3 scripts/convert-lavasr-enhancer-to-gguf.py \
    --backbone checkpoints/lavasr/enhancer_backbone.onnx \
    --spec-head checkpoints/lavasr/enhancer_spec_head.onnx \
    --out models/lavasr-enhancer-f32.gguf
```

### Run

`lavasr-bench` runs the two-stage pipeline over a wav and can write both
stage outputs:

```bash
./build/lavasr-bench --denoiser models/lavasr-denoiser-f32.gguf \
                     --enhancer models/lavasr-enhancer-f32.gguf \
                     --in noisy.wav \
                     --out-denoised denoised.wav --out-enhanced enhanced-48k.wav
```

`test-lavasr-gguf-load` needs no fixtures: it synthesizes tiny metadata-only
GGUFs and asserts both stage loaders fail closed on a missing, empty,
truncated, unmarked, cross-architecture, or tensorless file. The denoiser and
enhancer GGUFs differ only by `general.architecture`, so each loader has to
refuse the other's file.

### Native CPU, OpenCL and Hexagon checks

`lavasr-bench` selects the neural backend explicitly and can run either stage
independently. `--stage pipeline` runs denoising before enhancement. An explicit
backend request fails if the selected device is unavailable; it cannot report a
CPU fallback as an HTP/OpenCL result.

```bash
./build/lavasr-bench --stage denoiser --backend cpu \
    --denoiser models/lavasr-denoiser-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out denoiser-cpu.json \
    --out-denoised denoised.wav

./build/lavasr-bench --stage enhancer --backend opencl \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out enhancer-opencl.json \
    --out-enhanced enhanced.wav

GGML_HEXAGON_NDEV=1 GGML_HEXAGON_HOSTBUF=0 ./build/lavasr-bench \
    --stage pipeline --backend hexagon \
    --denoiser models/lavasr-denoiser-f32.gguf \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out pipeline-hexagon.json \
    --out-denoised htp-denoised.wav --out-enhanced htp-enhanced.wav
```

The Hexagon build must include the matching DSP skel and native backend
libraries in its deployment. `--backend hexagon` pins `HTP0`. Legacy
`--dn-gpu-layers`/`--n-gpu-layers` flags and `TTS_CPP_GPU_BACKEND` selection
remain available with `--backend auto`. The existing automatic GPU preference
is preserved; Hexagon is an explicit engine capability.

The optimized F32 Hexagon kernels, including source-layout energy fusion, were
device-validated on CPH2723 (SM8750, Android 16, HTP v79) with `qvac-ext-ggml` revision
`f36a6a2721b9530e05f8f36177b760d81daae13f`, merged into `speech` as
`ab1630bfba9083fd776b81b7b2e8a4ebb9213e6c`. Earlier validation used
`5eb6cb714bc84476f115ef491a534fc8b9104c8c`. Build the selected revision with
Hexagon enabled, use its installed CMake package through `CMAKE_PREFIX_PATH`,
and deploy the DSP skel from the same build. The original baseline revision
`380c5f9f3e1811dfd88adeb1c19ab887bcf8ab3e` uses the fallback graphs described
below. The older bundled pin in `scripts/setup-ggml.sh` is not the dependency
used for these Hexagon checks. Record the resolved shared GGML revision.

Both F32 and F16 GGUF inputs are supported. On native ggml routes, the loaders expand F16
weights to F32, and the neural graphs retain F32 tensors and F32-precision
matrix multiplication. On Hexagon those products use HVX rather than F16 HMX.
An F16 filename does not indicate F16 execution. Hexagon's denoiser graph queries
each actual F32 convolution, GRU and affine/PReLU candidate independently.
Supported ordinary 1x1/3x3 convolutions use direct `CONV_2D`, avoiding the
im2col buffer and final layout copy; supported GRU nodes run the recurrent sweep
as one op, and supported affine/PReLU nodes use `AFFINE_PRELU`. Unsupported
candidates retain the F32 im2col/matmul, unrolled recurrence or seven-op
affine/PReLU graph respectively. The original baseline revision above follows these
fallback routes. Precision and correctness thresholds are unchanged.

The Hexagon graph selects `MEAN` for supported F32 reduction shapes and retains
sum/scale means when the runtime rejects the candidate. Single-channel energy
omits the channel transpose and reduction; multi-channel energy exposes a
`CONT`/`SQR`/`MEAN` chain that the matching runtime can fuse directly from the
source layout. The channel shuffle copies complete spatial planes. The DPGRNN
output copy folds frequency and time into one spatial dimension for a batched
transpose, then reshapes back to the canonical `[F,T,C,Bc]` layout without changing
values or batch order. These Hexagon graph
choices preserve the CPU/OpenCL graph paths. The enhancer uses direct depthwise
convolution. Each graph is checked against the selected backend before
execution. Resampling, STFT/mel features, denoiser log features, overlap-add,
ISTFT and the FastLR merge remain host audio DSP.

Constructor time, the first processing call, additional warmups and measured
calls are recorded separately. With `--runs 1 --warmup 0 --min-seconds 0`
(the desktop time-wrapped runner's mode), the first call is also the single
measured sample: only one inference executes, with JSON `first_call_measured`
set to `true`. Other modes retain a separate first call excluded from the
measured samples. Each measured call reports denoiser, enhancer and combined
seconds. RTF is processing seconds divided by input audio seconds; lower is
faster. Native processing includes the host audio DSP and graph
construction/allocation. WAV I/O, reference verification, constructor time
and warmups are excluded from processing measurements. Output WAVs use float32
PCM, preserving the measured samples without int16 clipping or quantization.

`--verify` compares denoising against the scalar reference and enhancement
against ggml CPU on the exact same input. The combined mode also compares the
full output to an independently CPU-processed pipeline. Length mismatches,
non-finite output and numerical failures return a nonzero status. JSON includes
actual backend names, graph/precision policy, rates/sample counts, every timing
sample and numerical metrics. Passing the command is correctness evidence;
actual DSP execution still needs a returned Hexagon profile or trace.

For a sustained run, `--min-seconds 300 --runs 1` continues measuring in the
same warmed process until the sum of native processing times reaches at least
five minutes. The final call may cross the target. The JSON records every call
and the actual measured duration. Collect battery/thermal state separately and
match starting conditions before comparing accelerator throughput changes.

### Core ML

Build with `-DTTS_CPP_COREML=ON` on Apple to enable either or both neural cores.
The GGUF remains required for architecture and DSP parameters. Supply an
explicit compiled model path and compute policy for each selected stage;
there is no automatic discovery or fallback if a requested Core ML model fails
to load or predict. Existing callers keep their CPU/ggml/GPU routes.

The models exported below require macOS 13+ or iOS 16+. Export and compile on a
Mac with Xcode command-line tools. The converter uses `torch`, `numpy`, `gguf`
and `coremltools`; conversion was checked with Python 3.11, torch 2.7.0,
coremltools 9.0 and numpy 2.2.6. These are conversion dependencies, not runtime
dependencies of the C++ library.

```bash
python3 scripts/export-lavasr-coreml.py \
    --stage denoiser --gguf models/lavasr-denoiser-f32.gguf \
    --precision float32 --output models/denoiser-f32.mlpackage \
    --compile-dir models
python3 scripts/export-lavasr-coreml.py \
    --stage enhancer --gguf models/lavasr-enhancer-f32.gguf \
    --frames 256 --precision float32 --output models/enhancer-f32.mlpackage \
    --compile-dir models
```

Precision is required: `float32` preserves the reference numerics, while
`float16` permits lower precision execution including Neural Engine placement.
The denoiser's magnitude/log calculation retains FP32 precision to preserve
near-silence values and avoid overflow when squaring full-scale STFT bins.
Changing precision requires an explicit export; the loader never converts the
GGUF or changes the selected model precision at runtime.

`--frames` sets the enhancer's fixed inference window, not a maximum audio
duration. C++ processes longer inputs in overlapping windows and retains the
convolution context. The exporter masks padded activations after each block,
preserving short inputs and final-window boundaries. The denoiser retains its
GGUF-defined chunk size and resets recurrent state for each chunk, matching
the existing scalar and ggml implementations.
All chunks are submitted together through Core ML's batch prediction API;
the compiled model and its tensor shapes stay unchanged, and chunks keep
independent zero recurrent state. No audio-duration cap is imposed.

| Policy | Eligible Core ML compute units |
|---|---|
| `CPU_ONLY` | CPU |
| `CPU_AND_GPU` | CPU and GPU |
| `CPU_AND_NE` | CPU and Neural Engine |
| `ALL` | CPU, GPU and Neural Engine |

These are Apple's allowed compute units, not guarantees that every operation
runs on a particular device. `backend_name()` reports the requested policy
as `coreml-cpu`, `coreml-gpu`, `coreml-ane`, or `coreml-all`;
`Enhancer::backend_device()` reports `BackendDevice::CoreML`.
Resampling, STFT/ISTFT, overlap-add and FastLR remain in the existing C++ DSP
pipeline. See Apple's [compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)
and [precision guide](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

```cpp
#include <tts-cpp/lavasr/denoiser.h>
#include <tts-cpp/lavasr/enhancer.h>

tts_cpp::lavasr::DenoiserOptions dn_options;
dn_options.coreml = {"models/denoiser-f32.mlmodelc", "CPU_ONLY"};
auto dn = tts_cpp::lavasr::Denoiser::load("models/lavasr-denoiser-f32.gguf", dn_options);

tts_cpp::lavasr::EnhancerOptions enh_options;
enh_options.coreml = {"models/enhancer-f32.mlmodelc", "CPU_ONLY"};
auto enh = tts_cpp::lavasr::Enhancer::load("models/lavasr-enhancer-f32.gguf", enh_options);

auto cleaned = dn->denoise(pcm, sample_rate);
auto enhanced_48k = enh->enhance(cleaned, sample_rate);
```

Reuse these objects across calls: each compiled model is loaded once, and
access to its reusable inference buffers is serialized. A selected Core ML
stage cannot also select `use_gpu` or a nonzero `n_gpu_layers`; use the Core ML
compute policy instead. Other stages can independently keep their original
backend.

The benchmark accepts these Core ML flags with `--backend auto`. An explicit
`--backend cpu`, `opencl` or `hexagon` request rejects Core ML options instead
of changing the requested device. Default options do not discover adjacent
sidecars. CPU reference legs always use empty Core ML options and assert their
actual scalar/CPU backend identities. JSON records per-stage backend and
precision: native paths report F32; Core ML paths report precision defined by
the compiled model, and combined native/Core ML runs report mixed precision.
Requested Core ML compute units describe eligible placement, not a hardware
execution trace.

```bash
./build/lavasr-bench \
    --denoiser models/lavasr-denoiser-f32.gguf \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --denoiser-coreml models/denoiser-f32.mlmodelc \
    --enhancer-coreml models/enhancer-f32.mlmodelc \
    --coreml-compute-units CPU_ONLY --runs 3 --warmup 0 --verify \
    --out-denoised denoised.wav --out-enhanced enhanced-48k.wav
```

Load time is reported separately from processing time. `--verify` compares
each stage to scalar/ggml CPU and enforces the same length, finite-value and
numerical gates as the native backends; a failed comparison returns nonzero.
`test-lavasr-coreml-parity` enforces FP32 waveform parity and output lengths for
empty, short and multi-window inputs, repeated calls, and incompatible models:

```bash
./build/test-lavasr-coreml-parity \
    models/lavasr-denoiser-f32.gguf models/lavasr-enhancer-f32.gguf \
    models/denoiser-f32.mlmodelc models/enhancer-f32.mlmodelc
python3 test/test_export_lavasr_coreml.py
./build/test-lavasr-coreml-batch \
    models/lavasr-denoiser-f32.gguf models/denoiser-f32.mlmodelc CPU_ONLY
```

The native parity fixture is disabled in CTest until those four files exist in
`TTS_CPP_TEST_MODEL_DIR`. The options and exporter tests need no real weights;
the exporter tests report a skip when numpy/torch are unavailable.
The batch fixture checks serial/batch parity, output ordering, changing batch
sizes, invalid dimensions and concurrent calls on one loaded model.
