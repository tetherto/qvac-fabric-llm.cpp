# audiogen engine: tests and parity

Part of the [audiogen engine documentation](../README.md).

## Tests and parity

```sh
cmake -S audiogen -B build/audiogen -DAUDIOGEN_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
ctest --test-dir build/audiogen
```

ACE-Step model-free tests are split into `test-acestep-units` (math and
sampling), `test-acestep-tokenizer`, `test-acestep-backends`,
`test-acestep-loading`, `test-acestep-quantization`, `test-acestep-audio`, and
`test-acestep-generation` (request policies and editing). Each suite has its
own process and assertion counters. Run one domain with `ctest -R`.

`test-acestep-units`, `test-acestep-converter`, `test-minimax-units`,
`test-minimax-converter`, `test-minimax-quant-audit`, and
`test-minimax-dump-compare` cover weight-free CPU logic and need no GGUFs.
ACE-Step coverage includes the BPE tokenizer (UTF-8 decode, merges, byte
fallback, decode round-trip) on a hand-built vocabulary. MiniMax coverage
includes metadata compatibility, model-pair selection, Unicode token classes,
frame validation, prompt assembly and the request-utils caption/lyrics
cleanup helpers, unconditional masking, deterministic noise, flow scheduling,
condition length, window stitching, sampler edge cases, and converter output
transactions. `test-minimax-fit` writes a tiny MiniMax pair and checks the
memory-fit preflight against the buffers a real load and graph allocation of
that pair produce, byte for byte, along with its workload limits, rejections
and the weightless-GGUF case. Set `AUDIOGEN_TEST_MINIMAX_MODELS_DIR` to a
directory containing the MiniMax GGUF pair to run `test-minimax-integration`,
which covers model loading, generation output, progress, and cancellation, and
`test-minimax-fit-models`, which repeats the preflight parity check on the real
pair on the device `MM3_DEVICE` selects.
`test-minimax-tokenizer` loads a synthetic GGUF vocabulary and checks exact
IDs for BPE merges, byte fallback, Unicode, longest special-token matching,
empty inputs, and missing metadata. `test-minimax-converter-roundtrip` converts
a tiny safetensors conditioner and reads its GGUF back; serialization checks
cover F16, Q8_0, protected F32, overflow, and weight-norm folding. These real
conversion cases require `numpy` and `gguf`.

MiniMax integration distinguishes callback cancellation from explicit
`cancel()` and cross-thread cancellation. `MM3_DEVICE=gpu` requires successful
GPU execution; a missing GPU fails the test. `auto` can fall back to CPU.
The Q4 quality run requires `AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR` to resolve to
Q4_K_M or Q4_K_S; a mixed directory selecting Q8 or F16 fails.

On a Metal build, `ctest --test-dir build/audiogen -R
test-minimax-metal-ops` runs the model-free CPU/Metal condition and vocoder
parity regression. It skips with return code 77 either when Metal is
unavailable or when the Metal device cannot run `MUL_MAT` — both parity graphs
are matmul-based, and ggml gates `GGML_OP_MUL_MAT` on simdgroup reduction
(`MTLGPUFamilyApple7`+), which virtualized GPUs such as the ones on hosted macOS
CI runners do not report. Meaningful parity coverage therefore needs a
non-virtualized Apple GPU, which is why the audiogen CI macOS lane runs on the
self-hosted `qvac-macos26-arm64-gpu` runner and fails rather than passes if the
test skips there.
`test-acestep-integration` requires `AUDIOGEN_TEST_MODELS_DIR` with the four
ACE-Step GGUFs described in [model setup](pipeline.md#model-setup). From the
repository root, run the model-backed test against a standalone build:

```sh
AUDIOGEN_TEST_MODELS_DIR=/path/to/acestep \
ctest --test-dir build/audiogen -R '^test-acestep-integration$' --output-on-failure
```

The default reference uses CPU. Set `AUDIOGEN_TEST_GPU=1` to request GPU
placement; this variable uses presence semantics, so unset it for CPU coverage.
Set `AUDIOGEN_TEST_BASE_MODELS_DIR=/path/to/acestep-base` as well to exercise
optional base-model lego coverage. The primary fixture being unset skips the
test with return code 77; omitting the base fixture skips only that optional
coverage. A successful model-free test run does not establish generation parity.
For multi-config builds, add `-C Release` to `ctest`.

`test-minimax-conditioner-layout` compares tiled, strided Core ML IO with an
independent full convolution, including one-frame inputs and boundaries at
149, 150, 151, 299, 300 and 301 frames. With `AUDIOGEN_COREML=ON`, the
model-backed `test-minimax-coreml-conditioner` compares the original native
conditioner against CPU + ANE with recorded AR hiddens. It checks finite
outputs, identical resampled lengths, relative RMS error <= 0.002 and cosine
>= 0.9999. Set `AUDIOGEN_TEST_MINIMAX_SYNTH` to the F16 synth GGUF,
`AUDIOGEN_TEST_MINIMAX_COREML_CONDITIONER` to the compiled sidecar and
`AUDIOGEN_TEST_MINIMAX_HIDDENS` to `frame-hiddens.f32` from `mm3-replay`.
`AUDIOGEN_TEST_MINIMAX_DEVICE=gpu` selects the native GPU reference; the
test otherwise uses CPU. Missing artifacts skip with return code 77.

`test-audiogen-comparison-lib` runs the engine-comparison harness's own
`node:test` suites (`benchmarks/comparison/tests/`) on hosts with node,
so the harness's adapters, aggregation, and report logic stay verified.

Dumps are the tool for localising a backend divergence. Run the same prompt twice with `--dump-stages` (ACE-Step stages) or `--dump-iters` (MiniMax AR iterations), then compare:

| Script | Purpose |
|---|---|
| `scripts/stage_cos.py` | per-stage cosine and rel_l2 between two dump directories; fails under a worst-stage cosine of 0.999, and also on a missing, truncated, or mismatched counterpart |
| `scripts/dequant_gguf.py` | rewrite a GGUF with every quantized tensor dequantized to F32, giving a ground-truth trajectory to compare a quantized run against |
| `scripts/compare_ar_dumps.py` | finite-subset cosine, argmax agreement, and top-8 overlap between two `--dump-iters` directories; gates on `--min-cosine` / `--min-argmax-agree` and fails a pair it could not actually compare |
| `scripts/audit_quant_types.py` | audit a quantized GGUF against the deny-list of tensors that must never be quantized, plus a per-type byte summary |

`stage_cos.py` and `dequant_gguf.py` need `numpy`; `dequant_gguf.py` and
`audit_quant_types.py` also need `gguf`. `compare_ar_dumps.py` uses `numpy`
when it is installed and falls back to an equivalent stdlib path when it is not.

## Required CI coverage

The self-hosted macOS lane builds with both Core ML flags enabled and runs
the VAE/conditioner exporters and all three MiniMax ANE Python suites with
the pinned dependencies in `parakeet/scripts/requirements-coreml.txt`.
The ANE model and exporter suites are also registered with CTest; missing
Python dependencies produce an explicit skip (77). The ANE CLI suite is
registered when `AUDIOGEN_MINIMAX_COREML=ON` and executables are enabled.
Direct CI invocations fail if a required Python suite skips.

`test/ci-models.json` pins upstream revisions and SHA-256 hashes for the real
ACE-Step turbo/base stages, MiniMax checkpoint, and tokenizer. The staging
script verifies downloads, converts the MiniMax F16 pair with this checkout's
converter, and quantizes a separate Q4_K_M pair. Source downloads and outputs
persist under `$HOME/.cache/qvac-audiogen-ci-models`; converter, quantizer
header, or manifest changes invalidate the converted output cache. Allow at
least 80 GiB of disk for the fixtures and export artifacts. Staging removes
obsolete converted generations before conversion and refreshes shared base
stages when verified source weights change.

CI runs ACE-Step integration (including the base lego lane), ACE-Step fit
parity, MiniMax CPU/GPU integration, MiniMax fit parity, and MiniMax F16/Q4
quality tests directly. A skipped binary exits 77 and fails the step. These
model runs are separate from the weight-free Metal operation parity test.
To stage the same fixtures locally, run from the repository root:

```sh
python3 audiogen/scripts/prepare-ci-models.py \
  --cache-dir "$HOME/.cache/qvac-audiogen-ci-models" \
  --build-dir build/audiogen --output-env audiogen-models.env
```

The generated file contains the four fixture environment variables. Export
them before invoking the model-backed binaries. Set `MM3_DEVICE=gpu` to
require MiniMax GPU coverage and `AUDIOGEN_TEST_GPU=1` for ACE-Step GPU
placement. `test-acestep-xdna-generation` additionally requires XDNA0 and
enforces a PCM relative-L2 limit of 0.1 against CPU; it reports the measured
error before enforcing that bound.
