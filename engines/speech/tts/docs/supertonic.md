# tts engine: Supertonic

Part of the [tts engine documentation](../README.md).

## Supertonic GGUF

The tree also contains a Supertonic path.  It is
model-specific: the official Supertone ONNX files and assets are converted
into one GGUF, then a ggml C++ runtime runs the known Supertonic stages on
CPU or, with `--n-gpu-layers > 0`, on Metal (Apple), Vulkan, OpenCL (Adreno),
or CUDA when that backend is compiled in.  A recorded Apple benchmark found: on an Apple M5 (10-core GPU) Supertonic 3 `q8_0` renders a
3.1 s / 9.6 s / 16.3 s utterance in `32.5 / 52.8 / 77.0 ms` (`RTF 0.010 /
0.005 / 0.005`, medians of 40 warm runs), 2.2x / 2.0x / 2.0x faster than the
Metal path before the one-graph stages and kernel fusions below, measured back
to back against it; the `f16` GGUF renders the same texts in `31.6 / 51.7 /
75.9 ms`.  Supertonic 2 `q8_0` measured `91.4 ms` total on an M2 (see
[Supertonic port archive](history/supertonic-port.md)).

There are three related upstream bundles:

- `Supertone/supertonic` is the stable English bundle.  It should be used for
  English and does **not** wrap text in language tags.
- `Supertone/supertonic-2` is the multilingual bundle.  It should use the
  open/close language-tag path (`<lang>...</lang>`).  The older prefix-only
  form (`<lang>... `) can make English prompts stutter.
- `Supertone/supertonic-3` advertises 31 languages plus `na` for unknown source
  language and uses the open/close language-tag path.

Current status:

- `scripts/dump-supertonic-reference.py` dumps ONNX Runtime reference tensors;
  see [Fixtures and static validation](testing.md#fixtures-and-static-validation) for
  the exact v2/v3 artifact layout.
- `scripts/setup-supertonic2.sh` downloads the official Hugging Face bundle
  through `huggingface_hub` and writes the local GGUF.
- `scripts/convert-supertonic2-to-gguf.py` writes `models/supertonic.gguf`
  (English) or `models/supertonic2.gguf` (multilingual), depending on flags.
  When `--onnx-dir` is omitted, it downloads the selected repo just like the
  Chatterbox converters.
- `build/tts-cli` autodetects Supertonic GGUFs from `supertonic.arch` and can
  synthesize a 44.1 kHz wav on CPU or GPU (`--n-gpu-layers`, with
  `--vulkan-device` to pick an adapter), but rejects Supertonic streaming
  flags. `build/supertonic-cli` is the full batch/streaming surface.
- All five stages pass numerical parity against the ONNX reference
  (preprocess, duration, text encoder, vector estimator, vocoder), and the
  full pipeline (`test-supertonic-pipeline`) reproduces the ONNX reference
  waveform when fed the same initial noise tensor.
- The production path is GGML-backed for duration, text encoder, vector
  estimator, and vocoder.  Text relative-position self-attention and FFN blocks
  are expressed with stock GGML ops, and the speech-prompted text attention /
  vector attention blocks use `ggml_flash_attn_ext` where the math allows it.
- Vector attention uses strided Q/K/V GGML views where the time-channel layout
  permits it.  The vector runtime also keeps persistent graph/allocr caches for
  attention, ConvNeXt group, and tail islands, plus fused ConvNeXt boundary /
  tail update graphs and portable custom CPU kernels for pointwise Conv1D,
  depthwise Conv1D, row-wise layer norm, dense time matmul, and fused
  bias/GELU/residual elementwise work.
- The vocoder keeps a persistent GGML graph cache and uses portable
  BLAS/Accelerate-backed causal Conv1D custom ops for the hot projection paths.
  BLAS worker threads are capped by default to avoid nested oversubscription
  under GGML task-level threading.
- Off the CPU backend the duration encoder, the text encoder, the whole CFM
  loop and the vocoder each run as one graph compute, and the
  relative-position bias is a strided view over a padded tensor instead of a
  mask chain.  Where the fused Supertonic kernels exist (Metal) the CFM step
  runs in `[C, T]` layout with classifier-free guidance batched along time;
  other GPU backends keep the per-pass step inside the one graph.  ggml-metal fuses
  the Supertonic depthwise convolution into the following channel layer norm
  and the mat-mat epilogues (bias, bias + residual, bias + GELU, gamma +
  residual) into its mat-mat kernels.  `SUPERTONIC_DISABLE_CT_STEP=1`,
  `SUPERTONIC_DISABLE_TEXT_ONE_GRAPH=1`, `SUPERTONIC_DISABLE_DURATION_ONE_GRAPH=1`,
  `GGML_METAL_FUSION_MM_EPILOGUE_DISABLE=1` and `GGML_METAL_FUSION_DW_LN_DISABLE=1`
  restore the previous paths for A/B runs.
- `SUPERTONIC_VECTOR_PROFILE=1` and `SUPERTONIC_TEXT_PROFILE=1` print
  per-island timings for tuning graph boundaries.  Current text profiling shows
  stock-op relpos is ~0.7-0.8 ms/layer on the quick prompt, so a fused relpos
  op is deferred until backend profiling proves it necessary.
- CPU thread count is controlled by `--threads`.  The default follows the graph
  path the build takes.  Where an Accelerate / CBLAS pointwise path is compiled
  (Apple), the CPU backend runs the per-island path, which regresses when
  oversubscribed, so the default stays at 4 threads.  Where it is not (a Linux
  or Windows build without a pointwise BLAS), the CPU backend runs the same
  fused one-graph path the GPU backends take and the default leaves an eighth of
  the logical CPUs unsubscribed.  Full subscription regresses on both boxes
  measured (Ryzen 9 7950X3D, Ryzen AI MAX+ 395) and the regression survives an
  OpenMP barrier, so it is oversubscription rather than ggml's spin barrier.
  GPU backends keep the 4-thread default: they run only a handful of host ops.
- Current CPU benchmark artifacts live in
  `artifacts/supertonic-thread-matrix/`.  The final matched matrix on this
  machine uses F1, 5 denoise steps, speed `1.05`, `runs=3`, `warmup=1`, and
  ONNX Runtime `CPUExecutionProvider` only.  GGML wins 10 of 12 matched
  thread/prompt comparisons.  The only end-to-end losses are quick English at
  4 threads (`157.7 ms` vs `148.8 ms`) and long English at 4 threads
  (`361.2 ms` vs `351.5 ms`).
- The current quick prompt 4-thread medians are `13.5 ms` text encoder,
  `96.3 ms` vector estimator, `43.6 ms` vocoder, and `157.7 ms` total
  (`RTF 0.050`).  Portuguese 4-thread now wins end to end (`234.3 ms` GGML vs
  `250.8 ms` ONNX), with GGML vocoder at `68.9 ms` vs ONNX `95.6 ms`.

Latest matched CPU matrix, median total milliseconds:

| Prompt | GGML 1t | GGML 2t | GGML 3t | GGML 4t | ONNX 1t | ONNX 2t | ONNX 3t | ONNX 4t |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| quick English | 298.0 | 189.4 | 157.7 | 157.7 | 373.8 | 218.5 | 168.3 | 148.8 |
| longer English | 757.5 | 491.2 | 390.3 | 361.2 | 1103.0 | 580.6 | 555.7 | 351.5 |
| Portuguese smoke | 457.2 | 292.9 | 251.0 | 234.3 | 610.6 | 344.6 | 268.3 | 250.8 |

Example:

```bash
# Stable English bundle: no language wrapping.
bash scripts/setup-supertonic2.sh --arch supertonic

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Multilingual bundle: uses the <lang>...</lang> wrapping path.
bash scripts/setup-supertonic2.sh

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Bit-exact reproduction of the ONNX reference run (pass the same noise tensor)
./build/tts-cli --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --noise-npy artifacts/supertonic-ref-quick/noise.npy \
  --out /tmp/supertonic.wav

# Matched GGML benchmark with machine-readable metrics.
./build/supertonic-bench \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-bench.json

# Matched ONNX Runtime benchmark.  Use open_close wrapping for Supertonic 2.
python scripts/bench-supertonic-onnx.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --voice-style /path/to/supertonic-pytorch/assets/voice_styles/F1.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --lang en --language-wrap-mode open_close \
  --steps 5 --speed 1.05 --threads 1 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-onnx-bench.json
```

## Snapdragon Hexagon NPU

Supertonic 3 runs on the Snapdragon Hexagon NPU (HTP0; measured on a
Snapdragon 8 Elite, Hexagon v79) when it is requested explicitly: `EngineOptions::backend = "hexagon"` or
`--backend hexagon` on `supertonic-cli`, `supertonic-bench` and
`supertonic-fit-params`. There is no fallback: construction fails when HTP0 is
missing, and the automatic `--n-gpu-layers` walk never selects it. Weights are
uploaded into the Hexagon repack buffer type. On Android the
`EngineOptions::backends_dir` directory is prepended to `DSP_LIBRARY_PATH`, so
FastRPC finds the `libggml-htp-v*.so` skeletons staged next to the ggml
backends. It needs a fabric ggml build with the Hexagon GELU_ERF,
transpose and fused depthwise-convolution kernels.

Galaxy S25 (Snapdragon 8 Elite, Hexagon v79), `supertonic3` `q8_0`, voice `M1`,
5 steps, median of 5 warm runs, total ms:

| Utterance | CPU (6 threads) | OpenCL (Adreno 830) | Hexagon | Hexagon vs OpenCL |
|---|--:|--:|--:|--:|
| 6.7 s | 1,453 | 943 | 427 | 2.2x |
| 13.5 s | 3,659 | 1,490 | 752 | 2.0x |

Per stage on Hexagon, the 6.7 s utterance spends 4 ms in duration, 13 ms in the
text encoder, 358 ms in the vector estimator and 54 ms in the vocoder.

Each stage against the CPU backend on identical inputs, run as the whole graph
the engine computes:

| Stage | cosine | relative L2 |
|---|--:|--:|
| duration | 0.9999999 | 5.1e-4 |
| text encoder | 0.9999998 | 6.0e-4 |
| vector estimator (5 steps) | 0.99995 | 9.7e-3 |
| vocoder | 0.9999996 | 8.6e-4 |

HMX multiplies in F16, so Hexagon is not bit-close to CPU the way the F32
OpenCL path is. The rendered audio matches CPU at 0.99997 / 0.99993 log-mel
cosine (0.20 / 0.36 dB mean difference) for the two utterances above, and the
predicted duration differs by under 0.02%.

## Core ML vocoder sidecar

`TTS_CPP_COREML=ON` is Apple-only. It enables an optional Core ML sidecar for
the **vocoder** -- the latent unpack and denormalize, the embed convolution,
the ten causal ConvNeXt blocks, the pre-baked BatchNorm affine, and the head.
The vocoder is the second-largest stage of a synthesis (about a fifth of a
Metal run, more than a quarter of a CPU one) and the one Supertonic stage with
a fixed convolutional topology and a single variable dimension; the duration
predictor, text encoder, and vector estimator stay on ggml. The earlier
ONNX-Runtime CoreML-EP measurement (see the archived Supertonic port report) ran the whole
pipeline through shape-generic graphs and lost to ggml everywhere; the sidecar
is a fixed-shape, float16 `jit.trace` export of the vocoder alone, which is
the form Core ML places on the Neural Engine. Export it from the model GGUF:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install -r parakeet/scripts/requirements-coreml.txt
python tts/scripts/export-supertonic-coreml.py \
  --gguf models/supertonic2.gguf --compile-dir models
```

The compiled sidecar must sit next to the model GGUF as
`<basename-minus-quant>-vocoder.mlmodelc`: `supertonic2.gguf`,
`supertonic2-f16.gguf` and `supertonic2-q8_0.gguf` all resolve to
`supertonic2-vocoder.mlmodelc`, so one sidecar serves every tier. Export from
the f32 or f16 tier: a block-quantized source bakes its dequantized vocoder
weights under the same shared name (the exporter warns when that happens). The
exporter rebuilds the vocoder in PyTorch from the GGUF tensors -- replicate
(edge) left padding on every causal convolution, exactly like
`causal_replicate_pad_1d` in the ggml graph -- so it needs neither the ONNX
bundle nor its runtime; `--parity-dir <ref>` checks that rebuild against
`final_latent.npy` / `wav_full.npy` from `dump-supertonic-reference.py`
first. With the sidecar present the engine reports it on
`Engine::vocoder_on_coreml()` at load and per call on
`SynthesisResult::vocoder_synthesis_backend` ("mixed" when streamed chunks
used different vocoder paths); `supertonic-bench` prints and emits the same
label as `vocoder_synthesis_backend`.

The export is fixed-shape, `--window` latent frames (default 64, 4.46 s,
196608 samples) in and the matching samples out. The engine walks an
utterance in windows of exactly that width: every convolution in the vocoder
is causal, so a window's output is exact from the first frame whose receptive
field lies inside it, and the engine drops each window's leading causal
context (20 latent frames for this checkpoint, computed by
`supertonic_coreml_vocoder_context_frames` from the same kernel shapes the
ggml graph uses) just as a mid-utterance ggml frame ignores everything beyond
its receptive field. An utterance shorter than one window is zero-padded on
the right, which changes nothing before the padding because nothing in the
stack looks forward.

One exported sidecar serves every quantisation tier, but it always carries
reference-precision weights, so below 8 bits per weight it would substitute a
different vocoder rather than accelerate the one the GGUF ships (see the
per-tier parity table below). A GGUF whose vocoder weights are stored under
that floor -- `q4_0` and narrower -- therefore keeps the ggml vocoder even
with a sidecar staged next to it. The floor reads the GGUF storage width, not
the runtime type, because the loader dequantises the vocoder to f32 on the
way in.

Runtime controls, mirroring the Audio8 sidecar: `SUPERTONIC_COREML_DISABLE=1`
skips the sidecar at load; `SUPERTONIC_COREML_COMPUTE_UNITS` in `cpu_only` /
`cpu_and_gpu` / `cpu_and_ane` narrows the requested placement (default: all);
`SUPERTONIC_COREML_ALLOW_LOW_BIT=1` stages the sidecar on a tier below the
weight floor anyway (for measuring that substitution);
`SUPERTONIC_COREML_STRICT=1` (test-only) fails instead of falling back to the
ggml graph. Any sidecar failure -- missing, malformed, wrong shape, or a
prediction error -- falls back to the ggml vocoder for that call.

`test-supertonic-coreml-path` locks the naming rule without a model;
`test-supertonic-coreml-tier` locks the weight-width floor and the lookup that
applies it to the bound vocoder weights, also without a model;
`test-supertonic-coreml-parity` (staged via `SUPERTONIC_COREML_TEST_MODELS_DIR`)
gates the sidecar against the ggml vocoder at cosine 0.999 on a padded, an
exact-window, and a stitched multi-window latent, checks the
`SUPERTONIC_COREML_DISABLE` fallback, the strict refusal without a sidecar,
the invalid-sidecar fallback, that a CPU-backed engine's pre-warm reaches the
sidecar, and runs one public `Engine` synthesis end to
end; `test-supertonic-coreml-exporter` unit-tests the PyTorch rebuild
(causality, unpack layout, squeezed pointwise re-expansion, receptive field,
stem rule) without a checkpoint.

## Core ML vector export defaults

With `TTS_CPP_COREML=ON`, the engine can load Supertonic vector-estimator
sidecars alongside the vocoder. The vector exporter defaults to **4D,
channel-first layout, FP16 computation and uncompressed weights**. It uses
1×1 convolutions for projections and requests CPU + ANE. Guidance and Euler
integration remain FP32 on the host. `--layout 3d` retains the previous layout.

Use the Python 3.11 Core ML conversion environment to export from an F32/F16
GGUF:

```bash
python tts/scripts/export-supertonic-vector-coreml.py \
  --gguf models/supertonic3.gguf --buckets 64 128 --compile-dir models
```

Eight-bit weight palettization is optional and selected when building the
Core ML model. The same Core ML-enabled native binary supports either model;
no additional CMake option is needed. Install the optional conversion dependencies
and add `--palettize 8`:

```bash
python -m pip install "scikit-learn==1.5.1" "scipy==1.14.1"
python tts/scripts/export-supertonic-vector-coreml.py \
  --gguf models/supertonic3.gguf --buckets 64 128 --compile-dir models \
  --palettize 8
```

Palettization uses per-tensor k-means LUTs for convolution/linear constants
of at least 2,048 values. Normalization and other constants remain uncompressed;
computation remains FP16. Compression changes the weights and numerical outputs.
Omitting the flag, or using `--palettize 0`, retains uncompressed weights.

Both variants export the canonical `<basename-minus-quant>-vector-L64.mlmodelc`
and `-vector-L128.mlmodelc` names that the engine discovers automatically.
Use separate model directories when retaining alternative layouts or compression
variants, since exports share filenames. Larger 256/512-frame buckets are also
supported; text capacity defaults to 128. Over-capacity requests and unavailable
sidecars retain ggml fallback. The external field interface and denoising step
count are unchanged by layout or palettization.

## Core ML vs ggml Metal

Measured with `supertonic-bench` (voice `F1`, 5 steps, speed `1.05`, 4
threads, `--n-gpu-layers 99`, 10 runs after 3 warmups and one `--prewarm`;
ggml Metal runs every stage but the vocoder in both columns). Hosts: an Apple
M4 mini (10 GPU cores, macOS 15.3) and an Apple M3 Ultra (macOS 15.7).
Vocoder stage median, `supertonic2.gguf` f32:

| Utterance | Machine | ggml Metal | `coreml-all` | `coreml-ane` | `coreml-gpu` | total |
|---|---|---:|---:|---:|---:|---:|
| 1.3 s | M4 | 3.50 ms | **2.21 ms (1.6x)** | 2.22 ms | 8.76 ms | 24.5 -> 23.3 ms |
| 3.1 s | M4 | 6.44 ms | **2.24 ms (2.9x)** | 2.23 ms | 8.71 ms | 29.9 -> 25.8 ms |
| 9.7 s | M4 | 17.79 ms | **6.70 ms (2.7x)** | 6.71 ms | 26.21 ms | 52.4 -> 41.4 ms |
| 22.9 s | M4 | 40.02 ms | **15.30 ms (2.6x)** | 15.34 ms | 61.27 ms | 107.9 -> 83.5 ms |
| 28.2 s | M4 | 49.26 ms | **19.60 ms (2.5x)** | 19.55 ms | 78.95 ms | 147.2 -> 118.1 ms |
| 1.3 s | M3 Ultra | **2.46 ms** | 3.35 ms (0.7x) | 3.27 ms | 2.49 ms | 29.7 -> 30.7 ms |
| 3.1 s | M3 Ultra | **2.96 ms** | 3.44 ms (0.9x) | 3.56 ms | 2.54 ms | 29.6 -> 30.6 ms |
| 9.7 s | M3 Ultra | **5.49 ms** | 7.72 ms (0.7x) | 7.56 ms | 7.40 ms | 36.5 -> 39.3 ms |
| 22.9 s | M3 Ultra | **10.79 ms** | 16.22 ms (0.7x) | 16.62 ms | 17.26 ms | 45.0 -> 51.0 ms |
| 28.2 s | M3 Ultra | **13.33 ms** | 20.81 ms (0.6x) | 20.84 ms | 22.11 ms | 53.4 -> 61.4 ms |

`coreml-all` tracks `coreml-ane` to within noise on both hosts and
`coreml-gpu` is slower than both, which confirms the placement report (164 of
165 ops on the Neural Engine): the sidecar's speed is the ANE's. The ANE is
also the same part in both machines -- 15.3 ms on the M4 against 16.2 ms on
the M3 Ultra for the same 22.9 s utterance -- so what moves between rows is
the Metal baseline, not the sidecar. The 10-core M4 GPU loses to the ANE at
every length; the M3 Ultra's GPU wins at every length. The crossover is a
property of the GPU the sidecar is compared against, not of the utterance.

The verdict does not move with the quantisation tier. One f32-exported
sidecar serves every tier, and only 21 vocoder tensors quantise, so the ggml
vocoder costs the same in all three (`supertonic3`, short 3.1 s / long
22.2 s, vocoder stage median):

| Tier | M4 Metal | M4 Core ML | M3 Ultra Metal | M3 Ultra Core ML |
|---|---:|---:|---:|---:|
| f32 | 6.51 / 38.69 ms | **2.26 / 15.28 ms** | **2.96 / 10.58 ms** | 3.10 / 17.24 ms |
| q8_0 | 6.45 / 38.54 ms | **2.26 / 15.29 ms** | **2.97 / 10.62 ms** | 3.29 / 17.09 ms |
| q4_0 | 6.57 / 38.65 ms | **2.30 / 15.27 ms** | **2.96 / 10.58 ms** | 2.98 / 18.24 ms |

End to end on `supertonic3` that is 1.06x (short) to 1.13x (long) for the M4
and 0.90x to 1.00x for the M3 Ultra; the vocoder is a smaller share of the
larger checkpoint's pipeline than of `supertonic2`'s, so the same vocoder
ratio buys less.

Sidecar load is free at this scale: process wall clock for a one-shot synth
(GGUF load included) is 643-650 ms on the M4 and 627-632 ms on the M3 Ultra,
with and without the sidecar, and the spread between the four placements is
smaller than the spread between repeats.

Numerical agreement with the ggml vocoder is a per-tier question, because the
sidecar always carries f32 weights. Same seed, same latent, only the vocoder
differs (`supertonic3`, 16 s utterance; waveform cosine, and log-spectral
distance over a 1024-point STFT):

| Tier | wave cosine | log-spectral distance |
|---|---:|---:|
| f32 | 0.99970 | 4.4 dB |
| q8_0 | 0.99945 | 6.0 dB |
| q4_0 | 0.90692 | 23.6 dB |

f32 and q8_0 are substitutions; q4_0 is not. On q4_0 the sidecar would replace
a quantised vocoder with the reference-precision one -- closer to the original
checkpoint, but a change in output from what the pure-ggml q4_0 path produces
today. That is what the 8-bit weight floor above keeps out; lifting it needs a
listening pass, or a q4_0 export of its own.

Window sizing: sidecar compute scales with total padded frames, so the cost
of a window width is its padding overhead -- the 20-frame causal context
repaid per interior window plus the zero-padded tail of the last window. At
64 the steady-state overhead is 20/44 (+45%) per interior window, at 128 it
is 20/108 (+19%), at 192 it is 20/172 (+12%), but measured across 64 / 128 /
192 / 256 the wider windows never pay for themselves (`supertonic3` q8_0,
vocoder stage median):

| Utterance | Machine | Metal | w64 | w128 | w192 | w256 |
|---|---|---:|---:|---:|---:|---:|
| 3.1 s | M4 | 6.42 ms | **2.25 ms** | 4.87 ms | 7.45 ms | 9.68 ms |
| 22.2 s | M4 | 38.36 ms | 15.32 ms | **14.97 ms** | 15.20 ms | 19.88 ms |
| 28.2 s | M4 | 49.53 ms | **19.59 ms** | 19.99 ms | 22.44 ms | 20.02 ms |
| 3.1 s | M3 Ultra | **2.97 ms** | 3.24 ms | 6.57 ms | 9.63 ms | 11.06 ms |
| 22.2 s | M3 Ultra | **10.62 ms** | 17.01 ms | 16.71 ms | 17.00 ms | 21.33 ms |
| 28.2 s | M3 Ultra | **13.31 ms** | 22.20 ms | 21.45 ms | 23.83 ms | 20.88 ms |

The last window's zero-padded tail costs more than the repeated context
saves at every length tested, so 64 stays the default, and no window width
closes the M3 Ultra gap. On the streaming path each chunk vocodes
independently, so a short first chunk still pays one full window (~2.3 ms on
M4) -- a second, smaller exported window is the natural follow-up if
first-chunk latency matters.

Ship the sidecar where the GPU is consumer-class and skip it on Max / Ultra
parts. It is presence-driven, so this is a packaging decision per deployment,
not a build flag: the same binary uses the sidecar when the `.mlmodelc` sits
next to the GGUF and the ggml vocoder when it does not.
