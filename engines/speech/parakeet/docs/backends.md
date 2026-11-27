# parakeet engine: backends

Part of the [parakeet engine documentation](../README.md).

## Device encoder handoff

Offline TDT, Unified RNN-T and Nemotron ASR can decode the ggml encoder's output
without downloading it to the host. A synchronous internal consumer runs while
the cached encoder tensor is valid. It queues a same-backend device copy into
the decoder projection graph, then decodes before returning; no borrowed graph
tensor escapes its lifetime. Nemotron also concatenates its cached locale
one-hot tensor and computes the prompt projection on the backend, avoiding the
additional host packing, download and upload of conditioned features.

This path requires a graph decoder on the same non-CPU backend, a contiguous F32
encoder tensor, and a frame count within the decoder's projection capacity.
The tensor's frame count must equal the valid encoder length. When padding
produces extra tensor frames, the handoff is declined and host decoding uses
only the valid prefix.
CPU and OpenCL's CPU decoder, Core ML encoder output, diagnostic captures,
windowed long-form encoding, and cache-aware streaming retain their existing
feature-transfer routing.
CLI encoder dumps also retain host output. Errors propagate rather than
silently decoding through a second path. Encoder and decoder timing fields
charge the consumer's work to decoding.

Fixture tests cover CPU fallback and Vulkan handoff for TDT 0.6B-v3, Unified
0.6B and Nemotron 3.5 ASR Q8_0, including exact token IDs, graph reuse,
diagnostic output, consumer errors, invalid tensor shapes, padded Nemotron
host/device parity, prompt changes and
the public Engine API. CUDA and Metal variants register when those backends are
built; this change has been measured on the Radeon RX 7600 XT with Vulkan.

## Backends and runtime selection

Any combination of Metal, CUDA, Vulkan, OpenCL, and WebGPU may be built or
dynamically loaded. Parakeet selects one primary GPU through the ggml backend registry when
`n_gpu_layers > 0`; this is not a fixed compile-time CUDA/Metal/Vulkan/OpenCL
cascade. `n_gpu_layers` is currently a boolean offload request: positive means
the whole encoder, not a partial layer count.

Set `EngineOptions::backend` or CLI `--backend` to select a primary GGML backend
explicitly. `auto` (the default) preserves the policy below; `cpu`, `cuda`,
`vulkan`, `opencl`, `webgpu`, `metal`, `hexagon` and `xdna` override `n_gpu_layers`. `hexagon`
selects the `HTP0` device from the `HTP` registry. An exact GGML device name
such as `HTP0` is also accepted.
`xdna` selects AMD `XDNA0` explicitly; automatic GPU selection excludes XDNA.
An unavailable or failed explicit selection fails loading instead of selecting
another backend. The existing per-operation CPU scheduler still handles
unsupported operations; selecting HTP0 does not claim every operation ran there.
On Android, backend discovery also prepends `backends_dir` to FastRPC's
semicolon-separated `DSP_LIBRARY_PATH`, preserving existing user paths and
vendor defaults. This happens even when the first model requests CPU/OpenCL,
so later models can explicitly select Hexagon. If only legacy
`ADSP_LIBRARY_PATH` is set, its entries are retained. Set `backends_dir` before
the first Engine/loader call; registry discovery remains process-global.

Hexagon is experimental and currently targets Parakeet CTC and TDT 0.6B Q8_0,
TranscribePsy-250M-EU-sub and MOSS-Transcribe-Diarize. Build and ship the
matching GGML Hexagon backend and DSP library. How the FastConformer encoder and
the TDT decoder run on the NPU, with TranscribePsy measurements, is in
[Hexagon NPU](#hexagon-npu); MOSS-Transcribe-Diarize's requirements, graph
layout, device tests and measurements are in
[its guide](moss-transcribe.md#hexagon-npu). For Parakeet device validation, run
`parakeet --backend hexagon --verbose --bench --bench-json result.json`, with the
existing `--backends-dir` pointing at the installed backend libraries. The JSON
records the actual backend (`ggml-htp0`), transcript and stage timings. Use a
separate `GGML_HEXAGON_PROFILE=1` run to collect DSP operation evidence for
`IM2COL`, `CONV_2D_DW`, and eligible HMX matrix operations; backend identity alone
does not establish kernel execution or correct transcription. Measure CPU and
OpenCL with `--backend cpu` and `--backend opencl` using the same model/audio.
`Engine::backend_device()` identifies Hexagon as `BackendDevice::NPU`, while
`Engine::backend_name()` reports the resolved device name (`HTP0`).

Runtime tiering is:

1. Prefer OpenCL for Adreno 700+, where it is validated ahead of Vulkan.
2. Otherwise choose CUDA on NVIDIA, then the first registered discrete
   non-OpenCL GPU, such as Vulkan, Metal or WebGPU, then an integrated one
   such as Mali Vulkan.
3. Use non-Adreno OpenCL only when no non-OpenCL GPU can initialize.
4. Fall back to CPU.

Adreno 6xx OpenCL is skipped because it produces incorrect output. Set
`PARAKEET_ALLOW_ADRENO_6XX=1` to opt in explicitly. Mali uses Vulkan for the
encoder and CTC/TDT/EOU computation, but the Sortformer diarization head is
routed to CPU because that head is incorrect on Mali Vulkan.

WebGPU (ggml-webgpu on Dawn, `GGML_WEBGPU=ON`) registers one discrete
`WebGPU` device and takes the same tier as Vulkan and Metal, so it is chosen
automatically when no CUDA device is present and explicitly with
`--backend webgpu`. Validated on an RTX 5090 (Dawn on Vulkan, 2026-10-20) with
the TranscribePsy 250M TDT model: the transcript is identical to the CPU run
and the end-to-end time 2.8x shorter. The encoder keeps its unfused attention
graph on WebGPU (`flash_attn_allowed` admits only CUDA, Metal and Vulkan)
until the fused path is compared there too. The backend reports its
maximum buffer size as device memory, so a memory-fit projection is not
meaningful on WebGPU.

TDT predictor/joint decoding uses ggml graphs on every backend, including
ggml-cpu, where the quantised joint matmuls run about ten times faster than the
former host f32 gemv loop. ggml-opencl drops the in-place state writes those
graphs rely on, so with the encoder on OpenCL the TDT and RNN-T decoder graphs
run on the CPU backend from a CPU-resident copy of the decoder weights that
the loader makes (`model_tdt_decoder_on_cpu`; the fit projector counts the
copy); Nemotron keeps the host scalar decoder there, and
`PARAKEET_TDT_HOST_DECODE=1` forces the host decoder elsewhere for parity
testing. EOU decoding uses graphs on Metal, Vulkan and CUDA and the scalar
path on CPU and OpenCL.

Nemotron 3.5 ASR streaming is supported on OpenCL (Adreno 700+). The encoder
runs on the GPU and the cache-aware streaming operating points (80, 160, 320,
560, 1120 ms) are honoured. The transducer decode runs host-side on OpenCL,
same as TDT and EOU, because ggml-opencl drops the in-place `ggml_cpy` writes
that carry the persistent LSTM state. Adreno 6xx remains blocked by default;
opt in with `PARAKEET_ALLOW_ADRENO_6XX=1` (unvalidated).

Nemotron 3.5 ASR streaming is supported on CUDA. The encoder and the
transducer decode both run on the GPU at all five operating points, with the
fused LSTM-cell and transducer-step decode running up to eight greedy steps
per graph. Validated against the NeMo stream-step references at
80/160/320/560/1120 ms on an RTX 3090 (`test-nemotron-stream-step-cuda*`).

Nemotron 3.5 ASR streaming is also supported on Vulkan. The encoder and the
transducer decode both run on the GPU at all five operating points, and the
decode takes the same fused LSTM-cell and transducer-step graphs as CUDA and
Metal because ggml-vulkan implements `GGML_OP_LSTM_CELL` and
`GGML_OP_TDT_STEP`. Validated against the NeMo stream-step references at
80/160/320/560/1120 ms on an RTX 3090 (`test-nemotron-stream-step-vulkan*`).

Nemotron 3 Diarization runs on CUDA, Vulkan, and Metal. The feature-stacking
projection, the 31-layer RoPE encoder with fused flash attention, the subpixel
upsampler, and the speaker head form one GPU split, and offline inference and
cached streaming pass the NeMo-Speech.cpp reference test
(`test-nemotron-diarization-gpu`) on an RTX 5090 (CUDA and Vulkan), an AMD
Radeon RX 7600 XT (Vulkan), and an Apple M3 Ultra (Metal). When the active
backend cannot run fused flash attention for this graph, the encoder uses an
unfused attention (matmul, softmax, matmul) that stays on the GPU instead of
sending 31 attention ops per call to the CPU. ggml-opencl takes that path on
Adreno unless `GGML_OPENCL_FA_ADRENO=1` is set. Adreno's ggml-opencl and
ggml-vulkan matmul kernels narrow activations to half precision by default,
which put the Adreno 830 at the edge of the reference tolerance (relative L2
0.0030 on OpenCL, 0.0040 on Vulkan against 0.003), so on Adreno the graph
requests `GGML_PREC_F32` on every matmul; that halves the error for a 2-3 %
cost. Both Adreno backends then pass the reference test on a Snapdragon 8
Elite, and OpenCL, which the tier policy selects on Adreno 700+, runs the
model about twice as fast as its CPU. ggml-vulkan on that GPU is slower than
the CPU for this model. Mali Vulkan is untested on hardware; the Mali routing
of the Sortformer head does not apply to this model. See
[Nemotron 3 Diarization on GPU](performance.md#nemotron-3-diarization-on-gpu)
for timings and accuracy.

The graph decoder adapts to what the active backend reports through
`ggml_backend_supports_op`, probed once at load. Where the backend runs the
fused LSTM cell (`GGML_OP_LSTM_CELL`) and the transducer step control
(`GGML_OP_TDT_STEP`), which the fabric ggml implements for CPU, CUDA, Metal and
Vulkan, the TDT decoder runs up to eight greedy steps per graph: the joint,
argmax, LSTM update, duration bookkeeping and state selection stay on the
device and the
host reads the emitted tokens back once per graph. A backend without those
ops keeps one graph per step. Either path produces the same token sequence as
the sequential loop; the `test-tdt-unroll-parity` and `test-tdt-lstm-parity`
harnesses guard that. The encoder applies the same rule to the conformer's
depthwise convolution (`GGML_OP_CONV_2D_DW` in place where the backend
reports it, `im2col` and matmul elsewhere; the subsampler switches on a GPU
that passed the probe and on ggml-cpu, Mali and OpenCL keep the im2col
lowering) and to the gated
GLU (`a * sigmoid(b)` as one op where the backend reports it). Fused attention
is a build option (`PARAKEET_FLASH_ATTN`) that CUDA, Metal and Vulkan take, and
only after the backend accepts the exact node the encoder builds; CPU and
OpenCL keep the unfused graph in every build. A cached encoder graph computes
the 24 per-layer positional projections once per graph size (up to 256 MiB of
projections per graph) and the attention blocks read them from that buffer on
every forward; `PARAKEET_POS_PROJ_CACHE=0` recomputes them in-graph instead,
and `test-pos-proj-cache-parity` checks the encoder output is byte-identical
either way. The mel front-end runs on up
to eight host threads with output byte-equal to the single-thread result.

The CUDA path was validated on an RTX 3080 (TDT q8_0 and q4_0 transcripts,
Sortformer and streaming output byte-equal to the pre-change build, LibriSpeech
WER within noise of the CPU reference) but hardware decoder parity is not yet
covered by CI.

`GGML_BACKEND_DL=ON` builds load backend modules at runtime. Android defaults
this on and builds Vulkan, OpenCL, and CPU variants. Pass a module directory via
CLI `--backends-dir` or `EngineOptions::backends_dir`. Backend registration is
process-global: the first `Engine` construction loads from that directory and
later engines reuse the populated registry. Leave it empty for ggml's default
search path. In static `GGML_BACKEND_DL=OFF` builds the setting is a no-op.

`Engine::backend_name()` reports the primary active ggml backend. Individual
stages can run elsewhere: with OpenCL selected, TDT and RNN-T decoder graphs
run on CPU, while `backend_name()` still reports OpenCL. Benchmark log output
reports the decoder separately in `decoder_backend`.
`Engine::encoder_backend()` reports the loaded encoder-sidecar configuration;
benchmark JSON additionally reports whether every measured invocation actually
completed through Core ML.

Example backend configurations:

```bash
cmake -S parakeet -B build-metal -DGGML_METAL=ON
cmake -S parakeet -B build-cuda -DGGML_CUDA=ON
cmake -S parakeet -B build-vulkan -DGGML_VULKAN=ON
cmake -S parakeet -B build-opencl -DGGML_OPENCL=ON
```

## AMD XDNA NPU

Parakeet uses Fabric LLM's native GGML XDNA backend and the existing GGUF model.
It needs no ONNX export or ONNX Runtime session. Select it with
`EngineOptions::backend = "xdna"` or `--backend xdna`; `XDNA0` is accepted too.
`Engine::backend_device()` reports `BackendDevice::NPU` for this selection.
The encoder retains a scheduler with each cached graph so unsupported operations
can execute on CPU without invalidating that graph's copies between requests.
Other backends retain their direct encoder allocator.

The current backend targets XDNA2 devices, including Strix Point and Strix Halo,
through XRT. Its NPU matrix kernels accept eligible F16/BF16 and K-quantized
weights; Q8_0 matrices execute on its host path. Use the F16 model for NPU
validation. The backend also executes host operations inside its own chunks,
so selecting `XDNA0` does not mean every operation executes on the NPU.
Its `GGML_XDNA_GLUE_THREADS` and `GGML_XDNA_GLUE_THREADS_BIG` settings control
that host work independently of Parakeet's `--threads`.

Build the backend and matching IRON kernel artifacts as described in
[build.md](build.md#amd-xdna-dependency). Missing GEMM artifacts leave matrix
work on the host; check the loaded geometries and collect XRT submission
evidence when validating actual NPU execution. Compare CPU and XDNA using the
same model, audio and host thread settings. The backend's
`GGML_XDNA_PROF=1` diagnostics describe its host sections; backend identity
alone is insufficient evidence of NPU execution.

```bash
GGML_XDNA_KERNEL_DIR=/path/to/xdna-artifacts \
  build-parakeet/parakeet \
  --model models/parakeet-tdt-0.6b-v3.f16.gguf \
  --wav parakeet/test/samples/jfk.wav \
  --backend xdna --threads 4 --bench --bench-json xdna.json
```

`test-xdna-transcription` compares CPU and XDNA transcript text and token IDs,
repeated cached-graph use, streaming and a recreated Engine. It needs the F16
TDT fixture and XDNA0; hosts without that device skip the fixture test.

## Hexagon NPU

With `--backend hexagon` the FastConformer encoder and the TDT predictor and
joint graphs run on HTP0. Three things keep that numerically equal to the CPU:

- **Unfused attention.** `flash_attn_allowed` leaves Hexagon out. Its HMX fused
  attention holds the scores and the running output in F16 and drifts over
  long windows: on a 67.5 s window (844 encoder frames) the encoder output
  falls to cosine 0.76 against the CPU for Parakeet-TDT-0.6B and 0.81 for
  TranscribePsy. The unfused graph runs the same three batched matmuls (`ac`,
  `bd`, `attn·V`) on HMX in F16 with an F32 softmax: cosine 0.9996 and 0.99995
  on that window, and faster (0.94 s against 1.60 s for TDT-0.6B). HMX takes
  only 32-aligned reduction and row counts, so a head size that is not one
  (TranscribePsy's 80) is zero-padded to 96 on every attention operand
  (`attention_matmul_head_dim`); the padded columns add nothing to any score or
  output. Unpadded, those matmuls fall back to HVX and the window takes 6.2 s;
  padded, 0.91 s (the fused attention, on HVX for this head size, took 3.5 s).
- **Feed-forward headroom.** HMX writes matmul results through F16, so an
  element above 65504 reads back as inf. TranscribePsy's first block reaches
  37k to 76k in the second feed-forward linear on ordinary speech (69.5k on
  `hi-16k.wav`), while every other encoder matmul stays under 8.5k. On Hexagon
  that linear takes its input divided by 16 and multiplies the F32 result back
  (`matmul_output_headroom`, `affine_mul_mat_with_headroom`); other backends
  build the plain matmul.
- **Decoder weights in the repack buffer.** The TDT decoder graphs execute
  directly on the backend, without the scheduler's CPU fallback. The stacked
  `[w_ih | w_hh]` LSTM weight is therefore allocated in the same Hexagon repack
  buffer as the model weights and probed before use; in HTP's default buffer
  HTP refuses the quantized GEMV, and the graph decoded garbage.

TranscribePsy q8_0 on a Galaxy S25 (Snapdragon 8 Elite, HTP v79), six CPU
threads, default 60 s / 25 % windows, median of three runs after one warm-up;
transcripts identical to the CPU on every clip:

| Clip | CPU inference | Hexagon inference | Hexagon encoder | Hexagon decoder |
|---|--:|--:|--:|--:|
| `jfk.wav`, 11 s | 173 ms | 140 ms | 122 ms | 15 ms |
| 67.5 s (two windows) | 4.29 s | 1.71 s | 1.57 s | 122 ms |
| `abcdba.wav`, 191 s (five windows) | 14.8 s | 4.52 s | 4.13 s | 355 ms |

The device tests are `test-transcribepsy-hexagon`,
`test-encoder-backend-parity-transcribepsy`,
`test-attn-path-graph-transcribepsy-hexagon`, `test-tdt-backend-graphs-hexagon`
and `test-matmul-headroom-hexagon`; each exits with the skip code on a host
without HTP0 ([testing](testing.md#hexagon-npu-tests)).

## Experimental packed affine TDT recipe

`export-packed-affine.py` preserves an explicit per-output-channel INT4/INT8
recipe, including FP32 minima/steps and original FP16 overrides. It does not
apply another GGML weight quantizer. The nonstandard architecture
`parakeet-affine-v1` makes older engines reject these files instead of treating
carrier integers as model weights. Ordinary GGUF models retain their existing
load and execution paths.

Supply a JSON recipe with `source_archive_sha256` and a `tensors` object mapping
NeMo encoder weight names to `4`, `8`, or `16`; unlisted tensors remain FP16:

```bash
python parakeet/scripts/export-packed-affine.py \
  --archive models/parakeet-tdt-0.6b-v3.nemo \
  --recipe recipe.json \
  --output models/retained.affine.gguf \
  --dense-control models/retained.dense.gguf
```

The experimental native path stores integer codes in unit-scale Q4_0/Q8_0
carrier blocks, with separate FP32 affine metadata. These blocks are not
ordinary Q4_0/Q8_0 model weights. Block-scale headers add storage overhead over
pure packed codes. The native graph reconstructs each projection transiently
using separate FP32 multiply/add operations and rounds to FP16 before the
existing matrix multiply. Activation-first graph ordering permits workspace
reuse; the loader does not create a persistent dense model. This is a faithful
storage/execution baseline, **not a fused low-bit matrix multiply or a speed
optimization**. Reconstruction can be substantially slower than ordinary Q8.
Unsupported encoder graph operations fail rather than silently falling back.
`--profile` does not produce isolated substage timings for packed-affine models;
that graph builder has no affine reconstruction scope and rejects them explicitly.
Full encoder/inference measurements retain the normal affine execution path.
Only offline TDT has been evaluated; other backend/model combinations require
validation before use.

Depthwise kernels retain the recipe's reconstructed FP16 values, as in the
original packed-projection runtime. Selected FP16 attention weights remain
FP16. Both packed exports and their dense controls omit duplicate fused Q/K/V
copies, so size comparisons must account for that layout difference. The receipt
records the exported source values, physical bytes, coverage and file hashes.
`--expected-tensors` can require an independent SHA/shape/dtype receipt for every
source tensor before GGUF conversion.

Validate exact reconstruction before comparing transcription and timing:

```bash
python parakeet/test/test_packed_affine.py
# Build target test-packed-affine, then run on the exact requested backend:
./build/test-packed-affine models/retained.affine.gguf MTL0
```

The native audit compares every reconstructed FP16 weight bit-for-bit with the
saved affine arithmetic and refuses unsupported operations. Python reference
and Core ML export scripts deliberately reject carrier files; use the matching
`--dense-control` output when an expanded weight graph is required. A weight
identity check does not establish transcription accuracy or a speed advantage.

## Core ML encoder sidecar

`PARAKEET_COREML=ON` is Apple-only. It enables optional CTC/IndicConformer,
Unified RNN-T, TDT, EOU, Nemotron, and tagged Sortformer v2.1 FastConformer
encoder sidecars, and the MOSS-Transcribe-Diarize audio encoder sidecar, whose
export and routing are in [its guide](moss-transcribe.md#core-ml-encoder-sidecar).
Mel preprocessing, the CTC/RNN-T/EOU/Nemotron decoders, and the Sortformer
transformer/speaker head remain in the normal ggml pipeline; the TDT decoder
can additionally run on the Neural Engine through the
[batched decoder sidecar](#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).
Nemotron 3 Diarization and Sortformer v2.1 can instead run their whole network
on [Core ML diarizer sidecars](#core-ml-diarizer-sidecars).
Sortformer v1 and v2 have no sidecar. The English CTC checkpoints satisfy the
CTC/IndicConformer export contract; among CTC checkpoints, the Core ML parity
tests cover only IndicConformer.

Create an export environment with versions supported by Core ML Tools. NumPy 2
is not currently compatible with its TorchScript scalar conversion, and the
optional grouped-channel LUT pass uses scikit-learn:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r parakeet/scripts/requirements-coreml.txt
```

The compiled sidecar must sit next to the GGUF and use this name:

```text
<model-basename-with-quant-stripped>-encoder.mlmodelc
```

For example, both `parakeet-tdt-0.6b-v3.f16.gguf` and
`parakeet-tdt-0.6b-v3.q8_0.gguf` resolve to
`parakeet-tdt-0.6b-v3-encoder.mlmodelc`.
Sortformer v2.1 may also load
`diar_streaming_sortformer_4spk-v2.1-encoder.mlmodelc` for exact-shape batch
inference and
`diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlmodelc` for
masked AOSC inference.

Export and compile a fixed-shape sidecar:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/parakeet-tdt-0.6b-v3-encoder.mlpackage \
  --compile-dir parakeet/models
```

Fixed TDT exports now have two inputs: `mel` and `mel_valid_mask`. The latter
is a vector at mel capacity, with ones for the contiguous valid prefix and zeros
for padding. The runtime supplies this mask from the same valid-frame scan used
by the native encoder. The graph masks every subsampling stage, attention keys,
and intermediate Conformer states, including the GLU output before depthwise
convolution. Only the subsampled valid prefix reaches the decoder. This requires
both the updated exporter and runtime; an older runtime cannot load this new
interface. Existing single-input sidecars still load in the updated runtime,
but their padded frames remain visible to attention. Re-exporting is required
to obtain the masking fix; updating engine code alone cannot fix old artifacts.
Masking preserves natural-length semantics in the graph; quantization and Core
ML precision can still affect numerical and transcript parity. Validate the
new artifact on the intended audio before deploying it.

Checkpoint-free padding regressions can be run with:

```bash
python parakeet/test/test_coreml_padding.py
# Apple + Xcode command-line tools; also converts/compiles small test models:
python parakeet/test/test_coreml_padding_runtime.py
```

The runtime test covers fixed capacity, 1501-frame inputs, short and full-capacity
calls, integer lengths above 2048, stride boundaries, cache reuse, legacy single-input
models, and the Sortformer bypass interface on CPU-only and default Core ML
placement. FP32 conversions check padding semantics against the unpadded
PyTorch reference; FP16 and FP32 native I/O are compared against predictions
from the corresponding compiled Core ML graph. FP16-versus-PyTorch numerical
parity is a separate artifact validation, not asserted by the I/O test.
A tested conversion environment is Python 3.11, Torch 2.7.0,
coremltools 9.0, and NumPy 1.26.4. These small-model checks do not establish
full-model WER, latency, or ANE placement.

IndicConformer uses the same fixed-capacity contract. Core ML runs the encoder;
the multilingual CTC projection and `--language` token mask remain on ggml:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/indic-conformer-600m-multilingual.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/indic-conformer-600m-multilingual-encoder.mlpackage \
  --compile-dir parakeet/models
```

Nemotron supports an exact-shape offline encoder. Export it using `--wav` or
`--n-mel-frames`. Longer offline inputs and native cache-aware streaming run on
the cache-aware ggml path.

Unified RNN-T uses the same fixed-capacity contract:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/parakeet-unified-en-0.6b.q8_0.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/parakeet-unified-en-0.6b-encoder.mlpackage \
  --compile-dir parakeet/models
```

For EOU, first download and convert the checkpoint, verify the F16 round trip,
then compile the exact-shape sidecar from that GGUF:

```bash
parakeet/scripts/download-all-models.sh eou

python parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --quant f16

python parakeet/scripts/verify-gguf-roundtrip.py \
  --nemo parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --gguf parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf

python parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
  --quant q8_0

python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --wav parakeet/test/samples/jfk.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/parakeet_realtime_eou_120m-v1-encoder.mlpackage \
  --compile-dir parakeet/models
```
For Sortformer v2.1, export the batch and AOSC block-stack sidecars separately:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --wav parakeet/test/samples/diarization-sample-16k.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder.mlpackage \
  --compile-dir parakeet/models

python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --bypass-pre-encode --n-encoder-frames 410 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlpackage \
  --compile-dir parakeet/models
```


Benchmark fixed lengths and inspect ANE/GPU/CPU placement:

```bash
python parakeet/scripts/bench-encoder-coreml.py \
  --gguf parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16

python parakeet/scripts/bench-encoder-coreml.py \
  --gguf parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --mel-frames 1101 \
  --palettize-bits 6 --palettize-group-size 16
```

The default export uses Float16 input, output, weights, and intermediates.
CTC/IndicConformer, Unified RNN-T, and TDT shorter inputs are zero-padded to the
exported mel-frame capacity. Longer CTC and Unified RNN-T offline inputs use
overlapping windows. Standard TDT sidecars fall back to ggml beyond capacity:
forcing their 15-second windows can truncate or corrupt the transcript. TDT
Core ML windowing requires an export with a validated seam-context policy,
such as the masked Neural Engine export described below. EOU uses exact-shape
routing because padding future frames can change token and end-of-turn
decisions: only calls at the compiled mel-frame count use Core ML. Shorter
inputs, longer offline inputs, and mismatching streaming windows use ggml: EOU
never pads, and windowed causal/chunked attention parity has not been validated
for oversized inputs. The TDT example uses this addon's 15-second shape (1501
mel frames; its centred-STFT frontend emits `1 + samples/hop`) and optional
6-bit grouped-channel LUT weights. Grouped palettization requires coremltools 8+
and macOS 15 / iOS 18; omit both `--palettize-*` arguments for a macOS 13 / iOS
16 compatible Float16 model. `--flexible` exports a CTC/IndicConformer, Unified
RNN-T, or TDT RangeDim model, but it is a correctness/experimentation path:
measured flexible graphs place no operations on ANE and can be substantially
slower than ggml Metal. Flexible EOU export is rejected.

Nemotron routing requires the exact exported mel-frame count because its
causal, chunk-limited attention geometry is baked into the compiled program.
Longer offline inputs use the native cache-aware ggml encoder instead of
windowing the sidecar, because an independent exact-shape window cannot
preserve the encoder's full-depth receptive field. Shorter inputs are never
padded, and native cache-aware streaming stays on ggml. Flexible Nemotron
exports are rejected.

Sortformer batch routing requires the exact exported mel-frame count because
its graph has no validity-mask input; shorter and mismatched batch inputs stay
on ggml. The AOSC bypass sidecar masks padded keys and accepts encoder slabs up
to its exported capacity (410 frames for the default cache/FIFO/chunk geometry);
larger or custom geometries stay on ggml. Flexible Sortformer exports are
rejected. A Sortformer v2.1 diarizer sidecar takes precedence over both for
every call it covers. On an Apple M4, the bypass sidecar alone takes 53.6 ms
per 410-frame slab, about as long as the whole ggml Metal AOSC step, while the
diarizer sidecar takes 22.3 ms for the encoder and speaker head together.

At runtime the sidecar lets Core ML use all compute units and reuses its input,
feature-provider, and fixed-shape output-backing objects across predictions. The
graph is predominantly Neural Engine-backed, but a small number of operations
may prefer the GPU; forcing CPU + Neural Engine can move those operations onto
the CPU and reduce the speedup. Set `PARAKEET_COREML_COMPUTE_UNITS=cpu_and_gpu`,
`cpu_only`, or `cpu_and_ane` to override the default for placement comparisons.
The [Neural Engine exports](#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars)
are the exception: they are fully ANE-shaped and load pinned to CPU + Neural
Engine, since `all` is only a preference that may place them on the GPU;
`PARAKEET_COREML_COMPUTE_UNITS=all` restores Core ML's own choice.

A missing sidecar, load failure, incompatible shape, or runtime prediction
failure falls back to the ggml encoder. An incompatible shape (an input longer
than a fixed sidecar's capacity or window) affects only that call: the sidecar
stays loaded for the next one. A failed prediction (encoder or AOSC bypass)
retires that sidecar for the rest of the process, so later calls go directly
to ggml. Set `PARAKEET_COREML_DISABLE=1` to
force ggml, including for parity or benchmarking. Setting
`EngineOptions::long_form_window_frames` below zero disables automatic
windowing; an input larger than a fixed Core ML sidecar then falls back to the
single-pass ggml encoder.

For an unambiguous CTC/IndicConformer, Unified RNN-T, TDT, EOU, or Nemotron
benchmark, configure the exact build directory with Core ML enabled, compile
the sidecar beside the GGUF, and require Core ML. `--require-coreml` rejects
Sortformer GGUFs; their Core ML parity is covered by the diarization tests
instead:

```bash
cmake -S parakeet -B build-parakeet-coreml \
  -DCMAKE_BUILD_TYPE=Release \
  -DPARAKEET_COREML=ON \
  -DGGML_METAL=ON
cmake --build build-parakeet-coreml --target parakeet-cli -j

./build-parakeet-coreml/parakeet \
  --model parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav parakeet/test/samples/jfk.wav \
  --n-gpu-layers 999 \
  --bench --bench-warmup 2 --bench-runs 5 \
  --bench-json /tmp/parakeet-tdt-coreml.json \
  --require-coreml --verbose
```

The JSON may still contain `"backend": "ggml-metal"` because the decoder
continues to use Metal. Confirm encoder execution using `encoder_backend` and
`encoder_coreml_all_runs`. A `coreml-all` encoder label means Core ML may place
operations across ANE, GPU, and CPU; it does not mean ANE-only execution.

The desktop macOS benchmark generates this sidecar from the downloaded F16 GGUF
on its first run and stores the compiled bundle in the GitHub Actions cache.
Later runs restore that cache and go directly to the benchmark comparison.

Windowing bounds Core ML input shapes and memory, but full-context attention is
then local to each overlapping window. The stitched result should therefore be
treated as close to, rather than bit-identical with, a single full-length encode;
validate accuracy on representative long recordings before production use.

## Neural Engine fixed-window encoder and TDT decoder sidecars

Full TranscribePsy EU additionally needs separate English/adapted encoders and a
gate-tap encoder, with validated route metadata. Its offline, callback and
push-PCM paths support Core ML on macOS/iOS; [the full-EU guide](transcribepsy-eu-coreml.md)
contains export commands, streaming gate semantics and measured decoder dispatch policy.

TDT models can run on the Apple Neural Engine end to end with two sidecars that
TranscribePsy publishes ready-made
([qvac/TranscribePsy-250M-EU-sub-coreml](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml);
`scripts/download-all-models.sh transcribepsy` fetches them) and that
`export-encoder-coreml-ane.py` / `export-decoder-coreml-ane.py` export from any
offline TDT GGUF. Both sit next to the GGUF under the usual derived names:

```text
<model>.gguf
<model>-encoder.mlmodelc
<model>-decoder.mlmodelc
```

The **fixed-window encoder** is a channel-first `[1, C, 1, T]` graph over a
fixed window (`--window-mel-frames`, 1500 mel frames = 15 s by default) with
runtime masks: `mel_mask` zeroes the padded mel frames and, through its
even-index slices, the padded frames at every subsampling resolution;
`pad_mask` zeroes the padded encoder frames and `key_bias` masks them out of
attention, so a partially filled window uses the same graph and its valid
frames come out as the reference encoder's for the short input alone (the
exporter's `--check` verifies both a full and a partial window). The wrapper
derives all three masks from the frames that hold audio, so the mel
front-end's zeroed trailing padding frame counts as padding as well. The published
TranscribePsy export predates `mel_mask` and masks after subsampling only; the
wrapper fills `mel_mask` when an export declares it. The wrapper
recognises the inputs (`mel`, `pad_mask`, `key_bias`, optionally `mel_mask`), reports the window
as the sidecar's fixed capacity, and the engine's long-form plan tiles longer
audio across overlapping windows with 16 encoder frames (1.28 s) of seam
context each side, the setting the export was validated with
(`PARAKEET_COREML_ANE_CONTEXT_FRAMES` overrides it; an explicit
`EngineOptions::long_form_context_frames` wins). Weights are 6-bit palettized
(`--nbits 0` keeps fp16). Unlike the single-input export, this layout is fully
Neural Engine-resident on the published TranscribePsy model. The short window
and seam context cost accuracy on long recordings without pauses; the
[measurements](performance.md#transcribepsy-on-the-apple-neural-engine-mac-mini-m4-2026-10)
quantify it against the CPU windows.

```bash
python parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --out  parakeet/models/transcribepsy-250m-eu-sub-encoder.mlpackage \
  --compile-dir parakeet/models --check
```

The **TDT decoder sidecar** runs the prediction LSTM and the joint for 16
independent decoder states (`--lanes`) per dispatch; the prediction network is
memory-bandwidth bound, so one weight read serves every lane. The TDT walk and
the token-embedding lookup stay on the host (a gather is not a Neural Engine
op); the lookup converts one row at a time from the GGUF's own embedding table,
so the engine holds no dequantised copy of it, and the sequential ggml decoder
is built only when a call needs it (see below). The published model keeps the
joint output projection at 6 bits too (`--nbits-out 6`); the exporter's
default leaves it fp16. The exporter chooses an output-channel group size that
divides the projection width: 2 for TranscribePsy's 8198 output channels. It
verifies that the requested output projection was actually palettized before
reporting success; `--nbits 0 --nbits-out 6` compresses only that projection.

```bash
python parakeet/scripts/export-decoder-coreml-ane.py \
  --gguf parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --out  parakeet/models/transcribepsy-250m-eu-sub-decoder.mlpackage \
  --compile-dir parakeet/models --nbits-out 6 --check
```

To create the encoder-less EU-sub GGUF accompanying these sidecars, use
[`strip-encoder-gguf.py`](models.md#transcribepsy) after exporting from the full
GGUF. Match the compiled sidecar basenames to the stripped GGUF's basename.

With the decoder sidecar loaded, `Engine::transcribe*` on a TDT model takes the
**independent-segment path**: the mel is cut into segments no longer than the
encoder window, each cut at the quietest mel frame within the last 6 s before
the cap so the state reset falls in a pause rather than mid-word; every segment
is encoded in one dispatch and decoded from a fresh prediction state, 16
segments per decoder dispatch; a segment that emits nothing is re-decoded
warm-started from the previous segment's state; the texts are joined. The
output can differ from the sequential decoder's. `EngineResult::segment_plan_ms`
and `assemble_ms` report the planning and stitching time. The CLI's one-shot
path keeps the sequential decoder over the windowed encoder output.
Overrides: `PARAKEET_ANE_BATCH_DECODE=0` keeps the sequential path,
`PARAKEET_ANE_SEGMENT_MS` and `PARAKEET_ANE_PAUSE_SEARCH_MS` change the cap and
the pause search, and `PARAKEET_ANE_DECODE_HOST=1` decodes the same segments on
the host for parity checks. The lane walk's geometry is cross-checked against
the GGUF at load; a mismatched export is refused with a log line and the
sequential decoder runs. The sequential decoder's graphs and state are
allocated on its first call (streaming, these overrides, or a fallback), not at
load, and its encoder projection holds the model's long-form window (750
frames for TranscribePsy) rather than a fixed 4096; longer inputs are decoded
in window-sized pieces that share one prediction state, with the same tokens.

A **GGUF without encoder weights**, such as the 17 MB TranscribePsy Core ML
GGUF, loads only when the encoder sidecar can take the encoder over: it is then
loaded before weight allocation and the `encoder.*` tensors get no memory.
Without the sidecar, or on a build without `PARAKEET_COREML=ON`, loading fails
with a message saying so, and every path that would reach the ggml encoder
(`--profile`, the memory-fit projection) fails instead of running. The same
sidecars work next to a full GGUF renamed to match them; set
`PARAKEET_COREML_ONLY=1` so the ggml copy of the encoder weights, which the
Neural Engine path never reads, is not loaded.

Standard TDT sidecars without a validated seam policy retain the full GGUF's
native encoder weights even with `PARAKEET_COREML_ONLY=1`: oversized offline
inputs and push streams still need the native encoder. An encoder-less GGUF
paired with such a sidecar is rejected at load time with
`kGgufLoadNoEncoderWeights`. Validated Neural Engine exports and full
TranscribePsy EU keep their existing ownership behavior.

## Core ML family routing

| Family | Contract | Inputs the sidecar does not take |
|---|---|---|
| CTC (English, IndicConformer), Unified RNN-T | fixed capacity: shorter inputs are zero-padded to the compiled shape, longer offline inputs are split into overlapping windows | Unified RNN-T cache-aware streaming runs on ggml |
| TDT 0.6B-v3 / 1.1B | fixed capacity: shorter inputs are zero-padded; oversized inputs use ggml to preserve encoder context | standard TDT push streams require at least 30000 ms chunks; small PCM writes remain supported |
| Full TranscribePsy EU | fixed ANE English/adapted/gate programs for offline encoding; native-context FP32 CPU/GPU programs for streaming and true unwindowed calls | all public transcription modes execute Core ML with the matching sidecar family; iOS precise programs use Core ML CPU; optional decoder batches offline calls only |
| TDT with a fixed-window Neural Engine encoder (TranscribePsy EU-sub) | fixed window with padding masks: shorter inputs are padded and masked, longer offline inputs are split into overlapping windows with 16 frames of seam context, or into pause-cut independent segments when the TDT decoder sidecar is present | streaming runs on ggml; the decoder sidecar serves only the engine's offline transcribe path |
| EOU | exact compiled shape only | every other length, including mismatched streaming windows, runs on ggml; EOU never pads |
| Nemotron | exact compiled shape only | longer offline inputs and streaming take the cache-aware ggml path |
| Sortformer v2.1 (`sortformer-streaming-v2.1-aosc`) | masked whole-network diarizer sidecars for batch inputs and AOSC slabs up to their capacity; otherwise the exact-shape batch encoder sidecar plus a masked AOSC encoder sidecar | larger batch inputs and AOSC slabs, and every call without a sidecar, run on ggml |
| Nemotron 3 Diarization | masked whole-network diarizer sidecars for offline inputs, long-form blocks, and live chunks up to their capacity | larger calls, and every call without a sidecar, run on ggml |
| Sortformer v1, v2 | no sidecar | always ggml |
| MOSS-Transcribe-Diarize | Whisper encoder plus adaptor per 30 s window, the same zero-padded window the ggml encoder runs; separate `MOSS_COREML_*` switches | none; the Qwen3 decoder always runs on ggml. See [MOSS-Transcribe-Diarize](moss-transcribe.md#core-ml-encoder-sidecar) |

## Core ML diarizer sidecars

Nemotron 3 Diarization and Sortformer v2.1 can run their whole network on the
Apple Neural Engine. A diarizer sidecar holds everything after Nemotron's mel
stacking or Sortformer's ggml subsampling: the encoder, the speaker head, and
the sigmoid. Mel preprocessing, Sortformer subsampling, the speaker-cache
(AOSC) updates, and segment thresholding stay on the host, so the cache
semantics are unchanged.

Each sidecar has one fixed capacity in 80 ms encoder frames, counting speaker
cache, FIFO, and chunk rows together, and sits beside the GGUF as:

```text
<model-basename-with-quant-stripped>-diarizer-<frames>.mlmodelc
```

For example, `Nemotron-3-Diarization.q8_0.gguf` uses
`Nemotron-3-Diarization-diarizer-358.mlmodelc`. Several capacities can sit side
by side. Each call uses the smallest capacity that fits it; the unused rows are
zero-padded and masked out of attention and the speaker-head convolution, which
leaves the valid rows unchanged. Calls above the largest capacity run on ggml.
A capacity loads on first use. If it fails to load or predict, it is
quarantined for the lifetime of the engine and that call falls back to ggml.
`DiarizationResult::encoder_used_coreml` reports whether every network pass of
a call ran on a sidecar; `Engine::encoder_on_coreml()` only reports that one
was found.
`PARAKEET_COREML_DISABLE=1` and `PARAKEET_COREML_COMPUTE_UNITS` apply as for
the encoder sidecars. Like the Neural Engine encoder and TDT decoder exports,
the diarizers load with CPU and Neural Engine compute units unless
`PARAKEET_COREML_COMPUTE_UNITS` names others. `MLComputeUnitsAll` is only a
preference, and on an M3 Ultra under macOS 26.7 Core ML resolves it to the GPU,
where the program runs 1.4-1.8x slower than ggml Metal.
`PARAKEET_COREML_DIARIZER_DISABLE=1` ignores only the diarizer sidecars, so the
older Sortformer encoder sidecars still load.

| Model | Capacity | Covers |
|---|---:|---|
| Nemotron 3 Diarization | 358 | the default live chunk (264 cache + 80 FIFO + 14 chunk rows) and offline inputs up to 28.6 s |
| Nemotron 3 Diarization | 745 | the 30 s long-form block (344 + 401 rows) and offline inputs up to 59.6 s |
| Nemotron 3 Diarization | 1125 | offline inputs up to the 90 s long-form threshold |
| Sortformer v2.1 | 410 | the default AOSC step (188 cache + 188 FIFO + 33 chunk and context rows, 409 in all) and batch inputs up to 32.8 s |

Export from the GGUF in the Core ML environment above. The exporters trace a
Float16 program with Float32 inputs and outputs, and `--compile-dir` compiles
it with Core ML Tools, so a host with only the Command Line Tools works:

```bash
python parakeet/scripts/export-nemotron-diarization-coreml.py \
  --gguf parakeet/models/Nemotron-3-Diarization.q8_0.gguf \
  --frames 358 745 \
  --compile-dir parakeet/models

python parakeet/scripts/export-sortformer-diarizer-coreml.py \
  --gguf parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --frames 410 \
  --compile-dir parakeet/models
```

### Generated diarizers

The library can also make the Nemotron 3 Diarization and Sortformer v2.1
bundles itself, from the GGUF alone. macOS builds with `PARAKEET_COREML=ON`
compile the Core ML generator (`PARAKEET_COREML_GENERATE`, default `ON` on
macOS and `OFF` on iOS) and embed one template per capacity under
`parakeet/coreml-templates/`: `nemotron-diar-f{358,745,1125}/` and
`sortformer-diar-f410/`. A template is the exporter's weight-free Core ML
program plus a recipe that says how every weight of `weight.bin` is computed
from the GGUF's tensors. For Nemotron 3 Diarization:

- most weights are the GGUF tensor cast to fp16, after ggml dequantises it;
- each layer's `w_qkv` has its query rows scaled by `1/sqrt(head_width)`
  (`slice`, `mul`, `concat`), as the exporter does;
- the rotate-half matrix and the RoPE cos/sin tables of the capacity do not
  depend on the weights and are stored in the template (`literals.bin`);
- the compiler folded two constants of the layer-norm lowering (`zeros`,
  `ones`) and the negated speaker-head weight (`neg`);
- the negated speaker-head bias is an inline constant of the program, which a
  `patch` writes into the spec.

For Sortformer v2.1 (`sortformer-diar-f410`, the AOSC streaming variant):

- most weights are the GGUF tensor cast to fp16, after ggml dequantises it;
- scales are folded into weights (`mul`): the conformer's query scale
  `1/sqrt(64)` into each layer's query projection, its bias and the two
  position biases; the speaker transformer's `1/sqrt(24)` (as float32) into
  its query projection and bias; the feed-forward half-step `0.5` into each
  second feed-forward projection and bias;
- each conformer layer's relative-position projection is split into one 1x1
  convolution per head, with its input columns in phase order, even then odd
  (`slice` with step 2, `concat`, `permute`), as the exporter's
  angle-difference rewrite needs;
- the batch-norm scale and shift are folded into each depthwise convolution's
  weight and bias (`mul`, `add`);
- the query-side sine and cosine tables and the key-side table of the
  capacity do not depend on the weights and are stored in the template
  (`literals.bin`);
- the compiler folded the layer-norm constants (`zeros`, `ones`) and the
  negated speaker-head weight (`neg`); the negated speaker-head bias is an
  inline constant of the program, written by a `patch`.

When a capacity has no adjacent bundle, the engine plans route `generate` for
it if the GGUF matches the template (Nemotron 3 Diarization: architecture
`sortformer`, version `v3`; Sortformer v2.1: architecture `parakeet-ctc`,
model type `sortformer`, variant `sortformer-streaming-v2.1-aosc`; then the
template's hyperparameters, and every tensor's name and shape, with matrices
in f32, f16, bf16, q8_0, q6_k, q5_k or q4_k, for Sortformer v2.1 also q4_0),
a cache directory is
set (`CoremlOptions::cache_dir`, `QVAC_COREML_CACHE_DIR`, or the CLI's
default), `generate` is not `never`, the OS is macOS 15 or later, and the
resolved compute units are ones the template's memory envelope was validated
for (`cpu_and_ane`). After the model has loaded, the runtime's prepare queue
digests the source tensors, writes the package, compiles it with
`MLModel.compileModelAtURL`, publishes it under
`<cache_dir>/qvac-coreml-v1/artifacts/` and hands it to the load lane; the
capacity then loads like an adjacent bundle. A later load of the same GGUF
adopts the published bundle without generating it again (status route
`cached`). Until a capacity is ready its passes run on ggml and the result
reports `coreml_pending`.

The generated `weight.bin` is byte-identical to what the exporter writes for
the same GGUF dequantised to f32 (`qvac-coreml-tool f32`), and the spec is the
exporter's. CI checks this on random twins of the model (extraction and
cross-seed gates, `coreml/python/coreml_template.py report`), on the
real q8_0 GGUF (`coreml_template.py real`), and runs
`test-nemotron-diarization-coreml-generated-parity` on diarizers generated
from that GGUF.

| Capacity | Generated `weight.bin` | Compile (M2, macOS 27.2.1) |
|---:|---:|---:|
| 358 | 199.3 MB | 0.34 s |
| 745 | 200.1 MB | 0.33 s |
| 1125 | 200.8 MB | 0.34 s |

`MLModel.compileModelAtURL` took 0.33-0.35 s per capacity over two runs on
macOS 27.2.1 (0.69-0.93 s on macOS 15.7.3). The cold Neural Engine load that
follows took 11.4-14.2 s per capacity on macOS 27.2.1.

`parakeet-fit-params` charges each generated capacity a call can load (every
generated capacity under `prewarm = all`, which loads them all at
construction) at its template's Core ML envelope, in the Core ML-inclusive
verdict only: the resident part of each capacity plus one generation or load
transient. The calls an instance can make include shorter ones than the
projected workload: any input up to the single-pass graph (the long-form
window past the long-form threshold, 1125 frames by default) runs as one pass
and keeps its capacity loaded, so the default 300 s projection charges all
three capacities. A capacity that fails to load hands its calls to the next
larger one that has not failed, so fit takes every capacity of the model,
prebuilt and generated, in one ordered list, selects the ones the calls and
the construction prewarm load, lets each hold any serving capacity from
itself up to the next selected one, and charges the worst resident set that
can result (prebuilt bundles at today's estimate, generated capacities at
their envelope) plus its largest transient. Two capacities that replace each
other are never charged together. The templates' constants are measured values that already carry a
1.25 margin, so the envelope adds none (`memory.margin` is 1.0). The fit and
the engine choose the generated capacities with the same code, so they cannot
disagree. The route comes from the options and the GGUF, never from the cache,
so the projection is the same before and after the first generation.
`coreml.validated` is true only when every charged byte is a generated
envelope; an adjacent bundle's charge is today's unvalidated estimate.

The templates' memory envelope is validated for `cpu_and_ane` on two hosts
(`memory.validated_for`): an M2 with macOS 27.2.1 and an M4 Max with macOS
26.7.2 (the CI runner). The validation runs diarizer passes in a loop, on
ggml until a capacity is ready and on Core ML after, while the runtime
prepares (generates and compiles), loads and warms all three capacities in the
background (`prewarm = all`, `wait = none`).
`diarizer-coreml-memory` (built with the Core ML tests) samples
the process's `phys_footprint` every 2 ms and `ANECompilerService` every
100 ms, from fresh process identities and paths so every Neural Engine load is
cold. Per capacity it reports the prepare (from the start of its generation to
the end of its compile), the load and warm increment, and the resident step:
the process's settled footprint (the median sample) from the moment the
capacity is ready to the moment the next one is, less the same level for the
capacity before it, or less the peak of a ggml-only run of the same passes for
the first. The steps add up to the run's steady growth over ggml. It also
reports `ANECompilerService`'s growth during the loads: its peak less its size
when the first load started. `scripts/check-coreml-memory-schedule.py` checks
the reports against the templates: the peak increment against what fit
charges; each prepare, load, resident step and the compiler's growth, times
1.25 and rounded up to 1 MiB, against the template's constant (the rule the
constants were set by, so a constant that lost its margin fails); and the
steady growth against the three capacities' resident parts. The compiler's
growth is held to the reserve over the whole run and per load (from the
service's size when that load started). A run fails when a capacity shows no
generate, compile or load phase, a compile has no generate before it, a phase
has no footprint sample, a load has no `ANECompilerService` sample, the tool
reports sampler errors (a footprint or `ps` read that failed), a capacity does
not end the run ready or serves no Core ML pass after every capacity is ready
(counted by the capacity that served it), the compiler grew less than
64 MiB (a warm run), or the schedule was not overlapped: the report records
every pass's interval, and some ggml pass must have run during a prepare
(generate or compile) and some during a load or warm (one pass spanning both
counts for both). The tool itself exits 1 when a capacity never became ready,
served too few passes, or no ggml pass overlapped a prepare or a load. The Core ML lane of `parakeet-ci.yml` runs the generated schedule and
the check on its macOS runner. Increments are over a ggml-only run of the same
passes on the same host:

| Host | Runs | Peak increment | Largest prepare | Largest load and warm | Largest resident step | Steady growth | `ANECompilerService` growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2, macOS 27.2.1 | 18 | 64.7-138.0 MiB | 64.8-86.1 MiB | 12.7-96.7 MiB | 6.6-108.8 MiB | 30.6-138.0 MiB | 280.0-557.1 MiB |
| M4 Max, macOS 26.7.2 | 18 | 63.0-116.2 MiB | 63.6-88.3 MiB | 14.9-26.7 MiB | 5.9-33.1 MiB | 32.4-65.5 MiB | 279.9-628.2 MiB |

The ranges are over runs, each run's largest value. Steady growth comes from
the runs that recorded the settled footprint (14 on the M2, 10 on the M4 Max).
The earlier runs recorded the footprint when each load started, which gives
the resident steps of the first two capacities, and the compiler's growth is
taken from its smallest size during their prepares, which all come before the
first load. With the adjacent bundles loading in the background instead (2
runs on the M2), the peak increment was 88.0-91.6 MiB against today's
921.7 MiB charge. Each constant is 1.25 times the largest value over every run
on both hosts, rounded up to 1 MiB (`memory.measured` in `template.json`):

| Constant | Largest measured | Recorded |
|---|---:|---:|
| prepare: `generation_peak_bytes` (9 MiB) + 8 MiB + `compile_peak_bytes` | 88.3 MiB (M4 Max) | 111 MiB |
| `load_peak_bytes.cpu_and_ane` | 96.7 MiB (M2) | 121 MiB |
| resident part per capacity: `runtime_weight_bytes` + 2 x I/O (`workspace_bytes` 0) | 108.8 MiB (M2) | 137 MiB |
| `ane_compile_reserve_bytes` | 628.2 MiB (M4 Max) | 786 MiB |

The first prepare and load of a process pay Core ML's start-up whichever
capacity they are for, so every template records the same values. On
`cpu_and_ane` the resident part holds no copy of the fp16 weights: the Neural
Engine keeps them outside the process, so a loaded capacity costs the process
tens of MiB, not its 191 MiB `weight.bin`. On the M4 Max a load's increment
is released when the capacity is ready. On the M2, in 7 of 18 runs one load
grew the footprint by 59-97 MiB a few seconds in, and most of that stayed.
Because the next load starts as soon as a capacity is ready, a step also takes
in the start of the next load, so the largest step (108.8 MiB) is an upper
bound for one capacity; the steady growth of all three stayed within
138.0 MiB. For `prewarm = all`, fit charges 3 x 137 + 121 = 532 MiB, against
a largest measured peak increment of 138.0 MiB. The compiler is checked on its
growth because a shared runner's `ANECompilerService` holds whatever earlier jobs
left it (11-514 MiB before the loads on the M4 Max); its largest growth
(628.2 MiB) came on the M4 Max from a service holding 10.7 MiB, a clean start. The reserve only gates cold loads at run time (they wait for that much
available memory); it is not part of the fit charge.

On the pinned diarization set (`abcba`, `abcdba`, 250 ms collar) the
generated diarizers produce the same segments as the exporter's bundles, so
generation changes the DER by 0. Both match ggml on `abcdba` (DER 3.60 %). On
`abcba` the Neural Engine (`cpu_and_ane`, also `all`) gives 7.31 % against
ggml's 2.97 %, all of it speaker confusion in the long-form chunks around
90 s, with the generated and the exporter's bundles alike. The same bundles on
`cpu_only` (2.99 %) and `cpu_and_gpu` (2.97 %) match ggml, so the difference
comes from the Neural Engine's arithmetic, not from generation.

The Sortformer v2.1 template (`sortformer-diar-f410`) is validated the same
way, on the same two hosts. Its one capacity is in the construction set, so
`diarizer-coreml-memory` runs diarizer passes over the sample (342 encoder
frames each) on ggml while the runtime generates, compiles, loads and warms
the capacity in the background, and on Core ML after. Increments over a
ggml-only run of the same passes:

| Host | Runs | Peak increment | Prepare | Load and warm | Resident step | `ANECompilerService` growth |
|---|---:|---:|---:|---:|---:|---:|
| M2, macOS 27.2.1 | 22 | 62.5-108.9 MiB | 72.8-82.5 MiB | 36.5-90.3 MiB | 0.0-108.9 MiB | 252.1-674.2 MiB |
| M4 Max, macOS 26.7.2 | 12 | 55.8-73.3 MiB | 80.9-97.3 MiB | 35.8-51.2 MiB | 39.2-61.0 MiB | 120.0-460.4 MiB |

Each constant is 1.25 times the largest value over the 34 runs, rounded up
to 1 MiB:

| Constant | Largest measured | Recorded |
|---|---:|---:|
| prepare: `generation_peak_bytes` (8 MiB) + 8 MiB + `compile_peak_bytes` | 97.3 MiB (M4 Max) | 122 MiB |
| `load_peak_bytes.cpu_and_ane` | 90.3 MiB (M2) | 113 MiB |
| resident part: `runtime_weight_bytes` + 2 x I/O (`workspace_bytes` 0) | 108.9 MiB (M2) | 137 MiB |
| `ane_compile_reserve_bytes` | 674.2 MiB (M2) | 843 MiB |

The two [2026-11-25 CI runs](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/45614667211)
raised the largest preparation increment to 101,990,880 bytes. Its 1.25 margin
requires 122 MiB after rounding; the checker and its safety margin are unchanged.

Fit charges the capacity 137 + 122 = 259 MiB, against a largest measured
peak increment of 108.9 MiB. As with Nemotron, the M2 sometimes keeps most of
a load resident: in 3 of 22 runs the load grew the footprint by 83-90 MiB and
the step to the settled footprint was 104-109 MiB; the M4 Max never did. The
schedule checker accepts only runs from the current tool, which report no
sampler errors, a compile segment, an `ANECompilerService` sample during the
load and the passes that overlapped the background work. Runs recorded before
those fields existed count only towards the maxima.
`diarizer-coreml-memory` labels the compiler's size before the load as a
clean (at most 32 MiB), dirty or unknown baseline, because a service that
already holds memory grows less than a clean one would. Every run here
started from a dirty baseline (61-632 MiB held), and the largest growth came
from the smallest baseline (61.5 MiB, on the M2), so a clean host may grow
somewhat more.

On the pinned diarization set (AOSC streaming, 250 ms collar) the generated
diarizer produces the same segments as the exporter's bundle, so generation
changes the DER by 0: `abcba` 23.76 % (ggml 23.74 %) and `abcdba` 23.38 %
(ggml 23.44 %).

Each capacity is a separate bundle of Float16 weights, about 190 MB for
Nemotron 3 Diarization and 220 MB for Sortformer v2.1. A multifunction package
would share the weights, but on macOS 15.3 Core ML refuses to select a function
on the Neural Engine. 8-bit weights halve the size but cost accuracy:
per-channel quantization raised the Nemotron maximum error against NeMo to
0.116, and block quantization moved the program to the GPU.

The graphs are written for the Neural Engine. Activations are laid out as
`(1, channels, 1, frames)`, linear layers are 1x1 convolutions, and attention is
split per head, which measured 1.4-1.8x faster than batched attention at 358 to
1125 frames. The RoPE rotate-half is a grouped convolution. Sigmoids are written as
`1 / (1 + exp(-x))`, because the Neural Engine's native sigmoid is coarse
enough to triple the Sortformer error. Sortformer's relative-position scores
use the angle-difference identity of its sinusoidal table, with the constant
key-side table applied as a 1x1 convolution. A relative shift either compiled
for more than 15 minutes or ran on the CPU at 0.8 s per call, and an
activation-by-activation contraction over all 512 table channels lost too much
precision on the Neural Engine.

On an Apple M4 (10-core GPU, macOS 15.3.2), every operation of both programs
runs on the Neural Engine. Single calls at full capacity, against the ggml Metal
graph at the same size:

| Model and capacity | Neural Engine | ggml Metal |
|---|---:|---:|
| Nemotron 3 Diarization, 358 | 12.0 ms | 35.6 ms |
| Nemotron 3 Diarization, 745 | 31.9 ms | 76.5 ms |
| Nemotron 3 Diarization, 1125 | 81.2 ms | 129 ms |
| Sortformer v2.1, 410 | 22.3 ms | not measured alone |

End-to-end timings are in
[performance.md](performance.md#nemotron-3-diarization-and-sortformer-v21-on-the-apple-neural-engine).
The first load compiles the program for the Neural Engine: about 12 s per
Nemotron capacity and 22 s for Sortformer on the M4. macOS caches the result for
each application binary, after which a load takes about 2.7 s and 6 s. The cache
can evict programs, so a later process may compile again.
`EngineOptions::prewarm` moves the Nemotron cost into engine construction. If a
bundle is replaced in place and Core ML then aborts with an MPSGraph
`cannot open input file` assertion, remove
`~/Library/Caches/<binary>/com.apple.e5rt.e5bundlecache`.

`fit_params` charges the bundles a workload can load as
`FitResult::coreml_bytes` (`coremlBytes` in `parakeet-fit-params --json`):

- Nemotron: the bundle for the offline input, or every bundle up to the 30 s
  long-form block, plus the bundle for one default live chunk;
- Sortformer: the bundle for the offline input.

Core ML allocates this memory outside the process, and the ggml weights stay
loaded for the fallback. On the M4, one loaded bundle raised system memory use
(wired, anonymous and compressed pages) by 441-545 MiB, and the 358 and 745
pair together by 799 MiB. The resident footprint of the process grew by only
16-44 MiB. The projection therefore charges twice each weights file, plus
160 MiB once. `PARAKEET_COREML_DISABLE` and `PARAKEET_COREML_DIARIZER_DISABLE`
remove the charge.

The Neural Engine costs about the same on an M3 Ultra (60-core GPU, macOS 26.7.1)
as on the M4, but the Ultra's Metal GPU is two to three times faster. There the
sidecars only tie with ggml Metal: they win Nemotron offline at 27.3 s and live
at 273 s, and lose Nemotron long-form and every Sortformer path (see
[performance.md](performance.md#nemotron-3-diarization-and-sortformer-v21-on-the-apple-neural-engine)).
As with the Supertonic vocoder, stage them for base and Pro chips and iOS, and
leave them out on Max and Ultra machines. The engine does not decide this;
whether the bundles sit beside the GGUF does.

## IndicConformer and Unified Core ML export

IndicConformer CTC uses the fixed-capacity sidecar contract shared by the
offline TDT and Unified encoders. The FastConformer stack runs in Core ML,
while the multilingual CTC vocabulary projection and language mask stay on
ggml. Short inputs are padded to the compiled capacity and long inputs use the
overlapping encoder-window plan:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/indic-conformer-600m-multilingual.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/indic-conformer-600m-multilingual-encoder.mlpackage \
  --compile-dir parakeet/models
```

Place the compiled directory beside any quantization as
`indic-conformer-600m-multilingual-encoder.mlmodelc`. Missing or incompatible
sidecars and prediction failures fall back to the complete ggml path.

Unified RNN-T uses the same fixed-capacity sidecar contract as TDT. Shorter
inputs are zero-padded to the exported capacity, and oversized inputs use the
existing overlapping long-form window plan:

```bash
python parakeet/scripts/export-encoder-coreml.py \
  --gguf parakeet/models/parakeet-unified-en-0.6b.q8_0.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out parakeet/models/parakeet-unified-en-0.6b-encoder.mlpackage \
  --compile-dir parakeet/models
```

Place the compiled directory beside the GGUF as
`parakeet-unified-en-0.6b-encoder.mlmodelc`. Missing or incompatible sidecars
and prediction failures fall back to ggml.
