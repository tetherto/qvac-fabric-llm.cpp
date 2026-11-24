# audiogen engine: performance

Part of the [audiogen documentation](../README.md).

These are dated measurements. Compare timings only with the matching model,
workload, backend, and timing boundary.

## ACE-Step 1.5 multi-machine benchmark (2026-09)

Measured with the repo's own benchmark harness
([`benchmarks/comparison`](../benchmarks/comparison/README.md),
`node run-comparison.js --backend ...`): 10 fixed prompts, 1 warm-up + 3
timed runs each, 5 s cooldown, `failOnGpuFallback: true`. Build: engine
`46afe7d9`, ggml `speech@157b299f`; GGUFs `Qwen3-Embedding-0.6B-Q8_0`,
`acestep-5Hz-lm-0.6B-Q8_0`, `acestep-v15-turbo-Q8_0`, `vae-BF16`, identical
SHA-256 on every host. `gen` is generation compute, `e2e` adds the lazy
stage loads inside `generate()`.

| Device | Backend | gen | e2e | RTF |
|---|---|--:|--:|--:|
| MacBook Air M5 | Metal | 9,016 ms | 9,682 ms | 0.957 |
| RTX 3080 desktop | CUDA | 1,796 ms | 2,146 ms | 0.198 |
| RTX 3080 desktop | Vulkan | 1,580 ms | 1,912 ms | 0.164 |
| Strix Halo | Vulkan | 2,330 ms | 2,570 ms | 0.253 |
| RTX 5090 box | CUDA | 1,338 ms | 1,772 ms | 0.139 |
| RTX 5090 box | Vulkan | 1,435 ms | 1,864 ms | 0.157 |

Vulkan beats CUDA on the 3080 (1,580 vs 1,796 ms) — that is the ACE-Step LM
running on the GPU for every Vulkan device except Mali (PR #234). Output QC
from the harness: no failures and no silent renders in 180 rounds, 10/10
unique WAV hashes per lane, and duration error at most 1.2 s against the
requested song length.

## ACE-Step 1.5 on Snapdragon 8 Elite (2026-10)

Galaxy S25 (Adreno 830, Hexagon v79), `music-cli --dur 30` with
`ACESTEP_KEEP_STAGES=1`, `q8_0` LM and DiT, the run with the median total of
three, each started from thermal status 0. `generate` is the
`[acestep-timing]` total.

| Placement | LM | DiT | VAE | generate | Speedup vs OpenCL |
|---|--:|--:|--:|--:|--:|
| CPU | 26,459 ms | 51,861 ms | 58,725 ms | 147,482 ms | 0.35x |
| OpenCL | 10,661 ms | 10,989 ms | 28,519 ms | 51,574 ms | 1.00x |
| `--backend hexagon --lm-backend cpu` | 23,812 ms | 1,946 ms | 12,165 ms | 38,517 ms | 1.34x |
| `--backend hexagon --lm-backend opencl` | 10,830 ms | 1,870 ms | 13,440 ms | 26,686 ms | 1.93x |
| `--backend hexagon` | 7,522 ms | 1,920 ms | 12,458 ms | 22,479 ms | 2.29x |

At `--dur 8`, the same five placements generated in 45,743 / 13,412 /
16,267 / 8,973 / 7,832 ms respectively.

Per-stage parity and setup are in [docs/backends.md](backends.md#hexagon-npu).

## MiniMax-Music3 f16 on RTX 5090 (2026-09)

`mm3-replay --mode full` on the RTX 5090 box with the `f16` pair: one fixed
caption, seed 7, warm driver shader cache, in-process pipeline time (the
`[MM3-Pipe]` total, model load excluded). Before is engine `075045a2` on ggml
`speech@8b980299`; after adds the CFG batching, LM flash attention, stacked
projections and LM cache layout described [below](#minimax-native-backend-validation-and-optimization)
and runs on the same ggml plus two CUDA changes: `GGML_PREC_F32` served on
half-precision weights by the f32 matrix-vector kernel
([qvac-ext-ggml#104](https://github.com/tetherto/qvac-ext-ggml/pull/104)) and a
row-bias add folded into the half-precision cuBLAS GEMM's f32 conversion
([qvac-ext-ggml#105](https://github.com/tetherto/qvac-ext-ggml/pull/105)). CUDA 13.3 (CUDA graphs on), Vulkan SDK 1.4.341.1, driver
595.91.07.

| Song | Backend | Before | After | RTF after | Speedup |
|---|---|--:|--:|--:|--:|
| 20 s (500 frames) | CUDA | 15,120 ms | 11,660 ms | 0.58 | 1.30x |
| 20 s (500 frames) | Vulkan | 15,682 ms | 13,247 ms | 0.66 | 1.18x |
| 2 min (3000 frames) | CUDA | 102,289 ms | 73,949 ms | 0.62 | 1.38x |
| 2 min (3000 frames) | Vulkan | 105,393 ms | 84,222 ms | 0.70 | 1.25x |

On the 2-minute song the LM decode step drops from 15.8 to 9.6 ms (CUDA) and
15.7 to 11.3 ms (Vulkan), the depth decoder from 7.3 to 6.1 ms and 8.6 to
7.4 ms per frame, and the flow stage from 29.3 to 23.1 s and 28.5 to 24.2 s.
The LM and depth steps now stream their f16 weights at about 1.55 TB/s on
CUDA, close to what the card sustains, so the AR loop is bandwidth-bound.

The quantized pairs speed up as well (same box and method, 20 s track):

| Pair | Backend | Before | After | Speedup |
|---|---|--:|--:|--:|
| `q8_0` | CUDA | 11,229 ms | 9,070 ms | 1.24x |
| `q8_0` | Vulkan | 13,055 ms | 10,175 ms | 1.28x |
| `q4_k_m` | CUDA | 9,705 ms | 7,505 ms | 1.29x |
| `q4_k_m` | Vulkan | 11,388 ms | 8,504 ms | 1.34x |

## speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |


Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

## MiniMax native backend validation and optimization

The following records preserve the native backend measurements and implementation
context previously included in the pipeline guide. They describe the measured
models and devices, rather than guarantees for other hardware.

Vulkan and CUDA are the measured GPU backends at `q8_0`/`f16`, both on an RTX
5090. On each, `mm3-replay --mode replay` forcing the recorded official prompt
tokens, codes, and noise reproduces the CPU rendering (time-domain audio
correlation 0.9998 on Vulkan, 0.9993 on CUDA), `--mode condcheck` confirms
byte-identical DiT velocities across repeated computes, and
`test-minimax-quality` lands the final flow latents on the learned manifold.
Vulkan is measured with both the `q8_0` and the `f16` pair; the 22 GB `f16`
pair far exceeds ggml's 1 GiB Vulkan suballocation block and loads through the
chunked device buffers ggml creates automatically. The full
`test-backend-ops` suite passes on the same device, including the
`b_absmax=1e5` LM-shaped `mul_mat` stress cases. The `q4_k_m` pair is
measured on CPU, Metal (M5), Vulkan (Strix Halo RADV and RTX 3080), and CUDA
(RTX 3080): teacher-forced replays of the same inputs agree with the
F32-dequantized reference at 0.97 waveform correlation on every backend
(the `q8_0` pair's own figure on the same probe is 0.9995), condcheck stays
byte-identical, and `q4_k_m` generates faster than `q8_0` (Strix Vulkan
94.7 s vs 105.5 s, CPU 397 s vs 471 s for the same 12 s clip). OpenCL
remains unmeasured for MiniMax and takes the same all-on-GPU placement, so
measure it the same way before shipping it.

Every stage evaluates both classifier-free-guidance branches in one pass. The
LM and the RVQ depth decoder fold the conditional and unconditional rows into
the columns of each weight product, so a decode step streams every weight once
instead of once per branch. For float (`f16`/`bf16`/`f32`) weights those
two-column products request `GGML_PREC_F32`, which CUDA and Vulkan serve with
matrix-vector kernels that keep the activations and the accumulation in f32 at
no cost in speed; CUDA needs a ggml that routes the request on half-precision
weights to that kernel
([qvac-ext-ggml#104](https://github.com/tetherto/qvac-ext-ggml/pull/104)), and
an older one computes the same products with f16 activations. Quantized
weights keep the default precision and their integer kernels, since CUDA would
serve the request by dequantizing the whole matrix for an f32 GEMM. The LM and
the depth decoder also load each layer's q/k/v and gate/up projections stacked
into one matrix when the parts share a type (MiniMax's own conversions always do; a
mixed k-quant layer keeps them separate), so a layer reads its input with two
products instead of five, and the LM keeps its K/V cache as one
`[head_dim * heads]` row per position with the unconditional row's positions
after the conditional row's, which attention reads in place. The flow DiT runs
both branches as one batched forward with a condition gate per branch, which
adds about 100 MB of DiT compute buffer at the 689-frame window; its biases are
added straight after their products, where CUDA folds them into the GEMM's f32
conversion
([qvac-ext-ggml#105](https://github.com/tetherto/qvac-ext-ggml/pull/105)). With
the `f16` pair on an RTX 5090 this generates a 20 s vocal track (500 frames) in
11.7 s on CUDA and 13.2 s on Vulkan (15.1 s and 15.7 s before) and a 2-minute
song (3000 frames) in 73.9 s and 84.2 s (102.3 s and 105.4 s before); the LM
decode step drops from 15.8 to 9.6 ms on CUDA and from 15.7 to 11.3 ms on
Vulkan over the 2-minute context. A 3001-iteration teacher-forced replay of
that song agrees with the CPU rendering as closely as before (CFG-guided argmax
agreement 99.70 % on both backends, against 99.60 % on CUDA and 99.87 % on
Vulkan before), and a 20 s replay through the full flow reproduces the CPU
audio at 0.99982 (CUDA) and 0.99987 (Vulkan) waveform correlation (0.99988 and
0.99989 before).

### Residency and tiled vocoder measurements

The vocoder decodes in overlapped tiles whose 32-frame overlap exceeds its conv
stack's receptive field, so no tile interior depends on context outside the
tile: the tiled output is bit-identical to a single-shot decode on the CPU and
matches it to rounding on GPU backends (waveform correlation 0.99999994 on
CUDA and 0.999996 on Vulkan, RTX 5090), whose GEMM kernels depend on the
product shape. Releasing the LM KV cache and compute buffers after the AR
stage, together with tiled vocoder decoding, lets the `q4_k_m` pair complete full generations on a
10 GiB GPU (RTX 3080, peak 9.4 GiB alongside a desktop) and on a 16 GB
Apple-silicon Mac over Metal (peak RSS 8.4 GiB), both of which the `q8_0`
pair cannot fit. `tts_cpp::minimax::fit_params` projects this footprint from
the GGUF metadata before a load; see [memory-fit.md](memory-fit.md).

### Flow step count and depth quantization history

Files converted before 2026-09-17 carry `mm3.flow.steps = 30`; they run at 20
steps like everything else. On an AMD Strix Halo (Radeon 8060S, RADV) the flow
stage is exactly linear in the step count — 612 ms per step at L=689, flat from
8 steps to 30 — so 20 steps cuts a 10 s track from 57.0 s to 47.2 s end to end.

The depth decoder used to be held at `q8_0` whatever variant was requested, on the
grounds that its per-frame matvec graphs want the integer fast path. Since the
depth decoder gained a K/V cache every one of those matvecs runs with a single
new token, and at that width `q4_K` measured faster than `q8_0` on an AMD Strix
Halo under RADV (65 µs against 107 µs for the 4096×4096 projections) with a
teacher-forced correlation of 0.9998 against the `q8_0` render. That device
reports no integer dot-product support, so the integer fast path was not part of
the comparison; on a device that has it, `q8_0` may still be the faster choice
for the depth decoder and is one `--quant q8_0` conversion away.

## ACE-Step backend validation

Use an F32-dequantized model (`audiogen/scripts/dequant_gguf.py`)
as the reference for quantized-model comparisons. CPU quantized matmuls also
quantize activations internally, so CPU output alone is not a ground-truth
trajectory. The following recorded probes informed the
[stage placement policy](backends.md#backends):

- Metal: the Q8_0 LM reproduced the F32 argmax trajectory exactly where CPU
  Q8_0 diverged at the first token.
- CUDA, RTX 5090: the Q8_0 LM reached 0.9999 logit cosine against the F32
  reference with an identical greedy trajectory; CPU Q8_0 reached 0.994.
  The full `test-backend-ops` suite and `--quantized-batch-cfg-regression`
  passed with the fork's explicit-precision matmul support.
- Vulkan, Strix Halo RADV: 300-token Q8_0 prefills matched the F32 argmax on
  3/3 probes at cosine >= 0.99999995; CPU matched on 1/3 at about 0.9998.
  The GPU LM stage ran about twice as fast as CPU on that device.
- Vulkan, RTX 3080: 300-token Q8_0 prefills matched the F32 argmax on 3/3
  seeds at cosine >= 0.99999998 and top-50 overlap 49.8/50. CPU matched on
  0/9, 5/9 and 0/9 steps for those probes, at cosine about 0.9999.
- OpenCL: stage validation used an Adreno 740. Hexagon measurements and its
  activation-quantizer requirement are in the [backend guide](backends.md#hexagon-npu).

Mali-G715 Vulkan testing produced repeated codes and early termination,
while CPU LM placement produced a diverse full-length sequence. That report
predates the Vulkan strided-source binding fix; the LM remains on CPU until
new on-device measurements justify a placement change.
