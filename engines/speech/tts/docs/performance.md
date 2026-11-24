# tts engine: performance deep dive

Part of the [tts engine documentation](../README.md).

The dated CI snapshots and maintainer campaigns below use different models,
workloads and timing boundaries. Each section records its own provenance; old
results do not describe the current branch. Shared setup for the initial Apple rows:

- Text: *"Hello from native C plus plus. This audio was generated end
  to end on CPU using ggml."*
- Reference voice: `test/reference-audio/jfk.wav` (11 s mono 16 kHz)
- Seed: 42, warm 3-run average, inference only (excludes model load)

### speech-cpp CI (2026-09-07, CPU + macOS)

Fresh CPU baseline from `speech-benchmark-desktop.yml` on the hosted-Linux and
self-hosted macOS runners (5 timed runs + 1 warmup); parler on macOS runs on
Metal (`MTL0`).

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Chatterbox: chatterbox-t3-turbo-q8_0 | linux | CPU | 8078 | — | 1835 |
| Chatterbox: chatterbox-t3-turbo-q8_0 | macos | CPU | 2626 | — | 1746 |
| Supertonic: supertonic3-q8_0 | linux | (CPU) | 708 | 0.484 | 876 |
| Supertonic: supertonic3-q8_0 | macos | (CPU) | 49 | 0.0340 | 584 |
| Parler: parler-mini-v1-q8_0 | linux | CPU | 249362 | 18.8 | 1964 |
| Parler: parler-mini-v1-q8_0 | macos | MTL0 (Metal) | 12204 | 0.582 | 1307 |
| Cosyvoice: cosyvoice3-llm-q8_0 | linux | CPU | 67695 | 18.0 | 1416 |
| Cosyvoice: cosyvoice3-llm-q8_0 | macos | CPU | 46825 | 12.3 | 1467 |

`—` means the benchmark artifact did not report RTF; variable output length
does not prevent computing RTF from the generated audio duration.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Mac Studio M3 Ultra (96 GB unified memory)

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`tts-cpp` Q4_0**                    | **Metal**       |  573 ms / 155 tok  |    412 ms      |   **985 ms**    | 0.16  | **6.4×**     |
| `tts-cpp` Q4_0                        | CPU (NEON+Accel)| 2 045 ms / 178 tok |  5 523 ms      |    7 568 ms     | 1.05  | 0.96×        |

On Apple Silicon the Metal build runs end to end at **6.4× real time**; the
CPU-only build stays just under real time.

### Multilingual (Apple M4, F16 weights)

Same prompt + seed run through both variants on the same M4, for apples-to-
apples comparison.  MTL is 30 transformer layers vs Turbo's 24 plus CFG on
both T3 and CFM (2 forward passes per step), and it samples standard CFM
for 10 Euler steps instead of Turbo's meanflow 2.

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|---------------------:|-------------:|------:|--------:|
| Turbo, Metal                        |  788 ms /  73 tok   |    768 ms    | 3.04 s| 0.51    |
| Turbo, CPU 4t                       | 1 721 ms /  73 tok  |  3 334 ms    | 3.04 s| 1.66    |
| Multilingual, Metal *(batched CFM)* | 1 865 ms /  61 tok  |  2 247 ms    | 2.56 s| 1.61    |
| Multilingual, CPU 4t *(2-call CFM)*³| 3 210 ms /  85 tok  | 25 660 ms    | 3.52 s| 8.20    |

The MTL Metal path packs the CFG cond+uncond into a single batch=2
decoder forward (`use_b2 = !ggml_backend_is_cpu(...)`), since kernel
dispatch overhead amortises well across the bigger workload; on ggml-cpu
the extra permute+cont ops that a batched attention block needs regress
throughput, so CPU keeps the two-call path.  See
[Chatterbox port archive §3.19](history/chatterbox-port.md) for the measurement and a discussion
of where the MTL slowdown lives relative to Turbo.

³ Re-measured on the same M4 host after commit `6d9b42b` restored the
CFG combine on the non-`use_b2` (CPU) CFM path.  The previous row in
this table (`2 711 ms / 71 tok`, `8 029 ms`, `RTF 3.63`) was captured
while CPU was silently running only the conditional CFM pass — i.e.
half the CFM compute and no classifier-free guidance steering.  S3Gen
wall-time roughly doubled (one extra forward call per CFM step) and
RTF went from 3.63 → 8.20.  The remeasurement here uses a different
local reference wav (the original `jfk.wav` was not present on the
host) — that accounts for the slightly larger token count (85 vs the
original 71); the per-audio-second cost ratio is what shifted, ~2×,
in line with restoring the missing pass.  Turbo and the Metal
multilingual row are
unaffected — they always carried the CFG combine.

### Multilingual (Mac Studio M3 Ultra, after §3.21 optimisation pass)

Same Spanish prompt (`"Hola, esto es una demostración multilingüe."`,
`--language es`), `jfk.wav` voice, seed 42, greedy (`--temp 0 --top-k 1`),
3 warm runs averaged.  T3 is now CFG-batched into a single Metal forward
(B=2, mirrors S3Gen's `use_b2`); MLP uses `ggml_swiglu_split` so the 30
SiLU+Mul element-wise pairs collapse into one fused Metal kernel per
layer.  The new `--cfm-steps N` flag exposes the standard CFM step count
(default 10); N=7 is the recommended quality knee (log-mel cosine vs N=10
= **0.995**).

| Config                              | T3 infer           | S3Gen infer | Audio | **RTF** |
|-------------------------------------|-------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0, `--cfm-steps 7`    |  478 ms /  84 tok  |    576 ms   | 3.48 s|  0.30   |
| MTL, Metal Q4_0 (default N=10)      |  482 ms /  84 tok  |    730 ms   | 3.48 s|  0.35   |
| MTL, Metal F16, `--cfm-steps 7`     |  579 ms /  89 tok  |    586 ms   | 3.68 s|  0.32   |
| MTL, Metal F16 (default N=10)       |  613 ms /  89 tok  |    752 ms   | 3.68 s|  0.37   |

Compared to the M4 multilingual numbers above, the M3 Ultra hits
**RTF 0.30** on Q4_0 — a 4.6× speedup.  The CFG-batching alone drops T3
by 42–45% (see the archived Chatterbox port report §3.21 for the full bench matrix and the
NEGATIVE results for F16 KV cache and SwiGLU on F16).

### Multilingual (M3 Ultra, post §3.24–§3.31 Metal kernel portfolio)

Same prompt, voice, seed as §3.21 above.  Adds, on top of §3.21:

- **§3.24** — HiFT conv-kernel F16 quantisation (64 tensors).
- **§3.26** — `kernel_mul_mv_f32_f16{,_4,_short}` Metal kernel variants
  to unblock 21 more HiFT `source_*` F16 tensors
  (GGUF shrinks **754 → 747 MB**, WAV cos 1.000000 vs §3.24).
- **§3.27** — `kernel_mul_mm` + `ADD(bias)` [+ `ADD(residual)`] fusion
  for the CFM transformer Q4_0 mat-muls (1820 saved `ggml_add`
  dispatches per synth).
- **§3.28** — extends the fusion to absorb `GELU_ERF` (CFM FF ff0
  activation path; 1120 additional saved dispatches).
- **§3.30** — `test-metal-ops` fused-mul_mm parity harness + bias-only
  direct-store variant.
- **§3.31** — iOS-arm64 cross-build portability +
  `scripts/bench-m4-validation.sh` for M4 hand-off.

5-invocation averages (`default N=10` CFM — compare to the §3.21 N=10 row):

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|--------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0 + HiFT F16 v2 (§3.28) |  433 ms / 84 tok  |    706 ms   | 3.48 s| **0.33** |
| MTL, Metal Q4_0 baseline (§3.21 N=10) |  482 ms / 84 tok  |    730 ms   | 3.48 s|  0.35    |
| **Δ §3.21 → §3.28**                   |  **−49 ms / −10.2 %** | **−24 ms / −3.3 %** | — | **−0.02** |

WAV is byte-exact deterministic across runs (md5
`d8a1b22375dbcb2259c686426a7d76c5` ×5).  Parity harness
`test-metal-ops` passes 14 gates (3 base + 3 conv_transpose_1d + 8
fused `mul_mm`).  The 1088-line ggml-metal patch backing these
kernel changes is shipped pre-applied by the fabric ggml (`qvac-fabric`
vcpkg port); the standalone chatterbox.cpp
repo carries it under `patches/ggml-metal-chatterbox-ops.patch`
against pinned ggml `58c38058`.  All §3.24–§3.30 kernel changes
cross-compile cleanly for iOS-arm64 (portability verified; runtime
measurement deferred until an M4 / iPhone / iPad run of
[`scripts/bench-m4-validation.sh`](../scripts/bench-m4-validation.sh)).

M3 Ultra CFM time specifically drops from 541.9 ms → 534.0 ms
(**−1.5 %**) — modest on this chip because per-dispatch overhead
is very low; expected to be larger on bandwidth-limited silicon
(M4 / A-series) where each saved `ggml_add` dispatch is worth more
relative to compute.

### Parler-TTS on CPU (Mac Studio M3 Ultra)

CPU backend only (`GGML_METAL=OFF`, `GGML_NATIVE=OFF` as in the prebuilds),
16 threads, `parler-bench` greedy for a fixed 1200 decode steps (13.83 s of
audio) so both builds do identical work, median of 3 runs after a warm-up,
inference only. "Before" is the previous `master`; both use the same
ggml.

| Model | Decode ms/step | DAC s | Total s | RTF | Speedup |
|---|--:|--:|--:|--:|--:|
| mini-v1 `f32`  | 18.26 → 12.18 | 4.13 → 1.40 | 26.07 → 16.07 | 1.886 → **1.162** | 1.62x |
| mini-v1 `f16`  | 14.29 → 7.43  | 4.12 → 1.39 | 21.31 → 10.35 | 1.541 → **0.748** | 2.06x |
| mini-v1 `q8_0` | 12.62 → 5.85  | 4.12 → 1.39 | 19.28 → 8.45  | 1.394 → **0.611** | 2.28x |
| mini-v1 `q6_k` | 12.46 → 5.49  | 4.13 → 1.39 | 19.13 → 8.02  | 1.384 → **0.580** | 2.38x |
| Indic `f16`    | 14.12 → 7.40  | 4.13 → 1.39 | 21.09 → 10.30 | 1.525 → **0.745** | 2.05x |
| Indic `q8_0`   | 12.66 → 6.04  | 4.13 → 1.40 | 19.36 → 8.68  | 1.400 → **0.627** | 2.23x |
| Indic `q6_k`   | 12.36 → 5.85  | 4.13 → 1.40 | 19.03 → 8.45  | 1.376 → **0.611** | 2.25x |

One-shot `parler-cli` runs (a fresh process per utterance, so model load and
the cold first synthesis included; sampled, seed 42, 16 threads, median of 3)
on the 3 s / 17 s / 25-30 s benchmark prompts:

| Model | RTF before (short / med / long) | RTF after |
|---|---|---|
| mini-v1 `q8_0` | 1.51 / 2.04 / 2.62 | **0.82 / 0.83 / 0.82** |
| mini-v1 `f16`  | 1.88 / 2.46 / 2.77 | **1.37 / 1.24 / 1.29** |
| Indic `q8_0`   | 1.49 / 2.10 / 2.59 | **0.81 / 0.85 / 0.91** |

Where the time went, and what replaced it:

- **Attention.** The manual CPU path copied the whole K cache and a
  transposed V cache into fresh tensors for every layer of every step, so
  per-step cost grew with the utterance (over a third of the decode
  profile was `memmove`/`dup`). Flash attention over the existing F32 cache
  reads it in place.
- **GELU.** ggml-cpu splits unary ops by row, and a decode step's FFN
  activation is a single row, so the exact-erf GELU ran on one thread
  (46 µs per layer, 1.1 ms per step). It is now viewed as 32 rows.
- **DAC.** Per-element snake chains, `im2col` copies and ggml's F32 GEMM
  (about 0.5 TFLOPS here) became one fused snake node and Accelerate sgemm
  calls (1.6 to 4.9 TFLOPS on these shapes), and the DAC compute buffer
  dropped from 268 MiB to 99 MiB.

Decode is now bound by memory traffic: the 144 decode matmuls alone stream
`q8_0` weights at about 170 GB/s (2.2 ms per step, saturating at 8 threads),
`f32` weights cost four times that, and the F32 KV read grows with context
(about 470 MB per step at 2400 steps). The engine's default of
`min(cores, 4)` threads leaves speed on the table here: mini-v1 `q8_0` runs at
RTF 0.85 at 4 threads, 0.66 at 8, 0.63 at 12 and 0.62 at 16, so pass
`n_threads` on desktop-class CPUs, but stay at or below 16 on this machine
(20 threads drops `f32` to RTF 2.69). A persistent ggml thread pool was
measured and left out: it is 15-20 % faster once warm, but in a fresh process
at 15 threads or more macOS demotes its long-lived workers and decode runs 3-4x
slower for many seconds.

Quality is unchanged. Teacher-forced argmax agreement with the HF f32
reference over a 200-step trace (3600 codebook rows) is identical for `f32`
(100 %) and `f16` (99.69 %); `q8_0` (97.94 → 97.89 %) and `q6_k`
(95.17 → 94.72 %) flip rows in both directions (40/38 and 83/67, McNemar
p = 0.91 and 0.22), the rounding noise of quantized activations rather than
a bias.

```bash
./build/parler-bench --model models/parler-mini-v1-q8_0.gguf \
    --text "The quick brown fox jumps over the lazy dog." \
    --threads 16 --max-frames 1200 --runs 3 --warmup 1
```

### Reproducing these numbers

```bash
# Build tts-cpp, then:
./build/tts-cli \
    --model       models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf  models/chatterbox-s3gen.gguf \
    --reference-audio test/reference-audio/jfk.wav \
    --text "Hello from native C plus plus. This audio was generated end to end on CPU using ggml." \
    --out /tmp/bench.wav \
    --seed 42 \
    --n-gpu-layers 99   # 0 or omit for CPU
```

The binary prints both the per-stage timings and `BENCH:` lines that
scripts can scrape.  Note: the binary also prints an inner
`=== pipeline: … RTF=… ===` line — that RTF covers **only the
S3Gen + HiFT phase** (the timer around `s3gen_synthesize_to_wav`, which
runs after T3 is already done).  The tables above report the full
end-to-end number (T3_INFER + S3GEN_INFER).

`gen_RTF = (T3_INFER_MS + S3GEN_INFER_MS) / AUDIO_MS`

Token counts vary slightly across backends because the CPU-side
sampler reads logits that come out of different float-reduction orders
per backend; per-token T3 cost is the directly-comparable figure.
Full development history and older backend combinations (F16 vs
Q4_0 / Q5_0 / Q8_0, plus other machines) are in
[Chatterbox port archive §3.10 / §3.13](history/chatterbox-port.md).

### Streaming mode — low-latency playback

For interactive use cases, the binary can emit audio **chunk-by-chunk**
as it's generated instead of waiting for the whole sentence to finish.
Any non-zero `--stream-chunk-tokens N` turns streaming on.

**Flags:**

- `--stream-chunk-tokens N` — main knob; N speech tokens per chunk
  (25 ≈ 1 s of audio, 50 ≈ 2 s).
- `--stream-first-chunk-tokens N` — override the *first* chunk's size
  so first-audio-out lands early while later chunks stay big and keep
  overall RTF low.  Typical: 10.
- `--stream-cfm-steps N` — CFM Euler step count. Turbo defaults to 2 and
  supports 1 or 2; one step trades some quality for lower cost.
  Multilingual uses standard CFM, and streaming requests below its model
  timestep count are floored to 10.
- `--out -` — emit raw `s16le` mono @ 24 kHz to stdout instead of
  writing a wav file, so the output can be piped straight into a
  player.

**Recommended low-latency preset for interactive use:**

```bash
brew install sox      # one-time, for the `play` command

./build/tts-cli \
    --model      models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text       "Hello from streaming Chatterbox." \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens       25 \
    --stream-cfm-steps          1 \
    --n-gpu-layers              99 \
    --out - \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

`play` ships with `sox` and routes straight to CoreAudio.  If you
prefer, the same stdout stream works with `ffplay -f s16le -ar 24000
-ch_layout mono -nodisp -i -` or piped through a Python
`sounddevice.play()` one-liner; on some macOS 26 builds ffplay's SDL
output is silent for raw piped audio, so `sox play` is the safest
default.

You can also drop the `--out -` to get a regular wav:

```bash
./build/tts-cli … --stream-chunk-tokens 50 --out out.wav
afplay out.wav
```

**Latency and throughput** on an Apple M4 with the Metal backend and
the preset above, feeding the sentence *"Hello from streaming
Chatterbox, I am John and I work in Google since 2010. I love to go
out with my friends, eat some pizza and also drink some wine. I also
love to travel around the world alone."* (produces 317 speech tokens,
~12.7 s of audio):

| metric | value |
|---|---|
| first-audio-out latency | **279 ms** |
| chunk 1 (10-token bootstrap) | RTF 0.99 |
| chunks 2–13 (steady-state, 25 tokens each) | **RTF 0.30 – 0.63** |
| chunk 14 (tail finalise) | RTF 1.42 |
| total wall time | 11.5 s for 12.7 s of audio |
| overall RTF | **0.90** |

The steady-state RTFs stay comfortably below 1.0, so the streamer
sustainably pushes audio faster than real-time playback consumes it.
Chunk 1 is small by design so first audio lands in ~280 ms; the final
chunk is short and relatively slow (fixed encoder/CFM overhead
amortised over only 0.4 s of audio).

For the full journal of how streaming got there — bit-exact CFM parity,
`cache_source` + `trim_fade` port, `--out -` stdout wiring, per-chunk
tuning — see [Chatterbox port archive §B1](history/chatterbox-port.md).

## CI benchmarks (2026-08-12, Linux x86-64)

End-to-end RTF measured in CI on the `tetherto/qvac` self-hosted runners, using
the q4 GGUFs from the QVAC model registry, 1 warmup + 5 timed runs.
`RTF = generation_time / audio_duration` (lower is faster; RTF is the
backend-comparable metric, wall time is workload-specific).

| Engine                  | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|-------------------------|--------:|-----------:|------------:|-------------:|
| Chatterbox (Turbo)      |    1.54 |      0.099 |      410 ms |          173 |
| Chatterbox Multilingual |    5.81 |      0.182 |     1036 ms |           77 |
| Supertonic              |   0.113 |      0.018 |       78 ms |          952 |
| Supertonic Multilingual |   0.101 |      0.013 |       84 ms |         1087 |
| Supertonic 3            |   0.225 |      0.029 |      118 ms |          631 |

_Source: workflow run [#31603192731](https://github.com/tetherto/qvac/actions/runs/31603192731)
(2026-08-12), runner `qvac-ubuntu2204-x64-gpu`, GPU **NVIDIA RTX 4000 SFF Ada
Generation** (`backend=vulkan`), benchmarking the published
`@qvac/tts-ggml@0.6.2` addon (released 2026-08-03, pinning `tts-cpp`
2026-08-03#1). This run adds the previously missing Supertonic GPU lanes, so
the Vulkan columns are now recorded for all engines._

## Supertonic 3 multi-machine benchmark (2026-09)

Maintainer-run measurement on four machines. `supertonic-cli` performs one
synthesis per process (it has no repeat flag), so the number a user feels is
the **end-to-end process wall**, model load included, timed from outside the
process. Two text lengths, ~9.6 s and ~26.5 s of speech; f16 weights
(`supertonic3-f16.gguf`, 197 MiB); engine `46afe7d9`, ggml
`speech@157b299f`. Load is the two-point intercept at zero audio,
`wall_short - slope * audio_short`.

| Device | Backend | wall short / long | RTF (long) | load |
|---|---|--:|--:|--:|
| MacBook Air M5 | Metal | 0.66 / 0.65 s | 0.025 | 0.66 s |
| MacBook Air M5 | CPU | 1.72 / 2.66 s | 0.101 | 1.18 s |
| RTX 3080 desktop | CUDA | 0.71 / 0.74 s | 0.028 | 0.69 s |
| RTX 3080 desktop | Vulkan | 0.61 / 0.62 s | 0.024 | 0.61 s |
| Strix Halo | Vulkan | 0.61 / 0.65 s | 0.025 | 0.59 s |
| RTX 5090 box | CUDA | 0.79 / 0.82 s | 0.031 | 0.77 s |
| RTX 5090 box | Vulkan | 0.72 / 0.74 s | 0.028 | 0.70 s |

On the GPU lanes synthesis is cheap enough that 17 s of extra speech costs
less than the run-to-run noise on a 0.6 s process, so the compute-only slope
over text length — `(wall_long - wall_short) / (audio_long - audio_short)` —
is above the noise floor only on the M5 CPU (0.0561 s per audio-second), the
RTX 3080 CUDA (0.0022), Strix Halo Vulkan (0.0022), and the RTX 5090 (CUDA
0.0020, Vulkan 0.0014); start-up dominates the wall everywhere else. One
caveat: `supertonic-cli` caps one batch synthesis at ~28.5 s of audio, which
sets the long text point (3x the short text, not 8x).

## Audio8 multi-machine benchmark (2026-09)

Maintainer-run measurement on three machines. `audio8-cli` performs one
synthesis per process, so the number is the **end-to-end process wall**, model
load included, timed from outside the process. Two text lengths, ~9.6 s and
~24 s of speech; q8_0 weights (`audio8-lm-q8_0.gguf` 800 MiB +
`audio8-codec-decoder-q8_0.gguf` 201 MiB); engine `0f9fc817`, ggml
`speech@157b299f`, seed 42, 16 threads on the Linux boxes and 10 on the Mac.
Load is the two-point intercept at zero audio,
`wall_short - slope * audio_short`.

| Device | Backend | wall short / long | RTF (long) | load |
|---|---|--:|--:|--:|
| RTX 5090 box | CUDA | 2.12 / 4.82 s | 0.203 | 0.48 s |
| RTX 5090 box | Vulkan | 2.22 / 5.12 s | 0.215 | 0.37 s |
| RTX 5090 box | CPU | 8.53 / 19.96 s | 0.840 | 0.46 s |
| Strix Halo | Vulkan | 3.87 / 9.58 s | 0.403 | 0.24 s |
| Strix Halo | CPU | 7.93 / 18.40 s | 0.774 | 0.53 s |
| Mac mini M4 | Metal | 6.76 / 15.43 s | 0.649 | 0.92 s |
| Mac mini M4 | CPU | 10.82 / 26.97 s | 1.134 | ~0 s |

Audio8 is autoregressive, so unlike Supertonic its compute-only slope over
text length is well above the noise floor on every lane
(`(wall_long - wall_short) / (audio_long - audio_short)`): 0.183 s per
audio-second on the RTX 5090 CUDA, 0.200 on its Vulkan, 0.393 on Strix Halo
Vulkan, 0.610 on Mac Metal, and 0.75–1.14 on the CPU lanes. Every GPU lane
runs faster than real time; the Mac mini M4 CPU lane is the only one that does
not (RTF 1.13). Method: 1 untimed warm-up plus 5 timed runs per cell, engines
alternated, 3 s cooldown, worst rep-to-rep spread in any cell 3.2 % (10 %
gate), outputs deterministic per cell and non-silent (peak amplitude
0.47–0.89), device confirmed in every run log (`on CUDA0` / `on Vulkan0` /
`on MTL0`). One build note: the ggml Vulkan build needs the Khronos
SPIRV-Headers include path on hosts without a system copy.

The table is the campaign as measured at engine `0f9fc817`. The RTX 5090 GPU
lanes were re-measured after the decode-loop work and the fast-AR graph cache
(same harness, same protocol, same box): **CUDA 1.31 / 2.67 s, RTF 0.116**,
slope 0.096 s per audio-second, and **Vulkan 1.65 / 3.73 s, RTF 0.157**, slope
0.144. The intercept of the wall-versus-audio fit — what a caller pays before
the first second of speech — is 0.45 s on CUDA and 0.32 s on Vulkan. The CPU
lane and the other machines are unchanged and are not re-stated here.

## Audio8 and LavaSR CI (2026-09-07)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Audio8: audio8-lm-q8_0 | linux | (CPU) | 4906 | — | 1193 |
| Audio8: audio8-lm-q8_0 | macos | (CPU) | 1602 | — | 1208 |
| Lavasr: lavasr-denoiser-f16 | linux | (CPU) | 4102 | 0.684 | 155 |
| Lavasr: lavasr-denoiser-f16 | macos | (CPU) | 2090 | 0.348 | 209 |

`—` means RTF was not reported in the artifact.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218).
