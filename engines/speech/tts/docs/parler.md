# tts engine: Parler-TTS

Part of the [tts engine documentation](../README.md).

## Parler-TTS

Description-conditioned TTS: the transcript (`--text`) is spoken in a voice
controlled by a natural-language description (`--description`).  Supports
`parler-tts/parler-tts-mini-v1`, `parler-tts/parler-tts-large-v1` and
`ai4bharat/indic-parler-tts` (21 languages incl. Hindi/Gujarati);
model differences are pure GGUF metadata (`parler.*` keys), no code
branching.  Pipeline: Flan-T5 encoder (description → cross-attention K/V,
precomputed once and cached per description) → delay-pattern decoder LM
(9 DAC codebooks, MusicGen-style stagger, HF-faithful EOS gating) → DAC
codec decode → 44.1 kHz mono PCM.  Validated backends are CPU, Metal,
Vulkan, and OpenCL (Adreno): the GPU path (F16 flash attention, DAC
upsampling as phase matmuls) is gated on that allowlist in
`src/parler/gguf.cpp`, and any other GPU backend is released at load and
replaced by CPU rather than run unvalidated.  The Hexagon NPU runs only on
explicit request (see [Snapdragon Hexagon NPU](#snapdragon-hexagon-npu)).  GPU loads and mmap-backed CPU
loads fuse the per-layer QKV projections and the nine LM heads into single
row-concatenated matmuls (byte-exact; the CPU allocate-and-stream fallback
stays unfused so the projections are never resident twice, and
`PARLER_NO_FUSED` disables fusion everywhere).  Every backend samples each
step over the top-k candidate set with reused scratch buffers, and host
backends read the step logits in place from the graph buffer instead of
downloading a copy.  The graph dispatch uses the shared `sched_dispatch`
dual path like the other engines.

On the CPU backend the decoder's self- and cross-attention also run through
ggml's flash-attention kernel, over an F32 KV cache, so they multiply and
accumulate in F32 like the manual softmax(QK^T)V path while no longer copying
the whole cache every step (`PARLER_NO_FA` restores the manual path), and the
FFN's exact-erf GELU is viewed as rows so every thread shares it.  The CPU DAC
runs each snake activation as one fused node (over reciprocals precomputed at
load from the F32 snake alphas every converted GGUF carries); on Apple
builds its convolutions and transposed convolutions run on Accelerate's sgemm
with the snakes on vForce (`PARLER_DAC_NO_ACCEL` keeps the ggml kernels), which
also shrinks the DAC compute buffer from 268 MiB to 99 MiB.  Accelerate tiles
by shape, so a windowed or streamed DAC decode matches a whole-sequence one
within float tolerance there rather than bit for bit, as on a tinyBLAS build.
See [Performance](performance.md#parler-tts-on-cpu-mac-studio-m3-ultra) for
the Mac Studio M3 Ultra numbers.

Indic-class checkpoints ship a second, SentencePiece-BPE **prompt**
tokenizer (90k vocab covering the Indic scripts, byte fallback) alongside
the Flan-T5 description tokenizer; the converter embeds both
(`parler.prompt_tokenizer.*` keys) and the engine routes prompts through
the BPE tokenizer when present.  mini/large GGUFs are unaffected (one
shared tokenizer, byte-identical output).  The language is auto-detected
from the prompt script; descriptions stay English (name a recommended
voice — e.g. Rohit for Hindi, Yash for Gujarati — for stable speakers).

```sh
# convert (single GGUF: T5 + decoder + DAC + tokenizer; ~3.4 GB f32)
python3 scripts/parler/convert-to-gguf.py \
    --model-id parler-tts/parler-tts-mini-v1 --dtype f32 \
    --out models/parler-mini-v1-f32.gguf

# synthesize (also autodetected by tts-cli via parler.arch + --description)
./build/parler-cli --model models/parler-mini-v1-f32.gguf \
    --text "Hey, how are you doing today?" \
    --description "A female speaker with a calm, clear voice, close up." \
    --out out.wav
```

The indic checkpoint is gated on HF (`gated: auto` — any account gets
instant access); pass a downloaded snapshot directory as `--model-id`
with `--reference-repo ai4bharat/indic-parler-tts` for provenance.

Instead of writing a full `--description`, the voice can be configured
through template flags (`build_description()` in
`<tts-cpp/parler/description.h>` — the same renderer the downstream addon
uses).  The rendered text follows the models' training-caption phrasing,
and every flag has a working default — with no flags at all the engine
uses the models' recommended fallback caption ("The speaker speaks
naturally. The recording is very high quality with no background noise."):

| flag | values (default first) | rendered as |
|---|---|---|
| `--voice` | free name, e.g. Rohit, Laura | sentence subject |
| `--emotion` | one of the 12 below | tone clause + "The intended style is …" |
| `--pitch` | unset, low, moderate, high | "with a low pitch" |
| `--pace` | unset, slow, moderate, fast | "speaks slowly" / "at a fast pace" |
| `--expressivity` | unset, monotone, slightly expressive, expressive | "in an expressive manner" |
| `--noise` | clear, noisy | "with no/noticeable background noise" |
| `--reverb` | close, distant | "distant-sounding" (close is implied) |
| `--quality` | very high, high, basic | "The recording is … quality" |

Emotions come from the shared vocabulary in [Voice conditioning](voice-conditioning.md#voice-conditioning-cross-engine)
— the 12 speaking styles in the indic training set, case-insensitive and
validated.  Each renders an in-distribution clause ("with an angry tone",
"delivering the news", "perfect for narration") plus the trailing style
anchor sentence the training captions used.  The indic card lists 10 officially emotion-tested
languages (Assamese, Bengali, Bodo, Dogri, Kannada, Malayalam, Marathi,
Sanskrit, Nepali, Tamil); elsewhere — including Hindi/Gujarati and the
mini/large English checkpoints — emotion conditioning exists but is
best-effort: validate by ear.  `--description` and the template flags are
mutually exclusive (the CLI errors out rather than silently preferring
one).

`--dtype f16` (~2.5 GB) casts only the decoder matmul weights and
embeddings; the T5 encoder stays f32 (Flan-T5 activations overflow the
f16 range — the well-known T5 fp16 trap), as do norms, biases, snake
alphas, the positional table and all DAC tensors.

Quantized recipes (mini sizes; argmax agreement vs the f32 fixtures over a
200-step teacher-forced trace, f16 scoring 99.6%):

| `--dtype` | size | agree | bulk / tables / heads / T5 |
|---|---|---|---|
| `q8_0` | ~1.16 GB | 98.1% | q8_0 / f16 / f16 / q8_0 |
| `q6_k` | ~0.98 GB | 94.9% | q6_K / q6_K / f16 / q8_0 |

Sub-q6 tiers measured well below this quality floor (q5_0 90.4%, q4_k_m
83.1% on mini; 70.0% / 65.9% on large) and are deliberately not shipped —
reproduce them via `--recipe` if ever needed.  The recipes transfer to
the indic checkpoint unchanged (its own 200-step fixtures: f16 99.6%,
q8_0 98.3%; f32 scores 100%).  All recipes keep the f32
set above untouched and the T5 matmuls at q8_0 (quantized dot products
re-quantize activations per block, so the f16 trap does not apply — T5
must never go f16).  The 9 LM heads never drop below q8_0 (6-bit heads
derail sampled decoding); both tiers lift heads (q8_0 also tables) to
f16 — the dominant quality lever found by grid search
(`scripts/parler/quant-grid.py`; per-tier override via `--recipe`, optional
activation-weighted quantization via `scripts/parler/compute-imatrix.py` +
`--imatrix`).  Non-trivial encodes go through the built ggml library
(`ggml_quantize_chunk` via ctypes; auto-located, or pass `--ggml-lib`), so
build tts-cpp first.  Quantized output diverges from the f32 parity
fixtures by construction — validate by ear;
`PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-t5 MODEL.gguf REF_DIR` and
the corresponding `test-parler-decoder` invocation print stage metrics without
enforcing the f32 tolerance bars.

Digits in the prompt are expanded to English words before tokenization
("12" → "twelve"): parler-v1 ships no text front-end and voices raw digits
badly, so this deliberately diverges from stock HF.  English cardinals
(incl. thousands separators), decimals and ordinals are covered; times,
currency and units are not.  Opt out with `--no-normalize-numbers`
(`EngineOptions::normalize_numbers = false`); the description is never
rewritten.  On indic-class models, ASCII digit runs whose nearest letter
context is an Indic script are instead transliterated to that script's
native digits ("कमरा 12" → "कमरा १२"; all 13 digit-bearing scripts of the
21 languages covered); Latin-context runs still become English words, and
native numerals always pass through untouched.  Best effort by design:
the model only voices native numerals for scripts whose numerals were
frequent in its training text (Devanagari yes; e.g. Gujarati no —
verified by ear); full per-language number-words belong upstream where
the prompt language is known.

Verification: `ctest -R test-parler` runs tokenizer/T5/decoder/delay/DAC/
e2e parity against `.npy` fixtures produced by
`scripts/parler/dump-reference.py` (HF PyTorch reference, greedy).
`test-parler-gguf-load` needs no fixtures: it synthesizes tiny GGUFs and
asserts the loader fails closed on truncated files, missing metadata, and
wrong-typed metadata.  The
greedy token trace matches HF exactly; the DAC waveform matches at
>120 dB SNR. Default decoding is sampled (temperature 1.0, top-k 50, from
`generation_config.json`). End-user `--greedy` and `top_k=1` requests are
repaired to sampling with a warning because argmax does not terminate for this
architecture; use `--seed` for reproducibility. Greedy is retained only inside
the bounded parity harness.

Streaming is available through the `parler::Engine` callback API using
`stream_chunk_frames` and `stream_first_chunk_frames`. Neither `parler-cli` nor
the limited `tts-cli` Parler route exposes those controls.

## Core ML DAC sidecar (Apple Neural Engine)

With `TTS_CPP_COREML=ON` (Apple only), the DAC decoder can run on a Core ML
sidecar, `<gguf stem minus quant tag>-dac.mlmodelc` next to the GGUF
(`parler-mini-v1-q8_0.gguf` -> `parler-mini-v1-dac.mlmodelc`; one sidecar
serves every quantization of a model, since the DAC tensors are always F32).
The T5 encoder and the decoder LM stay on ggml (Metal or CPU); the residual
vector quantizer runs on the host from the GGUF's codebooks. The sidecar is
used whatever the `backend` option selects for the LM, as with the Audio8
and Supertonic sidecars. The exporter reads only the GGUF's `parler.dac.*`
geometry and `dac.*` tensors, so it applies to every Parler checkpoint; Indic
has the same measured DAC geometry as mini: all 116 DAC tensor shapes match,
with latent width 1024, decoder width 1536, nine 1024-entry codebooks,
upsampling rates `[8, 8, 4, 2]`, hop 512, 44.1 kHz output and 11 context
frames per side. Indic speech accuracy and the quiet-waveform error floor are
reported below:

```sh
python scripts/parler/export-dac-coreml.py \
    --gguf models/parler-mini-v1-f32.gguf --compile-dir models
# check the rebuild first: --parity-dir artifacts/parler-ref --no-export
```

### Generated sidecar and background loading

Without a prebuilt sidecar the engine can build the same sidecar itself from
the GGUF it loads: set a cache directory (`EngineOptions::coreml.cache_dir`,
or `QVAC_COREML_CACHE_DIR`). On macOS 15 or later, in a build with
`TTS_CPP_COREML=ON` and `TTS_CPP_COREML_GENERATE=ON` (the macOS default; iOS
builds embed no template), the engine checks the GGUF against the embedded
template `parler-dac-w128` (`coreml-templates/parler-dac-w128/`: the
exporter's weight-free graph for a 128-frame window plus a recipe), writes
the sidecar's `weight.bin` from the GGUF's `dac.dec.*` tensors, compiles the
package on the device and publishes it into the cache. Nothing is downloaded
and no Python runs. The weights are byte-identical to the exporter's fp16
export of the same GGUF, so a generated sidecar decodes exactly like the
prebuilt one. Every quantization of a model shares one artifact, since the
cache key covers only the DAC tensors (F32 in every tier); mini and Indic
share the template. large-v1's Hugging Face `audio_encoder` config equals
mini's (`dac_44khZ_8kbps`: latent 1024, 9 codebooks of 1024 entries, 86 Hz
at 44.1 kHz), so it should share it too, but that stays unverified at the
tensor level until a converted large-v1 GGUF is checked against the
template. A GGUF that does not match runs on ggml with state `no_template`.
The engine opens the GGUF once and reads its metadata and every weight
through that descriptor, and generates only from that file: a GGUF replaced
at the path during the load or since (an atomic rename, a re-pointed
symlink) fails generation with `source_changed`, and the DAC stays on ggml.

The sidecar, generated or prebuilt, loads in the background. Until it is
ready every DAC call decodes on ggml and `SynthesisResult::coreml_pending`
is true; each `parler_dac_decode` call takes its backend once, at its start,
so one call never mixes backends (a stream whose sidecar becomes ready
between chunks reports `mixed`). On an M2 (macOS 15.7.3) the first
generation of the mini DAC takes about 3 s (digest 1.6 s, write 1.4 s,
compile 0.12 s) and the first Neural Engine load 8.1 s; later processes
adopt the cached artifact, and a warm load takes about 0.1 s.

| `EngineOptions::coreml` field | Effect |
|---|---|
| `cache_dir` | where generated sidecars and warm-load hints live; unset -> `QVAC_COREML_CACHE_DIR` -> none (no generation) |
| `use` | `automatic` (default); `off` (ggml only); `require`: like `wait = call`, and a sidecar that cannot serve a call (failed to generate or load, a failed prediction) is an error |
| `generate` | `background` (default) or `never` (prebuilt sidecars only) |
| `wait` | `none` (default on macOS: never block, apart from a short bounded wait for a load this host finished quickly before); `construct` (the constructor waits until the DAC is ready or failed); `call` (each call waits) |
| `prewarm` | `none` (nothing until the first call) or `minimal` / `predicted` / `all` (the DAC is requested at construction: the default) |
| `compute_units` | as `coreml_compute_units`; unset falls back to it, then to `PARLER_COREML_COMPUTE_UNITS`, then `all` |
| `strict_wait_ms` | longest wait of a `use = require` call; unset waits without limit |
| `memory_budget_bytes` | the budget fit was assessed against; when only the ggml projection fits within the budget or the free memory, whichever is smaller, the DAC stays on ggml (memory-degrade) instead of fit refusing the model |

With `memory_budget_bytes` set, the constructor repeats the fit to decide
memory-degrade. It uses `EngineOptions::fit_description_tokens`,
`fit_prompt_tokens` and `fit_margin_bytes` (defaults 50, 128 and 256 MiB,
as in `FitOptions`): set them to the workload and headroom of the
`fit_params` call that assessed the engine, so the engine reaches the same
decision.

`coreml_strict` maps to `use` (`true` -> `require`, `false` -> `automatic`)
when `use` is unset. `PARLER_COREML_DISABLE` still overrides everything.
`Engine::coreml_status()` returns the stage's state as JSON (`route`
`prebuilt`, `generated` or `cached`; `state`; compute units; timings of the
digest, write, compile, load and warm-up; the artifact key and path; the
error and its class), and `Engine::coreml_wait(ms)` blocks until it is ready
or has failed; it returns false at once while nothing has requested the DAC
(`prewarm = none` before the first call) or its load waits for memory.
`Engine::cancel()` also interrupts a call that waits for the sidecar
(`wait = call` or `use = require`): the synthesis fails as cancelled. It
ends a `coreml_wait()` in progress too, which then returns false; the
sidecar keeps loading, and the next `coreml_wait()` waits for it again. A
wait clears nothing: `coreml_wait(0)` only polls, and a longer wait ends
on any `cancel()` made after it started (a count of cancels it compares
with the one it started with), so concurrent waits and a synthesis on
another thread each keep the cancels meant for them. With
`prewarm = none` the constructor issues nothing, so `wait = construct` does
not wait and `use = require` fails only when a call cannot be served.

```cpp
tts_cpp::parler::EngineOptions options;
options.model_gguf_path = "models/parler-mini-v1-q8_0.gguf";
options.coreml.cache_dir = "/Users/me/Library/Application Support/qvac-speech/coreml";
tts_cpp::parler::Engine engine(options);   // returns at once; the DAC builds in the background
auto first = engine.synthesize("Hello.", "A calm voice.");  // ggml DAC, first.coreml_pending
engine.coreml_wait(-1);
auto next = engine.synthesize("Hello.", "A calm voice.");   // next.dac_backend == "coreml-all"
```

With a generated route, `parler::fit_params` (with `FitOptions::coreml` set
like the engine's) adds the template's Core ML envelope to the projection
and reports it in `FitResult::coreml` (`route`, `bytes`, `validated`,
`disabled_reason`, `ggml_only_fits`). The ggml DAC stays resident, since it
serves every call until the sidecar is ready. The envelope is the sidecar's
resident weights, I/O buffers, prediction workspace and the host quantizer
copy, plus the larger of the generation and the load transients. Each of
its constants is measured with its own 1.25 margin (below) for the `all`
compute units, so the envelope adds none (`margin` 1.0, `coreml_bound`
`measured`). Other compute-unit policies are not generated until
validated. The route never reads the cache, so fit is the same before and
after the first generation.

Memory validation (`test-parler-dac-coreml --memory-schedule DIR`, compute
units `all`): ggml DAC decodes run on the calling thread while the sidecar is
generated, compiled, loaded and warmed into an empty cache in the background.
The test reads the process footprint every 2 ms and the slot's state every
10 ms. A run passes only when 1.25 times each phase it measured fits the
template constant that charges it:

- the prepare (digest, generation and compile, over the footprint just
  before the digest) against `generation_peak_bytes` + 8 MiB +
  `compile_peak_bytes`;
- the load and warm-up (over the footprint just before the load) against
  `load_peak_bytes.all`;
- the steady state (ready, Core ML decodes running, over the footprint just
  before the load) against the template's resident part (the host quantizer
  copy is built at construction, before the load, so that footprint already
  holds it);
- `ANECompilerService`'s absolute size while loading against
  `ane_compile_reserve_bytes`. Each `ps` query of the service is bracketed
  by two reads of the slot state; one that crossed the load's start or end
  is dropped, so a size read before the load never counts as one during it.

A run also fails when a state was never sampled (the digest does not count
for the generation), when a probe failed, or when a decode failed or ran on
another backend than ggml or the route's own label (only successful decodes
count towards the schedule's ggml and Core ML calls).

The whole run (over the warm ggml baseline) is held to the envelope as
measured, since the envelope adds phases that each carry the margin. A run
that sets a new maximum for any phase therefore fails until the constants
are recomputed.

`ANECompilerService` is a long-lived system process shared by every process
on the Mac: it runs as root under `aned`, so a test cannot restart it, and
it does not exit when idle. Its size includes whatever other processes
compiled, before this load or during it, summed over every instance of the
service. Each run also prints its growth during the load (its peak less its
size just before the load) and labels that baseline clean (at most 32 MiB),
dirty or unknown (no sample before the load). A service that already holds
memory reuses it, so on a dirty baseline the growth understates a cold
compile on a clean host. No run on either host could start the service
clean, so no isolated measurement compares the two placements, and the
reserve is 1.25 times the largest absolute size while loading on any host:
a conservative basis. In CI, where a pull request starts several macOS jobs
at once, another job compiling for the Neural Engine during the load counts
too; a run above 821.6 MiB fails the reserve and, by the rule above, raises
it.

Each run prints a per-phase table and a one-line JSON report. Each constant
is 1.25 times the largest value measured over every run on every validated
host, rounded up to 1 MiB (`memory.measured` in `template.json`), and the
envelope applies no further margin. Figures in MiB, over 82 run reports:

| Phase | M2, macOS 27.2.1 (37 runs) | M4 Max, macOS 26.7.2 (CI, 45 runs) | Charged |
|---|--:|--:|--:|
| prepare: generation and compile | 435.6-442.0 | 268.6-438.5 | 553 (1.25 x 442.0): `compile_peak_bytes` 329 on top of `generation_peak_bytes` 216 + 8 |
| load and warm-up | 16.3-25.2 | 148.5-326.4 (warm-up 134.4-308.2) | 409 (1.25 x 326.4): `load_peak_bytes.all` |
| steady state, over the footprint before the load | 0 (32 runs) | 150.1-326.7 (42 runs) | 409 (1.25 x 326.7): the weights 103.3, I/O 0.8 and `workspace_bytes` 305.0 |
| whole run, over the warm ggml baseline | 453.0-466.3 | 577.7-759.2 | 962.6: the envelope, with the quantizer copy 0.6 |
| `ANECompilerService` before the load | 2.9-393.8 (24 runs, one clean) | 123.3-485.8 (36 runs, none clean) | |
| `ANECompilerService` growth during the load | 104.3-330.0 (24 runs) | 0.8-3.1 (36 runs) | reported |
| `ANECompilerService` size while loading | 332.9-518.7 | 124.7-821.2 | 1,027 (1.25 x 821.2): `ane_compile_reserve_bytes` |

The runs without a figure in a row predate its measurement. The M4 Max
figures come from four CI runners of the same model, whose load and warm-up
ranged from about 150 MiB in some jobs to 326 MiB in others.

- **M2.** `all` runs the program on the Neural Engine. Its weights do not
  show in the process footprint, and the generation is the peak.
  `ANECompilerService` held 124-394 MiB before most loads and stayed at
  333-519 MiB while loading. One run found the service process at 2.9 MiB
  resident before the load (its pages compressed or purged meanwhile, not a
  new process); it grew 330.0 MiB. The same M2 on macOS 15.7.3 peaked at
  224-229 MiB over four runs (`ANECompilerService` 360-497 MiB absolute).
  Those runs recorded no per-phase breakdown and move no constant, but stay
  in `validated_for`.
- **M4 Max.** `all` runs most of the program on the GPU (below), and only
  the output stage on the Neural Engine, so `ANECompilerService` barely
  grows (at most 3.1 MiB in every run that recorded its baseline). Its size
  there comes mostly from other jobs' compiles on the runner's Mac, earlier
  or concurrent. The largest reading, 821.2 MiB, came from a run that
  recorded no baseline; the next two runs of the same job read 494 MiB. It
  sets the reserve all the same, since no clean measurement shows what a
  cold compile of this placement needs. The first prediction adds 134-308
  MiB, which stays resident: the prediction workspace. The whole schedule
  takes 3.5 s there, so the process still holds the generation's memory
  when the load starts (the M2 has returned it by the time the sidecar is
  ready). Its whole run is therefore the prepare plus the resident, which
  is the sum the envelope charges.

The envelope is 1,009,356,800 bytes (962.6 MiB), up from 475.4 MiB. With the
M2's values alone (the template's weights and I/O as the resident part) it
would be 657.6 MiB: the macOS 27 generation peak (442 MiB, against the
216 MiB of the template's generation estimate) causes most of the rise, and
the M4 Max's GPU resident adds the remaining 305.0 MiB, its
`workspace_bytes`. `validated_for` is matched on the compute units only, so every
chip is charged the larger figure. A cold Neural Engine load starts only
while 1,027 MiB are available; until then the DAC stays on ggml (state
`deferred(memory)`). Phones are not affected today: the runtime generates
sidecars on macOS only, so on iOS the route is none and fit charges no Core
ML envelope. Matching validation on the chip class, as well as the compute
units, is planned before generation is enabled on iOS, whose chips run the
program on the Neural Engine like the M2.

Generated-sidecar accuracy (mini `q8_0`, compute units `all`, fp16, on an
empty cache; the generated `weight.bin` is byte-identical to the exporter's
export of the same GGUF):

| Check | SNR | max abs |
|---|--:|--:|
| random codes, 265 frames (3 windows), vs ggml CPU | 56.2 dB | 1.7e-3 |
| random codes, 42 frames (one partial window), vs ggml CPU | 56.4 dB | 1.2e-3 |
| random codes, 1 frame, vs ggml CPU | 53.3 dB | 1.5e-3 |
| HF reference `case0` (greedy codes, 4.9 s), vs the PyTorch fp32 wav (macOS 27.2.1) | 49.7 dB | 1.4e-3 |

CI generates the sidecar from a staged `parler-mini-v1-q8_0.gguf` on every
pull request (`test-parler-dac-coreml --cache-dir DIR --min-snr 54
--require-coreml --units-sweep`), and runs the memory schedule above three
times in its own step, each from a re-signed copy of the test (a fresh
identity, so every Neural Engine load is cold) and an empty cache. Its
runner is an Apple M4 Max,
a chip with a recorded floor under `all` (below), so CI holds `all` to 50 dB
and the same artifact under `cpu_and_ane` to 54 dB. The 54 dB gate under
`all` on the M2 comes from the device run of the same command, not from CI.
A one-frame decode (512 samples of mostly masked context, an ill-conditioned
fp16 SNR) is held to 53 dB on the M2 only, for the generated sidecar under
`all`, which measures 53.3 dB there; every other chip, the units sweep and
the prebuilt sidecars hold it to `--min-snr` like the longer cases. With `--ref-dir` (the `case0_*.npy`
fixtures `scripts/parler/dump-reference.py` writes from the Hugging Face
checkpoint), the HF reference check uses `--min-speech-snr` (default 45 dB).
It runs only on a device with the fixtures, not in CI; the M2 (macOS 27.2.1)
measures 49.7 dB.

`--units-sweep` also prints the same artifact's SNR for every compute-unit
policy. The rows are reported only, except that on a chip with a floor the
`cpu_and_ane` row is held to `--min-snr` in every case. A case holds only
when the reference decoded on ggml and the sidecar decoded it under the
label the requested units give (`coreml-ane` for `cpu_and_ane`; a fallback
to ggml would compare ggml with itself, and a sidecar opened on other units
fails the row); on such a chip a sweep
that cannot find, link or load the artifact fails. Random codes, 265 / 42 / 1 frames,
vs ggml CPU:

| Host | `cpu_only` | `cpu_and_gpu` | `cpu_and_ane` | `all` |
|---|--:|--:|--:|--:|
| M2, macOS 15.7.3 (24G419) | 44.9 / 45.1 / 44.5 dB | 68.3 / 69.2 / 67.7 dB | 56.2 / 56.4 / 53.3 dB | 56.2 / 56.4 / 53.3 dB |
| M2, macOS 27.2.1 (26B131) | 45.0 / 45.2 / 44.6 dB | 68.3 / 69.2 / 67.7 dB | 56.2 / 56.4 / 53.3 dB | 56.2 / 56.4 / 53.3 dB |
| M4 Max, macOS 26.7.2 (25G332) | 45.0 / 45.2 / 44.6 dB | 68.3 / 69.2 / 67.7 dB | 54.8 / 55.0 / 54.2 dB | 51.3 / 51.4 / 51.8 dB |

The ggml reference is the same on every host (the `cpu_and_gpu` figures
match to 0.01 dB), and the M2 measures the same on macOS 15 and 27, so the
difference comes from the chip. Core ML partitions the program between the
GPU and the Neural Engine with its own per-chip cost model (`MLComputePlan`):

- On the M2, `all` runs every op on the Neural Engine, exactly as
  `cpu_and_ane`.
- On the M4 Max, `all` runs 443 of the 493 ops (89.5% of the estimated
  cost) on the GPU. It leaves eight folded-stage convolutions and the output
  stage (`conv_out`, `tanh`) on the Neural Engine. In that GPU-fed island
  the Neural Engine compiler fuses `tanh` into `conv_out` as its activation,
  which measures 51.7 dB from exact inputs (62.8 dB for `conv_out` alone).
  Up to the output stage, the M4 Max's `all` path is as accurate as the GPU
  (68 dB).

No single spelling of the program holds 54 dB under `all` on both chips
(265 / 42 / 1 frames, against the PyTorch fp32 rebuild):

| Program | M2 `all` | M4 Max `all` |
|---|--:|--:|
| as exported | 56.6 / 56.7 / 54.9 dB | 51.5 / 51.4 / 50.5 dB |
| output stage in fp32 (off the Neural Engine) | 52.8 / 52.2 / 51.5 dB | 67.9 / 68.5 / 65.9 dB |
| `tanh` after the output unfold | 52.3 / 51.8 / 50.5 dB | 51.5 / 51.4 / 50.5 dB |

Within the M2's all-Neural-Engine program, the fused output stage is the
more accurate one. `specializationStrategy = fastPrediction` does not change
the M4 Max figures. The template therefore keeps `all`, and the test holds
the M4 Max to a recorded 50 dB floor under `all`
(`GENERATED_ALL_UNITS_FLOORS` in `test/parler/test_dac_coreml.cpp`, keyed
on `sysctl machdep.cpu.brand_string`, matched exactly). Every other chip is
held to `--min-snr`. The floor expires: the test fails once every
multi-frame case on a floored chip reaches `--min-snr`, so the row is
removed. On the M4 Max the same artifact under `cpu_and_ane` holds 54 dB at
about twice the `all` window time, but the engine does not generate for
`cpu_and_ane`: compute-unit policies other than `all` are not generated
until their memory envelope is validated (see the fit envelope above).

| Environment | Effect |
|---|---|
| `PARLER_COREML_DISABLE` | never open the sidecar (the DAC runs on ggml) |
| `PARLER_COREML_STRICT` | a missing, unusable or failing sidecar is an error instead of a fallback |
| `PARLER_COREML_COMPUTE_UNITS` | `all` (default), `cpu_and_ane`, `cpu_and_gpu`, `cpu_only` |

SDK hosts can set `EngineOptions::coreml_compute_units` and
`EngineOptions::coreml_strict` per model instance without writing process
environment variables. The units string accepts exactly `all`, `cpu_and_ane`,
`cpu_and_gpu`, or `cpu_only`; an empty string inherits
`PARLER_COREML_COMPUTE_UNITS`, defaulting to `all`. An invalid explicit string
throws with the accepted values before loading the GGUF, including in builds
without Core ML. An invalid inherited value retains the DAC's legacy warning
and fallback to `all`; the decoder retains its rejection of an invalid inherited value.
`coreml_strict` is `std::optional<bool>`: unset inherits
`PARLER_COREML_STRICT`, while explicit `false` permits fallback even if that
variable is set. Legacy environment booleans use presence, so `0` and `false`
still mean enabled. The environment is read once at engine construction;
both stages and later runtime recovery use the captured policy.
`PARLER_COREML_DISABLE` remains a global construction-time kill switch for
both stages. As before, disabling Core ML while strict mode is enabled fails
construction; explicit `coreml_strict = false` permits ggml. Changing
environment variables does not reconfigure an existing engine.

These fields extend the public C++ options structure. Rebuild native SDK addons
and consumers against the matching library headers; do not mix old addon binaries
with this library or old libraries with a newly built addon.

```cpp
tts_cpp::parler::EngineOptions options;
options.model_gguf_path = model_path;
options.coreml_compute_units = "cpu_and_ane";
options.coreml_strict = false; // permit ggml recovery for this instance
options.coreml_decoder = true; // remains false by default; separate opt-in
tts_cpp::parler::Engine engine(options);
```

`Engine::dac_on_coreml()` reports whether the sidecar is ready;
`SynthesisResult::dac_backend` reports where each call's DAC ran (`ggml`, the
`coreml-*` compute label, or `mixed` across streamed chunks). `parler-cli`
prints it after synthesis and `parler-bench` on its `dac:` line and as
`dac_backend` in `--json-out` (the bench attaches the sidecar like the Engine;
`PARLER_COREML_DISABLE=1` benchmarks the ggml DAC). A sidecar whose
prediction fails is retired (for every engine using it in the process) and
the call's whole range is decoded again on ggml, so no output mixes a failed
call's windows; chunks already streamed are kept. Under the captured strict
policy that call and every later one fail instead. With an enabled prebuilt
sidecar present, `parler::fit_params` returns `Error` with
`coreml-memory-estimate-unavailable` (its resident memory is not projected
until validated); `PARLER_COREML_DISABLE=1` projects the ggml DAC.

How the sidecar decodes: the engine walks the utterance in fixed windows of
the exported width (128 frames, 1.49 s, by default) with
`parler_dac_rf_frames` (11) frames of real context on each side, exactly like
the ggml decode, and a 0/1 frame mask marks frames past the end of the
utterance. The exported graph zeroes the activation at masked frames after
the input convolution, every transposed convolution and every residual sum,
which is the zero padding a decode of the shorter sequence sees, so one
fixed-shape graph serves short utterances and the last window. For the
Neural Engine, sequences longer than 8192 samples are folded into rows of
8192 (each convolution takes its halo from the neighbouring rows), every
transposed convolution runs in its exact phase form, and each Snake computes
`sin^2(alpha x) / alpha` as a range-reduced even polynomial with `1 / alpha`
folded into per-channel coefficients. On the high-rate stages the Snakes,
not the convolutions, dominate Neural Engine time: per-channel constant
multiplies and adds fuse for free, but each tensor-by-tensor op is a full
pass over memory (about 1 ms per 96 x 131072 fp16 activation on an M2, 1.5 ms
for `round`), so the spelling minimises those ops. The conversions alone take
a 256-frame window from 334 to 245 ms.

Placement (`MLComputePlan`, M2, macOS 15): 491 ops on the Neural Engine, 2 on
the CPU (0.4% of the estimated cost).

Accuracy, mini `f32` GGUF, sidecar on the Neural Engine (fp16):

| Check | SNR | max abs |
|---|--:|--:|
| random codes, 265 frames (3 windows), vs ggml CPU | 56.2 dB | 1.7e-3 |
| random codes, 42 frames (one partial window), vs ggml CPU | 56.4 dB | 1.2e-3 |
| `case0` greedy codes vs the HF reference waveform | 49.7 dB | 1.4e-3 |

ggml's GPU DAC is held to 50 dB (or 5e-4 peak); the fp16 Neural Engine sits
just under it, about -57 dBFS of peak error. `test-parler-dac-coreml` gates
the sidecar at 45 dB for speech. For HF references below -60 dBFS RMS, it
instead requires error RMS at or below -70 dBFS and peak error at or below
1e-3 (-60 dBFS); both limits must pass. Random-code checks always retain
the requested SNR gate. All waveform metrics are printed.

Indic validation (M2, 24 GB, macOS 15.7.3, `f32` GGUF, fp16 sidecar,
`CPU_AND_NE`, 128-frame windows, 2026-10-19): random-code decodes against
ggml CPU passed at 56.16 dB (265 frames), 56.33 dB (42 frames), and
53.35 dB (one frame); stitched sub-ranges passed at 56.16 dB. The Gujarati
Yash greedy fixture passed against HF at 51.95 dB. Sampled Hindi Rohit speech
(seed 42, 1.51 s, -22.8 dBFS RMS) passed at 51.34 dB in C++; sampled Gujarati
passed at 52.23 dB in the exporter comparison.

The retained Hindi greedy fixture is mostly DC plus silence (RMS 1.18e-4,
mean 1.07e-4). It scores -1.75 dB relative SNR on the Neural Engine, with
error RMS 1.45e-4 (-76.8 dBFS) and peak error 5.12e-4. This passes the quiet
absolute bounds above. The error is mostly DC from fp16 rounding in the
last Snake activation and output convolution, whose per-channel offsets
cancel to produce silence. The same masked, windowed graph in fp32 scores
49 dB, and ggml CPU scores 80 dB. This is a measured fp16 error floor;
the quiet reference stays in validation alongside sampled speech. Fallback,
strict-mode and whole/streamed Engine backend checks pass. The `cpu_only`
override with an fp16 package scored 43.4 dB on Gujarati; it does not meet
the 45 dB speech gate. No Indic timing results were collected.

Speed, M2 (24 GB, macOS 15), mini `q8_0`,
`parler-bench` greedy for 498 decoder steps (5.70 s of audio), median of 3
warm runs, ms:

| LM backend | DAC on | T5 | prefill | decode | DAC | total | RTF |
|---|---|--:|--:|--:|--:|--:|--:|
| Metal | ggml (Metal) | 18 | 20 | 5,052 | 1,034 | 6,126 | 1.075 |
| Metal | Core ML (Neural Engine) | 26 | 20 | 5,048 | 639 | 5,739 | 1.007 |

These sidecar timing results cover the Metal LM backend; CPU LM timings
with the sidecar were not measured.

One sidecar call per window, standalone (Core ML, M2):

| Window (frames) | Neural Engine | Core ML GPU |
|--:|--:|--:|
| 128 | 128 ms | 115 ms |
| 192 | 194 ms | 170 ms |
| 256 | 245 ms | 239 ms |
| 384 | 413 ms | 340 ms |

The cost is about 1 ms per window frame on either unit, so the window trades
context overhead (22 frames per call) against the unused tail of the last
window; 128 frames also holds a 1 s streaming chunk in one call. The Core ML
GPU is slightly faster than the Neural Engine on an M2; `all` lets Core ML
pick (it places the graph on the Neural Engine) and keeps the GPU free.
Streaming correctness is covered by the same window/range parity checks
and whole/streamed Engine backend checks. Streaming latency was not measured.

Why only the DAC: a decoder step reads about 362 M parameters (724 MB in
fp16). A 24-layer step exported with its self-attention KV cache in an
`MLState` landed 98% on the Neural Engine but took 13.6 ms at 64 tokens of
context and 30.4 ms at 1024 (Core ML GPU 24.9 and 26.3 ms), against 10 ms
for the ggml Metal `q8_0` step, and its first Neural Engine compile took
about 45 minutes. A compressed-weight experiment at context 512 scored
99.56% argmax agreement for fp16 (15.44 ms per prediction) and 96.06% for
per-channel int8 (10.53 ms), against a teacher-forced fp32 twin. Those times
exclude the host's whole-cache state writes and therefore underestimate
the step cost. A 4-bit prototype took 6.26 ms at context 64 but 23.55 ms at
1024; its quality has not been established. None of these results establishes
a speed and quality improvement over Metal at production context lengths.
The T5 encoder runs once per
description and is cached (18 ms on Metal), so a sidecar would not pay for
its 670 MB of fp16 weights.

## Experimental Core ML decoder steps

`EngineOptions::coreml_decoder` (or `--coreml-decoder` on `parler-cli` and
`parler-bench`) opts into a separate `<stem>-decoder.mlmodelc` next to the
GGUF. This requires a `TTS_CPP_COREML` build and macOS 15 / iOS 18. It is
experimental: placing a decoder sidecar beside the model does not enable it
by default. T5 description encoding and prompt prefill still run on ggml.

Decoder contract 2 requires last-axis LayerNorm. Earlier experimental
channel-axis exports collapse to their learned bias on the tested macOS 26 /
M4 CPU path, even though the same package passes on macOS 15 / M2. Re-export
decoder sidecars with the current script; the runtime rejects contract 1 and
missing or incompatible normalization metadata with a re-export message.
The last-axis recipe passes the independent tiny graph and native gates on
both tested systems, including constant and wide-range normalization inputs.
Re-exported mini and Indic models retain ANE placement in the compute plan,
but the int8 decoder still fails the strict quality gate described below.

The decoder-step graph uses read-only self-attention and cross-attention
states, optionally grouping multiple decoder layers in one state tensor with
`--layers-per-state` (a divisor of the layer count). After ggml prefill, the runtime imports the prompt's K/V
and the description's projected cross K/V once. Each prediction emits only
the new K/V rows, which the host writes directly into Core ML's state views
with their actual strides. It does not rewrite the whole cache each step.
When all layers share one state group, the host submits two whole-output
row updates, one each for K and V. Smaller groups retain layer-selected
updates. Both paths write the same row payload and use the same native
stride and dtype checks.
A synthesis starts with fresh zeroed state even when the description is
cached; no state is shared between Engine instances.

```sh
python scripts/parler/export-decoder-coreml.py \
    --gguf models/parler-mini-v1-f32.gguf --ctx 512 --cross-context 128 \
    --weights int8 --heads-fp16 --layers-per-state 24 --compile-dir models
./build/parler-cli --model models/parler-mini-v1-q8_0.gguf \
    --text "Hey, how are you doing today?" \
    --description "A female speaker with a calm, clear voice, close up." \
    --n-gpu-layers 99 --coreml-decoder --max-frames 500 --out out.wav
```

The context must cover prompt tokens plus the intended delayed generation
steps, and cross context must cover the complete description. The runtime
validates the contract version, row layout, mask value, state grouping,
geometry and checkpoint variant (when present in the GGUF), and never clips
either sequence. The variant is a family identifier, not a weight checksum;
keep sidecars paired with their source checkpoint. A
missing, incompatible or failing decoder sidecar falls back to ggml. Requests
whose full generation bound or description exceeds the exported capacity use
ggml for that call, retaining the sidecar for later shorter requests. In
strict mode only the oversized request fails; it does not retire the model. After
a prediction failure it retires the sidecar and rebuilds the ggml cache by
replaying the exact already-consumed token history, preserving sampled code
frames and previously streamed chunks. The captured `coreml_strict` policy makes a
requested sidecar failure fatal, including later calls after retirement;
`PARLER_COREML_DISABLE` disables both Core ML stages.

`Engine::decoder_on_coreml()` reports whether the step sidecar is attached.
`SynthesisResult::decoder_backend` and benchmark JSON `decoder_backend`
report the call's step execution: `ggml`, the `coreml-*` compute label, or
`mixed` after successful Core ML steps followed by ggml recovery. Prefill is
always ggml in this implementation, and the benchmark prefill stage includes
native cache import/seeding. `FitOptions::coreml_decoder` (or the fitter's
`--coreml-decoder`) returns `coreml-memory-estimate-unavailable` rather than
underestimate additional Core ML model/state memory. Decoder-inclusive benchmarks must
include embedding preparation, prediction, logit readback and cache writes.
When integrating the fitter, copy the explicit engine decoder bool to
`FitOptions::coreml_decoder`. The compute-units and strictness options do not
affect the fitter's memory projection or make Core ML memory projectable, so
`FitOptions` has no matching fields. The enabled DAC sidecar path and decoder
opt-in continue to cause the conservative Core ML refusal for every units
selection, including `cpu_only`; strictness does not relax that refusal.
A successful small-model export does not establish production quality or an
end-to-end speed advantage. Uniform 6-bit palettes scored only 93.46% token
agreement against the fp32 rebuild on a 401-frame mini trace, so the faster
6-bit probe is not a recommended quality tier. Per-channel int8 scored
98.06%; integrated quality and speech validation remain required.

To inspect decoder-step costs, add `--profile-coreml-decoder` to a
`parler-bench --coreml-decoder` invocation. Profiling excludes warmup,
prefill and state seeding. Console output and the optional JSON
`coreml_decoder_profile` report requested, successful native and unprofiled
steps for the same measured runs accepted by the pipeline statistics.
Failed attempts, ggml steps and recovery do not contribute native timings;
if none qualify, JSON timing totals are `null`.

The profile separates embedding, input/update preparation, prediction,
cache row handling and logit readback. Session and step totals are inclusive;
their reported remainders reconcile the nested intervals. Row handling
includes output retrieval, validation and state handlers as well as copying.
Deferred synchronization can move work between prediction, row handling and
readback, so judge changes using inclusive step time and unprofiled pipeline
measurements. Profiling adds no clock samples when disabled.

A bounded M2 mini experiment measured prediction at 97.8% of decoder-step
time. Two whole-output row updates reduced measured row handling from
0.155 to 0.031 ms per step in separate profile invocations. An unprofiled
ABBA comparison showed total-time reductions of 1.1% and 6.7% in its two
pairs; the slower final control makes the aggregate 4.0% reduction tentative.
All final PCM16 outputs matched, and a fresh tiny native fixture matched
complete finite fp16 states materialized as fp32 and logits/K/V outputs
across positions and resets. Production mini replay also matched every
float32 logit in two 200-input cases across two resets. Six sampled pairs
(two prompts, seeds 1/42/777) matched complete raw float PCM, sample counts
and sample rates between the frozen baseline and candidate. Generated codes
and explicit EOS/cap reasons were not observed. This is bounded preservation
evidence for those mini inputs, not a universal quality guarantee. See the
[profiling record](history/parler-coreml-profiling-20261020.json) for scope
and individual timings. These battery-powered measurements should not be
compared directly with earlier AC-powered absolute timings.

A separate reshape-only export preserved all learned weights and removed
97 compiled transposes. It changed 8 of 3,600 teacher-forced top-token
choices and reduced total time by 0.7% and 1.6% in two AC-powered ABBA sets.
Both sets missed the predeclared 3% cutoff, so the export was not adopted
and further reference-quality and free-running checks were stopped.

A separate decoder-only `FastPrediction` specialization-hint experiment
preserved every float32 logit across the tested two 200-input histories
(199 native calls each) and two resets. Two unprofiled ABBA sets found total
time 0.14% and 0.30% slower
with the hint (0.22% pooled; baseline drift 0.41%). Both sets missed the
predeclared 5% improvement threshold, so the hint was not adopted. These
CPU-host/CPU+ANE, 498-step fixed-work measurements used unchanged mini q8
weights and decoder/DAC sidecars; they do not measure natural speech or
genuine cold-load latency. See the
[FastPrediction record](history/parler-coreml-fast-prediction-20261020.json)
for controls, individual timings and limits.

Three further isolated M2 experiments also missed the predeclared 5%
total-time improvement threshold. An `Infrequent` reshape hint was 0.91%
slower overall; replacing only the terminal new-K singleton transpose with
a reshape reduced total time by 1.47%; original-head self-QKV projection
fusion reduced it by 1.77% (decode loop 1.92%). Each complete experiment
used two ABBA sets, four CPU host threads, CPU+ANE decoder/DAC and 498-step
fixed work. These are separate within-experiment comparisons; gains are
not additive and the candidates were not compared directly. The initial study
did not adopt a candidate under that cutoff.

The tail and QKV candidates each matched all 800 float32 logit rows in the
tested production histories and resets. QKV still failed the separate tiny
CPU-only bitwise gate; production-history parity under CPU+ANE does not
establish general quality equivalence. Its first timing cohort was interrupted
by system idle sleep and excluded in full. A fresh cohort with temporary
idle-sleep prevention completed without sleep; the assertion was then removed.
All complete cohorts retained eight invocations and stayed within the 5%
baseline-drift limit. The initial study stopped before sampled-speech or ASR
evaluation. See the
[follow-up experiment record](history/parler-coreml-followup-experiments-20261021.json)
for individual measurements, power conditions and retained failures.

The QKV option was subsequently accepted for the tested mini CPU+ANE profile
at the user's request, despite its smaller measured gain. Six paired sampled
utterances (two prompts, seeds 1/42/777, twelve syntheses) then matched every
raw float PCM byte, sample count and rate. This supports those inputs on the
tested setup; the tiny CPU-only mismatch remains, and neither universal
quality equivalence nor an Indic speed gain is established. The original
5% experiment result remains unchanged. See the
[QKV adoption record](history/parler-coreml-qkv-adoption-20261021.json).

To opt into the same serialized transformation, first export an original
mini package with the settings above, then run:

```sh
python scripts/parler/optimize-decoder-coreml.py \
    --package models/parler-mini-v1-decoder.mlpackage \
    --out models/qkv/parler-mini-v1-decoder.mlpackage \
    --compile-dir models/qkv/compiled
```

Both output paths must be new; the package basename must stay the same.
Place the matching GGUF and unchanged DAC sidecar beside the returned
decoder `.mlmodelc`, and use the existing `--coreml-decoder` option. For the
validated profile, select a four-thread CPU host and
`PARLER_COREML_COMPUTE_UNITS=cpu_and_ane` with `PARLER_COREML_STRICT=1`.
The optimizer supports only the reviewed mini context-512/cross-128 geometry
(and its tiny test fixture), per-channel int8 weights, fp16 heads and all
layers per state. It copies the package, concatenates existing raw Q/K/V
codes and scales, and retains original head/current-token/cross-attention
operations. It does not retrace, requantize or change default exports.
The append-only format retains inactive original records: the tested compiled
package grew from 373.9 to 449.5 MB (about 20.2%). Validate output parity and
performance on other compute units before using it there.

The current contract 2 profile uses per-channel int8 weights with fp16
codebook heads (`--heads-fp16`), context 512 and all 24 layers per state.
On M2 / macOS 15, both mini and Indic have 1,211 operations assigned to the
Neural Engine and one transpose to CPU in the compute plan under
`CPU_AND_NE` and `ALL`. This is preferred-device planning, not an execution
trace. Teacher-forced quality runs use CPU T5/prefill and strict
`cpu_and_ane`, with no ggml fallback, repeat drift or invalid distributions
across two 200-step cases per model. Both f32 and the benchmark's q8 host
GGUF were measured; the figures below use the exact fast q8-host profile.

Against independent HF fp32 references, mini live-token argmax agreement is
98.19% / 98.02%, with mean KL 0.000963 / 0.001197. Indic agreement is
99.83% / 98.13%, with mean KL 0.000765 / 0.001058. This is close
to CPU q8 overall, but misses per-codebook criteria and has about three
times the KL error of Metal q8. The unchanged strict gate fails against
both CPU and Metal baseline pairs; default enablement is not justified.

The additional experimental `--ffn-down-fp16` option retains every FFN down
projection in fp16 and composes with `--heads-fp16`. It requires compressed
weights, records the exact selected projections in metadata, and fails the
export if selection or compression does not match the request. Full-model
trials retain the same preferred-device plan and add 95.95 MiB (26.91%) of
compiled storage per model. Mini mean KL improves 18% in case0 but worsens
4% in case1; Indic improves about 29–33%. All strict gates still fail.
The candidate stops the greedy benchmark at 209 steps, so its 2.50-second
time is incomparable with the 498-step controls. These results do not
justify replacing the current profile. Use a separate `--out` path when
comparing experimental exports with an existing sidecar.

A separate EOS-suppressed comparison completes 498 steps in every run:
mean total time is 5.915 s for the FFN-down candidate, 5.177 s for the
current profile, and 6.020 s for CPU decoding with the Core ML DAC.
The candidate costs 14.3% more time than current, with mixed quality and
greater storage. Its 1.7% advantage over the measured hybrid is small;
three calls, desktop activity and one M2 do not establish a general gain.
The original natural-stop result remains separately recorded.

For a separate bounded throughput comparison, `parler-bench` accepts
`--min-new-tokens N --max-frames N` to suppress EOS until the generation
bound. The default still uses the checkpoint's minimum; greedy generation
can stop early. Console and JSON report effective `min_new_tokens` and
`max_length`. Apply the same setting to every configuration, compare actual
step and audio lengths, and retain natural-stop measurements separately.
This controls work length and does not measure speech quality.

An exclusive sequential mini benchmark used the same q8 GGUF, prompt,
four threads, greedy 500-frame bound, one warmup and three measured calls.
Every configuration produced 498 decoder steps and 5.7005 seconds of audio:

| Execution | Mean total (ms) | p95 total (ms) |
|---|---:|---:|
| CPU | 7,671 | 7,714 |
| Metal | 6,278 | 6,306 |
| CPU decoder + Core ML DAC | 5,934 | 6,110 |
| CPU host + Core ML decoder and DAC | 5,120 | 5,126 |

The CPU host with Core ML decoder and DAC has 18.4% lower mean total latency
than Metal and 13.7% lower than CPU decoding with the Core ML DAC, the best
measured hybrid without the Core ML decoder on this prompt. Host T5/prefill, state
seeding, embedding, prediction, K/V row writes and logit readback are
included; cold model loading and compilation are excluded. An initially
noisy CPU run was repeated; both measurements are retained. Three-run p95
is a descriptive sample percentile, not an established tail-latency estimate.
Greedy benchmark audio is degenerate throughput output, as documented in
`bench.cpp`; it is not a speech-quality sample. These are warm measurements
on one M2, not a general speed or speech-quality guarantee.
See the [contract 2 record](history/parler-coreml-contract2-20261019.json)
for identities, per-codebook quality, failed gates and timing ranges.
Bounded sampled checks with the q8 host regenerated all three backends:
18 syntheses and ASR calls across mini and Indic, including three seeds for
one mini prompt. The five scored Core ML samples (two English prompts,
one repeated across three seeds, and one Hindi prompt) have zero normalized
WER/CER. Gujarati scores are unavailable because the recognizer returns the
wrong script on every backend. Three mini seeds show no consistent duration
increase, but do not establish a distribution; EOS/cap labels are inferred
from output length. These are ASR proxies, not listening/MOS or perceptual
equivalence. The strict teacher-forced quality gate remains failed.

Earlier contract 1 validation with 200 teacher-forced inputs on each of two
mini cases found a quality gap in the original int8 export. With CPU f32 prefill
and `cpu_and_ane`, live-token argmax agreement against independent HF fp32
references was 95.80% / 96.20%, compared with 98.02% / 98.19% for q8 on CPU.
Mean KL was 0.00238 / 0.00302, versus 0.00105 / 0.00120 for q8. There were
no invalid distributions or repeated-call drift. This candidate fails the
quality gate; its short-context speedup does not justify default enablement.
These quality runs use CPU f32 prefill, distinct from the q8/Metal-prefill
performance configuration below. See the
[quality record](history/parler-coreml-mini-quality-20261019.json) for class
and per-codebook metrics. Sampled speech equivalence is still unestablished.

The initial per-layer-state mini export failed ANE execution-plan compilation
with error `-14` at 13 or more decoder layers. Grouping all 24 layers into
four state tensors lets the full graph compile and run on the ANE. Preliminary
native probes with 6-bit weights and fp16 heads measured 7.41 ms per step at
context 512, 9.32 ms at 1024, and 12.40 ms at 2048 (including row updates,
excluding prompt/state initialization). These are decoder probe results;
quantized speech quality remains unestablished. Keep the default ggml decoder
for production until the quality gates are satisfied.

The following historical measurements use the contract 1 export on M2;
contract 2 must be measured separately. The integrated mini benchmark used
an Apple M2 with 24 GB memory and the same
q8_0 GGUF, four CPU threads, greedy generation capped at 500 frames, one
warmup and three measured runs. Every run produced 498 decoder steps and
5.7005 seconds of audio. Median warm synthesis times, including T5, prefill,
state seeding, host embedding/logit copies, decoder steps and DAC, were:

| Execution | Decoder loop (ms) | DAC (ms) | Total (ms) | RTF |
|---|---:|---:|---:|---:|
| CPU | 4,994 | 2,630 | 7,763 | 1.362 |
| Metal | 7,040 | 1,089 | 8,153 | 1.430 |
| Metal decoder + Core ML DAC | 7,068 | 705 | 7,847 | 1.376 |
| Core ML int8 decoder, context 512 + Core ML DAC | 4,632 | 682 | 5,377 | 0.943 |
| Core ML int8 decoder, context 3072 + Core ML DAC | 7,673 | 647 | 8,405 | 1.474 |

T5 and prefill ran on Metal in the Core ML rows. The 512-row profile reduced
median synthesis time by 31% versus CPU and 34% versus Metal on this trace;
the 3072-row profile was slower than both. Attention reads the exported
capacity on every step, so export the smallest context that covers the entire
request. Context 512 is deliberately bounded: a larger generation request
falls back to ggml, or fails under strict mode. Loading and cold compilation
are excluded. Runs were sequential, Metal timings varied, and equal step
counts do not establish equal speech quality. See the
[measurement record](history/parler-coreml-m2-20261019.json) for the protocol,
range and p95 values. This is evidence for a short-context speedup, not a
general speed or quality guarantee.


## Snapdragon Hexagon NPU

Parler-TTS runs on the Snapdragon Hexagon NPU (HTP0; measured on a Snapdragon
8 Elite, Hexagon v79) when it is requested explicitly:
`EngineOptions::backend = "hexagon"` or `--backend hexagon` on `parler-cli`,
`parler-bench` and `parler-fit-params`. There is no fallback: construction
fails when HTP0 is missing, and the automatic `--n-gpu-layers` walk never
selects it. The T5 encoder, the decoder (prefill and every step, flash
attention over an F16 KV cache) and the DAC all run on the NPU; the host only
samples. Weights and the fused decoder stacks go into the Hexagon repack
buffer type. HTP has no K-quant matmul and looks up embedding rows only from
F32/F16 tables, so a `q6_k` GGUF is refused at load: use `q8_0`, `f16` or
`f32`. It needs a fabric ggml build with the Hexagon fixes for
fused F32-precision matmuls and the tanh GELU.

The Hexagon DAC runs each snake as one fused node, each transposed
convolution as one GEMM into columns plus `ggml_col2im_1d` (the kernels are
rearranged at load, about 97 MB more for mini), and its convolutions over F16
im2col columns on HMX, which multiplies in F16 anyway; HTP has no fast F32
im2col.

Accuracy against the HF fixtures, `f32` GGUF (`PARLER_TEST_BACKEND=hexagon`
runs the fixture tests on any named backend):

| Check | Hexagon | CPU | OpenCL (Adreno 830) |
|---|--:|--:|--:|
| T5 cross states, relative error | 5.9e-7 | 2.7e-4 | 1.6e-6 |
| decoder argmax agreement, 360 teacher-forced predictions | 359 | 360 | 360 |
| DAC waveform SNR | 70.5 dB | 126.9 dB | 121.7 dB |
| greedy end-to-end token trace | 429 / 429 | 429 / 429 | 429 / 429 |

The F16 HMX products set the DAC's SNR, well inside the 50 dB accelerator bar.
With the `q8_0` GGUF the decoder agrees with the 200-step `f32` reference on
97.25% of 3,600 teacher-forced predictions on Hexagon and 97.72% on CPU.

Galaxy S25, `parler-mini-v1` `q8_0`, `parler-bench` greedy for 398 decoder
steps (4.54 s of audio), median of 3 warm runs, each backend started from
thermal status 0, ms:

| Backend | T5 | prefill | decode | DAC | total | vs OpenCL |
|---|--:|--:|--:|--:|--:|--:|
| CPU (4 threads) | 178 | 168 | 6,369 | 23,563 | 29,602 | 0.28x |
| OpenCL (Adreno 830) | 71 | 70 | 6,314 | 1,838 | 8,294 | 1.00x |
| Hexagon | 47 | 24 | 5,080 | 1,218 | 6,368 | 1.30x |
| Hexagon, `GGML_HEXAGON_OPPOLL=1` | 43 | 19 | 3,889 | 1,165 | 5,116 | 1.62x |

The same for 998 steps (11.51 s of audio), median of 2 warm runs. OpenCL's
decoder slows from 15.9 to 27.3 ms per step as the context grows; Hexagon's
from 12.8 to 13.8 ms:

| Backend | T5 | prefill | decode | DAC | total | vs OpenCL |
|---|--:|--:|--:|--:|--:|--:|
| CPU (4 threads) | 166 | 159 | 20,750 | 60,064 | 81,139 | 0.40x |
| OpenCL (Adreno 830) | 72 | 72 | 27,290 | 4,908 | 32,341 | 1.00x |
| Hexagon | 48 | 24 | 13,776 | 3,022 | 16,870 | 1.92x |
| Hexagon, `GGML_HEXAGON_OPPOLL=1` | 45 | 20 | 11,685 | 3,159 | 14,909 | 2.17x |

A decode step streams about 375 MB of `q8_0` weights through HVX at about
52 GB/s (7.2 ms of 9.5 ms DSP time at 220 tokens of context); attention adds about 3.3 us per
context token. By default the host blocks on the DSP queue during each step
and the CPU clocks down, so the host work between steps runs slowly;
`GGML_HEXAGON_OPPOLL=1` polls instead, which cuts decoding by a quarter but
keeps one core busy and heats the phone like the CPU backend (thermal status
2 after the benchmark, against 0 with the default wait).
