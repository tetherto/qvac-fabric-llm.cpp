# audiogen engine: pipeline and stages

Part of the [audiogen engine documentation](../README.md).

## Pipeline

```
caption + lyrics (+ bpm, key, time signature, language)
   |
   v
LM (acestep-5Hz-lm, Qwen3 causal) -> metadata + acoustic codes
FSQ detokenizer                    -> DiT context latents
text encoder (Qwen3-Embedding)     -> prompt embeddings
condition encoder                  -> cross-attention states
DiT (flow matching, Euler)         -> 64-channel acoustic latent
VAE (AutoencoderOobleck)           -> 48 kHz stereo PCM
```

The VAE upsamples by 1920, so `T_audio = T_latent * 1920`. Latents are time-major, `latent[t * 64 + c]`.

### MiniMax-Music3

```
caption + lyrics -> byte-exact Qwen prompt -> Qwen3 LM
                 -> RVQ depth decoder -> condition encoder
                 -> windowed flow DiT -> DAC vocoder -> 44.1 kHz stereo PCM
```

MiniMax uses two GGUF files: `mm3-lm-<quant>.gguf` for the Qwen3 global LM and
`mm3-synth-<quant>.gguf` for the RVQ depth decoder, condition encoder, flow DiT,
and vocoder. Set `EngineOptions::model_dir`, or provide `lm_model_path` and
`synth_model_path` explicitly. Explicit paths may use different quantizations;
the engine validates compatibility from the GGUF contents. Automatic discovery
requires matching quantizations, including when only one path is explicit. It
matches filenames case-insensitively and prefers `q8_0`, then `f16`, then `bf16`, then the k-quant
variants `acestep-quantize` can emit in descending fidelity (`q6_k`, `q5_k_m`,
`q5_k_s`, `q4_k_m`, `q4_k_s`, `q3_k_l`, `q3_k_m`, `q3_k_s`, `q2_k`), then
`f32` (a dequantized diagnostic pair, only picked when nothing else matches),
and rejects duplicate candidates. The engine is desktop-only.
It runs on CPU by default; `EngineOptions::device` (or, when that is empty,
the `MM3_DEVICE` environment variable) accepts `cpu`, `gpu`, or `auto`. `gpu`
requires a usable GPU backend and fails engine creation without one; `auto`
takes a GPU when available and otherwise falls back to the CPU. On a GPU the
weights and graphs live on the first usable backend the ggml registry offers
(CUDA, Vulkan, Metal, ...) and a CPU backend backs any unsupported op; the
full model pair must fit in device memory (~22 GB for the f16 pair, 12.7 GB
for `q8_0`, 7.6 GB for `q4_k_m`). The LM's KV cache and compute buffers are
released after the AR stage and rebuilt on the next generation, and the
vocoder decodes in overlapping tiles to bound decode memory. Use
`tts_cpp::minimax::fit_params` to project the full workload footprint before
loading; see [memory-fit.md](memory-fit.md). The flow DiT runs with flash attention by default on a GPU
backend (off on CPU); set `MM3_DIT_NO_FLASH=1` to force it off. The LM runs
flash attention by default on CUDA and Vulkan, where its teacher-forced logits
were measured against the CPU rendering, and keeps the explicit-softmax path on
every other backend; set `MM3_LM_FLASH=1` to opt another GPU backend in, or
`MM3_LM_NO_FLASH=1` to force it off (this wins if both are set). One MiniMax
engine instance may be active at a time because its compute graphs are shared.
The weight-free `test-minimax-metal-ops` regression compares the 4096-channel
condition projection and every DAC transposed-convolution stride against CPU on
Metal.

Backend parity measurements and optimization details are in
[performance.md](performance.md#minimax-native-backend-validation-and-optimization).
OpenCL remains unmeasured for MiniMax.

The frame rate, maximum frame count, CFG scale, and output sample rate come from
GGUF metadata. Current converted files specify 25 frames per second, at most
9000 frames, CFG 1.7, and 44100 Hz output. The flow step count is the one
default the engine does not take from the file: `mm3.flow.steps` is validated
but the engine uses its own recommendation of 20 unless `inference_steps` is set
on the request, because the metadata value is a fixed converter constant rather
than a property of the checkpoint.

Download the Comfy-Org single-file safetensors checkpoint (the converter's
preferred source; requires the Hugging Face CLI,
`pip install -U huggingface_hub`); conversion needs `numpy` and `gguf`.
Run these commands from the repository root:

```sh
audiogen/scripts/download-minimax-music3.sh --dir checkpoints/MiniMax-Music3
python3 audiogen/scripts/convert-minimax-music3-to-gguf.py \
  --src checkpoints/MiniMax-Music3 --out models/minimax --quant f16
```

An already-downloaded checkpoint converts the same way: point `--src` at the
local directory. The converter emits the two-file contract and writes the
MiniMax-Music3 Community License identifier into both files. Ship the upstream
model license with converted weights.

For a smaller footprint, quantize the `f16` pair to `Q4_K_M` (or `Q4_K_S`) with
`acestep-quantize` (built with `AUDIOGEN_BUILD_EXECUTABLES`, shared with the
ACE-Step engine):

```sh
./build/audiogen/acestep-quantize models/minimax/mm3-lm-f16.gguf \
                                          models/minimax/mm3-lm-q4_k_m.gguf Q4_K_M
./build/audiogen/acestep-quantize models/minimax/mm3-synth-f16.gguf \
                                          models/minimax/mm3-synth-q4_k_m.gguf Q4_K_M
```

This quantizes the LM and, within the synth file, both the flow DiT and the RVQ
depth decoder to the chosen k-quant, with two exceptions inside the depth
decoder: `depth.pos_embd.weight`, which the depth graph views raw and which
stays F32, and `depth.audio_embd.weight`, which is read with `get_rows` and is
held at `q8_0` because CUDA's `get_rows` has no k-quant path (a k-quant table
would silently fall back to a host copy on every depth step). The condition
encoder and vocoder keep their converted (F16/F32) precision. Quantize from the `f16` pair, not `q8_0` —
`acestep-quantize` only requantizes BF16/F16/F32 source tensors, so an
already-`q8_0` tensor passes through untouched.

`mm3-replay` (built with `AUDIOGEN_BUILD_EXECUTABLES`) is the MiniMax CLI and
parity harness:

```sh
./build/audiogen/mm3-replay --models models/minimax --out mm3-out --mode full \
                                    --caption "warm lo-fi beat with vinyl crackle" \
                                    --lyrics "[Instrumental]" --max-frames 300
```

`--mode full` runs the pipeline from a caption/lyrics pair and records the
prompt token ids it used (`tokens.i32`) next to the emitted semantic/acoustic
codes, so a full run's output directory doubles as a replay input set.
`--mode replay` forces recorded prompt tokens, semantic/acoustic codes, and
per-window initial noise through the native pipeline and dumps the per-window
latents, frame hiddens, and stitched audio for 1:1 comparison against the
official implementation, and `--mode condcheck` verifies the DiT emits
byte-identical velocities across repeated computes. `--dump-iters N` (optionally with
`--dump-dir <dir>`) additionally writes the first N AR iterations' semantic
logits, CFG-guided logits, LM hidden states, feedback embeddings, and depth
hiddens as raw f32 files; because the LM and depth decoder still run under
forced tokens, replay-mode dumps are teacher-forced and directly comparable
across quantization levels and backends. `test-minimax-quality` (built with
`AUDIOGEN_BUILD_TESTS`, skipped unless `AUDIOGEN_TEST_MINIMAX_MODELS_DIR` is
set) is the model-backed regression: it asserts DiT determinism, that the
tiled vocoder decode matches a single-shot decode (bit for bit on the CPU,
within rounding on a GPU) and that two shifted tiles agree bit for bit over
their shared interior on any backend, and that a short generation's final
flow latents land on the learned data manifold instead of stalling near the
Gaussian noise they started from.
`test-minimax-quality-q4` runs the same checks against the pair named by
`AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR`.

### ACE-Step audio editing

When `GenerateParams::edit_plan` is non-empty, the engine takes the editing
path instead: it VAE-encodes `source_audio`, skips the LM and FSQ
detokenizer, executes each `RepaintParams` or `FlowEditParams` operation in
order, and VAE-decodes the final latent.

### ACE-Step LRC generation

With `GenerateParams::generate_lrc` set, the engine aligns the request lyrics
with the generated audio and returns karaoke-style LRC timestamps in
`GenerateResult::metadata.lrc`, plus an alignment confidence score in
`metadata.lyrics_score` (`[0, 1]`). After sampling, one extra DiT forward at
the final timestep runs with explicit softmax on the validated lyric
cross-attention heads (the graph stops at the deepest captured layer), the
captured matrices are converted from the ggml column-major layout, and DTW
aligns each lyric line with the audio timeline. Requires lyrics — with Simple
Mode the LM-written lyrics are aligned — and the official 24-layer/16-head
DiT; the capture path is fully separate from the sampling graph, so normal
inference is untouched when disabled.

### ACE-Step Simple Mode

With `GenerateParams::simple_mode` set, `caption` is a short natural-language
query ("a romantic modern salsa with male lead vocals for a wedding") and the
LM inspire pass composes the full request before synthesis: a detailed
caption, complete lyrics, and every metadata field left unset — BPM,
key/scale, time signature, vocal language, and duration when `duration <= 0`.
Fields the caller sets are forced through the metadata FSM and kept. `lyrics`
must be empty (the LM writes them) or `[Instrumental]`, which forwards the
instrumental hint to the LM. In C++, `GenerateParams::lyrics` defaults to
`[Instrumental]`: call `params.lyrics.clear()` to request LM-written vocals.
The CLI clears lyrics automatically with `--simple` when `--lyrics` is omitted.
Simple Mode requires the plain `text2music` task
with no pre-supplied `audio_codes`; the composed request is reported back in
`GenerateResult::metadata`. The inspire pass emits `lm` progress ticks and
honors cancellation like every other stage.

### ACE-Step Query Rewriting

With `GenerateParams::rewrite_query` set, the LM FORMAT pass reworks a full
request before synthesis: the caption is rewritten into a detailed musical
description and the lyrics are regenerated preserving their content, with any
unset metadata filled through the same FSM the inspire pass uses. Unlike
Simple Mode — which expands a bare query and writes lyrics from scratch —
Query Rewriting takes caption AND lyrics as input, so both are required, and
the lyrics must be real lyric text: an `"[Instrumental]"` request belongs to
Simple Mode, which forwards the instrumental hint. The two modes are mutually
exclusive. The rewritten request is reported back in
`GenerateResult::metadata`. Faithful rewriting needs the 1.7B LM
(`acestep-5Hz-lm-1.7B`): the 0.6B drifts genre, voice, and language.

### ACE-Step quality scoring

With `GenerateParams::compute_quality_score` set, the generated audio codes
are teacher-forced back through the LM "understand" prompt and the request is
scored: caption and lyrics earn a normalized PMI (their mean log-prob given
the codes against the same text under a no-input prompt) and each set
metadata field earns a rank-weighted top-k recall of its YAML line. The
weight-normalized global score (caption 0.5, lyrics 0.3, metadata 0.2) lands
in `GenerateResult::metadata.quality_score` in `[0, 1]` with the per-condition
breakdown in `quality_report` — made for ranking a batch of takes (stem tasks
on the base DiT vary strongly by seed) and keeping the best. Scoring runs
extra LM forwards after code generation, reports `score` progress ticks, and
honors cancellation; it requires the LM code path, so cover / lego tasks and
the audio edit path reject it.

### ACE-Step audio understanding

`Engine::understand` runs the reverse pipeline: 48 kHz stereo audio is
VAE-encoded, the FSQ tokenizer (an attention pooler in the DiT GGUF under
`tokenizer.*`) turns the latents into 5 Hz semantic codes, and the LM
"listener" describes them — a descriptive caption plus BPM, key/scale, time
signature, duration estimate, and vocal language, decoded through the same
FSM-constrained metadata block generation uses. `UnderstandResult` also
returns the recovered codes, directly reusable as
`GenerateParams::audio_codes`. Lyrics are intentionally NOT reported: the
LM's transcription hallucinates on real songs, so the field is unsupported.
An optional `vocal_language` hint is forced through the FSM instead of the
LM's guess. Stages report as `source`, `tok`, and `understand` (unknown
total), all cancellable.

## Model stages

Four GGUF files provide six runtime weight sets. The DiT GGUF contains the
condition encoder and FSQ detokenizer weights as well as the DiT weights; the
other files provide the text encoder, LM, and VAE. Point `--models <dir>` at a
directory and the engine classifies files by filename stem, or pass explicit
per-file paths, which always win over the scan.

| Stage | Upstream model | Filename stems matched | Flag |
|---|---|---|---|
| Text encoder | Qwen3-Embedding | `embedding`, `text-enc`, `textenc` | `--text` |
| LM | ACE-Step 5 Hz LM (Qwen3 causal) | `-lm`, `lm-`, `_lm`, `ace-lm`, `5hz-lm` | `--lm` |
| DiT + condition encoder + FSQ detokenizer | ACE-Step v1.5 diffusion transformer | `turbo`, `dit`, `v15`, `sft` | `--dit` |
| VAE | AutoencoderOobleck | `vae` | `--vae` |

The most specific stems (`embedding`, `vae`) are tested first so no short token such as `lm` can claim an unrelated file.

### DiT variants

Detected from the `acestep.is_turbo` GGUF key (absent means base or sft) and
`general.name` (an "sft" substring marks the sft fine-tune).

| Variant | Default steps | Default shift | Default guidance | Notes |
|---|--:|--:|--:|---|
| turbo | 8 | 3.0 | 1.0 | fastest; guidance-distilled, CFG overrides are clamped to 1.0 |
| base | 50 | 1.0 | 7.0 | CFG via APG; the only variant supporting the lego stem task |
| sft | 50 | 1.0 | 7.0 | CFG via APG; no stem tasks |

### Multi-Track (lego)

`task_type: "lego"` generates a new instrument layer that follows
`source_audio` and returns only that stem, trimmed to the source length for
sample-for-sample mixing. It requires a base DiT (turbo and sft are rejected)
and a `track` name — one of vocals, backing_vocals, drums, bass, guitar,
keyboard, percussion, strings, synth, fx, brass, woodwinds:

```bash
music-cli --models <dir> --task lego --track guitar \
  --caption "clean electric guitar with syncopated fills" \
  --lyrics "[Instrumental]" --src-audio song.wav --out stem.wav
```

`--guidance F` overrides the DiT guidance scale on base/sft (0 = auto).

Weights load quantized. `f32`, `f16`, and `bf16` are handled for norms and biases, and the detokenizer's `special_tokens` may be `q8_0`. The stage GGUFs are built in tree by `scripts/convert-acestep-to-gguf.py` and the `acestep-quantize` binary (see [Model setup](#model-setup)), both adapted from [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp), the upstream C++/ggml implementation this port follows, which also publishes pre-quantized GGUFs at [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

### Model setup

Three four-file combinations are validated. All use the same fixed text
encoder, LM, and VAE; select one DiT:

| Role | Filename | Quantization | Approximate size | Variants |
|---|---|---|---:|---|
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `q8_0` | 748 MB | all |
| LM | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | `q8_0` | 677 MB | all |
| VAE | `vae-BF16.gguf` | `bf16` | 322 MB | all |
| DiT | `acestep-v15-turbo-Q4_K_M.gguf` | `q4_k_m` | 1.35 GB | `turbo-q4` |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | `q8_0` | 2.37 GB | `turbo-q8` |
| DiT | `acestep-v15-sft-Q8_0.gguf` | `q8_0` | 2.37 GB | `sft` |

Build these files locally in three steps: download the upstream safetensors
checkpoints from Hugging Face, convert each stage to a self-contained BF16
GGUF, and quantize the text encoder, LM, and DiT (the VAE always stays BF16).
The download script needs the Hugging Face CLI
(`pip install -U huggingface_hub`), the converter needs
`pip install numpy gguf`, and `acestep-quantize` builds with
`AUDIOGEN_BUILD_EXECUTABLES`. When built from the repository root, binaries
live under `./build/audiogen/`; a standalone `audiogen` build
places them under `./build/audiogen/` instead.

Run these commands from the repository root:

```sh
audiogen/scripts/download-acestep-checkpoints.sh --dir checkpoints
python3 audiogen/scripts/convert-acestep-to-gguf.py --checkpoints checkpoints --out models/bf16

mkdir -p models/acestep
./build/audiogen/acestep-quantize models/bf16/Qwen3-Embedding-0.6B-BF16.gguf \
                                        models/acestep/Qwen3-Embedding-0.6B-Q8_0.gguf Q8_0
./build/audiogen/acestep-quantize models/bf16/acestep-5Hz-lm-0.6B-BF16.gguf \
                                        models/acestep/acestep-5Hz-lm-0.6B-Q8_0.gguf Q8_0
./build/audiogen/acestep-quantize models/bf16/acestep-v15-turbo-BF16.gguf \
                                        models/acestep/acestep-v15-turbo-Q4_K_M.gguf Q4_K_M
cp models/bf16/vae-BF16.gguf models/acestep/
```

Quantize the turbo DiT to `Q8_0` instead for the `turbo-q8` combination; pass
`--sft` to the download script and quantize `acestep-v15-sft-BF16.gguf` to
`Q8_0` for the `sft` one. Keep one DiT per model directory rather than relying
on `--models` alone: directory iteration does not define which matching DiT is
selected, so with several present pass the intended file explicitly with
`--dit` / `EngineOptions::dit_model_path`. The external
`Serveurperso/ACE-Step-1.5-GGUF` files above have separate provenance and are
not the combinations validated by QVAC.
