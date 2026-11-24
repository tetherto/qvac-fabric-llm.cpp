# Music prompt-adherence diagnostics

ACE-Step and MiniMax-Music3 can optionally retain generated WAVs and score their
descriptive captions with CLAP. Higher cosine means closer text/audio embeddings;
it is not a percentage, MOS, lyric verifier or musical-quality guarantee. There
is no quality threshold. The existing performance workload remains the default.

## Environment and model

Use Python 3.12 and a separate environment. The resolved dependency pins were
validated on Linux with CPU Torch. macOS uses its native PyPI Torch build;
measure repeatability separately before comparing across scorer platforms.
The shared lock requires SHA-256-verified wheels for CPython 3.12 on Linux x86_64
and macOS 14+ arm64, including Torch; source builds and unpinned dependency installs
are disabled. The comparison scorer uses this same lock.
Linux requires glibc 2.28 or later. The security-updated runtime uses Torch 2.13.0
and Transformers 5.17.0; use the recorded dependency versions when comparing
scores. Rescoring retained ACE-Step audio changed cosine by approximately
2.81e-7 from the Torch 2.6.0 / Transformers 5.5.4 environment on Linux.
See the [dependency capability review](music-alignment-security.md) for the
remaining Socket findings, their dispositions and the limits of runtime tracing.

```sh
python3.12 -m venv .venv-music-alignment
.venv-music-alignment/bin/python -m pip install \
  -r scripts/benchmarks/requirements-music-alignment.txt
.venv-music-alignment/bin/python -m pip check
python3 scripts/benchmarks/prepare-music-alignment.py --models-root bench-models
export MUSIC_ALIGNMENT_PYTHON="$PWD/.venv-music-alignment/bin/python"
```

`music-alignment-model.json` pins `laion/larger_clap_music_and_speech` at
`195c3a3e68faebb3e2088b9a79e79b43ddbda76b`, including SHA-256 for every weight,
tokenizer and processor artifact. The weight is approximately 776 MB. Preparation
downloads and verifies files before generation. `--offline` verifies an existing
cache and fails on missing/corrupt artifacts. The cache directory is
`bench-models/reference-clap/<revision>`. Scoring verifies that directory and loads
strictly offline. No native CLAP GGUF runtime is required.

## Diagnostic driver

Build `music-cli` and/or `mm3-replay` using the repository's normal AudioGen build.
The driver needs `jq`. Supply provisioned generator models or use the existing
ACE-Step registry configuration:

```sh
MUSIC_ALIGNMENT=1 MUSIC_ALIGNMENT_MODEL_DIR=/path/to/acestep-models \
  scripts/benchmarks/run-family.sh --family acestep --runs 3 --warmup 1 \
  --build-dir build --models-root "$PWD/bench-models" \
  --out artifacts/acestep/result.json
```

For MiniMax use `--family minimax` and a directory containing its matching LM and
synthesis GGUFs. The CLI runs `--mode full` with caption and lyrics; replayed tokens
are not a prompt-to-music pilot. S3 registration is optional: prepare the models
from the pinned Hugging Face checkpoint as described below. Unprovisioned models
remain an explicit unavailable result.

`MUSIC_DEVICE=cpu|gpu` selects the requested generation device (default CPU).
CLAP itself runs CPU float32, one thread, irrespective of the generator device.
Generation logs and provenance must be inspected for actual backend/fallback.
`MUSIC_CAPTION`, `MUSIC_LYRICS`, `MUSIC_SEED`, `MUSIC_DURATION` (ACE-Step seconds),
and `MUSIC_MAX_FRAMES` (MiniMax cap) override diagnostic settings. MiniMax's cap
does not guarantee exact duration. The manifest-based pilot below supplies these
fields consistently.

Set `MUSIC_ALIGNMENT_MODEL` to a prepared CLAP directory to bypass automatic
preparation. `MUSIC_ALIGNMENT_MANIFEST` overrides the scorer manifest and
`MUSIC_ALIGNMENT_TIMEOUT` bounds each scorer process (default 300 seconds).

Each invocation owns a unique `result.music.*/` directory. Timed generations
retain `run-N/audio.wav`, `generation.json`, `score.json` and scorer logs.
`result.music-alignment.json` and `result.json.music_alignment` report individual
scores, mean across timed runs, errors and scored/expected coverage. Unavailable
scores are null and excluded from the mean, with partial coverage visible.
Scorer errors do not overwrite successful generation performance.

The diagnostic workload uses meaningful captions and longer output than the
default four-second ACE-Step benchmark. WAV output is included in time-wrapped
generation; CLAP preparation, loading and scoring are excluded. Diagnostic and
default RTF/wall times have different baselines and must not be compared as a
speedup. Workflow dispatch input `music_alignment=true` enables this path only
for music families and installs/caches its optional dependencies.

## MiniMax model preparation

MiniMax inference uses GGUF files, while the source download uses safetensors.
`minimax-model.json` pins the Comfy-Org checkpoint revision, source file sizes and
SHA-256 hashes. The tokenizer is embedded in the LM checkpoint. The separate
upstream MiniMax-Music3 Community License is pinned and retained as `LICENSE`;
it must accompany a prepared model bundle.

Prepare on a large-memory Linux or macOS machine. The current native converter
retains the LM in float32 plus writer buffers. The conservative cold-conversion
guard requires **64 GiB available RAM on Linux** (including cgroup limits), or
64 GiB physical RAM on macOS, and **50 GiB free conversion workspace plus missing
source downloads** (about 22 GiB). Close other large applications on macOS. These
are provisioning budgets, not measured minima or inference requirements. The
guard runs before downloading weights. A smaller inference host can use an
already-prepared bundle without conversion.

The workflow uses `q4_k_m` to reduce inference memory. The existing engine docs
estimate this pair at about 7.6 GB, before runtime buffers. Preparation converts
to f16 first, then quantizes both stages with the native `acestep-quantize` tool;
it never requantizes an already-q8 pair to q4. Build `mm3-replay` and
`acestep-quantize` against the normal speech ggml installation first.

```sh
python3.12 -m venv .venv-minimax
.venv-minimax/bin/python -m pip install -r scripts/benchmarks/requirements-minimax.txt
.venv-minimax/bin/python -m pip check
# Include the shared ggml implementation used by the quantizer in its identity.
quantizer_libraries=()
while IFS= read -r library; do
  quantizer_libraries+=(--quantizer-library "$library")
done < <(find "$PWD/ggml-install/lib" -maxdepth 1 \( -name 'libggml*' -o -name 'libqvac-speech-ggml*' \) \( -name '*.so*' -o -name '*.dylib' \))
.venv-minimax/bin/python scripts/benchmarks/prepare-minimax.py \
  --models-root "$PWD/bench-models" --quant q4_k_m \
  --quantizer "$PWD/build/audiogen/acestep-quantize" \
  "${quantizer_libraries[@]}" \
  > minimax-preparation.json
```

Use `--quant f16` or `--quant q8_0` without `--quantizer` for those tiers. Source
artifacts live under `minimax-sources/`; the final pair, `LICENSE` and
`provenance.json` live under `minimax/<cache-key>/`. Downloads use temporary files
and are checked before publication. The cache identity includes source manifest,
converter/preparer content, locked dependencies, quantizer binary and linked
ggml library identities; every
warm-cache run verifies the final pair and license. `--offline` forbids downloads;
`--verify-only` forbids both downloads and conversion.

On a rebuild, cached source checksums are computed once under the preparation
lock and reused by the download stage.

The workflow installs converter dependencies and builds the quantizer before
restoring the model cache. It obtains the cache key through `--cache-key-only`,
which computes the exact preparation identity without accessing models or
running resource checks. Python patch releases, native build metadata and linked
library changes therefore select a new workflow cache entry that can be saved
after preparation, instead of repeatedly restoring an incompatible bundle.

Transfer the entire final bundle to a smaller host, preserving all four files.
Verify it against the same benchmark source revision without installing converter
dependencies or requiring a matching local quantizer:

```sh
python3 scripts/benchmarks/prepare-minimax.py --models-root bench-models \
  --quant q4_k_m --prepared-dir /path/to/prepared-bundle \
  > minimax-preparation.json
MUSIC_ALIGNMENT=1 MINIMAX_PREPARATION_REPORT="$PWD/minimax-preparation.json" \
  scripts/benchmarks/run-family.sh --family minimax --runs 1 --warmup 0 \
  --build-dir build --models-root "$PWD/bench-models" \
  --out artifacts/minimax/result.json
```

The report names the verified directory and quantization. Preparation failures
produce `preparation-failed` with the failing stage and reason; they never run
generation against a stale directory. Manually provisioned legacy GGUF pairs can
still be supplied through `MUSIC_ALIGNMENT_MODEL_DIR`, but lack the preparer's
source/conversion evidence unless that metadata is also present.

Desktop dispatch with `model_families=minimax,music_alignment=true` attempts
pinned preparation and caches the final q4 bundle. The optional
`minimax_model_dir` input points to an already-prepared bundle on the selected
runner. That directory must survive self-hosted workspace cleanup, so keep it
outside `GITHUB_WORKSPACE`. The input is shared by the dispatch's selected
runners; use separate dispatches for different local directory paths. MiniMax
does not require AWS credentials. Linux requests CPU generation, macOS requests
GPU generation; logs determine the actual backend. Scoring stays on CPU.

To bootstrap an ephemeral hosted Linux runner, first dispatch macOS at the same
source revision with `publish_minimax_models=true`. After successful preparation,
that run publishes `minimax-models-q4_k_m-macos` with the pair, provenance and
license (seven-day retention). Dispatch Linux with `minimax_artifact_run_id` set
to that run ID and leave `minimax_model_dir` empty. The workflow downloads the
bundle using read-only Actions access and verifies it before inference. Imported
bundles retain their original builder provenance; the local generator binary and
backend are recorded separately. Publication is opt-in because the artifact is
several GB. Expired, missing or incompatible artifacts produce explicit
preparation failures.

Standard hosted Linux runners may not have sufficient RAM/disk for cold
conversion; a preparation refusal is a visible missing validation lane, not a
successful benchmark. No downloaded model or measured MiniMax CLAP result is
included with this implementation. Real generation, warm-cache resource
measurements, repeat/mismatch controls and listening observations must be attached
from adequately provisioned hosts. The existing pilot supports a clearly labelled
partial run with `--limit`, `--controls` and `--repeat-generation`; full-corpus and
human-listening calibration remain separate reported coverage.

The pilot fingerprints provisioned generator models once at startup, before any
generation timeout, and retains `model-fingerprints.json` in its output directory.
If models must first be downloaded by the driver, the first invocation creates
that fingerprint file. Subsequent records and repeat-generation controls reuse
the hashes after checking the model directory, file list, sizes, timestamps,
and file identities. A metadata change rejects reuse and requires a new pilot;
these checks do not rehash the contents. Keep model files unchanged for the
pilot's duration. Standalone `run-family.sh` invocations
continue hashing their models once per invocation, including manually provisioned
pairs without preparation metadata.

## Scoring policy

`clap-music-v2` uses caption-only text with an explicit tokenizer-length check;
it never silently truncates a caption. Decode PCM, reject empty/nonfinite/all-zero
audio and all-zero mono cancellation, average channels, and resample with pinned
SciPy `resample_poly` (Kaiser 5.0, constant padding) to 48 kHz. There is no loudness
normalization. RMS, peak, clipping fraction, channel energy and source WAV hash
are diagnostic metadata; quiet music is not assigned an arbitrary failure cutoff.
SciPy decodes PCM8/16/24/32 and floating-point WAV directly; SoundFile, CFFI and
pycparser are not required. Unsupported codecs and truncated WAVs fail explicitly.

Partition from sample zero into consecutive ten-second windows. Zero-pad only
the final short window. Score each full window, retaining its original valid
sample count, and compute a duration-weighted mean. Preserve all window offsets
and scores. Exact-window inputs bypass the processor's random crop path.
Reject zero-norm/nonfinite embeddings and invalid cosine values; clamp only
floating-point excursions within 1e-6 of the mathematical bounds [-1, 1].

The AudioGen comparison harness now shares this policy. Its CLI and JSON shape
remain supported, including its legacy MPS option, but **previous scores need a
new baseline**: random crops, linear resampling and silent caption truncation are
replaced. Legacy `--seed` is accepted but no longer affects scoring. For strictly
reproducible benchmarking use the verified local-model entry point; the older
comparison harness still has its pre-existing model-loading options.

## Pilot and paired reports

`fixtures/music-pilot-v1.json` defines 24 primary captions with three seeds each
(72 outputs), plus three separately reported stress captions (nine outputs).
Its engineering corpus and listening review still require human evaluation.
Create a configuration file with an informative name, original checkpoint
identity, requested `backend` (`CPU` or `GPU`) and nonempty build options:

```json
{"name":"cpu-q8","checkpoint_identity":"source-checkpoint-revision",
 "backend":"CPU","build_options":{"build_type":"Release","gpu":false}}
```

```sh
python3 scripts/benchmarks/run-music-pilot.py --family acestep \
  --configuration cpu-q8.json --build-dir build \
  --model-dir /path/to/acestep-models \
  --scorer-model-dir "$PWD/bench-models/reference-clap/195c3a3e68faebb3e2088b9a79e79b43ddbda76b" \
  --out-dir artifacts/pilot-cpu-q8 --controls --repeat-generation
python3 scripts/benchmarks/summarize-music-pilot.py \
  artifacts/pilot-cpu-q8/pilot.json --out artifacts/pilot-cpu-q8/summary.json
```

Use `--limit 1` for an initial smoke run: the report retains all expected IDs and
marks the unattempted cohort. `--controls` repeats scoring in a fresh process and
scores a deliberately mismatched caption for one output per category.
`--repeat-generation` repeats one successful identical prompt/seed configuration.
Neither control establishes overall listening quality. Scorer and generation
timeouts are separately configurable. Output directories must be new to preserve
prior evidence.

For a candidate, run the identical manifest with a separate configuration and
output directory. Pass `--baseline baseline/pilot.json` to the summarizer. It
rejects incompatible manifests, checkpoints, generation settings and scorer
provenance. Scores are averaged across available seeds within each prompt and
then equally across prompts. Paired deltas follow the same weighting, using only
available matching seeds and reporting missing coverage. Exploratory 95%
bootstrap intervals resample prompts, keeping seeds/windows together (10,000
replicates, fixed seed). They are not release thresholds. Stress prompts are
excluded from the primary aggregate.

FAD remains deferred until a licensed reference corpus and sample-size stability
study exist. Its distribution distance is not an alternative name for per-clip
CLAP cosine. See the [CLAP implementation](https://github.com/LAION-AI/CLAP) and
[generative-music FAD study](https://arxiv.org/abs/2311.01616).

## Tests

```sh
python3 -B scripts/benchmarks/test_music_preparation.py
.venv-music-alignment/bin/python -B scripts/benchmarks/test_music_alignment.py
python3 -B scripts/benchmarks/test-music-alignment-driver.py
python3 -B scripts/benchmarks/test_music_pilot.py
scripts/benchmarks/test-run-family.sh
node audiogen/benchmarks/comparison/tests/clap.test.js
```

Model-free tests establish behavior and failure handling, not musical usefulness.
Real-model scores, generation coverage and any incomplete listening calibration
must be reported separately with their artifacts before claiming a completed
quality pilot.
