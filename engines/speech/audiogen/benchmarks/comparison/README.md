# ACE-Step engine-to-engine comparison

Direct comparison of QVAC `audiogen` (`music-cli`) and
[`ServeurpersoCom/acestep.cpp`](https://github.com/ServeurpersoCom/acestep.cpp)
(`ace-lm` + `ace-synth`). The measured path is the two C++ CLIs only. It does
not use `packages/audiogen-ggml`, an addon, Bare, the SDK, RPC, or a model
downloader inside the timed interval.

This is an **implementation-level** comparison. The engines share ACE-Step 1.5
GGUF layout and sampling defaults, but they do not share a ggml pin, a process
shape, or identical default hyperparameters.

Read [`architecture.md`](architecture.md) before quoting a winner.

## Layout

```text
audiogen/benchmarks/comparison/
  config.json                 # shared run configuration
  prompts/manifest.json       # deterministic prompts, lyrics, seeds, durations
  adapters/qvac.js            # music-cli adapter
  adapters/acestep.js         # ace-lm + ace-synth adapter
  lib/                        # parse, backend, audio, aggregate, report
  run-comparison.js
  generate-report.js
  capture-environment.js
  score-clap.js                # optional CLAP post-pass
  quality/clap_score.py
  tests/
  reports/                    # reviewed platform reports
  models/                     # gitignored GGUFs
  vendor/acestep.cpp          # gitignored external checkout
  out/                        # gitignored generated results
```

## Dependencies

Install through the host package manager, then review local source before
building:

- CMake >= 3.20, a C++17 compiler, git, Node.js 20+
- Python is optional. CLAP scoring needs Python 3.12 and the pins in
  `quality/requirements.txt`. Fréchet Audio Distance is not run by default
  because this harness does not ship a licensed reference corpus.
- CUDA builds require the NVIDIA CUDA toolkit. RTX 50-series native (`sm_120`)
  compilation requires CUDA 12.8 or newer; older toolkits can use
  forward-compatible PTX as described below.

Do not pipe remote installers into a shell.

## Model acquisition

Download the same four Serveurperso GGUFs into `models/`. These files are data,
not executables. Record SHA-256 hashes in the report.

| Role | File | Quant |
|---|---|---|
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | q8_0 |
| LM | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | q8_0 |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | q8_0 |
| VAE | `vae-BF16.gguf` | bf16 |

Source: https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF

The 0.6B LM is used instead of the acestep.cpp README default 4B LM so both
engines run the QVAC-validated combination. Put all four files in one
directory and point both CLIs at it.

Conversion from safetensors, if you must rebuild, uses
`scripts/convert-acestep-to-gguf.py` plus the `acestep-quantize` binary (see
the engine README's Model setup), which follow the acestep.cpp GGUF layout
both engines load.

## Build QVAC music-cli

From `engines/speech` in the fabric repository. Review
`https://github.com/tetherto/qvac-ext-ggml` (`speech` branch) before compiling;
the VAE needs `ggml_snake` and `ggml_col2im_1d`.

The reviewed ggml revision for the committed reports is
`0a76e3ed969781da6de41d6c9a1c3fc471c0978b`. Check it out after cloning when
reproducing those reports. A newer revision is a new comparison input and must
be recorded in the platform verification report.

CPU (Metal disabled):

```sh
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src-cpu
cmake -S ggml-src-cpu -B ggml-src-cpu/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_METAL=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install-cpu
cmake --build ggml-src-cpu/build -j
cmake --install ggml-src-cpu/build
cmake -S audiogen -B audiogen/build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$PWD/ggml-install-cpu -DAUDIOGEN_BUILD_TESTS=ON
cmake --build audiogen/build-cpu --target music-cli -j
```

Metal:

```sh
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src-metal
cmake -S ggml-src-metal -B ggml-src-metal/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install-metal
cmake --build ggml-src-metal/build -j
cmake --install ggml-src-metal/build
cmake -S audiogen -B audiogen/build-metal -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$PWD/ggml-install-metal
cmake --build audiogen/build-metal --target music-cli -j
```

CUDA:

```sh
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src-cuda
git -C ggml-src-cuda checkout 0a76e3ed969781da6de41d6c9a1c3fc471c0978b
cmake -S ggml-src-cuda -B ggml-src-cuda/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_METAL=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install-cuda
cmake --build ggml-src-cuda/build -j
cmake --install ggml-src-cuda/build
cmake -S audiogen -B audiogen/build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/ggml-install-cuda
cmake --build audiogen/build-cuda --target music-cli -j
```

On an RTX 50-series GPU with CUDA older than 12.8, add
`-DCMAKE_CUDA_ARCHITECTURES=89-virtual` to both ggml CMake configurations.
The NVIDIA driver JIT-compiles that PTX for the installed GPU. This is valid
for an implementation comparison, but label the report as PTX rather than a
native `sm_120` build. Set `CUDA_VISIBLE_DEVICES` before environment capture
and generation when the host has more than one GPU.

Without `ACESTEP_QVAC_CLI` the harness looks for `music-cli` in the fabric root build (`build/bin`), then in a standalone speech build (`engines/speech/build/audiogen`), then in `audiogen/build`, `build-cpu` and `build-metal`. Set `ACESTEP_QVAC_CLI` to the `music-cli` you intend to measure.

## Build acestep.cpp

Checkout next to this comparison directory (`vendor/acestep.cpp`) or set
`ACESTEP_CPP_DIR`. Initialise its ggml submodule. Review the source, then
build. On macOS a default build enables Metal; a CPU comparison requires a
separate tree with Metal off.

The reviewed acestep.cpp revision for the committed reports is
`9761469d95fc204b5468623c68a1a2203e50b1f9`. Check it out before initialising
its pinned ggml submodule when reproducing those reports.

```sh
git clone https://github.com/ServeurpersoCom/acestep.cpp.git vendor/acestep.cpp
git -C vendor/acestep.cpp checkout 9761469d95fc204b5468623c68a1a2203e50b1f9
git -C vendor/acestep.cpp submodule update --init
cmake -S vendor/acestep.cpp -B vendor/acestep.cpp/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build vendor/acestep.cpp/build-cpu --target ace-lm ace-synth -j
cmake -S vendor/acestep.cpp -B vendor/acestep.cpp/build-metal \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build vendor/acestep.cpp/build-metal --target ace-lm ace-synth -j
cmake -S vendor/acestep.cpp -B vendor/acestep.cpp/build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF
cmake --build vendor/acestep.cpp/build-cuda --target ace-lm ace-synth -j
```

Use the same `CMAKE_CUDA_ARCHITECTURES` value for both engines.
Set `ACESTEP_CPP_LM` and `ACESTEP_CPP_SYNTH`.

## Run

```sh
cd audiogen/benchmarks/comparison
node --test tests/*.test.js
ACESTEP_QVAC_GGML_DIR=/path/to/ggml-src-cuda node capture-environment.js
node run-comparison.js --dry-run --backend cpu
ACESTEP_QVAC_CLI=... ACESTEP_CPP_LM=... ACESTEP_CPP_SYNTH=... \
  node run-comparison.js --backend cpu
ACESTEP_QVAC_CLI=... ACESTEP_CPP_LM=... ACESTEP_CPP_SYNTH=... \
  node run-comparison.js --backend metal
ACESTEP_QVAC_CLI=... ACESTEP_CPP_LM=... ACESTEP_CPP_SYNTH=... \
  node run-comparison.js --backend cuda
```

Interrupted runs resume from `out/rounds/`. `--force` rebuilds every round.
`--prompts instrumental-folk,short-drone` selects a subset.
`ACESTEP_COMPARE_ORDER=random|alternate|qvac-first|acestep-first` and
`ACESTEP_COMPARE_COOLDOWN_MS` control order and pauses.
Set `ACESTEP_QVAC_GGML_DIR` during environment capture so the report records
the exact external ggml revision. The acestep.cpp revision and its ggml
submodule are discovered automatically.

`node generate-report.js out/cpu.json` regenerates Markdown from JSON. Generated
reports include Mermaid bar charts for overall generation time, real-time
factor, peak memory, CLAP score, and per-prompt generation time. The same
visualizations are emitted for every platform and backend report.

## Optional CLAP post-pass

Generation commands do not change. After WAVs exist, score them in a separate
process so CLAP cannot affect RTF.

Review `quality/requirements.txt`, then:

```sh
python3 -m pip install -r quality/requirements.txt
node score-clap.js --backend cpu
node score-clap.js --backend metal
node score-clap.js --backend cuda
```

First run downloads `laion/larger_clap_music_and_speech` into the Hugging Face
cache (data, not a shell installer). Default text is the manifest **caption
only**. Override with `ACESTEP_CLAP_TEXT_POLICY=caption+lyrics`.
`--force` rescores. `--include-warmup` scores warm-up WAVs too.
`clap.elapsedMs` is scorer time and is not added to generation time.

Saved scores retain the scorer's `clap.policyVersion`. The post-pass reuses only
finite scores from the current policy (`clap-music-v2`); legacy scores without a
version and scores from other policies are automatically rescored. `--force`
also rescans scores already on the current policy. Reports label the policy and
reject mixed policies across engines or prompts, including legacy/unversioned
scores mixed with versioned scores. Rescore the saved WAVs before comparing such
results. A report containing only legacy scores is labeled `legacy/unversioned`.

Copy reviewed JSON/Markdown into `reports/<target>/` and complete the
`verification-report.md` checklist described in `reports/README.md`. Keep
large WAVs out of git; record checksums.

## Interpreting metrics

- **generation_ms**: QVAC `[acestep-timing]` total when present; otherwise
  process wall. For acestep.cpp, LM wall plus synth wall (or parsed load/gen
  lines when the CLIs print them).
- **e2e_ms**: process wall, including WAV write. QVAC is one process;
  acestep.cpp is two.
- **RTF**: generation seconds / generated audio seconds. Lower is faster.
- **init_ms**: QVAC is lazy-load, so init is folded into the first stage of
  `generate()`. acestep.cpp load lines are summed when logged.
- **CLAP**: cosine similarity of LAION CLAP text vs audio embeddings, from
  `score-clap.js` after generation. Median is reported; `n/a` until the
  post-pass runs. Higher is closer to the caption in CLAP space, not a MOS.

Medians and distributions are reported, not only the fastest run.

## Adding another platform

1. Set the GPU visibility variables, then capture the environment with
   `capture-environment.js`.
2. Build CPU and the platform GPU backend for **both** engines.
3. Run `--backend cpu` and the GPU name only if logs prove both used that GPU.
4. Store reports and the prompt manifest under `reports/<target>/`.
5. Complete `verification-report.md`, including revisions, build flags,
   environment, backend evidence, failures, and known limitations.
