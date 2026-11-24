# parakeet engine: build

Part of the [parakeet engine documentation](../README.md).

## Build modes

### Standalone engine build

The standalone build owns its ggml dependency. From the repository root, stage
the fabric ggml (the `ggml/` tree of
[qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp)) as
`parakeet/ggml`, then configure the engine directly. The setup script
pins the commit (`GGML_REF`, kept equal to the CI lanes' pin); check out the
`qvac-fabric` port's tag instead when reproducing a packaged build:

```bash
parakeet/scripts/setup-ggml.sh
cmake -S parakeet -B build-parakeet -DCMAKE_BUILD_TYPE=Release
cmake --build build-parakeet -j
```

Standalone defaults enable the library, CLI, tests, and examples. Outputs are
under the selected build directory, for example:

```text
build-parakeet/parakeet
build-parakeet/live-mic
build-parakeet/live-mic-attributed
build-parakeet/test-*
```

`setup-ggml.sh` does not apply local patches. Backend filename-prefix loading,
non-Adreno OpenCL support, and the OpenCL program-binary cache live as commits
in the fabric ggml.

### Umbrella speech-stack build

The repository-level build requires an installed fabric ggml (the
`qvac-fabric` package, or a local install as in the
[umbrella build guide](../../docs/BUILD.md#build)) and forces
`PARAKEET_USE_SYSTEM_GGML=ON`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build -j
```

The Parakeet CLI is then `build/parakeet/parakeet`. The umbrella
defaults follow `SPEECH_BUILD_EXECUTABLES` and `SPEECH_BUILD_TESTS`, and force
`PARAKEET_BUILD_EXAMPLES=OFF`. Enable examples only in a direct engine build.

### AMD XDNA dependency

XDNA lives in Fabric LLM's GGML. The speech GGML migration is tracked in
[Fabric Speech #390](https://github.com/tetherto/qvac-fabric-speech.cpp/pull/390).
The migration's initial LLM revision predates XDNA; the consumed revision must
contain both the XDNA backend and the speech operations and memory APIs.
This feature does not add XDNA to the older `qvac-ext-ggml@speech` dependency.

For a standalone validation, install the compatible GGML first and use the
existing system-GGML build mode. XDNA additionally requires XRT and an IRON
Python environment (`mlir_aie` and `llvm-aie`) to compile its kernel artifacts.
Follow that backend's `ggml/src/ggml-xdna/README.md` for toolchain and runtime
requirements. Keep the backend library and artifacts from the same revision.

```bash
cmake -S /path/to/qvac-fabric-llm.cpp/ggml -B build-ggml-xdna \
  -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_XDNA_GEMM_PYTHON=/path/to/iron-env/bin/python \
  -DCMAKE_INSTALL_PREFIX=/path/to/ggml-xdna-install
cmake --build build-ggml-xdna -j
cmake --install build-ggml-xdna

cmake -S parakeet -B build-parakeet-xdna \
  -DCMAKE_BUILD_TYPE=Release -DPARAKEET_USE_SYSTEM_GGML=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-xdna-install
cmake --build build-parakeet-xdna -j
```

If artifacts are staged separately, set `GGML_XDNA_KERNEL_DIR` to their directory.
See [backends](backends.md#amd-xdna-npu) for model formats and runtime selection.

### CMake options

| Option | Standalone default | As an umbrella subdirectory | Effect |
|---|---:|---:|---|
| `PARAKEET_BUILD_LIBRARY` | `ON` | `ON` | Build `qvac-parakeet`; linkage follows `BUILD_SHARED_LIBS`, otherwise static |
| `PARAKEET_BUILD_EXECUTABLES` | `ON` | inherited from `SPEECH_BUILD_EXECUTABLES` | Build target `parakeet-cli`, output name `parakeet` |
| `PARAKEET_BUILD_TESTS` | `ON` | inherited from `SPEECH_BUILD_TESTS` | Build and register test harnesses |
| `PARAKEET_BUILD_EXAMPLES` | `ON` | `OFF` | Build `live-mic` and `live-mic-attributed` |
| `PARAKEET_INSTALL` | `ON` | `ON` | Generate library, headers, CMake package, and pkg-config install rules |
| `PARAKEET_USE_SYSTEM_GGML` | `OFF` | `ON` | Use `find_package(ggml)` instead of `parakeet/ggml` |
| `PARAKEET_COREML` | `OFF` | `OFF` | Apple-only CTC/IndicConformer, Unified RNN-T, TDT, EOU, Nemotron, and tagged Sortformer encoder sidecars, the Neural Engine TDT decoder sidecar, and Nemotron 3 Diarization and Sortformer v2.1 diarizer sidecars |
| `PARAKEET_OPENMP` | `ON` | `ON` | Link OpenMP when available; auto-disabled on Windows non-MinGW unless explicitly overridden |
| `PARAKEET_FLASH_ATTN` | Metal, CUDA and Vulkan `ON`, otherwise `OFF` | same | Fused encoder attention with the rel-pos bias folded into the mask; selected per backend at load, the CPU path always keeps the unfused graph |
| `PARAKEET_CCACHE` | `ON` | `ON` | Use ccache for Parakeet targets when available |

### Installed package

With `-DCMAKE_INSTALL_PREFIX=<prefix>`, `cmake --install <build-dir>` installs:

```text
<prefix>/include/parakeet/*.h
<prefix>/lib/libqvac-parakeet.*
<prefix>/lib/cmake/qvac-parakeet/
<prefix>/lib/pkgconfig/qvac-parakeet.pc
```

The existing downstream fixture in `test/consumer/CMakeLists.txt` uses:

```cmake
find_package(qvac-parakeet CONFIG REQUIRED)
target_link_libraries(consumer PRIVATE qvac::parakeet)
```

Configure consumers with both the Parakeet and ggml prefixes when they are
separate:

```bash
cmake -S parakeet/test/consumer -B consumer-build \
  -DCMAKE_PREFIX_PATH="/path/to/parakeet;/path/to/ggml"
cmake --build consumer-build
```

For pkg-config, use `--static` when linking a static package so private OpenMP
or Apple framework dependencies are included:

```bash
export PKG_CONFIG_PATH=/path/to/parakeet/lib/pkgconfig:/path/to/ggml/lib/pkgconfig
pkg-config --cflags qvac-parakeet
pkg-config --static --libs qvac-parakeet
```

The complete install-tree check is:

```bash
parakeet/scripts/test-package-consumption.sh --help
parakeet/scripts/test-package-consumption.sh \
  --ggml-prefix /path/to/ggml-install
```

Options are `--ggml-prefix <dir>` (required), `--work-dir <dir>`, `--coreml`
(Apple only), and `--keep`. The script verifies artifact names, the CMake
consumer, and static pkg-config consumption. Set
`PARAKEET_PKGTEST_CMAKE_ARGS` for additional CMake definitions such as an
explicit OpenMP installation.

## Repository layout

| Path | Role |
|---|---|
| `CMakeLists.txt` | Library, CLI, tests, examples, and install package |
| `cmake/` | CMake package and pkg-config templates |
| `include/parakeet/` | Public API |
| `src/` | Engine, decoders, backend selection, preprocessing, and CLI |
| `examples/` | Microphone examples |
| `test/consumer/` | Installed-package consumer fixture |
| `scripts/` | ggml setup, conversion, Core ML, references, and package validation |
| `ggml-src/`, `ggml/` | Standalone bundled fabric ggml: the pinned qvac-fabric-llm.cpp clone and the symlink to its `ggml/` tree |
| `models/`, `artifacts/`, `test/samples/` | Local/runtime test fixtures |
| `docs/history/parakeet-port.md` | Archived implementation and parity history |
