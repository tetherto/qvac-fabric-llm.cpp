# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that offloads matrix multiplication (GEMM) to the AMD XDNA NPU
(Ryzen AI, NPU2: Strix Point / Strix Halo / Krackan) through XRT. It is a
scaffold: only `MUL_MAT` is implemented, everything else stays on the CPU.

## What runs on the NPU

`MUL_MAT` ops that satisfy `gemm_supported` (xdna-ops.cpp):

- `src0` weights of type `BF16` or `F16`, `src1` activations and result `F32`.
- 2D contiguous tensors (no batch dimensions).
- `N <= 16384`: wide projections (e.g. the vocabulary output layer) stay on
  the CPU.
- `K` is any multiple of `tile_k` (16); wide-K ops are split into blocks.

Both prefill and decode are covered. Prefill tiles the M dimension into
32-row blocks; decode runs as `M = 1`. The final vocab projection is the only
model GEMM that does not go to the NPU.

### Hardware constraints

- `K_div_k > 64` (K > 1024 with `tile_k = 16`) wedges the shared NPU context,
  so every K-block is capped at 1024 and partial blocks are zero-padded. This
  is a context-local hang that reproduces across IRON and hand-built streams.
- The kernel is BF16-in / F32-out only.

## Requirements

- Linux with an AMD NPU2 (XDNA2) device.
- XRT (tested with 2.25.37, installed under `/opt/xilinx/xrt`).
- A Python interpreter with mlir-aie (IRON) and llvm-aie to compile the kernel
  at build time (tested with mlir-aie 1.4.1).

## Build

```sh
cmake -B build -DGGML_XDNA=ON -DGGML_OPENMP=ON \
      -DGGML_XDNA_GEMM_PYTHON=$HOME/aie-env/bin/python
cmake --build build -j$(nproc)
```

- `GGML_XDNA=ON` enables the backend.
- `GGML_OPENMP=ON` is recommended: with `OMP_WAIT_POLICY=PASSIVE` the CPU idle
  cost of waiting on the NPU drops to near zero.
- `GGML_XDNA_BUILD_KERNELS=ON` (default) compiles the GEMM xclbin via
  `kernels/gemm.py`. It is skipped if IRON is not importable; the backend
  still builds.
- `GGML_XDNA_GEMM_PYTHON` overrides the interpreter used for the kernel build.
- `GGML_XDNA_GEMM_VARIANTS` lists the `"K N"` variants to compile (default
  `"1024 2048"`). A single xclbin serves every shape; the geometry is fixed by
  the `XDNA_*` variables in this directory's CMakeLists, which are also passed
  to the C++ code as `GGML_XDNA_*` compile definitions (xdna-seq.h).

The kernel artifacts land in the build output directory (`build/bin`). At
runtime they are looked up in the backend install dir, the executable dir and
the working directory. The `.insts.bin` files are reference copies of the
compiled DMA sequences; the runtime builds its own instruction streams.

## Running

The NPU shows up as a normal ACCEL device; use the default `-ngl 99` to place
all layers on it:

```sh
OMP_WAIT_POLICY=PASSIVE ./build/bin/llama-server -m model.gguf \
    --reasoning off --poll 0 --port 8080
```

- `--list-devices` shows the NPU; `--device none` forces pure-CPU execution
  (e.g. for a correctness baseline).
- `--poll 0` prevents the threadpool from busy-polling while waiting for the
  NPU.
- `GGML_XDNA_PROFILING=1` prints a per-`MUL_MAT` timing breakdown and a
  `FINALIZE` summary to stderr.

### Correctness check

Compare greedy output against the CPU baseline (sampling is not randomized, so
any difference is a real numerical mismatch):

```sh
./build/bin/llama-completion -m model.gguf -p "The capital of France is" \
    -n 64 -t 4 -ngl 99 -s 42 --poll 0 --top-k 1 --temp 0          # NPU
./build/bin/llama-completion -m model.gguf -p "The capital of France is" \
    -n 64 -t 4 -ngl 99 -s 42 --poll 0 --top-k 1 --temp 0 --device none  # CPU
```

The assistant text must match. With non-greedy sampling the outputs can diverge
between runs even with the same seed: M-tiling and K-splitting change the
accumulation order, which shifts logits within BF16 tolerance and flips sampled
tokens. This is expected.

### Performance expectations

The NPU path trades throughput for CPU headroom: on a small model (0.8B) the
CPU draw drops from ~30% to ~7% and CPU power from ~38W to ~30W, while both
prefill and decode get slower than the CPU. The benefit is freeing the CPU for
other work on an edge/fabric device, not raw speed.

## Code layout

| File | Role |
| :-- | :-- |
| `ggml-xdna.cpp` | Backend/device/registry scaffolding (ggml-backend-impl). |
| `xdna-runtime.h/.cpp` | XRT primitives: device, kernel load + insts bind, host buffers, run submit/wait. |
| `xdna-kernel-pool.cpp` | Kernel + buffer pools, artifact scanning. |
| `xdna-seq.h/.cpp` | TXN instruction-stream builder and the GEMM sequence. |
| `xdna-ops.h/.cpp` | Per-op dispatch: GEMM support check, compute, finalize. |
| `xdna-profile.h` | Optional per-op timing (`GGML_XDNA_PROFILING`). |
| `kernels/gemm.py` | IRON design compiled to the xclbin at build time. |
