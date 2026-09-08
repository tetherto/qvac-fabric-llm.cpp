# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that offloads matrix multiplication (GEMM) to the AMD XDNA NPU
(Ryzen AI, NPU2: Strix Point / Strix Halo / Krackan) through XRT. It is a
scaffold: only `MUL_MAT` is implemented, everything else stays on the CPU.

## What runs on the NPU

`MUL_MAT` ops that satisfy `gemm_supported` (xdna-ops.cpp):

- `src0` weights of type `BF16`, `F16` or `Q4_K` (Q4_K_S); `src1` activations
  and result `F32`. Quantized weights are dequantized once at pack time and
  cached as bf16 in the weight BO, so the NPU kernel is the same for every
  type.
- 2D contiguous tensors (no batch dimensions).
- `N <= 16384`: wide projections (e.g. the vocabulary output layer) stay on
  the CPU.
- `K` is any multiple of `tile_k` (64), and a multiple of 256 for `Q4_K`;
  wide-K ops are split into blocks.

Both prefill and decode are covered. Prefill tiles the M dimension into
32-row blocks; decode runs as `M = 1`. The final vocab projection is the only
model GEMM that does not go to the NPU.

### Hardware constraints

- The kernel is BF16-in / F32-out only.
- Every K-block is capped at 1024 (`GEMM_K_MAX`) and partial blocks are
  zero-padded. The cap was originally added for a context-local hang observed
  at `K_div_k > 64` with `tile_k = 16`; with the current driver the hang no
  longer reproduces, but the conservative cap is kept.

## Requirements

- Linux with an AMD NPU2 (XDNA2) device (Strix Point / Strix Halo / Krackan);
  the NPU shows up as `/dev/accel/accel0` and in `xrt-smi examine`.
- XRT 2.25.37 installed under `/opt/xilinx/xrt`. Put the XRT tools on the
  PATH: `export PATH=/opt/xilinx/xrt/bin:$PATH` (`xclbinutil` is needed when
  building kernels, `xrt-smi` to inspect the device).
- A Python interpreter with mlir-aie (IRON) and llvm-aie to compile the kernel
  at build time (tested with mlir-aie 1.4.1, llvm-aie 21).
- For the standalone kernel harness (`kernels/gemm.py --run`, `bstream.py`)
  also export `PYTHONPATH=/opt/xilinx/xrt/python` so `pyxrt` imports.

## Build

```sh
export PATH=/opt/xilinx/xrt/bin:$PATH

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON -DGGML_OPENMP=ON \
      -DGGML_XDNA_BUILD_KERNELS=ON \
      -DGGML_XDNA_GEMM_PYTHON=$HOME/aie-env/bin/python
cmake --build build -j$(nproc) --target llama-completion llama-server ggml-xdna-kernels
```

- `GGML_XDNA=ON` enables the backend.
- `GGML_OPENMP=ON` is recommended: with `OMP_WAIT_POLICY=PASSIVE` the CPU idle
  cost of waiting on the NPU drops to near zero.
- `GGML_XDNA_BUILD_KERNELS=ON` (default) compiles the GEMM xclbin via
  `kernels/gemm.py`. It is skipped if IRON is not importable; the backend
  still builds.
- `GGML_XDNA_GEMM_PYTHON` overrides the interpreter used for the kernel build;
  point it at the local IRON (mlir-aie) install, e.g. `$HOME/aie-env/bin/python`
  or `$HOME/ironenv-v2/bin/python`.
- `GGML_XDNA_GEMM_VARIANTS` lists the `"K N"` variants to compile (default
  `"1024 2048"`). A single xclbin serves every shape; the geometry is fixed by
  the `XDNA_*` variables in this directory's CMakeLists, which are also passed
  to the C++ code as `GGML_XDNA_*` compile definitions (xdna-seq.h).

The kernel artifacts (`gemm_bf16_f32_M32_K1024_N2048_c8.xclbin` and
`gemm_bf16_f32_M64_K1024_N2048_c8.xclbin`, plus `.insts.bin`) land in the
build output directory (`build/bin`). At
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

Decode benchmark (single stream):

```sh
OMP_WAIT_POLICY=PASSIVE ./build/bin/llama-completion -m model.gguf \
    -p "The capital of France is" -n 32 -t 8 -ngl 99 --poll 0 \
    -s 42 --top-k 1 --temp 0
```

- `--list-devices` shows the NPU; `--device none` forces pure-CPU execution
  (e.g. for a correctness baseline).
- `--poll 0` prevents the threadpool from busy-polling while waiting for the
  NPU.
- `GGML_XDNA_PROFILING=1` prints a per-`MUL_MAT` timing breakdown and a
  `FINALIZE` summary to stderr.

Server flags used for the numbers in "Performance expectations":

```sh
OMP_WAIT_POLICY=PASSIVE ./build/bin/llama-server -m model.gguf \
    -c 147456 -np 4 -b 4096 -ub 4096 --flash-attn off -t 16 --poll 0 \
    --reasoning off --port 8080
```

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

Measured with the `npu-two-kernel` branch (Qwen3.5-0.8B Q4_K_M, llama-server
`-c 147456 -np 4 -b 4096 -ub 4096 --flash-attn off -t 16 --poll 0
--reasoning off`, harness 3 warmup + 8 measured, 128 decode tokens; t/s):

| Context | CPU prefill | XDNA prefill | CPU decode | XDNA decode |
| ------: | ----------: | -----------: | ---------: | ----------: |
|    1024 |       669.3 |        415.5 |       66.4 |        23.0 |
|    4096 |       664.0 |        405.3 |       62.5 |        22.3 |
|   16384 |       561.9 |        357.1 |       49.8 |        20.8 |
|   32768 |       480.9 |        331.4 |       39.2 |        18.7 |

Decode runs as nested CPU on phase1 (`M = 1`). The NPU rail draws ~0.7-0.9 W
while decoding; the benefit on an edge/fabric device is freeing the CPU for
other work, not raw throughput (XDNA decode is ~0.35-0.48x the CPU baseline).
Time to first token at 1k context is ~2.5 s.

## Code layout

| File | Role |
| :-- | :-- |
| `ggml-xdna.cpp` | Backend/device/registry scaffolding (ggml-backend-impl). |
| `xdna-runtime.h/.cpp` | XRT primitives: device, kernel load + insts bind, host buffers, run submit/wait. |
| `xdna-kernel-pool.cpp` | Kernel + buffer pools, artifact scanning. |
| `xdna-seq.h/.cpp` | TXN instruction-stream builder and the GEMM sequence. |
| `xdna-ops.h/.cpp` | Per-op dispatch: GEMM support check, compute, finalize. |
| `xdna-profile.h` | Optional per-op timing (`GGML_XDNA_PROFILING`). |
| `kernels/gemm.py` | IRON design compiled to the xclbin at build time; also a standalone run/trace harness (`--run`, `--trace-size`). |
| `kernels/bstream.py` | DDR read-bandwidth probe for the shim DMA (sequential vs strided B patterns). |
