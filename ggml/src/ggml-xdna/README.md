# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that offloads matrix multiplication to the AMD XDNA NPU (Ryzen
AI, NPU2: Strix Point / Strix Halo / Krackan) through XRT.

## What runs on the NPU

- `MUL_MAT` with weights of type `BF16`, `F16`, `Q4_K`, `Q5_K` or `Q6_K`,
  f32 activations and f32 result. Prefill batches run the GEMM kernels
  (`kernels/gemm.py`); a single-row decode runs the GEMV kernels
  (`kernels/gemv_q4.py`), which keep the weights quantized on DDR and apply
  the group parameters to the accumulator.
- A single-token decode of a Qwen3.5 gated-delta-net layer: the whole
  recurrent layer runs on `fused_layer.xclbin` (conv + norm + GDN + gated
  epilogue, the decode GEMV its projections use, and the FFN transition tile),
  with the recurrent state kept on the device between tokens. This is the
  default decode path and it covers one weight set: `Q4_K` for the FFN gate and
  up, `Q4_K`/`Q5_K`/`Q6_K` for `ssm_out`, `Q4_K`/`Q6_K` for the FFN down. A
  model outside it is refused rather than run on the wrong kernels.
- The device-side C++ of every kernel lives in `kernels/*.cc`; the Python
  designs only declare the fifos and compile those sources.
- `GGML_OP_GATED_DELTA_NET`, `GGML_OP_SSM_CONV` and `GGML_OP_FLASH_ATTN_EXT`
  prefill, on their own kernels (`kernels/gdn_prefill.py`, `kernels/conv.py`,
  `kernels/fa.py`). GDN and FA prefill are opt-in (`GGML_XDNA_GDN=1`,
  `GGML_XDNA_FA=1`); conv prefill is on unless `GGML_XDNA_CONV=0`.

Everything else stays on the CPU.

## Requirements

- Linux with an AMD NPU2 (XDNA2) device; the NPU shows up as
  `/dev/accel/accel0` and in `xrt-smi examine`.
- XRT 2.25.37 installed under `/opt/xilinx/xrt`. Put the XRT tools on the
  PATH: `export PATH=/opt/xilinx/xrt/bin:$PATH`.
- A Python interpreter with mlir-aie (IRON) and llvm-aie to compile the
  kernels at build time.

## Build

```sh
export PATH=/opt/xilinx/xrt/bin:$PATH

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON -DGGML_OPENMP=ON \
      -DGGML_XDNA_BUILD_KERNELS=ON \
      -DGGML_XDNA_GEMM_PYTHON=$HOME/aie-env/bin/python
cmake --build build -j$(nproc) --target llama-completion llama-server ggml-xdna-kernels
```

- `GGML_OPENMP=ON` is recommended: with `OMP_WAIT_POLICY=PASSIVE` the CPU idle
  cost of waiting on the NPU drops to near zero.
- `GGML_XDNA_BUILD_KERNELS=ON` (default) compiles the xclbins. It is skipped
  if IRON is not importable; the backend still builds.
- `GGML_XDNA_GEMM_PYTHON` overrides the interpreter used for the kernel build.
- `GGML_XDNA_GATED_FMT` (default 1) picks the activation layout the fused
  layer's gated stage writes: 0 is the 4-bit form, for a `Q4_K` `ssm_out`;
  1 is the 8-bit form, for `Q5_K`/`Q6_K`. It is compiled into the artifact and
  into the backend, and the two have to agree - a model whose `ssm_out` needs
  the other form is refused with an error naming the flag, because nothing can
  translate between the layouts at run time. Q4_K_M quants differ here: the
  `qwen3.5-0.8b-q4km.gguf` recipe leaves `ssm_out` at Q5_K (build with the
  default), LM Studio's leaves it at Q4_K (`-DGGML_XDNA_GATED_FMT=0`).

The kernel artifacts land in the build output directory (`build/bin`) and are
looked up at runtime in the backend install dir, the executable dir and the
working directory. The two artifacts the recurrent path loads (`fused_layer`,
`gemv_n32_r4_c8`) carry a design tag over the design sources and the build
knobs (`kernels/design_tag.py`), so artifacts built from other sources are
invisible and a stale pair fails loudly instead of producing garbage output.
The tag does not cover the backend's own stream builder, so that side can be
changed without a kernel rebuild. The `.insts.bin` files next to them are
reference copies; the runtime builds its own instruction streams.

Changing the tile geometry or `GGML_XDNA_GATED_FMT` (the `XDNA_*` variables in
this directory's CMakeLists) invalidates every artifact, so rebuild both sides
together:

```sh
cmake --build build --target ggml-xdna-kernels && cmake --build build
```

## Running

```sh
OMP_WAIT_POLICY=PASSIVE ./build/bin/llama-server -m model.gguf \
    --reasoning off --poll 0 --port 8080
```

- `--list-devices` shows the NPU; `--device none` forces pure-CPU execution.
- `--poll 0` prevents the threadpool from busy-polling while waiting for the
  NPU.
- `GGML_XDNA_GLUE_THREADS` / `GGML_XDNA_GLUE_THREADS_BIG` size the host
  fallback pool used inside an NPU chunk (defaults 4 and 16).

The remaining knobs are for bisecting a regression, not for tuning: each one
disables the NPU path it names, and the default is the fast one.

| Variable | Default | Effect |
| :-- | :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | 1 | `0` runs every op on the per-op kernels instead of the fused recurrent layer. |
| `GGML_XDNA_GLUE` | 1 | `0` stops the backend from claiming the cheap host ops of a decode chunk. |
| `GGML_XDNA_INT8` | 1 | `0` drops the native int8 prefill GEMM. |
| `GGML_XDNA_CONV` | 1 | `0` runs the prefill conv on the host. |
| `GGML_XDNA_GDN` | 0 | `1` runs the GDN prefill body on the array. |
| `GGML_XDNA_FA` | 0 | `1` runs flash-attention prefill on the array. |
| `GGML_XDNA_GEMV_PROMOTE` | 0 | `1` lets one GEMV dispatch mix weight formats. |
| `GGML_XDNA_SPIN` | 0 | `1` polls for kernel completion instead of blocking.

## Code layout

| File | Role |
| :-- | :-- |
| `ggml-xdna.cpp` | Backend/device/registry scaffolding, graph dispatch, the fused recurrent-layer glue. |
| `xdna-types.h` | XRT-backed structs: device, kernel, buffer, kernel pool. |
| `xdna-runtime.h/.cpp` | XRT primitives: device, kernel load + insts bind, host buffers, run submit/wait, pool. |
| `xdna-util.h` | Small helpers shared by more than one translation unit. |
| `xdna-seq.h/.cpp` | TXN instruction-stream builder and the GEMM sequence. |
| `xdna-ops.h/.cpp` | Per-op dispatch: GEMM support check, compute, finalize; decode GEMV grouping. |
| `xdna-gemv.h/.cpp` | Decode GEMV: packed weight formats, stream builder, runner, FFN pair. |
| `xdna-quant.h/.cpp` | NPU weight formats (q4g32 / q8g16) and the ggml repack. |
| `xdna-rec.h/.cpp` | The fused recurrent layer: session state, packing, core stream, FFN transition. |
| `xdna-rec-gemv.h/.cpp` | The fused layer's projections (ssm_out, FFN) on the decode GEMV. |
| `xdna-seq-attn.cpp` | The fused core's phased TXN stream. |
| `xdna-gdn-prefill.h/.cpp` | GDN prefill kernel runner. |
| `xdna-conv-prefill.h/.cpp` | SSM_CONV prefill kernel runner. |
| `xdna-fa-prefill.h/.cpp` | Flash-attention prefill kernel runner. |
| `kernels/*.py` | IRON/MLIR-AIE designs: the AIE array configuration and the fifos. |
| `kernels/*.cc` | The device C++ the designs compile in, one file per kernel. |
