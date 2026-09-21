# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that runs part of a llama.cpp graph on the AMD XDNA NPU (Ryzen
AI, XDNA2 / NPU2: Strix Point, Strix Halo, Krackan) through XRT. It is built
around one model family - Qwen3.5 with gated-delta-net layers - whose
single-token decode runs almost entirely on the array; everything else falls
back to per-op kernels or the host.

## What runs on the NPU

### MUL_MAT

Weights of type `BF16`, `F16`, `Q4_K`, `Q5_K` or `Q6_K`, f32 activations and
f32 result, contiguous tensors, one sequence (`ne[2]*ne[3] == 1`), a `K` that is
a multiple of 256 for the quantized types (whole ggml super-blocks), and an `N`
a route can carry (16384 for the GEMM; 14 array passes, 14336 output columns,
for the GEMV). The route depends on the number of activation rows `M`:

| M | route |
| :-- | :-- |
| 1 | decode GEMV (`kernels/gemv_q4.py` + `kernels/gemv-q4.cc`): the weights stay quantized in DDR and the group parameters are applied to the accumulator |
| 2..63 | decode GEMM, the M32 block (`kernels/gemm.py`, native bf16 r=4) |
| >= 64 | prefill GEMM, the cheapest baked M block; bf16 for `F16`/`BF16`/`Q4_K`, and the native int8 route (1 B/value) for `Q4_K`/`Q5_K`/`Q6_K` |

The GEMM artifacts are compiled at K1024/N2048 but the runtime drives any `K`
and `N` (the stream builder writes the loop counts per dispatch), so only the
baked M block is fixed: one artifact per M block. A wide `N` such as the
vocabulary projection stays on the host.

### The fused decode recurrent layer

The default decode path (`GGML_XDNA_FUSED_LAYER=0` disables it). One design,
`fused_layer.xclbin`, holds the whole recurrent layer of a single-token decode:
conv + norm + GDN + gated epilogue, the decode GEMV the layer's projections run
on, and the FFN transition tile. It fires only for a complete recurrent layer
of `blk.N.*` present in one chunk, which means the conv1d, ssm_norm,
post_attention_norm, ssm_out and ffn gate/up/down weights, the qkv and gate
projections and the gate/beta/state nodes all have to be in that chunk. The
recurrent state stays on the device between tokens.

It covers one weight set: `Q4_K` for the FFN gate and up, `Q4_K`/`Q5_K`/`Q6_K`
for `ssm_out`, `Q4_K`/`Q6_K` for the FFN down, plus a `GGML_XDNA_GATED_FMT` that
matches the `ssm_out` layout. Anything else is refused with an error rather
than run on the wrong kernels; `GGML_XDNA_FUSED_LAYER=0` forces the per-op path
instead.

### Prefill ops

| op | kernel | default |
| :-- | :-- | :-- |
| `SSM_CONV`, from 64 tokens | `kernels/conv.py` | on, `GGML_XDNA_CONV=0` for the host |
| `GATED_DELTA_NET` | `kernels/gdn_prefill.py` | opt-in, `GGML_XDNA_GDN=1` |
| `FLASH_ATTN_EXT`, plain causal mask only | `kernels/fa.py` | opt-in, `GGML_XDNA_FA=1` |

### Everything else

The backend also claims the cheap contiguous f32 ops of an NPU chunk and runs
them on the host from inside that chunk (the "glue"). The point is not speed:
it keeps the scheduler from splitting the recurrent subgraph at every backend
change, which would put a layer's inputs back through the graph allocator
between NPU dispatches. `GGML_XDNA_GLUE=0` claims only the native NPU ops.

The glue runs on its own CPU backend with its own pool: `GGML_XDNA_GLUE_THREADS`
(4) for decode-sized chunks, `GGML_XDNA_GLUE_THREADS_BIG` (16) once a chunk has
32 tokens or more. `-t` does not size it.

## How it works

### Artifacts, the design tag, and the instruction streams

The AIE designs are Python (IRON / mlir-aie) under `kernels/`; the device C++
they compile in is `kernels/*.cc`, one file per kernel. They are compiled out of
line at build time into `<stem>.xclbin` next to their `.insts.bin`, in the build
output directory. The backend looks for artifacts in the backend install dir,
the executable dir and the working directory.

`fused_layer` and `gemv_n32_r4_c8` are loaded under a **design-tagged** name.
`kernels/design_tag.py` hashes the design sources and the build knobs, writes
`xdna-design-tag.h`, and the backend includes it, so the name a build produces
and the name the backend looks for come from one place: an artifact built from
other sources is invisible (and reported) rather than driven. The tag covers
the Python designs and the knobs, not the backend's own stream builders - those
can change without a kernel rebuild.

The `.insts.bin` files are reference copies. The runtime builds its own TXN
instruction streams in C++ (`xdna-seq.cpp`, `xdna-seq-attn.cpp`,
`xdna-gemv.cpp`), which is what lets one artifact serve every shape.

### The array

Eight AIE columns, four compute rows each. The GEMV designs stream every
column: one xclbin is one hardware context, and the device allows 16, so the
backend warms exactly the artifacts the active mode submits while the NPU is
idle - registering a hardware context in the middle of NPU work fails, and a
context spent on an artifact nothing submits is one the used ones cannot have.

Weights are packed into q4g32 (Q4_K: 4-bit codes plus an int8 scale and min per
32 values) or q8g16 (Q4_K/Q5_K/Q6_K: int8 codes plus scale and min per 16).
Both decode as `w = q*d + m` and both are exact with respect to the ggml block
they come from, so a single kernel streams either and the GEMV never switches
hardware context. The packed weights live in a device buffer for the process
lifetime, keyed by the tensor's data pointer and shape.

### One decode token

1. The scheduler has placed the ops here because `supports_op` claimed them.
2. `graph_compute` pre-scans the chunk and plans a fused run for every complete
   recurrent layer it finds.
3. The dispatch loop runs the rest of the chunk: projections on the decode GEMV
   (projections that read the same activation are grouped into one dispatch),
   prefill kernels where they apply, and the host glue batched over runs of
   consecutive host ops.
4. At a layer's fire point the fused run executes and writes `h_attn`/`h_out`
   directly into the layer's add tensors; the layer's own conv/gdn/ffn nodes are
   marked consumed and skipped.

Prefill chunks take the same path with `M > 1` (GEMM blocks and the conv/fa/gdn
prefill kernels); the fused layer never fires there.

## Requirements

- Linux with an AMD NPU2 (XDNA2) device; it shows up as `/dev/accel/accel0` and
  in `xrt-smi examine`.
- XRT 2.25.37 under `/opt/xilinx/xrt`, with its tools on the `PATH`.
- A Python interpreter with IRON (mlir-aie + llvm-aie) to compile the kernels.
  Verified against mlir-aie 1.4.3: the RTP buffer bases in `xdna-seq.h` are read
  back from that toolchain's placement, so a version bump needs them re-read
  from a freshly compiled project (the header says which values).
- A model the fused decode path covers (see above), or any model for the per-op
  path.

## Build

```sh
export PATH=/opt/xilinx/xrt/bin:$PATH

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON -DGGML_OPENMP=ON \
      -DGGML_XDNA_BUILD_KERNELS=ON \
      -DGGML_XDNA_GEMM_PYTHON=$HOME/aie-env/bin/python
cmake --build build -j$(nproc)
```

- The default target builds the backend and the kernel artifacts. To rebuild
  only the kernels: `cmake --build build --target ggml-xdna-kernels`.
- `GGML_XDNA_BUILD_KERNELS=ON` (default) compiles the xclbins out of line. If
  IRON is not importable it is skipped with a warning and the backend still
  builds - it then finds no artifacts and keeps everything on the host.
- `GGML_XDNA_GEMM_PYTHON` overrides the interpreter used for the kernel build.
- `GGML_OPENMP=ON` is recommended: the weight packing and the host fallback are
  OpenMP loops.
- `GGML_XDNA_GATED_FMT` (default 1) selects the activation layout the gated
  stage writes: `0` is the 4-bit form for a `Q4_K` `ssm_out`, `1` the 8-bit form
  for `Q5_K`/`Q6_K`. It is compiled into the artifact and into the backend, and
  the backend refuses a model whose `ssm_out` needs the other one. Q4_K_M
  quants differ here: `qwen3.5-0.8b-q4km.gguf` leaves `ssm_out` at Q5_K (the
  default), LM Studio's leaves it at Q4_K (`-DGGML_XDNA_GATED_FMT=0`).

Artifacts, all in the build output directory (`build/bin` by default):

| artifact | what |
| :-- | :-- |
| `gemm_bf16_f32_M{32,256,512,2048}_K1024_N2048_c8` | bf16 GEMM, one per M block; M32 is the decode one |
| `gemm_int8_int32_M{256,512,2048}_K1024_N2048_c8` | native int8 GEMM, prefill |
| `gdn_prefill_bf16_S128_H16_CS64_c8` | GATED_DELTA_NET prefill |
| `conv_prefill_bf16_c16_t256_kw4_mb8` | SSM_CONV prefill |
| `fa_prefill_bf16_D256_MT32_JT8_NJ64_c8` | flash attention prefill |
| `fused_layer_<tag>` | the fused recurrent layer, decode |
| `gemv_n32_r4_c8_<tag>` | the standalone decode GEMV |

The prefill M blocks come from `XDNA_GEMM_M_BIG` and `XDNA_GEMM_M_BIG_EXTRA`;
each has to be a multiple of `tile_m * n_compute_rows`, which differs between
the bf16 and int8 routes. Changing the geometry (`XDNA_*` in this directory's
CMakeLists) or `GGML_XDNA_GATED_FMT` invalidates the tagged artifacts, so
rebuild both sides together:

```sh
cmake --build build --target ggml-xdna-kernels && cmake --build build
```

## Running

```sh
OMP_WAIT_POLICY=PASSIVE ./build/bin/llama-server -m model.gguf \
    --reasoning off --poll 0 --port 8080
```

- `--list-devices` shows the NPU; `--device none` forces pure-CPU execution.
- `--poll 0` stops the threadpool busy-polling while the NPU works;
  `OMP_WAIT_POLICY=PASSIVE` does the same for the host fallback pool.
- Offload as usual: `-ngl 99` puts every layer on the device.

### Environment

The knobs below are for bisecting a regression rather than tuning: each one
disables the NPU path it names, and the default is the fast one.

| variable | default | effect |
| :-- | :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | 1 | `0` runs the decode on the per-op kernels instead of the fused layer |
| `GGML_XDNA_GLUE` | 1 | `0` stops the backend claiming the host ops of a decode chunk |
| `GGML_XDNA_GLUE_THREADS` | 4 | host-fallback threads for a decode-sized chunk |
| `GGML_XDNA_GLUE_THREADS_BIG` | 16 | host-fallback threads for a chunk of 32 tokens or more |
| `GGML_XDNA_INT8` | 1 | `0` drops the native int8 prefill GEMM |
| `GGML_XDNA_CONV` | 1 | `0` runs the prefill conv on the host |
| `GGML_XDNA_GDN` | 0 | `1` runs the GDN prefill body on the array |
| `GGML_XDNA_FA` | 0 | `1` runs flash-attention prefill on the array |
| `GGML_XDNA_GEMV_PROMOTE` | 0 | `1` lets one GEMV dispatch mix weight formats |
| `GGML_XDNA_SPIN` | 0 | `1` polls for kernel completion instead of blocking |

## Troubleshooting

- **`fused_layer.xclbin is present but not built for design tag ...`** - the
  artifacts and the backend are out of sync. Rebuild both:
  `cmake --build build --target ggml-xdna-kernels && cmake --build build`.
- **`ssm_out is ..., which needs the ...-bit activation layout, but the kernels
  were built with GATED_FMT=...`** - the model and the kernel build disagree.
  Reconfigure with the `-DGGML_XDNA_GATED_FMT` the message names and rebuild.
- **The fused layer is not used at all** - no tagged `fused_layer` artifact was
  found (check the build output dir and the tag in `xdna-design-tag.h`), or
  `GGML_XDNA_FUSED_LAYER=0` is set. Without it the decode still runs, on the
  per-op kernels.
- **`unsupported fused weight set`** - the model's quantization is outside the
  set the fused kernels were built for; run it with `GGML_XDNA_FUSED_LAYER=0`.
- **No NPU** - the backend still registers (llama.cpp expects every accelerator
  device to answer) but claims no ops, so everything runs on the CPU.

## Code layout

| file | role |
| :-- | :-- |
| `ggml-xdna.cpp` | backend/device/registry scaffolding, graph dispatch, the fused-layer glue |
| `xdna-types.h` | XRT-backed structs: device, kernel, buffer, kernel pool |
| `xdna-runtime.h/.cpp` | XRT primitives: device, kernel load and insts bind, host buffers, submit/wait, pool |
| `xdna-util.h` | small helpers shared by more than one translation unit |
| `xdna-seq.h/.cpp` | TXN instruction-stream builder and the GEMM sequence |
| `xdna-seq-attn.cpp` | the fused core's phased per-token TXN stream |
| `xdna-ops.h/.cpp` | per-op dispatch: GEMM/GEMV support, compute, finalize; decode GEMV grouping |
| `xdna-gemv.h/.cpp` | decode GEMV: packed weight formats, stream builder, runner, FFN pair |
| `xdna-quant.h/.cpp` | NPU weight formats (q4g32 / q8g16) and the ggml repack |
| `xdna-rec.h/.cpp` | the fused recurrent layer: session state, packing, core stream |
| `xdna-rec-gemv.h/.cpp` | the fused layer's projections (ssm_out, FFN) on the decode GEMV |
| `xdna-gdn-prefill.h/.cpp` | GDN prefill runner |
| `xdna-conv-prefill.h/.cpp` | SSM_CONV prefill runner |
| `xdna-fa-prefill.h/.cpp` | flash-attention prefill runner |
| `kernels/*.py` | IRON/MLIR-AIE designs: the array configuration, the fifos and the runtime sequence |
| `kernels/*.cc` | the device C++ those designs compile in |

## Known limits

- The fused decode layer is written around one geometry - 6144 conv channels,
  1024 `d_out`, 16 value heads of 128, a 262144-float recurrent state - and the
  designs are compiled for it, so another GDN model needs the designs rebuilt
  and the constants in `xdna-rec.h` revisited.
- The host-built instruction streams and the RTP/shim constants follow the
  mlir-aie placement. The design tag does not cover the toolchain version, so a
  toolchain bump has to be paired with re-reading those constants.
- One backend context per process holds the dispatch state, so two decodes at
  once from separate llama contexts share it (single-context use is what is
  tested).
- The device reports itself to the scheduler as a GPU while its buffers are host
  memory, so the reported free memory and any `--fit` accounting are nominal.
