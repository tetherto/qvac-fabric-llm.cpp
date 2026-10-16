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
| 1 | decode GEMV (`kernels/gemv_q4.py` + `kernels/gemv-q4.cc`): the weights stay quantized in DDR and the group parameters are applied to the accumulator; `Q5_K`/`Q6_K` `attn_qkv` and `attn_v` are taken at 4 bits by default (see [4-bit decode weights](#4-bit-decode-weights-on-by-default)) |
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
than run on the wrong kernels; the refusal names the weight set. The way on is
to requantize `ssm_out` (see Build). `GGML_XDNA_FUSED_LAYER=0` is a bisection
switch, not a way to run a model this layer refuses.

The designs are also built for one residual width (1024, `XDNA_RES_D`). A model
whose residual stream is wider - Qwen3.5-2B is 2048 - cannot be handed a
recurrent layer at all: its decode falls back by itself and stays on the array
as far as the width allows. The recurrent bodies (conv, GDN, norms, GLU) run on
the host, while the projections keep the decode GEMV on the same array, the
decode attention stays on the pool and the head stays on the array. One warning
names the width when it happens. `GGML_XDNA_FUSED_LAYER=0` still forces the
per-op path for a model the fused design does cover.

### Prefill ops

| op | kernel | default |
| :-- | :-- | :-- |
| `SSM_CONV`, from 64 tokens | `kernels/conv.py` | opt-in, `GGML_XDNA_CONV=1`; the host is cheaper and is what runs otherwise |
| the prefill GDN's conv input | `kernels/gdn_conv.py` | opt-in, `GGML_XDNA_GDN_CONV=1`; the host is faster (pp512 1900 against 1680 t/s on the 0.8B) and is what runs otherwise |
| `GATED_DELTA_NET` | `kernels/gdn_mm.py` | on, `GGML_XDNA_GDN_MM=0` for the host |
| `FLASH_ATTN_EXT` | `kernels/attn_mm.py` | on, `GGML_XDNA_FA_MM=0` for the host |

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
output directory. The backend looks for artifacts, in order, in
`GGML_XDNA_KERNEL_DIR` when it is set, the directory of the module the backend
is in (the backend library in a `GGML_BACKEND_DL` build, the binary it is
linked into otherwise), the backend install dir, the executable dir and the
working directory. A package that loads the backend with
`ggml_backend_load_all_from_path()` ships the artifacts next to the library.

`fused_layer` and `gemv_n32_r4_c8` are loaded under a **design-tagged** name.
`kernels/design_tag.py` hashes the design sources and the build knobs. Configure
passes that tag to the backend as a compile definition and the build writes it to
`xdna-design-tag.h` in the build tree, so the name a build produces and the name
the backend looks for are the same value: an artifact built from other sources is
invisible (and reported) rather than driven. The tag covers the Python designs
and the knobs, not the backend's own stream builders - those can change without a
kernel rebuild. A build that does not generate kernels compiles the committed
`xdna-design-tag.h` in this directory instead.

The `.insts.bin` files are reference copies. The runtime builds its own TXN
instruction streams in C++ (`xdna-seq.cpp`, `xdna-seq-attn.cpp`,
`xdna-gemv.cpp`), which is what lets one artifact serve every shape.

### Provenance and licences

Most of this backend is original work under the repository's MIT licence. Two
parts are worth stating plainly.

The host-side instruction-stream builders (`xdna-seq.cpp`, `xdna-seq-attn.cpp`,
`xdna-gemv.cpp` and the stream code in `xdna-pgemm.cpp`) are this backend's own
implementation of the NPU2 TXN format. They were written from the BD field
semantics documented in mlir-aie (`AIEDmaToNpu.cpp`,
`include/aie/Runtime/TxnEncoding.h`) and from instruction streams read back out
of artifacts this design built (`.insts.bin`); no mlir-aie or IRON source was
copied or translated into them. The one sentence quoted from mlir-aie, in
`xdna-seq.cpp`, is attributed where it appears.

`kernels/gemm.py` is the exception: its array and ObjectFifo plumbing is
adapted from mlir-aie's `programming_examples` matrix-multiplication designs,
which are `Copyright (C) 2024-2026 Advanced Micro Devices, Inc.` under
`Apache-2.0 WITH LLVM-exception`. The file carries that notice and the licence
texts ship beside it in `LICENSES/`. The other designs under `kernels/` use the
published IRON API only.

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists what is derived from
what, and says why the compiled `.xclbin` / `.insts.bin` need nothing added
(the LLVM exception covers compiled output).

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

#### 4-bit decode weights (on by default)

Two steps are lossy, and both are on by default by design. The first: for the single-token
decode, every `Q5_K` or `Q6_K` weight named `attn_qkv` or `attn_v` is
re-quantized to `Q4_K` from its dequantized values when it is packed, and the
decode GEMV streams it at 4 bits (q4g32) instead of the exact 8-bit form. That
applies wherever the decode runs those weights: the fused recurrent and
attention layers and the per-op GEMV. These two tensors buy the most decode
speed per unit of accuracy lost. On Qwen3.5-0.8B Q4_K_M, decode goes from 47.5
to 52.4 t/s at a 1k context, and the decode KLD against the CPU from 0.0051 to
0.0122. Greedy text can differ from the exact form where two tokens are close.

It does not touch:
- the prefill GEMM, which takes every weight in its own format (4 bits there
  would cost prefill KLD 0.0041 -> 0.0105);
- `Q4_K` tensors, which are 4-bit already;
- `ssm_out`;
- the weights in the model file.

`GGML_XDNA_W4` controls it:
- unset: `attn_qkv,attn_v`, the default above;
- set and empty (`GGML_XDNA_W4=`): every weight keeps its exact 8-bit form, at
  the slower decode;
- a comma-separated list (`GGML_XDNA_W4=attn_v`): the `Q5_K`/`Q6_K` weights it
  applies to, by the last part of their name (`blk.N.<name>.weight`).

The decode head (the vocabulary projection on the array) is the other lossy
default, on the same terms: its first 65536 vocabulary rows - a byte-pair
vocabulary's most frequent tokens - stay at 8 bits and the rest are 4-bit.
Against the CPU's `Q6_K` head on the 0.8B that is a head KLD of 0.0052, where
every row at 4 bits would be 0.0165. `GGML_XDNA_HEAD_EXACT_ROWS` sets the
number of exact rows (`0`: all 4-bit); `GGML_XDNA_HEAD=0` takes the head off
the array altogether.

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
- XRT 2.25.37 under `/opt/xilinx/xrt`, with its tools on the `PATH`. Older XRT
  is not supported: 2.21.75 has been seen to fail a ~1.2 GB host-memory BO
  allocation at startup and the decode was not reproducible on that
  installation. The version is not checked at run time.
- The libuuid development files (`uuid-dev` on Ubuntu). XRT's headers include
  `<uuid/uuid.h>` and the link needs `-luuid`, and XRT brings neither, so a
  build fails on the include and then on the link without them.
- A Python interpreter with IRON (mlir-aie + llvm-aie) to compile the kernels.
  Verified against mlir-aie 1.4.3: the RTP buffer bases in `xdna-seq.h` are read
  back from that toolchain's placement, so a version bump needs them re-read
  from a freshly compiled project (the header says which values).
- A model the fused decode path covers (see above). A refused quantization is
  not run by setting `GGML_XDNA_FUSED_LAYER=0`; requantize `ssm_out` (see Build).

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
  the backend refuses a model whose `ssm_out` needs the other one. Which type a
  quant ends up with is a property of the quant, not of its name:
  `llama-quantize ... Q4_K_M` built from this tree leaves `ssm_out` at `Q4_K`
  (so the fused layer needs `-DGGML_XDNA_GATED_FMT=0`), while other `Q4_K_M`
  conversions leave it at `Q5_K` (the default). Check the model with
  `llama-gguf model.gguf r n | grep ssm_out`, or read the refusal the backend
  prints. To keep the default `GATED_FMT=1`, requantize that one tensor:

  ```sh
  llama-quantize --tensor-type ssm_out=q5_K model-f16.gguf out.gguf Q4_K_M
  ```

  A `ssm_out` outside `Q4_K`/`Q5_K`/`Q6_K` is refused whatever `GATED_FMT` is -
  a stock `Qwen/Qwen3.5-0.8B Q4_K_M` leaves half of them at `Q8_0` and prints
  `unsupported fused weight set (so=q8_0 gate=q4_K up=q4_K down=q6_K)`. The
  `--tensor-type ssm_out=q5_K` recipe above is the way onto the fused path.
  `GGML_XDNA_FUSED_LAYER=0` does not run this model. It only moves a model the
  fused layer accepts onto the per-op kernels, for a bisection.

Artifacts, all in the build output directory (`build/bin` by default):

| artifact | what |
| :-- | :-- |
| `gemm_bf16_f32_M{32,256,512,2048}_K1024_N2048_c8` | bf16 GEMM, one per M block; M32 is the decode one |
| `gemm_int8_int32_M{256,512,2048}_K1024_N2048_c8` | native int8 GEMM, prefill |
| `pgemm_c8` | prefill GEMM in the decode GEMV's weight tiles |
| `gdn_mm_c8` | GATED_DELTA_NET prefill |
| `gdn_conv_c8` | the prefill GDN's conv input, opt-in |
| `attn_mm_c8` | FLASH_ATTN_EXT prefill |
| `conv_prefill_bf16_c16_t256_kw4_mb8` | SSM_CONV prefill, opt-in |
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

- `--list-devices` shows the NPU as `XDNA0`; `--device none` forces pure-CPU
  execution.
- The NPU is an integrated device to llama.cpp. In a build with another GPU
  backend, llama.cpp by default takes that GPU instead (a discrete one, or an
  iGPU registered before XDNA, as Vulkan's is); `--device XDNA0` selects the
  NPU.
- `--poll 0` stops the threadpool busy-polling while the NPU works;
  `OMP_WAIT_POLICY=PASSIVE` does the same for the host fallback pool.
- Offload as usual: `-ngl 99` puts every layer on the device.
- Leave flash attention on (`-fa on`, or `auto`, the default). The NPU's
  attention routes take llama's `FLASH_ATTN_EXT`; with `-fa off` llama builds
  attention from `MUL_MAT` and `SOFT_MAX`, which run on the host: about half
  the decode speed on the 0.8B, with ten times the host work. The backend
  logs a warning once when it sees that.

### Environment

Every `GGML_XDNA_*` variable the backend reads is listed here. Unset is the
default, and the default is what a release runs.

For users - a trade-off, a size or a location:

| variable | default | effect |
| :-- | :-- | :-- |
| `GGML_XDNA_W4` | `attn_qkv,attn_v` | the `Q5_K`/`Q6_K` weights the decode GEMV takes at 4 bits, a lossy default (see [4-bit decode weights](#4-bit-decode-weights-on-by-default)); empty keeps them at 8 bits |
| `GGML_XDNA_HEAD_EXACT_ROWS` | 65536 | the vocabulary rows the decode head keeps at 8 bits, from row 0 (a byte-pair vocabulary's most frequent tokens); the rest are 4-bit. Head KLD against the CPU's Q6_K head on the 0.8B: 0.0052 at 65536, 0.0165 with `0` (all 4-bit) |
| `GGML_XDNA_TOKEN` | 1 | how a token's queued runs go to the device: `1` as one joined command, `t > 1` at most `t` runs a command, `0` a command each - slightly faster decode at a higher package power (24.0 W at 50.9 t/s against 18.0 W at ~48 t/s with the default, 1k context) |
| `GGML_XDNA_SPIN_US` | 0 | poll this many microseconds for completion before blocking; `-1` polls until it completes |
| `GGML_XDNA_GLUE_THREADS` | 4 | host threads for the ops the backend runs on the host in a decode-sized chunk |
| `GGML_XDNA_GLUE_THREADS_BIG` | 16 | the same, for a chunk of 32 tokens or more |
| `GGML_XDNA_ARENA_MB` | 256 | size of each chunk of the device-buffer arena, in MiB |
| `GGML_XDNA_CONV` | 0 | `1` runs the standalone prefill `SSM_CONV` on the array; the host is cheaper, which is why it is off |
| `GGML_XDNA_GDN_CONV` | 0 | `1` computes the prefill GDN's conv input on the array; the host is faster, which is why it is off |
| `GGML_XDNA_KERNEL_DIR` | unset | a directory to look for the kernel artifacts in before the others (see Artifacts) |

For bisecting a regression - each `0` turns off the NPU route it names, and the
host (or a less fused path) runs it instead. They are for narrowing down a bug
report, not for tuning: every one is slower than the default.

| variable | `0` turns off |
| :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | the fused decode layers (the per-op path runs). It does not run a model the fused layer refuses |
| `GGML_XDNA_GEMV_GROUP` | the decode projections outside the fused layer, on the array |
| `GGML_XDNA_GEMV_PROMOTE` | merging a decode GEMV group of mixed weight formats into one dispatch |
| `GGML_XDNA_HEAD` | the decode's vocabulary projection on the array |
| `GGML_XDNA_QUEUE` | queuing the fused layers: each one is waited for |
| `GGML_XDNA_ATTN` | the decode attention on the array |
| `GGML_XDNA_ATTN_LAYER` | a whole attention layer, the q/k/v projections to `l_out`, in one dispatch (the tail and the decode attention still run) |
| `GGML_XDNA_ATTN_TAIL` | an attention layer's tail - output projection, residual, post-attention norm and FFN - in one dispatch |
| `GGML_XDNA_GLUE` | the host ops the backend runs inside an NPU chunk (the scheduler splits the graph at each instead) |
| `GGML_XDNA_HOST_BO` | the backend's host buffer type: plain CPU buffers |
| `GGML_XDNA_PGEMM` | the prefill GEMM on the array |
| `GGML_XDNA_INT8` | the native int8 prefill GEMM |
| `GGML_XDNA_PGEMM_ADD` | the residual ADD folded into the norm's pass |
| `GGML_XDNA_PGEMM_NORM_A` | the RMS norm written into the next GEMM's A |
| `GGML_XDNA_PGEMM_GATED_A` | a recurrent layer's gated output written into the output projection's A |
| `GGML_XDNA_PGEMM_GATE_A` | the attention gate written into the output projection's A |
| `GGML_XDNA_PGEMM_GLU` | the SwiGLU computed from the gate and up GEMMs on the array |
| `GGML_XDNA_PGEMM_GLU_A` | the SwiGLU written into the down projection's A |
| `GGML_XDNA_GDN_MM` | the prefill gated delta rule on the array |
| `GGML_XDNA_GDN_IN` | the prefill GDN's conv input, norms and state copy fed straight into the array |
| `GGML_XDNA_FA_MM` | the prefill attention on the array |

Diagnostics - nothing is measured unless they are set:

| variable | effect |
| :-- | :-- |
| `GGML_XDNA_PROF` | `1`: a table of where the time went, per section, at exit; `2`: also lists, once, the ops a single-token chunk still runs on the host |
| `GGML_XDNA_PROF_TRACE` | with `GGML_XDNA_PROF=1`, every section as `<name>\t<start ns>\t<end ns>` on the monotonic clock into this file, to line up with a power trace |

## Troubleshooting

- **`fused_layer.xclbin is present but not built for design tag ...`** - the
  artifacts and the backend are out of sync. Rebuild both:
  `cmake --build build --target ggml-xdna-kernels && cmake --build build`.
- **`ssm_out is ..., which needs the ...-bit activation layout, but the kernels
  were built with GATED_FMT=...`** - the model and the kernel build disagree.
  Reconfigure with the `-DGGML_XDNA_GATED_FMT` the message names and rebuild
  (`cmake --build build --target ggml-xdna-kernels && cmake --build build`), or
  requantize `ssm_out` to the other form (see Build). The refusal stops the
  first decode graph, so prefill still runs and, under `llama-bench`, all that
  shows is `llama_bench: error: failed to run gen warmup`.
- **The fused layer is not used at all** - no tagged `fused_layer` artifact was
  found (check the build output dir and the tag the build stamped, also written
  to `<build>/ggml/src/ggml-xdna/xdna-design-tag.h`), or
  `GGML_XDNA_FUSED_LAYER=0` is set. The flag is a bisection switch for a model
  the fused layer accepts: that decode then uses the per-op kernels. A model
  the layer refuses is not this case.
- **`unsupported fused weight set (...)`** - the model's quantization is outside
  the set the fused kernels were built for and no `GATED_FMT` value covers it
  (a `Q8_0` `ssm_out` is the common case). Requantize the tensor, e.g.
  `llama-quantize --tensor-type ssm_out=q5_K model-f16.gguf out.gguf Q4_K_M`.
  The failure has the same shape as the layout mismatch above.
  `GGML_XDNA_FUSED_LAYER=0` is not a way to run this model.
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
| `xdna-att-layer.h/.cpp` | a full-attention decode layer as one dispatch of the fused design |
| `xdna-att.h/.cpp` | the decode attention on the fused design's GEMV pool |
| `xdna-head.h/.cpp` | the decode's vocabulary projection on the pool, with its exact and 4-bit rows |
| `xdna-pgemm.h/.cpp` | the prefill GEMM in the decode GEMV's weight tiles, and what it folds in (norm, SwiGLU, gate into A) |
| `xdna-attn-mm.h/.cpp` | the prefill attention on the array |
| `xdna-gdn-mm.h/.cpp` | the prefill gated delta rule on the array, and the GDN input it is fed |
| `xdna-conv-prefill.h/.cpp` | SSM_CONV prefill runner (opt-in) |
| `xdna-norm.h` | the host RMS_NORM, rounded as ggml's CPU op rounds it |
| `xdna-prof.h` | the profiler behind `GGML_XDNA_PROF` |
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
- All llama contexts in a process share one backend context. Their graph
  computes take turns on it (the array runs one command stream at a time
  anyway).
- The device reports itself to the scheduler as an integrated GPU: its buffers
  are host memory, and the free memory it reports is all of system memory, so
  any `--fit` accounting is nominal.
