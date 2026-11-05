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
| 1 | decode GEMV (`kernels/gemv_q4.py` + `kernels/gemv-q4.cc`): the weights stay quantized in DDR and the group parameters are applied to the accumulator; in the fused layers a `Q5_K`/`Q6_K` `attn_qkv` is taken at 4 bits by default (see [4-bit decode weights](#4-bit-decode-weights-on-by-default)) |
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
matches the `ssm_out` layout. A model with any other format in those weights
is not handed to the fused layers: one warning names the first such weight, and
its decode runs as for a width they do not take (below). When `ssm_out` is the
only such weight - a stock `Qwen/Qwen3.5-0.8B Q4_K_M` leaves half of its
`ssm_out` at `Q8_0` - requantizing it is the way onto the fused path (see
Build). `GGML_XDNA_FUSED_LAYER=0` is a bisection switch, not a way to run such
a model on the fused layers.

The layers take the residual width from the model: any multiple of 512 up to
4096 (`XDNA_RES_MAX_D`, what the residual rows the layers hand each other are
sized for). They are tested at four - Qwen3.5-0.8B's 1024, the 2B's 2048, the
4B's 2560 and the 9B's 4096 - and the other widths they take are untested.
Past 1024 the projections take more than one pass
of the pool (the FFN's down projection, `ssm_out` and the attention output
each two at 2048).
`GGML_XDNA_FUSED_WIDE=0` keeps them to 1024. Past 1024 they take a model only
when its layers are Qwen3.5's: gated delta net layers of 16 or 32 value heads
with a dense FFN, and attention of 256-wide heads in one or two groups of
eight. A dense model (Qwen3-4B, Qwen3-8B) or a mixture of experts
(Qwen3.6-35B-A3B) is one they do not take. A model they do not take is not
handed a recurrent layer at all, and one warning names its width, or the layers
it lacks, when it happens. Its prefill keeps the array as far as the kernels
take its shapes.
Its decode (one token a sequence) runs whole on the host instead, because the
decode GEMV streams weights at ~31 GB/s, less than the host's cores draw.
One thread on every allowed core computes it, pinned one a core
(`GGML_XDNA_HOST_DECODE_THREADS` sets another count); unpinned, 8 threads do.
Against the array's per-op decode on a Ryzen AI MAX+ 395 (16 cores, with
`OMP_WAIT_POLICY` unset, see [Running](#running)), the 2B Q4_K_M goes from 20.1
to 61.9 t/s at 0.94 instead of 1.69 J a token, the 4B Q4_K_M from 9.6 to 30.1
t/s at 2.4 instead of 4.1 J, the 9B Q4_K_M from 6.4 to 18.8 t/s at 3.4 instead
of 6.4 J, and the 9B Q8_0 from 6.4 to 12.4 t/s at 4.7 instead of 10.5 J.

The fused layers decode these Q4_K_M models slower than that host decode, for
less package energy (tg512 in one session, J a token above idle, then in
total): the 2B at 30.0 against 63.6 t/s and 0.51 against 0.82 J (0.91 against
1.01), the 4B at 12.3 against 29.8 t/s and 1.10 against 1.93 J (2.07 against
2.34), and the 9B at 7.6 against 18.6 t/s and 1.77 against 2.78 J (3.34
against 3.42). `GGML_XDNA_FUSED_WIDE=0` restores the faster host decode.

`GGML_XDNA_HOST_DECODE_THREADS=0` keeps that decode on the array per op: the
recurrent bodies (conv, GDN, norms, GLU) run on the host, the projections on
the decode GEMV, the decode attention on the pool, and the head goes where
`GGML_XDNA_HEAD` puts it. `GGML_XDNA_FUSED_LAYER=0` still forces the per-op
path for a model the fused design does cover.

### Prefill ops

| op | kernel | default |
| :-- | :-- | :-- |
| `SSM_CONV`, from 64 tokens | `kernels/conv.py` | opt-in, `GGML_XDNA_CONV=1`; the host is cheaper and is what runs otherwise |
| the prefill GDN's conv input | `kernels/gdn_conv.py` | opt-in, `GGML_XDNA_GDN_CONV=1`; the host is faster (pp512 1900 against 1680 t/s on the 0.8B) and is what runs otherwise |
| `GATED_DELTA_NET`, from 144 tokens a sequence | `kernels/gdn_mm.py` | on, `GGML_XDNA_GDN_MM=0` for the host; below 144 the host is quicker at less energy a token |
| `FLASH_ATTN_EXT` without a K/V row list (`ggml_flash_attn_ext_rows`), above 512 x 512 (token, key) pairs a call | `kernels/attn_mm.py` | on, `GGML_XDNA_FA_MM=0` for the host; up to that the host is quicker at no more energy a token |

### Everything else

The backend also claims the cheap contiguous f32 ops of an NPU chunk and runs
them on the host from inside that chunk (the "glue"). The point is not speed:
it keeps the scheduler from splitting the recurrent subgraph at every backend
change, which would put a layer's inputs back through the graph allocator
between NPU dispatches. `GGML_XDNA_GLUE=0` claims only the native NPU ops.

The glue runs on its own CPU backend with its own pool: `GGML_XDNA_GLUE_THREADS`
(4) for decode-sized chunks, `GGML_XDNA_GLUE_THREADS_BIG` (16) once a chunk has
32 tokens or more, and 12 (at most `GGML_XDNA_GLUE_THREADS_BIG`) for a decode's
vocabulary projection. `-t` does not size it.

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

Most of this backend is original work, `Copyright (c) 2026 Tether Operations,
S.A. de C.V.`, under the MIT licence in [`LICENSE`](LICENSE). Two parts are worth
stating plainly.

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
`Apache-2.0 WITH LLVM-exception` (mlir-aie 1.4.3). The file carries that notice
and a notice of the changes, and the licence text ships beside it in
`LICENSES/Apache-2.0-WITH-LLVM-exception.txt`. The other designs under
`kernels/` use the published IRON API only.

[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) lists what is derived from what,
and why the compiled `.xclbin` / `.insts.bin` need nothing added (the LLVM
exception covers compiled output).

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

One step is lossy, and it is on by default by design: for the single-token
decode, every `Q5_K` or `Q6_K` weight named `attn_qkv` is re-quantized to
`Q4_K` from its dequantized values when it is packed, and the fused recurrent
layers stream it at 4 bits (q4g32) instead of the exact 8-bit form. It buys the
most decode speed per unit of accuracy lost. Teacher-forced decode KLD against
the CPU over 511 tokens, and tg512 on the fused layers (measured before the z
row reuse), on Qwen3.5 Q4_K_M (the CPU's own one-token decode scores 0.0029 on
the 9B, 0.0018 on the 4B, 0.0053 on the 2B and 0.0043 on the 0.8B):

| model | weights at 4 bits | KLD | t/s | J a token, above idle / total |
| :-- | :-- | --: | --: | --: |
| 9B | `attn_qkv` (default) | 0.0061 | 7.42 | 1.73 / 3.42 |
| 9B | none (`GGML_XDNA_W4=`) | 0.0033 | 7.17 | 1.77 / 3.53 |
| 4B | `attn_qkv` (default) | 0.0047 | 11.99 | 1.11 / 2.11 |
| 4B | none (`GGML_XDNA_W4=`) | 0.0028 | 11.57 | 1.15 / 2.18 |
| 2B | `attn_qkv` (default) | 0.0097 | 30.09 | 0.51 / 0.91 |
| 0.8B | `attn_qkv` (default) | 0.0099 | 58.63 | 0.25 / 0.46 |

Greedy text can differ from the exact form where two tokens are close.
`attn_v` is not worth it: a fused attention layer streams q|gate, k and v in
one format, and with `attn_v` at 4 bits as well the 9B reaches KLD 0.0091 for
1.5% more t/s and no less energy above idle.

It does not touch:
- the per-op decode GEMV, which takes every weight in its own format;
- the prefill GEMM, which does the same (4 bits there would cost prefill KLD
  0.0041 -> 0.0105);
- `Q4_K` tensors, which are 4-bit already;
- `ssm_out`;
- the weights in the model file.

`GGML_XDNA_W4` controls it:
- unset: `attn_qkv`, the default above;
- set and empty (`GGML_XDNA_W4=`): every weight keeps its exact 8-bit form, at
  the slower decode;
- a comma-separated list (`GGML_XDNA_W4=attn_v`): the `Q5_K`/`Q6_K` weights it
  applies to, by the last part of their name (`blk.N.<name>.weight`).

The decode head (the vocabulary projection) runs on the host by default, on
12 glue threads, in the model's own format: exact, and quicker than the array
on the 0.8B (tg256 57.8 against 51.4 t/s, at 0.454 against 0.437 J a token
of package energy). `GGML_XDNA_HEAD=1` puts it on the array, where it is
lossy: its first 65536 vocabulary rows - a byte-pair vocabulary's most
frequent tokens - stay at 8 bits and the rest are 4-bit. Against the CPU's
`Q6_K` head on the 0.8B that is a head KLD of 0.0052, where every row at 4 bits
would be 0.0165. `GGML_XDNA_HEAD_EXACT_ROWS` sets the number of exact rows
(`0`: all 4-bit).

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
- For the fused decode, a model whose weight set it covers (see above) and,
  past 1024, whose layers are Qwen3.5's; any other decodes as a model of a
  width it does not take, on the host by default.
  Requantizing `ssm_out` (see Build) moves a model onto the fused path when it
  is the only weight outside the set.

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
- `GGML_CPU` has to stay on (the default) unless `GGML_BACKEND_DL=ON`: the host
  ops inside an NPU chunk run on the CPU backend, and only a DL build compiles
  the backend its own copy. The configure stops otherwise.
- `GGML_XDNA_GATED_FMT` (default 1) selects the activation layout the gated
  stage writes: `0` is the 4-bit form for a `Q4_K` `ssm_out`, `1` the 8-bit form
  for `Q5_K`/`Q6_K`. It is compiled into the artifact and into the backend, and
  a model whose `ssm_out` needs the other one decodes on the host instead, with
  a warning that names the tensor. Which type a
  quant ends up with is a property of the quant, not of its name:
  `llama-quantize ... Q4_K_M` built from this tree leaves `ssm_out` at `Q4_K`
  (so the fused layer needs `-DGGML_XDNA_GATED_FMT=0`), while other `Q4_K_M`
  conversions leave it at `Q5_K` (the default). Check the model with
  `llama-gguf model.gguf r n | grep ssm_out`, or read the warning the backend
  prints. To keep the default `GATED_FMT=1`, requantize that one tensor:

  ```sh
  llama-quantize --tensor-type ssm_out=q5_K model-f16.gguf out.gguf Q4_K_M
  ```

  A `ssm_out` outside `Q4_K`/`Q5_K`/`Q6_K` is not taken whatever `GATED_FMT` is
  - a stock `Qwen/Qwen3.5-0.8B Q4_K_M` leaves half of them at `Q8_0`, and its
  decode runs on the host. The `--tensor-type ssm_out=q5_K` recipe above is the
  way onto the fused path. `GGML_XDNA_FUSED_LAYER=0` does not change that: it
  only moves a model the fused layer takes onto the per-op kernels, for a
  bisection.

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
  `OMP_WAIT_POLICY=PASSIVE` does the same for the host's OpenMP threads. On the
  0.8B it gives 3% less energy a decoded token and 11% less a prompt token; the
  fused 2B, 4B and 9B are measured with it unset. Leave it unset when a model's
  decode runs on host threads (a model the fused layers do not take, or any
  above 1024 with `GGML_XDNA_FUSED_WIDE=0`): it puts them to sleep between ops
  (the 9B Q4_K_M on the host decodes at 18.8 instead of 16.1 t/s and 3.4
  instead of 4.4 J a token without it).
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
| `GGML_XDNA_W4` | `attn_qkv` | the `Q5_K`/`Q6_K` weights the fused decode layers take at 4 bits, a lossy default (see [4-bit decode weights](#4-bit-decode-weights-on-by-default)); empty keeps them at 8 bits |
| `GGML_XDNA_HEAD` | 0 | `1` runs the decode's vocabulary projection on the array instead of the host: lossy (see `GGML_XDNA_HEAD_EXACT_ROWS`) and slower on the 0.8B, with fewer host threads busy |
| `GGML_XDNA_HEAD_EXACT_ROWS` | 65536 | with `GGML_XDNA_HEAD=1`, the vocabulary rows the decode head keeps at 8 bits, from row 0 (a byte-pair vocabulary's most frequent tokens); the rest are 4-bit. Head KLD against the CPU's Q6_K head on the 0.8B: 0.0052 at 65536, 0.0165 with `0` (all 4-bit) |
| `GGML_XDNA_TOKEN` | 1 | how a token's queued runs go to the device: `1` as one joined command, `t > 1` at most `t` runs a command, `0` a command each - slightly faster decode at a higher package power (24.0 W at 50.9 t/s against 18.0 W at ~48 t/s with the default, 1k context) |
| `GGML_XDNA_SPIN_US` | 0 | poll this many microseconds for completion before blocking; `-1` polls until it completes |
| `GGML_XDNA_GLUE_THREADS` | 4 | host threads for the ops the backend runs on the host in a decode-sized chunk, except the vocabulary projection, which takes 12 (at most `GGML_XDNA_GLUE_THREADS_BIG`) |
| `GGML_XDNA_GLUE_THREADS_BIG` | 16 | the same, for a chunk of 32 tokens or more |
| `GGML_XDNA_HOST_DECODE_THREADS` | one a core | host threads for the decode of a model the fused layers do not take, by its width, its weight set or its layers (see [The fused decode recurrent layer](#the-fused-decode-recurrent-layer)): one on every allowed core when they are pinned, 8 when they are not; `0` runs that decode per op on the array, slower at more energy a token |
| `GGML_XDNA_HOST_DECODE_PIN` | 1 | in an OpenMP build, pins those threads one a core, filling one L3 domain before the next, and hands every borrowed thread its CPU mask back after each decode; `0` leaves 8 threads to the OS, slower (the 4B: 18.7 against 30.1 t/s) |
| `GGML_XDNA_FUSED_WIDE` | 1 | the fused layers take a model with Qwen3.5's layers at any residual width that is a multiple of 512 up to 4096 (tested: Qwen3.5-2B, 4B and 9B; the other widths are untested); `0` or an empty value keeps them to 1024 (the 0.8B) and runs a wider model's decode on the host (see [The fused decode recurrent layer](#the-fused-decode-recurrent-layer)) |
| `GGML_XDNA_ARENA_MB` | 256 | size of each chunk of the device-buffer arena, in MiB |
| `GGML_XDNA_CONV` | 0 | `1` runs the standalone prefill `SSM_CONV` on the array; the host is cheaper, which is why it is off |
| `GGML_XDNA_GDN_CONV` | 0 | `1` computes the prefill GDN's conv input on the array; the host is faster, which is why it is off |
| `GGML_XDNA_KERNEL_DIR` | unset | a directory to look for the kernel artifacts in before the others (see Artifacts) |

For bisecting a regression - each `0` turns off the NPU route it names, and the
host (or a less fused path) runs it instead. They are for narrowing down a bug
report, not for tuning: every one costs more time or energy than the default.

| variable | `0` turns off |
| :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | the fused decode layers (the per-op path runs). A model whose weight set they do not take decodes on the host either way |
| `GGML_XDNA_GEMV_GROUP` | the decode projections outside the fused layer, on the array |
| `GGML_XDNA_GEMV_PROMOTE` | merging a decode GEMV group of mixed weight formats into one dispatch |
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
- **`the fused decode layers read ssm_out as ...; this model's ... is ...: its
  decode runs on the host`** - a weight the fused kernels do not read: the
  model decodes on the host instead of the fused layers. For a
  `Q4_K` `ssm_out` reconfigure with `-DGGML_XDNA_GATED_FMT=0` and rebuild
  (`cmake --build build --target ggml-xdna-kernels && cmake --build build`), or
  requantize `ssm_out` to `Q5_K` (see Build); a `Q8_0` one takes the
  requantize, e.g.
  `llama-quantize --tensor-type ssm_out=q5_K model-f16.gguf out.gguf Q4_K_M`.
  An FFN outside `Q4_K` gate/up and `Q4_K`/`Q6_K` down has no rebuild that
  takes it.
- **`the fused decode layers take Qwen3.5's gated delta net and attention layers
  with a dense FFN, this model's ...: its decode runs on the host`** - a model
  wider than 1024 whose layers the fused layers do not run (a dense model, a
  mixture of experts, other head counts): it decodes on the host. No rebuild or
  requantize puts it on the fused layers.
- **The fused layer is not used at all** - no tagged `fused_layer` artifact was
  found (check the build output dir and the tag the build stamped, also written
  to `<build>/ggml/src/ggml-xdna/xdna-design-tag.h`), or
  `GGML_XDNA_FUSED_LAYER=0` is set. The flag is a bisection switch for a model
  the fused layer accepts: that decode then uses the per-op kernels. A model
  whose weight set the layer does not take prints the warning above instead.
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
| `xdna-match.h` | how the graph planners recognise the nodes llama builds around a layer |
| `xdna-prof.h` | the profiler behind `GGML_XDNA_PROF` |
| `kernels/*.py` | IRON/MLIR-AIE designs: the array configuration, the fifos and the runtime sequence |
| `kernels/*.cc` | the device C++ those designs compile in |

## Known limits

- The fused decode layer is written around one family of geometries - 16 q/k
  groups of 128, 16 or 32 value heads of 128 (6144 or 8192 conv channels; the
  core runs once per 16 value heads, on a 262144-float recurrent state each
  time), a residual width that is a multiple of 512 up to 4096 - and the
  designs are compiled for it, so another GDN model needs the designs rebuilt
  and the constants in `xdna-rec.h` revisited.
- The prefill gated delta rule forms a chunk's decay factors as two powers of
  two about the chunk's middle token, which hold a decay of 2^126 across half a
  chunk. A chunk that decays faster has its gates floored at -10.4 (natural
  log, a decay of 3e-5 a token) before it goes to the array, and that head's
  output then differs from the host's by about 3e-5 relative. On the test
  prompt this touches only layer 8 head 15 of the 4B and the 9B; every other
  chunk runs as before.
- An eval callback (`llama-eval-callback`, `imatrix`) that watches a recurrent
  layer's `final_output` or `linear_attn_out` in a prefill splits the graph
  between the gated norm and `ssm_out`, and the array's outputs at those
  tensors come back empty, so the tool reads garbage there.
- The host-built instruction streams and the RTP/shim constants follow the
  mlir-aie placement. The design tag does not cover the toolchain version, so a
  toolchain bump has to be paired with re-reading those constants.
- All llama contexts in a process share one backend context. Their graph
  computes take turns on it (the array runs one command stream at a time
  anyway).
- The first decode call sets up every fused layer: it packs the layer's weights
  into the fused formats, builds the layer's instruction stream and loads its
  kernel. This takes about 3.2 s on Qwen3.5-4B and 6.4 s on Qwen3.5-9B, in that
  call only (`llama-bench` tg numbers exclude it). An attention layer also
  builds one more kernel the first time its K/V span reaches a new task count.
- The device reports itself to the scheduler as an integrated GPU: its buffers
  are host memory, and the free memory it reports is all of system memory, so
  any `--fit` accounting is nominal.
