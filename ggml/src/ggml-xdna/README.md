# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that runs part of a llama.cpp graph on the AMD XDNA NPU (Ryzen
AI, XDNA2 / NPU2: Strix Point, Strix Halo, Krackan) through XRT. It is built
around one model family - Qwen3.5 with gated-delta-net layers - whose
single-token decode runs on the array layer by layer, the vocabulary
projection included, and whose prompt runs its projections, attention and
gated delta rule there too; everything else falls back to per-op kernels or
the host.

## What runs on the NPU

### MUL_MAT

Weights of type `BF16`, `F16`, `Q4_K`, `Q5_K` or `Q6_K` (`Q8_0` too on the
prefill GEMM), f32 activations and f32 result, contiguous tensors, one sequence
(`ne[2]*ne[3] == 1`), a `K` that is a multiple of 256 for the quantized types
(whole ggml super-blocks), and an `N` a route can carry. The route depends on
the number of activation rows `M`:

| M | route |
| :-- | :-- |
| 1 | decode GEMV (`kernels/gemv_q4.py` + `kernels/gemv-q4.cc`): the weights stay quantized in DDR and the group parameters are applied to the accumulator. The fused layers' own projections run on it, and so do the ones outside them (`GGML_XDNA_GEMV_GROUP`) |
| 2..63 | decode GEMM, the M32 block (`kernels/gemm.py`, native bf16 r=4), for `BF16`/`F16`/`Q4_K` |
| >= 64 | quantized weights: the prefill GEMM on the whole array (`kernels/pgemm.py`), each core expanding the decode GEMV's own packed tiles into bf16 for the mmul, one artifact for every shape (`N` up to 16384, a multiple of 8). `BF16`/`F16`, and quantized weights with `GGML_XDNA_PGEMM=0`: the baked bf16 and native int8 GEMM blocks |

The prefill GEMM also takes over the work around it where the graph allows,
each step its own switch: the FFN's SwiGLU on the cores (`GGML_XDNA_PGEMM_GLU`),
written straight into `ffn_down`'s input layout (`_GLU_A`); an RMS norm times
its weight laid out as the next projections' input in one host pass
(`_NORM_A`), with the residual ADD before it (`_ADD`); a recurrent layer's
gated output into `ssm_out`'s input (`_GATED_A`); an attention output times its
gate (`_GATE_A`).

The vocabulary projection of a decode token runs on the NPU (`GGML_XDNA_HEAD`):
re-quantized once at load to `Q4_K` and packed as q4g32, except its first 64K
rows (`GGML_XDNA_HEAD_EXACT_ROWS`), which a byte-pair vocabulary's most
frequent tokens sit in and which stay in the 8-bit form. At prefill batch size
it stays on the host, which only needs its last row.

`GGML_XDNA_W4` chooses `Q5_K`/`Q6_K` weights to re-quantize at load into the
4-bit form, for decode speed at a measured accuracy cost: by default every
`attn_qkv` and `attn_v` (47.5 -> 52.4 tok/s at 1k for decode KLD 0.0051 ->
0.0122). Set it empty to keep every weight at its own type.

### The fused decode layers

The default decode path (`GGML_XDNA_FUSED_LAYER=0` disables it). One design,
`fused_layer.xclbin`, holds a whole decoder layer of a single-token decode as
one dispatch:

- a recurrent layer: the in-projection at the head of the dispatch
  (`GGML_XDNA_INPROJ`), conv + norm + GDN + gated epilogue, `ssm_out`, and the
  FFN in the same stream (`GGML_XDNA_FUSE_FFN`). The recurrent state stays on
  the device between tokens, one session a layer that follows llama's
  recurrent cache as it switches sequences;
- an attention layer (`GGML_XDNA_ATTN_LAYER`): the qkv projection, q/k norms
  and rope, this position's K and V rows written into llama's f16 cache where
  it lies in the backend's host BOs, attention over that cache, and
  `attn_output` + the FFN. Without the whole-layer dispatch, the attention
  runs on the fused design's GEMV pool on its own (`GGML_XDNA_ATTN`) and
  `attn_output` + the FFN as one dispatch (`GGML_XDNA_ATTN_TAIL`, which the
  whole-layer dispatch needs).

The layers hand the residual to each other in rows on the device rather than
through the graph's tensors, which are written only when a node outside the
fused layers reads them. A token's layers and its vocabulary projection go
into the queue back to back and the host waits once (`GGML_XDNA_QUEUE`); by
default their runs are joined into one command, the first sent on its own so
the array has work while the host prepares the rest (`GGML_XDNA_TOKEN`).

A recurrent layer fires only when it is complete in one chunk: the conv1d,
ssm_norm, post_attention_norm, ssm_out and ffn gate/up/down weights, the qkv
and gate projections and the gate/beta/state nodes.

It covers one weight set: `Q4_K` for the FFN gate and up, `Q4_K`/`Q5_K`/`Q6_K`
for `ssm_out`, `Q4_K`/`Q6_K` for the FFN down, plus a `GGML_XDNA_GATED_FMT` that
matches the `ssm_out` layout. Anything else is refused with an error rather
than run on the wrong kernels: prefill still runs on the array, and the first
decode graph fails. Which quants hit this, and the requantization that avoids
it, are under Build below.

### Prefill ops

| op | kernel | default |
| :-- | :-- | :-- |
| `FLASH_ATTN_EXT`, from 64 tokens, plain causal mask only | `kernels/attn_mm.py` (mmul) | on, `GGML_XDNA_FA_MM=0` for the host |
| `GATED_DELTA_NET`, from 64 tokens of a sequence | `kernels/gdn_mm.py` (mmul), the state on the array for the whole ubatch | on, `GGML_XDNA_GDN_MM=0` for the host |
| its conv input: conv, SiLU, the K/Q norms | `kernels/gdn_conv.py`, written straight into the GDN's input | on, `GGML_XDNA_GDN_CONV=0` / `GGML_XDNA_GDN_IN=0` |
| `SSM_CONV`, from 64 tokens | `kernels/conv.py` | opt-in, `GGML_XDNA_CONV=1` |
| `GATED_DELTA_NET` | `kernels/gdn_prefill.py` | opt-in, `GGML_XDNA_GDN=1`, where the mmul one does not run |
| `FLASH_ATTN_EXT` | `kernels/fa.py` | opt-in, `GGML_XDNA_FA=1`, where the mmul one does not run |

The last three are the earlier kernels. Each is slower here than what replaced
it or than the host (the source says by how much), so they are kept for
coverage and comparison, not for speed.

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
`xdna-gemv.cpp` and the prefill runners), which is what lets one artifact serve
every shape.

### The array

Eight AIE columns, four compute rows each. The GEMV designs stream every
column: one xclbin is one hardware context, and the device allows 16, so the
backend warms exactly the artifacts the active mode submits while the NPU is
idle - registering a hardware context in the middle of NPU work fails, and a
context spent on an artifact nothing submits is one the used ones cannot have.
The decode's attention and vocabulary projection run on the fused layer
design's own pool for the same reason: another xclbin costs ~2.5 ms each way.

Weights are packed into q4g32 (Q4_K: 4-bit codes plus an int8 scale and min per
32 values) or q8g16 (int8 codes plus an int8 scale and min per 32 values - the
name is historical), both under a bf16 pair shared by the 256 values of a ggml
super-block. Both decode as `w = q*d + m`, so a single kernel streams either
and the GEMV never switches hardware context. They are exact with respect to
the ggml block for `Q4_K` and `Q5_K`; a `Q6_K` group merges its two scales into
the larger one. The packed weights live in a device buffer keyed by the
tensor's data pointer and shape until the last context on the backend is
freed; so do the fused-layer sessions and the decode's arena, the few large
buffers the queued designs' own buffers are carved out of. A model loaded after
that starts from nothing, even at the same addresses.

### One decode token

1. The scheduler has placed the ops here because `supports_op` claimed them.
2. `graph_compute` pre-scans the chunk and plans a fused run for every complete
   decoder layer it finds, and the vocabulary projection.
3. The dispatch loop runs the rest of the chunk: the projections outside the
   fused layers on the decode GEMV (`GGML_XDNA_GEMV_GROUP=0` keeps them on the
   host), and the host glue batched over runs of consecutive host ops.
4. At a layer's fire point its fused run is queued; the layer's own nodes are
   marked consumed and skipped. The host waits where it reads something the
   queue wrote - at the latest, the logits.

Prefill chunks take the same path with `M > 1` (the prefill GEMM and the mmul
attention and GDN, from 64 tokens); the fused layers never fire there.

## Known limitations

Measured on npu2 (Ryzen AI MAX+ 395), Qwen3.5-0.8B-Q4_K_M, llama-server with
`-np 1 -b 4096 -ub 4096 --flash-attn off`, requests at `temperature 0`,
`top_k 1` and `cache_prompt false`.

**What reproducibility has been measured.** On the default path: five fresh
server processes, each given six one-token prompts, four 1466-token prompts and
three of each interleaved, every request with `n_probs 5`. All 30 one-token
answers were one result, all 20 long ones another, the 15 interleaved long ones
one result, and the 15 interleaved one-token ones matched those given on their
own - a one-token request after a long one is answered exactly as on its own.
With `GGML_XDNA_GEMV_GROUP=0` the same series splits the one-token answers two
ways: the first request of every process differs from the ones after it, by at
most 2e-6 in a log-probability, and does so the same way in every process; the
long ones are still one result. That series was on XRT 2.25.37. Older XRT is
not supported: 2.21.75 has been seen to fail a ~1.2 GB host-memory BO
allocation at startup, and the decode was not reproducible on that
installation. The version is not checked at run time.

Measure this on the logits, not on the generated text. The perturbation a
nondeterministic read leaves is small enough that the argmax token is usually
unchanged, so the output can be byte-identical while the numbers behind it are
not. Ask `llama-server` for `n_probs` and compare those.

**The array's numbers are not the CPU's.** The weights the decode re-quantizes
(`GGML_XDNA_W4`, the vocabulary projection past its exact rows) and the bf16
the kernels compute in move the logits, so where two tokens are close the
argmax can go the other way. Greedy, against the CPU backend: the 64 tokens
after a 1466-token prompt and after "The capital of France is" are the CPU's;
after the one-token prompt `A` the first 12 of 32 are.

**The per-op decode path is a bisection switch.** With
`GGML_XDNA_FUSED_LAYER=0` the decode runs on the per-op kernels; repeated
requests in one process get the same answer and the text is coherent, but
after a short prompt it leaves the CPU's greedy path at the first token (after
the 1466-token prompt its 64 tokens are the CPU's, with top-5 log-probabilities
up to 11.6 away). Do not use it to run a model the fused layers
refuse: requantize that model instead (see Build).

**A build writes into the source tree.** `kernels/design_tag.py` regenerates
`xdna-design-tag.h`, which is tracked, so `git status` reports the working tree
as modified after every build.

## Requirements

- Linux with an AMD NPU2 (XDNA2) device; it shows up as `/dev/accel/accel0` and
  in `xrt-smi examine`.
- XRT 2.25.37 under `/opt/xilinx/xrt`, with its tools on the `PATH`. Older XRT
  is not supported: 2.21.75 has been seen to fail a ~1.2 GB host-memory BO
  allocation at startup and the decode was not reproducible on that
  installation. The version is not checked at run time.
- libuuid development files (Ubuntu: `uuid-dev`). The XRT headers include
  `<uuid/uuid.h>` and the link needs `-luuid`; XRT does not bring either.
- A Python interpreter with IRON (mlir-aie + llvm-aie) to compile the kernels.
  Verified against mlir-aie 1.4.3: the RTP buffer bases in `xdna-seq.h` are read
  back from that toolchain's placement, so a version bump needs them re-read
  from a freshly compiled project (the header says which values).
- A model the fused decode path covers (see above and the requantization under
  Build).

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
  quants differ here:
  - `qwen3.5-0.8b-q4km.gguf` leaves `ssm_out` at Q5_K: the default covers it.
  - LM Studio's leaves it at Q4_K: `-DGGML_XDNA_GATED_FMT=0`.
  - The stock `Qwen/Qwen3.5-0.8B` `Q4_K_M` leaves half of them at `Q8_0`
    (9 x Q4_K + 9 x Q8_0 on the 0.8B), which is outside the set entirely, so
    neither value helps; it fails as
    `fused layer 0: unsupported fused weight set (so=q8_0 gate=q4_K up=q4_K
    down=q6_K)`.

  Requantizing with `ssm_out` pinned to a covered type gives a model the fused
  layer accepts on the default `GATED_FMT=1`:

  ```sh
  llama-quantize --tensor-type ssm_out=q5_K Qwen3.5-0.8B-BF16.gguf out.gguf Q4_K_M
  ```

Artifacts, all in the build output directory (`build/bin` by default):

| artifact | what |
| :-- | :-- |
| `fused_layer_<tag>` | the fused decode layers, the decode attention and the vocabulary projection |
| `gemv_n32_r4_c8_<tag>` | the standalone decode GEMV |
| `pgemm_c8` | the prefill GEMM, every quantized shape |
| `attn_mm_c8` | prefill attention on the mmul |
| `gdn_mm_c8` | prefill gated delta rule on the mmul |
| `gdn_conv_c8` | the prefill GDN's conv input |
| `gemm_bf16_f32_M{32,256,512,2048}_K1024_N2048_c8` | bf16 GEMM, one per M block; M32 is the decode one |
| `gemm_int8_int32_M{256,512,2048}_K1024_N2048_c8` | native int8 GEMM, prefill |
| `gdn_prefill_bf16_S128_H16_CS64_c8` | GATED_DELTA_NET prefill, opt-in |
| `conv_prefill_bf16_c16_t256_kw4_mb8` | SSM_CONV prefill, opt-in |
| `fa_prefill_bf16_D256_MT32_JT8_NJ64_c8` | flash attention prefill, opt-in |

The GEMM M blocks come from `XDNA_GEMM_M_BIG` and `XDNA_GEMM_M_BIG_EXTRA`;
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
- Leave flash attention on (`-fa on`, or `auto`, the default). The NPU's
  attention routes take llama's `FLASH_ATTN_EXT`; with `-fa off` llama builds
  attention from `MUL_MAT` and `SOFT_MAX`, which run on the host: about half
  the decode speed on the 0.8B, with ten times the host work. The backend
  logs a warning once when it sees that.

### Environment

The switches below are for bisecting a regression rather than tuning. The ones
that default to `1` name an NPU path that `0` disables; the ones that default
to `0` name a path that `1` turns on. Either way the default is the arm the
backend is tested on. The rest are sizes and diagnostics.

Decode:

| variable | default | effect |
| :-- | :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | 1 | `0` runs the decode on the per-op kernels instead of the fused layers |
| `GGML_XDNA_INPROJ` | 1 | `0` keeps a recurrent layer's in-projection a GEMV dispatch of its own |
| `GGML_XDNA_FUSE_FFN` | 1 | `0` keeps a recurrent layer's FFN a dispatch of its own |
| `GGML_XDNA_ATTN_LAYER` | 1 | `0` stops running a whole attention layer as one dispatch |
| `GGML_XDNA_ATTN` | 1 | `0` leaves the decode attention to the host |
| `GGML_XDNA_ATTN_TAIL` | 1 | `0` stops running an attention layer's `attn_output` + FFN as one dispatch, and with it the whole layer |
| `GGML_XDNA_HEAD` | 1 | `0` leaves the vocabulary projection to the host |
| `GGML_XDNA_HEAD_EXACT_ROWS` | 65536 | the vocabulary projection's rows kept in the 8-bit form, rounded down to a pass; `0` for all 4-bit |
| `GGML_XDNA_W4` | `attn_qkv,attn_v` | `Q5_K`/`Q6_K` weights re-quantized into the 4-bit form: tensor names without `.weight` or roles; empty for none. `ssm_out` is never chosen |
| `GGML_XDNA_GEMV_GROUP` | 1 | `0` keeps the projections outside the fused layers on the host |
| `GGML_XDNA_GEMV_PROMOTE` | 1 | `0` stops one GEMV dispatch mixing weight formats |
| `GGML_XDNA_QUEUE` | 1 | `0` waits for each fused layer instead of queueing a token's layers |
| `GGML_XDNA_TOKEN` | 1 | `1` joins a token's queued runs into one command; `t > 1` at most `t` runs a command; `0` a command each |
| `GGML_XDNA_BATCH_LEAD` | 1 | runs of a token started on their own before the joined command |
| `GGML_XDNA_BATCH` | 0 | with `GGML_XDNA_TOKEN=0`: `k > 1` sends queued runs `k` to a runlist |
| `GGML_XDNA_ARENA_MB` | 256 | size of the chunks the decode's arena grows by; `0` gives every buffer its own BO |

Prefill:

| variable | default | effect |
| :-- | :-- | :-- |
| `GGML_XDNA_PGEMM` | 1 | `0` drops the prefill GEMM for quantized weights |
| `GGML_XDNA_PGEMM_GLU` | 1 | `0` stops running the FFN's SwiGLU on the prefill GEMM's cores |
| `GGML_XDNA_PGEMM_GLU_A` | 1 | `0` stops writing that SwiGLU straight into `ffn_down`'s input |
| `GGML_XDNA_PGEMM_NORM_A` | 1 | `0` stops laying a norm's output out as the next GEMM's input |
| `GGML_XDNA_PGEMM_ADD` | 1 | `0` stops folding the residual ADD before such a norm into the same pass |
| `GGML_XDNA_PGEMM_GATED_A` | 1 | `0` stops laying a recurrent layer's gated output out as `ssm_out`'s input |
| `GGML_XDNA_PGEMM_GATE_A` | 1 | `0` stops laying an attention output times its gate out as the next GEMM's input |
| `GGML_XDNA_INT8` | 1 | `0` drops the native int8 GEMM |
| `GGML_XDNA_FA_MM` | 1 | `0` leaves the prefill attention to the host |
| `GGML_XDNA_FA_MM_MIN` | 0 | (token, key) pairs below which the prefill attention stays on the host |
| `GGML_XDNA_GDN_MM` | 1 | `0` leaves the prefill gated delta rule to the host |
| `GGML_XDNA_GDN_CONV` | 1 | `0` runs the GDN's conv input on the host |
| `GGML_XDNA_GDN_IN` | 1 | `0` stops writing a prompt's conv input straight into the GDN's input |
| `GGML_XDNA_CONV` | 0 | `1` runs the prefill `SSM_CONV` on `kernels/conv.py` |
| `GGML_XDNA_GDN` | 0 | `1` runs the GDN prefill on `kernels/gdn_prefill.py` |
| `GGML_XDNA_FA` | 0 | `1` runs flash-attention prefill on `kernels/fa.py` |

Both, and diagnostics:

| variable | default | effect |
| :-- | :-- | :-- |
| `GGML_XDNA_GLUE` | 1 | `0` stops the backend claiming the host ops of an NPU chunk |
| `GGML_XDNA_GLUE_THREADS` | 4 | host-fallback threads for a decode-sized chunk |
| `GGML_XDNA_GLUE_THREADS_BIG` | 16 | host-fallback threads for a chunk of 32 tokens or more |
| `GGML_XDNA_HOST_BO` | 1 | `0` keeps llama's tensors in the plain CPU buffer type instead of XRT host BOs |
| `GGML_XDNA_SPIN_US` | 0 | microseconds to poll for a run's completion before blocking; `-1` polls without limit |
| `GGML_XDNA_SETTLE_PASSES` | 2000 | reads of a marked output before a read gives up (`xdna_buffer_mark`) |
| `GGML_XDNA_SETTLE_PAUSE` | 400 | pause loops between two such reads |
| `GGML_XDNA_SETTLE_US` | 20 | microseconds between two reads of a settled read (`xdna_buffer_read_settled`) |
| `GGML_XDNA_PROF` | unset | `1` times the NPU dispatches, the host glue and the graph calls and reports them; `2` also lists, once, the nodes a decode's glue runs on the host |
| `GGML_XDNA_PROF_TRACE` | unset | with `GGML_XDNA_PROF=1`: every section and glue flush into this file, a tab-separated name, start and end in ns on the steady clock |
| `GGML_XDNA_ATTN_CHECK` | unset | set: runs the host's decode attention next to the NPU's and prints the difference; the host's result is kept unless `npu` |
| `GGML_XDNA_ATTN_SKIP` | unset | set (to anything): decode attention moves its data but skips the arithmetic: what the DMA alone costs; the output is garbage |

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
  set the fused kernels were built for. Prefill runs, then the first decode
  graph fails; under `llama-bench` the `ggml-xdna` line is not shown and all
  that is printed is `failed to run gen warmup`. Requantize as shown under
  Build. `GGML_XDNA_FUSED_LAYER=0` is a bisection switch, not a way to run it
  (see the limitations).
- **No NPU** - the backend still registers (llama.cpp expects every accelerator
  device to answer) but claims no ops, so everything runs on the CPU.

## Checks

- `test-xdna-repack` (ctest, no device) - which (type, format) repacks are
  accepted, and that a refused one writes nothing and an accepted one exactly
  its row.
- `tests/test-xdna-reload -m model.gguf` - load, run and free a model several
  times in one process; the tokens must match every cycle and the device
  buffers held after a free must not grow.
- `kernels/gemm.py --run`, `kernels/gemv_q4.py`, `kernels/gdn_prefill.py`
  (also `--S 64`) - each kernel against NumPy, nonzero exit on a mismatch.
- `probes/` - the checks and tools the designs were built with, kept as a
  record rather than as tests; each script's header says how to run it.
  Against the current designs, `fa_design_check.py`, `gdn_design_check.py`
  and `gdn_conv_check.py` run the whole-array prefill attention, gated delta
  rule and conv input against NumPy or ggml's recurrence, and pass;
  `gdn_chunk_ref.py` checks the chunked recurrence on the host. The one-core
  and one-pair checks (`expand_check.py`, `pair_check.py`, `pgemm_check.py`,
  `fa_pair_check.py`, `gdn_pair_check.py`, `act_att_check.py`,
  `attn_dec_check.py`) read the kernel sources directly, so they have not
  compiled since those include the shared headers (`xdna-math.h`,
  `xdna-vec.h`), and `pgemm_design_check.py` times out on the device.
  `mm_check.py`, `pair_speed.py`, `power_trace.py`, `xclbin_core_programs.py`
  and `head_probe.cpp` are measuring tools.

The kernel scripts need XRT's environment (`source /opt/xilinx/xrt/setup.sh`):
without `pyxrt` IRON cannot see the NPU and compiles for its default
architecture.

## Code layout

| file | role |
| :-- | :-- |
| `ggml-xdna.cpp` | backend/device/registry scaffolding, graph dispatch, the fused-layer and prefill planning, the glue |
| `xdna-types.h` | XRT-backed structs: device, kernel, buffer, kernel pool |
| `xdna-runtime.h/.cpp` | XRT primitives: device, kernel load and insts bind, host buffers and the decode's arena, submit/wait, joined token commands, pool |
| `xdna-util.h` | small helpers shared by more than one translation unit |
| `xdna-prof.h` | the profiler behind `GGML_XDNA_PROF` |
| `xdna-seq.h/.cpp` | TXN instruction-stream builder and the GEMM sequence |
| `xdna-seq-attn.cpp` | the fused core's phased per-token TXN stream |
| `xdna-ops.h/.cpp` | per-op dispatch: GEMM/GEMV support, compute, finalize; decode GEMV grouping |
| `xdna-gemv.h/.cpp` | decode GEMV: packed weight formats, stream builder, runner, FFN pair, residual rows |
| `xdna-quant.h/.cpp` | NPU weight formats (q4g32 / q8g16) and the ggml repack |
| `xdna-rec.h/.cpp` | the fused recurrent layer: session state, packing, core stream |
| `xdna-rec-gemv.h/.cpp` | the fused layer's projections (ssm_out, FFN) on the decode GEMV |
| `xdna-att.h/.cpp` | decode attention on the fused design's GEMV pool |
| `xdna-att-layer.h/.cpp` | a decode attention layer as one dispatch |
| `xdna-head.h/.cpp` | the vocabulary projection on the NPU |
| `xdna-norm.h` | the host's vectorized RMS norm, rounded as ggml's CPU op rounds it |
| `xdna-pgemm.h/.cpp` | the prefill GEMM and the layouts it reads its input in |
| `xdna-attn-mm.h/.cpp` | prefill attention on the mmul |
| `xdna-gdn-mm.h/.cpp` | prefill gated delta rule on the mmul, and its conv input |
| `xdna-gdn-prefill.h/.cpp` | GDN prefill runner (opt-in) |
| `xdna-conv-prefill.h/.cpp` | SSM_CONV prefill runner (opt-in) |
| `xdna-fa-prefill.h/.cpp` | flash-attention prefill runner (opt-in) |
| `kernels/*.py` | IRON/MLIR-AIE designs: the array configuration, the fifos and the runtime sequence |
| `kernels/*.cc`, `kernels/*.h` | the device C++ those designs compile in |
| `probes/` | the design checks and measuring tools under Checks |

## Known limits

- The fused decode layer is written around one geometry - 6144 conv channels,
  1024 `d_out`, 16 value heads of 128, a 262144-float recurrent state - and the
  designs are compiled for it, so another GDN model needs the designs rebuilt
  and the constants in `xdna-rec.h` revisited. The prefill attention and GDN
  designs are written for Qwen3.5's shapes in the same way.
- The host-built instruction streams and the RTP/shim constants follow the
  mlir-aie placement. The design tag does not cover the toolchain version, so a
  toolchain bump has to be paired with re-reading those constants.
- All llama contexts in a process share one backend context. Their graph
  computes take turns on it (the array runs one command stream at a time
  anyway).
- The device reports itself to the scheduler as a GPU while its buffers are host
  memory, so the reported free memory and any `--fit` accounting are nominal.
