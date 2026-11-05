# ggml-xdna: AMD XDNA (NPU) backend

A ggml backend that runs part of a llama.cpp graph on the AMD XDNA NPU (Ryzen
AI, XDNA2 / NPU2: Strix Point, Strix Halo, Krackan) through XRT. It is built
around one model family - Qwen3.5 with gated-delta-net layers - whose
single-token decode runs almost entirely on the array; everything else falls
back to per-op kernels or the host. That is also how it runs the EmbeddingGemma
2 embedding model (see [Embedding models](#embedding-models)).

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
| >= 64 | prefill GEMM on a baked M block, one for all the graph's GEMMs of a route and row count: the block of least summed cost, since each block is its own xclbin and a change of block costs a hardware-context switch; bf16 for `F16`/`BF16`/`Q4_K`, and the native int8 route (1 B/value) for `Q4_K`/`Q5_K`/`Q6_K` |

The GEMM artifacts are compiled at K1024/N2048 but the runtime drives any `K`
that is a multiple of 64 and any `N` that is a multiple of 512 (the stream
builder writes the loop counts per dispatch), so only the baked M block is
fixed: one artifact per M block. A wide `N` such as the vocabulary projection
stays on the host.

The bf16 route takes `K` in runs of 512, and the host adds the partial sums of
a deeper `K`. The cores read their K-loop count from the stream of each run,
and a run whose count differs from the previous run's on the same hardware
context hangs the context, so every run has the same width: a `K` that is not a
whole number of them ends in a run zero-padded out to 512. A block of A rows
runs only the sweeps its rows fill (`tile_m` rows on each of the four compute
rows); each sweep reads the weights from DDR once. The stream issues the next
sweep's descriptors before it waits for the previous sweep's output (two sweeps
in flight, `GEMM_SWEEP_WINDOW`); blocks of three or more sweeps run faster for
it, one and two sweeps are unchanged.

The device's writes to C are not cache coherent, so the host marks the rows a
run will write with a pattern before the submit (non-temporal stores, then a
cache write-back, so no line is fetched only to be overwritten) and reads a row
chunk only once the pattern is gone. C starts one page into its buffer, and a
block whose first C line stays marked is run again, up to twice, before the op
fails (see Known limits).

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

The layers take the residual width from the model; the residual rows the
layers hand each other are sized for 4096 (`XDNA_RES_MAX_D`). By default they
take a width of 1024, the 0.8B's. `GGML_XDNA_FUSED_WIDE=1` also gives them any
wider stream up to 2048 that is a multiple of 512 - Qwen3.5-2B is 2048 - whose
projections then take more than one pass of the pool (the FFN's down
projection, `ssm_out` and the attention output each two at 2048). Without it,
and for any wider model - the 4B is 2560, the 9B 4096 - the model is not handed
a recurrent layer at all, and one warning names the width when it happens. Its
prefill keeps the array as far as the kernels take its shapes. Its decode (one
token a sequence) runs whole on the host instead, because the decode GEMV
streams weights at ~31 GB/s, less than the host's cores draw.
One thread on every allowed core computes it, pinned one a core
(`GGML_XDNA_HOST_DECODE_THREADS` sets another count); unpinned, 8 threads do.
Against the array's per-op decode on a Ryzen AI MAX+ 395 (16 cores, with
`OMP_WAIT_POLICY` unset, see [Running](#running)), the 2B Q4_K_M goes from 20.1
to 61.9 t/s at 0.94 instead of 1.69 J a token, the 4B Q4_K_M from 9.6 to 30.1
t/s at 2.4 instead of 4.1 J, the 9B Q4_K_M from 6.4 to 18.8 t/s at 3.4 instead
of 6.4 J, and the 9B Q8_0 from 6.4 to 12.4 t/s at 4.7 instead of 10.5 J.

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
| `FLASH_ATTN_EXT` of 2 KV heads and D = 256 without a K/V row list (`ggml_flash_attn_ext_rows`), from 64M visible (query, key, head) triples a call (8M where ggml-cpu's op would run it) | `kernels/attn_mm.py` | on, `GGML_XDNA_FA_MM=0` for the host, `=2` for the array whatever the size; below that the host is quicker: a call pays two hardware-context switches (~5 ms) |

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

### The host or the array

Every array command costs a fixed floor whatever it computes, and a layer is a
serial chain of commands with host ops between them, so no floor hides behind
another. For a prefill the backend therefore compares what its commands cost
with what the host takes for the same products (`xdna_ops_plan_chunk`,
`xdna_ops_plan_gemm`), and the cheaper side runs. Both sides come from one
model:

- The array's prefill GEMM costs a command ~105 us at the least, ~30 us of host
  work around it (the layout of A, the start, the glue batch it splits), and
  ~21 us for each 1024 columns of N at K 512 for the first block of 128 rows and
  ~13.6 us for each further block, in proportion to K (`xdna_pgemm_command_ms`).
  A command runs `ceil(M / 128)` blocks whatever rows of the last one are real.
  A bf16 GEMM costs a run ~0.1 ms and ~23 us a sweep of rows over 512 columns,
  plus two hardware-context switches (~2.5 ms each) when it sits among pgemm
  commands: one in and one out.
- The host's time is ggml-cpu's MUL_MAT: `a + bytes * per_byte + 2MKN *
  per_flop + MK * per_mk + MN * per_mn`, one set of coefficients for each
  weight type (Q8_0, BF16, Q4_K, Q5_K, Q6_K) and for a team of 8 or more threads
  and fewer (`xdna-host-cost.h`), fitted with `GOMP_SPINCOUNT=20000`, the wait
  policy of the run line. The mean error over the measured shapes is 4 to 12%,
  the largest 16 to 39%. A weight type the model does not cover (F16 among
  them) keeps its array route.
- The projections of one activation (a layer's Q, K and V, its gate and up, a
  recurrent layer's alpha and beta) are one command of the array, and are
  costed as one, against the sum of the host's products.

`probes/pgemm_probe.cpp` and `probes/host_probe.cpp` measure the two sides.
Three rules follow:

- **A chunk runs whole on the host** when its array commands cost more than
  the host's products: by any amount within one block of 128 rows, where each
  command's cost is its measured floor, and by more than the fits' largest
  error (39%) above it, where both sides are estimates. Its rows must be within
  the fit of every weight type it holds: up to 2560 rows for Q8_0 and BF16, 128
  for the K-quants. It runs as CPU graphs on `GGML_XDNA_GLUE_THREADS_BIG`
  threads (16), with the backend's own kernels (the attention, the GDN's
  conv-input concat, SSM_CONV) between them on the same thread team, since
  alternating team sizes make the OpenMP runtime rebuild its team. A chunk with
  other array work (the array's attention, GDN or SSM_CONV, the int8 route, a
  weight type the model does not cover) keeps its commands.
  `GGML_XDNA_HOST_CHUNK=0` turns the rule off.
- **A pgemm command runs on the host** when its chunk keeps array work, the
  command has 2 to 128 rows, and the host does its products quicker on the
  glue's team (`GGML_XDNA_GLUE_THREADS` below 32 tokens a sequence,
  `GGML_XDNA_GLUE_THREADS_BIG` from 32). `GGML_XDNA_PGEMM_HOST=0` keeps it on
  the array.
- **A run of bf16 projections among pgemm commands** goes whole to the host or
  stays whole, whichever is cheaper counting its two switches
  (`GGML_XDNA_GEMM_HOST=0` keeps it on the array). The Q8_0 EmbeddingGemma 2
  file's `per_layer_model_proj` (K 512, N 12288) costs 3.0 ms on the array at
  15 tokens with the switches against 0.4 ms on the host, 3.3 against 0.75 at
  132 and 6.1 against 4.4 at 912. Among bf16 commands, as in the BF16 file, the
  projections pay no switch and stay.

The first two rules plan a prefill only: a step of one token a sequence is a
decode, however many sequences share the graph. The third is planned for any
step of more than one row, so a decode of several sequences takes it too.

### Embedding models

EmbeddingGemma 2 (`gemma-embedding2`, the `ggml-org/embeddinggemma-2-GGUF`
Q8_0 and BF16 files) runs without a model-specific path:

- Its weight projections take the [MUL_MAT](#mul_mat) routes, or the host's
  where [the cost model](#the-host-or-the-array) says the host is quicker. The
  `Q8_0` weights go through the prefill GEMM, repacked to the 8-bit form with
  one extra rounding of the codes. The `BF16` weights go through the bf16 GEMM,
  except the BF16 file's `output`: its N of 768 is not a multiple of 512, and
  zero-padded to 1024 columns it is no faster than the host's 16 threads and
  lowers the pooled cosine by 2e-5 to 6e-5, so it stays on the host.
- The prefill GEMM writes its result straight into the `MUL_MAT` node in whole
  blocks of 128 rows. The backend's host buffer type reports those rows in the
  node's allocation size (`get_alloc_size`), so no row is copied out of a
  staging buffer (`GGML_XDNA_HOST_BO=0` keeps the staging).
- The FFN's gate and up projections run as one prefill GEMM whose cores apply
  the GEGLU (the tanh-approximated GELU of the gate times the up) and write the
  down projection's input in bf16, so the activation never goes to the host
  (`GGML_XDNA_PGEMM_GLU=0` and `GGML_XDNA_PGEMM_GLU_A=0` turn the two steps
  off). The activation is evaluated at bf16, which is what the down projection
  reads of it. Qwen3.5-4B-Q8_0's prefill KLD against the CPU with and without
  the fold differs by less than the standard error (100 chunks of 512 tokens:
  0.00513 +- 0.00080 and 0.00496 +- 0.00075).
- A layer's Q, K and V projections run as one prefill GEMM that writes each
  output into its own node (`GGML_XDNA_PGEMM_GROUP=0`: one GEMM each). ggml
  places K and V after Q's norm and RoPE, so the backend's `graph_optimize`
  moves the projections of one activation up behind the first, before the
  scheduler allocates the graph, whatever its rows: a decode's graph keeps the
  node order of the prompt's whose allocation it reuses. All but the last end
  on a whole 512 columns, and every node must lie in one of the backend's
  buffers. On the bf16 GEMM, a
  layer's Q, K and V and its gate and up run as one GEMM the same way, their
  weights packed side by side along N (`GGML_XDNA_GEMM_GROUP=0`: one GEMM
  each).
- The F32 to F16 casts of K and V before each attention are host glue, as are
  casts to F32, so they do not split the graph (`GGML_XDNA_GLUE_CAST=0` leaves
  the F32 to F16 casts to the scheduler).
- The attention runs on the backend's own host kernel (`xdna-host-attn.cpp`;
  `GGML_XDNA_HOST_FA=0` runs ggml-cpu's op): AVX-512 tiles of f32 FMAs over f16
  K and V that read each row's visible keys off the mask, so a sliding window
  and packed sequences cost only the keys they show. Its results match
  ggml-cpu's to an NMSE of 5e-14 to 1.6e-12 from 64 queries up and ~1e-6 below,
  where ggml-cpu's own f16 path is the inexact one. At 912 tokens a
  sliding-window layer takes ~1.0 ms and a full-attention layer (heads of 512,
  one KV head) ~2.2 ms, against ggml-cpu's 2.4 and 4.9 ms. It takes 8 queries
  and up; a CPU without AVX-512 keeps ggml-cpu's op.
- The array's prefill attention kernel (4 query heads over 2 KV heads of 256, a
  window and the packed sequence) takes a call from 64M visible (query, key,
  head) triples, about an input of 16k tokens, where it repays its two
  hardware-context switches (~5 ms). It reads the mask of a graph input only.
- The norms, RoPE, the mean pooling and, in the BF16 file, the GEGLU are host
  glue. An input of one token runs whole on the host: the model's residual
  width is 512, and the fused decode layers take 1024.

Run it with the whole input in one ubatch, which non-causal attention needs:

```sh
GOMP_SPINCOUNT=20000 ./build/bin/llama-embedding -m embeddinggemma-2-Q8_0.gguf \
    --device XDNA0 -ngl 99 -fa on --poll 0 -c 2048 -b 2048 -ub 2048 -p "..."
```

A pass runs host glue between its array commands, a few nodes at a time, so
its threads should spin briefly at each barrier rather than sleep. At 912
tokens (ms and package J a pass, measured as in the table below) the Q8_0 file
takes 140.5 ms and 12.5 J with `GOMP_SPINCOUNT=20000` (~0.2 ms of spin),
150.9 ms and 12.1 J with `OMP_WAIT_POLICY=PASSIVE`, and 137.5 ms and 13.7 J
with libgomp's default spin (~3 ms); the BF16 file 160.7 / 15.6, 182.7 / 15.6
and 156.9 / 15.7. The short spin gives up 2.2 to 2.4% of the time against the
default and saves 0.5 to 8.5% of the energy. Sleeping saves at most 3.4% more
energy and costs 7 to 14% more time.

Against the CPU backend, the pooled embeddings agree to a cosine of at least
0.9994 (Q8_0) and 0.9996 (BF16), and every layer's output stays within an
NMSE of 1.4e-2. The drift comes from the GEMMs: the prefill blocks multiply in
bfp16-emulated bf16, and the CPU's own Q8_0 against BF16 moves the layers by up
to 7.1e-3. `test-xdna-embedding -m model.gguf` checks both, and
`scripts/snapdragon/embeddinggemma2_eval.py --device XDNA0 --flash-attn on`
runs the packing, ordering and CPU-parity checks.

Measured on a Ryzen AI MAX+ 395 against the CPU backend (16 threads, wait
policy unset), XDNA with `GOMP_SPINCOUNT=20000`: median ms and package joules
a pass over 6 runs, 3 rounds that each run the backends forward and then
backward. A run repeats the pass, and its package power is the mean of the
`PPT` sensor, read every 100 ms, over at least 5 s of passes after the reading
settles. The inputs marked `host` run whole on the host by the rule above.

| file | input | CPU ms | CPU J | XDNA ms | XDNA J |
| :-- | :-- | --: | --: | --: | --: |
| Q8_0 | one 15-token query (`host`) | 6.4 | 0.75 | 6.6 | 0.74 |
| Q8_0 | one 41-token text (`host`) | 11.8 | 1.57 | 11.7 | 1.51 |
| Q8_0 | one 67-token text (`host`) | 26.0 | 3.20 | 17.5 | 2.29 |
| Q8_0 | one 132-token text | 44.6 | 5.48 | 42.9 | 1.45 |
| Q8_0 | one 912-token text | 256.3 | 31.2 | 141.8 | 12.5 |
| Q8_0 | eight texts, 1188 tokens, one batch | 358.8 | 43.4 | 171.4 | 15.0 |
| BF16 | one 15-token query (`host`) | 6.5 | 0.52 | 7.5 | 0.34 |
| BF16 | one 132-token text (`host`) | 38.2 | 5.10 | 25.8 | 3.44 |
| BF16 | one 912-token text | 208.3 | 25.6 | 158.0 | 15.5 |
| BF16 | eight texts, 1188 tokens, one batch | 300.6 | 36.9 | 193.5 | 19.0 |

A pass the rule runs on the host is as quick as the CPU backend's at 41 tokens
and quicker from 67, and up to 1.15 times slower on a 15-token query, where
`--device none` is the quicker choice.

## How it works

### Artifacts, the design tag, and the instruction streams

The AIE designs are Python (IRON / mlir-aie) under `kernels/`; the device C++
they compile in is `kernels/*.cc`, one file per kernel. They are compiled out of
line at build time into `<stem>.xclbin` next to their `.insts.bin`, in the build
output directory. The backend looks for artifacts, in order, in
`GGML_XDNA_KERNEL_DIR` when it is set, the directory of the module the backend
is in (the backend library in a `GGML_BACKEND_DL` build, the binary it is
linked into otherwise), the backend install dir and the executable dir. The
first directory that holds an artifact wins. The working directory is not
searched, so a file placed there cannot replace a shipped kernel; point
`GGML_XDNA_KERNEL_DIR` at it instead, as an absolute path (a relative one
resolves against the working directory). A package that loads the backend with
`ggml_backend_load_all_from_path()` ships the artifacts next to the library.

`fused_layer` and `gemv_n32_r4_c8` are loaded under a **design-tagged** name.
`kernels/design_tag.py` hashes the design sources and the build knobs. Configure
passes that tag to the backend as a compile definition and the build writes it to
`xdna-design-tag.h` in the build tree, so the name a build produces and the name
the backend looks for are the same value: an artifact built from other sources is
invisible (and reported) rather than driven. The tag covers the Python designs
and the knobs, not the backend's own stream builders - those can change without a
kernel rebuild. A build that does not generate kernels compiles the committed
`xdna-design-tag.h` in this directory instead; `test-xdna-design-tag` (ctest)
fails when it is not the tag of the sources at the default
`GGML_XDNA_GATED_FMT=1`.

The `.insts.bin` files are reference copies. The runtime builds its own TXN
instruction streams in C++ (`xdna-seq.cpp`, `xdna-seq-attn.cpp`,
`xdna-gemv.cpp`), which is what lets one artifact serve every shape. The
prefill GEMM keeps the last 128 streams it built, by what they were built from,
so a layer or a pass that repeats a shape and its destination only looks one up.

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
the CPU over 511 tokens, and tg512 with `GGML_XDNA_FUSED_WIDE=1`, on Qwen3.5
Q4_K_M (the CPU's own one-token decode scores 0.0029 on the 9B, 0.0018 on the
4B, 0.0053 on the 2B and 0.0043 on the 0.8B):

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
- For a Qwen3.5 decode, a model the fused decode path covers (see above). A
  refused quantization is not run by setting `GGML_XDNA_FUSED_LAYER=0`;
  requantize `ssm_out` (see Build). An embedding model does not decode and
  needs none of this.

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
| `pfx_c8` | the merged prefill design: the prefill GEMM in its 8-bit tiles, the prefill attention and a row command (a sliding layer's Q, K and V norms and RoPE, under `GGML_XDNA_PFX_ROW`) in one hardware context, loaded with `GGML_XDNA_PFX=1` |
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
  `OMP_WAIT_POLICY=PASSIVE` does the same for the host's OpenMP threads. Use it
  for a model the fused layers cover (the 0.8B: 3% less energy a decoded token,
  11% less a prompt token) and leave it unset for a wider one, whose decode runs
  on host threads that it puts to sleep between ops (the 9B Q4_K_M decodes at
  18.8 instead of 16.1 t/s and 3.4 instead of 4.4 J a token without it).
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
| `GGML_XDNA_HOST_DECODE_THREADS` | one a core | host threads for the decode of a model wider than the fused layers (see [The fused decode recurrent layer](#the-fused-decode-recurrent-layer)): one on every allowed core when they are pinned, 8 when they are not; `0` runs that decode per op on the array, slower at more energy a token |
| `GGML_XDNA_HOST_DECODE_PIN` | 1 | in an OpenMP build, pins those threads one a core, filling one L3 domain before the next, and hands every borrowed thread its CPU mask back after each decode; `0` leaves 8 threads to the OS, slower (the 4B: 18.7 against 30.1 t/s) |
| `GGML_XDNA_FUSED_WIDE` | 0 | `1` runs the decode of a model with a residual width above 1024, up to 2048 and a multiple of 512 (Qwen3.5-2B), on the fused layers instead of the host (see [The fused decode recurrent layer](#the-fused-decode-recurrent-layer)) |
| `GGML_XDNA_ARENA_MB` | 256 | size of each chunk of the device-buffer arena, in MiB |
| `GGML_XDNA_CONV` | 0 | `1` runs the standalone prefill `SSM_CONV` on the array; the host is cheaper, which is why it is off |
| `GGML_XDNA_GDN_CONV` | 0 | `1` computes the prefill GDN's conv input on the array; the host is faster, which is why it is off |
| `GGML_XDNA_KERNEL_DIR` | unset | a directory to look for the kernel artifacts in before the others (see Artifacts) |
| `GGML_XDNA_PFX` | 0 | `1` loads `pfx_c8` in `pgemm_c8`'s place: the prefill attention runs on the GEMM's hardware context, so a layer pays no context switch between its projections and its attention. Its GEMM takes the 8-bit weight form only (`Q8_0`, `Q5_K`, `Q6_K`): a `Q4_K` projection runs on the int8 or bf16 GEMM instead, in their own hardware contexts, and a SwiGLU or GeGLU folds only into the down projection's A. A BF16 file's projections never run on the prefill GEMM, so there it moves only the attention and keeps the switches: EmbeddingGemma 2 BF16 takes 1.5% more energy a pass at 912 tokens and 3.8% more at 1188 with it than without, so leave it off for BF16. With every prefill op on the array (`GGML_XDNA_FA_MM=2`, `GGML_XDNA_GEMM_HOST=0`, `GGML_XDNA_HOST_CHUNK=0`, `GGML_XDNA_PGEMM_HOST=0`), run it with `OMP_PLACES=cores OMP_PROC_BIND=spread GOMP_SPINCOUNT=2000`: one host thread a core keeps the cores out of their boost clocks between array commands, which otherwise lifts some runs from ~38 to ~53 W, and the shorter spin idles them sooner. EmbeddingGemma 2 Q8_0 at 912 tokens then takes 165.0 ms and 5.89 J a pass, against 169.2 ms and 6.86 J without these settings and 265.3 ms and 7.99 J without `GGML_XDNA_PFX` (median of 6, 3 interleaved rounds). Without the artifact it warns at the first prefill GEMM and keeps `pgemm_c8` |
| `GGML_XDNA_PFX_ROW` | 0 | with `GGML_XDNA_PFX=1`, `1` runs each sliding layer's Q and K norms and RoPE and V's norm on the array's row-op columns, straight into the prefill attention's inputs: the host skips those nodes, their pads and f16 casts, and the attention's pack. A layer whose graph differs from EmbeddingGemma 2's falls back to the host. Opt-in: EmbeddingGemma 2 Q8_0 takes 4.1% less energy a pass at 912 tokens and 4.0% less at 1188 with it than without |
| `GGML_XDNA_PGEMM_GLU_X` | 0 | with `GGML_XDNA_PFX=1`, `1` runs EmbeddingGemma 2's per-layer input gate, `gelu(inp_gate x pe_in) * per-layer input`, as the projection's GEMM: the cores start the accumulator's up half from the per-layer input, so the GeGLU epilogue writes the product in bf16 straight into `per_layer_proj`'s input. The host skips the GELU and the MUL, and nothing of the gate goes to the host. A gate of other than 512 columns, or a product another node reads, stays on the host. Opt-in: EmbeddingGemma 2 Q8_0 with `GGML_XDNA_PFX_ROW=1` takes 0.9% less energy a pass at 912 tokens with it than without, and as much at 1188 within the spread of runs |

For bisecting a regression - each `0` turns off the NPU route it names, and the
host (or a less fused path) runs it instead. They are for narrowing down a bug
report, not for tuning: every one costs more time or energy than the default.

| variable | `0` turns off |
| :-- | :-- |
| `GGML_XDNA_FUSED_LAYER` | the fused decode layers (the per-op path runs). It does not run a model the fused layer refuses |
| `GGML_XDNA_GEMV_GROUP` | the decode projections outside the fused layer, on the array |
| `GGML_XDNA_GEMV_PROMOTE` | merging a decode GEMV group of mixed weight formats into one dispatch |
| `GGML_XDNA_QUEUE` | queuing the fused layers: each one is waited for |
| `GGML_XDNA_ATTN` | the decode attention on the array |
| `GGML_XDNA_ATTN_LAYER` | a whole attention layer, the q/k/v projections to `l_out`, in one dispatch (the tail and the decode attention still run) |
| `GGML_XDNA_ATTN_TAIL` | an attention layer's tail - output projection, residual, post-attention norm and FFN - in one dispatch |
| `GGML_XDNA_GLUE` | the host ops the backend runs inside an NPU chunk (the scheduler splits the graph at each instead) |
| `GGML_XDNA_GLUE_CAST` | the F32 to F16 casts among them, such as an attention's K and V (each is a CPU split instead) |
| `GGML_XDNA_HOST_BO` | the backend's host buffer type: plain CPU buffers |
| `GGML_XDNA_PGEMM` | the prefill GEMM on the array |
| `GGML_XDNA_INT8` | the native int8 prefill GEMM |
| `GGML_XDNA_PGEMM_ADD` | the residual ADD folded into the norm's pass |
| `GGML_XDNA_PGEMM_NORM_A` | the RMS norm written into the next GEMM's A |
| `GGML_XDNA_PGEMM_GATED_A` | a recurrent layer's gated output written into the output projection's A |
| `GGML_XDNA_PGEMM_GATE_A` | the attention gate written into the output projection's A |
| `GGML_XDNA_PGEMM_GLU` | the SwiGLU or GeGLU computed from the gate and up GEMMs on the array |
| `GGML_XDNA_PGEMM_GLU_A` | the SwiGLU or GeGLU written into the down projection's A |
| `GGML_XDNA_PGEMM_GROUP` | consecutive projections of one activation (Q, K and V) run as one GEMM, each output into its own node |
| `GGML_XDNA_GEMM_HOST` | a run of bf16 projections among pgemm commands runs on the host when that is quicker than the array's run plus the two context switches |
| `GGML_XDNA_HOST_CHUNK` | a prefill chunk whose array commands cost more than the host's products runs whole on the host as CPU graphs (see [The host or the array](#the-host-or-the-array)); `0` keeps it on the array |
| `GGML_XDNA_PGEMM_HOST` | in a chunk that keeps array work, a pgemm command of 2 to 128 rows whose products the host does quicker runs on the host; `0` keeps it on the array |
| `GGML_XDNA_GEMM_GROUP` | projections of one activation on the bf16 GEMM (Q, K and V; gate and up) run as one GEMM, each output into its own node |
| `GGML_XDNA_GEMM_ONE_BLOCK` | the GEMMs of one route and row count in a graph run on one prefill M block, the one of least summed cost; `0` lets each op pick its own, and every change of block costs a hardware-context switch |
| `GGML_XDNA_GDN_MM` | the prefill gated delta rule on the array |
| `GGML_XDNA_GDN_IN` | the prefill GDN's conv input, norms and state copy fed straight into the array |
| `GGML_XDNA_FA_MM` | the prefill attention on the array: `0` the host, `2` the array whatever the size |
| `GGML_XDNA_HOST_FA` | the backend's own host attention kernel: ggml-cpu's op runs instead. `2` runs the portable kernels in place of the AVX-512 ones |

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
- **No NPU** - with no accel device bound to `amdxdna`, or none the process
  can read and write, the backend lists no device. Listing the device does not
  open it: the first use does (a model placed on it, or its backend
  initialised), and loads the kernels then. If that open fails, the device
  stays listed but claims no ops, so everything runs on the CPU.

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
| `xdna-pgemm.h/.cpp` | the prefill GEMM in the decode GEMV's weight tiles, and what it folds in (norm, SwiGLU or GeGLU, gate into A, GELU times the per-layer input into A) and the projections it runs as one |
| `xdna-host-cost.h` | the host's time for a MUL_MAT (fitted per weight type and team size), which the routing plans compare with the array's commands |
| `xdna-attn-mm.h/.cpp` | the prefill attention on the array |
| `xdna-pfx-attn.h/.cpp` | the prefill attention as a mode of the merged prefill design (`GGML_XDNA_PFX`); its objects are prepared apart from the host's pack, so the row command can write them instead |
| `xdna-pfx-row.h/.cpp` | the merged prefill design's row command: a sliding layer's Q and K head norms and RoPE and V's norm on the row-op columns, straight into the prefill attention's Q, K and V^T objects: the sliding attention's Q, K and V under `GGML_XDNA_PFX_ROW` |
| `xdna-pfx-layout.h` | where the merged prefill design's attention reads its Q, K and V^T objects, shared by its host pack and the row command |
| `xdna-pgemm-path.h` | the prefill GEMM artifact's data paths and which artifact runs, shared by the GEMM and the merged design's attention |
| `xdna-host-attn.h/.cpp`, `xdna-host-attn-ops.h` | the prefill attention on the host: the work items, and the AVX-512 and portable kernels |
| `xdna-attn-mask.h` | an attention mask read on its f16 bits, as the array route and the host kernel both take it |
| `xdna-gdn-mm.h/.cpp` | the prefill gated delta rule on the array, and the GDN input it is fed |
| `xdna-conv-prefill.h/.cpp` | SSM_CONV prefill runner (opt-in) |
| `xdna-norm.h` | the host RMS_NORM, rounded as ggml's CPU op rounds it |
| `xdna-match.h` | how the graph planners recognise the nodes llama builds around a layer |
| `xdna-prof.h` | the profiler behind `GGML_XDNA_PROF` |
| `kernels/*.py` | IRON/MLIR-AIE designs: the array configuration, the fifos and the runtime sequence |
| `kernels/*.cc` | the device C++ those designs compile in |

## Known limits

- **A device write to the first line of a bf16 GEMM block's C has been seen not
  to land; the cause is not identified.** The 64 bytes at the start of C (row 0,
  columns 0-15) stay at the host's mark through a driver sync and 1.5 s of cache
  flushes, the rest of the block is correct, the line is nowhere else in the
  process, and running the block again writes it in full. It was seen only in
  processes where another buffer's data ended on the byte C is first written at
  (13 of 13 processes that lost a line, none of 77 that did not), and it recurs
  at the same blocks in every pass of such a process. Reproducer:
  `probes/bo_place_shim.c` (its header has the build and the command) places the
  buffers' address reservations back to back, and a fresh `llama-embedding` run
  of a BF16 EmbeddingGemma 2 file on 600 tokens then loses lines in every
  process. C one page into its buffer loses none under it, with the mappings
  still touching. So C starts one page in (`GEMM_C_GUARD`), and a block that
  still fails is run again (`GEMM_RERUNS`; about 18 ms and a warning each time).
  The int8 GEMM and the conv, decode and attention kernels write at the start of
  their buffers and have not been tried under the shim. The in-place pgemm
  output has no loss detection; audits of it found no loss, with and without the
  shim, but its buffer layout under the shim was not classified.
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
  anyway), and one context's support queries wait for another's compute.
- The first decode call sets up every fused layer: it packs the layer's weights
  into the fused formats, builds the layer's instruction stream and loads its
  kernel. This takes about 3.2 s on Qwen3.5-4B and 6.4 s on Qwen3.5-9B, in that
  call only (`llama-bench` tg numbers exclude it). An attention layer also
  builds one more kernel the first time its K/V span reaches a new task count.
- The device reports itself to the scheduler as an integrated GPU: its buffers
  are host memory, and the free memory it reports is the system's
  `MemAvailable`, which the CPU and any other device on host memory draw from
  too.
- EmbeddingGemma 2's attention runs on the host kernel below ~16k tokens, and
  its full-attention layers (heads of 512, one KV head) at any length: the
  prefill attention kernel takes heads of 256 over 2 KV heads, and each call
  pays two hardware-context switches.
- A prefill attention call on the array packs and flushes every key the cache
  holds, not only its ubatch's: a copy kept from an earlier ubatch can hold a
  cell the cache has freed and refilled since. On Qwen3.5-0.8B, a pp8192 in
  2048-token ubatches makes 12 such calls, and their packing takes 14 ms of the
  3.3 s prefill (~1.2 ms a call). In 512-token ubatches its attention stays on
  the host.
