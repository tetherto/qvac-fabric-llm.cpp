# XDNA decode: where this stands and what to do next

Branch `xdna/npu-decode-full`, sixteen commits ahead of origin, **not pushed**.
Everything below was measured on the bench, not inferred.

## The goal

Qwen3.5-0.8B fully on the NPU, decode within 20% of FastFlowLM (38.8 t/s, so
31.0), prefill after that. Decode is **31.3 t/s** today, from 30.1.

## Read this first

Four numbers this backend was built around turned out to be artifacts of how
they were measured. Expect more of them, and re-measure anything load-bearing
before building on it.

| recorded | actually |
| :-- | :-- |
| GEMV streams at 26 GB/s | **56 GB/s** - the fit had not subtracted the fixed per-dispatch cost |
| only arguments 0-5 can be patched | arguments >= 5 need the DDR aperture `0x80000000` folded into the patch offset |
| the gated drain's bit31 form needs offset zero | that bit *is* the aperture; the offset goes in the low bits |
| full-ELF stalls, the wall is in the driver | the probes bind zero-filled buffers, which hang the known-good stream too |

**Two probe methods here do not work and have each produced a wrong
conclusion.** Synthetic zero-filled buffers hang the array whatever the stream
is, so a probe built that way cannot tell a broken sequence from unprepared
data. And two concurrent processes do not alternate, so timing one against the
other measures scheduling, not the thing you wanted. What does work:
**alternate in one process against real buffers**, which is what
`~/twodesign.py` on the bench does.

## The cost model

Per token at 31.3 t/s (31.9 ms), 82 dispatches:

| | ms |
| :-- | --: |
| weight stream, 406 MB at 56 GB/s | 7.3 |
| dispatch floor, 82 x ~150 us | 12.3 |
| host | 9.9 |

- **~110 us of each dispatch's ~150 us is command submission**, not the array.
  An `xrt::runlist` removes it: 32 runs one at a time cost 456 us each, as a
  list 339 us (`GGML_XDNA_RUNLIST_PROBE=32`). A list also takes runs from more
  than one kernel of the same context (359 us each), which is what a layer
  needs - its dispatches are different instruction streams.
- **A context switch costs ~360 us per column the design occupies**: 368 / 693
  / 1518 / 2829 us for 1 / 2 / 4 / 8 columns. Nothing becomes co-resident; the
  placer starts every design at column zero.
- **So the array is not worth splitting.** Bandwidth scales with columns too, so
  a two-column design switches cheaply and then streams the token's 406 MB at
  ~14 GB/s. FastFlowLM's several xclbins are a wide artifact for the layers
  (about two submissions a token) and narrow ones for work that moves few
  bytes; the reason it wins is 4 submissions against our 82, not the artifacts.

## What is on the host, per token

From `GGML_XDNA_GLUE_PROF=2` (it runs each node alone and tallies by op):

| | ms | note |
| :-- | --: | :-- |
| `MUL_MAT q6_K 1024x248320` | 4.28 | the output projection |
| `MUL` / `GET_ROWS` / `RMS_NORM` / `CONCAT` / `ADD` / `CPY` | ~3.0 | elementwise glue |
| `MUL_MAT q8_0 1024x16` | 0.46 | `ssm_alpha` / `ssm_beta`, 36 a token, no NPU format |
| `FLASH_ATTN_EXT` + `ROPE` + `CONT` | 0.27 | the six attention layers |

The attention layers cost almost nothing at this context length - moving them
is a coverage question, not a speed one. The output projection is the big one,
and it is bandwidth: the CPU reads its 208 MB at ~48 GB/s where the array would
read 294 MB (Q6_K expands into q8g16) at 56, so it is about +1 ms to move, and
it needs `n_out() <= 14` lifted (243 chunks at N=248320).

## What landed

1. **`a657baa09`** The GDN prefill kernel is opt-in (`GGML_XDNA_GDN=1`). It left
   a recurrent state decode could not continue from - a 2048-token prompt
   answered with nothing at all - and it was slower than the host (310 t/s
   against 400). This was the reported bug.
2. **`85c46d1ab`** The DDR aperture for arguments >= 5, centrally in
   `xdna_seq_ddr_patch`. Retires the three "rules" above; the ssm_out projection
   now has arguments 6, 7 and 8 of its own instead of living inside one at
   offsets, which is also slightly faster.
3. **`259fa46ca`** A dispatch group may take the wider weight format so it can
   swallow a projection whose natural one is narrower (q8g16 represents Q4_K
   exactly). 102 -> 82 dispatches, 30.6 -> 31.9 t/s.
4. **`1c2ea8ada`** The FFN transition runs on the array - the residual add, the
   norm and the quantization. See below.
5. **`eea69a51b`** The projection drains into the FFN's tiles
   (`GGML_XDNA_SO_TO_ACT=1`, off - see the open bug below).

## The FFN transition, which is the current frontier

Between a layer's two dispatches the host used to do the residual add, the RMS
norm and the activation packing. It now does nothing:

- the activation tiles carry three bf16 runs at `ACT_RAW_OFF` - the previous
  dispatch's output, the residual and gamma - plus the row length at `ACT_D_W`;
- a **prologue tile** in the merged design (`gemv_q4.py`, `act_pro_tile` in
  `gemv_q4.cc` built with `-DACT_PRO=1`) turns them into codes, group sums and
  group scales, and carries the norm's reduction across the chunk;
- its rsqrt rides the last tile at `ACT_RMS_W` and a GEMV core applies it to the
  accumulator before silu. That is exact because the matmul is linear in the
  activation, and it is what lets a tile be quantized before the reduction it
  belongs to has finished.

Three things this cost, all of them worth knowing:

- **The quantizer does not fit in the GEMV cores' program memory** - 1120 bytes
  over as scalar code, 1616 vectorized. That is why it is on a tile of its own,
  built as a separate object so it does not link into the cores. The cores keep
  one multiply. Anything else moved onto the array lands in this same program.
- **A tile in a broadcast path must keep no count.** The first version read the
  dispatch header and looped `n_out * n_tiles`; it deadlocked, repeatably, 3 of
  3 runs, whether or not the path was used. One object in, one object out, with
  everything it needs carried in the tile, is 0 of 3.
- Placement did not matter (col 0 row 3, col 5 row 2, col 1 row 4 all behaved
  the same), and the shim endpoint map did not move - worth checking anyway
  whenever the design's fifos change, with `kernels/shim_map.py`.

Verified: same text as the host path token for token, short prompt and the
2048-token file; tg64 31.29 against 31.43. **No faster, and that is the point** -
the norm's microseconds were never the prize. The prize is that the two
dispatches now have nothing between them.

## The device-to-device handover, half done

`GGML_XDNA_SO_TO_ACT=1` (off by default) is the last step before a runlist: the
projection drains straight into the FFN's activation tiles instead of going
device -> host -> device. One drain stream to a tile - a stream carries
`og*rows*n_core` = 256 columns and a tile holds `K_TILE` = 256 - at
`out_off = ACT_TILE` with `out_stream_stride = ACT_TILE`. Argument 8 of the
fused run becomes the FFN's activation buffer.

**The handover itself is correct.** `GGML_XDNA_ACT_DUMP=1` prints the tiles and
what the host path collects, and they agree value for value across all four
tiles. llama-cli reports 28.0 t/s against 20.6 with the round trip, so the
saving is real.

**The decode text is wrong anyway, and that is the open bug.** What has been
ruled out:

- not the drain - the values above;
- not the tile layout - with the host writing acc into the same f32 slot at the
  front of the tile, the output is identical to the old path token for token;
- not the stride, once it was put into the strided branch of the output builder
  as well as the general one (before that two streams landed in the wrong tiles
  and two tiles stayed zero, which is what the dump catches);
- not the staging copy overwriting acc - the host part is written in place now
  and only its own byte ranges flushed;
- not the header going unflushed - syncing it changes the wrong output into a
  different wrong output, which is itself a clue.

The last point is where to pick this up. The header tile is the one object in
the activation stream the previous dispatch does not write, the host and the
array now write different parts of the same buffer, and the flushes are
partial - so look at what the cores actually receive. The prologue passes the
header through untouched, so dumping the prologue's *output* objects, or
having it stamp a counter into a spare word, would say whether the cores see
the header and the tiles in the order they expect.

## Next, in order

0. **Finish the handover above.** It is worth ~7 t/s on its own and it is the
   precondition for the runlist.
1. **Put a layer's two dispatches in one `xrt::runlist`.** This is the payoff for
   everything above: ~110 us on each of them. `xdna_runlist_probe` in
   `xdna-runtime.cpp` already shows the mechanism works, including across two
   kernels of one context.
2. **Widen the list across layers.** Wherever the host still sits between two
   dispatches, that is a submission. The remaining host work inside the
   recurrent layer is `xdna_rec_gemv_so_collect` (reads the projection, adds the
   residual) and packing x - both candidates for the same treatment.
3. **The elementwise glue**, ~3 ms a token. It can only be absorbed into
   existing kernels; moving any of it as its own dispatch costs 150 us against
   the ~10 us it takes on the host.
4. **The output projection**, +1 ms and `n_out() <= 14` to lift - descriptor
   reuse in `xdna_gemv_seq_build`, or a repeat descriptor over the chunks, whose
   destinations are a fixed stride apart.
5. **Prefill.** `GGML_XDNA_GDN=1` is broken and wants either fixing or replacing
   with the fused layer run over chunks. Conv prefill (`GGML_XDNA_CONV=1`) is
   correct but slower than the host.

## Working on the bench

The only model is `~/models/unsloth-Q4_K_M.gguf` - its `ssm_out` is Q5_K, which
is what `GATED_FMT=1` is built for. The other model on that machine is a
different quantization and produces a dispatch timeout that reads exactly like
a broken design.

Sources live in `~/qvac-clean`, an rsync target and not a git clone; the git
clones on that machine are stale. Build:

```sh
source /opt/xilinx/xrt/setup.sh
cmake -S . -B build -DGGML_XDNA=ON -DGGML_OPENMP=ON -DGGML_XDNA_BUILD_KERNELS=ON \
  -DGGML_XDNA_GEMM_PYTHON=/home/npu-bench/mlir-aie-1.4.3/ironenv-v1.4.3/bin/python
cmake --build build -j$(nproc) --target ggml-xdna-kernels   # kernels first
cmake --build build -j$(nproc)
```

The order matters: the design tag is regenerated by the kernel target and the
binaries have to be built after it.

Switches worth knowing: `GGML_XDNA_ACT_RAW=0` puts the transition back on the
host, `GGML_XDNA_GEMV_PROMOTE=0` keeps a dispatch group to one weight format,
`GGML_XDNA_RUNLIST_PROBE=N` and `GGML_XDNA_CTX_PROBE=<xclbin>` are the two
timing probes, `GGML_XDNA_GLUE_PROF=2` is the host inventory,
`GGML_XDNA_DISPATCH_COUNT=1` counts launches.
