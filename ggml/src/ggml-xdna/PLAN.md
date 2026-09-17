# What is left: full NPU coverage, and the speed to match FastFlowLM

Written 2026-09-16, against `6a9ece47b`. Model: `~/models/unsloth-Q4_K_M.gguf`
(Qwen3.5-0.8B, 18 recurrent + 6 full-attention layers). Bench: Ryzen AI MAX+
395, XRT 2.25.37, IRON 1.4.3.

Two goals, and they are not the same work:

- **Coverage** - every operator on the array. Settled; a correct-but-slower
  NPU path beats a host path. Section A.
- **Speed** - decode and prefill comparable to FastFlowLM. Section B.

Nothing here is a guess about where the time goes: every number was measured
with `GGML_XDNA_GLUE_PROF`, `GGML_XDNA_DESIGN_PROF` or the new
`GGML_XDNA_RUNNER_PROF`. The few places we still do not know are labelled.

## Where we are

FastFlowLM's own `/status` metrics, same 2056-token prompt, same machine
(`flm run qwen3.5:0.8b --pmode performance`, then `/input <file> <question>`):

| | us | FastFlowLM | gap |
| :-- | --: | --: | --: |
| **prefill** | ~3900 ms, **~520 t/s** | 1111 ms, **1856 t/s** | **3.5x** |
| **decode** | 31.2 t/s | **43.6 t/s** | **1.4x** |
| submissions per decode token | 83.7 | 4.44 | 19x, and it does not matter - see B1 |

`llama-bench`: pp512 680, pp2048 630, tg64 31.2.

Prefill was 303 t/s when this plan was first written; the fixes since (the GDN
packer, parallel and strip-wise weight re-quantisation, skipping the int32
accumulator, and the concat fusion before that) took it to ~520.

**Judge a change by the per-phase profile, not by prompt-eval time.** The
bench drifts about +-8% between runs - the same binary read 539 t/s one hour
and 463-486 the next - which is larger than most of the remaining items. The
profiles below are stable across runs.

## A. Coverage: what still runs on the host

### A1. Prefill glue - 414 ms

Nothing large is left. It is elementwise glue, and **each item is cheaper than
the ~150 us a dispatch costs**, so moving them one at a time makes the prompt
slower. They have to be folded into the dispatches already running.

| op | ms | calls | where it should go |
| :-- | --: | --: | :-- |
| `RMS_NORM` | 69 | 517 | into the projection dispatch that consumes it |
| `L2_NORM` | 41 | 216 | same (q/k norm of the recurrent layers) |
| `UNARY SIGMOID` | 37 | 168 | into the GDN prefill kernel's epilogue |
| `MUL` / `ADD` / `CPY` / `GLU` / `SILU` | ~120 | ~1800 | into whichever kernel produces the operand |
| `MUL_MAT q8_0 1024x16` | 19 | 252 | `ssm_alpha`/`ssm_beta`; no NPU weight format for Q8_0 at N=16 |
| `MUL_MAT q6_K 1024x248320` | 16 | 7 | the output projection - see A3 |
| `ROPE` / `GET_ROWS` / `CONT` / `SOFTPLUS` / `SCALE` | ~20 | ~1500 | glue |

### A2. Tail ubatches

Three kernels refuse batches that do not fill their geometry, so the last
ubatch of a prompt (8 tokens here) falls back to the host - `gdn` alone spends
40 ms of host tail:

- `fa_prefill` needs `n_tokens % 128 == 0` and at least 128.
- `conv_prefill` needs `n_tokens >= 64`.
- `gdn_prefill` finishes a partial chunk on the host (`cpu_tail`).

One fix for all three: pad the batch to the geometry and let the causal mask
(attention) or a zeroed tail (conv, gdn) make the padding harmless. For
attention that is already true - keys past `n_kv` are masked by position - so
only the query side needs a "how many rows are real" count, which can ride in
the header the kernel already reads.

### A3. Decode - ~9.9 ms a token on the host

| op | ms/token | note |
| :-- | --: | :-- |
| `MUL_MAT q6_K 1024x248320` | 4.3 | the output projection - the biggest host item in decode |
| `MUL` / `GET_ROWS` / `RMS_NORM` / `CONCAT` / `ADD` / `CPY` | ~3.0 | glue |
| `MUL_MAT q8_0 1024x16` | 0.5 | `ssm_alpha`/`ssm_beta`, 36 a token |
| `FLASH_ATTN_EXT` + `ROPE` + `CONT` | 0.3 | the six attention layers, one token each |

**The output projection is the one worth moving on its own.** It is bandwidth,
not launches: the CPU reads its 208 MB at ~48 GB/s; the array would read 294 MB
(Q6_K expands into q8g16) at 56, so about +1 ms of array time against -4.3 ms
of host time. It needs the GEMV's `n_out() <= 14` limit lifted - 243 chunks at
N=248320. FastFlowLM gives it its own xclbin (`lmhead`), which is the same
conclusion from the other side.

Decode attention is 0.3 ms a token and is a coverage item only. The prefill FA
kernel does not fit it: one token is one query row against the whole cache,
which is a GEMV shape, not a 128x512 tile.

## B. Speed

### B1. Decode: not the submission count (-9 ms wanted)

The first version of this plan led with an `xrt::runlist`, on the strength of
83.7 submissions a token against FastFlowLM's 4.44 and a probe that put ~110 us
of the ~150 us dispatch floor in the submission. **Measured, that is wrong.**
`GGML_XDNA_GEMV_STAGE_PROF` splits a real dispatch into host, submit and wait:

  host 6-15 us     submit 4-5 us     wait 122-311 us

The submission is 4 us. The probe's 110 us was the completion path of a short
stream run back to back, not what the decode does. And the 83.7 submissions
are 83.7 *dependent* steps with host work between them (18 core+so, 18 FFN
pairs, 43 projections, 4 fresh starts - `GGML_XDNA_DESIGN_PROF=1`), not a
batch waiting to be formed. A runlist would remove 4 us of 200.

What is actually in the decode's 32 ms:

1. **~9.9 ms on the host**, of which the output projection is 4.3 (A3) and the
   elementwise glue ~3.0. Both are real work to move, and the projection is
   the single biggest item in the whole decode.
2. **~10 ms waiting on GEMV weight traffic.** The K1024/N8192 projection moves
   9.7 MB, which is 173 us at 56 GB/s against 311 us measured - about 56% of
   peak. Worth understanding before anything else in decode: it is the floor
   everything else sits on.
3. The rest is the fused core and the FFN pairs.

Do **not** split the array into narrower designs to save context switches: a
switch costs ~360 us per column and nothing becomes co-resident - the placer
starts every design at column zero.

### B2. Prefill: ~3900 ms on the 2056-token prompt

| | ms | |
| :-- | --: | :-- |
| **int8 GEMM path** | **~2000** | `GGML_XDNA_PROFILING=1`, summed over 600 calls |
| - array wait | 900-1200 | **2.2 TFLOP/s effective - see below** |
| - weight re-quantisation (`wbo`) | 430 | first touch of each tensor |
| - int32 accumulate + rescale (`cacc`) | 200 | |
| - read / apack / dscan / seq | 180 | |
| `fa` dispatch | 580 | 240 dispatches |
| `gdn` dispatch | ~440 | 576 chunks |
| `conv` dispatch | 357 | 216 |
| `conv` pack + scatter | 339 | BO write/read bandwidth, ~5-6 GB/s |
| host glue ops (A1) | 414 | |
| `gdn` pack | 116 | was 744 |
| context switches | ~415 | 267 x ~1.9 ms, overlaps the dispatch times above |

**The array's GEMM time is no longer the suspect.** Per geometry, from the
same profile (FLOP / wait):

| M x K x N | calls | wait ms | effective |
| :-- | --: | --: | --: |
| 512 x 3584 x 1024 | 96 | 74 | 4.84 TFLOP/s |
| 512 x 1024 x 3584 | 192 | 263 | 2.74 |
| 512 x 1024 x 6144 | 72 | 192 | 2.41 |
| 512 x 2048 x 1024 | 96 | 113 | 1.83 |
| **512 x 1024 x 2048** | **72** | **218** | **0.71** |
| overall | | 919 | **2.21** |

2.2 TFLOP/s against the standalone kernel's 3.375 is 66%, not the 4x loss the
old note recorded - the concat fusion and the packer fixes closed it. The one
outlier is N=2048 at 0.71, and that is not the GEMM: those 72 calls are the
projection that follows the conv/gdn dispatches, so each one is the first
dispatch after a context switch, which costs ~1.6 ms extra (`DESIGN_PROF`
prints first-after-switch separately). 72 x 1.6 ms is most of its 218 ms.

So the remaining prefill work, in order:

1. **Flash attention on `aie::mmul`, 580 ms** -> ~60 ms if it reaches the GEMM
   kernel's rate. Now the largest single addressable item. Half done; the
   memory note `xdna-fa-design` records that the QK product on mmul is exact,
   the PV product is not, and that probing this kernel is treacherous (an `if`
   inside it crashes Peano, a probe with an early `return` gets reordered
   enough to lie). **Read it before restarting.**
2. **Context switches, ~415 ms**, of which the N=2048 GEMM alone eats ~115.
   The recurrent layer alternates three designs per layer (projections, conv,
   gdn). Merging conv and gdn into one artifact removes a third of the
   switches.
3. **The glue (A1), 414 ms** - only worth it folded into existing dispatches.
4. **`wbo`, 430 ms.** Two rounds already (parallel, then strip-wise writes);
   what is left is the Q4_K/Q6_K dequantisation itself. Caching the int8
   tensors on disk would remove it entirely after the first run.
5. **`conv` pack + scatter, 339 ms** - not a coding problem: it moves 911 MB
   in and 907 MB out of the BO over the prompt at 5-6 GB/s. Halving it means
   teaching the conv kernel bf16.
6. **`cacc`, 200 ms** - the rescale out of the device buffer. Contiguous and
   vectorisable, but it is also reading the BO, so expect bandwidth to bound it.

### B3. Accuracy, to keep an eye on

The device state is bf16 throughout. Measured: GDN prefill leaves a state
~4e-2 off the f32 reference after 512 tokens; flash attention is bf16 in Q, K,
V and the weights. Generation stays coherent and on topic but is not
token-identical to the host path. If that ever needs fixing, the state is the
place to look, not the products.

## Order of work

Done since the first version: FastFlowLM's prefill measured (1856 t/s), the
runner profiler added, the runlist idea retired on measurement, the GDN packer
(744 -> 116 ms), weight re-quantisation parallelised and strip-wise
(1638 -> 430 ms), and the int32 accumulator skipped for single-K GEMMs
(190 ms). Prefill 303 -> ~520 t/s. The int8 GEMM path is now understood and
the array runs it at 66% of the standalone kernel.

1. A3 output projection on the array - 4.3 ms a decode token, the biggest
   single decode item, and 16 ms of prefill too. Decode has had no work yet.
2. B2.1 flash attention on mmul, 580 -> ~60 ms.
3. B2.2 merge conv and gdn into one artifact - a third of the context
   switches.
4. B1/A1 fold the elementwise glue into the dispatches that produce it (414 ms
   prefill, ~3 ms a decode token).
5. B2.4 cache the int8 weight tensors on disk (430 ms of every run).
6. A2 pad the tail ubatches.
7. B2.5 teach the conv kernel bf16, halving its BO traffic.

## How to measure

```
GGML_XDNA_GLUE_PROF=1      host ops, batched as they actually run
GGML_XDNA_GLUE_PROF=2      host ops, each run alone and tallied by op (slower)
GGML_XDNA_RUNNER_PROF=1    pack / dispatch / scatter inside each NPU runner
GGML_XDNA_DESIGN_PROF=1    dispatches and context switches per xclbin, and the
                           cost of the first dispatch after a switch
GGML_XDNA_OPS_PROF=1       where each op of a decode graph ran
GGML_XDNA_RUNLIST_PROBE=32 submission cost, one at a time against a runlist
```

FastFlowLM, for comparison: `cd ~/flm && ./flm run qwen3.5:0.8b --pmode
performance`, then `/input /home/npu-bench/xdna-prompt-2048.txt <question>` and
`/status`. **Kill it before running ours** - it holds NPU contexts, and ours
then fails to create one (`CREATE_HWCTX IOCTL failed (err=-22)`).

Off switches, for bisecting: `GGML_XDNA_FA=0`, `GGML_XDNA_GDN=0`,
`GGML_XDNA_CONV=0`, `GGML_XDNA_CONV_FUSE=0` (`=2` traces the concat fusion).
