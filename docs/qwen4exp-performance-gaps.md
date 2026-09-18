# Qwen4Exp FP8 performance gaps and remaining work

Status: 2026-09-18

This is the short planning view of the experiments in
[the optimization worklog](../worklog.md). Estimates assume one engineer
familiar with GGML/CUDA and reliable access to two H100s. They are
engineering estimates, not commitments; kernel gains overlap and should not
be added directly.

## Current position

The matched baseline uses physical H100 GPUs 6 and 7, batch size one, FP8
weights, tensor/expert parallelism, PP2048, TG128, and a 2048-token llama.cpp
ubatch. llama.cpp also uses the optimized FP8 expert path, FlashInfer GDN,
compressed QSA, and pinned-host PLE through `-lm none`.

| Resident context | Workload | llama.cpp | SGLang | llama.cpp deficit | Gain needed to match |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | PP2048 | 9,835 tok/s | 12,014 tok/s | 18.1% | 22.2% |
| 0 | TG128 | 87.33 tok/s | 110.61 tok/s | 21.1% | 26.7% |
| 64K | PP2048 | 6,480 tok/s | 11,031 tok/s | 41.3% | 70.2% |
| 64K | TG128 | 74.16 tok/s | 103.58 tok/s | 28.4% | 39.7% |

The empty-context gap is now moderate. The dominant unresolved problem is
context scaling: from context 0 to 64K, llama.cpp loses 34.1% of PP throughput
and 15.1% of decode throughput, while SGLang loses 8.2% and 6.4%.

The first task should be a fresh pair of Nsight captures from this exact final
checkpoint. Older profiles identify the right classes of work, but they
predate the latest FP8 GEMV, QSA decode, selector, and BF16 GEMV changes.

## Remaining work

| Priority | Work package | Evidence and likely scope | Directional upside | Effort |
| --- | --- | --- | ---: | ---: |
| P0 | Profile the final checkpoint | Matched llama.cpp/SGLang PP2048 and TG128 captures at context 0 and 64K; verify PLE residency and rank balance. | Diagnostic | 2-3 days |
| P1 | Make QSA selection scale with context | The remaining 64K penalty is before/around sparse attention: block scoring, pooling, top-k, index expansion, and layout traffic still grow with cache length. Keep pooled/index data resident and fuse the score-to-index pipeline. | 15-35% PP64K; 5-10% TG64K | 7-15 days |
| P1 | Finish a production sparse-prefill kernel | The Triton AOT kernel helps at medium context but becomes neutral at 100K because selection dominates. Generalize dynamic widths and integrate a supported build/runtime path after fixing selection. | 5-15% prefill at useful contexts; uncertain at 64K alone | 5-10 days |
| P1 | Tune remaining batch-1 BF16 projections | The latest trace still showed about 2.35 ms/GPU/token in generic BF16 projection kernels. Extend tuned GEMV coverage only to hot shapes or fuse adjacent projections. | 5-10% TG | 5-10 days |
| P1 | Optimize and harden batch-1 FP8 MoE | The native two-stage expert GEMV is much better than DeepGEMM at batch one, but needs a current profile, rank-balance work, broader correctness coverage, and fewer staging operations. | 5-10% TG; smaller PP effect | 5-10 days |
| P2 | Remove recurrent BF16/F32 boundaries | FlashInfer GDN is fast, but its adapter still packs inputs and unpacks output. This requires changing the surrounding graph contract, not adding more casts. | 1-4% PP | 3-5 days |
| P2 | Resolve memory-residency parity | The worklog contains differing observations about SGLang PLE placement. First measure actual resident bytes and transfers. A larger GPU expert cache may help only if the deployment memory budget permits it. | Unknown until measured | 1-2 days to diagnose; 5-10 days if cache changes are justified |
| P2 | Production hardening | Replace experimental environment-only wiring where appropriate, define the FlashInfer/Triton dependency policy, add fallbacks and CI tests, and validate non-Hopper behavior. | Reliability, not direct speed | 5-10 days |

The QSA items are the best chance of recovering the extra long-context loss.
The MoE and dense-projection items are the best chance of closing the
context-independent decode gap. A realistic one-engineer plan is:

1. **2-3 days:** final matched profiling and residency verification.
2. **2-3 weeks:** QSA score/selection and sparse-prefill work.
3. **2-3 weeks:** batch-1 MoE and dense-projection work.
4. **1-2 weeks:** integration, fallbacks, tests, and repeatable benchmarks.

That is roughly **4-7 engineer-weeks to materially narrow the small-batch
gap**. Getting every context within 10% of SGLang is higher risk and may take
**6-10 weeks**, because the 64K PP path needs a 70% throughput increase and no
single measured kernel accounts for all of it.

## Work that should not be repeated without new evidence

- Whole-model tensor-parallel CUDA graphs reduced launches to approximately
  one replay per GPU/token but changed stable decode by only about 0.1%.
- Custom NVLink reductions and duplicated recurrent computation both lost to
  the existing NCCL path. NCCL is secondary, not the main gap.
- GDN/SSM decode is already competitive and occupies only a small fraction of
  token time.
- The old public `GGML_MOE_FFN` and MegaMoE prototypes were selected
  correctly but were catastrophically slow because of their data path.
- Maskless QSA decode was neutral, and the fused scalar HC-injection path
  regressed decode.

## High-throughput decoding is a separate architecture project

The work above can improve batch-1 and modest batch sizes, but SGLang-class
high-throughput continuous decoding cannot be reached by adding more CUDA
kernels to the current llama.cpp path. The limiting difference is the
KV-cache and serving architecture.

SGLang's throughput path combines block-paged request storage, cheap
request-to-token indirection, continuous admission and eviction, prefix reuse,
and graph-stable batch buckets. The current llama.cpp cache/scheduler does not
provide an equivalent end-to-end path. Qwen4Exp makes this harder because a
request owns more than ordinary attention KV: sparse-QSA indexer/pooled state
and recurrent GDN/SSM state must be allocated, copied, forked, evicted, and
restored consistently with the KV blocks.

A credible redesign would need:

1. A block-paged, per-request KV allocator with stable device-side block IDs.
2. Matching lifecycle management for QSA pooled/index state and recurrent
   GDN/SSM/conv state.
3. A continuous scheduler that forms graph-compatible batch buckets without
   copying or defragmenting the cache on the token path.
4. Prefix sharing, request fork/copy, eviction, rollback, and state restore.
5. Tensor/expert-parallel metadata and correctness tests across all cache
   mutations.

A focused prototype could establish feasibility in **3-5 engineer-weeks**.
A production-quality implementation is more realistically **8-12+ engineer-
weeks**, with substantial regression risk across other models and cache
operations. This should be planned independently from the Qwen4Exp kernel
optimization branch. Until that redesign exists, batch-8/16 kernel work may
improve aggregate throughput, but it should not be expected to match SGLang's
serving throughput.

## Recommended success criteria

- Near term: PP2048 at 64K above 8.5k tok/s, TG128 at context 0 above
  100 tok/s, and TG128 at 64K above 90 tok/s, with no regression at context 0.
- Small-batch completion: within 10-15% of SGLang for PP2048 and TG128 at both
  tested contexts, using repeatable warm measurements.
- High-throughput work: define a separate requests/second and
  tokens/second benchmark over batch/concurrency 1, 2, 4, 8, and 16; do not use
  batch-1 TG as a proxy for serving throughput.
