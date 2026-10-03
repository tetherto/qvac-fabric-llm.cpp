# DFlash2 GPU performance campaign

## Status
Campaign isolated at `perf/dflash2-gpu-next`, base `20e89b5a0b87f9f79e640e0bc3e209b9e192bc52`. Existing PR branches remain read-only. All six lanes are built and pass exact-workload canaries; width3/5/7/serial baseline characterization is running. No optimization or speedup is claimed.

Scope clarification: all prefill and decode optimizations are eligible, not only speculative-decoding paths. Rank general matmul, attention, recurrent kernels, graph scheduling/fusions, memory traffic and synchronization by measured end-to-end savings. Fixed workload, model precision and quality gates remain unchanged.

## Acceptance
- Equal-weight geometric mean across Metal/M3 Ultra, CUDA/RTX 5090, Vulkan/RTX 5090, CUDA/GB10, Vulkan/GB10 and Vulkan/Radeon 8060S: prefill >=1.50x, speculative decode >=1.35x; no lane regression.
- Context 16384, prompt 10000, completion 1024, greedy, one request, uncached; target Q4_0 and DFlash2 Q8_0. Characterize draft widths 3/5/7 and serial; freeze best baseline width.
- Same-lane baseline reference. Exact preferred; top-token agreement >=99%, mean KL <=0.002, perplexity ratio <=1.01. Snapshot/rollback correctness is mandatory.
- TensorFold v0.6.4 (`6ea5ade26c4335491c50275af0be32b81f75f525`) is an engine-plus-format comparison, not part of the Fabric aggregate.

## Evidence
Raw artifacts: `/Users/pratiknarola/workstuff/fabric-bench-dflash2/results-gpu-next/`. Record source/binary/toolchain/model/fixture hashes, full requests/outputs, cache counters and actual GPU identity. Historical measurements are not the pinned baseline.

## Scoreboard
| Lane | Baseline prefill/decode | Candidate | Quality | Status |
| --- | --- | --- | --- | --- |
| M3 Ultra Metal | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |
| RTX 5090 CUDA | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |
| RTX 5090 Vulkan | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |
| GB10 CUDA | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |
| GB10 Vulkan | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |
| Radeon 8060S Vulkan | full baseline pending | unmeasured | unmeasured | exact-workload canary passed |

## Hypothesis ledger
| ID | Hypothesis and prediction | Cheapest refutation / actual-path proof | Result | Verdict / next action |
| --- | --- | --- | --- | --- |
| H01 | Metal rollback K>1 excludes existing chunked GDN; prefix+serial-tail can reduce prefill time while retaining snapshot semantics | Actual server graph T512, q/k[128,16,512,1], v[128,48,512,1], scalar gate; pipeline prints K4 at n_max3 and uses serial f32_4 plus cache fusion | Source and reached-path exclusion confirmed; numerical fidelity and saved time unmeasured | PROVISIONAL gain; qualify existing chunked numerics and op cost before editing |
| H02 | Vulkan serial GDN is an avoidable prefill cost | Current Strix server profiler works with GGML_VK_PERF_LOGGER=1 and verbosity5; explicit stage boundaries, count*mean aggregation, no triplet double-count | GDN is1.95% of diagnostic prefill, while q4_0 GEMM57.68%, attention13.32%, CONCAT5.09%; 16-token diagnostic, not acceptance timing | PROVISIONAL gain, lower priority; profile GEMM/attention/CONCAT first without ruling out a smaller GDN improvement |
| H03 | CUDA few-row dispatch has a better measured crossover | Trace target/drafter types and compare existing routes on actual shapes and adjacent widths | No measurement | PROVISIONAL; keep current thresholds until evidence |
| H04 | Shared DFlash2 scratch allocation/copies materially affect rounds | Profile allocation/copy/wait share on Spark | No measurement | PROVISIONAL; no speculative refactor |

Each experiment appends configuration, prediction, canary, commands/artifacts, observed result, first divergence, mechanism evidence, quality/performance verdict and next action. DID_NOT_RUN is not REFUTED. Three consecutive refutations require independent review before another edit.

## Activity
- Created new local branch/worktree at the pinned base; no existing PR branch changed.
- Planning verified all four SSH routes and model presence. Hashes, toolchain probes and runtime canaries remain execution gates.
- The extended benchmark client passed five admission-boundary tests and real CUDA server smokes on RTX 5090 and GB10: 10000 prompt tokens, 1024 completion tokens, cache count zero, no server errors, complete text/hash/native chunks retained.
- Single canary only, not a baseline or speedup claim: RTX 5090 CUDA 4239.068 engine-prefill tok/s and 170.803 client-decode tok/s; GB10 CUDA 1038.475 and 29.472. Artifacts: `results-gpu-next/{rtx5090,spark}/base-cuda-n3-canary.*`.
- RTX 5090 manifest verifies pinned source archive, both model hashes, fixture hash, CUDA 13.3/120a-real and requested GPU UUID. Existing Vulkan SDK shaderc 2026.4 passed all seven feature probes; no system compiler upgrade needed.
- Startup provenance requires a separate verbosity-4 diagnostic: `common/log.cpp:557` maps inference INFO to TRACE, so normal verbosity 3 hides actual offload, ubatch and snapshot count. Never use a diagnostic row as acceptance timing.
- GB10 has an unrelated RPC process holding 810 MiB, observed idle before the canary. It remains untouched; timed baseline noise and contention still require measurement.
- All six same-base canaries passed. Additional single-canary engine-prefill/client-decode rates: M3 Metal 306.110/48.745, RTX 5090 Vulkan 2694.033/141.876, GB10 Vulkan 604.657/25.091, Strix Vulkan 316.868/26.740. These are smoke observations, not selected-width baselines or acceptance results.
- Separate diagnostics verify full target66/66 and draft6/6 GPU offload, context16384, batch2048, ubatch512, target n_rs_seq3/draft0. Frozen target and batch threads: Metal20, RTX5090 10, GB10 20, Strix16.
- Native quality fixture producer passed on Metal: exact10000 prefix IDs plus1024 returned IDs, zero cached tokens, no truncation; SHA `36420931ce54888d414336e58ca435b66d350811396bce8e3a654206c4e90666`. Full token/request/response equality verified on the controller. Two additional fixture-rejection tests passed.
- Harness port check initially rejected a recently stopped listener's address. No live listener existed; the production server itself uses SO_REUSEADDR (`tools/server/server-http.cpp:165`). Matching that option allowed the Metal diagnostic collector to start. Updated launcher is used for serialized characterization; no inference code changed.
- Strix diagnostic ranking is saved in `results-gpu-next/strix/profile-canary-debug/profile-summary.json`: q4_0 GEMV50.72%, q8_0 GEMV8.38%, q6_K GEMV7.93% of profiled decode. Exact dominant shapes are17408x5120 and5120x17408, n4 verify/n512 prefill. Profiled-stage denominators include synchronization/log overhead and are not clean production acceptance timings.
