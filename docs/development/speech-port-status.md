# Speech ggml port status

Snapshot: `temp-11259` at `4b955f55f`, 2026-10-26.

The combined [PR #346](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/346)
was closed as superseded by the reviewed stack below. The initial documentation
closeout landed in [#416](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/416);
this snapshot includes the completed Hexagon implementation from #417. Remaining
CUDA work and hardware validation are listed separately below.

## Merged stack

| Area | PR | Scope |
| --- | --- | --- |
| Infrastructure | [#363](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/363) | Tensor name capacity and RPC protocol update. |
| Ops and CPU | [#367](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/367) | Thirteen speech operations, generic names, CPU references, tests, and RPC protocol 111. |
| Metal | [#386](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/386) | Nine speech operations; the other four retain CPU fallback. |
| Vulkan | [#388](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/388) | Thirteen operations, alignment/grid gates, and stable channel normalization on Vulkan and Metal. |
| OpenCL | [#401](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/401) | Five operations and CI runtime detection. The source scanner was removed before merge. |
| Hexagon SNAKE | [#415](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/415) | SNAKE with QDC runner coverage. |
| Hexagon completion | [#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417) | Remaining speech kernels, shared-op changes, bounded extended-address DMA, and review fixes. |

The proposed Metal extras were assessed as unnecessary for the speech models or
already covered by the Metal/Vulkan parts; they are not a separate merged patch.

## Remaining implementation and validation

- CUDA implementations of the new speech operations are not part of the merged
  stack. Existing CUDA operations must not be confused with coverage of these
  new operation IDs.
- The SNAKE reviewer reported a successful Galaxy S25 (SM8750, v79) run. The
  three QDC jobs on the final #417 head also skipped device execution. The full
  port still needs hardware validation on QCS9075M, SM8750, and SM8850; the S25
  SNAKE result does not validate the remaining kernels. Green skipped jobs are
  not device passes. Snapdragon Android and Linux host builds passed.
- Keep performance/policy changes deferred unless required by a model or requested
  by the backend owners: Metal precision routing and copy fast paths; Vulkan
  precision routing, auxiliary fused/tiled shaders and Adreno policy; OpenCL
  fused im2col/matmul-add, wide-GEMV policy, BF16 GET_ROWS, and lazy initialization.

## Port-back list for the speech branch

Compare each candidate with the current speech branch before applying it. This is
an integration checklist, not a claim that each change is still missing there.

### From the merged stack

- Generic names for the five former `SUPERTONIC_*` operations and matching model
  call sites; preserve enum/GLU compatibility and RPC protocol coordination
  ([#367](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/367)).
- Shifted two-pass channel variance in standalone/fused Vulkan and Metal kernels,
  with the large-offset regression cases
  ([#388](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/388)).
- Vulkan view-alignment and dispatch-grid gates, element offsets for LSTM_CELL
  and TDT_STEP, and the minimum-workgroup gate for depthwise/norm fusion
  ([#388](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/388)).
- OpenCL GRU saturation clamps and host-supplied local-array limit; portable
  softplus conversions; explicit generic-device opt-in and honest CI reporting
  when no compatible device ran
  ([#401](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/401)).
- SNAKE vector-boundary cases and QDC checks that require actual HTP execution
  ([#415](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/415)). Keep the
  work-queue API and compute-flag adaptation specific to the target backend version.
- Bounded VTCM input windows and 40-bit DMA address preservation, including
  tiled copy conversion and F16 HMX activation row transfers. Reset cached
  windows per operation and synchronize DMA copy destinations
  ([#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417)).
- Shared host/DSP depthwise VTCM sizing, including weights and overflow checks,
  plus the shared matmul partial-tile predicate
  ([#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417)).
- IEEE accumulation between depthwise convolution taps and bounded one-time F16
  weight conversion, with zero-input and padded-edge regressions
  ([#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417)).
- F16 HMX activation alignment handling and precision gating limited to the new
  F16 two-dimensional path; retain existing LLM precision behavior
  ([#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417)).
- All-outer-row partitioning for strided copies, BF16 GET_ROWS vector conversion
  with bounded tails, and weight-buffer/window-boundary regression cases
  ([#417](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/417)).

### Candidates retained from the superseded combined PR

These were listed in [#346](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/346)
and are not all delivered by the smaller stack. Reassess them against the newer
upstream backend before transferring a patch:

- Metal generic-copy destination indexing, full-precision matmul routing, and
  capability-probe semaphore release.
- Vulkan conv_2d_dw view-alignment gate; check overlap with the merged alignment work.
- Hexagon width casts.
- WebGPU unaligned byte transfers, bounded memset/reduction dispatch, and related
  backend-specific tolerance changes.
- Mask-aware comparison metrics and their regression coverage.
- TurboQuant sign-table and environment-flag initialization race fixes.
- Neutral wording changes associated with those patches.

The library-prefix, package-consumer, sanitizer, and other infrastructure work
parked outside the minimal speech port remains separate from this closeout.

## Support table provenance

The four CSVs are complete capability snapshots from the merged revision above,
replacing the earlier partial refresh. `support` queries backend capability; it
does not execute kernels or certify numerical correctness. Device-specific
support can differ with hardware and drivers.

Collection uses CPU on Ryzen AI 9 HX 370 and CUDA0 on the WSL2 laptop
(RTX 5090 Laptop GPU), MTL0 on Apple M3 Ultra, and Vulkan0 on Radeon 8060S (RADV, Mesa 25.2.8).
All builds use Release mode; the standalone CPU build has `GGML_NATIVE=OFF`,
and CUDA targets architecture 120 with the default FlashAttention quantization
set (`GGML_CUDA_FA_ALL_QUANTS=OFF`). The WSL run selects the PTX compiler
from the matching Windows driver directory through `LD_LIBRARY_PATH`.

Each snapshot contains the same 24,304 cases: CPU reports 20,936 supported,
CUDA 18,485, Metal 16,908, and Vulkan 20,266. Other backend CSVs retain their
previous snapshots, including HTP; that column does not yet reflect #415/#417.
The generator also renders operations absent from a snapshot as unsupported,
so table entries describe collected coverage rather than exhaustive backend support.

Build `test-backend-ops` from the snapshot on each host, then collect the matching
backend without an operation filter:

```sh
./build/bin/test-backend-ops support -b CPU --output csv > CPU.csv
./build/bin/test-backend-ops support -b CUDA0 --output csv > CUDA.csv
./build/bin/test-backend-ops support -b MTL0 --output csv > Metal.csv
./build/bin/test-backend-ops support -b Vulkan0 --output csv > Vulkan.csv
```

Replace `docs/ops/{CPU,CUDA,Metal,Vulkan}.csv` with those outputs, preserving the
seven-column schema, then regenerate and verify:

```sh
python3 scripts/create_ops_docs.py
python3 scripts/create_ops_docs.py /tmp/ops-check.md
diff -u docs/ops.md /tmp/ops-check.md
```

The `update-ops-docs` workflow performs the same generation and comparison.
