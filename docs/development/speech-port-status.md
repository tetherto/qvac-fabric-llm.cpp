# Speech ggml port status

Snapshot: `temp-11259` at `d249ebc9c`, 2026-10-21.

The combined [PR #346](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/346)
is superseded by the reviewed stack below. Superseded does not mean that every
change from the combined PR has landed. Remaining implementation work stays in a
separate follow-up PR.

## Merged stack

| Area | PR | Scope |
| --- | --- | --- |
| Infrastructure | [#363](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/363) | Tensor name capacity and RPC protocol update. |
| Ops and CPU | [#367](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/367) | Thirteen speech operations, generic names, CPU references, tests, and RPC protocol 111. |
| Metal | [#386](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/386) | Nine speech operations; the other four retain CPU fallback. |
| Vulkan | [#388](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/388) | Thirteen operations, alignment/grid gates, and stable channel normalization on Vulkan and Metal. |
| OpenCL | [#401](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/401) | Five operations and CI runtime detection. The source scanner was removed before merge. |
| Hexagon | [#415](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/415) | SNAKE only, with QDC runner coverage. |

The proposed Metal extras were assessed as unnecessary for the speech models or
already covered by the Metal/Vulkan parts; they are not a separate merged patch.

## Remaining implementation and validation

- CUDA implementations of the new speech operations are not part of the merged
  stack. Existing CUDA operations must not be confused with coverage of these
  new operation IDs.
- Hexagon still needs col2im-1d, conv-2d-dw, depthwise-conv-1d,
  timestep-embedding, transpose, `hvx-erf.h`, DMA stride handling, and the speech
  changes to flash-attention, im2col, unary, matmul, get-rows, and copy.
- The SNAKE reviewer reported a successful Galaxy S25 (SM8750, v79) run. The
  three QDC jobs skipped device execution because the API key was unavailable.
  QCS9075M and SM8850 remain unvalidated; green skipped jobs are not device passes.
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

The four refreshed CSVs cover the thirteen speech operations from the merged
snapshot. Unrelated historical rows are retained to keep this closeout scoped.
`support` queries backend capability; it does not execute kernels or certify
numerical correctness. Unsupported CUDA rows describe the current base honestly.

Each backend snapshot contains 129 cases: CPU reports 129 supported, CUDA 0,
Metal 91, and Vulkan 128. These are case counts, not operation counts. Regenerating
`docs/ops.md` after the refresh produces the existing file unchanged.

Generate fresh rows with `test-backend-ops support -b <backend> --output csv`,
filtering `-o` to `CONV_1D_DW_DIRECT,NORM_CHANNEL,SCALE_RESIDUAL,BIAS_GELU,PAD_REPLICATE_1D,GRU,ZERO_UPSAMPLE,CHANNEL_SHUFFLE,AFFINE_PRELU,SNAKE,LSTM_CELL,TDT_STEP,SIGLU`.
Use CPU, CUDA0, and Vulkan0 on the RTX 5090 host and MTL0 on the Metal host.
Replace the matching operation rows in `docs/ops/{CPU,CUDA,Metal,Vulkan}.csv`,
retaining the existing seven-column schema, then run:

```sh
python3 scripts/create_ops_docs.py
python3 scripts/create_ops_docs.py /tmp/ops-check.md
diff -u docs/ops.md /tmp/ops-check.md
```

The `update-ops-docs` workflow performs the same generation and comparison.
