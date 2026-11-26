# Runner label catalog

CI runner labels live in one place: [`.github/runners.yaml`](./runners.yaml).
This lets a runner-image migration (OS bump, GPU-pool rename, hosted-image
retirement) update a single file instead of grepping every workflow.

## How it works

`runners.yaml` is the source of truth. A generated reusable workflow,
[`.github/workflows/reusable-runner-names.yml`](./workflows/reusable-runner-names.yml),
exports each catalog entry as a job output. Callers pull the label from that
output instead of hardcoding it, because `runs-on:` is evaluated before any step
runs, so a composite action cannot supply the label — a reusable workflow's
outputs can.

A catalog value is one of:

- a **scalar** label (`ubuntu-24.04`, `qvac-ubuntu2204-x64-gpu`) — consumed as
  `runs-on: ${{ needs.runner_names.outputs.<key> }}`
- a **composite label set** (`[self-hosted, Linux, X64]`) — exported as a JSON
  array string and consumed as
  `runs-on: ${{ fromJSON(needs.runner_names.outputs.<key>) }}`

Rolling `-latest` aliases (`ubuntu-latest`, `macos-latest`, `windows-latest`)
are intentionally left hardcoded — they are GitHub aliases, not fleet labels.

## `matrix.os` stays a logical identity

`matrix.os` values (and `matrix.os == '...'` conditionals) are **not** managed by
the catalog — they are frozen logical identities used in cache keys and
`runner.os` checks. A matrix lane pairs, e.g., `os: macos-26` with the
`qvac-macos26-arm64-gpu` runner.

A matrix row cannot carry the fork-authorization gate: a job-level `if:` is
evaluated before `matrix`, so it cannot see `matrix.os`. A matrix that mixes
hosted and self-hosted lanes therefore runs fork code on our hardware with no
approval. Split the self-hosted lane into its own job instead — which is what
`tts-ci.yml` does with `build-test` (hosted) and `build-test-macos` (gated):

```yaml
jobs:
  runner_names:
    permissions:
      contents: read
    uses: ./.github/workflows/reusable-runner-names.yml

  authorize:
    uses: ./.github/workflows/reusable-authorize-self-hosted.yml

  build-test:                        # hosted lanes only - no gate needed
    needs: runner_names
    strategy:
      matrix:
        os: [ubuntu-24.04]
    runs-on: ${{ needs.runner_names.outputs.ubuntu_2404 }}
    steps: ...

  build-test-macos:                  # self-hosted - its own job, gated
    needs: [runner_names, authorize]
    if: needs.authorize.outputs.allowed == 'true'
    runs-on: ${{ needs.runner_names.outputs.macos_arm64_gpu }}
    steps: ...
```

`validate-runner-names.mjs` does not check for the gate — it checks hardcoded
labels and the `runner_names` dependency — so this shape is the only thing
keeping fork code off the fleet.

Standalone (non-matrix) jobs are simpler:

```yaml
  gpu:
    needs: runner_names
    runs-on: ${{ needs.runner_names.outputs.linux_x64_gpu }}
    steps: ...
```

## Changing a label

1. Edit `.github/runners.yaml`.
2. Regenerate: `node .github/scripts/sync-runner-names.mjs`
3. Test: `node --test .github/scripts/test/runner-names.test.mjs`

CI enforces both invariants (catalog in sync with the generated workflow, and no
wired workflow hardcodes a catalog target) via
[`runner-names-validate.yml`](./workflows/runner-names-validate.yml), which runs
`node .github/scripts/validate-runner-names.mjs`.
