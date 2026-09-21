# Runner label catalog

The QVAC self-hosted runner labels (`qvac-*`) used by this fork's **QVAC-authored**
CI live in one place: [`.github/runners.yaml`](./runners.yaml). This lets a
fleet migration (GPU-pool rename, OS bump on the self-hosted boxes) update a
single file instead of grepping every workflow.

## Scope

This catalog is deliberately narrow. It manages **only** the `qvac-*` fleet
labels in QVAC-authored workflows (`build-self-hosted.yml`, `ui-self-hosted.yml`,
`ui-build-self-hosted.yml`, `python-*`, `code-style`, `check-vendor`,
`editorconfig`). It does **not** touch:

- GitHub rolling aliases (`ubuntu-latest`, …) or upstream llama.cpp hosted images
  (`ubuntu-24.04`, `windows-2025`, `macos-26`, …)
- upstream / third-party self-hosted pools that are not the QVAC fleet
  (`[self-hosted, llama-server, …]`, `[self-hosted, Linux, Intel]`, `ah-*`
  Actuated runners, `ai-run-*`)

Centralizing those would diverge this fork from `ggml-org/llama.cpp` on every
upstream sync, so they stay as-is.

## How it works

`runners.yaml` is the source of truth. A generated reusable workflow,
[`.github/workflows/reusable-runner-names.yml`](./workflows/reusable-runner-names.yml),
exports each catalog entry as a job output. Callers pull the label from that
output instead of hardcoding it, because `runs-on:` is evaluated before any step
runs, so a reusable workflow's outputs are the only way to feed it a
centrally-defined label.

```yaml
jobs:
  runner_names:
    permissions:
      contents: read
    uses: ./.github/workflows/reusable-runner-names.yml

  gpu-cuda:
    needs: runner_names
    runs-on: ${{ needs.runner_names.outputs.qvac_ubuntu2204_x64_gpu }}
    steps: ...
```

## Changing a label

1. Edit `.github/runners.yaml`.
2. Regenerate: `node .github/scripts/sync-runner-names.mjs`
3. Test: `node --test .github/scripts/test/runner-names.test.mjs`

CI enforces both invariants via
[`runner-names-validate.yml`](./workflows/runner-names-validate.yml).
