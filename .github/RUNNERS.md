# Runner label catalog

The QVAC self-hosted runner labels (`qvac-*`) used by this fork's **QVAC-authored**
CI live in one place: [`.github/runners.yaml`](./runners.yaml). This lets a
fleet migration (GPU-pool rename, OS bump on the self-hosted boxes) update a
single file instead of grepping every workflow.

## Scope

This catalog is deliberately narrow. It manages **only** the `qvac-*` fleet
labels, in the workflows listed in `ADDON_WORKFLOWS`
([`.github/scripts/lib/runner-names.mjs`](./scripts/lib/runner-names.mjs)) —
that list is the enforcement boundary, and a workflow wired to the catalog but
missing from it is not validated at all. It does **not** touch:

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

A reusable workflow called with `uses:` cannot read the caller's `needs:`, so it
takes the label as a `workflow_call` input instead of declaring its own
`runner_names` job — see `ui-build-self-hosted.yml`. A second `runner_names` job
would otherwise burn another GitHub-hosted job per run to echo the same
constants.

## Wiring a job to the catalog

Both lines matter. `runs-on:` is evaluated before any step runs, so a job that
reads `needs.runner_names.outputs.*` without listing `runner_names` in its own
`needs:` resolves the label to the empty string and becomes unschedulable while
the YAML stays valid:

```yaml
jobs:
  runner_names:
    permissions:
      contents: read
    uses: ./.github/workflows/reusable-runner-names.yml

  my-job:
    needs: [authorize, runner_names]   # runner_names, not just authorize
    if: needs.authorize.outputs.allowed == 'true'
    runs-on: ${{ needs.runner_names.outputs.qvac_ubuntu2404_x64 }}
```

`validate-runner-names.mjs` checks this per job, and also fails a workflow that
adds a new entry to `ADDON_WORKFLOWS`-listed files while hardcoding a catalog
label.

## Changing a label

1. Edit `.github/runners.yaml`.
2. Regenerate: `node .github/scripts/sync-runner-names.mjs`
3. Test: `node --test .github/scripts/test/runner-names.test.mjs`

CI enforces all of it via
[`runner-names-validate.yml`](./workflows/runner-names-validate.yml).
