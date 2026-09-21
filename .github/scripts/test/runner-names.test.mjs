import test from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import {
  REUSABLE_WORKFLOW,
  RUNNERS_YAML,
  assertReusableMatchesCatalog,
  findHardcodedLabelViolations,
  findMissingRunnerNamesNeeds,
  listAddonWorkflows,
  loadRunners,
  outputValue,
  parseRunnersYaml,
  readRepoFile,
  renderReusableWorkflow,
  repoRoot,
  runsOnExpression,
} from '../lib/runner-names.mjs'

test('runners.yaml parses qvac-* fleet with unique keys/targets', () => {
  const runners = loadRunners()
  assert.ok(runners.length >= 1)
  assert.ok(runners.every((e) => e.label.startsWith('qvac-')))
  assert.equal(new Set(runners.map((e) => e.key)).size, runners.length)
})

test('array vs scalar output value + runs-on expression', () => {
  const [scalar] = parseRunnersYaml('qvac_ubuntu2404_x64: qvac-ubuntu2404-x64\n')
  assert.equal(outputValue(scalar), 'qvac-ubuntu2404-x64')
  assert.equal(runsOnExpression(scalar), '${{ needs.runner_names.outputs.qvac_ubuntu2404_x64 }}')

  const [arr] = parseRunnersYaml('gpu: [self-hosted, Linux, X64]\n')
  assert.equal(outputValue(arr), '["self-hosted","Linux","X64"]')
  assert.equal(runsOnExpression(arr), '${{ fromJSON(needs.runner_names.outputs.gpu) }}')
})

test('parseRunnersYaml rejects quoted values and junk', () => {
  assert.throws(() => parseRunnersYaml('k: "qvac-ubuntu2404-x64"\n'), /invalid/)
  assert.throws(() => parseRunnersYaml('k: [self-hosted, "Linux"]\n'), /bare/)
  assert.throws(() => parseRunnersYaml('k: a\nk: b\n'), /duplicate runner key/)
  assert.throws(() => parseRunnersYaml('not yaml at all\n'), /invalid/)
})

test('reusable-runner-names.yml matches the catalog', () => {
  const runners = loadRunners()
  assert.doesNotThrow(() => assertReusableMatchesCatalog(runners, readRepoFile(REUSABLE_WORKFLOW)))
  const rendered = renderReusableWorkflow(runners)
  assert.match(rendered, /AUTO-GENERATED/)
  assert.match(rendered, /runs-on: ubuntu-latest/)
  assert.doesNotMatch(rendered, /actions\/checkout/)
})

test('wired workflows do not hardcode catalog runner targets', () => {
  const runners = loadRunners()
  const findings = []
  for (const file of listAddonWorkflows()) {
    const source = readRepoFile(file)
    findings.push(...findHardcodedLabelViolations(file, source, runners))
    findings.push(...findMissingRunnerNamesNeeds(file, source))
  }
  assert.deepEqual(findings, [], JSON.stringify(findings, null, 2))
})

test('detector flags qvac-* runs-on but leaves non-fleet self-hosted pools alone', () => {
  const runners = parseRunnersYaml(
    'qvac_gpu: qvac-ubuntu2204-x64-gpu\nqvac_mac: qvac-macos26-arm64-gpu\n',
  )
  const source = [
    'jobs:',
    '  a:',
    '    runs-on: qvac-ubuntu2204-x64-gpu',
    '  b:',
    '    runs-on: [self-hosted, Linux, Intel, OpenVINO]',
    '  c:',
    '    runs-on: ah-ubuntu_22_04-c8g_8x',
    '  d:',
    '    runs-on: ${{ needs.runner_names.outputs.qvac_mac }}',
    '  e:',
    '    runs-on: ubuntu-24.04',
    '',
  ].join('\n')
  const findings = findHardcodedLabelViolations('x.yml', source, runners)
  assert.deepEqual(
    findings.map((f) => [f.line, f.target]),
    [[3, 'qvac-ubuntu2204-x64-gpu']],
  )
})

test('validate-runner-names.mjs exits 0', () => {
  const result = spawnSync(process.execPath, [join(repoRoot, '.github/scripts/validate-runner-names.mjs')], {
    encoding: 'utf8',
    cwd: repoRoot,
  })
  assert.equal(result.status, 0, `stdout=${result.stdout}\nstderr=${result.stderr}`)
})

test('catalog path constant is correct', () => {
  assert.equal(RUNNERS_YAML, '.github/runners.yaml')
})
