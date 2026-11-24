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

test('runners.yaml parses with unique keys/targets', () => {
  const runners = loadRunners()
  assert.ok(runners.length >= 5)
  assert.ok(runners.some((e) => e.kind === 'scalar' && e.label === 'ubuntu-24.04'))
  assert.ok(runners.some((e) => e.label === 'qvac-ubuntu2204-x64-gpu'))
  assert.ok(runners.some((e) => e.label === 'qvac-macos26-arm64-gpu'))
  assert.equal(new Set(runners.map((e) => e.key)).size, runners.length)
})

test('array vs scalar output value + runs-on expression', () => {
  const [scalar] = parseRunnersYaml('linux_x64_gpu: qvac-ubuntu2204-x64-gpu\n')
  assert.equal(outputValue(scalar), 'qvac-ubuntu2204-x64-gpu')
  assert.equal(runsOnExpression(scalar), '${{ needs.runner_names.outputs.linux_x64_gpu }}')

  const [arr] = parseRunnersYaml('gpu: [self-hosted, Linux, X64]\n')
  assert.equal(outputValue(arr), '["self-hosted","Linux","X64"]')
  assert.equal(runsOnExpression(arr), '${{ fromJSON(needs.runner_names.outputs.gpu) }}')
})

test('parseRunnersYaml rejects quoted values and junk', () => {
  assert.throws(() => parseRunnersYaml('k: "ubuntu-24.04"\n'), /invalid/)
  assert.throws(() => parseRunnersYaml('k: [self-hosted, "Linux"]\n'), /bare/)
  assert.throws(() => parseRunnersYaml('k: macos\nk: other\n'), /duplicate runner key/)
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

test('CI workflows do not hardcode catalog runner targets', () => {
  const runners = loadRunners()
  const findings = []
  for (const file of listAddonWorkflows()) {
    const source = readRepoFile(file)
    findings.push(...findHardcodedLabelViolations(file, source, runners))
    findings.push(...findMissingRunnerNamesNeeds(file, source))
  }
  assert.deepEqual(findings, [], JSON.stringify(findings, null, 2))
})

test('detector catches scalar hardcoded runs-on but not matrix.os identities', () => {
  const runners = parseRunnersYaml(
    'ubuntu_2404: ubuntu-24.04\nmacos_14: macos-14\nlinux_x64_gpu: qvac-ubuntu2204-x64-gpu\n',
  )
  const source = [
    'jobs:',
    '  a:',
    '    runs-on: ubuntu-24.04',
    '  b:',
    '    runs-on: qvac-ubuntu2204-x64-gpu',
    '  c:',
    "    runs-on: ${{ matrix.os == 'macos-14' && needs.runner_names.outputs.macos_14 || needs.runner_names.outputs.ubuntu_2404 }}",
    '  d:',
    '    runs-on: ${{ matrix.os }}',
    '  e:',
    '    runs-on: ubuntu-latest',
    '',
  ].join('\n')
  const findings = findHardcodedLabelViolations('x.yml', source, runners)
  assert.deepEqual(
    findings.map((f) => [f.line, f.target]).sort((a, b) => a[0] - b[0]),
    [
      [3, 'ubuntu-24.04'],
      [5, 'qvac-ubuntu2204-x64-gpu'],
    ],
  )
})

test('detector flags a quoted catalog label used as a value in an expression', () => {
  const runners = parseRunnersYaml('linux_x64_gpu: qvac-ubuntu2204-x64-gpu\n')
  const source = ['jobs:', '  a:', "    runs-on: ${{ inputs.gpu && 'qvac-ubuntu2204-x64-gpu' || 'ubuntu-latest' }}", ''].join('\n')
  assert.equal(findHardcodedLabelViolations('x.yml', source, runners).length, 1)
})

test('validate-runner-names.mjs exits 0', () => {
  const result = spawnSync(process.execPath, [join(repoRoot, '.github/scripts/validate-runner-names.mjs')], {
    encoding: 'utf8',
    cwd: repoRoot,
  })
  assert.equal(result.status, 0, `stdout=${result.stdout}\nstderr=${result.stderr}`)
})

test('catalog path constant points at a tracked file', () => {
  assert.equal(RUNNERS_YAML, '.github/runners.yaml')
  assert.match(readRepoFile(RUNNERS_YAML), /qvac-macos26-arm64-gpu/)
})
