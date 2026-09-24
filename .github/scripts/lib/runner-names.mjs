/**
 * Shared helpers for the qvac-fabric-llm.cpp runner-label catalog.
 *
 * Source of truth: .github/runners.yaml
 * Generated consumer: .github/workflows/reusable-runner-names.yml
 *
 * SCOPE: only the QVAC self-hosted fleet (`qvac-*`) in QVAC-authored workflows.
 * Upstream llama.cpp workflows and their hosted / third-party / non-QVAC
 * self-hosted labels are intentionally left alone (see .github/runners.yaml).
 *
 * A catalog entry is either a scalar label or an ordered array of labels
 * (composite self-hosted set). Array entries are exported as a JSON-array
 * string and consumed with `fromJSON(...)` in runs-on.
 */
import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
export const repoRoot = resolve(here, '../../..')

export const RUNNERS_YAML = '.github/runners.yaml'
export const REUSABLE_WORKFLOW = '.github/workflows/reusable-runner-names.yml'
export const REUSABLE_USES = './.github/workflows/reusable-runner-names.yml'

// Workflows wired to the catalog and validated. Upstream llama.cpp workflows
// that still run on upstream pools (release, docker, ui-build, ui, ui-publish,
// etc.) are intentionally excluded.
//
// QVAC-24501: the server* entries below are upstream workflows, added here
// because their jobs now target the QVAC fleet. They previously asked for
// upstream pools ([self-hosted, llama-server, ...], ah-* Actuated) that no
// fleet runner carries, so they queued to GitHub's 24h ceiling and were
// cancelled on every run. A wired workflow that is not listed here is not
// validated, which is how they drifted unnoticed.
const ADDON_WORKFLOWS = new Set([
  'build-self-hosted.yml',
  'ui-self-hosted.yml',
  'ui-build-self-hosted.yml',
  'python-lint.yml',
  'python-check-requirements.yml',
  'python-type-check.yml',
  'code-style.yml',
  'check-vendor.yml',
  'editorconfig.yml',
  'server-self-hosted.yml',
  'server-sanitize.yml',
  'build-cmake-pkg.yml',
  'build-sanitize.yml',
  'pre-tokenizer-hashes.yml',
  'update-ops-docs.yml',
])

const KEY = '[a-z][a-z0-9_]*'
const COMMENT = '(?:#\\s*(.*\\S))?'
const SCALAR_RE = new RegExp(`^(${KEY}):\\s+([^#\\[\\s'"][^#]*?)\\s*${COMMENT}\\s*$`)
const ARRAY_RE = new RegExp(`^(${KEY}):\\s+\\[([^\\]]+)\\]\\s*${COMMENT}\\s*$`)

export function parseRunnersYaml(source) {
  const runners = []
  const seenKeys = new Set()
  const seenTargets = new Set()

  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    const arrayMatch = line.match(ARRAY_RE)
    const scalarMatch = arrayMatch ? null : line.match(SCALAR_RE)
    if (!arrayMatch && !scalarMatch) {
      throw new Error(`invalid runners.yaml line (expected "key: label" or "key: [a, b]"): ${JSON.stringify(line)}`)
    }

    let entry
    if (arrayMatch) {
      const [, key, body, description = ''] = arrayMatch
      const labels = body.split(',').map((token) => token.trim()).filter(Boolean)
      if (labels.length === 0) throw new Error(`empty runner array: ${JSON.stringify(line)}`)
      for (const label of labels) {
        if (/["'{}[\]]/.test(label)) {
          throw new Error(`runner label must be bare, not quoted/structured: ${JSON.stringify(line)}`)
        }
      }
      entry = { key, kind: 'array', labels, description }
    } else {
      const [, key, rawLabel, description = ''] = scalarMatch
      const label = rawLabel.trim()
      if (/["'{}[\]]/.test(label)) {
        throw new Error(`runner label must be bare, not quoted/structured: ${JSON.stringify(line)}`)
      }
      entry = { key, kind: 'scalar', label, description }
    }

    const target = targetKeyString(entry)
    if (seenKeys.has(entry.key)) throw new Error(`duplicate runner key: ${entry.key}`)
    if (seenTargets.has(target)) throw new Error(`duplicate runner target: ${target}`)
    seenKeys.add(entry.key)
    seenTargets.add(target)
    runners.push(entry)
  }

  if (runners.length === 0) throw new Error('runners.yaml has no entries')
  return runners
}

/** Canonical string for a target, used for duplicate detection + matching. */
export function targetKeyString(entry) {
  return entry.kind === 'array' ? `[${entry.labels.join(',')}]` : entry.label
}

/** The output value the reusable exports: scalar label, or compact JSON array. */
export function outputValue(entry) {
  return entry.kind === 'array' ? JSON.stringify(entry.labels) : entry.label
}

/** How a caller references the output in `runs-on:`. */
export function runsOnExpression(entry) {
  const ref = `needs.runner_names.outputs.${entry.key}`
  return entry.kind === 'array' ? `\${{ fromJSON(${ref}) }}` : `\${{ ${ref} }}`
}

export function loadRunners() {
  return parseRunnersYaml(readFileSync(join(repoRoot, RUNNERS_YAML), 'utf8'))
}

export function listAddonWorkflows() {
  const directory = join(repoRoot, '.github/workflows')
  return readdirSync(directory)
    .filter((name) => ADDON_WORKFLOWS.has(name))
    .map((name) => `.github/workflows/${name}`)
    .sort()
}

export function renderReusableWorkflow(runners) {
  const outputsBlock = runners
    .map((entry) => {
      const description = entry.description || `Runner target ${targetKeyString(entry)}`
      return [
        `      ${entry.key}:`,
        `        description: ${yamlDoubleQuoted(description)}`,
        `        value: \${{ jobs.export.outputs.${entry.key} }}`,
      ].join('\n')
    })
    .join('\n')

  const jobOutputs = runners
    .map((entry) => `      ${entry.key}: \${{ steps.export.outputs.${entry.key} }}`)
    .join('\n')

  const exportLines = runners
    .map((entry) => `          echo '${entry.key}=${outputValue(entry)}' >> "$GITHUB_OUTPUT"`)
    .join('\n')

  return `# AUTO-GENERATED by .github/scripts/sync-runner-names.mjs
# Source of truth: ${RUNNERS_YAML}
# Do not edit this file by hand.

name: Runner names

on:
  workflow_call:
    outputs:
${outputsBlock}

permissions:
  contents: read

jobs:
  export:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    outputs:
${jobOutputs}
    steps:
      - name: Export runner labels
        id: export
        run: |
${exportLines}
`
}

export function assertReusableMatchesCatalog(runners, reusableSource) {
  const expected = renderReusableWorkflow(runners)
  if (normalizeNewlines(reusableSource) !== normalizeNewlines(expected)) {
    throw new Error(`${REUSABLE_WORKFLOW} is out of date. Run: node .github/scripts/sync-runner-names.mjs`)
  }
}

const LABEL_BOUNDARY_BEFORE = `(^|[\\s"'\\[,])`
const LABEL_BOUNDARY_AFTER = `([\\s"'\\],]|$)`

function escapeLabel(label) {
  return label.replaceAll(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function normalizeArrayTokens(body) {
  return body.split(',').map((token) => token.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean)
}

function isCommentLine(line) {
  return /^\s*#/.test(line)
}

/**
 * Flags catalog targets hardcoded in `runs-on:` where they select the machine.
 * Handles scalar forms (bare / quoted / inside a `${{ }}` expression) and the
 * composite flow-array form. Frozen logical identities are intentionally NOT
 * flagged: `matrix.os` values and `matrix.<x> == '<label>'` conditionals.
 */
export function findHardcodedLabelViolations(relativePath, source, runners) {
  const findings = []
  const lines = source.split(/\r?\n/)
  const scalars = runners.filter((entry) => entry.kind === 'scalar')
  const arrays = runners.filter((entry) => entry.kind === 'array')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (isCommentLine(line)) continue

    const assign = line.match(/^\s+runs-on:\s*(.*?)\s*$/)
    if (!assign) continue
    const value = assign[1]

    const arrayValue = value.match(/^\[([^\]]+)\]$/)
    if (arrayValue) {
      const tokens = normalizeArrayTokens(arrayValue[1])
      const match = arrays.find((entry) => sameSet(entry.labels, tokens))
      if (match) findings.push({ file: relativePath, line: i + 1, target: targetKeyString(match), text: line.trim() })
      continue
    }

    if (value.includes('${{')) {
      const scan = value.replace(/(==|!=)\s*(['"])[^'"]*\2/g, '')
      const hit = scalars.find((entry) => new RegExp(`['"]${escapeLabel(entry.label)}['"]`).test(scan))
      if (hit) findings.push({ file: relativePath, line: i + 1, target: hit.label, text: line.trim() })
      continue
    }

    const hit = scalars.find((entry) =>
      new RegExp(`${LABEL_BOUNDARY_BEFORE}${escapeLabel(entry.label)}${LABEL_BOUNDARY_AFTER}`).test(value),
    )
    if (hit) findings.push({ file: relativePath, line: i + 1, target: hit.label, text: line.trim() })
  }

  return findings
}

export function hasRunnerNamesJob(source) {
  return /^\s+runner_names:\s*$/m.test(source) && source.includes(`uses: ${REUSABLE_USES}`)
}

const RUNNER_NAMES_OUTPUTS = 'needs.runner_names.outputs'

/**
 * Split a workflow into its jobs. Line-based, like the rest of this module: a
 * job owns every line until the next key at its own indent.
 */
export function parseJobs(source) {
  const lines = source.split(/\r?\n/)
  const jobs = []
  let inJobs = false
  let jobIndent = null
  let current = null

  const close = () => {
    if (current) jobs.push(current)
    current = null
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (!inJobs) {
      if (/^jobs:\s*$/.test(line)) inJobs = true
      continue
    }
    // A key back at column 0 ends the mapping.
    if (line.trim() !== '' && !/^\s/.test(line)) {
      close()
      inJobs = false
      continue
    }

    const key = line.match(/^(\s+)([A-Za-z_][\w.-]*):\s*(.*)$/)
    if (key) {
      const indent = key[1].length
      if (jobIndent === null) jobIndent = indent
      if (indent === jobIndent) {
        close()
        current = { name: key[2], line: i + 1, indent, lines: [] }
        continue
      }
    }
    if (current) current.lines.push(line)
  }
  close()
  return jobs
}

function unquote(token) {
  return token.trim().replace(/^['"]|['"]$/g, '')
}

/** Derived, not assumed: workflows here nest at both 2 and 4 spaces. */
function propertyIndentOf(job) {
  let indent = null
  for (const line of job.lines) {
    const key = line.match(/^(\s+)[A-Za-z_][\w.-]*:/)
    if (!key) continue
    if (indent === null || key[1].length < indent) indent = key[1].length
  }
  return indent
}

/** Job names listed in a job's own `needs:`, in any of the three YAML forms. */
export function declaredNeeds(job) {
  const needs = []
  const propertyIndent = propertyIndentOf(job)
  if (propertyIndent === null) return needs

  for (let i = 0; i < job.lines.length; i++) {
    const line = job.lines[i]
    const match = line.match(/^(\s+)needs:\s*(.*?)\s*$/)
    // Indent check: the job's own `needs:`, not one inside a step script.
    if (!match || match[1].length !== propertyIndent) continue

    const value = match[2].replace(/\s+#.*$/, '').trim()
    if (value === '') {
      // Block sequence.
      for (let j = i + 1; j < job.lines.length; j++) {
        const item = job.lines[j].match(/^\s+-\s*(.+?)\s*$/)
        if (!item) break
        needs.push(unquote(item[1]))
      }
      continue
    }

    const array = value.match(/^\[(.*)\]$/)
    if (array) needs.push(...array[1].split(',').map(unquote).filter(Boolean))
    else needs.push(unquote(value))
  }

  return needs
}

/**
 * Two ways a consumer loses its label - no `runner_names` job in the file, or a
 * job reading its outputs without depending on it. Both resolve `runs-on` to
 * the empty string and leave the job unschedulable while the YAML stays valid.
 */
export function findMissingRunnerNamesNeeds(relativePath, source) {
  if (!source.includes(RUNNER_NAMES_OUTPUTS)) return []
  if (!hasRunnerNamesJob(source)) {
    return [{ file: relativePath, message: `references ${RUNNER_NAMES_OUTPUTS} but has no runner_names job using ${REUSABLE_USES}` }]
  }

  const findings = []
  for (const job of parseJobs(source)) {
    if (!job.lines.some((line) => line.includes(RUNNER_NAMES_OUTPUTS))) continue
    if (declaredNeeds(job).includes('runner_names')) continue
    findings.push({
      file: relativePath,
      line: job.line,
      message: `job "${job.name}" reads ${RUNNER_NAMES_OUTPUTS} but does not list runner_names in its own needs:`,
    })
  }
  return findings
}

function sameSet(a, b) {
  if (a.length !== b.length) return false
  const sortedA = [...a].sort()
  const sortedB = [...b].sort()
  return sortedA.every((value, index) => value === sortedB[index])
}

function yamlDoubleQuoted(value) {
  return `"${value.replaceAll('\\', '\\\\').replaceAll('"', '\\"')}"`
}

function normalizeNewlines(text) {
  return text.replaceAll('\r\n', '\n')
}

export function readRepoFile(relativePath) {
  return readFileSync(join(repoRoot, relativePath), 'utf8')
}
