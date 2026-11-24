'use strict'

function formatNumber (value, digits = 3) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return Number(value).toFixed(digits)
}

function formatBytes (value) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`
}

function chartNumber (value, digits) {
  return Number(Number(value).toFixed(digits))
}

function renderBarChart (title, yAxis, entries, digits) {
  const values = entries.map(entry => entry.value)
  if (values.length === 0 || values.some(value => !Number.isFinite(value))) return []

  const lowerBound = Math.min(0, Math.min(...values) * 1.1)
  let upperBound = Math.max(0, Math.max(...values) * 1.1)
  if (lowerBound === upperBound) upperBound = lowerBound + 1

  return [
    '```mermaid',
    'xychart-beta',
    `  title ${JSON.stringify(title)}`,
    `  x-axis ${JSON.stringify(entries.map(entry => entry.label))}`,
    `  y-axis ${JSON.stringify(yAxis)} ${chartNumber(lowerBound, digits)} --> ${chartNumber(upperBound, digits)}`,
    `  bar [${values.map(value => chartNumber(value, digits)).join(', ')}]`,
    '```',
    ''
  ]
}

function overallMetricEntries (engines, selectValue) {
  return Object.entries(engines || {}).map(([name, stats]) => ({
    label: name,
    value: selectValue(stats)
  }))
}

function renderOverallCharts (engines) {
  const definitions = [
    {
      title: 'Median generation time (lower is better)',
      yAxis: 'Milliseconds',
      digits: 1,
      selectValue: stats => stats.generationMs.median
    },
    {
      title: 'Median real-time factor (lower is better)',
      yAxis: 'RTF',
      digits: 3,
      selectValue: stats => stats.rtf.median
    },
    {
      title: 'Median peak memory (lower is better)',
      yAxis: 'MiB',
      digits: 1,
      selectValue: stats => stats.peakRssBytes.median == null
        ? null
        : stats.peakRssBytes.median / (1024 * 1024)
    },
    {
      title: 'Median CLAP score (higher is better)',
      yAxis: 'Cosine similarity',
      digits: 3,
      selectValue: stats => stats.clap && stats.clap.median
    }
  ]
  const lines = ['## Visualizations', '']
  for (const definition of definitions) {
    const entries = overallMetricEntries(engines, definition.selectValue)
    lines.push(...renderBarChart(definition.title, definition.yAxis, entries, definition.digits))
  }
  return lines
}

function promptMetricEntries (byPrompt, engine) {
  return Object.entries(byPrompt || {}).map(([promptId, engines]) => ({
    label: promptId,
    value: engines[engine] && engines[engine].generationMs.median
  }))
}

function renderPromptCharts (byPrompt, engines) {
  const lines = ['### Median generation time by prompt', '']
  for (const engine of Object.keys(engines || {})) {
    const entries = promptMetricEntries(byPrompt, engine)
    lines.push(...renderBarChart(`${engine} by prompt`, 'Milliseconds', entries, 1))
  }
  return lines
}

function renderEngineTable (title, engines) {
  const lines = [
    `## ${title}`,
    '',
    '| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP | CLAP policy |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|'
  ]
  for (const [name, stats] of Object.entries(engines || {})) {
    const clapMedian = stats.clap && stats.clap.median
    const policy = clapMedian == null ? 'n/a' : (stats.clap.policyVersion || 'legacy/unversioned')
    lines.push(`| ${name} | ${stats.successCount} | ${stats.failureCount} | ${formatNumber(stats.generationMs.median, 1)} | ${formatNumber(stats.e2eMs.median, 1)} | ${formatNumber(stats.rtf.median, 3)} | ${formatBytes(stats.peakRssBytes.median)} | ${stats.uniqueWavHashes} | ${stats.silentCount} | ${formatNumber(clapMedian, 3)} | ${policy} |`)
  }
  lines.push('')
  return lines
}

function renderMarkdownReport (data) {
  const lines = [
    `# ACE-Step engine comparison (${data.meta.backend})`,
    '',
    `Generated: ${data.meta.generatedAt}`,
    '',
    `Comparison class: **${data.meta.comparisonClass}**. Platform: \`${data.meta.platform}\`. Threads: ${data.meta.threads}. Warm-ups: ${data.meta.warmups}. Timed runs: ${data.meta.runs}. Order: \`${data.meta.order}\`. Cooldown: ${data.meta.cooldownMs} ms.`,
    '',
    'Both engines used the same GGUF files, prompt manifest, duration, seed, LM sampling defaults, Euler sampler, and Haar DCW double-mode scalers. QVAC `music-cli` is a single process. `acestep.cpp` is `ace-lm` then `ace-synth`. QVAC lazy-loads stages inside `generate()`; acestep.cpp loads modules in each CLI process.',
    '',
    ...renderEngineTable('Overall', data.overall),
    ...renderOverallCharts(data.overall),
    ...renderPromptCharts(data.byPrompt, data.overall)
  ]
  for (const [promptId, engines] of Object.entries(data.byPrompt || {})) {
    lines.push(...renderEngineTable(promptId, engines))
  }
  lines.push('Failed rounds are retained in the JSON. WAV QC and optional CLAP scores are computed from files after generation and are not included in generation time or RTF.')
  lines.push('')
  return lines.join('\n')
}

module.exports = {
  formatBytes,
  formatNumber,
  renderMarkdownReport,
  renderOverallCharts
}
