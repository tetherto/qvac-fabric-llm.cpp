'use strict'

const fs = require('fs')
const os = require('os')
const path = require('path')
const { aggregateRounds, groupByPrompt } = require('./aggregate')
const { fileSha256, writeJson } = require('./process')
const { renderMarkdownReport } = require('./report')

function collectModelHashes (config) {
  const hashes = {}
  for (const [role, filename] of Object.entries(config.models)) {
    const filePath = path.join(config.modelsDir, filename)
    hashes[role] = fs.existsSync(filePath) ? fileSha256(filePath) : null
  }
  return hashes
}

function collectJsonFiles (dir, acc) {
  if (!fs.existsSync(dir)) return acc
  for (const name of fs.readdirSync(dir)) {
    const filePath = path.join(dir, name)
    const stat = fs.statSync(filePath)
    if (stat.isDirectory()) {
      collectJsonFiles(filePath, acc)
    } else if (name.endsWith('.json') && !name.endsWith('.log.json')) {
      acc.push(filePath)
    }
  }
  return acc
}

function loadRoundRecords (roundsRoot) {
  return collectJsonFiles(roundsRoot, []).map(filePath => JSON.parse(fs.readFileSync(filePath, 'utf8')))
}

function buildResultsDocument (config, rounds) {
  const timed = rounds.filter(round => round.kind === 'timed')
  return {
    meta: {
      generatedAt: new Date().toISOString(),
      platform: `${process.platform}-${process.arch}`,
      hostname: os.hostname(),
      backend: config.backend,
      comparisonClass: config.comparisonClass,
      threads: config.threads,
      warmups: config.warmups,
      runs: config.runs,
      order: config.order,
      cooldownMs: config.cooldownMs,
      qvacCli: config.qvacCli,
      aceLm: config.aceLm,
      aceSynth: config.aceSynth,
      models: config.models,
      modelHashes: collectModelHashes(config),
      parameters: config.parameters,
      clap: config.clap || null
    },
    overall: aggregateRounds(timed),
    byPrompt: groupByPrompt(timed),
    rounds
  }
}

function writeReports (config, rounds) {
  const data = buildResultsDocument(config, rounds)
  const jsonPath = path.join(config.outDir, `${config.backend}.json`)
  const markdownPath = path.join(config.outDir, `${config.backend}.md`)
  writeJson(jsonPath, data)
  const markdown = renderMarkdownReport(data)
  fs.writeFileSync(markdownPath, markdown.endsWith('\n') ? markdown : `${markdown}\n`)
  return { jsonPath, markdownPath }
}

module.exports = {
  buildResultsDocument,
  collectJsonFiles,
  collectModelHashes,
  loadRoundRecords,
  writeReports
}
