#!/usr/bin/env node
'use strict'

const { spawnSync } = require('child_process')
const fs = require('fs')
const path = require('path')
const {
  CLAP_POLICY_VERSION,
  buildClapText,
  lookupPrompt,
  mergeClapScore,
  parseClapBatchOutput,
  shouldScoreRound
} = require('./lib/clap')
const { loadHarnessConfig } = require('./lib/config')
const { writeJson } = require('./lib/process')
const { comparisonRoot } = require('./lib/paths')
const { loadRoundRecords, collectJsonFiles, writeReports } = require('./lib/results')

function parseArguments (argv) {
  const args = argv.slice(2)
  const flags = {
    help: false,
    force: false,
    backend: null,
    includeWarmup: false
  }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '--help' || arg === '-h') flags.help = true
    else if (arg === '--force') flags.force = true
    else if (arg === '--backend') flags.backend = args[++i]
    else if (arg === '--include-warmup') flags.includeWarmup = true
    else throw new Error(`unknown argument: ${arg}`)
  }
  return flags
}

function printHelp () {
  console.log(`Score saved comparison WAVs with LAION CLAP (off the inference clock).

Usage:
  node score-clap.js [--backend cpu|metal|cuda|vulkan] [--force] [--include-warmup]

Does not run music-cli or ace-synth. Generation commands are unchanged.
Requires Python deps from quality/requirements.txt and a CLAP checkpoint
(downloaded by transformers on first use into the Hugging Face cache).
`)
}

function pythonCheck (pythonBin) {
  const probe = spawnSync(pythonBin, ['--version'], { encoding: 'utf8' })
  if (probe.error && probe.error.code === 'ENOENT') {
    throw new Error(`CLAP python not found: ${pythonBin}. Install Python 3.10+ or set ACESTEP_CLAP_PYTHON`)
  }
  if (probe.status !== 0) {
    throw new Error(`${pythonBin} --version failed: ${probe.stderr || probe.stdout}`)
  }
}

function collectPendingRounds (records, prompts, options, textPolicy) {
  const pending = []
  for (const record of records) {
    if (!shouldScoreRound(record, options)) continue
    const prompt = lookupPrompt(prompts, record.promptId)
    pending.push({
      record,
      item: {
        id: `${record.engine}/${record.promptId}/${record.kind}-${record.index}`,
        wav: record.wavPath,
        text: buildClapText(prompt, textPolicy)
      }
    })
  }
  return pending
}

function runClapBatch (config, items) {
  const script = path.join(comparisonRoot(), 'quality', 'clap_score.py')
  if (!fs.existsSync(script)) {
    throw new Error(`missing CLAP scorer: ${script}`)
  }
  const batchPath = path.join(config.outDir, `clap-batch-${config.backend}.json`)
  writeJson(batchPath, { items })
  const args = [
    script,
    '--batch', batchPath,
    '--model', config.clap.model,
    '--device', config.clap.device
  ]
  if (config.clap.revision) args.push('--revision', config.clap.revision)
  const result = spawnSync(config.clap.python, args, {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    env: {
      ...process.env,
      PYTHONWARNINGS: 'ignore',
      TRANSFORMERS_VERBOSITY: 'error',
      HF_HUB_DISABLE_PROGRESS_BARS: '1',
      TOKENIZERS_PARALLELISM: 'false'
    }
  })
  if (result.status !== 0) {
    const output = `${result.stdout || ''}\n${result.stderr || ''}`.trim()
    try {
      const parsed = JSON.parse((result.stdout || '').trim())
      if (parsed && parsed.error) {
        throw new Error(parsed.error)
      }
    } catch (error) {
      if (error.message && error.message.includes('CLAP')) throw error
    }
    throw new Error(`clap_score.py exited ${result.status}: ${output.slice(0, 2000)}`)
  }
  return parseClapBatchOutput(result.stdout)
}

function applyScores (pending, batch, textPolicy) {
  if (batch.policyVersion !== CLAP_POLICY_VERSION) {
    throw new Error(`incompatible CLAP scorer policy: ${batch.policyVersion || 'missing'}; expected ${CLAP_POLICY_VERSION}`)
  }
  const byId = new Map()
  for (const score of batch.scores) {
    byId.set(score.id, score)
  }
  const meta = {
    model: batch.model,
    revision: batch.revision,
    samplingRate: batch.samplingRate,
    policyVersion: batch.policyVersion,
    device: batch.device,
    textPolicy
  }
  const updated = []
  for (const entry of pending) {
    const scoreItem = byId.get(entry.item.id) || {
      id: entry.item.id,
      ok: false,
      score: null,
      error: 'CLAP batch omitted this item'
    }
    updated.push(mergeClapScore(entry.record, scoreItem, meta))
  }
  return updated
}

function replaceScoredRounds (records, scored) {
  const byKey = new Map()
  for (const round of scored) {
    byKey.set(`${round.engine}/${round.promptId}/${round.kind}-${round.index}`, round)
  }
  return records.map(record => {
    const key = `${record.engine}/${record.promptId}/${record.kind}-${record.index}`
    return byKey.get(key) || record
  })
}

function persistRounds (config, records) {
  const roundsRoot = path.join(config.outDir, 'rounds', config.backend)
  const files = collectJsonFiles(roundsRoot, [])
  const byKey = new Map()
  for (const record of records) {
    byKey.set(`${record.engine}/${record.promptId}/${record.kind}-${record.index}`, record)
  }
  for (const filePath of files) {
    const current = JSON.parse(fs.readFileSync(filePath, 'utf8'))
    const key = `${current.engine}/${current.promptId}/${current.kind}-${current.index}`
    const next = byKey.get(key)
    if (next) writeJson(filePath, next)
  }
}

function main () {
  const flags = parseArguments(process.argv)
  if (flags.help) {
    printHelp()
    return
  }
  const config = loadHarnessConfig({ backend: flags.backend })
  pythonCheck(config.clap.python)
  const roundsRoot = path.join(config.outDir, 'rounds', config.backend)
  const records = loadRoundRecords(roundsRoot)
  if (!records.length) {
    throw new Error(`no round JSON under ${roundsRoot}. Run node run-comparison.js --backend ${config.backend} first`)
  }
  const pending = collectPendingRounds(
    records,
    config.prompts,
    { force: flags.force, timedOnly: !flags.includeWarmup },
    config.clap.textPolicy
  )
  if (!pending.length) {
    console.log('No WAV rounds need CLAP (already scored, or none timed). Use --force to rescore.')
  } else {
    console.log(`Scoring ${pending.length} WAV(s) with ${config.clap.model} on ${config.clap.device}`)
    const batch = runClapBatch(config, pending.map(entry => entry.item))
    const scored = applyScores(pending, batch, config.clap.textPolicy)
    persistRounds(config, replaceScoredRounds(records, scored))
  }
  const written = writeReports(config, loadRoundRecords(roundsRoot))
  console.log(`Wrote ${written.jsonPath}`)
  console.log(`Wrote ${written.markdownPath}`)
}

if (require.main === module) {
  try {
    main()
  } catch (error) {
    console.error(`error: ${error.message}`)
    process.exitCode = 1
  }
}

module.exports = { applyScores, collectPendingRounds, persistRounds, replaceScoredRounds }
