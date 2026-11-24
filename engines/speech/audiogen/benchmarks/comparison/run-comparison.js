#!/usr/bin/env node
'use strict'

const { spawnSync } = require('child_process')
const fs = require('fs')
const path = require('path')
const { runAcestepRound } = require('./adapters/acestep')
const { runQvacRound } = require('./adapters/qvac')
const { loadHarnessConfig } = require('./lib/config')
const { writeJson } = require('./lib/process')
const { writeReports } = require('./lib/results')

function sleep (ms) {
  if (!ms) return
  spawnSync('sleep', [String(ms / 1000)])
}

function parseArguments (argv) {
  const args = argv.slice(2)
  const flags = {
    help: false,
    dryRun: false,
    force: false,
    backend: null,
    promptIds: null
  }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '--help' || arg === '-h') flags.help = true
    else if (arg === '--dry-run') flags.dryRun = true
    else if (arg === '--force') flags.force = true
    else if (arg === '--backend') flags.backend = args[++i]
    else if (arg === '--prompts') flags.promptIds = args[++i]
    else throw new Error(`unknown argument: ${arg}`)
  }
  return flags
}

function printHelp (config) {
  console.log(`ACE-Step direct engine comparison

Usage:
  node run-comparison.js [--dry-run] [--force] [--backend cpu|metal|cuda|vulkan] [--prompts id,id]

Measures QVAC music-cli against acestep.cpp ace-lm + ace-synth. No addon, SDK,
or packages/audiogen-ggml is used.

Current configuration:
  QVAC CLI:     ${config.qvacCli}
  ace-lm:       ${config.aceLm}
  ace-synth:    ${config.aceSynth}
  models:       ${config.modelsDir}
  output:       ${config.outDir}
  backend:      ${config.backend}
  threads:      ${config.threads}
  warmups/runs: ${config.warmups}/${config.runs}
  order:        ${config.order}
  prompts:      ${config.prompts.map(prompt => prompt.id).join(', ')}
`)
}

function requireFile (label, filePath, checks) {
  const exists = fs.existsSync(filePath)
  console.log(`  ${exists ? 'found  ' : 'missing'} ${label}: ${filePath}`)
  checks.push(exists)
  return exists
}

function validateConfig (config) {
  console.log('Discovery:')
  const checks = []
  requireFile('QVAC music-cli', config.qvacCli, checks)
  requireFile('ace-lm', config.aceLm, checks)
  requireFile('ace-synth', config.aceSynth, checks)
  for (const [role, filename] of Object.entries(config.models)) {
    requireFile(`${role} GGUF`, path.join(config.modelsDir, filename), checks)
  }
  const ready = checks.every(Boolean)
  if (!ready) {
    console.log('Configuration is incomplete. Stage models and build both CLIs before running.')
  }
  return ready
}

function roundPath (config, engine, promptId, kind, index) {
  return path.join(config.outDir, 'rounds', config.backend, engine, promptId, `${kind}-${index}.json`)
}

function wavPath (config, engine, promptId, kind, index) {
  return path.join(config.outDir, 'samples', config.backend, engine, promptId, `${kind}-${index}.wav`)
}

function engineOrder (config, promptIndex) {
  if (config.order === 'qvac-first') return ['qvac', 'acestep.cpp']
  if (config.order === 'acestep-first') return ['acestep.cpp', 'qvac']
  if (config.order === 'random') return Math.random() < 0.5 ? ['qvac', 'acestep.cpp'] : ['acestep.cpp', 'qvac']
  return promptIndex % 2 === 0 ? ['qvac', 'acestep.cpp'] : ['acestep.cpp', 'qvac']
}

function runOne (config, engine, prompt, kind, index, force) {
  const jsonPath = roundPath(config, engine, prompt.id, kind, index)
  const outputWav = wavPath(config, engine, prompt.id, kind, index)
  if (!force && fs.existsSync(jsonPath)) {
    return JSON.parse(fs.readFileSync(jsonPath, 'utf8'))
  }
  fs.mkdirSync(path.dirname(outputWav), { recursive: true })
  const started = new Date().toISOString()
  let result
  try {
    result = engine === 'qvac'
      ? runQvacRound(config, prompt, outputWav)
      : runAcestepRound(config, prompt, outputWav)
  } catch (error) {
    result = {
      engine,
      promptId: prompt.id,
      ok: false,
      error: error.message,
      code: error.code || null,
      e2eMs: null,
      generationMs: null
    }
  }
  const record = {
    ...result,
    kind,
    index,
    started,
    finished: new Date().toISOString(),
    wavPath: fs.existsSync(outputWav) ? outputWav : null
  }
  writeJson(jsonPath, record)
  writeJson(jsonPath.replace(/\.json$/, '.log.json'), {
    stdout: record.stdout || '',
    stderr: record.stderr || ''
  })
  return record
}

function collectRounds (config, force) {
  const rounds = []
  config.prompts.forEach((prompt, promptIndex) => {
    const engines = engineOrder(config, promptIndex)
    for (const engine of engines) {
      for (let i = 0; i < config.warmups; i++) {
        console.log(`warmup ${engine} ${prompt.id} #${i}`)
        rounds.push(runOne(config, engine, prompt, 'warmup', i, force))
        sleep(config.cooldownMs)
      }
      for (let i = 0; i < config.runs; i++) {
        console.log(`timed ${engine} ${prompt.id} #${i}`)
        rounds.push(runOne(config, engine, prompt, 'timed', i, force))
        sleep(config.cooldownMs)
      }
    }
  })
  return rounds
}

function main () {
  const flags = parseArguments(process.argv)
  const config = loadHarnessConfig({
    backend: flags.backend,
    promptIds: flags.promptIds
  })
  if (flags.help) {
    printHelp(config)
    return
  }
  printHelp(config)
  const ready = validateConfig(config)
  if (flags.dryRun) return
  if (!ready) throw new Error('comparison configuration is incomplete')
  fs.mkdirSync(config.outDir, { recursive: true })
  const written = writeReports(config, collectRounds(config, flags.force))
  console.log(`Wrote ${written.jsonPath}`)
  console.log(`Wrote ${written.markdownPath}`)
}

try {
  main()
} catch (error) {
  console.error(`error: ${error.message}`)
  process.exitCode = 1
}
