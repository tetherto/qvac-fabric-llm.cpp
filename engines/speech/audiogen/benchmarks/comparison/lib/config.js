'use strict'

const fs = require('fs')
const path = require('path')
const { comparisonRoot, engineDir, repositoryDir, workspaceDir } = require('./paths')

const DEFAULT_CONFIG_PATH = path.join(comparisonRoot(), 'config.json')
const DEFAULT_MANIFEST_PATH = path.join(comparisonRoot(), 'prompts', 'manifest.json')

function firstSet (...values) {
  return values.find(value => value !== undefined && value !== '')
}

function integerEnv (name, fallback) {
  const value = process.env[name]
  if (value === undefined || value === '') return fallback
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer`)
  }
  return parsed
}

function firstExisting (root, candidates) {
  return candidates
    .map(candidate => path.join(root, candidate))
    .find(candidate => fs.existsSync(candidate))
}

function readJson (filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

function loadHarnessConfig (overrides = {}) {
  const configPath = path.resolve(overrides.configPath || process.env.ACESTEP_COMPARE_CONFIG || DEFAULT_CONFIG_PATH)
  const manifestPath = path.resolve(overrides.manifestPath || process.env.ACESTEP_COMPARE_MANIFEST || DEFAULT_MANIFEST_PATH)
  const fileConfig = readJson(configPath)
  const manifest = readJson(manifestPath)
  if (!Array.isArray(manifest.prompts) || manifest.prompts.length === 0) {
    throw new Error(`prompt manifest must contain a non-empty prompts array: ${manifestPath}`)
  }

  const acestepDir = path.resolve(firstSet(
    process.env.ACESTEP_CPP_DIR,
    path.join(comparisonRoot(), 'vendor', 'acestep.cpp'),
    path.join(workspaceDir(), 'acestep.cpp')
  ))

  const qvacCli = path.resolve(firstSet(
    process.env.ACESTEP_QVAC_CLI,
    firstExisting(repositoryDir(), [
      'build/audiogen/music-cli',
      'build/audiogen/music-cli',
      'audiogen/build/music-cli',
      'audiogen/build-cpu/music-cli',
      'audiogen/build-metal/music-cli'
    ]),
    path.join(engineDir(), 'build', 'music-cli')
  ))

  const aceLm = path.resolve(firstSet(
    process.env.ACESTEP_CPP_LM,
    firstExisting(acestepDir, [
      'build/ace-lm',
      'build/Release/ace-lm',
      'build-cpu/ace-lm',
      'build-metal/ace-lm'
    ]),
    path.join(acestepDir, 'build', 'ace-lm')
  ))

  const aceSynth = path.resolve(firstSet(
    process.env.ACESTEP_CPP_SYNTH,
    firstExisting(acestepDir, [
      'build/ace-synth',
      'build/Release/ace-synth',
      'build-cpu/ace-synth',
      'build-metal/ace-synth'
    ]),
    path.join(acestepDir, 'build', 'ace-synth')
  ))

  const requestedIds = (overrides.promptIds || process.env.ACESTEP_COMPARE_PROMPTS || fileConfig.promptIds || [])
  const idList = Array.isArray(requestedIds)
    ? requestedIds
    : String(requestedIds).split(',').map(value => value.trim()).filter(Boolean)
  const prompts = idList.length
    ? manifest.prompts.filter(prompt => idList.includes(prompt.id))
    : manifest.prompts
  if (idList.length && prompts.length !== idList.length) {
    const found = new Set(prompts.map(prompt => prompt.id))
    const missing = idList.filter(id => !found.has(id))
    throw new Error(`unknown prompt id(s): ${missing.join(', ')}`)
  }

  return {
    ...fileConfig,
    configPath,
    manifestPath,
    prompts,
    threads: integerEnv('ACESTEP_COMPARE_THREADS', fileConfig.threads),
    warmups: integerEnv('ACESTEP_COMPARE_WARMUPS', fileConfig.warmups),
    runs: integerEnv('ACESTEP_COMPARE_RUNS', fileConfig.runs),
    cooldownMs: integerEnv('ACESTEP_COMPARE_COOLDOWN_MS', fileConfig.cooldownMs),
    order: process.env.ACESTEP_COMPARE_ORDER || fileConfig.order || 'alternate',
    backend: process.env.ACESTEP_COMPARE_BACKEND || overrides.backend || 'cpu',
    modelsDir: path.resolve(process.env.ACESTEP_COMPARE_MODELS_DIR || path.join(comparisonRoot(), 'models')),
    outDir: path.resolve(process.env.ACESTEP_COMPARE_OUT_DIR || path.join(comparisonRoot(), 'out')),
    qvacCli,
    aceLm,
    aceSynth,
    acestepDir,
    engineDir: engineDir(),
    repositoryDir: repositoryDir(),
    failOnGpuFallback: fileConfig.failOnGpuFallback !== false,
    clap: {
      model: firstSet(
        process.env.ACESTEP_CLAP_MODEL,
        fileConfig.clap && fileConfig.clap.model,
        'laion/larger_clap_music_and_speech'
      ),
      revision: firstSet(
        process.env.ACESTEP_CLAP_REVISION,
        fileConfig.clap && fileConfig.clap.revision
      ) || null,
      textPolicy: firstSet(
        process.env.ACESTEP_CLAP_TEXT_POLICY,
        fileConfig.clap && fileConfig.clap.textPolicy,
        'caption'
      ),
      device: firstSet(
        process.env.ACESTEP_CLAP_DEVICE,
        fileConfig.clap && fileConfig.clap.device,
        'cpu'
      ),
      python: firstSet(process.env.ACESTEP_CLAP_PYTHON, 'python3')
    }
  }
}

module.exports = {
  DEFAULT_CONFIG_PATH,
  DEFAULT_MANIFEST_PATH,
  firstExisting,
  loadHarnessConfig,
  readJson
}
