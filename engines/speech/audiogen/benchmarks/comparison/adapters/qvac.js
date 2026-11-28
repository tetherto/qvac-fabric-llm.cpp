'use strict'

const fs = require('fs')
const path = require('path')
const crypto = require('crypto')
const { analyseWav, durationError } = require('../lib/audio')
const { assertBackend, detectQvacBackend } = require('../lib/backend')
const { runTimedProcess } = require('../lib/process')
const { parseQvacTimings } = require('../lib/timing')

function modelArgs (config) {
  return [
    '--dit', path.join(config.modelsDir, config.models.dit),
    '--lm', path.join(config.modelsDir, config.models.lm),
    '--text', path.join(config.modelsDir, config.models.text),
    '--vae', path.join(config.modelsDir, config.models.vae)
  ]
}

function buildQvacArgs (config, prompt, outputWav) {
  const args = [
    ...modelArgs(config),
    '--caption', prompt.caption,
    '--lyrics', prompt.lyrics,
    '--dur', String(prompt.duration),
    '--seed', String(prompt.seed),
    '--bpm', String(prompt.bpm),
    '--key', prompt.keyscale,
    '--tsig', prompt.timesignature,
    '--lang', prompt.vocal_language,
    '--steps', String(config.parameters.inference_steps),
    '--shift', String(config.parameters.shift),
    '--temp', String(config.parameters.lm_temperature),
    '--topp', String(config.parameters.lm_top_p),
    '--cfg', String(config.parameters.lm_cfg_scale),
    '--threads', String(config.threads),
    '--out', outputWav
  ]
  if (config.parameters.lm_top_k) args.push('--topk', String(config.parameters.lm_top_k))
  if (!config.parameters.dcw_enabled) args.push('--no-dcw')
  if (config.backend !== 'cpu') args.push('--gpu')
  return args
}

function runQvacRound (config, prompt, outputWav) {
  if (!fs.existsSync(config.qvacCli)) {
    throw new Error(`QVAC music-cli not found: ${config.qvacCli}`)
  }
  const env = { ...process.env }
  if (config.keepStages) env.ACESTEP_KEEP_STAGES = '1'
  const execution = runTimedProcess(
    config.qvacCli,
    buildQvacArgs(config, prompt, outputWav),
    { cwd: config.engineDir, env }
  )
  const timings = parseQvacTimings(execution.logText)
  const backend = config.backend === 'cpu' ? 'cpu' : detectQvacBackend(execution.logText)
  if (config.backend !== 'cpu') {
    assertBackend(backend, config.backend, { failOnFallback: config.failOnGpuFallback })
  }
  let audio = null
  let wavSha256 = null
  if (fs.existsSync(outputWav)) {
    const buffer = fs.readFileSync(outputWav)
    wavSha256 = crypto.createHash('sha256').update(buffer).digest('hex')
    audio = analyseWav(buffer)
  }
  const ok = execution.status === 0 && Boolean(audio)
  const generationMs = timings.generationMs
  const e2eMs = execution.wallMs
  return {
    engine: 'qvac',
    promptId: prompt.id,
    ok,
    status: execution.status,
    backend,
    e2eMs,
    generationMs,
    initMs: generationMs == null ? null : Math.max(0, e2eMs - generationMs),
    rtf: require('../lib/aggregate').rtf(generationMs || e2eMs, audio && audio.durationSeconds),
    peakRssBytes: execution.peakRssBytes,
    userSeconds: execution.userSeconds,
    sysSeconds: execution.sysSeconds,
    audio,
    durationErrorSeconds: durationError(prompt.duration, audio && audio.durationSeconds),
    wavSha256,
    nCodes: timings.nCodes,
    seed: timings.seed,
    stages: timings.stages,
    stdout: execution.stdout,
    stderr: execution.stderr,
    error: ok ? null : (execution.status !== 0 ? `music-cli exited ${execution.status}` : 'missing or invalid WAV')
  }
}

module.exports = {
  buildQvacArgs,
  runQvacRound
}
