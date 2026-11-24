'use strict'

const crypto = require('crypto')
const fs = require('fs')
const path = require('path')
const { analyseWav, durationError } = require('../lib/audio')
const { assertBackend, detectAcestepBackend } = require('../lib/backend')
const { runTimedProcess, writeJson } = require('../lib/process')
const { parseAcestepTimings } = require('../lib/timing')
const { rtf } = require('../lib/aggregate')

function buildRequest (config, prompt) {
  const p = config.parameters
  return {
    caption: prompt.caption,
    lyrics: prompt.lyrics,
    bpm: prompt.bpm,
    duration: prompt.duration,
    keyscale: prompt.keyscale,
    timesignature: prompt.timesignature,
    vocal_language: prompt.vocal_language,
    seed: prompt.seed,
    lm_seed: prompt.seed,
    lm_mode: 'generate',
    lm_temperature: p.lm_temperature,
    lm_cfg_scale: p.lm_cfg_scale,
    lm_top_p: p.lm_top_p,
    lm_top_k: p.lm_top_k,
    use_cot_caption: p.use_cot_caption,
    inference_steps: p.inference_steps,
    guidance_scale: p.guidance_scale,
    shift: p.shift,
    dcw_scaler: p.dcw_enabled ? p.dcw_scaler : 0.0,
    dcw_high_scaler: p.dcw_enabled ? p.dcw_high_scaler : 0.0,
    dcw_mode: p.dcw_mode,
    solver: p.solver,
    task_type: p.task_type,
    output_format: p.output_format,
    peak_clip: p.peak_clip,
    lm_model: config.models.lm,
    synth_model: config.models.dit,
    vae: config.models.vae
  }
}

function findOutputWav (workDir) {
  const files = fs.readdirSync(workDir)
    .filter(name => name.toLowerCase().endsWith('.wav'))
    .map(name => path.join(workDir, name))
  if (!files.length) return null
  files.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)
  return files[0]
}

function prepareWorkDir (outputWav) {
  const workDir = `${outputWav}.work`
  fs.rmSync(outputWav, { force: true })
  fs.rmSync(workDir, { recursive: true, force: true })
  fs.mkdirSync(workDir, { recursive: true })
  return workDir
}

function runAcestepRound (config, prompt, outputWav) {
  if (!fs.existsSync(config.aceLm) || !fs.existsSync(config.aceSynth)) {
    throw new Error(`acestep.cpp CLIs not found: lm=${config.aceLm} synth=${config.aceSynth}`)
  }
  const workDir = prepareWorkDir(outputWav)
  const requestPath = path.join(workDir, 'request.json')
  writeJson(requestPath, buildRequest(config, prompt))
  const env = { ...process.env }
  if (config.backend === 'cpu') {
    env.GGML_BACKEND = 'cpu'
    env.GGML_METAL = '0'
  }

  const lm = runTimedProcess(
    config.aceLm,
    ['--models', config.modelsDir, '--request', requestPath],
    { cwd: workDir, env }
  )
  const enriched = path.join(workDir, 'request0.json')
  const synth = runTimedProcess(
    config.aceSynth,
    ['--models', config.modelsDir, '--request', fs.existsSync(enriched) ? enriched : requestPath],
    { cwd: workDir, env }
  )

  const logText = `${lm.logText}\n${synth.logText}`
  const backend = config.backend === 'cpu' ? 'cpu' : detectAcestepBackend(logText)
  if (config.backend !== 'cpu') {
    assertBackend(backend, config.backend, { failOnFallback: config.failOnGpuFallback })
  }

  const produced = findOutputWav(workDir)
  if (produced && path.resolve(produced) !== path.resolve(outputWav)) {
    fs.copyFileSync(produced, outputWav)
  }

  let audio = null
  let wavSha256 = null
  if (fs.existsSync(outputWav)) {
    const buffer = fs.readFileSync(outputWav)
    wavSha256 = crypto.createHash('sha256').update(buffer).digest('hex')
    audio = analyseWav(buffer)
  }

  const lmTimings = parseAcestepTimings(lm.logText)
  const synthTimings = parseAcestepTimings(synth.logText)
  const e2eMs = lm.wallMs + synth.wallMs
  const generationMs = (lmTimings.generationMs || lm.wallMs) + (synthTimings.generationMs || synth.wallMs)
  const initMs = (lmTimings.loadMs || 0) + (synthTimings.loadMs || 0)
  const ok = lm.status === 0 && synth.status === 0 && Boolean(audio)

  return {
    engine: 'acestep.cpp',
    promptId: prompt.id,
    ok,
    status: synth.status || lm.status,
    backend,
    e2eMs,
    generationMs,
    initMs: initMs || null,
    lmWallMs: lm.wallMs,
    synthWallMs: synth.wallMs,
    rtf: rtf(generationMs, audio && audio.durationSeconds),
    peakRssBytes: Math.max(lm.peakRssBytes || 0, synth.peakRssBytes || 0) || null,
    userSeconds: (lm.userSeconds || 0) + (synth.userSeconds || 0),
    sysSeconds: (lm.sysSeconds || 0) + (synth.sysSeconds || 0),
    audio,
    durationErrorSeconds: durationError(prompt.duration, audio && audio.durationSeconds),
    wavSha256,
    stdout: `${lm.stdout}\n${synth.stdout}`,
    stderr: `${lm.stderr}\n${synth.stderr}`,
    error: ok
      ? null
      : `ace-lm exited ${lm.status}; ace-synth exited ${synth.status}${audio ? '' : '; missing WAV'}`
  }
}

module.exports = {
  buildRequest,
  prepareWorkDir,
  runAcestepRound
}
