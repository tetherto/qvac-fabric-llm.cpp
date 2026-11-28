'use strict'

// Must match scripts/benchmarks/music_alignment.py (checked by clap.test.js).
const CLAP_POLICY_VERSION = 'clap-music-v2'

function buildClapText (prompt, textPolicy) {
  if (!prompt || typeof prompt.caption !== 'string') {
    throw new Error('CLAP prompt must include a caption string')
  }
  if (textPolicy === 'caption+lyrics') {
    const lyrics = prompt.lyrics == null ? '' : String(prompt.lyrics)
    return `${prompt.caption}\n${lyrics}`
  }
  if (textPolicy && textPolicy !== 'caption') {
    throw new Error(`unknown CLAP text policy: ${textPolicy}`)
  }
  return prompt.caption
}

function parseClapBatchOutput (stdout) {
  const text = String(stdout || '').trim()
  if (!text) throw new Error('CLAP scorer printed no JSON')
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new Error(`CLAP scorer stdout is not JSON: ${error.message}`)
  }
  if (!parsed || parsed.ok !== true || !Array.isArray(parsed.scores)) {
    const detail = parsed && parsed.error ? parsed.error : 'missing scores array'
    throw new Error(`CLAP scorer failed: ${detail}`)
  }
  return parsed
}

function shouldScoreRound (record, options) {
  const timedOnly = options.timedOnly !== false
  if (!record) return false
  if (timedOnly && record.kind !== 'timed') return false
  if (!record.wavPath) return false
  if (!options.force && record.clap && record.clap.ok !== false &&
      Number.isFinite(record.clap.score) && record.clap.policyVersion === CLAP_POLICY_VERSION) return false
  return true
}

function clapPolicyVersion (rounds) {
  const versions = new Set(rounds
    .filter(round => round.ok && round.clap && Number.isFinite(round.clap.score))
    .map(round => round.clap.policyVersion || null))
  if (versions.size > 1) {
    const labels = [...versions].map(version => version || 'legacy/unversioned')
    throw new Error(`mixed CLAP policies: ${labels.join(', ')}. Run node score-clap.js --force to rescore saved WAVs before reporting.`)
  }
  return versions.size ? [...versions][0] : null
}

function mergeClapScore (record, scoreItem, meta) {
  const clap = {
    ok: Boolean(scoreItem && scoreItem.ok),
    score: scoreItem && typeof scoreItem.score === 'number' ? scoreItem.score : null,
    error: scoreItem && scoreItem.error ? scoreItem.error : null,
    elapsedMs: scoreItem && scoreItem.elapsedMs != null ? scoreItem.elapsedMs : null,
    model: meta.model,
    revision: meta.revision || null,
    samplingRate: meta.samplingRate || null,
    policyVersion: meta.policyVersion || null,
    device: meta.device,
    textPolicy: meta.textPolicy
  }
  return { ...record, clap }
}

function lookupPrompt (prompts, promptId) {
  const prompt = prompts.find(entry => entry.id === promptId)
  if (!prompt) throw new Error(`CLAP prompt id not in manifest: ${promptId}`)
  return prompt
}

module.exports = {
  CLAP_POLICY_VERSION,
  buildClapText,
  clapPolicyVersion,
  lookupPrompt,
  mergeClapScore,
  parseClapBatchOutput,
  shouldScoreRound
}
