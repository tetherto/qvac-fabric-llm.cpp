'use strict'

const { clapPolicyVersion } = require('./clap')

function sortedNumbers (values) {
  return values.filter(value => typeof value === 'number' && !Number.isNaN(value)).sort((a, b) => a - b)
}

function median (values) {
  const sorted = sortedNumbers(values)
  if (!sorted.length) return null
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2
  return sorted[mid]
}

function mean (values) {
  const sorted = sortedNumbers(values)
  if (!sorted.length) return null
  return sorted.reduce((sum, value) => sum + value, 0) / sorted.length
}

function min (values) {
  const sorted = sortedNumbers(values)
  return sorted.length ? sorted[0] : null
}

function max (values) {
  const sorted = sortedNumbers(values)
  return sorted.length ? sorted[sorted.length - 1] : null
}

function stddev (values) {
  const sorted = sortedNumbers(values)
  if (sorted.length < 2) return null
  const average = mean(sorted)
  const variance = sorted.reduce((sum, value) => sum + (value - average) ** 2, 0) / (sorted.length - 1)
  return Math.sqrt(variance)
}

function summarise (values) {
  return {
    n: sortedNumbers(values).length,
    mean: mean(values),
    median: median(values),
    min: min(values),
    max: max(values),
    stddev: stddev(values)
  }
}

function rtf (generationMs, audioSeconds) {
  if (!generationMs || !audioSeconds) return null
  return (generationMs / 1000) / audioSeconds
}

function uniqueCount (values) {
  return new Set(values.filter(Boolean)).size
}

function validateCompatibleClapPolicies (rounds) {
  clapPolicyVersion(rounds)
}

function aggregateRounds (rounds) {
  validateCompatibleClapPolicies(rounds)
  const byEngine = {}
  for (const round of rounds) {
    const key = round.engine
    if (!byEngine[key]) byEngine[key] = []
    byEngine[key].push(round)
  }
  const engines = {}
  for (const [engine, engineRounds] of Object.entries(byEngine)) {
    const successful = engineRounds.filter(round => round.ok)
    const failed = engineRounds.filter(round => !round.ok)
    engines[engine] = {
      successCount: successful.length,
      failureCount: failed.length,
      generationMs: summarise(successful.map(round => round.generationMs)),
      e2eMs: summarise(successful.map(round => round.e2eMs)),
      initMs: summarise(successful.map(round => round.initMs)),
      warmupMs: summarise(engineRounds.filter(round => round.kind === 'warmup').map(round => round.e2eMs)),
      rtf: summarise(successful.map(round => round.rtf)),
      peakRssBytes: summarise(successful.map(round => round.peakRssBytes)),
      durationSeconds: summarise(successful.map(round => round.audio && round.audio.durationSeconds)),
      durationErrorSeconds: summarise(successful.map(round => round.durationErrorSeconds)),
      silenceRatio: summarise(successful.map(round => round.audio && round.audio.silenceRatio)),
      clippingRatio: summarise(successful.map(round => round.audio && round.audio.clippingRatio)),
      uniqueWavHashes: uniqueCount(successful.map(round => round.wavSha256)),
      uniqueCodeHashes: uniqueCount(successful.map(round => round.codesSha256)),
      clap: {
        ...summarise(successful.map(round => round.clap && round.clap.score).filter(Number.isFinite)),
        policyVersion: clapPolicyVersion(successful)
      },
      silentCount: successful.filter(round => round.audio && round.audio.silent).length
    }
  }
  return engines
}

function groupByPrompt (rounds) {
  validateCompatibleClapPolicies(rounds)
  const groups = {}
  for (const round of rounds) {
    const key = round.promptId
    if (!groups[key]) groups[key] = []
    groups[key].push(round)
  }
  const out = {}
  for (const [promptId, promptRounds] of Object.entries(groups)) {
    out[promptId] = aggregateRounds(promptRounds)
  }
  return out
}

module.exports = {
  aggregateRounds,
  groupByPrompt,
  max,
  mean,
  median,
  min,
  rtf,
  stddev,
  summarise
}
