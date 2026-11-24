'use strict'

function parseQvacTimings (logText) {
  const text = String(logText || '')
  const totalMatch = text.match(/\[acestep-timing\] per-stage wall clock \(total\s+([\d.]+)\s+ms\)/i)
  const stages = {}
  const stageRe = /^\[acestep-timing\][ \t]{2,}(\S+)\s+([\d.]+)\s+ms/gm
  let match
  while ((match = stageRe.exec(text)) !== null) {
    stages[match[1]] = Number.parseFloat(match[2])
  }
  const codesMatch = text.match(/generated\s+(\d+)\s+codes, seed=(-?\d+),\s+(\d+)\s+frames\s+\(([\d.]+)s\)/i)
  return {
    generationMs: totalMatch ? Number.parseFloat(totalMatch[1]) : null,
    stages,
    nCodes: codesMatch ? Number.parseInt(codesMatch[1], 10) : null,
    seed: codesMatch ? Number.parseInt(codesMatch[2], 10) : null,
    frames: codesMatch ? Number.parseInt(codesMatch[3], 10) : null,
    audioSecondsLogged: codesMatch ? Number.parseFloat(codesMatch[4]) : null
  }
}

function parseAcestepTimings (logText) {
  const text = String(logText || '')
  const loadMatches = [...text.matchAll(/loaded\s+\S+\s+in\s+([\d.]+)\s*ms/gi)].map(match => Number.parseFloat(match[1]))
  const loadMs = loadMatches.length ? loadMatches.reduce((sum, value) => sum + value, 0) : null
  const genMatch = text.match(/generated\s+in\s+([\d.]+)\s*ms/i) || text.match(/synth(?:esis)?\s+([\d.]+)\s*ms/i)
  return {
    loadMs,
    generationMs: genMatch ? Number.parseFloat(genMatch[1]) : null
  }
}

function parseTimeLMemory (logText) {
  const text = String(logText || '')
  const rssMatch = text.match(/([\d.]+)\s+maximum resident set size/i) ||
    text.match(/maximum resident set size\s*[:=]?\s*([\d.]+)/i) ||
    text.match(/maxresident\s*=\s*([\d.]+)/i)
  const userMatch = text.match(/user\s+([\d.]+)/i)
  const sysMatch = text.match(/sys\s+([\d.]+)/i)
  const rss = rssMatch ? Number.parseFloat(rssMatch[1]) : null
  // macOS `time -l` reports bytes; Linux `-v` often reports kilobytes. Callers
  // pass platform so we can normalise to bytes.
  return {
    peakRssRaw: rss,
    userSeconds: userMatch ? Number.parseFloat(userMatch[1]) : null,
    sysSeconds: sysMatch ? Number.parseFloat(sysMatch[1]) : null
  }
}

function normalisePeakRssBytes (raw, platform) {
  if (raw == null || Number.isNaN(raw)) return null
  if (platform === 'darwin') return raw
  return raw * 1024
}

module.exports = {
  normalisePeakRssBytes,
  parseAcestepTimings,
  parseQvacTimings,
  parseTimeLMemory
}
