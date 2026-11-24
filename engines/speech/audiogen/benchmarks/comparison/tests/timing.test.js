'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { parseQvacTimings, parseTimeLMemory, normalisePeakRssBytes } = require('../lib/timing')

test('parseQvacTimings extracts stage totals', () => {
  const log = `[acestep-timing] per-stage wall clock (total 1234 ms)
[acestep-timing]   lm           800.0 ms   64.8%
[acestep-timing]   dit          200.5 ms   16.2%
[music-cli] generated 40 codes, seed=42, 384000 frames (8.00s)
`
  const parsed = parseQvacTimings(log)
  assert.equal(parsed.generationMs, 1234)
  assert.equal(parsed.stages.lm, 800)
  assert.equal(parsed.stages['(total'], undefined)
  assert.equal(parsed.nCodes, 40)
  assert.equal(parsed.audioSecondsLogged, 8)
})

test('parseTimeLMemory reads Darwin time -l output', () => {
  const parsed = parseTimeLMemory('         123456789  maximum resident set size')
  assert.equal(parsed.peakRssRaw, 123456789)
  assert.equal(normalisePeakRssBytes(parsed.peakRssRaw, 'darwin'), 123456789)
})
