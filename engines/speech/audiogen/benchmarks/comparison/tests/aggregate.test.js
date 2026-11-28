'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { aggregateRounds, median, rtf } = require('../lib/aggregate')

test('median ignores nulls and uses the middle value', () => {
  assert.equal(median([3, 1, 2, null]), 2)
  assert.equal(median([1, 2, 3, 4]), 2.5)
})

test('rtf is generation seconds over audio seconds', () => {
  assert.equal(rtf(2000, 4), 0.5)
})

test('aggregateRounds retains failures and unique hashes', () => {
  const stats = aggregateRounds([
    { engine: 'qvac', ok: true, generationMs: 10, e2eMs: 12, rtf: 0.5, peakRssBytes: 100, wavSha256: 'a', audio: { durationSeconds: 8, silenceRatio: 0.1, clippingRatio: 0, silent: false } },
    { engine: 'qvac', ok: true, generationMs: 14, e2eMs: 16, rtf: 0.6, peakRssBytes: 110, wavSha256: 'a', audio: { durationSeconds: 8.1, silenceRatio: 0.1, clippingRatio: 0, silent: false } },
    { engine: 'qvac', ok: false, error: 'boom' }
  ]).qvac
  assert.equal(stats.successCount, 2)
  assert.equal(stats.failureCount, 1)
  assert.equal(stats.generationMs.median, 12)
  assert.equal(stats.uniqueWavHashes, 1)
})
