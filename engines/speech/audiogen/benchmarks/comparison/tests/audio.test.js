'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { analyseWav, durationError } = require('../lib/audio')

function pcm16Wav (frames, channels, sampleRate, sampleValue) {
  const dataSize = frames * channels * 2
  const buffer = Buffer.alloc(44 + dataSize)
  buffer.write('RIFF', 0)
  buffer.writeUInt32LE(36 + dataSize, 4)
  buffer.write('WAVE', 8)
  buffer.write('fmt ', 12)
  buffer.writeUInt32LE(16, 16)
  buffer.writeUInt16LE(1, 20)
  buffer.writeUInt16LE(channels, 22)
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * channels * 2, 28)
  buffer.writeUInt16LE(channels * 2, 32)
  buffer.writeUInt16LE(16, 34)
  buffer.write('data', 36)
  buffer.writeUInt32LE(dataSize, 40)
  for (let i = 0; i < frames * channels; i++) buffer.writeInt16LE(sampleValue, 44 + i * 2)
  return buffer
}

test('analyseWav reports stereo 48 kHz duration and silence', () => {
  const silent = analyseWav(pcm16Wav(48000, 2, 48000, 0))
  assert.equal(silent.sampleRate, 48000)
  assert.equal(silent.channels, 2)
  assert.equal(silent.durationSeconds, 1)
  assert.equal(silent.silent, true)
  const loud = analyseWav(pcm16Wav(24000, 2, 48000, 30000))
  assert.equal(loud.silent, false)
  assert.ok(loud.peakNorm > 0.9)
})

test('durationError is actual minus requested', () => {
  assert.equal(durationError(8, 8.5), 0.5)
})

test('analyseWav rejects non-PCM files', () => {
  const bogus = Buffer.alloc(64, 0)
  bogus.write('XXXX', 0)
  assert.throws(() => analyseWav(bogus), /RIFF/)
})
