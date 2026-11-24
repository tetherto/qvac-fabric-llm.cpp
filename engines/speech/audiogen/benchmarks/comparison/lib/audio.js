'use strict'

function readPcm16Wav (buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 44) {
    throw new Error('WAV buffer is too small')
  }
  const riff = buffer.toString('ascii', 0, 4)
  const wave = buffer.toString('ascii', 8, 12)
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error('not a RIFF/WAVE file')
  }
  let offset = 12
  let audioFormat = null
  let channels = null
  let sampleRate = null
  let bitsPerSample = null
  let dataOffset = null
  let dataSize = null
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4)
    const size = buffer.readUInt32LE(offset + 4)
    const next = offset + 8 + size
    if (id === 'fmt ') {
      audioFormat = buffer.readUInt16LE(offset + 8)
      channels = buffer.readUInt16LE(offset + 10)
      sampleRate = buffer.readUInt32LE(offset + 12)
      bitsPerSample = buffer.readUInt16LE(offset + 22)
    } else if (id === 'data') {
      dataOffset = offset + 8
      dataSize = size
      break
    }
    offset = next + (size % 2)
  }
  if (dataOffset == null) throw new Error('WAV data chunk missing')
  if (audioFormat !== 1) throw new Error(`unsupported WAV format ${audioFormat}`)
  if (bitsPerSample !== 16) throw new Error(`unsupported bit depth ${bitsPerSample}`)
  const sampleCount = Math.floor(dataSize / 2)
  const samples = new Int16Array(sampleCount)
  for (let i = 0; i < sampleCount; i++) {
    samples[i] = buffer.readInt16LE(dataOffset + i * 2)
  }
  const frames = channels ? Math.floor(sampleCount / channels) : 0
  return { channels, sampleRate, bitsPerSample, frames, samples }
}

function analyseWav (buffer) {
  const wav = readPcm16Wav(buffer)
  const { samples, frames, channels, sampleRate } = wav
  let peak = 0
  let sumSquares = 0
  let silent = 0
  let clipped = 0
  const silenceThreshold = 512
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i])
    if (abs > peak) peak = abs
    sumSquares += samples[i] * samples[i]
    if (abs <= silenceThreshold) silent++
    if (samples[i] <= -32768 || samples[i] >= 32767) clipped++
  }
  const durationSeconds = sampleRate ? frames / sampleRate : 0
  const rms = samples.length ? Math.sqrt(sumSquares / samples.length) / 32768 : 0
  const peakNorm = peak / 32768
  return {
    sampleRate,
    channels,
    frames,
    durationSeconds,
    fileBytes: buffer.length,
    peak,
    peakNorm,
    rms,
    silenceRatio: samples.length ? silent / samples.length : 1,
    clippingRatio: samples.length ? clipped / samples.length : 0,
    silent: peak <= silenceThreshold
  }
}

function durationError (requestedSeconds, actualSeconds) {
  if (requestedSeconds == null || actualSeconds == null) return null
  return actualSeconds - requestedSeconds
}

module.exports = {
  analyseWav,
  durationError,
  readPcm16Wav
}
