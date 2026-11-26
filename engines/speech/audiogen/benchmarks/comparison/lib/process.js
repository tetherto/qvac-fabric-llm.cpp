'use strict'

const { spawnSync } = require('child_process')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { normalisePeakRssBytes, parseTimeLMemory } = require('./timing')

function combineOutput (result) {
  return `${result.stdout || ''}\n${result.stderr || ''}`
}

function runTimedProcess (command, args, options = {}) {
  const started = process.hrtime.bigint()
  const platform = options.platform || process.platform
  const useTime = options.useTime !== false
  const timeBin = '/usr/bin/time'
  let wrappedCommand = command
  let wrappedArgs = args
  let timeFormat = null
  if (useTime && fs.existsSync(timeBin)) {
    if (platform === 'darwin') {
      wrappedCommand = timeBin
      wrappedArgs = ['-l', command, ...args]
    } else {
      wrappedCommand = timeBin
      wrappedArgs = ['-f', 'user %U\nsys %S\nmaxresident %M', command, ...args]
      timeFormat = 'gnu'
    }
  }
  const result = spawnSync(wrappedCommand, wrappedArgs, {
    encoding: 'utf8',
    cwd: options.cwd,
    env: options.env || process.env,
    timeout: options.timeoutMs,
    maxBuffer: options.maxBuffer || 32 * 1024 * 1024
  })
  const ended = process.hrtime.bigint()
  const wallMs = Number(ended - started) / 1e6
  const logText = combineOutput(result)
  const memory = parseTimeLMemory(logText)
  if (result.error) {
    throw result.error
  }
  return {
    command,
    args,
    wrappedCommand,
    status: result.status,
    signal: result.signal,
    stdout: result.stdout || '',
    stderr: result.stderr || '',
    logText,
    wallMs,
    peakRssBytes: normalisePeakRssBytes(memory.peakRssRaw, timeFormat === 'gnu' ? 'linux' : platform),
    userSeconds: memory.userSeconds,
    sysSeconds: memory.sysSeconds,
    hostname: os.hostname()
  }
}

function writeText (filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, text.endsWith('\n') ? text : `${text}\n`)
}

function writeJson (filePath, value) {
  writeText(filePath, JSON.stringify(value, null, 2))
}

function fileSha256 (filePath) {
  const crypto = require('crypto')
  const hash = crypto.createHash('sha256')
  const fd = fs.openSync(filePath, 'r')
  try {
    hashFileChunks(hash, fd)
  } finally {
    fs.closeSync(fd)
  }
  return hash.digest('hex')
}

function hashFileChunks (hash, fd) {
  const buffer = Buffer.alloc(1024 * 1024)
  let bytesRead = fs.readSync(fd, buffer, 0, buffer.length, null)
  while (bytesRead > 0) {
    hash.update(buffer.subarray(0, bytesRead))
    bytesRead = fs.readSync(fd, buffer, 0, buffer.length, null)
  }
}

module.exports = {
  combineOutput,
  fileSha256,
  hashFileChunks,
  runTimedProcess,
  writeJson,
  writeText
}
