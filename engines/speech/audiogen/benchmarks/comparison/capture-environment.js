#!/usr/bin/env node
'use strict'

const os = require('os')
const { spawnSync } = require('child_process')
const { writeJson } = require('./lib/process')
const { loadHarnessConfig } = require('./lib/config')
const path = require('path')
const fs = require('fs')

function runText (command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8' })
  if (result.status !== 0) return (result.stderr || result.stdout || '').trim()
  return (result.stdout || '').trim()
}

function readCmakeBuildValue (buildDir, key) {
  const cachePath = path.join(buildDir, 'CMakeCache.txt')
  if (!fs.existsSync(cachePath)) return null
  const prefix = `${key}:`
  const line = fs.readFileSync(cachePath, 'utf8')
    .split(/\r?\n/)
    .find(candidate => candidate.startsWith(prefix))
  if (!line) return null
  const separator = line.indexOf('=')
  return separator === -1 ? null : line.slice(separator + 1)
}

function buildDirectoryMetadata (buildDir) {
  return {
    directory: buildDir,
    cudaArchitectures: readCmakeBuildValue(buildDir, 'CMAKE_CUDA_ARCHITECTURES'),
    cudaCompiler: readCmakeBuildValue(buildDir, 'CMAKE_CUDA_COMPILER')
  }
}

function buildMetadata (executable) {
  return {
    executable,
    ...buildDirectoryMetadata(path.dirname(executable))
  }
}

function gitHead (directory) {
  return directory && fsExists(directory)
    ? runText('git', ['-C', directory, 'rev-parse', 'HEAD'])
    : null
}

function main () {
  const config = loadHarnessConfig()
  const qvacGgmlDir = process.env.ACESTEP_QVAC_GGML_DIR || null
  const qvacGgmlBuildDir = process.env.ACESTEP_QVAC_GGML_BUILD_DIR ||
    (qvacGgmlDir ? path.join(qvacGgmlDir, 'build') : null)
  const info = {
    generatedAt: new Date().toISOString(),
    hostname: os.hostname(),
    platform: process.platform,
    arch: process.arch,
    release: os.release(),
    cpus: os.cpus().map(cpu => cpu.model),
    cpuCount: os.cpus().length,
    totalMemBytes: os.totalmem(),
    node: process.version,
    cmake: runText('cmake', ['--version']),
    clang: runText('clang', ['--version']),
    cxx: runText('c++', ['--version']),
    uname: process.platform === 'linux' ? runText('uname', ['-a']) : null,
    nvidiaSmi: process.platform === 'linux'
      ? runText('nvidia-smi', [
          '--query-gpu=index,name,driver_version,memory.total',
          '--format=csv,noheader'
        ])
      : null,
    nvcc: process.platform === 'linux' ? runText('nvcc', ['--version']) : null,
    cudaEnvironment: {
      visibleDevices: process.env.CUDA_VISIBLE_DEVICES || null,
      cudaHome: process.env.CUDA_HOME || null,
      cudaCompiler: process.env.CUDACXX || null,
      cudaArchitectures: process.env.CMAKE_CUDA_ARCHITECTURES || null
    },
    builds: {
      qvac: buildMetadata(config.qvacCli),
      qvacGgml: qvacGgmlBuildDir ? buildDirectoryMetadata(qvacGgmlBuildDir) : null,
      acestep: buildMetadata(config.aceSynth)
    },
    sysctl: process.platform === 'darwin' ? {
      brand: runText('sysctl', ['-n', 'machdep.cpu.brand_string']),
      memsize: runText('sysctl', ['-n', 'hw.memsize']),
      machine: runText('sysctl', ['-n', 'hw.model'])
    } : null,
    swVers: process.platform === 'darwin' ? runText('sw_vers', []) : null,
    qvacHead: gitHead(config.repositoryDir),
    qvacGgmlHead: gitHead(qvacGgmlDir),
    acestepHead: gitHead(config.acestepDir),
    acestepGgmlHead: gitHead(path.join(config.acestepDir, 'ggml'))
  }
  const outPath = path.join(config.outDir, 'environment.json')
  writeJson(outPath, info)
  console.log(`Wrote ${outPath}`)
}

function fsExists (filePath) {
  return fs.existsSync(filePath)
}

if (require.main === module) {
  try {
    main()
  } catch (error) {
    console.error(`error: ${error.message}`)
    process.exitCode = 1
  }
}

module.exports = {
  buildDirectoryMetadata,
  buildMetadata,
  gitHead,
  readCmakeBuildValue,
  runText
}
