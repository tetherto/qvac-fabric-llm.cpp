'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { findQvacCli } = require('../lib/config')
const { comparisonRoot, engineDir, repositoryDir, speechDir } = require('../lib/paths')

const CLI_NAME = 'music-cli'
const SPEECH_SUBDIR = path.join('engines', 'speech')

function makeFabricTree (t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'acestep-compare-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const speech = path.join(root, SPEECH_SUBDIR)
  fs.mkdirSync(speech, { recursive: true })
  return { repositoryDir: root, speechDir: speech }
}

function placeCli (directory) {
  fs.mkdirSync(directory, { recursive: true })
  const cli = path.join(directory, CLI_NAME)
  fs.writeFileSync(cli, '')
  return cli
}

test('path roots follow the tree the harness lives in', () => {
  assert.equal(path.relative(engineDir(), comparisonRoot()), path.join('benchmarks', 'comparison'))
  assert.equal(path.relative(speechDir(), engineDir()), 'audiogen')
  assert.equal(path.relative(repositoryDir(), speechDir()), SPEECH_SUBDIR)
  assert.ok(fs.existsSync(path.join(speechDir(), 'audiogen', 'CMakeLists.txt')))
  assert.ok(fs.existsSync(path.join(repositoryDir(), 'ggml', 'CMakeLists.txt')))
})

test('findQvacCli finds the fabric root build', t => {
  const roots = makeFabricTree(t)
  const cli = placeCli(path.join(roots.repositoryDir, 'build', 'bin'))
  assert.equal(findQvacCli(roots), cli)
})

test('findQvacCli finds the standalone speech build', t => {
  const roots = makeFabricTree(t)
  const cli = placeCli(path.join(roots.speechDir, 'build', 'audiogen'))
  assert.equal(findQvacCli(roots), cli)
})

test('findQvacCli finds an engine-local build', t => {
  const roots = makeFabricTree(t)
  const cli = placeCli(path.join(roots.speechDir, 'audiogen', 'build-cpu'))
  assert.equal(findQvacCli(roots), cli)
})

test('findQvacCli reports no binary when nothing is built', t => {
  assert.equal(findQvacCli(makeFabricTree(t)), undefined)
})
