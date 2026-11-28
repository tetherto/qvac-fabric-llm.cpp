import test from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { SPEECH_ROOT, buildMatrices, changedPaths, comparisonBase, selectEngines } from '../lib/engine-matrix.mjs'

const allEngines = ['parakeet', 'tts', 'audiogen', 'whisper']

test('engine changes select only that engine across supported platforms', () => {
  for (const engine of allEngines.slice(0, 3)) {
    const selected = selectEngines([`${SPEECH_ROOT}${engine}/src/model.cpp`])
    assert.deepEqual(selected, [engine])
    for (const matrix of Object.values(buildMatrices(selected))) {
      assert.deepEqual(matrix.include.map((entry) => entry.engine), [engine])
      assert.equal(matrix.include[0].src, engine)
    }
  }
})

test('multiple engines and deleted paths retain all affected engines', () => {
  assert.deepEqual(selectEngines([
    'engines/speech/tts/src/deleted.cpp', 'engines/speech/parakeet/src/new.cpp', 'engines/speech/tts/README.md',
  ]), ['parakeet', 'tts'])
})

test('shared dependencies and CI configuration run the full matrix', () => {
  for (const path of [
    'engines/speech/CMakeLists.txt', 'engines/speech/third_party/whisper.cpp/include/whisper.h',
    'engines/speech/test/moss_gpu_arm.h', 'engines/speech/whisper-gpu-picker.h',
    'engines/speech/cmake/toolchain.cmake', 'engines/speech/scripts/benchmarks/test_audio8_streaming.py',
    '.github/workflows/speech-engines-cross-ci.yml', 'engines/speech/ci/scripts/select-engine-matrix.mjs',
    'engines/speech/ci/runners.yaml', '.github/workflows/speech-reusable-runner-names.yml',
    'ggml/src/ggml.c', 'CMakeLists.txt', 'tts/src/model.cpp',
  ]) {
    assert.deepEqual(selectEngines(['engines/speech/tts/src/model.cpp', path]), allEngines, path)
  }
})

test('manual runs and unavailable comparisons run all ten lanes', () => {
  const matrices = buildMatrices(selectEngines(null))
  assert.equal(matrices.windows.include.length, 3)
  assert.equal(matrices.android.include.length, 4)
  assert.equal(matrices.ios.include.length, 3)
  assert.equal(matrices.android.include.at(-1).src, '.')
  assert.match(matrices.android.include.at(-1).cmake_flags, /-DSPEECH_BUILD_WHISPER=ON/)
})

test('an empty diff produces no platform jobs', () => {
  for (const matrix of Object.values(buildMatrices(selectEngines([])))) {
    assert.deepEqual(matrix.include, [])
  }
})

test('comparison bases support PRs, multi-commit pushes and new branches', () => {
  assert.equal(comparisonBase('pull_request', { pull_request: { base: { sha: 'base' } } }), 'base')
  assert.equal(comparisonBase('push', { before: 'previous-push' }), 'previous-push')
  assert.equal(comparisonBase('push', { before: '0'.repeat(40) }), null)
  assert.equal(comparisonBase('workflow_dispatch', {}), null)
  assert.equal(comparisonBase('pull_request', {}), null)
})

test('diff includes both sides of moves and preserves spaces and newlines in paths', () => {
  const paths = ['engines/speech/tts/old name.cpp', 'engines/speech/parakeet/new\nname.cpp']
  const commands = []
  const run = (_command, args) => {
    commands.push(args)
    return args[0] === 'diff' ? `${paths.join('\0')}\0` : ''
  }
  assert.deepEqual(changedPaths('base', run), paths)
  assert.ok(commands[1].includes('--no-renames'))
  assert.deepEqual(selectEngines(paths), ['parakeet', 'tts'])
})

test('unavailable bases and failed diffs fall back to the full matrix', (t) => {
  t.mock.method(console, 'warn', () => {})
  assert.equal(changedPaths(null, () => assert.fail('should not invoke git')), null)
  for (const failedCommand of ['fetch', 'diff']) {
    const paths = changedPaths('base', (_command, args) => {
      if (args[0] === failedCommand) throw new Error('comparison unavailable')
      return ''
    })
    assert.deepEqual(selectEngines(paths), allEngines)
  }
})

test('the workflow entry point writes usable matrices and platform gates', () => {
  const directory = mkdtempSync(join(tmpdir(), 'engine-matrix-'))
  try {
    const eventPath = join(directory, 'event.json')
    const outputPath = join(directory, 'output')
    writeFileSync(eventPath, '{}')
    execFileSync(process.execPath, [fileURLToPath(new URL('../select-engine-matrix.mjs', import.meta.url))], {
      env: { ...process.env, GITHUB_EVENT_NAME: 'workflow_dispatch',
        GITHUB_EVENT_PATH: eventPath, GITHUB_OUTPUT: outputPath },
    })
    const outputs = Object.fromEntries(readFileSync(outputPath, 'utf8').trim().split('\n').map((line) => {
      const separator = line.indexOf('=')
      return [line.slice(0, separator), line.slice(separator + 1)]
    }))
    for (const [platform, matrix] of Object.entries(buildMatrices(allEngines))) {
      assert.deepEqual(JSON.parse(outputs[platform]), matrix)
      assert.equal(outputs[`${platform}_enabled`], 'true')
    }
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
