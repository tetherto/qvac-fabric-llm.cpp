'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { readCmakeBuildValue } = require('../capture-environment')

test('readCmakeBuildValue reads CUDA build metadata', t => {
  const buildDir = fs.mkdtempSync(path.join(os.tmpdir(), 'acestep-env-'))
  t.after(() => fs.rmSync(buildDir, { recursive: true, force: true }))
  fs.writeFileSync(
    path.join(buildDir, 'CMakeCache.txt'),
    'CMAKE_CUDA_ARCHITECTURES:STRING=89-virtual\nCMAKE_CUDA_COMPILER:FILEPATH=/usr/bin/nvcc\n'
  )
  assert.equal(readCmakeBuildValue(buildDir, 'CMAKE_CUDA_ARCHITECTURES'), '89-virtual')
  assert.equal(readCmakeBuildValue(buildDir, 'CMAKE_CUDA_COMPILER'), '/usr/bin/nvcc')
  assert.equal(readCmakeBuildValue(buildDir, 'MISSING'), null)
})
