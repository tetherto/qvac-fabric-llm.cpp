'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const crypto = require('crypto')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { fileSha256 } = require('../lib/process')

test('fileSha256 streams and matches a known digest', () => {
  const filePath = path.join(os.tmpdir(), `acestep-hash-${process.pid}.bin`)
  const payload = Buffer.alloc(2 * 1024 * 1024, 7)
  fs.writeFileSync(filePath, payload)
  try {
    const expected = crypto.createHash('sha256').update(payload).digest('hex')
    assert.equal(fileSha256(filePath), expected)
  } finally {
    fs.unlinkSync(filePath)
  }
})
