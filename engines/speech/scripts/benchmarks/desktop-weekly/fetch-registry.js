'use strict'
const fs = require('fs')
const path = require('path')
const { QVACRegistryClient } = require(process.env.REGISTRY_CLIENT)

const REGISTRY_PREFIX = 'qvac_models_compiled/ggml'
const DOWNLOAD_TIMEOUT_MS = 4 * 60 * 60 * 1000
const PARTIAL_SUFFIX = '.part'

function targetPath (outDir, key) {
  return path.join(outDir, path.basename(key))
}

function partialPath (dest) {
  const shared = dest + PARTIAL_SUFFIX
  if (!fs.existsSync(shared)) return shared
  try {
    fs.rmSync(shared, { force: true })
    console.log(`discard stale ${shared}`)
    return shared
  } catch (err) {
    console.log(`cannot remove stale ${shared} (${err.code}); downloading beside it`)
    return `${shared}-${process.pid}`
  }
}

async function fetchKey (client, outDir, key) {
  const dest = targetPath(outDir, key)
  if (fs.existsSync(dest)) return console.log(`present ${dest}`)
  const partial = partialPath(dest)
  console.log(`fetch ${REGISTRY_PREFIX}/${key}`)
  await client.downloadModel(`${REGISTRY_PREFIX}/${key}`, 's3', { outputFile: partial, timeout: DOWNLOAD_TIMEOUT_MS })
  fs.renameSync(partial, dest)
  console.log(`done ${dest} ${fs.statSync(dest).size}`)
}

async function fetchKeys (client, outDir, keys) {
  let failures = 0
  for (const key of keys) {
    try {
      await fetchKey(client, outDir, key)
    } catch (err) {
      failures++
      console.error(`FAIL ${key}: ${err.message}`)
    }
  }
  return failures
}

async function main () {
  const outDir = process.argv[2]
  const keys = process.argv.slice(3)
  fs.mkdirSync(outDir, { recursive: true })
  const client = new QVACRegistryClient()
  await client.ready()
  const failures = await fetchKeys(client, outDir, keys)
  await client.close().catch(() => {})
  process.exit(failures ? 1 : 0)
}

main()
