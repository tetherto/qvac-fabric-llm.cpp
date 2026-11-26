#!/usr/bin/env node
'use strict'

const fs = require('fs')
const path = require('path')
const { renderMarkdownReport } = require('./lib/report')

function main () {
  const jsonPath = process.argv[2]
  if (!jsonPath) throw new Error('usage: node generate-report.js <results.json>')
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'))
  const markdown = renderMarkdownReport(data)
  const outPath = process.argv[3] || jsonPath.replace(/\.json$/i, '.md')
  fs.writeFileSync(outPath, markdown.endsWith('\n') ? markdown : `${markdown}\n`)
  console.log(`Wrote ${outPath}`)
}

try {
  main()
} catch (error) {
  console.error(`error: ${error.message}`)
  process.exitCode = 1
}
