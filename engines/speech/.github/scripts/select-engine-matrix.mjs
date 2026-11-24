import { appendFileSync, readFileSync } from 'node:fs'
import { buildMatrices, changedPaths, comparisonBase, selectEngines } from './lib/engine-matrix.mjs'

const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'))
const base = comparisonBase(process.env.GITHUB_EVENT_NAME, event)
const matrices = buildMatrices(selectEngines(changedPaths(base)))
for (const [platform, matrix] of Object.entries(matrices)) {
  appendFileSync(process.env.GITHUB_OUTPUT, `${platform}=${JSON.stringify(matrix)}\n`)
  appendFileSync(process.env.GITHUB_OUTPUT, `${platform}_enabled=${matrix.include.length > 0}\n`)
  console.log(`${platform}: ${matrix.include.map(({ engine }) => engine).join(', ') || 'skipped'}`)
}
