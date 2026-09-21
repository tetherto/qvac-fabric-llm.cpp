#!/usr/bin/env node
/**
 * Regenerate .github/workflows/reusable-runner-names.yml from .github/runners.yaml.
 *
 * After editing the catalog:
 *   node .github/scripts/sync-runner-names.mjs
 *   node --test .github/scripts/test/runner-names.test.mjs
 */
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { REUSABLE_WORKFLOW, RUNNERS_YAML, loadRunners, renderReusableWorkflow, repoRoot } from './lib/runner-names.mjs'

const runners = loadRunners()
writeFileSync(join(repoRoot, REUSABLE_WORKFLOW), renderReusableWorkflow(runners), 'utf8')
console.log(`wrote ${REUSABLE_WORKFLOW} (${runners.length} targets from ${RUNNERS_YAML})`)
