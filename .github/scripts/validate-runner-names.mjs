#!/usr/bin/env node
/**
 * Fail if wired CI workflows hardcode catalog runner targets or drift from the
 * generated reusable-runner-names workflow.
 *
 * Usage: node .github/scripts/validate-runner-names.mjs
 */
import {
  REUSABLE_WORKFLOW,
  assertReusableMatchesCatalog,
  findHardcodedLabelViolations,
  findMissingRunnerNamesNeeds,
  listAddonWorkflows,
  loadRunners,
  readRepoFile,
} from './lib/runner-names.mjs'

function main() {
  const runners = loadRunners()
  const errors = []

  try {
    assertReusableMatchesCatalog(runners, readRepoFile(REUSABLE_WORKFLOW))
  } catch (error) {
    errors.push(error.message)
  }

  for (const file of listAddonWorkflows()) {
    const source = readRepoFile(file)
    for (const finding of findHardcodedLabelViolations(file, source, runners)) {
      errors.push(`${finding.file}:${finding.line} hardcodes runner target ${finding.target}: ${finding.text}`)
    }
    for (const finding of findMissingRunnerNamesNeeds(file, source)) {
      errors.push(`${finding.file}: ${finding.message}`)
    }
  }

  if (errors.length > 0) {
    console.error(`validate-runner-names: ${errors.length} finding(s):`)
    for (const error of errors) console.error(`  ${error}`)
    process.exit(1)
  }

  console.log(`validate-runner-names: ok (${runners.length} targets, ${listAddonWorkflows().length} workflow(s))`)
}

main()
