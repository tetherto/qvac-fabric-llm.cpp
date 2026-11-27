import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  GGML_BUILD_WORKFLOWS, GGML_TRIGGER_PATH, findMissingTriggerPaths, triggerPaths,
} from '../lib/workflow-triggers.mjs'

const SAMPLE = [
  'on:',
  '  push:',
  '    branches: [master]',
  '    paths:',
  "      - 'engines/speech/tts/**'",
  '  pull_request:',
  '    paths:',
  "      - 'engines/speech/tts/**'",
  "      - 'ggml/**'",
  '  workflow_dispatch:',
  'jobs:',
  '  build:',
  '    steps:',
  "      - 'not/a/trigger'",
].join('\n')

test('trigger paths are read per event', () => {
  assert.deepEqual(triggerPaths(SAMPLE, 'push'), ['engines/speech/tts/**'])
  assert.deepEqual(triggerPaths(SAMPLE, 'pull_request'), ['engines/speech/tts/**', 'ggml/**'])
  assert.deepEqual(triggerPaths(SAMPLE, 'schedule'), [])
})

test('every workflow that builds the in-tree ggml runs on ggml changes', () => {
  assert.deepEqual(findMissingTriggerPaths(GGML_BUILD_WORKFLOWS, GGML_TRIGGER_PATH), [])
})
