import { readRepoFile } from './runner-names.mjs'

export const GGML_TRIGGER_PATH = 'ggml/**'
export const PATH_FILTERED_EVENTS = ['push', 'pull_request']
export const GGML_BUILD_WORKFLOWS = [
  '.github/workflows/speech-audiogen-ci.yml',
  '.github/workflows/speech-engines-cross-ci.yml',
  '.github/workflows/speech-parakeet-ci.yml',
  '.github/workflows/speech-tts-ci.yml',
  '.github/workflows/speech-whisper-ci.yml',
]

const EVENT_INDENT = '  '
const PATH_ENTRY = /^ {6}- '([^']+)'\s*$/

function isEventHeader(line) {
  return line.startsWith(EVENT_INDENT) && !line.startsWith(`${EVENT_INDENT} `) && line.trimEnd().endsWith(':')
}

function eventBlock(lines, event) {
  const start = lines.indexOf(`${EVENT_INDENT}${event}:`)
  if (start < 0) return []
  const rest = lines.slice(start + 1)
  const end = rest.findIndex((line) => isEventHeader(line) || /^\S/.test(line))
  return end < 0 ? rest : rest.slice(0, end)
}

export function triggerPaths(source, event) {
  return eventBlock(source.split('\n'), event)
    .map((line) => PATH_ENTRY.exec(line))
    .filter(Boolean)
    .map((match) => match[1])
}

export function findMissingTriggerPaths(workflows, requiredPath) {
  return workflows.flatMap((workflow) =>
    PATH_FILTERED_EVENTS
      .filter((event) => !triggerPaths(readRepoFile(workflow), event).includes(requiredPath))
      .map((event) => `${workflow}: ${event}`))
}
