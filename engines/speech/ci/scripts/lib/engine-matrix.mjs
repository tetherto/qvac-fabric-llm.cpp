import { execFileSync } from 'node:child_process'

const ENGINE_FLAGS = {
  parakeet: {
    windows: '-DPARAKEET_USE_SYSTEM_GGML=ON -DPARAKEET_BUILD_TESTS=ON -DPARAKEET_BUILD_EXAMPLES=OFF',
    android: '-DPARAKEET_USE_SYSTEM_GGML=ON -DPARAKEET_BUILD_TESTS=OFF -DPARAKEET_BUILD_EXAMPLES=OFF',
    ios: '-DPARAKEET_USE_SYSTEM_GGML=ON -DPARAKEET_BUILD_TESTS=OFF -DPARAKEET_BUILD_EXAMPLES=OFF',
  },
  tts: {
    windows: '-DTTS_CPP_BUILD_TESTS=ON',
    android: '-DTTS_CPP_BUILD_TESTS=OFF',
    ios: '-DTTS_CPP_BUILD_TESTS=OFF -DTTS_CPP_COREML=ON',
  },
  audiogen: {
    windows: '-DAUDIOGEN_BUILD_TESTS=ON -DAUDIOGEN_BUILD_EXECUTABLES=OFF',
    android: '-DAUDIOGEN_BUILD_TESTS=OFF -DAUDIOGEN_BUILD_EXECUTABLES=OFF',
    ios: '-DAUDIOGEN_BUILD_TESTS=OFF -DAUDIOGEN_BUILD_EXECUTABLES=OFF -DAUDIOGEN_COREML=ON',
  },
  whisper: {
    android: '-DSPEECH_BUILD_WHISPER=ON -DSPEECH_BUILD_PARAKEET=OFF -DSPEECH_BUILD_TTS=OFF -DSPEECH_BUILD_AUDIOGEN=OFF -DSPEECH_BUILD_EXECUTABLES=ON -DSPEECH_BUILD_TESTS=OFF',
  },
}

const ALL_ENGINES = Object.keys(ENGINE_FLAGS)
export const SPEECH_ROOT = 'engines/speech/'

export function selectEngines(paths) {
  // A missing comparison or a shared dependency change requires every lane.
  if (paths === null) return ALL_ENGINES
  const engines = new Set()
  for (const path of paths) {
    const engine = ALL_ENGINES.find((name) => path.startsWith(`${SPEECH_ROOT}${name}/`))
    if (!engine) return ALL_ENGINES
    engines.add(engine)
  }
  return ALL_ENGINES.filter((engine) => engines.has(engine))
}

export function buildMatrices(engines) {
  return Object.fromEntries(['windows', 'android', 'ios'].map((platform) => [
    platform,
    { include: engines.filter((engine) => ENGINE_FLAGS[engine][platform]).map((engine) => ({
      engine,
      src: engine === 'whisper' ? '.' : engine,
      cmake_flags: ENGINE_FLAGS[engine][platform],
    })) },
  ]))
}

export function comparisonBase(eventName, event) {
  if (eventName === 'pull_request') return event.pull_request?.base?.sha ?? null
  if (eventName === 'push' && event.before && !/^0+$/.test(event.before)) return event.before
  return null
}


export function changedPaths(base, run = execFileSync) {
  if (!base) return null
  try {
    run('git', ['fetch', '--no-tags', '--depth=1', 'origin', base], { stdio: 'pipe' })
    // HEAD is the PR merge commit or the pushed commit from actions/checkout.
    // Disable rename detection so moves select both the old and new engines.
    return run('git', ['diff', '--no-renames', '--name-only', '-z', base, 'HEAD'], {
      encoding: 'utf8',
    }).split('\0').filter(Boolean)
  } catch {
    console.warn('Unable to compare changes; running all engines.')
    return null
  }
}
