# CI build selection and caching

The cross-platform workflow selects engines from the complete PR diff or all
commits in a push. Changes confined to `parakeet/`, `tts/`, or
`audiogen/` run that engine's Windows, Android, and iOS jobs. Changes to
shared files, including the Whisper subtree, root CMake configuration, scripts,
shared engine helpers, and runner configuration, run all ten jobs. Manual runs
and unavailable comparison commits also run the full matrix. Moves select both
the source and destination engines. Platform job names remain stable.

Selection lives in `engines/speech/ci/scripts/lib/engine-matrix.mjs`, alongside each
platform's CMake flags. Keep the workflow path filters current when adding shared
build dependencies. Every workflow that builds the in-tree `ggml/` also runs on
`ggml/**` changes, and a ggml change selects the full matrix;
`engines/speech/ci/scripts/lib/workflow-triggers.mjs` lists those workflows and its
test fails when one drops the filter. Run the tests with:

```sh
node --test engines/speech/ci/scripts/test/*.test.mjs
```

Windows builds use Ninja with the x64 MSVC environment and four compile workers.
Ccache stores compiler outputs separately for each engine; cache keys include the
ggml pin and commit, with fallbacks to earlier builds of the same engine. Ccache
also checks compiler contents, source files, headers, and compiler flags before
reusing an object. Each job prints cache statistics. The installed ggml cache is
separate and remains shared across engines. Cold builds still compile every
target, and Windows jobs continue to run their full non-GPU test suites.

The Whisper VAD contract tests use one CPU thread for the complete `jfk.wav`
fixture to avoid worker synchronization on each small inference graph. All
full-input probability, segment, reset, and CPU/GPU comparisons remain enabled.
The streaming test additionally exercises the default thread count on a short
input with a partial final window. Production VAD defaults are unchanged.

Run these tests from a configured Whisper test build with:

```sh
ctest --test-dir build -R '^test-vad(-streaming|-gpu)?$' --output-on-failure
```

Measure Windows build/cache timings and hosted macOS VAD timings on subsequent
CI runs; local hardware timings do not establish hosted-runner speedups.
