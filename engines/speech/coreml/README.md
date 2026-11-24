# Shared Core ML runtime

`coreml` holds the Core ML code that the tts, parakeet and audiogen
engines share:

- the load scheduler and model manager (`src/scheduler.cpp`,
  `src/manager.cpp`), compiled into each engine library under the engine's
  own namespace by `qvac_coreml_add_to()` in `qvac_coreml.cmake`;
- the portable weight generator: the recipe parser and interpreter
  (`src/recipe.cpp`, `src/interp.cpp`), the MILBlob v2 writer
  (`src/milblob.cpp`), the GGUF tensor source (`src/gguf_source.cpp`), the
  source digest (`src/digest.cpp`) and the on-disk artifact cache
  (`src/cache.cpp`, `src/failed.cpp`). Given a template directory and a GGUF,
  it writes the `weight.bin` of an `.mlpackage` byte for byte the same on
  Linux and macOS;
- `qvac-coreml-tool`, the host CLI for that generator (below);
- on Apple, the on-device compile of a generated package
  (`src/apple/compile.mm`) and the templates embedded in each engine library
  (`src/templates.h`);
- `python/coreml_template.py`, which makes templates from an unchanged
  exporter and checks them against it, and `python/golden.py`, the
  generator tripwire;
- the generation glue engines use (`include/qvac_coreml/generation.h`,
  below).

No library is installed: the headers under `include/` are build-interface
only.

## Process exit

The scheduler's threads are never joined and its manager is leaked on
purpose, but a Core ML compile or load cannot be cancelled, and `exit()`
destroys the frameworks' static state. Apple's compiler builds some of that
state lazily, during the first compile (a lookup table in
`ANECompiler`, another in MetalPerformanceShadersGraph's validator), so its
destructors are the first ones `exit()` runs. A process that ended while a
worker was still inside such a call could crash there (a SIGSEGV in
`ANECompiler` after the process had written its output), or spin on a freed
table.

**What the runtime does.** Each engine library (tts-cpp, qvac-parakeet,
audiogen-cpp: each has its own copy of the runtime) has one bounded exit wait:
it closes admission in that library, then waits for the job its worker is
running (a prepare, or a load and its warm-up, and any model the worker is
releasing) to return. The automatic default bound is 15 s on macOS and 2 s on
iOS. Each wait ends at the earlier of its own deadline and the library's one
exit deadline, which the first wait with a positive bound sets, so the exit
guard after an explicit shutdown gets only what is left. Several libraries in
one process each have their own. Once admission is closed, a job not yet
started never starts (also one a worker was selecting at that moment), and a
slot with no model and no running job fails with the reason
`the process is exiting: no new Core ML work starts` (logged once per library
on stderr): a fresh request, and a slot whose job was still queued. A strict
caller gets an error instead of waiting, also one already waiting. A ready
model stays usable, and a running job still publishes.

The wait runs from:

- **The exit guard**, a `thread_local` object of the main thread. When the
  main thread exits the process (`exit()`, or `return` from `main`, which
  first destroys `main`'s automatic objects), its `thread_local` destructors
  run before every static destructor and `atexit` handler, including those
  the frameworks register during a compile; `thread_local` objects of the
  main thread built after the guard are destroyed before it. The guard is
  armed automatically when the main thread makes a slot that can load or
  requests work, which pins the library image first; or explicitly, by the
  arming hook below.
- **The explicit shutdown hook**, which a host calls itself right before it
  exits.

When exit starts with neither (the engine was made and loaded off the main
thread and nothing armed the main thread, or `exit()` is called on another
thread), an `atexit` handler only closes admission, as before this wait
existed. It runs after the destructors registered since the manager was
created, and waiting there was measured to do harm: the job kept running on
the destroyed tables, crashing or spinning until the bound. A load still
running at that point can crash as it could before. The same holds past the
bound. After that handler has run nothing waits any more (the shutdown hook
only reports, and arming fails).

**Hooks for hosts.** Each engine re-exports them (`tts-cpp/coreml_exit.h`,
`parakeet/coreml_exit.h`, `audiogen-cpp/coreml_exit.h`; no-ops in builds
without Core ML):

- `coreml_arm_exit_guard()`: call on the main thread; on any other thread it
  returns false and does nothing. A JS host whose engines are made in a
  worker calls it on its JS thread when that thread is the process's main
  thread (Bare and Node on the desktop), for example where the addon instance
  is created. The qvac TTS and AudioGen addons make their engines in
  `activate()` on a worker; the ASR addon's activation initialises Parakeet
  synchronously on the calling JS thread, so whether it arms automatically
  depends on the host. Until an addon arms, a desktop process that exits while
  a load runs is not covered.
- `coreml_shutdown_for_exit(timeout_ms)`: closes admission and waits, from
  any thread. It is terminal, for every user of that library for the rest of
  the process. It pins the image even before the first request, so reloading
  an addon does not reopen admission. Slot status and prewarm results report
  queued work that cannot start as failed. Call it right before the process
  exits, while the host's main queue
  can still make progress, never on an addon unload in a process that goes on.
  It must also run before process finalization: from an `atexit` handler or a
  static destructor it is too late (the frameworks' lazily registered
  destructors may already have run; it then only reports). An acquisition
  during finalization does not wait for a running slot. The timeout is
  clamped to one hour; 0 or less only closes admission and fixes no deadline.
  It returns true when no job is running, false on timeout and always from
  inside a job.
- `coreml_set_automatic_exit_guard(false)`, per library: an automatically
  armed guard does not wait, whenever the call is made (a guard armed by the
  hook still does). On Apple platforms a main thread that ends with
  `pthread_exit()` while the process goes on (`dispatch_main()` does this)
  runs the guard's destructor too, and the runtime then closes admission for
  the rest of the process: later requests fail with the reason above. Hosts
  built that way turn automatic arming off and call the shutdown hook right
  before they exit. (glibc does not run the main thread's `thread_local`
  destructors at `pthread_exit()`, so there the guard does not run and
  nothing stops.)

**Cost.** Only a process that ends with a job in flight pays: it exits after
the rest of that job, at most the longest cold load (about 1 s for the
Supertonic vocoder, 6-11 s for Pocket, ACE-Step and the Parler DAC on an M2)
and never more than the bound; the compiled plan is then cached for the next
process. A job in its digest or generation phase notices the exit and stops
within milliseconds. On iOS an app is normally suspended or killed, with no
`exit()`; when it does exit, the automatic wait is 2 s at most (Mac Catalyst
gets the macOS default).

## Engine build options

`qvac_coreml_add_to(<engine-lib> NAMESPACE <ns> PREFIX <P> ENABLE <bool> [TEMPLATES <dirs>])`
in `qvac_coreml.cmake` folds the runtime into an engine library when the
engine's Core ML option (`TTS_CPP_COREML`, `PARAKEET_COREML`,
`AUDIOGEN_COREML`) is on. With it come these cache options:

| Option | Default | Effect |
|---|---|---|
| `<P>_COREML_GENERATE` (`TTS_CPP_COREML_GENERATE`, `PARAKEET_COREML_GENERATE`, `AUDIOGEN_COREML_GENERATE`) | `ON` on macOS, `OFF` on iOS | Compiles the generator (recipe interpreter, MILBlob writer, `GgufSource`, cache, on-device compile) into the library and embeds the template directories passed as `TEMPLATES`. Needs the `ggml` target. With `OFF`, only the runtime half (background loads of prebuilt sidecars) is compiled and no template is embedded. |
| `QVAC_COREML_TEMPLATE_BUDGET_MB` | `32` | The most template bytes (uncompressed) one library may embed. Configure fails above it with `the embedded Core ML templates take N bytes, over QVAC_COREML_TEMPLATE_BUDGET_MB=…`. |
| `QVAC_COREML_DEV` | `OFF` | Compiles the template override (below) into a release build. Test builds (`<P>_BUILD_TESTS=ON`) always have it. |

**Embedding.** Each file of each template (`Manifest.json`, `model.mlmodel`,
`recipe.txt`, `template.json` and an optional `literals.bin`; never the
test-only `reference-gguf.json`) becomes one generated assembler stub that
places it uncompressed in `__TEXT,__const` with `.incbin`, so the bytes are
paged in only when a package is written. A generated table lists
`{template id, file, begin, end, sha256}`; CMake computes each `sha256`, so
`template_sha256` (part of every cache key) is known without reading the
files. Editing a template file reconfigures the build. A template directory
must be named after its `template.json` `id`. Linux builds embed nothing.

**Template override.** Where compiled in, the `template_dir` option
(`Options::template_dir`) or the `QVAC_COREML_TEMPLATE_DIR` environment
variable makes the generator read `<dir>/<template-id>/` instead of the
embedded copy. Elsewhere `template_dir` is ignored with a warning.

**On-device compile.** `apple::compile_package()` calls
`+[MLModel compileModelAtURL:completionHandler:]`. Core ML writes the bundle
to `$TMPDIR/<package name>.mlmodelc`, a directory every process and every
cache directory shares. The pipeline gives each attempt its own package name,
`qvac-coreml-<key>-<pid>-<random>.mlpackage`, so prepares of the same weights
into different cache directories (each with its own key lock) can compile at
the same time without meeting there. A process killed mid-compile leaves
`$TMPDIR/qvac-coreml-<key>-<pid>-<random>.mlmodelc`, which a routine prune
removes after an hour. The bundle is then renamed into staging, or copied when
`$TMPDIR` is on another volume. `artifact.json` records the OS build and the
`coremlc` version read from the compiled `model.mil`. Generation needs
macOS 15. When Core ML rejects the package, the failure is recorded as a
compile failure: it is retried by the next process, and the same error code from
another process on the same OS build marks the artifact deterministic (not
retried until the OS build or the generator version changes). A missing
package or a failed move or copy is an I/O error: it is retried by the next
process and is never promoted, however often it repeats.

## Generating a stage's sidecar in an engine

`include/qvac_coreml/generation.h` is what an engine calls to give a stage
route `generate`:

1. `generation::probe(ids, gguf, resolved)` looks the templates up (embedded,
   or under the development override) and matches each against the GGUF:
   `general.architecture`, `match.metadata` and every recipe `source` (name,
   allowed type, shape). It reads the GGUF's header only, never the cache. A
   `Candidate` carries the template, its match result and its `memory` section.
2. `plan()` gets `generation_built`, `template_available`,
   `template_matched` and the template's `validated_for` units in its
   `PlanInput`. With no adjacent prebuilt it returns `generate` when the
   build has the generator, the GGUF matches, a cache directory is set,
   `generate` is not `never`, the OS is macOS 15 or later and the resolved
   units are validated.
3. The slot's `SlotSpec` gets `prepare = generation::prepare_request(...)`
   (the template, the identity of the GGUF the engine loaded its weights
   from, the cache directory), `prepare_fn = generation::prepare` (the cache
   pipeline with the on-device compile) and `tmpl`. Its `artifact_path` stays
   empty: the prepare job supplies the published bundle. The identity comes
   from `generation::source_stat_fd`: the engine opens the GGUF once, reads
   its metadata, mapping and streamed weights through that one descriptor,
   and records fstat of it, so the identity is the loaded file whatever the
   path named during the load. The prepare job opens the path and fails with
   `source_changed` unless it is that file; an identity marked `changed`
   (no descriptor, or a request for another path than the load's) fails the
   same way, and the stage stays on ggml. `generation::loaded_source` (stats
   of the path before the first open and after the last read) is the weaker
   form for an engine that cannot read through one descriptor: a file
   swapped in and restored between its two samples passes unseen.
   A stat taken only when the slot is made would follow a replacement.
   Owners of one artifact share a loaded model, or a running load, only
   with equal `SlotSpec::geometry`, so each geometry is validated by its own
   `LoadFn`; a prediction failure retires the (artifact, function, units)
   for every owner, whatever its geometry: their ready models, queued loads
   and later acquisitions.
4. The stage's `LoadFn` calls `generation::begin_load` before loading and
   `generation::end_load` after: a generated bundle loads under the shared
   lease of its cache entry, and `used/<key>` is touched after a successful
   load. Both do nothing for a prebuilt bundle.
   - The preparation releases its lease when it returns, so a prune can
     delete the artifact before the load takes its own. `begin_load` then
     sets `LoadResult::artifact_gone`, and the scheduler sends the slot back
     to preparation, which regenerates the artifact (or adopts what another
     process published).
   - After a deterministic load failure (the engine rejected the model or
     its geometry), `end_load` moves the artifact to `quarantine/` under the
     key's exclusive lock and sets `artifact_gone`, so the slot regenerates
     it at the same key. A transient failure, or an artifact another holder
     has leased, keeps it.
   - The scheduler allows two such regenerations in a row per slot (a ready
     load resets the count); after that the slot is `failed`.
5. A cache directory on a volume where generation cannot run (not a local
   apfs or hfs volume) gives `PrepareResult::unsupported = "cache_fs"`: the
   slot reports `unsupported(cache_fs)`, terminal and with no error class,
   and the stage stays on ggml.

Fit charges a generated stage with `generation::envelope_bytes(memory,
units)`: resident (runtime weights, twice the function I/O, state,
workspace) plus one transient (the larger of `generation_peak_bytes + 8 MiB +
compile_peak_bytes` and the unit class's `load_peak`), times the margin.
`resident_bytes`, `prepare_peak_bytes`, `load_peak_bytes` and
`transient_bytes` return those parts before the margin, so a memory
validation can hold each phase to its own constant.
`ane_compile_reserve_bytes` gives the memory admission reserve of a cold ANE
load. Tests replace the Core ML compile with
`generation::set_compile_override`.

### Measured memory constants

`extract` writes a template's `memory` section with the defaults above,
`margin` 1.25 and `coreml_bound: "conservative"`. A model's memory validation
replaces them with measured constants. It runs an overlapped schedule:
ggml work on the calling thread while the stage is generated, compiled,
loaded and warmed in the background, sampling the process footprint and
`ANECompilerService`'s size. `test-parler-dac-coreml --memory-schedule` is the
Parler DAC's. Its constants follow this convention:

- Each constant is 1.25 times the largest value measured over every run on
  every host in `validated_for`, rounded up to 1 MiB:
  - `compile_peak_bytes`: the prepare (from the start of generation to the end
    of the compile, over the footprint just before generation), less
    `generation_peak_bytes` and 8 MiB;
  - `load_peak_bytes.<units>`: the load and warm-up, over the footprint just
    before the load;
  - the resident part (`runtime_weight_bytes` + 2 x I/O + `state_bytes` +
    `workspace_bytes`) must cover 1.25 times the steady state once ready,
    over the footprint just before the load. `runtime_weight_bytes` stays
    the weights' size. `workspace_bytes` is what that figure needs beyond
    the weights, I/O and state, as on a GPU placement that holds its own
    copy of the weights and working memory, or 0 when the charged weights
    already cover it, as on the Neural Engine, which keeps the weights
    outside the process;
  - `ane_compile_reserve_bytes`: below.
- A schedule run passes only when 1.25 times each phase it measured fits the
  constant that charges it, so a run that sets a new maximum fails until the
  constants are recomputed. The whole run is held to the envelope as
  measured, since the envelope adds phases that each carry the margin.
- Each constant carries its own 1.25, so the envelope adds none: `margin` 1.0
  and `coreml_bound: "measured"`. The runtime reads `margin`; `coreml_bound`
  and `measured` are records.
- `memory.measured` records the derivation: `margin` (the 1.25 each constant
  carries), `runs` (how many runs the maxima cover), `resident_bytes` (the
  resident part the template charges, the sum above: at least 1.25 times the
  largest measured steady state, not the raw measurement) and `basis` (what
  each constant covers and which host set its maximum).
- `validated_for` lists every measured host as `{units, os, chip}`. `plan()`
  matches the units only, so every chip is charged the largest placement's
  values.

`ane_compile_reserve_bytes` is not part of the envelope. It is runtime
admission: a cold Neural Engine load waits until that much memory is free.
`ANECompilerService` is a long-lived system process (it runs as root under
`aned`, so a test cannot restart it, and it does not exit when idle). A
schedule measures it in two ways:

- its **growth** during the load: its peak while loading, less its size just
  before the first load;
- its **absolute** size while loading. On a shared host it includes what
  other processes compiled, before the load or during it.

Both sum every instance of the service, so another process that compiles
for the Neural Engine during the load counts in both. A service that
already holds memory from earlier compiles reuses it, so on such a dirty
baseline the growth understates a cold compile on a clean host, where the
growth is the whole size. The reserve is therefore 1.25 times the largest
growth measured from a clean start (at most 32 MiB before the load,
`ANE_COMPILER_CLEAN_BYTES` in `test/memory_probe.h`) on every validated
host, when every host has such runs. Otherwise it is 1.25 times the largest
absolute size while loading on any validated host, a conservative basis
that compares across placements, and the schedule holds 1.25 times the
absolute size to it. Each run report prints both, the size before the load,
and labels that baseline clean, dirty or unknown (no sample before the
load, in which case the growth is the whole size while loading).

## Building the tests and the tool

The portable tests and `qvac-coreml-tool` build on Linux and macOS (not on
Windows) with CMake 3.20 or newer and a C++17 compiler. The tool and the
GGUF test cases also need ggml (`ggml-base` for GGUF parsing and
`ggml_get_type_traits()->to_float` dequantisation).

Standalone, against a ggml install (for example the `ggml-speech` vcpkg
port):

```sh
cmake -S coreml -B build-coreml -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build-coreml
ctest --test-dir build-coreml -L coreml-runtime
./build-coreml/qvac-coreml-tool dump model.gguf
```

Without ggml on `CMAKE_PREFIX_PATH` the standalone build still configures,
builds the scheduler, recipe, MILBlob and cache tests without their GGUF
cases, and prints
`qvac-coreml-tool and the GGUF cases of the generator tests: not built (no ggml target)`.

Inside an engine build the tests and the tool come with the engine's test
option: `TTS_CPP_BUILD_TESTS`, `PARAKEET_BUILD_TESTS` or
`AUDIOGEN_BUILD_TESTS` (or `SPEECH_BUILD_TESTS` in the umbrella build). There
ggml is required: configure fails with
`qvac_coreml_add_tests(REQUIRE_GGML): no ggml target` instead of silently
dropping the tool and the GGUF cases.

```sh
cmake -S tts -B build-tts -DTTS_CPP_BUILD_TESTS=ON
cmake --build build-tts --target qvac-coreml-tool test-coreml-recipe
ctest --test-dir build-tts -L coreml-runtime
```

`test-coreml-exit-load` (label `fixture`) checks process exit on the device
with a real model. Each run copies the model to a fresh path and the program
to a copy signed with a fresh identifier, so Core ML compiles it again into a
cache of that run's own, and starts a child process that exits at a different
point of that load. Every child must get its load in flight, exit 0 within
20 s (the exit wait's bound is 15 s) in the state ready, loading or warming,
and, where the wait applies, show that its in-flight load returned before any
destructor registered during the load had run. A control child must first
load the model to ready, and at least half of the runs must exit with the load
in flight. Point it at a compiled model whose cold load takes about a second
or more and includes dilated convolutions, for example an exported Supertonic
vocoder sidecar:

```sh
cmake -S coreml -B build-coreml \
      -DQVAC_COREML_TEST_EXIT_MODEL=/path/to/supertonic2-vocoder.mlmodelc
cmake --build build-coreml --target test-coreml-exit-load
ctest --test-dir build-coreml -R test-coreml-exit-load --output-on-failure
```

Without that option the test is registered DISABLED; the macOS tts CI job
runs the binary directly on the vocoder it stages. Options for measurements:
`--engine-thread worker` (the engine is made off the main thread, as in the
addons), `--arm`, `--drain-ms D`, `--no-scribble`, `--exit-wait-ms 0` (no
wait: the behaviour before it existed) and `--report-only`.

The tool and the generator are compiled with `-ffp-contract=off`, so no
floating-point expression is fused into an FMA and the output does not
depend on the host CPU (x86-64 or arm64).

## Making a template

Templates live in `engines/<engine>/coreml-templates/<template-id>/`. The
exporter (`engines/*/scripts/export-*-coreml.py`) is never modified; it stays
the numerical reference. `python/coreml_template.py` needs the coremltools
venv pinned by `parakeet/scripts/requirements-coreml.txt` (numpy,
gguf-py, coremltools), the exporter's own requirements and a built
`qvac-coreml-tool` (`--tool`, or `$QVAC_COREML_TOOL`). A relative exporter
path, on the command line or in `template.json`, is resolved against
`--root` (every command takes it; the default is the checkout that holds the
script).

1. Write the shape manifest of a real GGUF:
   `qvac-coreml-tool dump model.gguf -o <dir>/reference-gguf.json`.
2. Draft the recipe from the exporter run on twin A (seed 1, f32):

   ```sh
   python coreml/python/coreml_template.py --tool build/qvac-coreml-tool suggest \
       --manifest <dir>/reference-gguf.json --id <template-id> -o <dir>/recipe.txt \
       --exporter tts/scripts/parler/export-dac-coreml.py \
       --exporter-arg=--gguf --exporter-arg={gguf} --exporter-arg=--output --exporter-arg={out}
   ```

   Every `weight.bin` blob is matched against `f16(t(name))`, transposes,
   scalar multiples, negation, `zeros` and `ones`. A blob it cannot explain is
   written as `blob <offset> <dtype> <count> ???` under an `UNEXPLAINED`
   comment; a blob that several tensors explain gets an `AMBIGUOUS` comment.
   Suggested constants come from the twin's values: confirm them against the
   exporter.
3. Complete `recipe.txt` by hand (the recipe language: `include/qvac_coreml/recipe.h`
   and the vocabulary below).
4. Write `template.json` with the fields only the author knows (`family`,
   `stage`, `export_kind`, `match.metadata`, `exporter` with `{gguf}` and
   `{out}` in its `args`, `default_compute_units`, `memory`), then extract:

   ```sh
   python coreml/python/coreml_template.py --tool build/qvac-coreml-tool extract --template-dir <dir>
   ```

   `extract` runs the exporter on twins A and B (seeds 1 and 2) and compares
   the two specs field by field. An inline fp16 or fp32 constant whose value
   differs between A and B (a weight the compiler folded into the program,
   such as a bias it negated) becomes a patch site: `model.mlmodel` holds a
   deterministic sentinel there, `template.json` `patches` records
   `{spec_offset, len, sentinel_hex}`, and `extract` adds a drafted `patch`
   line to `recipe.txt` for every site it can explain from twin A. Any other
   difference between the specs, or between the two `weight.bin` layouts,
   fails the extraction. Literals declared in `template.json` `literals`
   with an `exporter_blob` (a `weight.bin` blob offset) are copied from that
   blob into `literals.bin`, 64-byte aligned, after checking that A and B
   agree on it, so a literal never depends on the weights. `extract` then
   writes `model.mlmodel` (A's spec, deterministically serialised, sentinels
   in place), `Manifest.json` and the computed `template.json` fields
   (`functions`, `stateful`, `weight_bin_size`, `generation_peak_bytes`,
   `patches`, the literals' offsets, the exporter's sha256, the toolchain
   versions), and refreshes the header of `recipe.txt`. Every output is
   staged in a temporary copy of the template first, and the twin-A
   `generate` runs on that copy; the template directory is written only after
   every check has passed. A rejected extraction (a twin check, a recipe
   header or body, a recipe with no `end` line for drafted patch lines, the
   twin-A `generate`) therefore leaves the template's files untouched,
   including `reference-gguf.json` when `--manifest` names another file. A
   `model.mlmodel` that equals the
   new spec (the coremltools conversion date aside, sentinels in place) and a
   `Manifest.json` that differs only in its random item identifiers are kept
   byte for byte, so extracting an unchanged template again leaves its files,
   and so its `template_sha256`, as they were.
5. Check it against the exporter:

   ```sh
   python coreml/python/coreml_template.py --tool build/qvac-coreml-tool report --template-dir <dir>
   ```

   For twins A and B (f32) and Q (`--quant-types`, default `q8_0`; the
   exporter gets `qvac-coreml-tool f32` of it), `report` generates the
   package with the C++ generator (`qvac-coreml-tool generate`), compares its
   `weight.bin` blob by blob with the exporter's (`qvac-coreml-tool verify`)
   and its spec, patches applied, with the exporter's. `--all` checks every
   template under `engines/*/coreml-templates/`.

   On a production GGUF the same comparison is the real-weights gate:

   ```sh
   python coreml/python/coreml_template.py --tool build/qvac-coreml-tool real \
       --gguf models/Nemotron-3-Diarization.q8_0.gguf --template-dir <dir> [--template-dir <dir> ...]
   ```
6. Add the template's golden line:
   `python3 coreml/python/golden.py update --tool build/qvac-coreml-tool`.

### Recipe vocabulary

This generator version knows `t f32 f64 f16 add mul div recip clip reshape
permute take concat`, `let` bindings, and:

| Op | Semantics |
|---|---|
| `slice(x, axis, begin, end[, step])` | half-open, `0 <= begin < end <= dim`, `step >= 1` |
| `neg(x)` | IEEE negation (`neg(0)` is `-0`) |
| `zeros([d...])`, `ones([d...])` | f32 constants |
| `lit("id")` | a `template.json` `literals` entry `{id, dtype, shape, offset}`: bytes of `literals.bin` (fp16 or fp32, numpy order), as f32 |

`patch <spec_offset> fp16|fp32 <count> <expr>` lines write `count` values
into `model.mlmodel` at `spec_offset`, replacing the sentinel that
`template.json` `patches` records there. Patch lines come in spec offset
order and must not overlap; each lies on a declared patch site of exactly its
length, and every site has a line. `patch ... str`, `provided()`, `meta_f`
and `meta_s` belong to a later generator version.

### Gates in CI

`.github/workflows/coreml-templates-ci.yml` runs on changes to the exporters,
the templates, `coreml` and the venv pin, on the self-hosted
Apple-silicon lane: the `coreml_template.py` unit checks, `report` on the toy
templates (a harness self-check), `report --all` (the extraction and
cross-seed gates) and `golden.py check`.

The real-weights gate (`coreml_template.py real`) runs in the engine's Core
ML lane on a staged production GGUF: the Parakeet lane in `parakeet-ci.yml`
checks the Nemotron 3 Diarization templates on the official q8_0 GGUF
(pinned revision and sha256), then runs the parity test on diarizers the
engine generated from it.

`coreml/test/golden.txt` holds, per template, the generator version,
`template_sha256` and the sha256 of the `weight.bin` generated from twin A.
`golden.py check` (also the `test-coreml-golden` ctest on Linux and macOS)
fails when the generated bytes change while the generator version and the
template stay the same: a generator change that alters output must bump
`GENERATOR_VERSION` (`include/qvac_coreml/recipe.h`). A version or template
change, a template without a line or a line without a template also fail
until `golden.py update` is run and `golden.txt` committed.

### Toy templates

`test/toy-templates/` holds three test-only templates made by
`test/toy/export_toy.py` with coremltools' MIL builder: `toy-stateless`,
`toy-multi` (two functions sharing weights) and `toy-state` (an MLState
accumulator). They are routing and lifetime fixtures, never a gate on a
model. `test-coreml-toy` (label `coreml-toy`, macOS 15 or later) embeds them
in a test library through `qvac_coreml_add_to(... TEMPLATES ...)` and checks,
on the device: the embedded table against the source files; generate,
compile, load and predict for each toy, compared with the same arithmetic on
the GGUF's weights (CPU, and all compute units for the stateless toy);
`artifact.json`; the user metadata of an on-device-compiled bundle; the
memo hit on a second preparation; the compile output naming and the copy
fallback; the repair of an unusable artifact (moved to `quarantine/`,
regenerated, compiled again); and two preparations of the same weights into
two cache directories whose compiles run at the same time, each of which must
publish a bundle of its own that loads and predicts. It prints
`compile_peak_bytes` per toy.

```sh
ctest --test-dir build-coreml -L coreml-toy --output-on-failure
```

To rebuild the toys after changing `export_toy.py`:

```sh
python coreml/test/toy/export_toy.py --kind stateless --init seed.gguf
qvac-coreml-tool dump seed.gguf -o coreml/test/toy-templates/toy-stateless/reference-gguf.json
python coreml/python/coreml_template.py --tool qvac-coreml-tool extract \
    --template-dir coreml/test/toy-templates/toy-stateless
```

## qvac-coreml-tool

```
qvac-coreml-tool <command> [arguments]
```

Every command exits 0 on success and 2 on a usage error or a failure (the
message goes to stderr, prefixed `qvac-coreml-tool:`). `verify` exits 1 when a
blob does not match. JSON results go to stdout unless `-o` names a file.

A template directory, as `generate`, `digest` and `verify` take with
`--template`, holds `recipe.txt`, `template.json`, `model.mlmodel`,
`Manifest.json` and optionally `literals.bin`. Whenever one of these commands
reads a GGUF, it first checks that the GGUF matches the template (tensor
names, types and shapes as the recipe's `source` lines declare them) and fails
with `no_template: <reason>` when it does not.

### dump

```
qvac-coreml-tool dump <model.gguf> [-o <manifest.json>]
```

Writes a shape manifest: every metadata key with its GGUF type and value, and
every tensor's name, type, `n_dims`, `ne` and value statistics (min, max,
mean, std). `twin` reads it. `n_dims` is the count the file records: ggml's
GGUF API drops trailing dimensions of 1 (a `[1, 96, 7]` weight reads as 2-D),
while exporters read the rank through gguf-py, so `dump` reads it from the
file header and every GGUF the tool writes (`twin`, `f32`) records it too.

```sh
qvac-coreml-tool dump models/mimi.gguf -o mimi-manifest.json
```

### twin

```
qvac-coreml-tool twin --manifest <manifest.json> --seed <N> --types <type[,type...]> -o <out.gguf>
                      [--set key=value ...]
```

Writes a random GGUF with the same metadata and tensor geometry as the
manifest. Each tensor's values are drawn from a normal distribution with the
recorded mean and standard deviation, clamped to the recorded range. A tensor
whose recorded values are all equal (a scalar bias, a constant vector) is drawn
within ±25% of that value (±0.00025 around zero) instead, so every tensor
differs between seeds and `extract` can see any value the exporter folds into
the program. Tensors
that are not f32 in the original (quantised, f16, bf16) take the `--types`
entries in turn, quantised with ggml's own quantisers, when their row length is
a multiple of that type's block size; the rest stay f32. `--set` sets a string
metadata value (overriding the manifest's) and may be repeated. The same manifest,
seed and types give the same file on every host, so tests can use a twin in
place of a production model they cannot ship.

`--types` takes the types ggml can both write and read back, so a twin is
always a GGUF that the generator and `f32` can read:

| | `--types` values |
|---|---|
| float | `f32`, `f16`, `bf16` |
| legacy quants | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0` |
| k-quants | `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k` |
| non-linear quants | `iq4_nl`, `iq4_xs` |

Any other type, for example `f64` or `q8_k` (ggml has no quantiser for them)
or `q8_1` (ggml cannot read it back), fails with
`--types: cannot write <type> (supported: ...)` and exit status 2. This
happens before the manifest is read, so no output file is created.

```sh
qvac-coreml-tool twin --manifest mimi-manifest.json --seed 7 --types q8_0,f16 -o mimi-twin.gguf
```

### f32

```
qvac-coreml-tool f32 <in.gguf> -o <out.gguf>
```

Writes a copy whose tensors are dequantised to f32 by ggml
(`ggml_get_type_traits(type)->to_float`). Tensors of a type ggml cannot
convert are copied unchanged. Use it to give a Python exporter the same
values the generator reads from a quantised GGUF.

```sh
qvac-coreml-tool f32 mimi-twin.gguf -o mimi-twin-f32.gguf
```

### generate

```
qvac-coreml-tool generate --template <dir> --gguf <model.gguf> -o <out.mlpackage> [--options <opts>]
```

Runs the recipe interpreter and the MILBlob writer and writes an
`.mlpackage` (no Core ML compile, so it works on Linux). Prints the template
id, `template_sha256`, `source_digest`, `artifact_key`, the SHA-256 and size
of the written `weight.bin`, the generator's peak memory
(`generation_peak_bytes`) and `generator_version`. `--options` only feeds
the printed `artifact_key` (default `out=fp16`).

```sh
qvac-coreml-tool generate --template templates/mimi-decoder --gguf models/mimi.gguf -o mimi.mlpackage
```

### digest

```
qvac-coreml-tool digest --template <dir> --gguf <model.gguf> [--options <opts>]
```

Prints the cache identities of a GGUF for a template without generating
anything: `template_sha256`, `source_digest` (with the number of tensors and
bytes it covers), `artifact_key`, `memo_id`, the GGUF's real path and the
options (default `out=fp16`). A runtime with the same inputs looks up the same
cache entry.

```sh
qvac-coreml-tool digest --template templates/mimi-decoder --gguf models/mimi.gguf
```

### verify

```
qvac-coreml-tool verify --template <dir> (--gguf <model.gguf> | --ours <weight.bin>) --ref <weight.bin>
```

Compares a `weight.bin` with a reference one (usually the coremltools export
of the same weights), blob by blob. The left side is either generated from
`--gguf` in memory or read from `--ours`. For each blob it prints the offset,
the comparison class (`exact`, or `ulp` for blobs whose recipe uses
operations that may differ in the last place), the dtype, the number of
differing elements, the largest ULP distance and `ok` or `FAIL`; layout
errors (header, metadata, offsets, padding) are reported first. The last line
is `PASS` or `FAIL`.

```sh
qvac-coreml-tool verify --template templates/mimi-decoder --gguf models/mimi.gguf \
    --ref export/mimi.mlpackage/Data/com.apple.CoreML/weights/weight.bin
```
