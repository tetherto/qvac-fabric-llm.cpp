#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SPEECH_SUBDIR="engines/speech"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { PASS=$((PASS + 1)); echo "ok $PASS - $*"; }

check_scripts_parse() {
  local script
  for script in "$HERE"/*.sh; do bash -n "$script" || fail "$script does not parse"; done
  python3 -c 'import ast, sys; [ast.parse(open(path).read(), path) for path in sys.argv[1:]]' "$HERE"/*.py || fail "a python script does not parse"
  if command -v node > /dev/null; then
    node --check "$HERE/fetch-registry.js" || fail "fetch-registry.js does not parse"
    node --check "$HERE/dashboard/app.js" || fail "dashboard/app.js does not parse"
  fi
  ok "every script parses"
}

check_spec_fields() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
    | select([.value.bench_kind, .value.binary, (.value.cmake_target // .value.engine), .value.task, .value.model_label, .value.weights] | any(. == null))
    | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families without bench_kind/binary/cmake_target (or engine)/task/model_label/weights: $bad"
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | select(.value.engine != null)
    | select((.value.platform == null) or (.value.backends | type != "array") or (.value.binary | startswith("${BENCH_HOME}/") | not)) | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families of another engine without platform, backends and a \${BENCH_HOME} binary: $bad"
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
    | select([.value.runs, .value.warmup, .value.cpu_runs, .value.cpu_warmup, .value.timeout_minutes] | any(. == null))
    | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families without runs/warmup/cpu_runs/cpu_warmup/timeout_minutes: $bad"
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | select(.value.runs < 1 or .value.cpu_runs < 1 or .value.timeout_minutes < 1) | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families with zero runs or timeout: $bad"
  ok "every family declares the weekly fields"
}

check_args_are_provisioned() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family
    | ([.value.provision[] | (.as // ((.registry // .huggingface.name // .fixture // .prepared // .sidecar.name // .derived.name) | split("/") | last))]) as $files
    | [.value.args[] | select(startswith("${MODEL_DIR}/")) | sub("^\\$\\{MODEL_DIR\\}/"; "")] | map(select(. as $f | $files | index($f) | not))
    | select(length > 0) | "\($family): \(join(","))"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "args reference files no provision entry supplies: $bad"
  ok "every \${MODEL_DIR} file in args is provisioned"
}

check_derived_sources() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family | .value.provision as $p
    | range(0; $p | length) as $i | $p[$i].derived // empty | select(.kind == "requantize") | .source as $source
    | select(([$p[0:$i][] | .registry // "" | split("/") | last] | index($source)) == null) | "\($family): \($source)"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "requantize sources listed after or without their registry file: $bad"
  ok "requantized files follow their source in the provision list"
}

check_huggingface_pins() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family
    | .value.provision[] | .huggingface // empty | select((.revision // "" | length) != 40 or (.sha256 // "" | length) != 64) | "\($family): \(.name)"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "HuggingFace files without a 40-char revision and a 64-char sha256: $bad"
  ok "every HuggingFace file pins a revision and a sha256"
}

make_repo() {
  local dir="$1"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m init
}

setup_home() {
  export BENCH_HOME="$TMP/home" CONFIG_FILE="$TMP/none" KILL_GRACE_SECONDS=1 DEADLINE_POLL_SECONDS=1 LOCK_POLL_SECONDS=1
  mkdir -p "$BENCH_HOME/runs"
  make_repo "$BENCH_HOME/src/speech"
  make_repo "$BENCH_HOME/src/ggml"
  mkdir -p "$BENCH_HOME/src/speech/$SPEECH_SUBDIR/parakeet/test/samples"
  cp "$REPO/parakeet/test/samples/jfk.wav" "$BENCH_HOME/src/speech/$SPEECH_SUBDIR/parakeet/test/samples/jfk.wav"
  echo "project(speech)" > "$BENCH_HOME/src/speech/$SPEECH_SUBDIR/CMakeLists.txt"
  git -C "$BENCH_HOME/src/speech" add -A
  git -C "$BENCH_HOME/src/speech" -c user.name=test -c user.email=test@example.com commit -q -m fixture
}

check_lib_backends() {
  . "$HERE/lib.sh"
  load_config
  local expected="cpu,cuda,vulkan"
  [ "$OS_NAME" = "Darwin" ] && expected="cpu,metal"
  [ "$(resolve_backends)" = "$expected" ] || fail "auto backends on $OS_NAME: $(resolve_backends)"
  BACKENDS="cpu,cuda"
  [ "$(resolve_backends)" = "cpu,cuda" ] || fail "BACKENDS override ignored"
  BACKENDS="auto"
  ok "backends resolve per OS and from BACKENDS"
}

check_private_tool_dirs() {
  MODELS_DIR="$TMP/shared-models"
  load_config
  [ "$MODELS_DIR" = "$TMP/shared-models" ] || fail "load_config ignored MODELS_DIR"
  [ "$REGISTRY_CLIENT_DIR" = "$BENCH_HOME/registry-client" ] || fail "registry client dir left the bench home: $REGISTRY_CLIENT_DIR"
  [ "$MODELS_VENV" = "$BENCH_HOME/venv" ] || fail "venv left the bench home: $MODELS_VENV"
  case "$REGISTRY_CLIENT_DIR$MODELS_VENV" in *"$MODELS_DIR"*) fail "tool dirs inside MODELS_DIR" ;; esac
  printf 'BACKENDS=cpu\n' > "$TMP/config-backends.env"
  [ "$(BACKENDS=cuda CONFIG_FILE="$TMP/config-backends.env" bash -c '. "$0"; load_config; echo "$BACKENDS"' "$HERE/lib.sh")" = "cuda" ] || fail "config.env overrode an exported BACKENDS"
  MODELS_DIR="$BENCH_HOME/models"
  load_config
  ok "executable tool directories stay under the private bench home and exported BACKENDS wins over config.env"
}

check_lib_iterations() {
  [ "$(family_runs parakeet-tdt cuda)" = 10 ] || fail "parakeet-tdt gpu runs"
  [ "$(family_runs parakeet-tdt cpu)" = 5 ] || fail "parakeet-tdt cpu runs"
  [ "$(family_warmup moss-tts-f16 cpu)" = 0 ] || fail "moss-tts-f16 cpu warmup"
  [ "$(family_field acestep-turbo-q8_0 timeout_minutes)" = 120 ] || fail "acestep timeout"
  ok "iterations come from the backend-specific fields"
}

check_lib_targets() {
  local targets
  targets="$(cmake_targets)"
  for target in whisper-cli parakeet-cli supertonic-bench audio8-cli moss-cli music-cli; do
    case " $targets " in *" $target "*) ;; *) fail "cmake target $target missing from '$targets'" ;; esac
  done
  targets="$(SELECTED_FAMILIES=parakeet-tdt,supertonic3 cmake_targets)"
  [ "$targets" = "parakeet-cli supertonic-bench whisper-cli " ] || fail "selected-family targets: '$targets'"
  [ "$(PATH=/nonexistent cmake_generator)" = "Unix Makefiles" ] || fail "generator fallback without ninja"
  ok "cmake targets follow the selected families and the generator falls back to make"
}

check_lib_lock() {
  gpu_busy() { return 1; }
  GPU_LOCK="$TMP/lock"
  LOCK_WAIT_HOURS=0
  echo "someone" > "$GPU_LOCK"
  if wait_for_gpu; then fail "wait_for_gpu ignored a held lock"; fi
  if take_gpu_lock; then fail "take_gpu_lock overwrote a held lock"; fi
  [ "$(cat "$GPU_LOCK")" = "someone" ] || fail "held lock content changed"
  if acquire_gpu_lock; then fail "acquire_gpu_lock ignored a held lock"; fi
  rm -f "$GPU_LOCK"
  wait_for_gpu || fail "wait_for_gpu failed with a free GPU"
  acquire_gpu_lock || fail "acquire_gpu_lock failed on a free GPU"
  [ -f "$GPU_LOCK" ] || fail "take_gpu_lock wrote nothing"
  grep -q "speech weekly bench" "$GPU_LOCK" || fail "lock content"
  release_gpu_lock
  [ ! -e "$GPU_LOCK" ] || fail "release_gpu_lock left the file"
  [ "$GPU_LOCK_HELD" = 0 ] || fail "release_gpu_lock left the held flag set"
  echo "theirs" > "$GPU_LOCK"
  release_gpu_lock
  [ "$(cat "$GPU_LOCK")" = "theirs" ] || fail "release_gpu_lock removed a lock this run does not hold"
  rm -f "$GPU_LOCK"
  mkfifo "$GPU_LOCK"
  [ -z "$(lock_holder)" ] || fail "lock_holder read a FIFO"
  rm -f "$GPU_LOCK"
  [ "$(unset USER LOGNAME; current_user)" = "$(id -un)" ] || fail "current_user without USER and LOGNAME"
  GPU_LOCK="$TMP/no-such-dir/lock"
  if acquire_gpu_lock; then fail "acquire_gpu_lock succeeded on an unwritable lock path"; fi
  GPU_LOCK="$TMP/lock"
  trap 'rm -rf "$TMP"' EXIT
  ok "lock is taken atomically, released once, read only from regular files, and acquisition is bounded"
}

check_run_lock() {
  take_run_lock || fail "take_run_lock on a free bench home"
  [ -f "$BENCH_HOME/run.lock/owner" ] || fail "run lock owner file missing"
  if take_run_lock; then fail "take_run_lock succeeded while this process holds it"; fi
  release_run_lock
  [ ! -e "$BENCH_HOME/run.lock" ] || fail "release_run_lock left the lock"
  mkdir "$BENCH_HOME/run.lock"
  echo "99999999 stale" > "$BENCH_HOME/run.lock/owner"
  take_run_lock || fail "take_run_lock did not reclaim a stale lock"
  release_run_lock
  mkdir "$BENCH_HOME/run.lock"
  echo "$$ live" > "$BENCH_HOME/run.lock/owner"
  local status=0
  bash "$HERE/weekly.sh" --no-update --no-mail --label concurrent > /dev/null 2>&1 || status=$?
  [ "$status" = 1 ] || fail "a second weekly.sh did not fail fast (exit $status)"
  grep -q "another weekly.sh run holds" "$BENCH_HOME/runs/$(date +%F)-concurrent/FAILED" || fail "second run did not name the run lock"
  [ "$(cat "$BENCH_HOME/run.lock/owner")" = "$$ live" ] || fail "second run touched the live lock"
  rm -rf "${BENCH_HOME:?}/run.lock" "${BENCH_HOME:?}/runs/$(date +%F)-concurrent"
  trap 'rm -rf "$TMP"' EXIT
  ok "a second weekly.sh on the same bench home fails fast; stale run locks are reclaimed"
}

check_cell_signature() {
  local results="$TMP/cells/results/cpu" spec="$TMP/families-cell.json"
  mkdir -p "$results" "$BUILDS_DIR/cpu"
  cp "$TMP/families-test.json" "$spec"
  FAMILIES_JSON="$spec"
  run_family_driver() { echo '{"status":"ok","backend":"CPU","wall_ms_median":1}' > "$3"; }
  run_family clone-test cpu "$results" > "$TMP/cell1.log" 2>&1
  [ -f "$results/clone-test.json.sig" ] || fail "no cell signature written"
  run_family clone-test cpu "$results" > "$TMP/cell2.log" 2>&1
  grep -q "skip clone-test on cpu" "$TMP/cell2.log" || fail "unchanged inputs were not reused"
  jq '."clone-test".args = ["--changed"]' "$spec" > "$spec.new" && mv "$spec.new" "$spec"
  run_family clone-test cpu "$results" > "$TMP/cell3.log" 2>&1
  grep -q "rerun clone-test on cpu (inputs changed)" "$TMP/cell3.log" || fail "changed args did not trigger a rerun"
  echo "0000" > "$BENCH_HOME/models/store/jfk.wav.sha256"
  run_family clone-test cpu "$results" > "$TMP/cell4.log" 2>&1
  grep -q "rerun clone-test on cpu (inputs changed)" "$TMP/cell4.log" || fail "a changed model hash did not trigger a rerun"
  unset -f run_family_driver
  FAMILIES_JSON="$HERE/families.json"
  ok "a cell is reused only when build, iterations, spec and model hashes match"
}

check_resume_keeps_load_samples() {
  local run="$TMP/resume" results="$TMP/resume/results/cpu" sample
  mkdir -p "$results"
  FAMILIES_JSON="$TMP/families-test.json"
  run_family_driver() { echo '{"status":"ok","backend":"CPU","wall_ms_median":1}' > "$3"; }
  run_family clone-test cpu "$results" > "$TMP/resume1.log" 2>&1
  [ -f "$results/clone-test.load.json" ] || fail "a measured cell has no load sample"
  sample="$(cat "$results/clone-test.load.json")"
  echo '{"status":"timeout"}' > "$results/failed-test.json"
  echo '{"busy_before_pct":1.0,"load1_before":0.5,"cpu_count":8}' > "$results/failed-test.load.json"
  discard_failed_results "$run"
  [ -f "$results/clone-test.json" ] && [ -f "$results/clone-test.load.json" ] || fail "a resumed run dropped the load sample of an ok cell"
  [ ! -e "$results/failed-test.json" ] && [ ! -e "$results/failed-test.load.json" ] || fail "a resumed run kept a failed cell or its load sample"
  run_family clone-test cpu "$results" > "$TMP/resume2.log" 2>&1
  grep -q "skip clone-test on cpu" "$TMP/resume2.log" || fail "the ok cell was not reused on resume"
  [ "$(cat "$results/clone-test.load.json")" = "$sample" ] || fail "the reused cell lost or resampled its load sample"
  unset -f run_family_driver
  FAMILIES_JSON="$HERE/families.json"
  ok "a resumed run reuses an ok cell with its load sample and drops a failed cell with its sample"
}

check_machine_quirks() {
  MEMORY_SAMPLER=meminfo
  [ "$(memory_sampler_kind)" = "meminfo" ] || fail "MEMORY_SAMPLER override ignored"
  if [ -r /proc/meminfo ]; then
    start_gpu_sampler "$TMP/meminfo.csv"
    sleep 1
    stop_gpu_sampler
    [ "$(head -1 "$TMP/meminfo.csv" | awk -F', ' '{print NF}')" = 3 ] || fail "meminfo sampler row shape: $(head -1 "$TMP/meminfo.csv")"
  fi
  MEMORY_SAMPLER=auto
  [ "$(perl "$HERE/nsdate" +%N | wc -c | tr -d ' ')" = 10 ] || fail "nsdate +%N is not nine digits"
  [[ "$(perl "$HERE/nsdate")" =~ ^[0-9]{19}$ ]] || fail "nsdate default is not seconds plus nanoseconds"
  mkdir -p "$TMP/evict" && printf 'data' > "$TMP/evict/a.bin"
  [[ "$(python3 "$HERE/prepare-derived.py" evict "$TMP/evict")" =~ ^advised\ [01]\ files$ ]] || fail "evict subcommand"
  [ "$(busy_percent_between "100 50" "300 110")" = "70.0" ] || fail "busy_percent_between: $(busy_percent_between "100 50" "300 110")"
  [ "$(busy_percent_between "100 50" "100 50")" = "0.0" ] || fail "busy_percent_between without elapsed ticks"
  [ "$(busy_percent_between "5119870000 5062390000" "5119870600 5062390400")" = "33.3" ] || fail "busy_percent_between loses precision on long-uptime tick counts: $(busy_percent_between "5119870000 5062390000" "5119870600 5062390400")"
  if [ -r /proc/stat ]; then [[ "$(cpu_ticks)" =~ ^[0-9]+\ [0-9]+$ ]] || fail "cpu_ticks is not two integers: $(cpu_ticks)"; fi
  [ "$(echo "CPU usage: 3.57% user, 7.14% sys, 89.28% idle" | busy_percent_from_top)" = "10.7" ] || fail "busy_percent_from_top: $(echo "CPU usage: 3.57% user, 7.14% sys, 89.28% idle" | busy_percent_from_top)"
  [ "$(printf '' | busy_percent_from_top)" = "0.0" ] || fail "busy_percent_from_top without a top line"
  write_host_load "$TMP/load.json"
  [ "$(jq -r '[.busy_before_pct, .load1_before, .cpu_count] | map(type) | join(",")' "$TMP/load.json")" = "number,number,number" ] || fail "write_host_load shape: $(cat "$TMP/load.json")"
  printf '{}' > "$TMP/cell.json"; printf 'x' > "$TMP/cell.load.json"
  discard_cell "$TMP/cell.json"
  [ ! -e "$TMP/cell.load.json" ] || fail "discard_cell keeps the load sample"
  mkdir -p "$ENGINE_SRC/scripts/benchmarks"
  echo 'now=$(/bin/date +%s%N)' > "$ENGINE_SRC/scripts/benchmarks/run-family.sh"
  install_nanosecond_shim
  grep -q "bin/nsdate +%s%N" "$ENGINE_SRC/scripts/benchmarks/run-family.sh" || fail "clock shim not applied to run-family.sh"
  [ -x "$BENCH_HOME/bin/nsdate" ] || fail "clock shim not installed"
  bash -n "$HERE/scheduler.sh" || fail "scheduler.sh does not parse"
  ok "meminfo sampler, host load sample, nanosecond clock shim and scheduler are in place for the Spark, Strix and Mac mini"
}

check_page_cache_eviction() {
  (
    BENCH_HOME="$TMP/bench home"; MODELS_DIR="$TMP/shared models"; BUILDS_DIR="$BENCH_HOME/builds"; SRC_DIR="$BENCH_HOME/src"
    EVICT_PAGE_CACHE=1
    python3() { printf '%s\n' "$@" > "$TMP/evict-args"; }
    evict_page_cache
    [ "$(cat "$TMP/evict-args")" = "$(printf '%s\n' "$TOOLS_DIR/prepare-derived.py" evict "$MODELS_DIR/store" "$BUILDS_DIR" "$SRC_DIR" "$BENCH_HOME/deps")" ] ||
      { echo "eviction arguments: $(tr '\n' '|' < "$TMP/evict-args")" >&2; exit 1; }
    rm -f "$TMP/evict-args"; EVICT_PAGE_CACHE=0
    evict_page_cache
    [ ! -e "$TMP/evict-args" ] || { echo "eviction ran with EVICT_PAGE_CACHE=0" >&2; exit 1; }
  ) || fail "page cache eviction"
  ok "page cache eviction passes the model store and the build, source and dependency trees whole, spaces included"
}

check_skip_families() {
  (
    . "$HERE/weekly.sh"
    load_config
    SKIP_FAMILIES="supertonic3,moss-tts-f16"
    QUICK=1
    [ "$(selected_families 2>/dev/null)" = "parakeet-tdt,audio8-lm-q8_0-dec-q8_0" ] || { echo "SKIP_FAMILIES not applied: $(selected_families 2>/dev/null)" >&2; exit 1; }
    FAMILIES_ARG="moss-tts-f16"
    [ -z "$(selected_families 2>/dev/null)" ] || { echo "an explicitly requested skipped family slipped through" >&2; exit 1; }
    FAMILIES_ARG="parakeet-mlx,parakeet-tdt"
    BENCH_TEST_PLATFORM=Linux
    [ "$(selected_families 2>/dev/null)" = "parakeet-tdt" ] || { echo "a Darwin-only family ran on Linux: $(selected_families 2>/dev/null)" >&2; exit 1; }
    BENCH_TEST_PLATFORM=Darwin
    [ "$(selected_families 2>/dev/null)" = "parakeet-mlx,parakeet-tdt" ] || { echo "a Darwin-only family was skipped on Darwin: $(selected_families 2>/dev/null)" >&2; exit 1; }
  ) || fail "SKIP_FAMILIES"
  ok "SKIP_FAMILIES and the family platform remove families from quick, default and explicit selections"
}

check_skip_cells() {
  (
    . "$HERE/weekly.sh"
    load_config
    SKIP_FAMILIES="moss-tts-f16,parakeet-tdt:cuda,supertonic3:vulkan"
    QUICK=1
    [ "$(selected_families 2>/dev/null)" = "parakeet-tdt,supertonic3,audio8-lm-q8_0-dec-q8_0" ] || { echo "a cell skip removed its whole family: $(selected_families 2>/dev/null)" >&2; exit 1; }
    [ "$(skipped_cells)" = "parakeet-tdt:cuda,supertonic3:vulkan" ] || { echo "skipped_cells: $(skipped_cells)" >&2; exit 1; }
    run_family() { echo "ran $1 on $2"; }
    case "$(run_family_if_allowed parakeet-tdt cuda "$TMP")" in *"skip parakeet-tdt on cuda (SKIP_FAMILIES)") ;; *) echo "a skipped cell ran" >&2; exit 1 ;; esac
    [ "$(run_family_if_allowed parakeet-tdt vulkan "$TMP")" = "ran parakeet-tdt on vulkan" ] || { echo "a cell skip leaked to another backend" >&2; exit 1; }
    SKIP_FAMILIES="moss-tts-f16"
    [ -z "$(skipped_cells)" ] || { echo "skipped_cells without cell entries: $(skipped_cells)" >&2; exit 1; }
  ) || fail "SKIP_FAMILIES cells"
  ok "family:backend entries of SKIP_FAMILIES skip one cell and keep the family on the other backends"
}

check_engine_platform_options() {
  (
    . "$HERE/lib.sh"
    load_config
    [ "$(engine_platform_options Darwin)" = "-DPARAKEET_COREML=ON -DTTS_CPP_COREML=ON" ] || { echo "Darwin builds must enable the Parakeet and TTS Core ML sidecars" >&2; exit 1; }
    [ -z "$(engine_platform_options Linux)" ] || { echo "Linux builds must not carry Darwin options" >&2; exit 1; }
    build_signature cuda | grep -q "engine_options=" || { echo "engine options missing from the build signature" >&2; exit 1; }
  ) || fail "engine platform options"
  ok "Darwin engine builds enable the Core ML sidecars and the option is part of the build signature"
}

check_family_backends() {
  (
    . "$HERE/lib.sh"
    load_config
    family_runs_on_backend parakeet-mlx metal || { echo "parakeet-mlx must run on metal" >&2; exit 1; }
    ! family_runs_on_backend parakeet-mlx cpu || { echo "parakeet-mlx must not run on cpu" >&2; exit 1; }
    family_runs_on_backend parakeet-tdt cpu || { echo "a family without backends must run everywhere" >&2; exit 1; }
    [ "$(family_backends mlx-whisper-abcba)" = "metal" ] || { echo "family_backends: $(family_backends mlx-whisper-abcba)" >&2; exit 1; }
  ) || fail "family backends"
  ok "a family's backends list keeps it off the other backends"
}

check_lib_deadline() {
  local status=0
  run_with_deadline 0 sleep 30 || status=$?
  [ "$status" = "$DEADLINE_EXIT_CODE" ] || fail "run_with_deadline returned $status for a timed-out command"
  run_with_deadline 1 true || fail "run_with_deadline failed a quick command"
  status=0
  run_with_deadline 1 false || status=$?
  [ "$status" = 1 ] || fail "run_with_deadline hid the exit code ($status)"
  ok "run_with_deadline kills on timeout and passes exit codes through"
}

check_failed_result() {
  write_failed_result moss-tts-f16 cuda timeout "$TMP/failed.json"
  [ "$(jq -r .status "$TMP/failed.json")" = "timeout" ] || fail "failed result status"
  [ "$(jq -r .family "$TMP/failed.json")" = "moss-tts-f16" ] || fail "failed result family"
  ok "write_failed_result emits a driver-shaped row"
}

make_run() {
  mkdir -p "$BENCH_HOME/runs/$1"
  echo '{"run":{}}' > "$BENCH_HOME/runs/$1/summary.json"
  [ "${2:-done}" = "done" ] && touch "$BENCH_HOME/runs/$1/DONE"
  return 0
}

check_previous_and_prune() {
  make_run 2026-01-04
  make_run 2026-01-11
  make_run 2026-01-12-quick
  make_run 2026-01-13-fix1
  make_run 2026-01-18 pending
  [ "$(previous_summary "$BENCH_HOME/runs/2026-01-18")" = "$BENCH_HOME/runs/2026-01-11/summary.json" ] || fail "previous of a pending run skips quick and labelled runs"
  [ "$(previous_summary "$BENCH_HOME/runs/2026-01-11")" = "$BENCH_HOME/runs/2026-01-04/summary.json" ] || fail "previous skips the current run"
  KEEP_RUNS=2
  prune_old_runs
  [ ! -d "$BENCH_HOME/runs/2026-01-04" ] || fail "prune kept the oldest run"
  [ ! -d "$BENCH_HOME/runs/2026-01-12-quick" ] || fail "prune kept an old quick run"
  [ -d "$BENCH_HOME/runs/2026-01-18" ] || fail "prune removed a recent run"
  ok "previous_summary and prune_old_runs walk runs/ newest first"
}

make_engine_origin() {
  local origin="$1" subdir="$2" tree="$1${2:+/$2}"
  make_repo "$origin"
  mkdir -p "$tree/tts/scripts"
  cp "$REPO/CMakeLists.txt" "$tree/CMakeLists.txt"
  cp "$REPO/tts/scripts/setup-ggml.sh" "$tree/tts/scripts/setup-ggml.sh"
  git -C "$origin" add -A && git -C "$origin" -c user.name=test -c user.email=test@example.com commit -q -m tree
}

check_default_source_layout() {
  (
    unset ENGINE_REPO ENGINE_BRANCH ENGINE_SUBDIR
    . "$HERE/lib.sh"
    load_config
    local origin="$TMP/engine-origin" expected
    expected="$(sed -n 's/^GGML_REF="\([0-9a-f]\{40\}\)".*/\1/p' "$REPO/tts/scripts/setup-ggml.sh" | head -n 1)"
    [ -n "$expected" ] || { echo "the tree's own setup-ggml.sh has no GGML_REF" >&2; exit 1; }
    [ "$ENGINE_REPO" = "https://github.com/tetherto/qvac-fabric-llm.cpp.git" ] || { echo "default engine repository: $ENGINE_REPO" >&2; exit 1; }
    [ "$ENGINE_SRC" = "$BENCH_HOME/src/speech/$SPEECH_SUBDIR" ] || { echo "default engine sources: $ENGINE_SRC" >&2; exit 1; }
    [ "$REPO" = "$(git -C "$REPO" rev-parse --show-toplevel)/$ENGINE_SUBDIR" ] || { echo "ENGINE_SUBDIR $ENGINE_SUBDIR is not where this tree lives: $REPO" >&2; exit 1; }
    make_engine_origin "$origin" "$ENGINE_SUBDIR"
    ENGINE_REPO="file://$origin"; ENGINE_BRANCH="$(git -C "$origin" rev-parse --abbrev-ref HEAD)"; ENGINE_CHECKOUT="$TMP/engine-checkout"
    ENGINE_SRC="$ENGINE_CHECKOUT/$ENGINE_SUBDIR"
    update_repo "$ENGINE_REPO" "$ENGINE_BRANCH" "$ENGINE_CHECKOUT" "" || { echo "engine checkout failed" >&2; exit 1; }
    engine_source_present || { echo "the default layout has no engine sources" >&2; exit 1; }
    [ "$(engine_ggml_pin)" = "$expected" ] || { echo "engine pin under the default layout: $(engine_ggml_pin)" >&2; exit 1; }
    [ "$(ggml_ref)" = "$expected" ] || { echo "ggml ref under the default layout: $(ggml_ref)" >&2; exit 1; }
    make_engine_origin "$TMP/flat-origin" ""
    ENGINE_REPO="file://$TMP/flat-origin"; ENGINE_CHECKOUT="$TMP/flat-checkout"; ENGINE_SRC="$ENGINE_CHECKOUT/$ENGINE_SUBDIR"
    update_repo "$ENGINE_REPO" "$ENGINE_BRANCH" "$ENGINE_CHECKOUT" "" || { echo "flat checkout failed" >&2; exit 1; }
    engine_source_present > /dev/null && { echo "a repository without $ENGINE_SUBDIR passed as an engine source" >&2; exit 1; }
    ENGINE_SUBDIR="" load_config
    [ "$ENGINE_SRC" = "$BENCH_HOME/src/speech" ] || { echo "an empty ENGINE_SUBDIR resolves to $ENGINE_SRC" >&2; exit 1; }
  ) || fail "default source layout"
  ok "the default engine repository, branch and subdirectory resolve this tree's layout and its ggml pin, and a repository without that subdirectory is refused"
}

check_ggml_source() {
  (
    . "$HERE/lib.sh"
    load_config
    local pin_file="$ENGINE_SRC/$ENGINE_GGML_PIN_FILE" origin="$TMP/ggml-origin" first second
    [ "$GGML_REPO" = "https://github.com/tetherto/qvac-fabric-llm.cpp.git" ] || { echo "default ggml repository: $GGML_REPO" >&2; exit 1; }
    make_repo "$origin"
    mkdir -p "$origin/ggml" && echo "project(ggml)" > "$origin/ggml/CMakeLists.txt"
    git -C "$origin" add -A && git -C "$origin" -c user.name=test -c user.email=test@example.com commit -q -m first
    first="$(git -C "$origin" rev-parse HEAD)"
    git -C "$origin" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m second
    second="$(git -C "$origin" rev-parse HEAD)"
    git -C "$origin" config uploadpack.allowAnySHA1InWant true
    [ -z "$(ggml_ref)" ] || { echo "a ggml ref without any pin: $(ggml_ref)" >&2; exit 1; }
    update_ggml > /dev/null && { echo "update_ggml ran without a ggml commit" >&2; exit 1; }
    mkdir -p "$(dirname "$pin_file")"
    printf 'GGML_REPO_URL="x"\nGGML_REF="%s"   # the pin\n' "$first" > "$pin_file"
    [ "$(ggml_ref)" = "$first" ] || { echo "engine pin not read: $(ggml_ref)" >&2; exit 1; }
    GGML_REPO="file://$origin"; GGML_SRC="$TMP/ggml-checkout"
    update_ggml || { echo "update_ggml failed at the engine pin" >&2; exit 1; }
    [ "$(repo_sha "$GGML_SRC")" = "$first" ] || { echo "checkout is not at the engine pin" >&2; exit 1; }
    [ "$(ggml_cmake_source)" = "$GGML_SRC/ggml" ] || { echo "cmake source: $(ggml_cmake_source)" >&2; exit 1; }
    BUILDS_DIR="$TMP/ggml-builds"
    mkdir -p "$BUILDS_DIR/cpu/ggml" "$BUILDS_DIR/cuda/ggml"
    echo "CMAKE_HOME_DIRECTORY:INTERNAL=$GGML_SRC" > "$BUILDS_DIR/cpu/ggml/CMakeCache.txt"
    echo "CMAKE_HOME_DIRECTORY:INTERNAL=$GGML_SRC/ggml" > "$BUILDS_DIR/cuda/ggml/CMakeCache.txt"
    discard_ggml_build_of_another_source cpu; discard_ggml_build_of_another_source cuda; discard_ggml_build_of_another_source vulkan
    [ ! -e "$BUILDS_DIR/cpu/ggml" ] || { echo "a ggml build of the old source layout was kept" >&2; exit 1; }
    [ -f "$BUILDS_DIR/cuda/ggml/CMakeCache.txt" ] || { echo "a ggml build of the current source was discarded" >&2; exit 1; }
    git -C "$GGML_SRC" remote set-url origin "file://$TMP/moved-away"
    printf 'GGML_REF="%s"\n' "$second" > "$pin_file"
    update_ggml || { echo "update_ggml did not follow a moved pin or repository" >&2; exit 1; }
    [ "$(repo_sha "$GGML_SRC")" = "$second" ] || { echo "checkout did not move to the new engine pin" >&2; exit 1; }
    GGML_BRANCH="some-branch"
    [ "$(ggml_ref)" = "some-branch" ] && [ "$(ggml_ref_source)" = "branch some-branch" ] || { echo "GGML_BRANCH does not override the engine pin" >&2; exit 1; }
    GGML_PIN="$first"
    [ "$(ggml_ref)" = "$first" ] && [ "$(ggml_ref_source)" = "GGML_PIN" ] || { echo "GGML_PIN does not override the branch" >&2; exit 1; }
    GGML_SRC="$TMP/flat-ggml"; mkdir -p "$GGML_SRC"
    [ "$(ggml_cmake_source)" = "$GGML_SRC" ] || { echo "cmake source of a flat ggml tree: $(ggml_cmake_source)" >&2; exit 1; }
  ) || fail "ggml source"
  ok "ggml is built from qvac-fabric-llm.cpp at the commit the engine pins, follows a moved pin, and GGML_PIN or GGML_BRANCH override it"
}

check_gpu_lease() {
  (
    . "$HERE/weekly.sh"
    export HOME="$TMP/lease-home" LOCK_WAIT_HOURS=2
    mkdir -p "$HOME/.local/bin"
    load_config
    BACKENDS="cpu,cuda"
    needs_gpu_lease && { echo "a lease is needed without the lease tool" >&2; exit 1; }
    printf '#!/bin/sh\nprintf "%%s\\n" "$*" > "%s/lease-args"\n' "$TMP" > "$HOME/.local/bin/gpu-run"
    chmod +x "$HOME/.local/bin/gpu-run"
    PATH="$HOME/.local/bin:$PATH"
    needs_gpu_lease || { echo "no lease asked for a cuda run on a host with the lease tool" >&2; exit 1; }
    BACKENDS="cpu"
    needs_gpu_lease && { echo "a cpu-only run asked for a lease" >&2; exit 1; }
    BACKENDS="cpu,vulkan"; GPU_INDEX=1
    export_gpu_selection
    [ "$CUDA_VISIBLE_DEVICES" = 1 ] && [ "$GGML_VK_VISIBLE_DEVICES" = 1 ] || { echo "the configured GPU is not selected outside a lease" >&2; exit 1; }
    ( BENCH_HOME="$TMP/home" CONFIG_FILE="$TMP/none" bash "$HERE/weekly.sh" --no-update --no-mail --backends vulkan --label lease > /dev/null 2>&1 & echo $! > "$TMP/lease-pid"; wait )
    [ -s "$TMP/lease-args" ] || { kill "$(cat "$TMP/lease-pid")" 2> /dev/null; echo "the run did not go through the lease tool" >&2; exit 1; }
    [ "$(cat "$TMP/lease-args")" = "-n 1 --timeout 7200 -- /bin/bash $HERE/weekly.sh --no-update --no-mail --backends vulkan --label lease" ] || { echo "lease command: $(cat "$TMP/lease-args")" >&2; exit 1; }
    [ ! -d "$TMP/home/runs/$(date +%F)-lease" ] || { echo "the run started outside the lease" >&2; exit 1; }
    export GPU_RUN_LEASE=7 CUDA_VISIBLE_DEVICES="GPU-uuid"
    needs_gpu_lease && { echo "a second lease asked for inside a lease" >&2; exit 1; }
    GPU_INDEX=1; load_config; export_gpu_selection
    [ "$GPU_INDEX" = 0 ] && [ "$CUDA_VISIBLE_DEVICES" = "GPU-uuid" ] && [ -z "${GGML_VK_VISIBLE_DEVICES:-}" ] && [ "$GPU_LOCK_QUIET" = 1 ] ||
      { echo "inside a lease: index $GPU_INDEX cuda $CUDA_VISIBLE_DEVICES vk ${GGML_VK_VISIBLE_DEVICES:-}" >&2; exit 1; }
    cat > "$HOME/.local/bin/nvidia-smi" <<'FAKE'
#!/bin/sh
echo "$*" >> "$SMI_CALLS"
case "$*" in
  *"-i 0 --query-compute-apps"*) echo 4242 ;;
  *"-i GPU-leased-1 --query-compute-apps"*) ;;
  *"-i GPU-leased-1 --query-gpu=timestamp"*) echo "2026/11/15 10:00:00.000, 2048, 37" ;;
  *"-i GPU-leased-1 --query-gpu=name"*) echo "RTX leased" ;;
  *"--query-gpu"*) echo "RTX busy neighbour" ;;
esac
FAKE
    chmod +x "$HOME/.local/bin/nvidia-smi"
    export SMI_CALLS="$TMP/smi-calls" CUDA_VISIBLE_DEVICES="GPU-leased-1" GPU_RUN_GPUS=1
    [ "$(gpu_monitor_target)" = "GPU-leased-1" ] && [ "$(gpu_physical_index)" = 1 ] || { echo "leased GPU 1 is monitored as $(gpu_monitor_target), index $(gpu_physical_index)" >&2; exit 1; }
    gpu_busy_linux && { echo "GPU 0's job of another user counted as busy while GPU 1 is leased" >&2; exit 1; }
    MEMORY_SAMPLER=auto
    start_gpu_sampler "$TMP/lease-gpu.csv"; sleep 1; stop_gpu_sampler
    grep -q "2048, 37" "$TMP/lease-gpu.csv" || { echo "the sampler did not read the leased GPU: $(cat "$TMP/lease-gpu.csv")" >&2; exit 1; }
    [ "$(bash "$HERE/machine-info.sh" | jq -r '[.gpu.name, .gpu.index] | join("|")')" = "RTX leased|1" ] || { echo "machine info names another GPU: $(bash "$HERE/machine-info.sh" | jq -c .gpu)" >&2; exit 1; }
    grep -q -- "-i 0 " "$SMI_CALLS" && { echo "nvidia-smi was asked about index 0 inside a lease: $(grep -- "-i 0 " "$SMI_CALLS" | head -n 1)" >&2; exit 1; }
    unset GPU_RUN_LEASE; GPU_INDEX=0
    [ "$(gpu_monitor_target)" = 0 ] && [ "$(gpu_physical_index)" = 0 ] || { echo "outside a lease the configured index is not monitored" >&2; exit 1; }
    gpu_busy_linux || { echo "a busy configured GPU is not seen outside a lease" >&2; exit 1; }
  ) || fail "gpu lease"
  ok "on a host with gpu-run a GPU run takes one lease for itself, and busy checks, sampling and machine info follow the leased GPU by its UUID"
}

check_build_marker() {
  mkdir -p "$BUILDS_DIR/cpu"
  if build_is_current cpu; then fail "build_is_current without a marker"; fi
  build_signature cpu > "$(build_marker cpu)"
  build_is_current cpu || fail "build_is_current with a matching marker"
  git -C "$ENGINE_SRC" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m bump
  if build_is_current cpu; then fail "build_is_current after the engine moved"; fi
  mkdir -p "$BUILDS_DIR/cuda"
  build_signature cuda > "$(build_marker cuda)"
  build_is_current cuda || fail "build_is_current for cuda with a fresh marker"
  CUDA_ARCH=89-real
  if build_is_current cuda; then fail "build_is_current after CUDA_ARCH changed"; fi
  CUDA_ARCH=120a-real
  CMAKE_PREFIX_PATH=/opt/deps
  if build_is_current cuda; then fail "build_is_current after CMAKE_PREFIX_PATH changed"; fi
  [ "$(cmake_prefix_path)" = "/opt/deps" ] || fail "cmake_prefix_path without an own prefix"
  [ "$(cmake_prefix_path /b/ggml-install)" = "/b/ggml-install;/opt/deps" ] || fail "cmake_prefix_path with an own prefix"
  CMAKE_PREFIX_PATH=
  [ "$(cmake_prefix_path /b/ggml-install)" = "/b/ggml-install" ] || fail "cmake_prefix_path without extra prefixes"
  mkdir -p "$TMP/rebuild/results/cuda"
  touch "$TMP/rebuild/results/cuda/old.json"
  discard_backend_results "$TMP/rebuild" cuda
  [ ! -e "$TMP/rebuild/results/cuda/old.json" ] || fail "discard_backend_results kept old rows"
  ok "build markers track the commits, the build options and the cmake prefixes, and a rebuild discards that backend's rows"
}

check_fetch_models_offline() {
  cat > "$TMP/families-test.json" <<'SPEC'
{
  "clone-test": {
    "cmake_target": "x", "bench_kind": "time-wrapped", "binary": "x", "models": [], "args": [],
    "task": "TTS", "model_label": "x", "weights": "x", "runs": 1, "warmup": 0, "cpu_runs": 1, "cpu_warmup": 0, "timeout_minutes": 1,
    "provision": [
      {"fixture": "parakeet/test/samples/jfk.wav"},
      {"fixture": "parakeet/test/samples/jfk.wav", "as": "voice.wav"},
      {"derived": {"name": "jfk-24k.wav", "kind": "resample", "source": "parakeet/test/samples/jfk.wav", "rate": 24000}},
      {"sidecar": {"exporter": "engines/stub-export.py", "source": "jfk.wav", "name": "jfk-vocoder.mlmodelc", "args": ["--palettize", "6"]}}
    ]
  }
}
SPEC
  cat > "$TMP/stub-coreml-python" <<'STUB'
#!/usr/bin/env bash
out=""; dir=""; pal=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift ;; --compile-dir) dir="$2"; shift ;; --palettize) pal="$2"; shift ;; esac; shift; done
[ "$pal" = "6" ] || { echo "palettize argument missing" >&2; exit 1; }
mkdir -p "$out" "$dir/$(basename "${out%.mlpackage}").mlmodelc"
STUB
  chmod +x "$TMP/stub-coreml-python"
  FAMILIES_JSON="$TMP/families-test.json" COREML_PYTHON="$TMP/stub-coreml-python" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch.log" 2>&1 || fail "fetch-models.sh failed: $(cat "$TMP/fetch.log")"
  [ -d "$BENCH_HOME/models/store/jfk-vocoder.mlmodelc" ] || fail "sidecar not exported into the store"
  [ ! -e "$BENCH_HOME/models/store/jfk-vocoder.mlpackage" ] || fail "sidecar package not removed after compiling"
  [ -L "$BENCH_HOME/models/families/clone-test/jfk-vocoder.mlmodelc" ] || fail "sidecar not linked into the family"
  [ -L "$BENCH_HOME/models/families/clone-test/jfk.wav" ] || fail "fixture not linked"
  [ -L "$BENCH_HOME/models/families/clone-test/jfk-24k.wav" ] || fail "derived file not linked"
  [ "$(readlink "$BENCH_HOME/models/families/clone-test/voice.wav")" = "$BENCH_HOME/models/store/jfk.wav" ] || fail "as link not pointing at the store file"
  [ -s "$BENCH_HOME/models/store/jfk.wav.sha256" ] || fail "sha256 sidecar"
  grep -q "^fixture parakeet" "$BENCH_HOME/models/store/jfk.wav.source" || fail "source sidecar"
  grep -q "^derived resample 24000" "$BENCH_HOME/models/store/jfk-24k.wav.source" || fail "derived source sidecar"
  ok "fetch-models.sh provisions fixtures and derived files offline and links a file under its as name"
}

check_fetch_models_reprovisions_on_origin_change() {
  local before after
  before="$(cat "$BENCH_HOME/models/store/jfk.wav.source")"
  git -C "$ENGINE_SRC" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m moved
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch2.log" 2>&1 || fail "second fetch-models.sh failed"
  after="$(cat "$BENCH_HOME/models/store/jfk.wav.source")"
  [ "$before" != "$after" ] || fail "fixture origin not refreshed after the engine moved"
  printf 'stale' > "$BENCH_HOME/models/store/jfk-24k.wav"
  rm -f "$BENCH_HOME/models/store/jfk-24k.wav.source"
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch3.log" 2>&1 || fail "third fetch-models.sh failed"
  [ "$(wc -c < "$BENCH_HOME/models/store/jfk-24k.wav" | tr -d ' ')" -gt 1000 ] || fail "a file without an origin sidecar was kept"
  ok "files are re-provisioned when their recorded origin differs or is missing"
}

check_fetch_registry_discards_partial() {
  command -v node > /dev/null || { ok "fetch-registry.js partial check skipped without node"; return 0; }
  mkdir -p "$TMP/registry-stub" "$TMP/registry-store"
  cat > "$TMP/registry-stub/index.js" <<'STUB'
class QVACRegistryClient {
  async ready () {}
  async close () {}
  async downloadModel (key, source, options) { require('fs').writeFileSync(options.outputFile, `fresh ${key}`) }
}
module.exports = { QVACRegistryClient }
STUB
  echo stale > "$TMP/registry-store/model.gguf.part"
  REGISTRY_CLIENT="$TMP/registry-stub/index.js" node "$HERE/fetch-registry.js" "$TMP/registry-store" acestep/2026-07-22/model.gguf > "$TMP/registry.log" 2>&1 || fail "fetch-registry.js failed: $(cat "$TMP/registry.log")"
  grep -q "discard stale" "$TMP/registry.log" || fail "a stale .part was not discarded"
  [ ! -e "$TMP/registry-store/model.gguf.part" ] || fail "the .part file survived the download"
  grep -q "fresh qvac_models_compiled/ggml/acestep/2026-07-22/model.gguf" "$TMP/registry-store/model.gguf" || fail "the model was not downloaded afresh"
  REGISTRY_CLIENT="$TMP/registry-stub/index.js" node "$HERE/fetch-registry.js" "$TMP/registry-store" acestep/2026-07-22/model.gguf | grep -q "^present" || fail "a present model was fetched again"
  ok "fetch-registry.js discards a stale partial download and keeps a present model"
}

check_weekly_helpers() {
  (
    . "$HERE/weekly.sh"
    RUN_DIR="$TMP/run-helpers"
    mkdir -p "$RUN_DIR/results/cpu"
    touch "$RUN_DIR/DONE" "$RUN_DIR/build-failures.txt" "$RUN_DIR/narrative.md" "$RUN_DIR/summary.json"
    echo '{"status":"ok"}' > "$RUN_DIR/results/cpu/good.json"
    echo '{"status":"timeout"}' > "$RUN_DIR/results/cpu/bad.json"
    touch "$RUN_DIR/results/cpu/bad.log"
    clear_run_outputs
    [ ! -e "$RUN_DIR/build-failures.txt" ] && [ ! -e "$RUN_DIR/narrative.md" ] && [ ! -e "$RUN_DIR/summary.json" ] || { echo "clear_run_outputs kept stale outputs" >&2; exit 1; }
    [ -e "$RUN_DIR/results/cpu/good.json" ] || { echo "clear_run_outputs dropped an ok row" >&2; exit 1; }
    [ ! -e "$RUN_DIR/results/cpu/bad.json" ] && [ ! -e "$RUN_DIR/results/cpu/bad.log" ] || { echo "clear_run_outputs kept a failed row" >&2; exit 1; }
    [ -z "$(existing_attachments)" ] || { echo "existing_attachments listed missing files" >&2; exit 1; }
    touch "$RUN_DIR/report.md"
    [ "$(existing_attachments | tr '\n' ' ')" = "--attach $RUN_DIR/report.md " ] || { echo "existing_attachments: $(existing_attachments)" >&2; exit 1; }
  ) || fail "weekly.sh helpers"
  ok "weekly.sh clears stale outputs, drops failed rows for a rerun and attaches only existing files"
}

check_info_scripts() {
  bash "$HERE/machine-info.sh" > "$TMP/machine.json" || fail "machine-info.sh failed"
  [ "$(jq -r .hostname "$TMP/machine.json")" = "$(hostname -s)" ] || fail "machine.json hostname"
  jq -e '.tools.cmake != null and .gpu != null' "$TMP/machine.json" > /dev/null || fail "machine.json shape"
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins.json" 2> "$TMP/pins.err" || fail "collect-pins.sh failed"
  [ ! -s "$TMP/pins.err" ] || fail "collect-pins.sh wrote errors: $(cat "$TMP/pins.err")"
  [ "$(jq -r .engine.sha "$TMP/pins.json")" = "$(git -C "$ENGINE_SRC" rev-parse HEAD)" ] || fail "pins engine sha"
  [ "$(jq -r '.models[] | select(.name == "jfk-24k.wav") | .present' "$TMP/pins.json")" = "true" ] || fail "pins model entry"
  jq -e '.models[] | select(.name == "jfk-vocoder.mlmodelc")' "$TMP/pins.json" > /dev/null || fail "pins list the sidecar entry"
  ok "machine-info.sh and collect-pins.sh emit the expected JSON"
}

check_nvcc_release() {
  mkdir -p "$TMP/cuda/bin"
  printf '#!/bin/sh\nprintf "nvcc: NVIDIA (R) Cuda compiler driver\\nCuda compilation tools, release 13.3, V13.3.48\\n"\n' > "$TMP/cuda/bin/nvcc"
  chmod +x "$TMP/cuda/bin/nvcc"
  CUDA_HOME="$TMP/cuda" bash "$HERE/machine-info.sh" > "$TMP/machine-cuda.json" || fail "machine-info.sh with a stub nvcc"
  [ "$(jq -r .tools.nvcc "$TMP/machine-cuda.json")" = "13.3" ] || fail "nvcc release parse: $(jq -r .tools.nvcc "$TMP/machine-cuda.json")"
  BACKENDS=cpu FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins-cpu.json" || fail "collect-pins.sh without cuda"
  [ "$(jq -r .build.cuda_arch "$TMP/pins-cpu.json")" = "" ] || fail "cuda_arch recorded without a cuda backend"
  BACKENDS=cpu,cuda FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins-cuda.json" || fail "collect-pins.sh with cuda"
  [ "$(jq -r .build.cuda_arch "$TMP/pins-cuda.json")" = "120a-real" ] || fail "cuda_arch missing with a cuda backend"
  ok "nvcc release and CUDA arch pins follow the toolchain and the backends"
}

check_input_validation() {
  if bash "$HERE/weekly.sh" --label 'x /' > /dev/null 2>&1; then fail "weekly.sh accepted an unsafe label"; fi
  if BENCH_HOME="$TMP/install" bash "$HERE/install.sh" --no-cron --hour 99 > /dev/null 2>&1; then fail "install.sh accepted hour 99"; fi
  if BENCH_HOME="$TMP/bad home" bash "$HERE/install.sh" --no-cron > /dev/null 2>&1; then fail "install.sh accepted a bench home with a space"; fi
  printf 'secret\n' > "$TMP/secret-open"
  chmod 644 "$TMP/secret-open"
  printf 'secret\n' > "$TMP/secret-private"
  chmod 600 "$TMP/secret-private"
  [ -z "$(read_secret "$TMP/secret-open" 2>/dev/null)" ] || fail "read_secret returned a world-readable secret"
  [ "$(read_secret "$TMP/secret-private" 2>/dev/null)" = "secret" ] || fail "read_secret rejected a private secret"
  [ "$(public_repo_url "https://x-access-token:abc@github.com/o/r.git")" = "https://github.com/o/r.git" ] || fail "public_repo_url kept the credential"
  [ "$(public_repo_url "git@github.com:o/r.git")" = "git@github.com:o/r.git" ] || fail "public_repo_url changed an ssh url"
  ok "unsafe labels, schedules, bench homes, open secret files and repo credentials are rejected or stripped"
}

check_entry_points() {
  bash "$HERE/weekly.sh" --help > /dev/null 2>&1 || fail "weekly.sh --help"
  bash "$HERE/install.sh" --help > /dev/null 2>&1 || fail "install.sh --help"
  BENCH_HOME="$TMP/install" bash "$HERE/install.sh" --no-cron > /dev/null || fail "install.sh --no-cron"
  mkdir -p "$TMP/stubcron"
  printf '#!/bin/sh\nexit 1\n' > "$TMP/stubcron/crontab"
  chmod +x "$TMP/stubcron/crontab"
  PATH="$TMP/stubcron:$PATH" BENCH_HOME="$TMP/install" bash "$HERE/install.sh" --hour 18 > "$TMP/install-fallback.log" 2>&1 || fail "install.sh with a refused crontab"
  grep -q "scheduler.sh (pid" "$TMP/install-fallback.log" || fail "install.sh did not fall back to the scheduler: $(cat "$TMP/install-fallback.log")"
  kill "$(cat "$TMP/install/scheduler.pid")" 2>/dev/null || fail "scheduler pid file missing after the fallback"
  [ -x "$TMP/install/tools/weekly.sh" ] || fail "install.sh did not copy the tools"
  [ -f "$TMP/install/config.env" ] || fail "install.sh did not seed config.env"
  [ "$(ls -l "$TMP/install/config.env" | cut -c5-10)" = "------" ] || fail "config.env is not private"
  BENCH_HOME="$TMP/install" bash "$TMP/install/tools/install.sh" --no-cron > "$TMP/reinstall.log" 2>&1 || fail "install.sh from the installed tools failed"
  ! grep -q "identical" "$TMP/reinstall.log" || fail "install.sh copied the tools onto themselves"
  ok "entry points answer --help and install.sh deploys without cron"
}

check_python_tests() {
  local test
  for test in "$HERE"/test_*.py; do python3 -B "$test" > "$TMP/pytest.log" 2>&1 || fail "$(basename "$test"): $(tail -20 "$TMP/pytest.log")"; done
  ok "python unit tests pass"
}

main() {
  check_scripts_parse
  check_spec_fields
  check_args_are_provisioned
  check_derived_sources
  check_huggingface_pins
  setup_home
  check_lib_backends
  check_private_tool_dirs
  check_lib_iterations
  check_lib_targets
  check_lib_lock
  check_run_lock
  check_machine_quirks
  check_page_cache_eviction
  check_skip_families
  check_skip_cells
  check_family_backends
  check_engine_platform_options
  check_lib_deadline
  check_failed_result
  check_previous_and_prune
  check_default_source_layout
  check_ggml_source
  check_gpu_lease
  check_build_marker
  check_fetch_models_offline
  check_fetch_models_reprovisions_on_origin_change
  check_cell_signature
  check_resume_keeps_load_samples
  check_weekly_helpers
  check_info_scripts
  check_nvcc_release
  check_input_validation
  check_entry_points
  check_fetch_registry_discards_partial
  check_python_tests
  echo "all $PASS checks passed"
}

main "$@"
