#!/usr/bin/env bash

[ -n "${SPEECH_BENCH_LIB_LOADED:-}" ] && return 0
SPEECH_BENCH_LIB_LOADED=1

readonly TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OS_NAME="$(uname -s)"
readonly DEFAULT_LINUX_BACKENDS="cpu,cuda,vulkan"
readonly DEFAULT_DARWIN_BACKENDS="cpu,metal"
readonly DEFAULT_QUICK_FAMILIES="parakeet-tdt,supertonic3,audio8-lm-q8_0-dec-q8_0"
readonly DEFAULT_LOCK_WAIT_HOURS=12
readonly DEFAULT_KEEP_RUNS=12
readonly LOCK_POLL_SECONDS="${LOCK_POLL_SECONDS:-60}"
readonly LOCK_CONTENT_LIMIT=200
readonly DEADLINE_POLL_SECONDS="${DEADLINE_POLL_SECONDS:-5}"
readonly KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-15}"
readonly GPU_SAMPLE_MS=250
readonly HOST_LOAD_WINDOW_SECONDS=1
readonly GPU_LEASE_TOOL="gpu-run"
readonly LEASED_GPU_INDEX=0
readonly ENGINE_GGML_PIN_FILE="tts/scripts/setup-ggml.sh"
readonly HOST_LOAD_SUFFIX=".load.json"
readonly MEMORY_SAMPLE_SECONDS=0.25
readonly KIB_PER_MIB=1024
readonly NANOSECOND_SHIM="nsdate"
readonly SECONDS_PER_MINUTE=60
readonly SECONDS_PER_HOUR=3600
readonly DEADLINE_EXIT_CODE=124
readonly DARWIN_BUSY_PATTERN='test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64'
readonly EXTRA_TARGETS="whisper-cli"

BENCH_HOME="${BENCH_HOME:-$HOME/speech-bench}"
CONFIG_FILE="${CONFIG_FILE:-$BENCH_HOME/config.env}"

load_config() {
  local backends_override="${BACKENDS:-}"
  if [ -f "$CONFIG_FILE" ]; then
    set -a
    . "$CONFIG_FILE"
    set +a
  fi
  if [ -n "$backends_override" ]; then BACKENDS="$backends_override"; fi
  MODELS_DIR="${MODELS_DIR:-$BENCH_HOME/models}"
  GPU_LOCK="${GPU_LOCK:-/tmp/gpu-bench.lock}"
  LOCK_WAIT_HOURS="${LOCK_WAIT_HOURS:-$DEFAULT_LOCK_WAIT_HOURS}"
  ENGINE_REPO="${ENGINE_REPO:-https://github.com/tetherto/qvac-fabric-speech.cpp.git}"
  ENGINE_BRANCH="${ENGINE_BRANCH:-master}"
  GGML_REPO="${GGML_REPO:-https://github.com/tetherto/qvac-fabric-llm.cpp.git}"
  GGML_BRANCH="${GGML_BRANCH:-}"
  BACKENDS="${BACKENDS:-auto}"
  CUDA_ARCH="${CUDA_ARCH:-120a-real}"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  GPU_INDEX="${GPU_INDEX:-0}"
  KEEP_RUNS="${KEEP_RUNS:-$DEFAULT_KEEP_RUNS}"
  QUICK_FAMILIES="${QUICK_FAMILIES:-$DEFAULT_QUICK_FAMILIES}"
  SKIP_FAMILIES="${SKIP_FAMILIES:-}"
  MEMORY_SAMPLER="${MEMORY_SAMPLER:-auto}"
  EVICT_PAGE_CACHE="${EVICT_PAGE_CACHE:-0}"
  CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  COREML_PYTHON="${COREML_PYTHON:-}"
  FAMILIES_JSON="${FAMILIES_JSON:-$TOOLS_DIR/families.json}"
  SRC_DIR="$BENCH_HOME/src"
  BUILDS_DIR="$BENCH_HOME/builds"
  RUNS_DIR="$BENCH_HOME/runs"
  ENGINE_SRC="$SRC_DIR/speech"
  GGML_SRC="$SRC_DIR/ggml"
  REGISTRY_CLIENT_DIR="$BENCH_HOME/registry-client"
  MODELS_VENV="$BENCH_HOME/venv"
  adopt_gpu_lease
  export BENCH_HOME CONFIG_FILE MODELS_DIR FAMILIES_JSON BACKENDS
}

inside_gpu_lease() { [ -n "${GPU_RUN_LEASE:-}" ]; }

adopt_gpu_lease() {
  if inside_gpu_lease; then
    GPU_INDEX="$LEASED_GPU_INDEX"
    export GPU_LOCK_QUIET=1
  fi
}

leased_gpu_uuid() {
  case "${CUDA_VISIBLE_DEVICES:-}" in GPU-*) echo "${CUDA_VISIBLE_DEVICES%%,*}" ;; *) echo "" ;; esac
}

gpu_monitor_target() {
  local leased=""
  if inside_gpu_lease; then leased="$(leased_gpu_uuid)"; fi
  echo "${leased:-$GPU_INDEX}"
}

gpu_physical_index() {
  local leased=""
  if inside_gpu_lease; then leased="${GPU_RUN_GPUS:-}"; fi
  echo "${leased:-$GPU_INDEX}"
}

gpu_backends_requested() {
  case ",$(resolve_backends)," in *",cuda,"*|*",vulkan,"*) return 0 ;; *) return 1 ;; esac
}

needs_gpu_lease() {
  ! inside_gpu_lease && command -v "$GPU_LEASE_TOOL" > /dev/null 2>&1 && gpu_backends_requested
}

gpu_lease_timeout_seconds() { echo $(( LOCK_WAIT_HOURS * SECONDS_PER_HOUR )); }

export_gpu_selection() {
  if inside_gpu_lease; then unset GGML_VK_VISIBLE_DEVICES; return 0; fi
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX" GGML_VK_VISIBLE_DEVICES="$GPU_INDEX"
}

setup_path() {
  if [ "$OS_NAME" = "Darwin" ]; then
    export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  else
    export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$CUDA_HOME/bin"
  fi
}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

utc_now() { date -u +%FT%TZ; }

epoch_now() { date +%s; }

cpu_count() {
  if [ "$OS_NAME" = "Darwin" ]; then sysctl -n hw.ncpu; else nproc; fi
}

build_jobs() { echo "${BUILD_JOBS:-$(cpu_count)}"; }

default_backends() {
  if [ "$OS_NAME" = "Darwin" ]; then echo "$DEFAULT_DARWIN_BACKENDS"; else echo "$DEFAULT_LINUX_BACKENDS"; fi
}

resolve_backends() {
  if [ "$BACKENDS" = "auto" ]; then default_backends; else echo "$BACKENDS"; fi
}

split_list() { echo "$1" | tr ',' '\n' | sed '/^$/d'; }

all_families() {
  jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key' "$FAMILIES_JSON"
}

family_field() {
  jq -r --arg f "$1" --arg k "$2" '.[$f][$k] // empty' "$FAMILIES_JSON"
}

family_runs() {
  local family="$1" backend="$2"
  if [ "$backend" = "cpu" ]; then family_field "$family" cpu_runs; else family_field "$family" runs; fi
}

family_warmup() {
  local family="$1" backend="$2"
  if [ "$backend" = "cpu" ]; then family_field "$family" cpu_warmup; else family_field "$family" warmup; fi
}

cmake_targets() {
  local family
  { for family in $(split_list "${SELECTED_FAMILIES:-$(all_families | tr '\n' ',')}"); do family_field "$family" cmake_target; done
    echo "$EXTRA_TARGETS"; } | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' '
}

cmake_generator() {
  if command -v ninja > /dev/null; then echo "Ninja"; else echo "Unix Makefiles"; fi
}

gpu_busy_linux() {
  command -v nvidia-smi > /dev/null || return 1
  [ -n "$(nvidia-smi -i "$(gpu_monitor_target)" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

gpu_busy_darwin() {
  ps -Ao command | grep -E "$DARWIN_BUSY_PATTERN" | grep -v grep > /dev/null
}

gpu_busy() {
  if [ "$OS_NAME" = "Darwin" ]; then gpu_busy_darwin; else gpu_busy_linux; fi
}

lock_held_elsewhere() { [ -e "$GPU_LOCK" ] || [ -L "$GPU_LOCK" ] || gpu_busy; }

lock_holder() {
  if [ -f "$GPU_LOCK" ] && [ ! -L "$GPU_LOCK" ]; then head -c "$LOCK_CONTENT_LIMIT" "$GPU_LOCK"; fi
}

wait_for_gpu() {
  local deadline
  deadline=$(( $(epoch_now) + LOCK_WAIT_HOURS * SECONDS_PER_HOUR ))
  while lock_held_elsewhere; do
    if [ "$(epoch_now)" -ge "$deadline" ]; then
      log "GPU busy for $LOCK_WAIT_HOURS hours ($(lock_holder))"
      return 1
    fi
    sleep "$LOCK_POLL_SECONDS"
  done
}

GPU_LOCK_HELD=0
RUN_LOCK_HELD=0

release_gpu_lock() {
  if [ "$GPU_LOCK_HELD" = 1 ]; then rm -f "${GPU_LOCK:?}"; GPU_LOCK_HELD=0; fi
}

run_lock_dir() { echo "${BENCH_HOME:?}/run.lock"; }

run_lock_holder() { cat "$(run_lock_dir)/owner" 2>/dev/null; }

release_run_lock() {
  if [ "$RUN_LOCK_HELD" = 1 ]; then rm -rf "${BENCH_HOME:?}/run.lock"; RUN_LOCK_HELD=0; fi
}

release_locks() {
  release_gpu_lock
  release_run_lock
}

run_lock_is_stale() {
  local pid
  pid="$(run_lock_holder | cut -d' ' -f1)"
  [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null
}

try_take_run_lock() {
  mkdir "$(run_lock_dir)" 2>/dev/null && echo "$$ $(utc_now)" > "$(run_lock_dir)/owner"
}

take_run_lock() {
  if ! try_take_run_lock; then
    run_lock_is_stale || return 1
    rm -rf "${BENCH_HOME:?}/run.lock"
    try_take_run_lock || return 1
  fi
  RUN_LOCK_HELD=1
  trap release_locks EXIT
}

current_user() { echo "${USER:-${LOGNAME:-$(id -un)}}"; }

try_take_gpu_lock() {
  ( set -o noclobber; echo "$(current_user) speech weekly bench since $(utc_now)" > "$GPU_LOCK" ) 2>/dev/null
}

take_gpu_lock() {
  try_take_gpu_lock || return 1
  GPU_LOCK_HELD=1
  trap release_locks EXIT
}

acquire_gpu_lock() {
  local deadline
  deadline=$(( $(epoch_now) + LOCK_WAIT_HOURS * SECONDS_PER_HOUR ))
  while true; do
    wait_for_gpu || return 1
    if take_gpu_lock; then return 0; fi
    if [ "$(epoch_now)" -ge "$deadline" ]; then log "could not create $GPU_LOCK"; return 1; fi
    sleep "$LOCK_POLL_SECONDS"
  done
}

keep_awake() {
  if [ "$OS_NAME" = "Darwin" ] && command -v caffeinate > /dev/null; then caffeinate -i -w $$ & fi
}

repo_head_line() {
  git -C "$1" log -1 --format='%H %cs'
}

repo_sha() { repo_head_line "$1" | cut -d' ' -f1; }

fetch_ref() {
  local dir="$1" ref="$2"
  git -C "$dir" fetch -q --depth 1 origin "$ref" && git -C "$dir" reset -q --hard FETCH_HEAD
}

init_repo() {
  local url="$1" dir="$2"
  git init -q "$dir" && git -C "$dir" remote add origin "$url"
}

update_repo() {
  local url="$1" branch="$2" dir="$3" pin="$4"
  if [ ! -d "$dir/.git" ]; then init_repo "$url" "$dir" || return 1; fi
  git -C "$dir" remote set-url origin "$url" && fetch_ref "$dir" "${pin:-$branch}"
}

engine_ggml_pin() {
  sed -n 's/^GGML_REF="\([0-9a-f]\{40\}\)".*/\1/p' "$ENGINE_SRC/$ENGINE_GGML_PIN_FILE" 2>/dev/null | head -n 1
}

ggml_ref() {
  if [ -n "${GGML_PIN:-}" ]; then echo "$GGML_PIN"
  elif [ -n "$GGML_BRANCH" ]; then echo "$GGML_BRANCH"
  else engine_ggml_pin; fi
}

ggml_ref_source() {
  if [ -n "${GGML_PIN:-}" ]; then echo "GGML_PIN"
  elif [ -n "$GGML_BRANCH" ]; then echo "branch $GGML_BRANCH"
  else echo "the engine's pin in $ENGINE_GGML_PIN_FILE"; fi
}

update_ggml() {
  local ref
  ref="$(ggml_ref)"
  if [ -z "$ref" ]; then log "no ggml commit: $ENGINE_GGML_PIN_FILE of the engine has no GGML_REF and neither GGML_PIN nor GGML_BRANCH is set"; return 1; fi
  update_repo "$GGML_REPO" "" "$GGML_SRC" "$ref"
}

ggml_cmake_source() {
  if [ -f "$GGML_SRC/ggml/CMakeLists.txt" ]; then echo "$GGML_SRC/ggml"; else echo "$GGML_SRC"; fi
}

date_has_nanoseconds() { [[ "$(/bin/date +%N 2>/dev/null)" =~ ^[0-9]+$ ]]; }

install_nanosecond_shim() {
  mkdir -p "$BENCH_HOME/bin"
  install -m 755 "$TOOLS_DIR/$NANOSECOND_SHIM" "$BENCH_HOME/bin/$NANOSECOND_SHIM"
  perl -pi -e "s{/bin/date }{$BENCH_HOME/bin/$NANOSECOND_SHIM }g" "$ENGINE_SRC/scripts/benchmarks/run-family.sh"
}

apply_clock_shim() {
  if date_has_nanoseconds; then return 0; fi
  log "system date has no nanoseconds; routing run-family.sh through $NANOSECOND_SHIM"
  install_nanosecond_shim
}

update_sources() {
  mkdir -p "$SRC_DIR"
  update_repo "$ENGINE_REPO" "$ENGINE_BRANCH" "$ENGINE_SRC" "${ENGINE_PIN:-}" || return 1
  update_ggml || return 1
  apply_clock_shim
  log "engine $(repo_head_line "$ENGINE_SRC") ggml $(repo_head_line "$GGML_SRC") from $(ggml_ref_source)"
}

ggml_backend_options() {
  case "$1" in
    cpu) echo "-DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF" ;;
    cuda) echo "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DGGML_VULKAN=OFF -DGGML_METAL=OFF" ;;
    vulkan) echo "-DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF" ;;
    metal) echo "-DGGML_METAL=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF" ;;
    *) return 1 ;;
  esac
}

build_signature() {
  echo "engine=$(repo_sha "$ENGINE_SRC") ggml=$(repo_sha "$GGML_SRC") backend=$1 options=$(ggml_backend_options "$1") engine_options=$(engine_platform_options "$OS_NAME") prefix=$CMAKE_PREFIX_PATH targets=$(cmake_targets)"
}

build_marker() { echo "${BUILDS_DIR:?}/${1:?}/BUILT"; }

build_is_current() {
  local marker
  marker="$(build_marker "$1")"
  [ -f "$marker" ] && [ "$(cat "$marker")" = "$(build_signature "$1")" ]
}

cmake_prefix_path() {
  local own="${1:-}"
  echo "$own${own:+${CMAKE_PREFIX_PATH:+;}}$CMAKE_PREFIX_PATH"
}

cached_cmake_source() { sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$1/CMakeCache.txt" 2>/dev/null; }

discard_ggml_build_of_another_source() {
  local build="${BUILDS_DIR:?}/${1:?}/ggml" cached
  cached="$(cached_cmake_source "$build")"
  if [ -n "$cached" ] && [ "$cached" != "$(ggml_cmake_source)" ]; then rm -rf "$build"; fi
}

configure_ggml() {
  local backend="$1" dir="$BUILDS_DIR/$1" log_dir="$2"
  discard_ggml_build_of_another_source "$backend"
  cmake -S "$(ggml_cmake_source)" -B "$dir/ggml" -G "$(cmake_generator)" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF \
    -DCMAKE_INSTALL_PREFIX="$dir/ggml-install" -DCMAKE_PREFIX_PATH="$(cmake_prefix_path)" $(ggml_backend_options "$backend") \
    > "$log_dir/ggml-configure.log" 2>&1
}

build_ggml() {
  local dir="${BUILDS_DIR:?}/${1:?}" log_dir="$2"
  cmake --build "$dir/ggml" -j"$(build_jobs)" > "$log_dir/ggml-build.log" 2>&1 &&
    rm -rf "${dir:?}/ggml-install" &&
    cmake --install "$dir/ggml" > "$log_dir/ggml-install.log" 2>&1
}

engine_platform_options() {
  case "$1" in
    Darwin) echo "-DPARAKEET_COREML=ON -DTTS_CPP_COREML=ON" ;;
    *) echo "" ;;
  esac
}

configure_engine() {
  local dir="$BUILDS_DIR/$1" log_dir="$2"
  cmake -S "$ENGINE_SRC" -B "$dir/speech" -G "$(cmake_generator)" -DCMAKE_BUILD_TYPE=Release -DSPEECH_BUILD_TESTS=OFF \
    -DCMAKE_PREFIX_PATH="$(cmake_prefix_path "$dir/ggml-install")" $(engine_platform_options "$OS_NAME") > "$log_dir/speech-configure.log" 2>&1
}

build_engine() {
  local dir="$BUILDS_DIR/$1" log_dir="$2"
  cmake --build "$dir/speech" -j"$(build_jobs)" --target $(cmake_targets) > "$log_dir/speech-build.log" 2>&1
}

build_backend() {
  local backend="$1" log_dir="$2/build-$1"
  mkdir -p "$log_dir" "$BUILDS_DIR/$backend"
  if build_is_current "$backend"; then log "build $backend is current"; return 0; fi
  rm -f "$(build_marker "$backend")"
  log "build $backend"
  configure_ggml "$backend" "$log_dir" && build_ggml "$backend" "$log_dir" &&
    configure_engine "$backend" "$log_dir" && build_engine "$backend" "$log_dir" || return 1
  build_signature "$backend" > "$(build_marker "$backend")"
}

export_backend_environment() {
  local lib="$BUILDS_DIR/$1/ggml-install/lib"
  export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH="$lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  export_gpu_selection
  export_coreml_python
}

export_coreml_python() {
  if [ -n "$COREML_PYTHON" ]; then export PARAKEET_COREML_PYTHON="$COREML_PYTHON"; fi
}

kill_process_group() {
  kill -TERM -- "-$1" 2>/dev/null
  sleep "$KILL_GRACE_SECONDS"
  kill -KILL -- "-$1" 2>/dev/null
}

watch_deadline() {
  local pid="$1" limit=$(( $2 * SECONDS_PER_MINUTE )) waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then kill_process_group "$pid"; return 1; fi
    sleep "$DEADLINE_POLL_SECONDS"
    waited=$(( waited + DEADLINE_POLL_SECONDS ))
  done
}

run_with_deadline() {
  local minutes="$1"
  shift
  (
    set -m
    "$@" &
    local pid=$!
    if watch_deadline "$pid" "$minutes"; then wait "$pid"; else wait "$pid" 2>/dev/null; return "$DEADLINE_EXIT_CODE"; fi
  )
}

sample_system_memory() {
  while true; do
    awk -v ts="$(utc_now)" -v kib="$KIB_PER_MIB" \
      '/^MemTotal:/ { t = $2 } /^MemAvailable:/ { a = $2 } END { printf "%s, %d, 0\n", ts, (t - a) / kib }' /proc/meminfo
    sleep "$MEMORY_SAMPLE_SECONDS"
  done
}

memory_sampler_kind() {
  if [ "$MEMORY_SAMPLER" != "auto" ]; then echo "$MEMORY_SAMPLER"; return; fi
  if command -v nvidia-smi > /dev/null; then echo "nvidia-smi"; else echo "none"; fi
}

start_gpu_sampler() {
  GPU_SAMPLER_PID=""
  case "$(memory_sampler_kind)" in
    nvidia-smi)
      nvidia-smi -i "$(gpu_monitor_target)" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
        -lms "$GPU_SAMPLE_MS" > "$1" 2>/dev/null & ;;
    meminfo) sample_system_memory > "$1" & ;;
    *) return 0 ;;
  esac
  GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
  if [ -n "${GPU_SAMPLER_PID:-}" ]; then kill "$GPU_SAMPLER_PID" 2>/dev/null; wait "$GPU_SAMPLER_PID" 2>/dev/null || true; fi
  GPU_SAMPLER_PID=""
  return 0
}

write_failed_result() {
  local family="$1" backend="$2" status="$3" out="$4"
  jq -n --arg family "$family" --arg runner "$(hostname -s)-$backend" --arg os "$OS_NAME" --arg status "$status" \
    --arg notes "the weekly driver wrote this row: $status" \
    '{family:$family, model:$family, runner:$runner, os:$os, backend:"unknown", wall_ms_median:null, wall_ms_min:null,
      wall_ms_max:null, rtf_median:null, peak_rss_mib:null, runs:0, status:$status, notes:$notes}' > "$out"
}

run_family_driver() {
  local family="$1" backend="$2" out="$3" runs warmup
  runs="$(family_runs "$family" "$backend")"
  warmup="$(family_warmup "$family" "$backend")"
  ( cd "$ENGINE_SRC" && BENCH_FAMILIES_JSON="$FAMILIES_JSON" BENCH_HOME="$BENCH_HOME" bash scripts/benchmarks/run-family.sh \
      --family "$family" --runs "$runs" --warmup "$warmup" --build-dir "$BUILDS_DIR/$backend/speech" \
      --models-root "$MODELS_DIR/families" --runner "$(hostname -s)-$backend" --out "$out" )
}

family_spec_signature() {
  jq -c --arg f "$1" '.[$f] | {binary, args, models, provision, correctness, audio_duration_seconds}' "$FAMILIES_JSON"
}

family_model_signature() {
  local link target
  for link in "$MODELS_DIR/families/$1"/*; do
    [ -e "$link" ] || continue
    target="$(readlink "$link" || echo "$link")"
    echo "$(basename "$link")=$(cat "$target.sha256" 2>/dev/null || echo unknown)"
  done | tr '\n' ' '
}

cell_signature() {
  local family="$1" backend="$2"
  echo "$(build_signature "$backend") runs=$(family_runs "$family" "$backend")/$(family_warmup "$family" "$backend") spec=$(family_spec_signature "$family") models=$(family_model_signature "$family")"
}

cell_is_reusable() {
  local out="$1" signature="$2"
  [ -e "$out" ] && [ -f "$out.sig" ] && [ "$(cat "$out.sig")" = "$signature" ]
}

discard_cell() {
  local out="$1"
  rm -f "${out:?}" "${out:?}.sig" "${out%.json}.log" "${out%.json}.gpu.csv" "$(host_load_file "$out")"
}

cpu_ticks() { awk '/^cpu /{total=0; for (i=2; i<=NF; i++) total+=$i; printf "%.0f %.0f\n", total, $5+$6}' /proc/stat; }

busy_percent_between() {
  awk -v a="$1" -v b="$2" 'BEGIN{split(a,x," "); split(b,y," "); dt=y[1]-x[1]; di=y[2]-x[2]; if (dt<=0) print "0.0"; else printf "%.1f\n", 100*(dt-di)/dt}'
}

linux_busy_percent() {
  local before after
  before="$(cpu_ticks)"; sleep "$HOST_LOAD_WINDOW_SECONDS"; after="$(cpu_ticks)"
  busy_percent_between "$before" "$after"
}

busy_percent_from_top() { sed -n 's/.* \([0-9.]*\)% idle.*/\1/p' | awk 'END{if (NR==0) print "0.0"; else printf "%.1f\n", 100-$1}'; }

darwin_busy_percent() { top -l 2 -s "$HOST_LOAD_WINDOW_SECONDS" -n 0 2>/dev/null | grep "CPU usage" | tail -n 1 | busy_percent_from_top; }

host_busy_percent() { case "$OS_NAME" in Darwin) darwin_busy_percent ;; *) linux_busy_percent ;; esac; }

load_average_one() { case "$OS_NAME" in Darwin) sysctl -n vm.loadavg | awk '{print $2}' ;; *) awk '{print $1}' /proc/loadavg ;; esac; }

online_cpus() { getconf _NPROCESSORS_ONLN; }

host_load_file() { echo "${1%.json}$HOST_LOAD_SUFFIX"; }

is_host_load_file() { case "$1" in *"$HOST_LOAD_SUFFIX") return 0 ;; *) return 1 ;; esac; }

write_host_load() {
  jq -n --arg at "$(utc_now)" --argjson busy "$(host_busy_percent)" --argjson load "$(load_average_one)" --argjson cpus "$(online_cpus)" \
    '{sampled_at: $at, busy_before_pct: $busy, load1_before: $load, cpu_count: $cpus}' > "$1"
}

evict_trees() { python3 "$TOOLS_DIR/prepare-derived.py" evict "$@" > /dev/null; }

evict_page_cache() {
  if [ "$EVICT_PAGE_CACHE" = 1 ]; then evict_trees "$MODELS_DIR/store" "$BUILDS_DIR" "$SRC_DIR" "$BENCH_HOME/deps"; fi
}

run_family() {
  local family="$1" backend="$2" results="$3" out="$3/$1.json" status signature
  signature="$(cell_signature "$family" "$backend")"
  if cell_is_reusable "$out" "$signature"; then log "skip $family on $backend (result present with the same inputs)"; return 0; fi
  if [ -e "$out" ]; then log "rerun $family on $backend (inputs changed)"; discard_cell "$out"; fi
  evict_page_cache
  log "run $family on $backend"
  write_host_load "$(host_load_file "$out")"
  start_gpu_sampler "$results/$family.gpu.csv"
  run_with_deadline "$(family_field "$family" timeout_minutes)" run_family_driver "$family" "$backend" "$out" > "$results/$family.log" 2>&1
  status=$?
  stop_gpu_sampler
  if [ "$status" -eq "$DEADLINE_EXIT_CODE" ]; then write_failed_result "$family" "$backend" "timeout" "$out"; fi
  if [ ! -e "$out" ]; then write_failed_result "$family" "$backend" "driver-failed" "$out"; fi
  echo "$signature" > "$out.sig"
  log "done $family on $backend: $(jq -r '[.status, .backend, .wall_ms_median] | map(tostring) | join(" ")' "$out")"
}

family_backends() {
  jq -r --arg f "$1" '.[$f].backends // [] | join(",")' "$FAMILIES_JSON"
}

family_runs_on_backend() {
  local allowed; allowed="$(family_backends "$1")"
  [ -z "$allowed" ] || case ",$allowed," in *",$2,"*) return 0 ;; *) return 1 ;; esac
}

cell_skipped() {
  case ",$SKIP_FAMILIES," in *",$1:$2,"*) return 0 ;; *) return 1 ;; esac
}

skipped_cells() { split_list "$SKIP_FAMILIES" | sed -n '/:/p' | paste -sd, -; }

run_family_if_allowed() {
  if cell_skipped "$1" "$2"; then log "skip $1 on $2 (SKIP_FAMILIES)"
  elif family_runs_on_backend "$1" "$2"; then run_family "$1" "$2" "$3"
  else log "skip $1 on $2 (family backends: $(family_backends "$1"))"; fi
}

run_families_on_backend() {
  local backend="$1" families="$2" results="$3/results/$1" family
  mkdir -p "$results"
  export_backend_environment "$backend"
  for family in $(split_list "$families"); do run_family_if_allowed "$family" "$backend" "$results"; done
}

discard_backend_results() { rm -rf "${1:?}/results/${2:?}"; }

bench_backend() {
  local backend="$1" families="$2" run_dir="$3"
  if ! build_is_current "$backend"; then discard_backend_results "$run_dir" "$backend"; fi
  if ! build_backend "$backend" "$run_dir/logs"; then
    log "build $backend FAILED"
    echo "$backend" >> "$run_dir/build-failures.txt"
    return 1
  fi
  run_families_on_backend "$backend" "$families" "$run_dir"
}

result_failed() { [ "$(jq -r '.status // ""' "$1" 2>/dev/null)" != "ok" ]; }

failed_cell_result() { ! is_host_load_file "$1" && result_failed "$1"; }

discard_failed_results() {
  local result
  for result in "${1:?}"/results/*/*.json; do
    [ -f "$result" ] || continue
    if failed_cell_result "$result"; then discard_cell "$result"; fi
  done
}

bench_all_backends() {
  local families="$1" run_dir="$2" backend
  for backend in $(split_list "$(resolve_backends)"); do bench_backend "$backend" "$families" "$run_dir"; done
}

run_dirs_newest_first() {
  local dirs=("${RUNS_DIR:?}"/*/) index
  for (( index = ${#dirs[@]} - 1; index >= 0; index-- )); do
    [ -d "${dirs[$index]}" ] && echo "${dirs[$index]%/}"
  done
}

weekly_run_dir() { [[ "$(basename "$1")" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; }

finished_runs_newest_first() {
  local dir
  while IFS= read -r dir; do
    if weekly_run_dir "$dir" && [ -f "$dir/summary.json" ] && [ -e "$dir/DONE" ]; then echo "$dir"; fi
  done < <(run_dirs_newest_first)
}

previous_summary() {
  local current="$1" dir
  while IFS= read -r dir; do
    if [ "$dir" != "$current" ]; then echo "$dir/summary.json"; return 0; fi
  done < <(finished_runs_newest_first)
}

prune_old_runs() {
  local dir count=0
  while IFS= read -r dir; do
    count=$(( count + 1 ))
    if [ "$count" -gt "$KEEP_RUNS" ]; then rm -rf "${dir:?}"; fi
  done < <(run_dirs_newest_first)
}

valid_label() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]; }

private_file() { [ -f "$1" ] && [ "$(ls -l "$1" | cut -c5-10)" = "------" ]; }

read_secret() {
  [ -n "$1" ] || return 0
  if private_file "$1"; then cat "$1"; else log "secret file $1 is missing or readable by the group or others; ignoring it" >&2; fi
}

public_repo_url() { echo "$1" | sed -E 's#^([a-z]+://)[^/@]*@#\1#'; }
