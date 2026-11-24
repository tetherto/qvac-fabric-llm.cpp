#!/usr/bin/env bash
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

QUICK=0
UPDATE=1
MAIL=1
LABEL=""
FAMILIES_ARG=""
BACKENDS_ARG=""

usage() {
  cat <<USAGE >&2
usage: weekly.sh [--quick] [--families a,b] [--backends a,b] [--no-update] [--no-mail] [--label NAME]
  --quick       run the QUICK_FAMILIES subset with one timed iteration per family
  --families    comma-separated family names from families.json (default: all)
  --backends    comma-separated backends (default: cpu,cuda,vulkan on Linux; cpu,metal on macOS)
  --no-update   keep the current engine and ggml checkouts
  --no-mail     write the report but do not e-mail it
  --label NAME  suffix for the run directory (runs/<date>-NAME)
USAGE
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --quick) QUICK=1; shift ;;
      --families) FAMILIES_ARG="$2"; shift 2 ;;
      --backends) BACKENDS_ARG="$2"; shift 2 ;;
      --no-update) UPDATE=0; shift ;;
      --no-mail) MAIL=0; shift ;;
      --label) valid_label "$2" || { usage; exit 2; }; LABEL="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) usage; exit 2 ;;
    esac
  done
}

compute_run_name() {
  local name
  name="$(date +%F)"
  if [ "$QUICK" = 1 ]; then name="$name-quick"; fi
  if [ -n "$LABEL" ]; then name="$name-$LABEL"; fi
  echo "$name"
}

run_name() { echo "$RUN_NAME"; }

requested_families() {
  if [ -n "$FAMILIES_ARG" ]; then echo "$FAMILIES_ARG"
  elif [ "$QUICK" = 1 ]; then echo "$QUICK_FAMILIES"
  else all_families | tr '\n' ',' | sed 's/,$//'
  fi
}

family_skipped() {
  case ",$SKIP_FAMILIES," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

family_platform() { family_field "$1" platform; }

host_platform() { echo "${BENCH_TEST_PLATFORM:-$OS_NAME}"; }

family_off_platform() {
  local platform; platform="$(family_platform "$1")"
  [ -n "$platform" ] && [ "$platform" != "$(host_platform)" ]
}

select_family() {
  if family_skipped "$1"; then log "skip $1 (SKIP_FAMILIES)" >&2
  elif family_off_platform "$1"; then log "skip $1 (platform $(family_platform "$1"))" >&2
  else echo "$1"; fi
}

selected_families() {
  local family
  for family in $(split_list "$(requested_families)"); do select_family "$family"; done | tr '\n' ',' | sed 's/,$//'
}

apply_quick_iterations() {
  if [ "$QUICK" = 1 ]; then
    FAMILIES_JSON_QUICK="$RUN_DIR/families-quick.json"
    jq 'with_entries(if (.key | startswith("_")) then . else .value += {runs: 1, warmup: 0, cpu_runs: 1, cpu_warmup: 0} end)' \
      "$FAMILIES_JSON" > "$FAMILIES_JSON_QUICK"
    FAMILIES_JSON="$FAMILIES_JSON_QUICK"
  fi
  export FAMILIES_JSON
}

existing_attachments() {
  local file
  for file in "$RUN_DIR/report.md" "$RUN_DIR/summary.json"; do [ -f "$file" ] && printf -- '--attach\n%s\n' "$file"; done
}

mail_report() {
  [ "$MAIL" = 1 ] || return 0
  SMTP_PASSWORD="$(read_secret "${SMTP_PASSWORD_FILE:-}")" python3 "$TOOLS_DIR/send-report.py" \
    --subject "$1" --body-file "$2" $(existing_attachments) || log "mail FAILED"
}

fail() {
  log "FAILED: $1"
  echo "$1" > "$RUN_DIR/FAILED"
  printf '%s\n\nlog: %s\n' "$1" "$RUN_DIR/weekly.log" > "$RUN_DIR/failure.txt"
  mail_report "[speech-bench] $(hostname -s) $(run_name): FAILED" "$RUN_DIR/failure.txt"
  exit 1
}

run_under_gpu_lease() {
  exec "$GPU_LEASE_TOOL" -n 1 --timeout "$(gpu_lease_timeout_seconds)" -- /bin/bash "${BASH_SOURCE[0]}" "$@"
}

write_narrative() {
  ANTHROPIC_API_KEY="$(read_secret "${ANTHROPIC_API_KEY_FILE:-}")" python3 "$TOOLS_DIR/narrate.py" \
    --report "$RUN_DIR/report.md" --out "$RUN_DIR/narrative.md" || log "narrative skipped"
}

compose_mail_body() {
  if [ -s "$RUN_DIR/narrative.md" ]; then cat "$RUN_DIR/narrative.md"; echo; echo "---"; echo; fi
  cat "$RUN_DIR/report.md"
}

summary_subject() {
  jq -r --arg host "$(hostname -s)" --arg run "$(run_name)" \
    '"[speech-bench] \($host) \($run): \(.run.status) (\(.counts.ok)/\(.counts.expected) ok)"' "$RUN_DIR/summary.json"
}

build_report() {
  local previous
  previous="$(previous_summary "$RUN_DIR")"
  python3 "$TOOLS_DIR/summarize.py" --run-dir "$RUN_DIR" --families "$FAMILIES_JSON" --started "$STARTED" \
    --backends "$(resolve_backends)" --families-selected "$FAMILIES" --skip-cells "$(skipped_cells)" || fail "summarize.py failed"
  python3 "$TOOLS_DIR/report.py" "$RUN_DIR/summary.json" ${previous:+--previous "$previous"} -o "$RUN_DIR/report.md" ||
    fail "report.py failed"
}

clear_run_outputs() {
  local name
  for name in DONE FAILED build-failures.txt summary.json report.md narrative.md mail-body.md failure.txt; do
    rm -f "${RUN_DIR:?}/$name"
  done
  discard_failed_results "$RUN_DIR"
}

prepare_run_dir() {
  RUN_NAME="$(compute_run_name)"
  RUN_DIR="$RUNS_DIR/$RUN_NAME"
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results"
  clear_run_outputs
  exec >> "$RUN_DIR/weekly.log" 2>&1
}

main() {
  parse_args "$@"
  load_config
  setup_path
  if [ -n "$BACKENDS_ARG" ]; then BACKENDS="$BACKENDS_ARG"; fi
  if needs_gpu_lease; then run_under_gpu_lease "$@"; fi
  prepare_run_dir
  STARTED="$(utc_now)"
  log "start $(run_name) backends=$(resolve_backends)"
  take_run_lock || fail "another weekly.sh run holds $(run_lock_dir) ($(run_lock_holder))"
  keep_awake
  if [ "$UPDATE" = 1 ]; then update_sources || fail "source update failed"; fi
  [ -d "$ENGINE_SRC/.git" ] || fail "no engine checkout at $ENGINE_SRC"
  FAMILIES="$(selected_families)"
  export SELECTED_FAMILIES="$FAMILIES"
  apply_quick_iterations
  bash "$TOOLS_DIR/fetch-models.sh" "$FAMILIES" > "$RUN_DIR/logs/fetch-models.log" 2>&1 || log "fetch-models.sh reported errors"
  bash "$TOOLS_DIR/machine-info.sh" > "$RUN_DIR/machine.json"
  acquire_gpu_lock || fail "GPU busy for $LOCK_WAIT_HOURS hours"
  bench_all_backends "$FAMILIES" "$RUN_DIR"
  release_gpu_lock
  bash "$TOOLS_DIR/collect-pins.sh" "$FAMILIES" > "$RUN_DIR/pins.json"
  release_run_lock
  build_report
  write_narrative
  compose_mail_body > "$RUN_DIR/mail-body.md"
  mail_report "$(summary_subject)" "$RUN_DIR/mail-body.md"
  touch "$RUN_DIR/DONE"
  prune_old_runs
  log "end $(run_name)"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
