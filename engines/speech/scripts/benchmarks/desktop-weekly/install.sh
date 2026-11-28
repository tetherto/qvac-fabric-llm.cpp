#!/usr/bin/env bash
set -uo pipefail

readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CRON_TAG="# speech-bench-weekly"
readonly DEFAULT_WEEKDAY=0
readonly DEFAULT_HOUR=18
readonly DEFAULT_MINUTE=0
readonly MAX_WEEKDAY=6
readonly MAX_HOUR=23
readonly MAX_MINUTE=59
readonly CONFIG_MODE=600

BENCH_HOME="${BENCH_HOME:-$HOME/speech-bench}"
WEEKDAY="$DEFAULT_WEEKDAY"
HOUR="$DEFAULT_HOUR"
MINUTE="$DEFAULT_MINUTE"
CRON=1

usage() {
  cat <<USAGE >&2
usage: install.sh [--bench-home DIR] [--weekday 0-6] [--hour H] [--minute M] [--no-cron]
Copies this directory to <bench home>/tools, seeds <bench home>/config.env from config.example.env when it is
missing, and installs a crontab line that runs weekly.sh on the given weekday (0 = Sunday) at the given machine-local time
(default Sunday 18:00; pass --hour 17 on a UTC machine for 18:00 CET).
USAGE
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --bench-home) BENCH_HOME="$2"; shift 2 ;;
      --weekday) WEEKDAY="$2"; shift 2 ;;
      --hour) HOUR="$2"; shift 2 ;;
      --minute) MINUTE="$2"; shift 2 ;;
      --no-cron) CRON=0; shift ;;
      -h|--help) usage; exit 0 ;;
      *) usage; exit 2 ;;
    esac
  done
}

integer_within() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -le "$2" ]; }

validate_schedule() {
  integer_within "$WEEKDAY" "$MAX_WEEKDAY" || { echo "weekday must be 0-6" >&2; exit 2; }
  integer_within "$HOUR" "$MAX_HOUR" || { echo "hour must be 0-23" >&2; exit 2; }
  integer_within "$MINUTE" "$MAX_MINUTE" || { echo "minute must be 0-59" >&2; exit 2; }
  [[ "$BENCH_HOME" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "bench home may only contain letters, digits, / . _ -" >&2; exit 2; }
}

installing_in_place() {
  mkdir -p "$BENCH_HOME/tools"
  [ "$(cd "$BENCH_HOME/tools" && pwd -P)" = "$(cd "$SOURCE_DIR" && pwd -P)" ]
}

install_tools() {
  mkdir -p "$BENCH_HOME/tools" "$BENCH_HOME/runs"
  if ! installing_in_place; then find "$SOURCE_DIR" -maxdepth 1 -type f -exec cp {} "$BENCH_HOME/tools/" \; ; fi
  chmod +x "$BENCH_HOME"/tools/*.sh
}

seed_config() {
  [ -f "$BENCH_HOME/config.env" ] || cp "$SOURCE_DIR/config.example.env" "$BENCH_HOME/config.env"
  chmod "$CONFIG_MODE" "$BENCH_HOME/config.env"
}

cron_line() {
  echo "$MINUTE $HOUR * * $WEEKDAY BENCH_HOME=$BENCH_HOME /bin/bash $BENCH_HOME/tools/weekly.sh >> $BENCH_HOME/cron.log 2>&1 $CRON_TAG"
}

other_cron_lines() {
  crontab -l 2>/dev/null | grep -v -F "$CRON_TAG" || true
}

scheduler_running() {
  local pid
  pid="$(cat "$BENCH_HOME/scheduler.pid" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stop_scheduler() {
  if scheduler_running; then kill "$(cat "$BENCH_HOME/scheduler.pid")"; fi
  rm -f "$BENCH_HOME/scheduler.pid"
}

start_scheduler() {
  stop_scheduler
  BENCH_HOME="$BENCH_HOME" nohup /bin/bash "$BENCH_HOME/tools/scheduler.sh" "$WEEKDAY" "$HOUR" "$MINUTE" > "$BENCH_HOME/scheduler.log" 2>&1 < /dev/null &
  echo $! > "$BENCH_HOME/scheduler.pid"
  echo "scheduler: no cron on this host; scheduler.sh (pid $!) runs weekly.sh on weekday $WEEKDAY at $HOUR:$MINUTE and does not survive a reboot"
}

crontab_holds_line() {
  crontab -l 2>/dev/null | grep -q -F "$CRON_TAG"
}

write_crontab() {
  { other_cron_lines; cron_line; } | crontab - 2>/dev/null && crontab_holds_line
}

install_cron() {
  [ "$CRON" = 1 ] || return 0
  if command -v crontab > /dev/null && write_crontab; then echo "crontab: $(cron_line)"; return 0; fi
  echo "crontab is missing or refused the write on this host"
  start_scheduler
}

main() {
  parse_args "$@"
  validate_schedule
  install_tools
  seed_config
  install_cron
  echo "installed in $BENCH_HOME (edit $BENCH_HOME/config.env, then try: $BENCH_HOME/tools/weekly.sh --quick --no-mail)"
}

main "$@"
