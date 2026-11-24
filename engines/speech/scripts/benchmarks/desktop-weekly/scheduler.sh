#!/usr/bin/env bash
set -uo pipefail

readonly POLL_SECONDS=60
readonly BENCH_HOME="${BENCH_HOME:-$HOME/speech-bench}"
readonly WEEKDAY="${1:?weekday 0-6}"
readonly HOUR="${2:?hour}"
readonly MINUTE="${3:?minute}"
readonly STAMP_FILE="$BENCH_HOME/scheduler.last"

due_now() {
  local now_day now_hour now_minute
  now_day="$(date +%w)"
  now_hour="$(date +%-H)"
  now_minute="$(date +%-M)"
  [ "$now_day" = "$WEEKDAY" ] && [ "$now_hour" = "$HOUR" ] && [ "$now_minute" -ge "$MINUTE" ]
}

ran_today() { [ "$(cat "$STAMP_FILE" 2>/dev/null)" = "$(date +%F)" ]; }

run_weekly() {
  date +%F > "$STAMP_FILE"
  /bin/bash "$BENCH_HOME/tools/weekly.sh" >> "$BENCH_HOME/cron.log" 2>&1
}

wait_and_run() {
  while true; do
    if due_now && ! ran_today; then run_weekly; fi
    sleep "$POLL_SECONDS"
  done
}

echo $$ > "$BENCH_HOME/scheduler.pid"
wait_and_run
