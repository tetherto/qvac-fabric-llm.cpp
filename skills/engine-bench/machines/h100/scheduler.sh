#!/bin/bash
# User-space stand-in for cron on cosmicac-a26cbb09, a docker container with no cron daemon (and no root to add one).
# Runs weekly.sh at 00:00 every Saturday Europe/Zurich, DST included, and appends its output to cron.log. Start it
# detached:  setsid nohup ~/bench-weekly/scheduler.sh >> ~/bench-weekly/cron.log 2>&1 < /dev/null &
# It does not survive a container restart; status.sh reports whether it is running, and the publish routine restarts it.
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=$HOME/.local/bin:$W/venvs/tools/bin:/usr/local/bin:/usr/bin:/bin
# Zurich time as a POSIX rule: the container has no tzdata, and TZ=Europe/Zurich silently falls back to UTC there
ZURICH="CET-1CEST,M3.5.0,M10.5.0/3"
exec 8> $W/.scheduler.lock
flock -n 8 || { echo "$(date): another scheduler.sh is running"; exit 1; }
echo $$ > $W/scheduler.pid
echo "$(date): scheduler started, pid $$"
while true; do
  next=$(TZ=$ZURICH date -d "next saturday 00:00" +%s)
  echo "next run $(TZ=$ZURICH date -d @$next '+%F %T %Z') ($(date -d @$next -u '+%F %T UTC'))" > $W/scheduler.next
  # short sleeps, so a suspended or clock-adjusted machine still starts on time
  while [ $(date +%s) -lt $next ]; do sleep 60; done
  echo "$(date): starting weekly.sh"
  /bin/bash $W/weekly.sh
  echo "$(date): weekly.sh exited with $?"
  sleep 120
done
