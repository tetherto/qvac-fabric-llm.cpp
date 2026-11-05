#!/bin/bash
# User-space stand-in for cron on qvac-dev-strix-0, which has no cron daemon and no lingering systemd user manager
# (both need root to set up). Runs weekly.sh at 00:00 every Saturday Europe/Zurich, DST included, and appends its
# output to cron.log. Start it detached:  setsid nohup ~/bench-weekly/scheduler.sh >> ~/bench-weekly/cron.log 2>&1 < /dev/null &
# It does not survive a reboot; status.sh reports whether it is running.
#   scheduler.sh --test-in N   run gpu-check.sh once, N seconds from now, then exit (checks GPU access when detached)
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=$HOME/.local/bin:/opt/rocm-7.2.4/bin:/usr/local/bin:/usr/bin:/bin
if [ "${1:-}" = "--test-in" ]; then
  sleep $2; echo "test $(date)"; $W/gpu-check.sh; exit
fi
exec 8> $W/.scheduler.lock
flock -n 8 || { echo "$(date): another scheduler.sh is running"; exit 1; }
echo $$ > $W/scheduler.pid
echo "$(date): scheduler started, pid $$"
while true; do
  next=$(TZ=Europe/Zurich date -d "next saturday 00:00" +%s)
  echo "next run $(TZ=Europe/Zurich date -d @$next '+%F %T %Z') ($(date -d @$next -u '+%F %T UTC'))" > $W/scheduler.next
  # short sleeps, so a suspended or clock-adjusted machine still starts on time
  while [ $(date +%s) -lt $next ]; do sleep 60; done
  echo "$(date): starting weekly.sh"
  /bin/bash $W/weekly.sh
  echo "$(date): weekly.sh exited with $?"
  sleep 120
done
