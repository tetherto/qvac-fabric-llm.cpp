#!/bin/bash
# usage: memguard.sh <pid to follow> <log file>
# Protects the shared Spark while the bench runs: every 5 s, if MemAvailable falls below 6 GiB, stops this user's own
# engine servers and kernel compiles (never anyone else's processes) and logs it. Exits when <pid> exits.
FOLLOW=$1; LOG=$2
LIMIT_KB=$((6 * 1024 * 1024))
OWN_PGID=$(ps -o pgid= -p $FOLLOW | tr -d ' ')
while kill -0 $FOLLOW 2>/dev/null; do
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$avail" -lt $LIMIT_KB ]; then
    echo "$(date) memguard: MemAvailable $((avail / 1024)) MiB, stopping own engines and compiles" >> $LOG
    # the bracket keeps these patterns from matching this script's own command line
    for p in $(pgrep -u $(id -u) -f "sglang[.]launch_server|vllm serve[ ]|llama-server[ ]"); do
      g=$(ps -o pgid= -p $p | tr -d ' ')
      [ -n "$g" ] && [ "$g" != "$OWN_PGID" ] && kill -KILL -- -$g 2>/dev/null
      kill -KILL $p 2>/dev/null
    done
    pkill -KILL -u $(id -u) -x ninja; pkill -KILL -u $(id -u) -x nvcc; pkill -KILL -u $(id -u) -x cicc; pkill -KILL -u $(id -u) -x ptxas
    sleep 30
  fi
  sleep 5
done
