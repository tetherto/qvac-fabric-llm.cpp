#!/bin/bash
# Where the weekly TurboVec bench stands on this machine: lock, a run in progress, builds, latest runs, trigger.
T=${TV_DIR:-$HOME/bench-turbovec}
LLM=${BENCH_DIR:-$HOME/bench-weekly}
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
source $T/machine.conf
echo "machine: $MACHINE ($HW)"
echo "lock: $(cat $LOCK 2>/dev/null || echo none)"
if [ -d $T/.run.lock ] && kill -0 "$(cat $T/.run.lock/pid 2>/dev/null)" 2>/dev/null; then
  echo "running: turbovec.sh pid $(cat $T/.run.lock/pid)"
else
  echo "running: no"
fi
sed 's/^/  /' $T/engines.env 2>/dev/null
for d in $(ls -d $T/runs/20??-??-??* 2>/dev/null | tail -4); do
  echo "$(basename $d): $(cd $d && ls DONE FAILED 2>/dev/null | tr '\n' ' ')$([ -e $d/DONE -o -e $d/FAILED ] || echo 'no marker')"
done
echo "trigger (runs after the LLM bench):"
crontab -l 2>/dev/null | grep turbovec | sed 's/^/  /'
grep -n turbovec $LLM/scheduler.sh 2>/dev/null | sed 's/^/  /'
