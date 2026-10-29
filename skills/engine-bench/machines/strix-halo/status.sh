#!/bin/bash
# One-screen status of the weekly bench on this machine: scheduler, lock, latest runs.
W=${BENCH_DIR:-$HOME/bench-weekly}
p=$(cat $W/scheduler.pid 2>/dev/null)
if [ -n "$p" ] && kill -0 $p 2>/dev/null; then echo "scheduler: running (pid $p), $(cat $W/scheduler.next 2>/dev/null)"; else echo "scheduler: NOT running"; fi
L=${GPU_LOCK:-/tmp/gpu-bench.lock}; [ -e $L ] && echo "lock: $(cat $L)" || echo "lock: free"
for d in $(ls -d $W/runs/20* 2>/dev/null | tail -3); do
  s=running; [ -e $d/DONE ] && s=DONE; [ -e $d/FAILED ] && s="FAILED: $(cat $d/FAILED)"
  echo "$(basename $d): $s"
done
