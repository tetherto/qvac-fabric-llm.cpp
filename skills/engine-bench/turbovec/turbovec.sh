#!/bin/bash
# Weekly TurboVec bench on this machine's CPU: qvac-fabric's ggml-vector-index (newest temp-<n> branch) against
# upstream turbovec (newest stable PyPI release), with the scripts in bench/ (dim 1024, 100,000 random unit vectors,
# 1,000 queries, 2 and 4 bit): batch search and cold add on 1 and 8 threads, one query per call (p50/p95, k 10 and 64),
# and recall against an exact float32 search. It runs right after the Saturday LLM engine bench, so the two never share
# the CPU. Writes runs/<date>/summary.json and marks the run with DONE or FAILED.
#   turbovec.sh              full run now
#   turbovec.sh --after-llm  as the weekly trigger calls it after weekly.sh: runs only on a Saturday (Zurich) once this
#                            Saturday's LLM run has a DONE or FAILED marker, and only once per date
#   turbovec.sh --quick      pipeline test: no updates, one sample, 1 thread
# env: TV_DIR (this folder, default ~/bench-turbovec), BENCH_DIR (the LLM bench, default ~/bench-weekly), GPU_LOCK
set -u
T=${TV_DIR:-$HOME/bench-turbovec}
LLM=${BENCH_DIR:-$HOME/bench-weekly}
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
ZURICH=Europe/Zurich
source $T/machine.conf  # MACHINE, HW, BUILD_JOBS, PATH
export TV_DIR=$T
ZDATE() { TZ=$ZURICH date "$@"; }
MODE=full; case "${1:-}" in --after-llm) MODE=after;; --quick) MODE=quick;; esac
DATE=$(ZDATE +%F)

if [ $MODE = after ]; then
  [ "$(ZDATE +%u)" = 6 ] || exit 0
  if [ ! -e $LLM/runs/$DATE/DONE ] && [ ! -e $LLM/runs/$DATE/FAILED ]; then
    echo "$(date): LLM run $DATE has no DONE or FAILED marker yet; TurboVec not started"; exit 0
  fi
  [ -e $T/runs/$DATE ] && exit 0
fi
[ $MODE = quick ] && DATE="$DATE-quick"
# one run at a time (macOS has no flock: a lock directory holding the owner's pid)
if ! mkdir $T/.run.lock 2>/dev/null; then
  pid=$(cat $T/.run.lock/pid 2>/dev/null)
  if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then echo "$(date): another turbovec.sh (pid $pid) is running"; exit 1; fi
  rm -rf $T/.run.lock; mkdir $T/.run.lock || exit 1
fi
echo $$ > $T/.run.lock/pid
HAVE_LOCK=0; LOADPID=
cleanup() { [ $HAVE_LOCK = 1 ] && rm -f $LOCK; [ -n "$LOADPID" ] && kill $LOADPID 2>/dev/null; rm -rf $T/.run.lock; }
trap cleanup EXIT
RUN=$T/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/turbovec.log 2>&1
echo "start $(date) ($MODE)"
rm -f $RUN/DONE $RUN/FAILED $RUN/errors.jsonl $RUN/jobs.jsonl

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; $SUMPY $T/summarize.py $RUN --status failed --reason "$1"; exit 1; }
SUMPY=python3

# wait until the CPU is quiet: no lock (the GPU bench lock, which the LLM bench takes too) and a 1-minute
# load under max(2, cores/8), so other users' jobs do not skew the timings; at most 6 hours
load1() { if [ "$(uname)" = Darwin ]; then sysctl -n vm.loadavg | awk '{print $2}'; else cut -d' ' -f1 /proc/loadavg; fi; }
NCPU=$(getconf _NPROCESSORS_ONLN)
busy() {
  [ -e $LOCK ] && return 0
  awk -v l=$(load1) -v n=$NCPU 'BEGIN { t = n / 8; if (t < 2) t = 2; exit !(l > t) }'
}
deadline=$(( $(date +%s) + 6 * 3600 ))
while busy; do
  [ $(date +%s) -gt $deadline ] && fail "CPU busy for 6 hours (load $(load1), lock: $(cat $LOCK 2>/dev/null || echo none))"
  sleep 60
done
echo "${USER:-$(id -un)} TurboVec bench since $(date +%H:%M) (pid $$)" > $LOCK; HAVE_LOCK=1
echo "lock taken $(date), load $(load1)"
date -u +%FT%TZ > $RUN/started

# the 1-minute load every 30 s, so the summary can show whether anything else ran meanwhile
( while true; do echo "$(date +%s) $(load1)"; sleep 30; done ) > $RUN/load.log 2>/dev/null &
LOADPID=$!
disown $LOADPID 2>/dev/null || true  # so killing it at the end prints no job notice into the log

if [ $MODE != quick ]; then
  $T/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $T/engines.env ] || fail "no engines.env; run update.sh once by hand"
source $T/engines.env
cp $T/engines.env $RUN/engines.env
[ -x "${FAB_BIN:-}/bench_fabric" ] || fail "no qvac-fabric build"
[ -x "${TV_PY:-}" ] || fail "no turbovec venv"
SUMPY=$TV_PY

THREADS=1,8; BS=5; SS=3; [ $MODE = quick ] && { THREADS=1; BS=1; SS=1; }
# a build without the thread/staged API is named "...base" for the drivers: 1-thread rows only, no staged switch
FAB=fabric; [ "$FAB_API" = 1 ] || FAB=fabric-base
RECALL=(--fabric $FAB=$FAB_BIN/bench_fabric); [ "$FAB_API" = 1 ] && RECALL+=(--fabric fabric-staged-off=$FAB_BIN/bench_fabric)
mkdir -p $T/data

job() { # <label> <cmd...>: runs one driver; a failure is recorded in errors.jsonl and the run goes on
  local label=$1; shift
  echo "{\"label\": \"$label\", \"at\": \"$(date -u +%FT%TZ)\"}" >> $RUN/jobs.jsonl
  echo "== $label $(date +%H:%M:%S)"
  if ! "$@" > $RUN/logs/$label.log 2>&1; then
    $TV_PY -c 'import json, sys; print(json.dumps({"label": sys.argv[1], "at": sys.argv[2], "error": sys.argv[3]}))' \
      "$label" "$(date -u +%FT%TZ)" "$(grep -v '^\s*$' $RUN/logs/$label.log | tail -1 | cut -c1-300)" >> $RUN/errors.jsonl
    echo "$label failed: $(tail -1 $RUN/logs/$label.log)"
  fi
}
B=$T/bench
job batch $TV_PY $B/run_bench.py --out $RUN/batch.md --json $RUN/batch.json --data-dir $T/data --threads $THREADS \
  --samples $BS --fabric $FAB=$FAB_BIN/bench_fabric --upstream turbovec=$TV_PY
job single $TV_PY $B/run_single.py --out $RUN/single.md --json $RUN/single.json --data-dir $T/data --threads $THREADS \
  --samples $SS --fabric $FAB=$FAB_BIN/bench_fabric --upstream turbovec=$TV_PY
job recall $TV_PY $B/recall.py --out $RUN/recall.md --json $RUN/recall.json --data-dir $T/data "${RECALL[@]}" \
  --upstream turbovec=$TV_PY

kill $LOADPID 2>/dev/null; LOADPID=
$TV_PY $T/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
