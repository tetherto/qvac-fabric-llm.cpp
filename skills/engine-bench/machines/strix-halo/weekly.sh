#!/bin/bash
# Weekly Qwen3.8-27B engine benchmark on qvac-dev-strix-0 (Strix Halo, gfx1151), started by scheduler.sh at 00:00
# every Saturday Europe/Zurich. Updates every engine (update.sh), runs the matrix under the GPU lock and writes
# runs/<date>/summary.json, which a scheduled Claude routine publishes to the dashboard. Marks the run DONE or FAILED.
# The matrix is trimmed to about 2-3 hours of GPU time (user's request, 2026-10-21): one round and one rep of the spec
# prompts, C = 1, 4, 16 without speculation and C = 4 with it, a 1k-32k depth sweep, Vulkan builds plus gufo (ROCm-only).
#   weekly.sh            full run
#   weekly.sh --quick    smoke run: no updates, 64-token generations, C = 1, 4 (for testing the pipeline)
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}; ROCM=/opt/rocm-7.2.4
export PATH=$HOME/.local/bin:$ROCM/bin:/usr/local/bin:/usr/bin:/bin
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
DATE=$(TZ=Europe/Zurich date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

# wait for the GPU lock, for other inference processes (any user) and for an idle GPU, at most 6 hours.
# Never kills anything that is not ours.
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
BUSY="llama-(bench|server|batched|cli|perplexity)|gufo (serve|bench|prompt)|vllm|sglang|ollama|lemonade|rpc-server|test-backend-ops"
gpu_busy() { # mean gpu_busy_percent over 5 s above 20
  local s=0 i
  for i in 1 2 3 4 5; do s=$((s + $(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null || echo 0))); sleep 1; done
  [ $((s / 5)) -gt 20 ]
}
deadline=$(( $(date +%s) + 6 * 3600 ))
while [ -e $LOCK ] || ps -eo args | grep -E "$BUSY" | grep -v grep >/dev/null || gpu_busy; do
  [ $(date +%s) -gt $deadline ] && fail "GPU busy for 6 hours ($(cat $LOCK 2>/dev/null; ps -eo user,args | grep -E "$BUSY" | grep -v grep | head -3))"
  sleep 60
done
echo "$USER weekly bench since $(date +%H:%M) (pid $$)" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "lock taken $(date)"

# update and build every engine; a failed update keeps the previous install
if [ $QUICK = 0 ]; then
  $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
cp $W/engines.env $RUN/engines.env; cp $W/matrix.conf $W/engine-opts.conf $W/backends.conf $RUN/
source $W/engines.env; source $W/matrix.conf

M=${MODELS_DIR:-$HOME/models/qwen38-27b}/Qwen3.8-27B-UD-Q4_K_XL.gguf
for f in $M ${MODELS_DIR:-$HOME/models/qwen38-27b}/Qwen3.8-27B-DFlash2-Q4_K_M.gguf ; do
  [ -s "$f" ] || fail "missing model file $f"
done
[ "$(stat -c %s $M)" = 17559178144 ] || fail "model file $M is incomplete"
CORPUS="$W/corpus/**/*.md"

# 1) one sequence, engine only: llama-bench decode and prefill for every llama.cpp-family build
step() { echo "== $1 $(date +%H:%M:%S)"; }
step "llama-bench"
BR=3; [ $QUICK = 1 ] && BR=1
for c in $CONFIGS; do
  e=${c%:*}; be=${c#*:}; [ $e = gufo ] && continue
  P=$(echo $e | tr a-z A-Z); V=${P}_$(echo $be | tr a-z A-Z)_BIN; bin=${!V:-}
  [ -x "$bin/llama-bench" ] || { echo "no llama-bench for $c"; continue; }
  $bin/llama-bench -m $M -fa 1 -p 512 -n 128 -r $BR -o jsonl > $RUN/bench-$e-$be.jsonl 2> $RUN/logs/bench-$e-$be.err \
    || echo "llama-bench $c failed"
  sleep 5
done

# 2) one request per speculative mode: the four fixed prompts once each, identical prefix for every engine
#    (one rep since the depth sweep was added, to stay within about 2-3 hours)
step "spec"
NR=1; GEN=512; REPS=1; [ $QUICK = 1 ] && GEN=64
for r in $(seq 1 $NR); do
  for c in $CONFIGS; do
    for mode in ar mtp dflash2; do
      $W/run_engine.sh $RUN ${c%:*} ${c#*:} $mode spec --mode spec --fixed-prefix --gen $GEN --reps $REPS --corpus "$CORPUS"
    done
  done
done

# 3) concurrent requests without speculation: unique essay prompts (some servers ignore ignore_eos)
step "conc"
CONC=1,4,16; ROUNDS=1; [ $QUICK = 1 ] && CONC=1,4
for c in $CONFIGS; do
  $W/run_engine.sh $RUN ${c%:*} ${c#*:} ar conc --mode conc --long-output --concurrency $CONC --rounds $ROUNDS \
    --prompt-tokens 128 --gen $GEN --corpus "$CORPUS"
done

# 4) speculation under concurrency: MTP and DFlash2 at 4 requests
step "concspec"
SCONC=4
for c in $CONFIGS; do
  for mode in mtp dflash2; do
    $W/run_engine.sh $RUN ${c%:*} ${c#*:} $mode concspec --mode conc --long-output --concurrency $SCONC --rounds 1 \
      --prompt-tokens 128 --gen $GEN --corpus "$CORPUS"
  done
done

# 5) depth sweep: one request per prompt length through each server without speculation; prefill t/s is prompt
#    tokens / time to first token, decode t/s is measured at that depth (128 tokens)
step "depth"
DP="1024 4096 16384 32768"; [ $QUICK = 1 ] && DP="1024 4096"
for c in $CONFIGS; do
  DEPTHS="$DP" $W/run_engine.sh $RUN ${c%:*} ${c#*:} ar depth
done

step "summary"
python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
