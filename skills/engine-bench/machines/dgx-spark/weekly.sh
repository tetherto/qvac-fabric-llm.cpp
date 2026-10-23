#!/bin/bash
# Weekly Qwen3.8-27B engine benchmark on spark-d9ca (DGX Spark, GB10), run by cron at 00:00 every Saturday Europe/Zurich.
# Updates every engine (update.sh), runs the suite under the GPU lock, and writes runs/<date>/summary.json, which a
# scheduled Claude task publishes to the "DGX Spark Engine Bench" dashboard. Marks the run with DONE or FAILED.
#   weekly.sh            full run
#   weekly.sh --cron     as cron calls it: the machine runs on UTC and Debian cron has no CRON_TZ, so cron fires at
#                        22:00 and 23:00 UTC on Friday and only the call that is 00:00 Saturday in Zurich goes on
#   weekly.sh --quick    pipeline test: no updates, 128-token outputs, one rep, C = 1,4 (still ~40 min: server starts)
#   weekly.sh --heavy    on demand, not in cron: no updates; adds the heavy profile (engines tuned for throughput, up to
#                        64 requests, 8k-token prompts, depths to 128k) to today's run directory, plus the standard depth
#                        sweep if that run lacks it, and re-summarizes the whole run (about 2.5 hours)
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
if [ "${1:-}" = "--cron" ]; then
  [ "$(TZ=Europe/Zurich date +%u%H)" = "600" ] || exit 0
fi
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
HEAVY=0; [ "${1:-}" = "--heavy" ] && HEAVY=1
DATE=$(TZ=Europe/Zurich date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

# wait for the GPU lock and for any other GPU process (any user), at most 6 hours; never touch other users' processes
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
gpu_busy() {
  [ -e $LOCK ] && return 0
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
  # utilization alone can stick at 96% on an idle GB10 after a context goes away, so also require real power draw
  local u p
  read u p <<< "$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
  [ "${u:-0}" -gt 10 ] 2>/dev/null && [ "${p%.*}" -gt 30 ] 2>/dev/null && return 0
  return 1
}
deadline=$(( $(date +%s) + 6 * 3600 ))
while gpu_busy; do
  [ $(date +%s) -gt $deadline ] && fail "GPU busy for 6 hours ($(cat $LOCK 2>/dev/null; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' '))"
  sleep 60
done
echo "$USER weekly bench ($([ $HEAVY = 1 ] && echo heavy || echo cron)) since $(date +%H:%M) UTC" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "lock taken $(date)"
# stop our own engines if the shared machine runs out of memory (first-use kernel builds once did)
$W/memguard.sh $$ $RUN/weekly.log &

# update and build every engine; a failed update keeps the previous install
if [ $QUICK = 0 ] && [ $HEAVY = 0 ]; then
  $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
cp $W/engines.env $RUN/engines.env
cp $W/weights.json $RUN/weights.json 2>/dev/null
source $W/engines.env

MD=${MODELS_DIR:-$HOME/models/qwen38-27b}
GGUF=$MD/gguf/Qwen3.8-27B-NVFP4-modelopt.gguf
for f in $GGUF $MD/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf $MD/nvfp4-hf/config.json $MD/dflash2-hf/config.json; do
  [ -e "$f" ] || fail "missing model file $f"
done

# --heavy: the throughput-tuned profile, labels <kind>-<engine>-<mode>-heavy (run_engine.sh PROFILE=heavy has the
# server sizing). Short prompts at 1-64 requests, 8k-token prompts at 1-16, MTP and DFlash2 at 8-32, depths to 128k
# (the top level is 100000 requested tokens: real prompts run 8-28% longer than requested, so about 108-128k, which
# fits the 131072-token context of vLLM and SGLang with 128 out).
# llama.cpp starts a fresh server per job: with a unified cache, idle slots keep their cells and long jobs would leave
# too little room for the next one.
run_heavy() {
  local T0=$(date +%s) DSTD=1024,4096,16384,32768
  local DH=1024,4096,16384,32768,65536,100000
  local SHORT=(--mode conc --long-output --concurrency 1,8,16,32,64 --rounds 1 --prompt-tokens 128 --gen 512)
  local LONG=(--mode conc --long-output --concurrency 1,4,8,16 --rounds 1 --prompt-tokens 8192 --gen 256)
  local SPECC=(--mode conc --long-output --concurrency 8,16,32 --rounds 1 --prompt-tokens 128 --gen 512)
  # the standard depth sweep, when this run predates it
  if ! grep -q '"label": "depth-up-ar"' $RUN/results.jsonl 2>/dev/null; then
    for e in up fab; do $W/run_engine.sh $RUN $e ar depth-$e-ar --depths $DSTD; done
    for e in vllm sglang; do $W/run_engine.sh $RUN $e ar depth-$e-ar --depths $DSTD; done
  fi
  for e in vllm sglang; do
    PROFILE=heavy $W/run_engine.sh $RUN $e ar depth-$e-ar-heavy --depths $DH :: conc-$e-ar-heavy "${SHORT[@]}" :: conc-$e-ar-heavy "${LONG[@]}"
    for m in mtp dflash2; do
      PROFILE=heavy $W/run_engine.sh $RUN $e $m concspec-$e-$m-heavy "${SPECC[@]}"
    done
  done
  for e in up fab; do
    # stay near the user's 2-3 hour budget: qvac-fabric's heavy jobs are the first to go
    if [ $e = fab ] && [ $(( $(date +%s) - T0 )) -gt 8400 ]; then echo "heavy: trimmed qvac-fabric (over 2h20m)"; break; fi
    PROFILE=heavy $W/run_engine.sh $RUN $e ar depth-$e-ar-heavy --depths $DH
    PROFILE=heavy $W/run_engine.sh $RUN $e ar conc-$e-ar-heavy "${SHORT[@]}"
    PROFILE=heavy $W/run_engine.sh $RUN $e ar conc-$e-ar-heavy "${LONG[@]}"
    for m in mtp dflash2; do
      PROFILE=heavy $W/run_engine.sh $RUN $e $m concspec-$e-$m-heavy "${SPECC[@]}"
    done
  done
  echo "heavy took $(( ($(date +%s) - T0) / 60 )) min"
}

if [ $HEAVY = 1 ]; then
  run_heavy
else
# The matrix is held to about two hours of GPU time: one round of each test, concurrency without speculation at 1, 4
# and 16 requests, speculation under concurrency at 4 requests and also 8 while the run is under 2h15m. vLLM and SGLang
# take ~5 minutes to start, so one server per mode serves both its one-request and its concurrency jobs; llama.cpp
# starts in seconds and keeps its one-slot and 32-slot servers apart.
T0=$(date +%s)
DSTD=1024,4096,16384,32768; [ $QUICK = 1 ] && DSTD=1024,4096
GEN=512; REPS=2; CONC=1,4,16; ROUNDS=2; [ $QUICK = 1 ] && { GEN=128; REPS=1; CONC=1,4; ROUNDS=1; }
SPEC=(--mode spec --fixed-prefix --gen $GEN --reps $REPS)
CONCA=(--mode conc --long-output --concurrency $CONC --rounds $ROUNDS --prompt-tokens 128 --gen $GEN)
sconc() { local c=4; [ $QUICK = 0 ] && [ $(( $(date +%s) - T0 )) -lt 8100 ] && c=4,8
  echo --mode conc --long-output --concurrency $c --rounds 1 --prompt-tokens 128 --gen $GEN; }

# 1) one sequence, engine only: llama-bench decode and prefill
for e in up fab; do
  bin=$UP_BIN; [ $e = fab ] && bin=$FAB_BIN
  $bin/llama-bench -m $GGUF -fa 1 -p 512 -n 128 -r 3 -o jsonl > $RUN/bench-$e.jsonl 2> $RUN/logs/bench-$e.err \
    || echo "llama-bench $e failed"
done

# 2) llama.cpp and qvac-fabric. One request per mode: four fixed prompts x reps, identical prefix for every engine.
#    fab0 is qvac-fabric with LLAMA_GDN_DEFERRED=0, the reference for its deferred path. Then concurrency.
for e in up fab; do
  for m in ar mtp dflash2; do
    $W/run_engine.sh $RUN $e $m spec-$e-$m "${SPEC[@]}"
  done
done
for m in mtp dflash2; do
  FAB_DEFERRED=0 $W/run_engine.sh $RUN fab $m spec-fab0-$m "${SPEC[@]}"
done
for e in up fab; do
  $W/run_engine.sh $RUN $e ar conc-$e-ar "${CONCA[@]}"
  for m in mtp dflash2; do
    $W/run_engine.sh $RUN $e $m concspec-$e-$m $(sconc)
  done
done

# 3) vLLM and SGLang: one server per mode runs the one-request prompts, then its concurrency levels (and, without
#    speculation, the depth sweep of step 4)
for e in vllm sglang; do
  $W/run_engine.sh $RUN $e ar spec-$e-ar "${SPEC[@]}" :: conc-$e-ar "${CONCA[@]}" :: depth-$e-ar --depths $DSTD
  for m in mtp dflash2; do
    $W/run_engine.sh $RUN $e $m spec-$e-$m "${SPEC[@]}" :: concspec-$e-$m $(sconc)
  done
done
# 4) prefill and decode by depth, llama.cpp and qvac-fabric (vLLM and SGLang ran theirs in step 3): one request per
#    prompt length on a one-slot server with a 40960-token context
for e in up fab; do
  $W/run_engine.sh $RUN $e ar depth-$e-ar --depths $DSTD
done
echo "matrix took $(( ($(date +%s) - T0) / 60 )) min"

fi

python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
