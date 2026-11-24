#!/bin/bash
# Weekly Qwen3.8-27B engine benchmark on one H100 of cosmicac-a26cbb09 (shared 8x H100 container; the bench uses the
# last GPU, BENCH_GPU=7, as agreed). scheduler.sh starts it at 00:00 every Saturday Europe/Zurich (the box has
# no cron). Updates every engine (update.sh), runs the suite under the GPU lock, and writes runs/<date>/summary.json,
# which a scheduled Claude task publishes to the "H100 Engine Bench" dashboard. Marks the run with DONE or FAILED.
# vLLM and SGLang serve Qwen's FP8 checkpoint (native FP8 on Hopper), llama.cpp and qvac-fabric its Q8_0 GGUF.
#   weekly.sh            full run
#   weekly.sh --quick    pipeline test: no updates, 128-token outputs, one rep, C = 1,4, depths 1k and 4k
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=$HOME/.local/bin:$W/venvs/tools/bin:/usr/local/bin:/usr/bin:/bin
export BENCH_GPU=${BENCH_GPU:-7}
# Zurich time as a POSIX rule: the container has no tzdata, and TZ=Europe/Zurich silently falls back to UTC there
ZURICH="CET-1CEST,M3.5.0,M10.5.0/3"
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
DATE=$(TZ=$ZURICH date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

# wait for the lock and for any other process on the bench's GPU (any user, any container), at most 6 hours; never
# touch other users' processes. The other seven GPUs are other people's and do not count.
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
gpu_busy() {
  [ -e $LOCK ] && return 0
  [ -n "$(nvidia-smi -i $BENCH_GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
  # processes in other containers may not be listed: also count real use (an idle H100 draws ~70 W) or held memory
  local u p m
  read u p m <<< "$(nvidia-smi -i $BENCH_GPU --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"
  [ "${u:-0}" -gt 10 ] 2>/dev/null && [ "${p%.*}" -gt 120 ] 2>/dev/null && return 0
  [ "${m:-0}" -gt 2048 ] 2>/dev/null && return 0
  return 1
}
deadline=$(( $(date +%s) + 6 * 3600 ))
while gpu_busy; do
  [ $(date +%s) -gt $deadline ] && fail "GPU $BENCH_GPU busy for 6 hours ($(cat $LOCK 2>/dev/null; nvidia-smi -i $BENCH_GPU --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' '))"
  sleep 60
done
echo "${USER:-$(id -un)} weekly bench on GPU $BENCH_GPU since $(date +%H:%M) UTC" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "lock taken $(date)"

# update and build every engine; a failed update keeps the previous install
if [ $QUICK = 0 ]; then
  $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
cp $W/engines.env $RUN/engines.env
cp $W/weights.json $RUN/weights.json 2>/dev/null
source $W/engines.env

MD=${MODELS_DIR:-$HOME/models/qwen38-27b}
GGUF=$MD/gguf/Qwen3.8-27B-Q8_0.gguf
for f in $GGUF $MD/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf $MD/fp8-hf/config.json $MD/fp8-hf/mtp.safetensors $MD/dflash2-hf/config.json; do
  [ -e "$f" ] || fail "missing model file $f"
done

# The standard matrix in full (the H100 is fast enough): concurrency 1-16 in two rounds, speculation under concurrency
# at 1, 4 and 8, depths to 100000 requested tokens (about 108-128k real; contexts 131072). vLLM and SGLang take minutes
# to start (compile, CUDA graphs), so one server per mode serves its one-request, concurrency and depth jobs;
# llama.cpp starts in seconds and keeps its one-slot, 32-slot and 8-slot servers apart.
T0=$(date +%s)
DSTD=1024,4096,16384,32768,65536,100000; [ $QUICK = 1 ] && DSTD=1024,4096
GEN=512; REPS=2; CONC=1,2,4,8,16; ROUNDS=2; CS=1,4,8; [ $QUICK = 1 ] && { GEN=128; REPS=1; CONC=1,4; ROUNDS=1; CS=1,4; }
SPEC=(--mode spec --fixed-prefix --gen $GEN --reps $REPS)
CONCA=(--mode conc --long-output --concurrency $CONC --rounds $ROUNDS --prompt-tokens 128 --gen $GEN)
CONCS=(--mode conc --long-output --concurrency $CS --rounds 1 --prompt-tokens 128 --gen $GEN)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$BENCH_GPU

# 1) one sequence, engine only: llama-bench decode and prefill
for e in up fab; do
  bin=$UP_BIN; [ $e = fab ] && bin=$FAB_BIN
  $bin/llama-bench -m $GGUF -fa 1 -p 512 -n 128 -r 3 -o jsonl > $RUN/bench-$e.jsonl 2> $RUN/logs/bench-$e.err \
    || echo "llama-bench $e failed"
done

# 2) llama.cpp and qvac-fabric. One request per mode: four fixed prompts x reps, identical prefix for every engine.
#    fab0 is qvac-fabric with LLAMA_GDN_DEFERRED=0, the reference for its deferred path. Then concurrency: 32 slots
#    without speculation; 8 slots with it (each slot keeps a rollback copy of its recurrent state per draft position,
#    which 32 DFlash2 slots would not fit next to the Q8_0 weights in 80 GB).
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
    NP=8 CTX=32768 $W/run_engine.sh $RUN $e $m concspec-$e-$m "${CONCS[@]}"
  done
done

# 3) vLLM and SGLang: one server per mode runs the one-request prompts, then its concurrency levels (and, without
#    speculation, the depth sweep)
for e in vllm sglang; do
  $W/run_engine.sh $RUN $e ar spec-$e-ar "${SPEC[@]}" :: conc-$e-ar "${CONCA[@]}" :: depth-$e-ar --depths $DSTD
  for m in mtp dflash2; do
    $W/run_engine.sh $RUN $e $m spec-$e-$m "${SPEC[@]}" :: concspec-$e-$m "${CONCS[@]}"
  done
done
# 4) prefill and decode by depth, llama.cpp and qvac-fabric: one request per prompt length on a one-slot server with
#    a 131072-token context
for e in up fab; do
  $W/run_engine.sh $RUN $e ar depth-$e-ar --depths $DSTD
done
echo "matrix took $(( ($(date +%s) - T0) / 60 )) min"

python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
