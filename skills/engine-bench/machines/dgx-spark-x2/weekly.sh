#!/bin/bash
# Weekly Qwen3.8-27B benchmark of one model split over two DGX Sparks (spark-1ff6 + spark-d9ca by default, direct
# ConnectX-7 cable, see pair.conf): llama.cpp master and qvac-fabric over RPC with -sm tensor, against vLLM with tensor
# parallel 2. Run by cron on the head node at 04:00 every Saturday Europe/Zurich, after the single-Spark bench on
# spark-d9ca (00:00, ~2.5 h). Updates every engine (update.sh), runs the suite under the GPU lock of both nodes, writes
# runs/<date>/summary.json, which a scheduled Claude task publishes to the "2x DGX Spark Engine Bench" dashboard, and
# marks the run with DONE or FAILED.
# Limits over RPC with -sm tensor (tested 2026-10-27/28):
#   qvac-fabric runs with its deferred gated delta net commit off (no -sm tensor split rules for the deferred ops); it
#                needs the rpc-server graph cache fix, in temp-11259 since 2026-10-28 (PR #440);
#   DFlash2 is left out for llama.cpp and qvac-fabric: the draft shares the target's lm_head, which -sm tensor keeps
#                in the split (meta) buffer; a split draft asserts in ggml-backend-meta (split axis 0), a draft on one
#                GPU cannot read that buffer.
# LL_ENGINES and LL_MODES change the llama.cpp-family part of the matrix. EXTRA_LOCKS lists more lock files that mean
# a node is busy (e.g. the single-Spark bench's own lock, when it is not GPU_LOCK).
#   weekly.sh            full run
#   weekly.sh --cron     as cron calls it: the Sparks run on UTC and cron fires at 02:00 and 03:00 UTC on Saturday;
#                        only the call that is 04:00 Saturday in Zurich goes on (CEST or CET)
#   weekly.sh --quick    pipeline test: no updates, 128-token outputs, one rep, C = 1,4
set -u
W=${BENCH_DIR:-$HOME/bench-pair}
source $W/pair.conf
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
if [ "${1:-}" = "--cron" ]; then
  [ "$(TZ=Europe/Zurich date +%u%H)" = "604" ] || exit 0
fi
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
DATE=$(TZ=Europe/Zurich date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

# wait for the GPU lock and for other GPU work on either node (any user), at most 6 hours; never touch other users'
# processes
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
gpu_busy() { $W/gpu-busy.sh $LOCK ${EXTRA_LOCKS:-} || peer "$W/gpu-busy.sh $LOCK ${EXTRA_LOCKS:-}"; }
deadline=$(( $(date +%s) + 6 * 3600 ))
while gpu_busy; do
  [ $(date +%s) -gt $deadline ] && fail "GPU busy for 6 hours (lock: $(cat $LOCK 2>/dev/null; peer cat $LOCK 2>/dev/null))"
  sleep 60
done
# cron sets no USER, and set -u would stop the run here
MSG="${USER:-$(id -un)} 2x Spark weekly bench ($([ $QUICK = 1 ] && echo quick || echo full)) since $(date +%H:%M) $(date +%Z)"
echo "$MSG" > $LOCK; peer "echo '$MSG' > $LOCK"
trap 'rm -f $LOCK; peer rm -f $LOCK; $W/rpc.sh stop' EXIT
echo "lock taken $(date)"
# stop our own engines if the head node runs out of memory (it runs the servers and the client)
$W/memguard.sh $$ $RUN/weekly.log &

if [ $QUICK = 0 ]; then
  $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
cp $W/engines.env $RUN/engines.env
cp $W/weights.json $RUN/weights.json 2>/dev/null
source $W/engines.env

MD=${MODELS_DIR:-$W/models}
GGUF=$MD/gguf/Qwen3.8-27B-NVFP4-modelopt.gguf
for f in $GGUF $MD/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf $MD/nvfp4-hf/config.json $MD/dflash2-hf/config.json; do
  [ -e "$f" ] || fail "missing model file $f"
done

# The matrix follows the single Spark's standard profile, held near two hours: one request on four prompts, concurrency
# without speculation at 1, 4 and 16 requests, speculation under concurrency at 4 (and 8 while the run is under
# 1h45m), and a depth sweep. vLLM takes minutes to start on two nodes, so one server per mode serves all its jobs.
LL_ENGINES=${LL_ENGINES:-up fab}; LL_MODES=${LL_MODES:-ar mtp}
T0=$(date +%s)
DSTD=1024,4096,16384,32768; [ $QUICK = 1 ] && DSTD=1024,4096
GEN=512; REPS=2; CONC=1,4,16; ROUNDS=2; [ $QUICK = 1 ] && { GEN=128; REPS=1; CONC=1,4; ROUNDS=1; }
SPEC=(--mode spec --fixed-prefix --gen $GEN --reps $REPS)
CONCA=(--mode conc --long-output --concurrency $CONC --rounds $ROUNDS --prompt-tokens 128 --gen $GEN)
sconc() { local c=4; [ $QUICK = 0 ] && [ $(( $(date +%s) - T0 )) -lt 6300 ] && c=4,8
  echo --mode conc --long-output --concurrency $c --rounds 1 --prompt-tokens 128 --gen $GEN; }

# 1) llama-bench decode and prefill through the two rpc-servers, -sm tensor
for e in $LL_ENGINES; do
  bin=$UP_BIN; [ $e = fab ] && bin=$FAB_BIN
  $W/rpc.sh start $bin $RUN/logs/bench-$e || echo "rpc-servers did not start for llama-bench $e"
  envs=(DUMMY_ENV=0); [ $e = fab ] && envs=(LLAMA_GDN_DEFERRED=0)
  env "${envs[@]}" $bin/llama-bench -m $GGUF --rpc $SELF:$RPC_PORT,$PEER:$RPC_PORT -dev RPC0/RPC1 -sm tensor -fa 1 \
      -p 512 -n 128 -r 3 -o jsonl > $RUN/bench-$e.jsonl 2> $RUN/logs/bench-$e.err || echo "llama-bench $e failed"
  $W/rpc.sh stop
done

# 2) llama.cpp and qvac-fabric: one request per mode (four fixed prompts x reps), then concurrency
for e in $LL_ENGINES; do
  for m in $LL_MODES; do
    $W/run_engine.sh $RUN $e $m spec-$e-$m "${SPEC[@]}"
  done
done
for e in $LL_ENGINES; do
  $W/run_engine.sh $RUN $e ar conc-$e-ar "${CONCA[@]}"
  for m in $LL_MODES; do
    [ $m = ar ] || $W/run_engine.sh $RUN $e $m concspec-$e-$m $(sconc)
  done
done

# 3) vLLM: one server per mode runs the one-request prompts, then its concurrency levels (and, without speculation,
#    the depth sweep)
$W/run_engine.sh $RUN vllm ar spec-vllm-ar "${SPEC[@]}" :: conc-vllm-ar "${CONCA[@]}" :: depth-vllm-ar --depths $DSTD
for m in mtp dflash2; do
  $W/run_engine.sh $RUN vllm $m spec-vllm-$m "${SPEC[@]}" :: concspec-vllm-$m $(sconc)
done

# 4) prefill and decode by depth, llama.cpp and qvac-fabric: one request per prompt length, one-slot server
for e in $LL_ENGINES; do
  $W/run_engine.sh $RUN $e ar depth-$e-ar --depths $DSTD
done
echo "matrix took $(( ($(date +%s) - T0) / 60 )) min"

python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
