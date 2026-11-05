#!/bin/bash
# Weekly Qwen3.8-27B engine benchmark on qvac-dev-mac-arm64, run by cron at 00:00 every Saturday (Mac local time).
# Updates every engine (update.sh), runs the suite under the shared GPU lock, and writes runs/<date>/summary.json,
# which a scheduled Claude task publishes to the results artifact. Marks the run with DONE or FAILED.
#   weekly.sh            full run
#   weekly.sh --quick    smoke run: no updates, C = 1,4 only, one round of each mode (for testing the pipeline)
set -u
W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
DATE=$(date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE; mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

# keep the Mac awake for the whole run
caffeinate -i -w $$ &

# wait for the shared GPU lock and for other engines to stop, at most 6 hours
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
deadline=$(( $(date +%s) + 6 * 3600 ))
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do
  [ $(date +%s) -gt $deadline ] && fail "GPU busy for 6 hours ($(cat $LOCK 2>/dev/null))"
  sleep 60
done
echo "$USER weekly bench (cron) since $(date +%H:%M)" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "lock taken $(date)"

# update and build every engine; a failed update keeps the previous install
if [ $QUICK = 0 ]; then
  $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"
fi
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
cp $W/engines.env $RUN/engines.env
source $W/engines.env

MODELS=${MODELS_DIR:-/Users/shared/models/mlx-vs-llama}
Q0=$MODELS/gguf/Qwen3.8-27B-Q4_0-pure.gguf
for f in $Q0 $MODELS/gguf/Qwen3.8-27B-DFlash2-Q4_0.gguf $MODELS/mlx/Qwen3.8-27B-4bit/config.json $MODELS/mlx/Qwen3.8-27B-MTP-4bit/config.json $MODELS/hf/Qwen3.8-27B-DFlash2/config.json; do
  [ -e "$f" ] || fail "missing model file $f"
done

# 1) one request, engine only: llama-bench decode and prefill
for e in up fab; do
  bin=$UP_BIN; [ $e = fab ] && bin=$FAB_BIN
  $bin/llama-bench -m $Q0 -fa 1 -p 512 -n 128 -r 3 -o jsonl > $RUN/bench-$e.jsonl 2> $RUN/logs/bench-$e.err \
    || echo "llama-bench $e failed"
done

# 2) concurrent requests, no speculation, through each server: essay prompts (oMLX ignores ignore_eos), 245 tokens
#    in, 512 out. Splash cannot turn its DFlash2 draft off, so it runs here in that mode.
CONC=1,2,4,8,16; ROUNDS=2; [ $QUICK = 1 ] && { CONC=1,4; ROUNDS=1; }
for em in omlx:ar sglang:ar splash:dflash2 fab:ar up:ar mlx:ar; do
  e=${em%%:*}; m=${em#*:}
  $W/run_engine.sh $RUN $e $m conc-$e-$m --mode conc --long-output --concurrency $CONC --rounds $ROUNDS --prompt-tokens 128 --gen 512
done

# 3) one request in every mode each engine supports, on the four spec prompts with an identical prefix, interleaved
#    rounds. mlx-lm and SGLang on MLX have no MTP or DFlash2; Splash only DFlash2.
SPECS="up:ar up:mtp up:dflash2 fab:ar fab:mtp fab:dflash2 omlx:ar omlx:mtp omlx:dflash2 splash:dflash2 mlx:ar sglang:ar"
# one round (each round already runs every prompt twice); the time goes to the DFlash2 depth sweep in step 6
NR=1
for r in $(seq 1 $NR); do
  for em in $SPECS; do
    e=${em%%:*}; m=${em#*:}
    NP=1 CTX=16384 $W/run_engine.sh $RUN $e $m spec-$e-$m --mode spec --fixed-prefix --gen 512
  done
done

# 4) speculation under concurrency: MTP and DFlash2 at 1, 4 and 8 requests (Splash's are in step 2)
CS=1,4,8; [ $QUICK = 1 ] && CS=1,4
for em in up:mtp up:dflash2 fab:mtp fab:dflash2 omlx:mtp omlx:dflash2; do
  e=${em%%:*}; m=${em#*:}
  NP=8 CTX=32768 $W/run_engine.sh $RUN $e $m concspec-$e-$m --mode conc --long-output --concurrency $CS --rounds 1 --prompt-tokens 128 --gen 512
done

# 5) depth sweep: one request per prompt length through each server, no speculation (Splash: DFlash2); prefill t/s
#    is prompt tokens / time to first token, decode t/s is measured at that depth
DP="1024 4096 16384 32768"; [ $QUICK = 1 ] && DP="1024 4096"
# 57344 and 104857 requested come out at about 70k and 123k real tokens (the client's prompts run 17-25% over), so the
# contexts are sized for that. The ~123k level adds about 1.5 h (steps 5 and 6), for a run of about 4 h.
[ $QUICK = 0 ] && DP="$DP 57344 104857"
for em in up:ar fab:ar omlx:ar mlx:ar sglang:ar splash:dflash2; do
  e=${em%%:*}; m=${em#*:}
  NP=1 CTX=163840 SGL_CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN $e $m depth-$e-$m
done

# 6) the same depth sweep with DFlash2, for the engines that can switch it on (Splash's is in step 5)
for e in fab up omlx; do
  NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN $e dflash2 depth-$e-dflash2
done

python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
