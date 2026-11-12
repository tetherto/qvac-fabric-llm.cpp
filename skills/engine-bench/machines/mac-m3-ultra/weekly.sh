#!/bin/bash
# Weekly Qwen3.8-27B engine benchmark on qvac-dev-mac-arm64, run by cron at 00:00 every Saturday (Mac local time)
# through ssh localhost (ssh-entry.sh), as jobs cron starts directly run at a lower priority and decode slower.
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
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64|uzu_server.py"
deadline=$(( $(date +%s) + 6 * 3600 ))
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do
  [ $(date +%s) -gt $deadline ] && fail "GPU busy for 6 hours ($(cat $LOCK 2>/dev/null))"
  sleep 60
done
echo "${USER:-$(id -un)} weekly bench (cron) since $(date +%H:%M)" > $LOCK
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
# TensorFold's weights live in a shared Hugging Face cache (see run_engine.sh)
TF_HF=${TF_HF_HOME:-/Users/Shared/huggingface}/hub
for f in $MODELS/gguf/Qwen3.8-27B-Q4_0-pure.gguf $MODELS/gguf/Qwen3.8-27B-DFlash2-Q4_0.gguf $MODELS/mlx/Qwen3.8-27B-4bit/config.json \
    $MODELS/mlx/Qwen3.8-27B-MTP-4bit/config.json $MODELS/hf/Qwen3.8-27B-DFlash2/config.json \
    $MODELS/uzu/Qwen3.8-27B-M-nospec/model.safetensors $MODELS/uzu/Qwen3.8-27B-M-nospec/encoding.json \
    $TF_HF/models--TensorFold--Qwen3.8-27B-MLX-4bit/refs/main $TF_HF/models--z-lab--Qwen3.8-27B-DFlash2/refs/main; do
  [ -e "$f" ] || fail "missing model file $f"
done

# 1) one sequence, llama.cpp builds: prefill of 512 and 4096 tokens and decode of 128 from llama-server's own timings,
#    with the server flags of the other steps. Not llama-bench: it never turns qvac-fabric's Core ML FFN sidecar on
for e in up fab; do
  NP=1 CTX=16384 SINGLE=512,4096 $W/run_engine.sh $RUN $e ar single-$e-ar
done

# 2) concurrent requests, no speculation, through each server: essay prompts (oMLX ignores ignore_eos), 245 tokens
#    in, 512 out. Splash cannot turn its DFlash2 draft off, so it runs here in that mode. Uzu serves one request at
#    a time, so its levels queue. TensorFold runs 16 streams.
CONC=1,2,4,8,16; ROUNDS=2; [ $QUICK = 1 ] && { CONC=1,4; ROUNDS=1; }
for em in omlx:ar sglang:ar splash:dflash2 fab:ar up:ar mlx:ar uzu:ar tf:ar; do
  e=${em%%:*}; m=${em#*:}
  $W/run_engine.sh $RUN $e $m conc-$e-$m --mode conc --long-output --concurrency $CONC --rounds $ROUNDS --prompt-tokens 128 --gen 512
done

# 3) one request in every mode each engine supports, on the four spec prompts with an identical prefix, interleaved
#    rounds. mlx-lm and SGLang on MLX have no MTP or DFlash2; Splash only DFlash2. Uzu runs without speculation only:
#    its drafter has tree shapes for M5/M6 chips only, so the engine leaves it off on the M3 Ultra. TensorFold has no MTP.
SPECS="up:ar up:mtp up:dflash2 fab:ar fab:mtp fab:dflash2 omlx:ar omlx:mtp omlx:dflash2 splash:dflash2 mlx:ar sglang:ar uzu:ar tf:ar tf:dflash2"
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
for em in up:mtp up:dflash2 fab:mtp fab:dflash2 omlx:mtp omlx:dflash2 tf:dflash2; do
  e=${em%%:*}; m=${em#*:}
  NP=8 CTX=32768 $W/run_engine.sh $RUN $e $m concspec-$e-$m --mode conc --long-output --concurrency $CS --rounds 1 --prompt-tokens 128 --gen 512
done

# 5) depth sweep: one request per prompt length through each server, no speculation (Splash: DFlash2); prefill t/s
#    is prompt tokens / time to first token, decode t/s is measured at that depth
DP="1024 4096 16384 32768"; [ $QUICK = 1 ] && DP="1024 4096"
# 57344 and 104857 requested come out at about 70k and 123k real tokens (the client's prompts run 17-25% over), so the
# contexts are sized for that. Steps 5 and 6 take about 3 h 40 min of a full run of about 5 h 20 min (2026-11-09).
[ $QUICK = 0 ] && DP="$DP 57344 104857"
for em in up:ar fab:ar omlx:ar mlx:ar sglang:ar splash:dflash2 uzu:ar tf:ar; do
  e=${em%%:*}; m=${em#*:}
  NP=1 CTX=163840 SGL_CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN $e $m depth-$e-$m
done

# 6) the same depth sweep with DFlash2, for the engines that can switch it on (Splash's is in step 5)
for e in fab up omlx tf; do
  NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN $e dflash2 depth-$e-dflash2
done

python3 $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
