#!/bin/bash
# Overlap in the server: correctness, then prompt-length sweeps of the current setup, the overlap and Splash in one session
set -u
W=$HOME/bench-weekly; RUN=$W/runs/2026-10-31-ovl; O=$RUN
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p $RUN/logs; exec >> $RUN/run.log 2>&1
sed -e "s#^FABCML_BIN=.*#FABCML_BIN=\x27$HOME/ovl/build-cml\x27#" -e "s#^FABCML_VER=.*#FABCML_VER=\x27Core ML + kernel + overlap (uncommitted)\x27#" $W/runs/2026-10-31-final/engines.env > $RUN/engines.env
export ENGINES_ENV=$RUN/engines.env FABCML_BIN=$HOME/ovl/build-cml CML_ASSETS=$W/cml/sel-q40
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER Core ML overlap sweep since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
M=/Users/shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
PROMPT=$(head -c 4200 /Users/ishan/bench-weekly/corpus/docs/function-calling.md)
for cfg in base ovl; do
  if [ $cfg = ovl ]; then E="FFN_COREML_ASYNC=1 LLAMA_QWEN35_HALVES=1"; X="-ub 1024"; else E=""; X=""; fi
  env $E FFN_COREML_ASSETS=$W/cml/sel-q40 FFN_COREML_GPU_RESIDENT=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_CACHE_COMPILED=1     $HOME/ovl/build-cml/llama-server-coreml -m $M -ngl 99 -fa on --port 8098 --no-cache-prompt -cram 0 -ctxcp 0 --no-webui -np 1 -kvu -c 16384 $X > $O/logs/srv-$cfg.log 2>&1 &
  P=$!
  until curl -sf http://127.0.0.1:8098/health >/dev/null; do sleep 2; kill -0 $P || { echo "server died $cfg"; tail -5 $O/logs/srv-$cfg.log; exit 1; }; done
  python3 -c "import json,sys; print(json.dumps({\"prompt\": sys.argv[1], \"n_predict\": 48, \"temperature\": 0, \"n_probs\": 3, \"cache_prompt\": False}))" "$PROMPT" > $O/req.json
  curl -s http://127.0.0.1:8098/completion -d @$O/req.json > /dev/null
  curl -s http://127.0.0.1:8098/completion -d @$O/req.json > $O/out-$cfg.json
  kill $P; wait $P 2>/dev/null; sleep 5
done
DP="1024 4096 16384 32768 57344 104857"
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar
NP=1 CTX=163840 DEPTHS="$DP" FABCML_ENV="FFN_COREML_ASYNC=1 LLAMA_QWEN35_HALVES=1" FABCML_EXTRA="-ub 1024" $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar-ovl
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN splash dflash2 depth-splash-dflash2
python3 $W/summarize.py $RUN --status ok > /dev/null && echo "summary written"
touch $RUN/DONE; echo "end $(date)"
