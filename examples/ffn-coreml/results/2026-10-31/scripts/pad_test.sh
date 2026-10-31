#!/bin/bash
# Padding of partial microbatches: outputs and short-prompt prefill with FFN_COREML_PAD_MIN off (9999) and on (256)
set -u
W=$HOME/bench-weekly; RUN=$W/runs/2026-10-31-pad
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p $RUN/logs; cp $W/runs/2026-10-31-fa/engines.env $RUN/
exec >> $RUN/run.log 2>&1
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER Core ML padding test since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
B=$W/src/fabric-coreml-fa/build-cml-fa
M=/Users/shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
PROMPT=$(head -c 4200 /Users/ishan/bench-weekly/corpus/docs/function-calling.md)
for pad in 9999 256; do
  FFN_COREML_PAD_MIN=$pad FFN_COREML_ASSETS=$W/cml/sel-q40 FFN_COREML_GPU_RESIDENT=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_CACHE_COMPILED=1 \
  LLAMA_GDN_DEFERRED=1 $B/llama-server-coreml -m $M -ngl 99 -fa on --port 8098 --no-cache-prompt -cram 0 -ctxcp 0 --no-webui -np 1 -kvu -c 16384 \
    > $RUN/logs/srv-pad$pad.log 2>&1 &
  P=$!
  until curl -sf http://127.0.0.1:8098/health >/dev/null; do sleep 2; kill -0 $P || { echo "server died"; exit 1; }; done
  python3 -c "import json,sys; print(json.dumps({\"prompt\": sys.argv[1], \"n_predict\": 48, \"temperature\": 0, \"n_probs\": 3, \"cache_prompt\": False}))" "$PROMPT" > $RUN/req.json
  curl -s http://127.0.0.1:8098/completion -d @$RUN/req.json > /dev/null
  curl -s http://127.0.0.1:8098/completion -d @$RUN/req.json > $RUN/out-pad$pad.json
  kill $P; wait $P 2>/dev/null; sleep 5
done
for pad in 9999 256 9999 256; do
  FFN_COREML_PAD_MIN=$pad ENGINES_ENV=$RUN/engines.env FABCML_BIN=$B CML_ASSETS=$W/cml/sel-q40 NP=1 CTX=16384 DEPTHS="1024 2048 3072 4096" \
    $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar-pad$pad
done
touch $RUN/DONE
echo "end $(date)"
