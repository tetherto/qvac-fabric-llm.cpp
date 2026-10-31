#!/bin/bash
# Padding of partial microbatches: outputs and short-prompt prefill with FFN_COREML_PAD_MIN off (9999) and on (256)
set -u
W=$HOME/bench-weekly; RUN=$W/runs/2026-10-31-pad3
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
for pad in 9999 200 9999 200; do
  FFN_COREML_PAD_MIN=$pad ENGINES_ENV=$RUN/engines.env FABCML_BIN=$B CML_ASSETS=$W/cml/sel-q40 NP=1 CTX=16384 DEPTHS="640 700 760 820" \
    $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar-pad$pad
done
touch $RUN/DONE
echo "end $(date)"
