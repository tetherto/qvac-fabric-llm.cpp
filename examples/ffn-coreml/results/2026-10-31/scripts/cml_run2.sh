#!/bin/bash
# Follow-up to runs/2026-10-30: Splash with and without its Neural Engine FFN split, and the Core ML engine with DFlash2.
set -u
W=$HOME/bench-weekly; RUN=$W/runs/2026-10-31-x
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export ENGINES_ENV=$W/runs/2026-10-30/engines.env FABCML_BIN=$W/src/fabric-coreml/build-cml CML_ASSETS=$W/cml/sel-q40
mkdir -p $RUN/logs; cp $ENGINES_ENV $RUN/
exec >> $RUN/run.log 2>&1
echo "start $(date)"
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 60; done
echo "$USER Core ML follow-up bench since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
DP="1024 4096 16384 32768 57344 104857"
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN splash dflash2 depth-splash-dflash2
NP=1 CTX=163840 DEPTHS="$DP" SPLASH_EXTRA=--disable-ane $W/run_engine.sh $RUN splash dflash2 depth-splash-dflash2-noane
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN fabcml dflash2 depth-fabcml-dflash2
for em in fab:dflash2 fabcml:dflash2 splash:dflash2; do
  e=${em%%:*}; m=${em#*:}
  NP=1 CTX=16384 $W/run_engine.sh $RUN $e $m spec-$e-$m --mode spec --fixed-prefix --gen 512
done
NP=8 CTX=32768 $W/run_engine.sh $RUN fabcml dflash2 concspec-fabcml-dflash2 --mode conc --long-output --concurrency 1,4,8 --rounds 1 --prompt-tokens 128 --gen 512
touch $RUN/DONE
echo "end $(date)"
