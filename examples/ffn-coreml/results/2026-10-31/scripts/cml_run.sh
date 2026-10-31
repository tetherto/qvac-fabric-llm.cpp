#!/bin/bash
# After weekly.sh: the Core ML engine (fabcml) through the skill's client into the same run, then the full
# held-out quality gate for the Q4_0 sidecar. Takes the shared GPU lock like weekly.sh.
#   cml_run.sh <run dir>
set -u
RUN=$1; W=${BENCH_DIR:-$HOME/bench-weekly}; F=~/ffn-probe-20261029
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
exec >> $RUN/cml.log 2>&1
echo "cml start $(date)"
until [ -e $RUN/DONE ] || [ -e $RUN/FAILED ]; do sleep 60; done
LOCK=${GPU_LOCK:-/tmp/gpu-bench.lock}
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 60; done
echo "$USER Core ML engine bench since $(date +%H:%M)" > $LOCK
trap 'rm -f $LOCK' EXIT
caffeinate -i -w $$ &
echo "lock taken $(date)"
export FABCML_BIN=$W/src/fabric-coreml/build-cml CML_ASSETS=$W/cml/sel-q40
$W/run_engine.sh $RUN fabcml ar conc-fabcml-ar --mode conc --long-output --concurrency 1,2,4,8,16 --rounds 2 --prompt-tokens 128 --gen 512
NP=1 CTX=16384 $W/run_engine.sh $RUN fabcml ar spec-fabcml-ar --mode spec --fixed-prefix --gen 512
NP=1 CTX=163840 DEPTHS="1024 4096 16384 32768 57344 104857" $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar
grep -q FABCML_VER $RUN/engines.env || {
  echo "FABCML_BIN='$FABCML_BIN'" >> $RUN/engines.env
  echo "FABCML_VER='temp-11259 @ d3d019a3b + feat/coreml-qwen-3.8 Core ML server (fbb340d)'" >> $RUN/engines.env; }
python3 $W/summarize.py $RUN --status ok && echo "summary rewritten"
touch $RUN/CML_DONE
echo "engine runs done $(date)"
$W/cml/run_gate_q40.sh $W/cml/sel-q40 q40full > $W/cml/gate-q40full.log 2>&1
python3 -I $F/heldout/analyze_gate.py $F/heldout/out-q40full 64 > $W/cml/gate-q40full-summary.txt 2>&1
touch $RUN/CML_GATE_DONE
echo "cml end $(date)"
