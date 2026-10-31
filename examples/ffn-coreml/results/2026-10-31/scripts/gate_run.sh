#!/bin/bash
# Full held-out quality gate for the Q4_0 sidecar with the new prefill attention kernel
W=$HOME/bench-weekly; C=$W/cml; F=~/ffn-probe-20261029
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER quality gate since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
bash $C/run_gate_q40fa.sh $C/sel-q40 q40fa > $C/gate-q40fa.log 2>&1
python3 -I $F/heldout/analyze_gate.py $F/heldout/out-q40fa 64 > $C/gate-q40fa-summary.txt 2>&1
touch $C/GATE_FA_DONE
