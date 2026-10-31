#!/bin/bash
# Full held-out gate for the rebalanced sidecar with the overlap on
set -u
C=~/bench-weekly/cml; O=~/ovl/out-p3; H=~/ffn-probe-20261029/heldout
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER quality gate since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
export FFN_COREML_ASYNC=1 FFN_COREML_UBATCH=1024 LLAMA_QWEN35_HALVES=1
bash ~/ovl/examples/ffn-coreml/gate/run_gate.sh ~/ovl/build-cml/llama-coreml-decoder-bench /Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf $C/sel-s5120 $H $O/gate > $O/gate.log 2>&1
touch $O/GATE_DONE
