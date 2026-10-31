#!/bin/bash
# Overlap feasibility: (1) GPU/ANE contention, (2) per-layer timeline of the split at 8192 tokens
set -u
O=~/ovl/out; mkdir -p $O; exec > $O/overlap.log 2>&1
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER overlap feasibility since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
A=~/ovl/ane/ane_loop; L=~/bench-weekly/cml/sel-q40/layer40; FA=~/ovl/fa/build/fa_bench; MM=~/ovl/fa/build/mm_bench
echo "== ANE alone"; $A $L 10
for kv in 65536 122880; do
  echo "== attention kv=$kv alone"; $FA $kv 15 512 2>/dev/null | grep kv=
  echo "== attention kv=$kv with ANE busy"; $A $L 40 & P=$!; sleep 3; $FA $kv 15 512 2>/dev/null | grep kv=; wait $P
done
echo "== matmul alone"; $MM 17408 512 5120 2>/dev/null | grep m=
echo "== matmul with ANE busy"; $A $L 15 & P=$!; sleep 3; for i in 1 2 3 4 5; do $MM 17408 512 5120 2>/dev/null | grep m=; done; wait $P
echo "== ANE with attention kv=122880 looping"; ( for i in $(seq 1 6); do $FA 122880 15 512 > /dev/null 2>&1; done ) & P=$!; sleep 5; $A $L 15; kill $P 2>/dev/null; wait $P 2>/dev/null
echo "== ANE with matmul looping"; ( while true; do $MM 17408 512 5120 > /dev/null 2>&1; done ) & P=$!; sleep 3; $A $L 10; kill $P; wait $P 2>/dev/null; pkill -f mm_bench
B=~/ovl/build-cml/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
F=$(ls ~/ffn-probe-20261029/heldout/w8192/*.txt | head -1)
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
echo "== decoder metal 8192"; $B $M ~/bench-weekly/cml/sel-q40 metal $F 8192 16 2 $O/tok-metal | grep -E "^(prefill|timing|result)" | head
echo "== decoder split 8192 profiled"; FFN_COREML_PROFILE=1 $B $M ~/bench-weekly/cml/sel-q40 split $F 8192 16 2 $O/tok-split > $O/split-profile.txt 2>&1; grep -vE "^profile " $O/split-profile.txt | grep -E "^(prefill|timing|result|resident)" | head
echo "end $(date)"
