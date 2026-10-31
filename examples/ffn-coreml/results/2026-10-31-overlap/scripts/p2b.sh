#!/bin/bash
# Phase 2 step 2: half-batch graph with Core ML start/finish ops. Quality against Metal, then timing against 512-token async
set -u
O=~/ovl/out-p2; mkdir -p $O; exec > $O/p2b.log 2>&1
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER half-batch Core ML test since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
B=~/ovl/build-cml/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf; S=~/bench-weekly/cml/sel-q40
H=~/ffn-probe-20261029/heldout
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1 FFN_COREML_ASYNC=1
W2=$H/w2048/cpp-arg.txt; W8=$H/w8192/cpp-long.txt
halves() { FFN_COREML_UBATCH=1024 LLAMA_QWEN35_HALVES=1 "$@"; }
echo "== quality halves 2048"; halves $B $M $S quality $W2 2048 16 1 $O/tq2 2>&1 | grep -E "^quality |rror|failed|ASSERT" | cut -c1-300
echo "== quality halves 8192"; halves $B $M $S quality $W8 8192 16 1 $O/tq8 2>&1 | grep -E "^quality |rror|failed|ASSERT" | cut -c1-300
for r in 1 2; do
  for n in 2048 8192; do
    echo "== split 512-async $n round $r"; $B $M $S split $W8 $n 16 3 $O/ts 2>&1 | grep -E "^request |rror|failed" | cut -c1-130
    echo "== split halves $n round $r"; halves $B $M $S split $W8 $n 16 3 $O/ts 2>&1 | grep -E "^request |rror|failed|ASSERT" | cut -c1-130
  done
done
echo "end $(date)"
