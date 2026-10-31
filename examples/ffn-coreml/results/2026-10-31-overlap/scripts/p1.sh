#!/bin/bash
# Phase 1: asynchronous ANE calls (FFN_COREML_ASYNC) against the synchronous path, same build
set -u
O=~/ovl/out-p1; mkdir -p $O; exec > $O/p1.log 2>&1
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER async ANE test since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
B=~/ovl/build-cml/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf; S=~/bench-weekly/cml/sel-q40
H=~/ffn-probe-20261029/heldout
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
W2=$(ls $H/w2048/*.txt | head -1); W8=$(ls $H/w8192/*.txt | head -1)
for a in sync async; do
  [ $a = async ] && export FFN_COREML_ASYNC=1 || unset FFN_COREML_ASYNC
  echo "== quality $a 2048"; $B $M $S quality $W2 2048 16 1 $O/tok-q2-$a 2>&1 | grep -E "^quality |rror|failed"
  echo "== quality $a 8192"; $B $M $S quality $W8 8192 16 1 $O/tok-q8-$a 2>&1 | grep -E "^quality |rror|failed"
done
for r in 1 2; do for a in sync async; do
  [ $a = async ] && export FFN_COREML_ASYNC=1 || unset FFN_COREML_ASYNC
  for n in 2048 8192; do echo "== split $a $n round $r"; $B $M $S split $W8 $n 16 3 $O/tok-s-$a-$n 2>&1 | grep -E "^request |rror|failed" | cut -c1-200; done
done; done
echo "end $(date)"
