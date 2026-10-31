#!/bin/bash
# Phase 2 step 1: half-batch graph with the FFN on Metal against the normal graph, 1024-token microbatches
set -u
O=~/ovl/out-p2; mkdir -p $O; exec > $O/p2a.log 2>&1
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER half-batch graph test since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
echo "start $(date)"
B=~/ovl/build-cml/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf; S=~/bench-weekly/cml/sel-q40
H=~/ffn-probe-20261029/heldout
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
export FFN_COREML_UBATCH=1024 LLAMA_QWEN35_HALVES=1 FFN_COREML_HALVES_METAL=1 FFN_COREML_ASYNC=1
for w in $(ls $H/w2048/*.txt | head -2) $(ls $H/w8192/*.txt | head -1); do
  n=2048; case $w in *w8192*) n=8192;; esac
  echo "== $(basename $w) $n"; $B $M $S quality $w $n 16 1 $O/tok 2>&1 | grep -E "^quality |rror|failed|abort|ASSERT" | cut -c1-400
done
echo "end $(date)"
