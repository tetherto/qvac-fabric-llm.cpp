#!/bin/bash
# llama-bench prefill at depth with the new attention kernel (default) and the old one (GGML_METAL_FA_MQ=0)
LOCK=/tmp/gpu-bench.lock
[ -e $LOCK ] && { echo "lock held"; exit 1; }
echo "$USER attention kernel bench since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
B=~/fa-work/build-fa/bin/llama-bench
for mode in 4 0; do
  echo "== GGML_METAL_FA_MQ=$mode $(date +%H:%M:%S)"
  GGML_METAL_FA_MQ=$mode LLAMA_GDN_DEFERRED=1 $B -m $M -ngl 99 -fa 1 -p 512 -n 0 -d 0,65536 -r 2 -o md 2>/dev/null
done
echo "end $(date +%H:%M:%S)"
