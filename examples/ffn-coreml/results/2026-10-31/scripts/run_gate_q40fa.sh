#!/bin/bash
# usage: run_gate.sh selection-dir label
S=$1; L=$2; H=~/ffn-probe-20261029/heldout; BIN=/Users/ishan/bench-weekly/src/fabric-coreml-fa/build-cml-fa/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
mkdir -p $H/out-$L
for group in w2048 w8192; do
  n=${group#w}
  for f in ${3:-$H}/$group/*.txt; do
    w=$(basename $f .txt)
    for mode in quality quality-metal; do
      log=$H/out-$L/$mode-$w.log
      $BIN $M $S $mode $f $n 16 1 $H/out-$L/tok-$mode-$w > $log 2>&1 || echo "RUNFAIL $mode $w"
      grep "^quality " $log | sed "s/^/$mode $w /"
    done
  done
done
echo GATE_DONE
