#!/bin/bash
# Held-out quality gate: every window in quality mode (split against Metal) and quality-metal mode (all-Metal control
# through the same offload boundary), then analyze_gate.py.
# usage: run_gate.sh <llama-coreml-decoder-bench> <model.gguf> <sidecar dir> <windows dir> <output dir>
set -u
BIN=$1; MODEL=$2; SIDECAR=$3; WINDOWS=$4; OUT=$5
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
mkdir -p "$OUT"
for group in w2048 w8192; do
  n=${group#w}
  for f in "$WINDOWS"/$group/*.txt; do
    w=$(basename "$f" .txt)
    for mode in quality quality-metal; do
      log=$OUT/$mode-$w.log
      "$BIN" "$MODEL" "$SIDECAR" $mode "$f" $n 16 1 "$OUT/tok-$mode-$w" > "$log" 2>&1 || echo "RUNFAIL $mode $w"
      grep "^quality " "$log" | sed "s/^/$mode $w /"
    done
  done
done
python3 "$(dirname "$0")/analyze_gate.py" "$OUT" "$(grep -c . "$SIDECAR/layers.txt")"
