#!/bin/bash
# Rebalanced sidecar (5120 Metal / 6144 + 6144 ANE) with the overlap: timing and quality check, gate, server sweep
set -u
W=~/bench-weekly; C=$W/cml; O=~/ovl/out-p3; RUN=$W/runs/2026-10-31-s5120
mkdir -p $O $RUN/logs; exec >> $O/p3.log 2>&1
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER rebalanced sidecar test since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
B=~/ovl/build-cml/llama-coreml-decoder-bench; M=/Users/Shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf; H=~/ffn-probe-20261029/heldout
export FFN_COREML_GPU_RESIDENT=1 FFN_COREML_THREADS_BATCH=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_DECODE_PLAN=512 FFN_COREML_CACHE_COMPILED=1
export FFN_COREML_ASYNC=1 FFN_COREML_UBATCH=1024 LLAMA_QWEN35_HALVES=1
W8=$H/w8192/cpp-long.txt
echo "== quality s5120 cpp-arg 2048"; $B $M $C/sel-s5120 quality $H/w2048/cpp-arg.txt 2048 16 1 $O/tq 2>&1 | grep -E "^quality |rror|failed" | cut -c1-300
for r in 1 2; do for s in sel-q40 sel-s5120; do for n in 2048 8192; do
  echo "== split $s $n round $r"; $B $M $C/$s split $W8 $n 16 3 $O/ts 2>&1 | grep -E "^request |rror|failed" | cut -c1-120
done; done; done
med() { grep -A4 "== split $1 $2 round" $O/p3.log | grep -o "pp_ms=[0-9.]*" | cut -d= -f2 | sort -n | awk "{a[NR]=\$1} END {print a[int((NR+1)/2)]}"; }
old2=$(med sel-q40 2048); new2=$(med sel-s5120 2048); old8=$(med sel-q40 8192); new8=$(med sel-s5120 8192)
echo "median pp_ms 2048: $old2 -> $new2, 8192: $old8 -> $new8"
if ! python3 -c "import sys; sys.exit(0 if $new2 < $old2 and $new8 < $old8 else 1)"; then echo "no gain, stopping"; echo "end $(date)"; exit 0; fi
echo "== gate s5120 with overlap"
tail -40 $O/gate.log | grep -E "\"(mean_dnll|dnll_upper95|mean_kl|worst_window|worst_window_dnll|pass)\"|RUNFAIL|problems"
sed -e "s#^FABCML_BIN=.*#FABCML_BIN=\x27$HOME/ovl/build-cml\x27#" -e "s#^FABCML_VER=.*#FABCML_VER=\x27Core ML 5120/6144x2 + kernel + overlap (uncommitted)\x27#" $W/runs/2026-10-31-final/engines.env > $RUN/engines.env
DP="1024 4096 16384 32768 57344 104857"
ENGINES_ENV=$RUN/engines.env FABCML_BIN=$HOME/ovl/build-cml CML_ASSETS=$C/sel-s5120 NP=1 CTX=163840 DEPTHS="$DP" \
  FABCML_ENV="FFN_COREML_ASYNC=1 LLAMA_QWEN35_HALVES=1" FABCML_EXTRA="-ub 1024" $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar-ovl
python3 $W/summarize.py $RUN --status ok > /dev/null && echo "summary written"
touch $RUN/DONE
echo "end $(date)"
