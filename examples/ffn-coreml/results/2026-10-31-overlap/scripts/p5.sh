#!/bin/bash
# Partial microbatches in the half-batch path: correctness at ~600 and ~980 tokens, then the prompt-length sweep
set -u
W=$HOME/bench-weekly; C=$W/cml; RUN=$W/runs/2026-10-31-small-b; O=$RUN
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p $RUN/logs; exec >> $RUN/run.log 2>&1
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 30; done
echo "$USER Core ML final sweep since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
echo "start $(date)"
M=/Users/shared/models/mlx-vs-llama/gguf/Qwen3.8-27B-Q4_0-pure.gguf
for chars in 2150; do
  PROMPT=$(head -c $chars /Users/ishan/bench-weekly/corpus/docs/function-calling.md)
  python3 -c "import json,sys; print(json.dumps({\"prompt\": sys.argv[1], \"n_predict\": 48, \"temperature\": 0, \"n_probs\": 3, \"cache_prompt\": False}))" "$PROMPT" > $O/req.json
  for cfg in base ovl; do
    if [ $cfg = ovl ]; then E="FFN_COREML_ASYNC=1 LLAMA_QWEN35_HALVES=1"; X="-ub 1024"; else E=""; X=""; fi
    env $E FFN_COREML_ASSETS=$C/sel-s5120 FFN_COREML_GPU_RESIDENT=1 FFN_COREML_SINGLE_ENCODER=1 FFN_COREML_CACHE_COMPILED=1       $HOME/ovl/build-cml/llama-server-coreml -m $M -ngl 99 -fa on --port 8098 --no-cache-prompt -cram 0 -ctxcp 0 --no-webui -np 1 -kvu -c 16384 $X > $O/logs/srv-$cfg-$chars.log 2>&1 &
    P=$!
    until curl -sf http://127.0.0.1:8098/health >/dev/null; do sleep 2; kill -0 $P || { echo "server died $cfg"; tail -5 $O/logs/srv-$cfg-$chars.log; exit 1; }; done
    curl -s http://127.0.0.1:8098/completion -d @$O/req.json > /dev/null
    curl -s http://127.0.0.1:8098/completion -d @$O/req.json > $O/out-$cfg-$chars.json
    kill $P; wait $P 2>/dev/null; sleep 5
  done
  python3 - $O/out-base-$chars.json $O/out-ovl-$chars.json <<"PY"
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
d = [abs(x["top_logprobs"][0]["logprob"] - y["top_logprobs"][0]["logprob"]) for x, y in zip(a["completion_probabilities"], b["completion_probabilities"])]
print("check tokens=%d same_text=%s max_logprob_diff=%.4f pp_tps %.1f -> %.1f" % (a["tokens_evaluated"], a["content"] == b["content"], max(d), a["timings"]["prompt_per_second"], b["timings"]["prompt_per_second"]))
PY
done
touch $RUN/DONE; echo "end $(date)"
