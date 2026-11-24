#!/bin/bash
# Stops this user's own bench processes (weekly.sh, run_engine.sh and the engine servers on BENCH_PORT, default 18098),
# never anyone else's. Use it instead of typing pgrep/pkill patterns into an ssh command line, which match that shell.
# The bracketed patterns do not match a shell that merely mentions them.
P=${BENCH_PORT:-18098}; PP="${P%?}[${P: -1}]"
for p in $(pgrep -u $(id -u) -f "weekly[.]sh|run_engine[.]sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -u $(id -u) -f "llama-server .*--port $PP|vllm serve[ ]|sglang[.]launch_server|VLLM::EngineCor[e]"); do
  g=$(ps -o pgid= -p $p | tr -d " "); [ -n "$g" ] && kill -TERM -- -$g 2>/dev/null; kill $p 2>/dev/null
done
sleep 5
pgrep -u $(id -u) -fa "weekly[.]sh|run_engine[.]sh|llama-serve[r]|vllm serve[ ]|sglang[.]launch_server" | cut -c1-120
nvidia-smi -i ${BENCH_GPU:-7} --query-gpu=memory.used --format=csv,noheader
