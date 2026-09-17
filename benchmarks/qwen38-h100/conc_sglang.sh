#!/bin/bash
# Concurrency sweep of SGLang FP8: one server for every level (the scheduler admits up to --max-running-requests), the
# client varies the concurrency. Defaults otherwise (bf16 KV, default chunked prefill and GEMM backend), fa3 attention
# as in the earlier single-request runs on this host. Host-side script (/home/pratik/qwen38-bench).
# usage: [MODEL=<hf snapshot dir>] [EXTRA="..."] ./conc_sglang.sh <gpu> <port> <label> [levels]
set -uo pipefail
gpu=$1; port=$2; label=$3; levels=${4:-1,2,4,8,16,32}
model=${MODEL:-/home/pratik/qwen38-bench/hf-cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
bench=/home/pratik/qwen38-bench/qvac-fabric-llm.cpp/benchmarks/qwen38-h100
cd /home/pratik/qwen38-bench
export CUDA_VISIBLE_DEVICES=$gpu CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=/home/pratik/qwen38-bench/hf-cache
mkdir -p prof $bench/results/sglang
log=/home/pratik/qwen38-bench/prof/sgl-$label.log
.venv-sglang-nightly/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen38-fp8 \
    --host 127.0.0.1 --port "$port" --context-length 262144 --attention-backend fa3 --max-running-requests 32 \
    --cuda-graph-max-bs 32 --mem-fraction-static 0.8 --enable-cache-report ${EXTRA:-} > "$log" 2>&1 &
pid=$!
for i in $(seq 1 360); do
    kill -0 $pid 2>/dev/null || { echo "$label: server died"; tail -5 "$log"; exit 3; }
    curl -s -o /dev/null -w "%{http_code}" -m 5 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200 && break
    sleep 5
done
echo "$label: sglang pid $pid load-start $(cut -d' ' -f1 /proc/loadavg)"
grep -oE "chunked_prefill_size=[0-9]*|max_prefill_tokens=[0-9]*|max_running_requests=[0-9]*|kv_cache_dtype=[a-z0-9_]*|mem_fraction_static=[0-9.]*|attention_backend=[a-z0-9_]*|fp8_gemm_backend=[a-z_]*|max_total_num_tokens=[0-9]*|cuda_graph_max_bs=[0-9]*" "$log" | sort -u | tr '\n' ' '; echo
python3 $bench/conc_bench.py --base-url "http://127.0.0.1:$port" --model qwen38-fp8 \
    --label "$label" --prompt-10k $bench/perf-prompt-10k.jsonl --max-tokens 1024 \
    --levels "$levels" --reps 2 --out "$bench/results/sglang/$label.json" 2>&1 | grep -v '^{'
rc=${PIPESTATUS[0]}
kill "$pid" 2>/dev/null; sleep 5; pkill -f "sglang.launch_server" 2>/dev/null; wait "$pid" 2>/dev/null
for i in $(seq 1 60); do nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu" | grep -q "^0 MiB" && break; sleep 1; done
echo "$label: rc $rc load-end $(cut -d' ' -f1 /proc/loadavg)"
