#!/bin/bash
# Concurrency sweep of llama-server: one server per level (--parallel N, ctx = max(262144, N * 12288) so every slot
# holds a 10k prompt plus 1k output), conc_bench.py at that level, 2 reps. Host-side script (/home/pratik/qwen38-bench).
# usage: [BUILD=build-h100-r1] [GGUF=...] ./conc_fabric.sh <gpu> <port> <label> [levels]
set -uo pipefail
gpu=$1; port=$2; label=$3; levels=${4:-1,2,4,8,16,32}
build=${BUILD:-build-h100-r1}
gguf=${GGUF:-/home/pratik/qwen38-bench/models/Qwen3.8-27B-FP8-q8head.gguf}
bench=/home/pratik/qwen38-bench/qvac-fabric-llm.cpp/benchmarks/qwen38-h100
cd /home/pratik/qwen38-bench/qvac-fabric-llm.cpp
export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=$PWD/$build/bin:/home/pratik/qwen38-bench/.venv-vllm/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu12/lib
export GGML_CUDA_GDN_AOT_LIB=/home/pratik/qwen38-bench/flashinfer-gdn/libflashinfer_gdn_sm90_aot.so
mkdir -p /home/pratik/qwen38-bench/prof $bench/results/fabric
for n in ${levels//,/ }; do
    ctx=$((n * 12288)); [ $ctx -lt 262144 ] && ctx=262144
    log=/home/pratik/qwen38-bench/prof/srv-$label-n$n.log
    $build/bin/llama-server -m $gguf --alias qwen38-fp8 --host 127.0.0.1 --port $port \
        --ctx-size $ctx --parallel $n --batch-size 4096 --ubatch-size 4096 -ctk f16 -ctv f16 \
        --cache-ram 32768 --cache-prompt -fa on -ngl 999 --threads 16 > "$log" 2>&1 &
    pid=$!
    for i in $(seq 1 150); do
        kill -0 $pid 2>/dev/null || { echo "$label n=$n: server died"; tail -3 "$log"; break; }
        curl -s -o /dev/null -w "%{http_code}" -m 5 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200 && break; sleep 2
    done
    echo "$label n=$n: build $build ctx $ctx pid $pid load-start $(cut -d' ' -f1 /proc/loadavg)"
    python3 $bench/conc_bench.py --base-url "http://127.0.0.1:$port" --model qwen38-fp8 \
        --label "$label-n$n" --prompt-10k $bench/perf-prompt-10k.jsonl --max-tokens 1024 \
        --levels "$n" --reps 2 --out "$bench/results/fabric/$label-n$n.json" 2>&1 | grep -v '^{'
    rc=${PIPESTATUS[0]}
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    for i in $(seq 1 30); do nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu" | grep -q "^0 MiB" && break; sleep 1; done
    echo "$label n=$n: rc $rc load-end $(cut -d' ' -f1 /proc/loadavg)"
done
