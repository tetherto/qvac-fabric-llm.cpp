#!/bin/bash
# Why fabric decode does not scale below batch 9: nsys decode windows at B=4 and B=16 (PDL off, true per-kernel
# durations) plus a fusion-on/off A/B at B=1,4,8,16. Host-side script (/home/pratik/qwen38-bench).
# usage: [BUILD=build-h100-r1] ./conc_attrib.sh <gpu>
set -uo pipefail
gpu=$1
build=${BUILD:-build-h100-r1}
gguf=/home/pratik/qwen38-bench/models/Qwen3.8-27B-FP8-q8head.gguf
prof=/home/pratik/qwen38-bench/prof
bench=/home/pratik/qwen38-bench/qvac-fabric-llm.cpp/benchmarks/qwen38-h100
cd /home/pratik/qwen38-bench/qvac-fabric-llm.cpp
export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=$PWD/$build/bin:/home/pratik/qwen38-bench/.venv-vllm/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu12/lib
export GGML_CUDA_GDN_AOT_LIB=/home/pratik/qwen38-bench/flashinfer-gdn/libflashinfer_gdn_sm90_aot.so
common="-m $gguf -c 393216 -b 4096 -ub 4096 -fa 1 -ctk f16 -ctv f16 -ngl 999 -npp 10240 -ntg 64"

for b in 4 16; do
    echo "== nsys B=$b load $(cut -d' ' -f1 /proc/loadavg)"
    GGML_CUDA_PDL=0 nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
        -o $prof/bs-b$b-nopdl $build/bin/llama-batched-bench $common -npl $b 2>&1 | grep -E "^\| 10240|error"
    python3 $bench/kern_window.py $prof/bs-b$b-nopdl.sqlite --n-tokens $((64 * b)) --top 12 2>&1 | head -24
done

echo "== fusion A/B (-ntg 128)"
for b in 1 4 8 16; do
    a=$($build/bin/llama-batched-bench $common -ntg 128 -npl $b 2>/dev/null | grep -E "^\| 10240" | tail -1)
    f=$(GGML_CUDA_DISABLE_FUSION=1 $build/bin/llama-batched-bench $common -ntg 128 -npl $b 2>/dev/null | grep -E "^\| 10240" | tail -1)
    echo "B=$b fusion-on : $a"
    echo "B=$b fusion-off: $f"
done
echo ATTRIBDONE
