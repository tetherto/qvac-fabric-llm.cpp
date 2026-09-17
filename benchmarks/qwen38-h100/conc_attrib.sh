#!/bin/bash
# Decode attribution for the batch-scaling campaign: nsys decode windows at B=2 and B=8 with PDL off (true
# per-kernel durations), at 10k and once at 110k, each printed under both window markers, plus the CUDA-graph
# node fraction and an ncu arithmetic-against-memory check on the batch-8 F8 matmul. The fusion-on/off A/B of the
# first concurrency entry is kept behind FUSION_AB=1. Host-side script (/home/pratik/qwen38-bench).
# usage: [BUILD=build-h100-r1] [FUSION_AB=1] ./conc_attrib.sh <gpu>
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
common="-m $gguf -c 393216 -b 4096 -ub 4096 -fa 1 -ctk f16 -ctv f16 -ngl 999 -ntg 64"
# The default kern_window.py marker predates the CUTLASS prefill route, so every window is printed twice: once
# with that marker and once with a CUTLASS-aware one. Fractional launches per token mean the window leaked.
cutlass_marker='cutlass_gemm|fmha_cutlass|f8_quantize'

window() { # window <tag> <npp> <npl>
    local tag=$1 npp=$2 b=$3
    echo "== nsys npp=$npp B=$b load $(cut -d' ' -f1 /proc/loadavg)"
    GGML_CUDA_PDL=0 nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
        -o $prof/$tag $build/bin/llama-batched-bench $common -npp $npp -npl $b 2>&1 | grep -E "^\| $npp|error"
    nsys export --type sqlite --force-overwrite=true -o $prof/$tag.sqlite $prof/$tag.nsys-rep 2>&1 | grep -iE "error|fail"
    echo "-- marker=default"
    python3 $bench/kern_window.py $prof/$tag.sqlite --n-tokens $((64 * b)) --top 12 2>&1 | head -24
    echo "-- marker=cutlass"
    python3 $bench/kern_window.py $prof/$tag.sqlite --n-tokens $((64 * b)) --top 12 --marker "$cutlass_marker" 2>&1 | head -24
    echo -n "graph fraction (kernels, graph nodes): "
    python3 -c "import sqlite3,sys; d=sqlite3.connect(sys.argv[1]); print(d.execute('select count(*), count(graphNodeId) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone())" $prof/$tag.sqlite
}

window bs-b2-nopdl      10240  2
window bs-b8-nopdl      10240  8
window bs-b8-110k-nopdl 110000 8

# Is the batch-8 F8 matmul memory bound or arithmetic bound? Below 40% DRAM with a hot FMA/ALU pipe means the
# scalar FP32 inner loop is the cost, not the weight traffic. Kernel replay can fail on graph-captured kernels,
# so the run is repeated with graphs disabled if no throughput line comes back.
ncu_run() {
    GGML_CUDA_PDL=0 $1 ncu --kernel-name regex:mul_mat_vec_f8_e4m3 --launch-skip 400 --launch-count 4 \
        --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
        $build/bin/llama-batched-bench $common -npp 10240 -npl 8 2>&1
}
echo "== ncu batch-8 F8 matmul"
out=$(ncu_run "")
if ! echo "$out" | grep -q "Memory Throughput"; then
    echo "-- kernel replay failed under CUDA graphs, retrying with GGML_CUDA_DISABLE_GRAPHS=1 (times not comparable)"
    out=$(ncu_run "GGML_CUDA_DISABLE_GRAPHS=1")
fi
echo "$out" | grep -E "mul_mat_vec_f8|Memory Throughput|Compute \(SM\)|DRAM|Duration|Elapsed|pipe|Utilization|error|ERR_" | head -40

if [ "${FUSION_AB:-0}" = 1 ]; then
    echo "== fusion A/B (-ntg 128)"
    for b in 1 4 8 16; do
        a=$($build/bin/llama-batched-bench $common -npp 10240 -ntg 128 -npl $b 2>/dev/null | grep -E "^\| 10240" | tail -1)
        f=$(GGML_CUDA_DISABLE_FUSION=1 $build/bin/llama-batched-bench $common -npp 10240 -ntg 128 -npl $b 2>/dev/null | grep -E "^\| 10240" | tail -1)
        echo "B=$b fusion-on : $a"
        echo "B=$b fusion-off: $f"
    done
fi
echo ATTRIBDONE
