#!/bin/bash
# Usage: bash run.sh DEVICE BUILD_DIR BACKEND N_MAX LABEL
# Required: LANE_MANIFEST. Optional: ROOT, MODEL_ROOT, PROMPTS, REPS, PROMPT_START, PORT, THREADS.
set -euo pipefail
DEVICE=$1; BUILD_DIR=$2; BACKEND=$3; N_MAX=$4; LABEL=$5
ROOT=${ROOT:-$HOME/fabric-dflash2-gpu-next}
MODEL_ROOT=${MODEL_ROOT:-$HOME/fabric-bench/models}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
PROMPTS=${PROMPTS:-$ROOT/prompt.json}
REPS=${REPS:-3}
PROMPT_START=${PROMPT_START:-0}
PORT=${PORT:-8093}
PYTHON=${PYTHON:-python3}
RES=$ROOT/results/$DEVICE
BIN=$BUILD_DIR/bin/llama-server
: "${LANE_MANIFEST:?Set LANE_MANIFEST to the verified lane provenance JSON}"
export PATH=/opt/homebrew/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$RES"
[[ ! -e "$RES/$LABEL.json" && ! -e "$RES/$LABEL.server.log" ]] || { echo "Refusing to overwrite $LABEL"; exit 1; }
mkdir "$ROOT/run.lock" || { echo "Another campaign run owns $ROOT/run.lock"; exit 1; }
PID=
stop_server() {
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    rmdir "$ROOT/run.lock"
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

pin_device() {
    case "$DEVICE:$BACKEND" in
        rtx5090:cuda) export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-6174b261-1aa2-79ad-73d0-63aac6adfb18 ;;
        spark:cuda) export CUDA_VISIBLE_DEVICES=GPU-0550cf62-38d2-4b96-90b1-0a082c20dde5 ;;
        *:vulkan) : "${GGML_VK_VISIBLE_DEVICES:?Set the verified physical Vulkan device ordinal}" ;;
        mac:metal) ;;
        *) echo "Unsupported lane $DEVICE/$BACKEND"; exit 2 ;;
    esac
}

wait_ready() {
    local waited=0 limit=900
    while (( waited < limit )); do
        kill -0 "$PID" 2>/dev/null || return 1
        if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then return 0; fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

record_context() {
    date -u
    uptime
    "$BIN" --version
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    fi
}

pin_device
"$PYTHON" -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT"
ARGS=(-m "$MODEL_ROOT/Qwen3.8-27B-Q4_0.gguf" -ngl 99 -fa on -c 16384 -np 1
      -b 2048 -ub 512 -ctk f16 -ctv f16 --jinja --alias bench --host 127.0.0.1 --port "$PORT")
if (( N_MAX > 0 )); then
    ARGS+=(-md "$MODEL_ROOT/dflash-Qwen3.8-27B-Q8_0.gguf" --spec-type draft-dflash
           --spec-draft-n-max "$N_MAX" -ctkd f16 -ctvd f16)
fi
if [[ -n "${THREADS:-}" ]]; then ARGS+=(-t "$THREADS" -tb "${BATCH_THREADS:-$THREADS}"); fi
record_context >"$RES/$LABEL.context.log" 2>&1
"$PYTHON" "$SCRIPTS/manifest.py" --lane "$LANE_MANIFEST" --binary "$BIN" --label "$LABEL" \
    --device "$DEVICE" --backend "$BACKEND" --draft-width "$N_MAX" --output "$RES/$LABEL.manifest.json" \
    -- "${ARGS[@]}"
printf 'CAMPAIGN_CANARY %s\n' "$LABEL" >"$RES/$LABEL.server.log"
"$BIN" "${ARGS[@]}" >>"$RES/$LABEL.server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$ROOT/run.lock/pid"
# Remove only the PID file created by this run before the lock directory is released.
trap 'rm -f "$ROOT/run.lock/pid"; stop_server' EXIT
wait_ready || { echo "SERVER_FAIL $LABEL: $RES/$LABEL.server.log"; exit 1; }
"$PYTHON" "$SCRIPTS/bench_pp_tg.py" "http://127.0.0.1:$PORT" --engine fabric --prompts "$PROMPTS" \
    --max-tokens 1024 --reps "$REPS" --prompt-start "$PROMPT_START" --label "$LABEL" \
    --manifest "$RES/$LABEL.manifest.json" --output "$RES/$LABEL.json"
record_context >>"$RES/$LABEL.context.log" 2>&1
