# shellcheck shell=bash
# shellcheck disable=SC2034  # globals here are consumed by the sourcing scripts
#
# Shared helpers for tests/test-kv-cache-quantization-{perf,perp}.sh.
#
# Callers must set before sourcing / calling the helpers:
#   KVQ_PREFIX - CSV/log filename prefix, e.g. "kv-perf" or "kv-perp"
#   KVQ_TOOL   - main binary name under <build>/bin, e.g. "llama-bench"

# --- Model presets ---

declare -A MODEL_PRESETS_NAME MODEL_PRESETS_URL
MODEL_PRESETS_NAME[mistral-q8]="Mistral-7B-Instruct-v0.3-Q8_0.gguf"
MODEL_PRESETS_URL[mistral-q8]="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q8_0.gguf"
MODEL_PRESETS_NAME[llama-q8]="Llama-3.1-8B-Instruct-Q8_0.gguf"
MODEL_PRESETS_URL[llama-q8]="https://huggingface.co/second-state/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q8_0.gguf"
MODEL_PRESETS_NAME[qwen25-05b-q8]="Qwen2.5-0.5B-Instruct-Q8_0.gguf"
MODEL_PRESETS_URL[qwen25-05b-q8]="https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
MODEL_PRESETS_NAME[qwen35-4b-q8]="Qwen3.5-4B-Q8_0.gguf"
MODEL_PRESETS_URL[qwen35-4b-q8]="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q8_0.gguf"
MODEL_PRESETS_NAME[qwen36-35b-a3b-ud-q4km]="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL_PRESETS_URL[qwen36-35b-a3b-ud-q4km]="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

ALL_MODEL_PRESETS=("mistral-q8" "llama-q8" "qwen25-05b-q8" "qwen35-4b-q8" "qwen36-35b-a3b-ud-q4km")

# Re-invoke the calling script once per model preset. Does not return.
# usage: _kvq_run_all_presets <self> [pass-through args...]
_kvq_run_all_presets() {
    local self="$1"
    shift

    echo "=========================================="
    echo " Running all model presets sequentially"
    echo "=========================================="
    echo ""
    local preset
    for preset in "${ALL_MODEL_PRESETS[@]}"; do
        echo ">>> Starting: $preset (${MODEL_PRESETS_NAME[$preset]})"
        echo ""
        "$self" -m "$preset" "$@" || echo ">>> FAILED: $preset"
        echo ""
        echo ">>> Finished: $preset"
        echo ""
    done
    echo "=========================================="
    echo " All model presets complete. Log files:"
    echo "=========================================="
    ls -1t "${KVQ_PREFIX}"_*.txt 2>/dev/null | head -20
    exit 0
}

# Resolve MODEL_NAME/MODEL_URL/MODEL_PATH from MODEL_PRESET or env/defaults.
# usage: _kvq_resolve_model <default_model_name> <default_model_url>
_kvq_resolve_model() {
    MODEL_DIR="${MODEL_DIR:-models}"

    if [ -n "$MODEL_PRESET" ]; then
        if [ -z "${MODEL_PRESETS_NAME[$MODEL_PRESET]:-}" ]; then
            echo "Error: unknown model preset '$MODEL_PRESET'"
            echo "Available: ${ALL_MODEL_PRESETS[*]}"
            exit 1
        fi
        MODEL_NAME="${MODEL_PRESETS_NAME[$MODEL_PRESET]}"
        MODEL_URL="${MODEL_PRESETS_URL[$MODEL_PRESET]}"
    else
        MODEL_NAME="${MODEL_NAME:-$1}"
        MODEL_URL="${MODEL_URL:-$2}"
    fi
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
}

# Download MODEL_PATH from MODEL_URL unless already present and plausible.
_kvq_download_model() {
    mkdir -p "$MODEL_DIR"

    if [ ! -f "$MODEL_PATH" ] || [ "$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)" -lt 1000000 ]; then
        [ -f "$MODEL_PATH" ] && rm -f "$MODEL_PATH"
        echo "Downloading model: $MODEL_NAME ..."
        if command -v curl &> /dev/null; then
            curl -L --fail -C - -o "$MODEL_PATH" "$MODEL_URL"
        elif command -v wget &> /dev/null; then
            wget -O "$MODEL_PATH" "$MODEL_URL"
        else
            echo "Error: neither curl nor wget found"
            exit 1
        fi
        FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
        if [ "$FILE_SIZE" -lt 1000000 ]; then
            echo "Error: downloaded file is only $FILE_SIZE bytes - likely a redirect or error page."
            echo "Download the model manually:"
            echo "  curl -L --fail -o $MODEL_PATH $MODEL_URL"
            rm -f "$MODEL_PATH"
            exit 1
        fi
        echo "Model downloaded to $MODEL_PATH ($((FILE_SIZE / 1048576)) MB)"
    else
        echo "Model already exists: $MODEL_PATH"
    fi
}

# Set REF_BIN/REF_CONFIGS/REF_GIT_* from REF_IMPL, REF_KS and REF_VS.
_kvq_setup_ref_impl() {
    local tool_rel="build/bin/${KVQ_TOOL}"

    REF_BIN=""
    REF_CONFIGS=()
    REF_GIT_INFO=""
    REF_GIT_BRANCH=""
    REF_GIT_DIRTY=""
    REF_GIT_ORIGIN=""

    REF_IMPL="${REF_IMPL:-../llama-cpp-turboquant}"
    [ ${#REF_KS[@]} -eq 0 ] && REF_KS=("turbo3" "turbo4")
    [ ${#REF_VS[@]} -eq 0 ] && REF_VS=("turbo3" "turbo4")

    if [ ! -d "$REF_IMPL" ]; then
        echo "Note: reference implementation not found at $REF_IMPL - skipping ref rows."
    elif [ ! -f "$REF_IMPL/$tool_rel" ]; then
        echo "Note: reference ${KVQ_TOOL} binary not found at $REF_IMPL/$tool_rel - skipping ref rows."
    else
        REF_BIN="$REF_IMPL/$tool_rel"

        local rk rv
        for rk in "${REF_KS[@]}"; do
            for rv in "${REF_VS[@]}"; do
                REF_CONFIGS+=("${rk}:${rv}")
            done
        done

        if command -v git &> /dev/null && git -C "$REF_IMPL" rev-parse --git-dir &> /dev/null; then
            REF_GIT_INFO=$(git -C "$REF_IMPL" log --oneline -3 2>/dev/null || true)
            REF_GIT_BRANCH=$(git -C "$REF_IMPL" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
            REF_GIT_DIRTY=$(git -C "$REF_IMPL" diff --quiet 2>/dev/null && echo "" || echo " (dirty)")
            REF_GIT_ORIGIN=$(git -C "$REF_IMPL" remote get-url origin 2>/dev/null || echo "")
        fi
    fi
}

# --- GPU device detection ---

_kvq_detect_gpu_device() {
    local gpu_info=""
    local bench_bin="$BUILD_DIR/bin/llama-bench"

    if [ -f "$bench_bin" ]; then
        gpu_info=$("$bench_bin" --list-devices 2>&1 | grep -E '^\s+(Vulkan|CUDA|Metal|SYCL|ROCm)' | head -1 | sed 's/^[[:space:]]*//' || true)
    fi

    if [ -z "$gpu_info" ]; then
        if command -v nvidia-smi &> /dev/null; then
            gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        elif command -v vulkaninfo &> /dev/null; then
            gpu_info=$(vulkaninfo 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= //' || true)
        fi
    fi

    echo "${gpu_info:-unknown}"
}

_kvq_sanitize() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g; s/__*/_/g; s/^_//; s/_$//'; }

# Detect GPU_DEVICE, auto-name CSV_FILE (unless given or CSV=0), stage it via
# mktemp with the final name in CSV_FINAL, and tee all output to LOG_FILE.
_kvq_setup_csv_and_log() {
    GPU_DEVICE=$(_kvq_detect_gpu_device)

    CSV_FINAL=""
    if [ -z "$CSV_FILE" ] && [ "${CSV:-1}" != "0" ]; then
        local gpu_short gpu_tag model_tag cfg_tag
        gpu_short=$(echo "$GPU_DEVICE" | sed 's/^[A-Za-z]*[0-9]*: //; s/ ([0-9].*//')
        gpu_tag=$(_kvq_sanitize "$gpu_short")
        model_tag=$(_kvq_sanitize "${MODEL_NAME%.gguf}")
        cfg_tag=$(printf '%s-' "${CONFIGS[@]}" | sed 's/-$//')
        CSV_FILE="${KVQ_PREFIX}_${gpu_tag}_${model_tag}_${cfg_tag}_$(date +%Y%m%d_%H%M%S).csv"
    fi
    if [ -n "$CSV_FILE" ]; then
        CSV_FINAL="$CSV_FILE"
        CSV_FILE=$(mktemp "/tmp/${KVQ_PREFIX}-XXXXXX.csv")
    fi

    LOG_FILE=""
    if [ -n "$CSV_FINAL" ]; then
        LOG_FILE="${CSV_FINAL%.csv}.txt"
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi
}
