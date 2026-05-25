#!/usr/bin/env bash
#
# ZeroSCROLLS benchmark for KV-cache quantization quality evaluation.
#
# Runs a chosen (model, K, V) cell against the ZeroSCROLLS validation split
# (~20 samples per task, 8 tasks, ~160 samples total). Uses a persistent
# llama-server for inference — same pattern as longbench-bench.sh, so server
# startup is paid once per cell instead of once per sample.
#
# Why validation, not test? The ZeroSCROLLS test split ships with
# `output: null` — official scoring happens server-side via leaderboard
# submission. The validation split (~20 samples/task) has full references
# and is what every offline implementation actually uses for development.
#
# Why 8 tasks not 10? space_digest (exponential-similarity on a numeric
# rating) and book_sum_sort (Concordance-Index / Kendall-tau on a sort
# order) need bespoke metrics that aren't part of the standard ROUGE/F1/EM
# triad and aren't published as a self-contained Python package. They're
# scoped for a follow-up; everything else maps cleanly to the same
# metric machinery we already implement for LongBench.
#
# Usage (direct):
#   ./tests/zeroscrolls-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/zeroscrolls-bench.sh
#   zeroscrolls_bench -m model.gguf -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1
#
# After zeroscrolls_bench returns:
#   zeroscrolls_global_score - unweighted mean across the 8 task means (%)
#   zeroscrolls_task_scores  - assoc array of task -> mean% (no stdev)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZS_DIR="${SCRIPT_DIR}/zeroscrolls"
ZS_DATA_BASE_URL="https://huggingface.co/datasets/tau/zero_scrolls/resolve/main"
ZS_RAW_DIR="${ZS_DIR}/_raw"        # per-task ZIPs + extracted JSONL files
ZS_VENV="${ZS_DIR}/.venv"
ZS_PREPARE="${SCRIPT_DIR}/zeroscrolls-prepare.py"
ZS_SCORE="${SCRIPT_DIR}/zeroscrolls-score.py"
ZS_DATA_DIR=""  # set per-tokenizer/preset combination in zeroscrolls_bench()
ZS_SERVER_PORT=""
ZS_SERVER_PID=""
ZS_SERVER_URL=""
ZS_SERVER_LOG=""

# The 8 ZeroSCROLLS tasks we score offline.
ZS_TASKS_ALL=(
    gov_report summ_screen_fd qmsum squality
    qasper narrative_qa musique
    quality
)

# Paper / leaderboard category bucketing. Used by the orchestrator's
# results table and by this script's own summary.
declare -gA ZS_TASK_CATEGORY=(
    [gov_report]="Summarization"     [summ_screen_fd]="Summarization"
    [qmsum]="Summarization"          [squality]="Summarization"
    [qasper]="QA"                    [narrative_qa]="QA"   [musique]="QA"
    [quality]="MultiChoice"
)
ZS_CATEGORY_ORDER=(Summarization QA MultiChoice)

zeroscrolls_usage() {
    cat <<'EOF'
ZeroSCROLLS benchmark for KV cache quantization quality evaluation

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                K cache type                        (default: f16)
  -ctv  TYPE                V cache type                        (default: f16)
  --num-samples N           Samples per task                    (default: -1 = all validation)
                            Validation split has ~20 samples per task so even
                            the "full" run is small; --num-samples N caps it
                            further for plumbing checks.
  --max-length N            Token cap for input prompts         (default: 16384)
                            Inputs longer than this are suffix-preserving
                            truncated (port of upstream run_hf_model.py).
  --tasks "T ..."           Subset of the 8 tasks               (default: all)
  --tokenizer PATH          HF tokenizer name/path              (default: auto-detect)
  --server-bin PATH         Path to llama-server binary         (default: build/bin/llama-server)
  --cli-bin PATH            Deprecated alias for --server-bin (back-compat).
  --seed N                  Sample-selection seed               (default: 42)
  -q, --quiet               Suppress per-sample output
  --csv FILE                Write per-task CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-server (e.g. -ngl 99, -fa 1,
--split-mode none).

ZeroSCROLLS tasks (paper category bucketing):
  Summarization  gov_report, summ_screen_fd, qmsum, squality
  QA             qasper, narrative_qa, musique
  MultiChoice    quality

Example:
  ./tests/zeroscrolls-bench.sh -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1
EOF
}

# ── auto-detect HF tokenizer from model filename (same as longbench) ────────
_zs_detect_tokenizer() {
    local model_path=$1
    local base
    base=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    case "$base" in
        *llama-3.1-8b*|*llama-3-8b*|*llama-3.1*|*meta-llama*)
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
        *ministral*)
            echo "mistralai/Ministral-8B-Instruct-2410" ;;
        *mistral*)
            echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
        *qwen2.5*)
            echo "Qwen/Qwen2.5-7B-Instruct" ;;
        *qwen*)
            echo "Qwen/Qwen2-7B-Instruct" ;;
        *)
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
    esac
}

# ── ensure venv (verbatim from longbench-bench.sh, paths swapped) ──────────
_zs_ensure_uv() {
    local uv_dir="${ZS_DIR}/.uv"
    local uv_bin="${uv_dir}/uv"
    if [[ -x "$uv_bin" ]]; then return 0; fi
    echo "Installing uv into ${uv_dir} ..."
    UV_INSTALL_DIR="$uv_dir" curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    [[ -x "$uv_bin" ]]
}

# See longbench-bench.sh for the flock rationale (parallel-cell race on
# fresh dirs creating the same venv twice).
_zs_ensure_venv() {
    mkdir -p "${ZS_DIR}"
    local lock="${ZS_DIR}/.venv.lock"
    (
        flock -x 9
        if [[ -f "${ZS_VENV}/bin/activate" ]]; then exit 0; fi
        echo "Creating Python venv at ${ZS_VENV} ..."
        local uv_bin="${ZS_DIR}/.uv/uv"
        if [[ "${ZS_USE_UV:-0}" == "1" ]]; then
            if command -v uv &>/dev/null; then
                uv venv "${ZS_VENV}"
            elif _zs_ensure_uv; then
                "$uv_bin" venv "${ZS_VENV}"
            else
                echo "ERROR: ZS_USE_UV=1 but failed to install uv" >&2
                exit 1
            fi
        elif python3 -m venv "${ZS_VENV}" 2>/dev/null; then
            :
        elif command -v uv &>/dev/null; then
            echo "python3 -m venv unavailable, using uv ..."
            uv venv "${ZS_VENV}"
        elif _zs_ensure_uv; then
            "$uv_bin" venv "${ZS_VENV}"
        else
            echo "ERROR: cannot create a Python venv (no python3-venv, no uv)" >&2
            exit 1
        fi
    ) 9>"$lock"
    local rc=$?
    if (( rc != 0 )); then return $rc; fi
    # shellcheck disable=SC1091
    source "${ZS_VENV}/bin/activate"
}

_zs_ensure_deps() {
    _zs_ensure_venv
    local missing=0
    # `rouge` (original Python pkg, not rouge-score) for the summarization
    # tasks. transformers + jinja2 for tokenizer + apply_chat_template.
    # No jieba/fuzzywuzzy needed (no Chinese tasks, no code tasks here).
    for pkg in rouge numpy transformers jinja2; do
        python3 -c "import ${pkg}" 2>/dev/null || missing=1
    done
    if (( missing )); then
        echo "Installing ZeroSCROLLS Python dependencies into venv ..."
        python3 -m pip install --quiet rouge numpy transformers jinja2 2>&1 | tail -5
    fi
}

# ── download + unzip per-task data ──────────────────────────────────────────
_zs_ensure_task_data() {
    local task=$1
    local marker="${ZS_RAW_DIR}/${task}/validation.jsonl"
    if [[ -f "$marker" ]]; then return 0; fi
    mkdir -p "$ZS_RAW_DIR"
    local zip="${ZS_RAW_DIR}/${task}.zip"
    if [[ ! -f "$zip" ]] || [[ "$(stat -c%s "$zip" 2>/dev/null || echo 0)" -lt 1000 ]]; then
        echo "Downloading ${task}.zip ..."
        rm -f "$zip"
        local url="${ZS_DATA_BASE_URL}/${task}.zip"
        if command -v curl &>/dev/null; then
            curl -L --fail -o "$zip" "$url" 2>&1 | tail -1
        elif command -v wget &>/dev/null; then
            wget -O "$zip" "$url" 2>&1 | tail -1
        else
            echo "ERROR: neither curl nor wget found" >&2
            return 1
        fi
    fi
    ( cd "$ZS_RAW_DIR" && unzip -oq "${task}.zip" ) || {
        echo "ERROR: failed to unzip ${task}.zip" >&2
        return 1
    }
    [[ -f "$marker" ]]
}

# ── server lifecycle (verbatim shape from longbench-bench.sh) ──────────────
_zs_pick_port() {
    awk -v seed="$$$(date +%N)" 'BEGIN { srand(seed); printf "%d\n", 20000 + int(rand() * 40000) }'
}

_zs_start_server() {
    local server_bin=$1 model=$2 ctk=$3 ctv=$4 max_length=$5
    shift 5
    local -a extra=("$@")

    ZS_SERVER_PORT=$(_zs_pick_port)
    ZS_SERVER_URL="http://127.0.0.1:${ZS_SERVER_PORT}"
    ZS_SERVER_LOG=$(mktemp)

    # max_length input + 1024 max_gen (upstream run_hf_model.py default) +
    # 512 slack for any chat-template / BOS overhead.
    local n_ctx=$(( max_length + 1024 + 512 ))

    echo "Starting llama-server (port=${ZS_SERVER_PORT}, n_ctx=${n_ctx}, K=${ctk}, V=${ctv}) ..."
    "$server_bin" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -c "$n_ctx" \
        --host 127.0.0.1 \
        --port "$ZS_SERVER_PORT" \
        --no-webui \
        "${extra[@]}" \
        >"$ZS_SERVER_LOG" 2>&1 &
    ZS_SERVER_PID=$!

    local i
    for ((i=0; i<120; i++)); do
        if ! kill -0 "$ZS_SERVER_PID" 2>/dev/null; then
            echo "ERROR: llama-server died during startup (see $ZS_SERVER_LOG):" >&2
            tail -20 "$ZS_SERVER_LOG" >&2
            return 1
        fi
        if curl -sf "${ZS_SERVER_URL}/health" >/dev/null 2>&1; then
            echo "Server ready after ${i}s."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: llama-server did not become healthy within 120s." >&2
    tail -20 "$ZS_SERVER_LOG" >&2
    return 1
}

_zs_stop_server() {
    if [[ -n "$ZS_SERVER_PID" ]] && kill -0 "$ZS_SERVER_PID" 2>/dev/null; then
        kill "$ZS_SERVER_PID" 2>/dev/null
        local i
        for ((i=0; i<5; i++)); do
            kill -0 "$ZS_SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$ZS_SERVER_PID" 2>/dev/null || true
    fi
    [[ -n "$ZS_SERVER_LOG" && -f "$ZS_SERVER_LOG" ]] && rm -f "$ZS_SERVER_LOG"
}

# ── prepare one task's data via the Python helper ───────────────────────────
_zs_prepare_task() {
    local task=$1 tokenizer=$2 max_length=$3 num_samples=$4 seed=$5
    local out_file="${ZS_DATA_DIR}/${task}.jsonl"

    if [[ -s "$out_file" ]]; then
        local existing_n
        existing_n=$(wc -l < "$out_file")
        # Validation split is small (~20/task); if existing >= requested,
        # we trust the cache. -1 (use all) always re-validates the cache.
        if [[ "$num_samples" -lt 0 && "$existing_n" -ge 1 ]]; then return 0; fi
        if [[ "$num_samples" -ge 0 && "$existing_n" -ge "$num_samples" ]]; then return 0; fi
    fi

    python3 "$ZS_PREPARE" \
        --task "$task" \
        --tokenizer "$tokenizer" \
        --max-length "$max_length" \
        --num-samples "$num_samples" \
        --output-file "$out_file" \
        --data-dir "${ZS_RAW_DIR}" \
        --seed "$seed" \
        2>&1 | tail -3
}

# ── single-sample inference via the persistent server ──────────────────────
_zs_infer_single() {
    local tokens_to_gen=$1 prompt=$2

    local body
    body=$(jq -nc \
        --arg p "$prompt" \
        --argjson n "$tokens_to_gen" \
        '{prompt: $p, n_predict: $n, temperature: 0, cache_prompt: false, add_special: false}')

    local resp rc=0
    resp=$(curl -sf --max-time 600 \
        -X POST "${ZS_SERVER_URL}/completion" \
        -H "Content-Type: application/json" \
        -d "$body") || rc=$?
    if (( rc != 0 )); then
        echo "ERROR: /completion request failed (curl rc=$rc)" >&2
        return 1
    fi
    jq -r '.content' <<<"$resp"
}

# ── score one prediction via zeroscrolls-score.py ──────────────────────────
_zs_score() {
    local task=$1 prediction=$2 references_json=$3
    local payload
    payload=$(python3 -c '
import json, sys
print(json.dumps({
    "task": sys.argv[1],
    "prediction": sys.argv[2],
    "references": json.loads(sys.argv[3]),
}))' "$task" "$prediction" "$references_json")

    python3 "$ZS_SCORE" --stdin <<<"$payload"
}

_zs_mean_stdev() {
    echo "$1" | awk '{
        n = NF; if (n == 0) { print "- -"; exit }
        sum = 0; for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        sumsq = 0; for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        sd = (n > 1) ? sqrt(sumsq / (n - 1)) : 0
        printf "%.1f %.1f", mean * 100, sd * 100
    }'
}

zeroscrolls_bench() {
    local server_bin="build/bin/llama-server"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local num_samples=-1
    local max_length=16384
    local tokenizer=""
    local seed=42
    local quiet=0
    local csv_file=""
    local log_file=""
    local -a extra_args=()
    local -a tasks=("${ZS_TASKS_ALL[@]}")

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        zeroscrolls_usage; return 0 ;;
            -m|--model)       model="$2";       shift 2 ;;
            -ctk)             ctk="$2";         shift 2 ;;
            -ctv)             ctv="$2";         shift 2 ;;
            --num-samples)
                if [[ "$2" == "all" ]]; then num_samples=-1; else num_samples="$2"; fi
                shift 2 ;;
            --max-length)     max_length="$2";  shift 2 ;;
            --tasks)
                # shellcheck disable=SC2206
                tasks=($2); shift 2 ;;
            --tokenizer)      tokenizer="$2";   shift 2 ;;
            --cli-bin|--server-bin) server_bin="$2"; shift 2 ;;
            --seed)           seed="$2";        shift 2 ;;
            -q|--quiet)       quiet=1;          shift ;;
            --csv)            csv_file="$2";    shift 2 ;;
            --log)            log_file="$2";    shift 2 ;;
            *)                extra_args+=("$1"); shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2
        zeroscrolls_usage >&2
        return 1
    fi
    if [[ ! -x "$server_bin" ]]; then
        echo "ERROR: llama-server binary not found at '$server_bin'" >&2
        echo "       Build with: cmake --build build --target llama-server" >&2
        return 1
    fi
    if [[ -z "$tokenizer" ]]; then
        tokenizer=$(_zs_detect_tokenizer "$model")
        echo "Auto-detected tokenizer: $tokenizer"
    fi

    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    local ns_tag
    if (( num_samples < 0 )); then ns_tag="all"; else ns_tag="n${num_samples}"; fi
    ZS_DATA_DIR="${ZS_DIR}/_data/${tok_slug}/${ns_tag}_m${max_length}"
    mkdir -p "$ZS_DATA_DIR"

    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _zs_ensure_deps

    # Banner ---------------------------------------------------------------
    echo ""
    echo "=========================================="
    echo " ZeroSCROLLS Benchmark"
    echo "=========================================="
    echo "  Model:        $(basename "$model")"
    echo "  K type:       $ctk"
    echo "  V type:       $ctv"
    echo "  Tasks:        ${tasks[*]}"
    echo "  Samples:      $([[ "$num_samples" -lt 0 ]] && echo 'all validation' || echo "$num_samples per task (capped at validation size)")"
    echo "  Max length:   $max_length tokens"
    echo "  Tokenizer:    $tokenizer"
    echo "  Seed:         $seed"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:        ${extra_args[*]}"
    echo "=========================================="
    echo ""

    # Download per-task data --------------------------------------------------
    echo "Fetching ZeroSCROLLS data ..."
    for task in "${tasks[@]}"; do
        echo -n "  ${task} ... "
        if _zs_ensure_task_data "$task"; then
            echo "OK"
        else
            echo "FAILED"
        fi
    done
    echo ""

    # Prepare prompts per task -----------------------------------------------
    echo "Preparing ZeroSCROLLS task data ..."
    for task in "${tasks[@]}"; do
        echo -n "  ${task} ... "
        if _zs_prepare_task "$task" "$tokenizer" "$max_length" "$num_samples" "$seed"; then
            echo "OK"
        else
            echo "FAILED"
        fi
    done
    echo ""

    # Start persistent server for this cell ----------------------------------
    trap '_zs_stop_server' EXIT INT TERM
    _zs_start_server "$server_bin" "$model" "$ctk" "$ctv" "$max_length" "${extra_args[@]}" || return 1

    # Inference + scoring ----------------------------------------------------
    declare -A task_score_list task_counts
    local global_count=0

    for task in "${tasks[@]}"; do
        local data_file="${ZS_DATA_DIR}/${task}.jsonl"
        if [[ ! -f "$data_file" ]]; then
            echo "SKIP: no data for $task"
            continue
        fi
        task_score_list[$task]=""
        task_counts[$task]=0
        local sample_idx=0

        while IFS= read -r line; do
            if (( num_samples >= 0 && sample_idx >= num_samples )); then
                break
            fi
            local input_text references_json max_gen
            input_text=$(jq -r '.input' <<<"$line")
            references_json=$(jq -c '.references' <<<"$line")
            max_gen=$(jq -r '.max_gen' <<<"$line")
            if [[ -z "$input_text" ]]; then
                sample_idx=$((sample_idx + 1))
                continue
            fi
            local prediction
            prediction=$(_zs_infer_single "$max_gen" "$input_text") || {
                echo "ERROR: inference failed for $task sample $sample_idx" >&2
                sample_idx=$((sample_idx + 1))
                continue
            }
            local score
            score=$(_zs_score "$task" "$prediction" "$references_json") || {
                echo "ERROR: scoring failed for $task sample $sample_idx" >&2
                sample_idx=$((sample_idx + 1))
                continue
            }
            task_score_list[$task]="${task_score_list[$task]} $score"
            task_counts[$task]=$((${task_counts[$task]} + 1))
            global_count=$((global_count + 1))
            sample_idx=$((sample_idx + 1))

            if (( quiet == 0 )); then
                local pct
                pct=$(awk "BEGIN { printf \"%.0f\", $score * 100 }")
                # UTF-8-safe 80-byte truncation — see longbench-bench.sh
                # for rationale (printf "%.80s" splits multi-byte codepoints).
                local short_pred
                short_pred=$(printf '%s' "$prediction" | tr '\n' ' ' \
                    | head -c 80 | iconv -c -f UTF-8 -t UTF-8)
                printf "  [%s] sample %d: %s%%  got=%s\n" \
                    "$task" "$sample_idx" "$pct" "$short_pred"
            fi
        done < "$data_file"
    done

    # Per-task table + category aggregates -----------------------------------
    echo ""
    echo "=========================================="
    local _bpw_k _bpw_v
    case "$ctk" in
        f16) _bpw_k=16.0 ;; q8_0) _bpw_k=8.5 ;; q4_0) _bpw_k=4.5 ;;
        tbq3_0) _bpw_k=4.25 ;; tbq4_0) _bpw_k=5.25 ;;
        pq3_0) _bpw_k=3.25 ;; pq4_0) _bpw_k=4.25 ;; *) _bpw_k="?" ;;
    esac
    case "$ctv" in
        f16) _bpw_v=16.0 ;; q8_0) _bpw_v=8.5 ;; q4_0) _bpw_v=4.5 ;;
        tbq3_0) _bpw_v=4.25 ;; tbq4_0) _bpw_v=5.25 ;;
        pq3_0) _bpw_v=3.25 ;; pq4_0) _bpw_v=4.25 ;; *) _bpw_v="?" ;;
    esac
    echo " Results: K=$ctk(${_bpw_k}bpw)  V=$ctv(${_bpw_v}bpw)  Model=$(basename "$model")"
    echo "=========================================="
    echo ""

    printf "%-24s %12s %14s\n" "Task" "Samples" "Score"
    printf "%-24s %12s %14s\n" "----" "-------" "-----"

    if [[ -n "$csv_file" ]]; then
        echo "model,cache_k,cache_v,task,category,samples,mean_pct,stdev_pct" > "$csv_file"
    fi

    declare -gA zeroscrolls_task_scores=()
    local model_base
    model_base=$(basename "$model")

    declare -A cat_sum cat_count
    for cat in "${ZS_CATEGORY_ORDER[@]}"; do
        cat_sum[$cat]=0
        cat_count[$cat]=0
    done

    for task in "${tasks[@]}"; do
        local cnt=${task_counts[$task]:-0}
        local scores_list="${task_score_list[$task]:-}"
        if (( cnt > 0 )); then
            local ms
            ms=$(_zs_mean_stdev "$scores_list")
            local cell_mean="${ms% *}"
            local cell_sd="${ms#* }"
            printf "%-24s %12d %14s\n" "$task" "$cnt" "${cell_mean}±${cell_sd}%"
            zeroscrolls_task_scores[$task]="$cell_mean"
            local cat="${ZS_TASK_CATEGORY[$task]}"
            cat_sum[$cat]=$(awk "BEGIN { printf \"%.4f\", ${cat_sum[$cat]} + $cell_mean }")
            cat_count[$cat]=$((${cat_count[$cat]} + 1))
            if [[ -n "$csv_file" ]]; then
                echo "${model_base},${ctk},${ctv},${task},${ZS_TASK_CATEGORY[$task]},${cnt},${cell_mean},${cell_sd}" \
                    >> "$csv_file"
            fi
        else
            printf "%-24s %12s %14s\n" "$task" "-" "-"
        fi
    done

    echo ""
    printf "%-24s %12s %14s\n" "Category" "Tasks" "Avg"
    printf "%-24s %12s %14s\n" "--------" "-----" "---"
    local cat_avg_sum=0 cat_avg_count=0
    for cat in "${ZS_CATEGORY_ORDER[@]}"; do
        local c=${cat_count[$cat]}
        if (( c > 0 )); then
            local cavg
            cavg=$(awk "BEGIN { printf \"%.1f\", ${cat_sum[$cat]} / $c }")
            printf "%-24s %12d %14s\n" "$cat" "$c" "${cavg}%"
            cat_avg_sum=$(awk "BEGIN { printf \"%.4f\", $cat_avg_sum + $cavg }")
            cat_avg_count=$((cat_avg_count + 1))
        fi
    done

    if (( cat_avg_count > 0 )); then
        zeroscrolls_global_score=$(awk "BEGIN { printf \"%.1f\", $cat_avg_sum / $cat_avg_count }")
    else
        zeroscrolls_global_score="N/A"
    fi

    echo ""
    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo "  Category-avg score: ${zeroscrolls_global_score}%  (${global_count} samples)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

# ── auto-run when executed directly (not sourced) ───────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    zeroscrolls_bench "$@"
fi
