#!/usr/bin/env bash
#
# L-Eval (closed-ended subset) benchmark for KV-cache quantization quality.
#
# Runs the 7 closed-ended L-Eval tasks on a chosen (model, K, V) cell
# using a persistent llama-server. Same server-lifecycle / scoring shape
# as zeroscrolls-bench.sh; the substantive differences are the data
# source (shallow clone of OpenLMLab/LEval), the per-task prompt templates
# (handled by leval-prepare.py), and the per-task exam-style scorer
# (leval-score.py).
#
# Scope: closed-ended group only (exact-match scoring). The open-ended
# group officially uses GPT-4-judge or ROUGE/F1 with length-instruction
# correction — both require either an API budget or extra implementation
# effort that doesn't move the KV-cache quality signal. The 7 closed-ended
# tasks (tpo / quality / coursera / gsm100 / topic_retrieval_longchat /
# sci_fi / codeU) are what the paper itself reports per-model Table 3
# results on, so they're enough for cache-type comparison.
#
# Usage (direct):
#   ./tests/leval-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/leval-bench.sh
#   leval_bench -m model.gguf -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=kv_cache_bench_common.sh
source "${SCRIPT_DIR}/kv_cache_bench_common.sh"

LV_DIR="${SCRIPT_DIR}/leval"
LV_REPO="https://github.com/OpenLMLab/LEval.git"
LV_VENV="${LV_DIR}/.venv"
LV_PREPARE="${SCRIPT_DIR}/leval-prepare.py"
LV_SCORE="${SCRIPT_DIR}/leval-score.py"
LV_DATA_SRC=""   # set after clone (LEval-data/Closed-ended-tasks)
LV_DATA_DIR=""

LV_TASKS_ALL=(
    tpo quality coursera
    gsm100
    topic_retrieval_longchat
    sci_fi
    codeU
)

# Category bucketing mirrors the L-Eval paper's grouping in Table 1 /
# Table 3 — they don't formally cluster, but it helps the orchestrator
# show a per-task-type aggregate alongside the per-task numbers.
declare -gA LV_TASK_CATEGORY=(
    [tpo]="MultiChoice"             [quality]="MultiChoice"
    [coursera]="MultiResponse"
    [gsm100]="Math"
    [topic_retrieval_longchat]="Retrieval"
    [sci_fi]="TrueFalse"
    [codeU]="Code"
)
LV_CATEGORY_ORDER=(MultiChoice MultiResponse Math Retrieval TrueFalse Code)

leval_usage() {
    cat <<'EOF'
L-Eval (closed-ended) benchmark for KV cache quantization quality evaluation

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                K cache type                        (default: f16)
  -ctv  TYPE                V cache type                        (default: f16)
  --num-samples N           Samples per task                    (default: -1 = all)
                            L-Eval closed-ended has 64–269 instructions per
                            task (small enough to run fully end-to-end).
  --max-length N            Token cap for input prompts         (default: 16384)
                            Middle-truncation per LongBench recipe.
  --tasks "T ..."           Subset of the 7 closed-ended tasks  (default: all)
  --tokenizer PATH          HF tokenizer name/path              (default: auto-detect)
  --server-bin PATH         Path to llama-server binary         (default: build/bin/llama-server)
  --cli-bin PATH            Deprecated alias for --server-bin.
  --seed N                  Sample-selection seed               (default: 42)
  -q, --quiet               Suppress per-sample output
  --csv FILE                Write per-task CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-server.

L-Eval closed-ended tasks:
  MultiChoice   tpo, quality
  MultiResponse coursera
  Math          gsm100
  Retrieval     topic_retrieval_longchat
  TrueFalse     sci_fi
  Code          codeU
EOF
}

# ── ensure upstream repo cloned (for the JSONL data files) ─────────────────
_lv_ensure_repo() {
    if [[ -d "${LV_DIR}/LEval-data/Closed-ended-tasks" ]]; then
        return 0
    fi
    echo "Cloning OpenLMLab/LEval into ${LV_DIR} ..."
    if [[ -d "${LV_DIR}" && -n "$(ls -A "${LV_DIR}" 2>/dev/null)" ]]; then
        local tmp
        tmp=$(mktemp -d)
        git clone --depth 1 "${LV_REPO}" "${tmp}/lv" 2>&1 | tail -2
        cp -a "${tmp}/lv/." "${LV_DIR}/"
        rm -rf "${tmp}"
    else
        mkdir -p "$(dirname "${LV_DIR}")"
        git clone --depth 1 "${LV_REPO}" "${LV_DIR}" 2>&1 | tail -2
    fi
    [[ -d "${LV_DIR}/LEval-data/Closed-ended-tasks" ]]
}

# ── venv + deps (smaller dep list — closed-ended scoring is pure Python) ──
# No `rouge` / `fuzzywuzzy` / `jieba` — closed-ended scoring is pure-Python
# (regex + string ops).
_lv_ensure_deps() {
    _kv_ensure_venv "${LV_VENV}" "${LV_USE_UV:-0}" "${LV_DIR}/.uv"
    _kv_ensure_pip_deps "L-Eval" \
        "numpy transformers jinja2" \
        "numpy transformers jinja2"
}

# Server lifecycle (_kv_pick_port / _kv_start_server / _kv_stop_server) and
# /completion inference (_kv_infer_completion) live in kv_cache_bench_common.sh.

# ── prep one task ──────────────────────────────────────────────────────────
_lv_prepare_task() {
    local task=$1 tokenizer=$2 max_length=$3 num_samples=$4 seed=$5
    local out_file="${LV_DATA_DIR}/${task}.jsonl"
    if [[ -s "$out_file" ]]; then
        local existing_n
        existing_n=$(wc -l < "$out_file")
        if [[ "$num_samples" -lt 0 && "$existing_n" -ge 1 ]]; then return 0; fi
        if [[ "$num_samples" -ge 0 && "$existing_n" -ge "$num_samples" ]]; then return 0; fi
    fi
    python3 "$LV_PREPARE" \
        --task "$task" \
        --tokenizer "$tokenizer" \
        --max-length "$max_length" \
        --num-samples "$num_samples" \
        --output-file "$out_file" \
        --data-dir "$LV_DATA_SRC" \
        --seed "$seed" \
        2>&1 | tail -3
}

# ── scoring ────────────────────────────────────────────────────────────────
_lv_score() {
    local task=$1 prediction=$2 references_json=$3
    local payload
    payload=$(python3 -c '
import json, sys
print(json.dumps({"task": sys.argv[1], "prediction": sys.argv[2],
                  "references": json.loads(sys.argv[3])}))' \
        "$task" "$prediction" "$references_json")
    python3 "$LV_SCORE" --stdin <<<"$payload"
}

leval_bench() {
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
    local -a tasks=("${LV_TASKS_ALL[@]}")

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        leval_usage; return 0 ;;
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
        echo "ERROR: --model is required" >&2; leval_usage >&2; return 1
    fi
    if [[ ! -x "$server_bin" ]]; then
        echo "ERROR: llama-server binary not found at '$server_bin'" >&2
        echo "       Build with: cmake --build build --target llama-server" >&2
        return 1
    fi
    if [[ -z "$tokenizer" ]]; then
        tokenizer=$(_kv_detect_tokenizer "$model")
        echo "Auto-detected tokenizer: $tokenizer"
    fi

    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    local ns_tag
    if (( num_samples < 0 )); then ns_tag="all"; else ns_tag="n${num_samples}"; fi
    LV_DATA_DIR="${LV_DIR}/_data/${tok_slug}/${ns_tag}_m${max_length}"
    mkdir -p "$LV_DATA_DIR"

    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _lv_ensure_repo || { echo "ERROR: failed to clone OpenLMLab/LEval" >&2; return 1; }
    LV_DATA_SRC="${LV_DIR}/LEval-data/Closed-ended-tasks"
    _lv_ensure_deps

    echo ""
    echo "=========================================="
    echo " L-Eval (closed-ended) Benchmark"
    echo "=========================================="
    echo "  Model:        $(basename "$model")"
    echo "  K type:       $ctk"
    echo "  V type:       $ctv"
    echo "  Tasks:        ${tasks[*]}"
    echo "  Samples:      $([[ "$num_samples" -lt 0 ]] && echo 'all' || echo "$num_samples per task")"
    echo "  Max length:   $max_length tokens"
    echo "  Tokenizer:    $tokenizer"
    echo "  Seed:         $seed"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:        ${extra_args[*]}"
    echo "=========================================="
    echo ""

    echo "Preparing L-Eval task data ..."
    for task in "${tasks[@]}"; do
        echo -n "  ${task} ... "
        if _lv_prepare_task "$task" "$tokenizer" "$max_length" "$num_samples" "$seed"; then
            echo "OK"
        else
            echo "FAILED"
        fi
    done
    echo ""

    # L-Eval per-task max_gen tops out at 64; +512 slack for safety.
    local n_ctx=$(( max_length + 64 + 512 ))

    trap '_kv_stop_server' EXIT INT TERM
    _kv_start_server "$server_bin" "$model" "$ctk" "$ctv" "$n_ctx" "${extra_args[@]}" || return 1

    declare -A task_score_list task_counts
    local global_count=0
    for task in "${tasks[@]}"; do
        local data_file="${LV_DATA_DIR}/${task}.jsonl"
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
            prediction=$(_kv_infer_completion "$max_gen" "$input_text" "$KV_SERVER_URL") || {
                echo "ERROR: inference failed for $task sample $sample_idx" >&2
                sample_idx=$((sample_idx + 1))
                continue
            }
            local score
            score=$(_lv_score "$task" "$prediction" "$references_json") || {
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
                # UTF-8-safe 80-byte truncation — see longbench-bench.sh.
                local short_pred
                short_pred=$(printf '%s' "$prediction" | tr '\n' ' ' \
                    | head -c 80 | iconv -c -f UTF-8 -t UTF-8)
                printf "  [%s] sample %d: %s%%  got=%s\n" \
                    "$task" "$sample_idx" "$pct" "$short_pred"
            fi
        done < "$data_file"
    done

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

    printf "%-30s %10s %14s\n" "Task" "Samples" "Score"
    printf "%-30s %10s %14s\n" "----" "-------" "-----"

    if [[ -n "$csv_file" ]]; then
        echo "model,cache_k,cache_v,task,category,samples,mean_pct,stdev_pct" > "$csv_file"
    fi

    declare -gA leval_task_scores=()
    local model_base
    model_base=$(basename "$model")

    declare -A cat_sum cat_count
    for cat in "${LV_CATEGORY_ORDER[@]}"; do
        cat_sum[$cat]=0
        cat_count[$cat]=0
    done

    for task in "${tasks[@]}"; do
        local cnt=${task_counts[$task]:-0}
        local scores_list="${task_score_list[$task]:-}"
        if (( cnt > 0 )); then
            local ms
            ms=$(_kv_mean_stdev "$scores_list")
            local cell_mean="${ms% *}"
            local cell_sd="${ms#* }"
            printf "%-30s %10d %14s\n" "$task" "$cnt" "${cell_mean}±${cell_sd}%"
            leval_task_scores[$task]="$cell_mean"
            local cat="${LV_TASK_CATEGORY[$task]}"
            cat_sum[$cat]=$(awk "BEGIN { printf \"%.4f\", ${cat_sum[$cat]} + $cell_mean }")
            cat_count[$cat]=$((${cat_count[$cat]} + 1))
            if [[ -n "$csv_file" ]]; then
                echo "${model_base},${ctk},${ctv},${task},${LV_TASK_CATEGORY[$task]},${cnt},${cell_mean},${cell_sd}" \
                    >> "$csv_file"
            fi
        else
            printf "%-30s %10s %14s\n" "$task" "-" "-"
        fi
    done

    echo ""
    printf "%-30s %10s %14s\n" "Category" "Tasks" "Avg"
    printf "%-30s %10s %14s\n" "--------" "-----" "---"
    local cat_avg_sum=0 cat_avg_count=0
    for cat in "${LV_CATEGORY_ORDER[@]}"; do
        local c=${cat_count[$cat]}
        if (( c > 0 )); then
            local cavg
            cavg=$(awk "BEGIN { printf \"%.1f\", ${cat_sum[$cat]} / $c }")
            printf "%-30s %10d %14s\n" "$cat" "$c" "${cavg}%"
            cat_avg_sum=$(awk "BEGIN { printf \"%.4f\", $cat_avg_sum + $cavg }")
            cat_avg_count=$((cat_avg_count + 1))
        fi
    done

    if (( cat_avg_count > 0 )); then
        leval_global_score=$(awk "BEGIN { printf \"%.1f\", $cat_avg_sum / $cat_avg_count }")
    else
        leval_global_score="N/A"
    fi

    echo ""
    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo "  Category-avg score: ${leval_global_score}%  (${global_count} samples)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    leval_bench "$@"
fi
