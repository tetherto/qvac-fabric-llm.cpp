#!/usr/bin/env bash
#
# LongBench-E benchmark for KV-cache quantization quality evaluation.
#
# Runs upstream LongBench-E (THUDM/LongBench) on a single (model, K, V) cell
# using a persistent llama-server for local inference. Clones THUDM/LongBench into
# tests/longbench/, bootstraps its own Python venv at tests/longbench/.venv/,
# and caches prepared per-tokenizer task data under tests/longbench/_data/.
#
# Mirrors the structure of tests/ruler-bench.sh: same flag set, same per-cell
# CSV schema, same sourced-mode contract. The substantive differences live in
# _lb_prepare_task (HF dataset load + middle-truncate + chat template) and
# _lb_score (per-task metric dispatch via tests/longbench-score.py).
#
# Why LongBench-E (not full LongBench-V1)? The TurboQuant paper (arXiv
# 2504.19874, §4.3) reports Table 1 on LongBench-E specifically: "we employ
# LongBench-E, a subset designed with a more uniform length distribution".
# Matching the paper means the same task subset (13 English tasks) and the
# same per-bucket length distribution (0-4k / 4-8k / 8k+).
#
# Usage (direct):
#   ./tests/longbench-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/longbench-bench.sh
#   longbench_bench -m model.gguf -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1
#
# After longbench_bench returns:
#   longbench_global_score   - unweighted mean across the 13 task means (%)
#   longbench_task_scores    - assoc array of task -> mean% (no stdev)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=kv_cache_bench_common.sh
source "${SCRIPT_DIR}/kv_cache_bench_common.sh"

LB_DIR="${SCRIPT_DIR}/longbench"
LB_REPO="https://github.com/THUDM/LongBench.git"
LB_DATA_URL="https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
LB_RAW_DIR="${LB_DIR}/_raw"        # unzipped upstream JSONL files (one per task)
LB_VENV="${LB_DIR}/.venv"
LB_PREPARE="${SCRIPT_DIR}/longbench-prepare.py"
LB_SCORE="${SCRIPT_DIR}/longbench-score.py"
LB_DATA_DIR=""  # set per-tokenizer/preset combination in longbench_bench()

# The 13 LongBench-E tasks, paper order, English only (Chinese tasks dropped
# to match Llama-3.1-8B-Instruct's training language).
LB_TASKS_ALL=(
    qasper multifieldqa_en
    hotpotqa 2wikimqa
    gov_report multi_news
    trec triviaqa samsum
    passage_count passage_retrieval_en
    lcc repobench-p
)

# Category buckets used for the per-cell results table at the end. The
# orchestrator (test-kv-cache-longbench.py) reproduces the paper's Table 1
# six-column layout, but having the bash runner also emit a category view is
# useful when invoking this script directly.
declare -gA LB_TASK_CATEGORY=(
    [qasper]="SingleQA"           [multifieldqa_en]="SingleQA"
    [hotpotqa]="MultiQA"          [2wikimqa]="MultiQA"
    [gov_report]="Summarization"  [multi_news]="Summarization"
    [trec]="FewShot"              [triviaqa]="FewShot"  [samsum]="FewShot"
    [passage_count]="Synthetic"   [passage_retrieval_en]="Synthetic"
    [lcc]="Code"                  [repobench-p]="Code"
)
LB_CATEGORY_ORDER=(SingleQA MultiQA Summarization FewShot Synthetic Code)

longbench_usage() {
    cat <<'EOF'
LongBench-E benchmark for KV cache quantization quality evaluation

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                K cache type                        (default: f16)
  -ctv  TYPE                V cache type                        (default: f16)
  --num-samples N           Samples per task                    (default: 10)
                            Use -1 / all to run the full LongBench-E split
                            (~3.9k samples — the paper preset).
  --max-length N            Token cap for input prompts         (default: 31500)
                            Inputs longer than this are middle-truncated
                            per upstream pred.py:get_pred.
  --tasks "T ..."           Subset of the 13 LongBench-E tasks  (default: all)
  --tokenizer PATH          HF tokenizer name/path              (default: auto-detect)
                            Used for both middle-truncation and chat-template
                            wrapping. Must match the model.
  --server-bin PATH         Path to llama-server binary         (default: build/bin/llama-server)
  --cli-bin PATH            Deprecated alias for --server-bin (back-compat).
  --seed N                  Sample-selection seed               (default: 42)
  -q, --quiet               Suppress per-sample output
  --no-print-cmd            (No-op; preserved for back-compat with older callers.)
  --csv FILE                Write per-task CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-server (e.g. -ngl 99, -fa 1,
--split-mode none).

LongBench-E tasks (paper category bucketing):
  SingleQA       qasper, multifieldqa_en
  MultiQA        hotpotqa, 2wikimqa
  Summarization  gov_report, multi_news
  FewShot        trec, triviaqa, samsum
  Synthetic      passage_count, passage_retrieval_en
  Code           lcc, repobench-p

Example:
  ./tests/longbench-bench.sh -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      -ctk tbq3_0 -ctv pq3_0 --num-samples 10 -ngl 99 -fa 1
EOF
}

# ── ensure upstream repo is cloned ───────────────────────────────────────────
_lb_ensure_repo() {
    if [[ -f "${LB_DIR}/LongBench/eval.py" ]]; then
        return 0
    fi
    # Two cases:
    #   1. Fresh state — LB_DIR doesn't exist. Plain `git clone` works.
    #   2. Stale state — LB_DIR exists from a previous run that created
    #      _data/ or .venv/ before the clone (or an interrupted clone).
    #      We can't `git clone` into a non-empty dir, so clone to a tempdir
    #      and rsync the contents in, then clean up.
    echo "Cloning THUDM/LongBench into ${LB_DIR} ..."
    if [[ -d "${LB_DIR}" && -n "$(ls -A "${LB_DIR}" 2>/dev/null)" ]]; then
        local tmp
        tmp=$(mktemp -d)
        # depth=1 keeps the clone small (~30 MB); we only read LongBench/*.py
        # and LongBench/config/*.json from upstream.
        git clone --depth 1 "${LB_REPO}" "${tmp}/lb" 2>&1 | tail -2
        # Use cp -a so we preserve the .git dir for future `git pull` (cheap).
        cp -a "${tmp}/lb/." "${LB_DIR}/"
        rm -rf "${tmp}"
    else
        mkdir -p "$(dirname "${LB_DIR}")"
        git clone --depth 1 "${LB_REPO}" "${LB_DIR}" 2>&1 | tail -2
    fi
}

# ── ensure pip deps ──────────────────────────────────────────────────────────
# Minimal subset of upstream's requirements.txt — we load JSONL directly
# (the `datasets` package's script-loader API was removed in v3, so the
# upstream `load_dataset('THUDM/LongBench', ...)` recipe no longer works).
# `rouge` here is the original Python package used by upstream metrics.py,
# NOT the similarly named `rouge-score` package (different API/scores).
# jinja2 is needed by transformers' apply_chat_template (soft dep upstream).
# jieba is imported unconditionally by upstream's metrics.py at module load
# even though we only use the English metric subset — pulling it in is
# cheaper than patching the import.
_lb_ensure_deps() {
    _kv_ensure_venv "${LB_VENV}" "${LB_USE_UV:-0}" "${LB_DIR}/.uv"
    _kv_ensure_pip_deps "LongBench" \
        "rouge fuzzywuzzy numpy transformers jinja2 jieba" \
        "rouge fuzzywuzzy python-Levenshtein numpy transformers jinja2 jieba"
}

# ── ensure unzipped data is on disk ──────────────────────────────────────────
# Upstream distributes the per-task JSONL files in a single data.zip on the
# HF dataset page. The `datasets` library can no longer load their script-
# based loader, so we just fetch + unzip the archive once. The ZIP is small
# enough (~270 MB) and contains both LongBench-V1 and LongBench-E files
# (task.jsonl + task_e.jsonl side-by-side).
_lb_ensure_data() {
    local marker="${LB_RAW_DIR}/data/qasper_e.jsonl"
    if [[ -f "$marker" ]]; then
        return 0
    fi
    mkdir -p "$LB_RAW_DIR"
    local zip="${LB_RAW_DIR}/data.zip"
    if [[ ! -f "$zip" ]] || [[ "$(stat -c%s "$zip" 2>/dev/null || echo 0)" -lt 1000000 ]]; then
        echo "Downloading LongBench data.zip ..."
        rm -f "$zip"
        if command -v curl &>/dev/null; then
            curl -L --fail -o "$zip" "$LB_DATA_URL" 2>&1 | tail -2
        elif command -v wget &>/dev/null; then
            wget -O "$zip" "$LB_DATA_URL" 2>&1 | tail -2
        else
            echo "ERROR: neither curl nor wget found" >&2
            return 1
        fi
    fi
    echo "Unzipping LongBench data into ${LB_RAW_DIR} ..."
    ( cd "$LB_RAW_DIR" && unzip -oq data.zip ) || {
        echo "ERROR: unzip failed; archive may be corrupt or truncated" >&2
        return 1
    }
    [[ -f "$marker" ]]
}

# ── prepare one task's data via the Python helper ───────────────────────────
_lb_prepare_task() {
    local task=$1 tokenizer=$2 max_length=$3 num_samples=$4 seed=$5
    local out_file="${LB_DATA_DIR}/${task}.jsonl"

    if [[ -s "$out_file" ]]; then
        local existing_n
        existing_n=$(wc -l < "$out_file")
        if [[ "$num_samples" -lt 0 || "$existing_n" -ge "$num_samples" ]]; then
            return 0
        fi
    fi

    python3 "$LB_PREPARE" \
        --task "$task" \
        --tokenizer "$tokenizer" \
        --max-length "$max_length" \
        --num-samples "$num_samples" \
        --output-file "$out_file" \
        --upstream-dir "${LB_DIR}/LongBench" \
        --data-dir "${LB_RAW_DIR}/data" \
        --seed "$seed" \
        --variant longbench-e \
        2>&1 | tail -3
}

# Server lifecycle (_kv_pick_port / _kv_start_server / _kv_stop_server) and
# /completion inference (_kv_infer_completion) live in kv_cache_bench_common.sh.

# ── score one prediction using upstream metrics via longbench-score.py ─────
_lb_score() {
    local task=$1 prediction=$2 references_json=$3 all_classes_json=$4
    local payload
    payload=$(python3 -c '
import json, sys
print(json.dumps({
    "task": sys.argv[1],
    "prediction": sys.argv[2],
    "references": json.loads(sys.argv[3]),
    "all_classes": json.loads(sys.argv[4]) if sys.argv[4] not in ("", "null") else None,
}))' \
        "$task" "$prediction" "$references_json" "$all_classes_json")

    python3 "$LB_SCORE" --stdin <<<"$payload"
}

# ── main benchmark function ──────────────────────────────────────────────────
longbench_bench() {
    local server_bin="build/bin/llama-server"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local num_samples=10
    local max_length=31500
    local tokenizer=""
    local seed=42
    local quiet=0
    local csv_file=""
    local log_file=""
    local -a extra_args=()
    local -a tasks=("${LB_TASKS_ALL[@]}")

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        longbench_usage; return 0 ;;
            -m|--model)       model="$2";       shift 2 ;;
            -ctk)             ctk="$2";         shift 2 ;;
            -ctv)             ctv="$2";         shift 2 ;;
            --num-samples)
                # accept both "-1" and "all" as the "use full split" sentinel
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
            --no-print-cmd)   shift ;;  # back-compat no-op (server-based runner has nothing to print)
            --csv)            csv_file="$2";    shift 2 ;;
            --log)            log_file="$2";    shift 2 ;;
            *)                extra_args+=("$1"); shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2
        longbench_usage >&2
        return 1
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

    # Per-(tokenizer × max_length × num_samples) data cache. Different
    # presets get different slices, so we key by all three to avoid
    # silently reusing data from a smaller run when the user asks for more
    # samples — and the deterministic seed in longbench-prepare.py means
    # a re-run with the same triple is byte-stable.
    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    local ns_tag
    if (( num_samples < 0 )); then ns_tag="all"; else ns_tag="n${num_samples}"; fi
    LB_DATA_DIR="${LB_DIR}/_data/${tok_slug}/${ns_tag}_m${max_length}"
    mkdir -p "$LB_DATA_DIR"

    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _lb_ensure_repo
    _lb_ensure_data
    _lb_ensure_deps

    # Spin up the persistent server for this cell and tear it down on any
    # exit path (normal return, error, ^C). All sample inferences below
    # reuse the same loaded model.
    # Server context budget: longest prompt (max_length) + the largest
    # generation budget across LongBench-E tasks (512 tokens, for the
    # summarization tasks per upstream dataset2maxlen.json) + slack for any
    # chat-template / BOS overhead. 1024 of slack is generous but cheap.
    local n_ctx=$(( max_length + 512 + 1024 ))

    trap '_kv_stop_server' EXIT INT TERM
    _kv_start_server "$server_bin" "$model" "$ctk" "$ctv" "$n_ctx" "${extra_args[@]}" || return 1

    # ── banner ──────────────────────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo " LongBench-E Benchmark"
    echo "=========================================="
    echo "  Model:        $(basename "$model")"
    echo "  K type:       $ctk"
    echo "  V type:       $ctv"
    echo "  Tasks:        ${tasks[*]}"
    echo "  Samples:      $([[ "$num_samples" -lt 0 ]] && echo 'all (paper preset)' || echo "$num_samples per task")"
    echo "  Max length:   $max_length tokens"
    echo "  Tokenizer:    $tokenizer"
    echo "  Seed:         $seed"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:        ${extra_args[*]}"
    echo "=========================================="
    echo ""

    # ── prepare data for all selected tasks ─────────────────────────────────
    echo "Preparing LongBench-E task data ..."
    for task in "${tasks[@]}"; do
        echo -n "  ${task} ... "
        if _lb_prepare_task "$task" "$tokenizer" "$max_length" "$num_samples" "$seed"; then
            echo "OK"
        else
            echo "FAILED"
        fi
    done
    echo ""

    # ── run inference + score per task ──────────────────────────────────────
    declare -A task_score_list task_counts
    local global_score_sum=0
    local global_count=0

    for task in "${tasks[@]}"; do
        local data_file="${LB_DATA_DIR}/${task}.jsonl"
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

            local input_text answers_json all_classes_json max_gen
            input_text=$(jq -r '.input' <<<"$line")
            answers_json=$(jq -c '.answers' <<<"$line")
            all_classes_json=$(jq -c '.all_classes // null' <<<"$line")
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
            score=$(_lb_score "$task" "$prediction" "$answers_json" "$all_classes_json")
            if [[ -z "$score" ]]; then
                echo "ERROR: scoring failed for $task sample $sample_idx" >&2
                sample_idx=$((sample_idx + 1))
                continue
            fi

            task_score_list[$task]="${task_score_list[$task]} $score"
            task_counts[$task]=$((${task_counts[$task]} + 1))
            global_score_sum=$(awk "BEGIN { printf \"%.6f\", $global_score_sum + $score }")
            global_count=$((global_count + 1))

            sample_idx=$((sample_idx + 1))

            if (( quiet == 0 )); then
                local pct
                pct=$(awk "BEGIN { printf \"%.0f\", $score * 100 }")
                # `printf "%.80s"` (the obvious form) truncates at byte
                # boundaries and can split multi-byte UTF-8 codepoints,
                # producing invalid bytes in stdout that crash the
                # orchestrator's text-mode subprocess capture. The pipeline
                # below: flatten newlines → cut to 80 bytes → iconv strips
                # any trailing partial-codepoint bytes. Result is valid
                # UTF-8, <= 80 bytes.
                # Append `|| true` so that an iconv non-zero exit code (which
                # happens when the 80-byte head -c cut lands inside a
                # multi-byte codepoint -- common with Qwen3.5 which emits a
                # lot of Unicode) doesn't kill the whole bench under
                # `set -eo pipefail`. `-c` already silently drops invalid
                # bytes; we just don't want a partial-trail-byte error to
                # propagate. 2>/dev/null hides the cosmetic stderr line.
                local short_pred
                short_pred=$(printf '%s' "$prediction" | tr '\n' ' ' \
                    | head -c 80 | iconv -c -f UTF-8 -t UTF-8 2>/dev/null || true)
                printf "  [%s] sample %d: %s%%  got=%s\n" \
                    "$task" "$sample_idx" "$pct" "$short_pred"
            fi
        done < "$data_file"
    done

    # ── per-task table + category aggregates ────────────────────────────────
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

    declare -gA longbench_task_scores=()
    local model_base
    model_base=$(basename "$model")

    declare -A cat_sum cat_count
    for cat in "${LB_CATEGORY_ORDER[@]}"; do
        cat_sum[$cat]=0
        cat_count[$cat]=0
    done

    for task in "${tasks[@]}"; do
        local cnt=${task_counts[$task]:-0}
        local scores_list="${task_score_list[$task]:-}"
        local ms cell_mean cell_sd
        if (( cnt > 0 )); then
            ms=$(_kv_mean_stdev "$scores_list")
            cell_mean="${ms% *}"
            cell_sd="${ms#* }"
            printf "%-24s %12d %14s\n" "$task" "$cnt" "${cell_mean}±${cell_sd}%"
            longbench_task_scores[$task]="$cell_mean"
            local cat="${LB_TASK_CATEGORY[$task]}"
            cat_sum[$cat]=$(awk "BEGIN { printf \"%.4f\", ${cat_sum[$cat]} + $cell_mean }")
            cat_count[$cat]=$((${cat_count[$cat]} + 1))
            if [[ -n "$csv_file" ]]; then
                echo "${model_base},${ctk},${ctv},${task},${LB_TASK_CATEGORY[$task]},${cnt},${cell_mean},${cell_sd}" \
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
    for cat in "${LB_CATEGORY_ORDER[@]}"; do
        local c=${cat_count[$cat]}
        if (( c > 0 )); then
            local cavg
            cavg=$(awk "BEGIN { printf \"%.1f\", ${cat_sum[$cat]} / $c }")
            printf "%-24s %12d %14s\n" "$cat" "$c" "${cavg}%"
            cat_avg_sum=$(awk "BEGIN { printf \"%.4f\", $cat_avg_sum + $cavg }")
            cat_avg_count=$((cat_avg_count + 1))
        fi
    done

    # Paper's "Average" column = unweighted mean of the 6 category means
    # (NOT a weighted mean across samples). Replicated here so direct
    # invocations of this script match the orchestrator's aggregation.
    if (( cat_avg_count > 0 )); then
        longbench_global_score=$(awk "BEGIN { printf \"%.1f\", $cat_avg_sum / $cat_avg_count }")
    else
        longbench_global_score="N/A"
    fi

    echo ""
    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo "  Category-avg score: ${longbench_global_score}%  (${global_count} samples)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

# ── auto-run when executed directly (not sourced) ───────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    longbench_bench "$@"
fi
