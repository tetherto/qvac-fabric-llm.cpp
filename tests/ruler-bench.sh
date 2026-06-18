#!/usr/bin/env bash
#
# RULER benchmark for KV cache quantization quality evaluation.
#
# Runs a subset of NVIDIA RULER tasks (NIAH variants, variable tracking,
# common-words extraction) at configurable context lengths, using llama-cli
# for local inference. Clones NVIDIA/RULER into tests/ruler/ if needed.
#
# Usage (direct):
#   ./tests/ruler-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/ruler-bench.sh
#   ruler_bench -m model.gguf -ctk tbq3_0 -ctv tbq3_0 -ngl 99
#
# After ruler_bench returns, the following variables are set:
#   ruler_global_score   - overall accuracy %
#   ruler_task_scores    - associative array of task -> score%

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULER_DIR="${SCRIPT_DIR}/ruler"
RULER_REPO="https://github.com/NVIDIA/RULER.git"
RULER_DATA_DIR=""  # set per-tokenizer in ruler_bench()
RULER_VENV="${RULER_DIR}/.venv"

ruler_usage() {
    cat <<'EOF'
RULER benchmark for KV cache quantization quality evaluation

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                KV cache type for K             (default: f16)
  -ctv  TYPE                KV cache type for V             (default: f16)
  --ctx-lengths "N ..."     Context lengths to test         (default: "4096 8192")
  --num-samples N           Samples per task per length     (default: 5)
  --tasks "T ..."           RULER tasks to run              (default: "niah_single_1 niah_single_2 vt cwe")
  --tokenizer PATH          HF tokenizer name/path          (default: auto-detect from model)
  --cli-bin PATH            Path to llama-cli binary        (default: build/bin/llama-cli)
  --seed N                  Random seed                     (default: 42)
  -q, --quiet               Suppress per-sample output
  --no-print-cmd            Suppress llama-cli command logging (printed by default)
  --csv FILE                Write per-cell CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-cli (e.g. -ngl 99, --threads 8, -fa 1).

Supported tasks (from RULER synthetic.yaml):
  niah_single_1    Single NIAH, noise haystack, word keys, number values
  niah_single_2    Single NIAH, essay haystack, word keys, number values
  niah_single_3    Single NIAH, essay haystack, word keys, UUID values
  niah_multikey_1  Multi-key NIAH (4 keys), essay haystack
  niah_multikey_2  Multi-key NIAH (8 keys), essay haystack
  niah_multikey_3  Multi-key NIAH (16 keys), essay haystack
  niah_mk_k8q4_noise    Multi-key NIAH (8 keys, 4 queries), noise haystack
  niah_mk_k8q4v2_noise  Multi-key NIAH (8 keys, 4 queries, 2 values), noise haystack
  vt               Variable tracking (1 chain, 4 hops)
  cwe              Common words extraction

Example:
  ./tests/ruler-bench.sh -m model.gguf -ctk pq3_0 -ctv pq3_0 --ctx-lengths "4096 8192" -ngl 99 -fa 1
EOF
}

# ── ensure RULER repo is cloned ──────────────────────────────────────────────
_ruler_ensure_repo() {
    if [[ ! -d "${RULER_DIR}/scripts/data/synthetic" ]]; then
        echo "Cloning NVIDIA/RULER into ${RULER_DIR} ..."
        git clone --depth 1 "${RULER_REPO}" "${RULER_DIR}" 2>&1 | tail -2
    fi
}

# ── ensure venv + Python dependencies ────────────────────────────────────────
_ruler_ensure_uv() {
    local uv_dir="${RULER_DIR}/.uv"
    local uv_bin="${uv_dir}/uv"
    if [[ -x "$uv_bin" ]]; then
        return 0
    fi
    echo "Installing uv into ${uv_dir} ..."
    UV_INSTALL_DIR="$uv_dir" curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    [[ -x "$uv_bin" ]]
}

_ruler_ensure_venv() {
    if [[ ! -f "${RULER_VENV}/bin/activate" ]]; then
        echo "Creating Python venv at ${RULER_VENV} ..."
        local uv_bin="${RULER_DIR}/.uv/uv"
        if [[ "${RULER_USE_UV:-0}" == "1" ]]; then
            if command -v uv &>/dev/null; then
                echo "RULER_USE_UV=1, using uv ..."
                uv venv "${RULER_VENV}"
            elif _ruler_ensure_uv; then
                echo "RULER_USE_UV=1, using local uv ..."
                "$uv_bin" venv "${RULER_VENV}"
            else
                echo "ERROR: RULER_USE_UV=1 but failed to install uv" >&2
                return 1
            fi
        elif python3 -m venv "${RULER_VENV}" 2>/dev/null; then
            : # success
        elif command -v uv &>/dev/null; then
            echo "python3 -m venv unavailable, using uv ..."
            uv venv "${RULER_VENV}"
        elif _ruler_ensure_uv; then
            echo "python3 -m venv unavailable, using local uv ..."
            "$uv_bin" venv "${RULER_VENV}"
        else
            echo "python3 -m venv unavailable, creating manual venv ..."
            local py_bin
            py_bin="$(command -v python3)"
            mkdir -p "${RULER_VENV}/bin"
            ln -sf "$py_bin" "${RULER_VENV}/bin/python3"
            ln -sf "$py_bin" "${RULER_VENV}/bin/python"
            cat > "${RULER_VENV}/bin/activate" <<'ACTIVATE'
VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
ACTIVATE
            # shellcheck disable=SC1091
            source "${RULER_VENV}/bin/activate"
            echo "Bootstrapping pip via get-pip.py ..."
            curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - 2>&1 | tail -3
        fi
    fi
    # shellcheck disable=SC1091
    source "${RULER_VENV}/bin/activate"
}

_ruler_ensure_deps() {
    _ruler_ensure_venv

    local missing=0
    for pkg in wonderwords nltk numpy transformers yaml html2text tenacity; do
        python3 -c "import ${pkg}" 2>/dev/null || missing=1
    done
    if (( missing )); then
        echo "Installing RULER Python dependencies into venv ..."
        python3 -m pip install --quiet wonderwords nltk numpy transformers pyyaml html2text beautifulsoup4 tenacity 2>&1 | tail -5
        python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true
    fi

    local essay_json="${RULER_DIR}/scripts/data/synthetic/json/PaulGrahamEssays.json"
    if [[ ! -f "$essay_json" ]]; then
        echo "Downloading Paul Graham essays for RULER haystack (timeout 60s) ..."
        (cd "${RULER_DIR}/scripts/data/synthetic/json" && timeout 60 python3 download_paulgraham_essay.py 2>&1 | tail -5) || {
            echo "WARNING: download_paulgraham_essay.py failed, creating minimal fallback ..."
            python3 -c "
import json, pathlib
text = ' '.join(['The quick brown fox jumps over the lazy dog.'] * 5000)
pathlib.Path('${essay_json}').write_text(json.dumps({'text': text}))
print('Created fallback PaulGrahamEssays.json')
"
        }
    fi
}

# ── generate RULER task data ─────────────────────────────────────────────────
_ruler_generate_data() {
    local task_name=$1
    local ctx_len=$2
    local num_samples=$3
    local tokenizer=$4
    local seed=$5
    local out_dir="${RULER_DATA_DIR}/${task_name}/${ctx_len}"

    mkdir -p "$out_dir"
    local out_file="${out_dir}/validation.jsonl"

    if [[ -f "$out_file" ]] && [[ $(wc -l < "$out_file") -ge $num_samples ]]; then
        return 0
    fi

    local task_type args_extra=""
    case "$task_name" in
        niah_single_1)
            task_type="niah"
            args_extra="--type_haystack noise --type_needle_k words --type_needle_v numbers --num_needle_k 1 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_single_2)
            task_type="niah"
            args_extra="--type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k 1 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_single_3)
            task_type="niah"
            args_extra="--type_haystack essay --type_needle_k words --type_needle_v uuids --num_needle_k 1 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_multikey_1)
            task_type="niah"
            args_extra="--type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k 4 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_multikey_2)
            task_type="niah"
            args_extra="--type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k 8 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_multikey_3)
            task_type="niah"
            args_extra="--type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k 16 --num_needle_v 1 --num_needle_q 1"
            ;;
        niah_mk_k8q4_noise)
            task_type="niah"
            args_extra="--type_haystack noise --type_needle_k words --type_needle_v numbers --num_needle_k 8 --num_needle_v 1 --num_needle_q 4"
            ;;
        niah_mk_k8q4v2_noise)
            task_type="niah"
            args_extra="--type_haystack noise --type_needle_k words --type_needle_v numbers --num_needle_k 8 --num_needle_v 2 --num_needle_q 4"
            ;;
        vt)
            task_type="variable_tracking"
            args_extra="--type_haystack noise --num_chains 1 --num_hops 4"
            ;;
        cwe)
            task_type="common_words_extraction"
            args_extra="--freq_cw 30 --freq_ucw 3 --num_cw 10"
            ;;
        *)
            echo "ERROR: unknown task '$task_name'" >&2
            return 1
            ;;
    esac

    local script="${RULER_DIR}/scripts/data/synthetic/${task_type}.py"
    if [[ ! -f "$script" ]]; then
        echo "ERROR: RULER script not found: $script" >&2
        return 1
    fi

    # Read template + tokens_to_generate from constants.py
    local tokens_to_gen
    local template answer_prefix
    tokens_to_gen=$(python3 -c "
import sys; sys.path.insert(0, '${RULER_DIR}/scripts/data/synthetic')
from constants import TASKS
print(TASKS['${task_type}']['tokens_to_generate'])
")
    template=$(python3 -c "
import sys; sys.path.insert(0, '${RULER_DIR}/scripts/data/synthetic')
from constants import TASKS
t = TASKS['${task_type}']
print(t['template'] + t.get('answer_prefix', ''))
")

    python3 "$script" \
        --save_dir "$RULER_DATA_DIR" \
        --save_name "${task_name}/${ctx_len}" \
        --tokenizer_path "$tokenizer" \
        --tokenizer_type hf \
        --max_seq_length "$ctx_len" \
        --tokens_to_generate "$tokens_to_gen" \
        --num_samples "$num_samples" \
        --random_seed "$seed" \
        --remove_newline_tab \
        --template "$template" \
        $args_extra 2>&1 | tail -5

    if [[ ! -f "$out_file" ]]; then
        echo "ERROR: data generation failed for ${task_name} @ ${ctx_len}" >&2
        return 1
    fi
}

# ── run inference on a single sample ─────────────────────────────────────────
_ruler_infer_single() {
    local cli_bin=$1 model=$2 ctk=$3 ctv=$4 ctx_len=$5
    shift 5
    local print_cmd=${RULER_PRINT_CMD:-0}
    local extra_args=()
    local prompt=""
    local answer_prefix=""
    local tokens_to_gen=128

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prompt)         prompt="$2";         shift 2 ;;
            --answer-prefix)  answer_prefix="$2";  shift 2 ;;
            --tokens-to-gen)  tokens_to_gen="$2";  shift 2 ;;
            *)                extra_args+=("$1");  shift ;;
        esac
    done

    local full_prompt="${prompt}${answer_prefix}"
    local tmp_prompt
    tmp_prompt=$(mktemp)
    printf '%s' "$full_prompt" > "$tmp_prompt"

    local output
    local rc=0
    local tmp_stderr
    tmp_stderr=$(mktemp)

    if (( print_cmd == 1 )); then
        printf '  command: ' >&2
        printf '%q ' "$cli_bin" -m "$model" -ctk "$ctk" -ctv "$ctv" -c $((ctx_len + tokens_to_gen + 256)) -n "$tokens_to_gen" --temp 0 -f "$tmp_prompt" --no-display-prompt --no-conversation >&2
        printf '%q ' "${extra_args[@]}" >&2
        printf '\n' >&2
    fi

    output=$(
        "$cli_bin" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -c $((ctx_len + tokens_to_gen + 256)) \
        -n "$tokens_to_gen" \
        --temp 0 \
        -f "$tmp_prompt" \
        --no-display-prompt \
        --no-conversation \
        "${extra_args[@]}" 2>"$tmp_stderr"
    ) || rc=$?

    rm -f "$tmp_prompt"

    if (( rc != 0 )); then
        echo "ERROR: $cli_bin exited with code $rc" >&2
        cat "$tmp_stderr" >&2
        rm -f "$tmp_stderr"
        return 1
    fi

    rm -f "$tmp_stderr"
    echo "$output"
}

# ── scoring: RULER string_match_all ──────────────────────────────────────────
_ruler_score() {
    local prediction=$1
    shift
    local -a refs=("$@")
    local matched=0
    local total=${#refs[@]}

    for ref in "${refs[@]}"; do
        local ref_lower="${ref,,}"
        local pred_lower="${prediction,,}"
        if [[ "$pred_lower" == *"$ref_lower"* ]]; then
            matched=$((matched + 1))
        fi
    done

    if (( total > 0 )); then
        awk "BEGIN { printf \"%.4f\", $matched / $total }"
    else
        echo "0.0000"
    fi
}

# ── compute mean±stdev from space-separated values ──────────────────────────
_ruler_mean_stdev() {
    echo "$1" | awk '{
        n = NF; if (n == 0) { print "- -"; exit }
        sum = 0; for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        sumsq = 0; for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        sd = (n > 1) ? sqrt(sumsq / (n - 1)) : 0
        printf "%.1f %.1f", mean * 100, sd * 100
    }'
}

# ── auto-detect tokenizer from model path ───────────────────────────────────
_ruler_detect_tokenizer() {
    local model_path=$1
    local model_base
    model_base=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')

    if [[ "$model_base" == *mistral* ]]; then
        echo "mistralai/Mistral-7B-Instruct-v0.3"
    elif [[ "$model_base" == *llama*3* ]] || [[ "$model_base" == *meta-llama* ]]; then
        echo "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif [[ "$model_base" == *llama*2* ]]; then
        echo "meta-llama/Llama-2-7b-chat-hf"
    elif [[ "$model_base" == *qwen* ]]; then
        echo "Qwen/Qwen2-7B-Instruct"
    elif [[ "$model_base" == *phi* ]]; then
        echo "microsoft/Phi-3-mini-128K-instruct"
    else
        echo "mistralai/Mistral-7B-Instruct-v0.3"
    fi
}

# ── main benchmark function ─────────────────────────────────────────────────
ruler_bench() {
    local cli_bin="build/bin/llama-cli"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local ctx_lengths="4096 8192"
    local num_samples=5
    local tasks="niah_single_1 niah_single_2 vt cwe"
    local tokenizer=""
    local seed=42
    local extra_args=()
    local quiet=0
    local print_cmd=1
    local csv_file=""
    local log_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        ruler_usage; return 0 ;;
            -m|--model)       model="$2";         shift 2 ;;
            -ctk)             ctk="$2";           shift 2 ;;
            -ctv)             ctv="$2";           shift 2 ;;
            --ctx-lengths)    ctx_lengths="$2";   shift 2 ;;
            --num-samples)    num_samples="$2";   shift 2 ;;
            --tasks)          tasks="$2";         shift 2 ;;
            --tokenizer)      tokenizer="$2";     shift 2 ;;
            --cli-bin)        cli_bin="$2";       shift 2 ;;
            --seed)           seed="$2";          shift 2 ;;
            -q|--quiet)       quiet=1;            shift ;;
            --no-print-cmd)   print_cmd=0;        shift ;;
            --csv)            csv_file="$2";      shift 2 ;;
            --log)            log_file="$2";      shift 2 ;;
            *)                extra_args+=("$1"); shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2
        ruler_usage >&2
        return 1
    fi

    if [[ ! -x "$cli_bin" ]]; then
        echo "ERROR: llama-cli binary not found at '$cli_bin'" >&2
        echo "       Build with: cmake --build build --target llama-cli" >&2
        return 1
    fi

    if [[ -z "$tokenizer" ]]; then
        tokenizer=$(_ruler_detect_tokenizer "$model")
        echo "Auto-detected tokenizer: $tokenizer"
    fi

    # Per-tokenizer data dir so switching models doesn't reuse stale prompts
    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    RULER_DATA_DIR="${RULER_DIR}/_data/${tok_slug}"
    export RULER_PRINT_CMD=$print_cmd

    # Auto-generate log file from csv name, or use explicit --log
    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _ruler_ensure_repo
    _ruler_ensure_deps

    # ── banner ───────────────────────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo " RULER Benchmark"
    echo "=========================================="
    echo "  Model:       $(basename "$model")"
    echo "  K type:      $ctk"
    echo "  V type:      $ctv"
    echo "  Contexts:    $ctx_lengths"
    echo "  Tasks:       $tasks"
    echo "  Samples:     $num_samples per task per length"
    echo "  Tokenizer:   $tokenizer"
    echo "  Seed:        $seed"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:       ${extra_args[*]}"
    echo "=========================================="
    echo ""

    # ── generate data for all tasks x lengths ────────────────────────────────
    echo "Generating RULER task data ..."
    for task in $tasks; do
        for ctx_len in $ctx_lengths; do
            echo -n "  ${task} @ ${ctx_len} ... "
            if _ruler_generate_data "$task" "$ctx_len" "$num_samples" "$tokenizer" "$seed"; then
                echo "OK"
            else
                echo "FAILED"
            fi
        done
    done
    echo ""

    # ── determine task type for scoring ──────────────────────────────────────
    _ruler_task_type() {
        case "$1" in
            niah_*)     echo "niah" ;;
            vt)         echo "variable_tracking" ;;
            cwe)        echo "common_words_extraction" ;;
            fwe)        echo "freq_words_extraction" ;;
            qa_*)       echo "qa" ;;
            *)          echo "niah" ;;
        esac
    }

    # ── run inference + scoring ──────────────────────────────────────────────
    declare -A task_scores task_counts task_score_list
    local global_score_sum=0
    local global_count=0

    for task in $tasks; do
        local task_type
        task_type=$(_ruler_task_type "$task")
        local tokens_to_gen
        tokens_to_gen=$(python3 -c "
import sys; sys.path.insert(0, '${RULER_DIR}/scripts/data/synthetic')
from constants import TASKS
print(TASKS['${task_type}']['tokens_to_generate'])
" 2>/dev/null || echo 128)

        for ctx_len in $ctx_lengths; do
            local data_file="${RULER_DATA_DIR}/${task}/${ctx_len}/validation.jsonl"
            if [[ ! -f "$data_file" ]]; then
                echo "SKIP: no data for ${task} @ ${ctx_len}"
                continue
            fi

            local cell_key="${task}_${ctx_len}"
            task_scores[$cell_key]=0
            task_counts[$cell_key]=0
            task_score_list[$cell_key]=""
            local sample_idx=0

            while IFS= read -r line; do
                if (( sample_idx >= num_samples )); then
                    break
                fi
                local input_text answer_prefix
                input_text=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['input'])" 2>/dev/null)
                answer_prefix=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('answer_prefix',''))" 2>/dev/null)

                local -a outputs=()
                while IFS= read -r ans; do
                    outputs+=("$ans")
                done < <(echo "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
for o in d['outputs']:
    print(o)
" 2>/dev/null)

                if [[ -z "$input_text" ]] || [[ ${#outputs[@]} -eq 0 ]]; then
                    sample_idx=$((sample_idx + 1))
                    continue
                fi

                local prediction
                prediction=$(_ruler_infer_single "$cli_bin" "$model" "$ctk" "$ctv" "$ctx_len" \
                    --prompt "$input_text" \
                    --answer-prefix "$answer_prefix" \
                    --tokens-to-gen "$tokens_to_gen" \
                    "${extra_args[@]}")

                local score
                score=$(_ruler_score "$prediction" "${outputs[@]}")

                task_scores[$cell_key]=$(awk "BEGIN { printf \"%.4f\", ${task_scores[$cell_key]} + $score }")
                task_counts[$cell_key]=$((${task_counts[$cell_key]} + 1))
                task_score_list[$cell_key]="${task_score_list[$cell_key]} $score"
                local num_refs=${#outputs[@]}
                global_score_sum=$(awk "BEGIN { printf \"%.4f\", $global_score_sum + ($score * $num_refs) }")
                global_count=$((global_count + num_refs))

                sample_idx=$((sample_idx + 1))

                if [[ $quiet -eq 0 ]]; then
                    local pct
                    pct=$(awk "BEGIN { printf \"%.0f\", $score * 100 }")
                    printf "  [%s @ %s] sample %d: %s%%  expected=%s  got=%.80s\n" \
                        "$task" "$ctx_len" "$sample_idx" "$pct" \
                        "$(IFS=,; echo "${outputs[*]}")" \
                        "$prediction"
                fi
            done < "$data_file"
        done
    done

    # ── results table ────────────────────────────────────────────────────────
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

    printf "%-24s" "Task"
    for ctx_len in $ctx_lengths; do
        printf "%16s" "${ctx_len}"
    done
    printf "%16s\n" "Avg"

    local total_cols=$(( $(echo "$ctx_lengths" | wc -w) + 2 ))
    printf '%*s\n' $(( total_cols * 16 + 24 )) '' | tr ' ' '-'

    # CSV header
    if [[ -n "$csv_file" ]]; then
        echo "model,cache_k,cache_v,task,ctx_len,samples,mean_pct,stdev_pct" > "$csv_file"
    fi

    declare -A ruler_task_scores_export
    local model_base
    model_base=$(basename "$model")

    for task in $tasks; do
        printf "%-24s" "$task"
        local row_scores=""

        for ctx_len in $ctx_lengths; do
            local cell_key="${task}_${ctx_len}"
            local cnt=${task_counts[$cell_key]:-0}
            local scores_list="${task_score_list[$cell_key]:-}"
            if (( cnt > 0 )); then
                local ms
                ms=$(_ruler_mean_stdev "$scores_list")
                local cell_mean="${ms% *}"
                local cell_sd="${ms#* }"
                printf "%15s%%" "${cell_mean}±${cell_sd}"
                row_scores="$row_scores $scores_list"
                if [[ -n "$csv_file" ]]; then
                    echo "${model_base},${ctk},${ctv},${task},${ctx_len},${cnt},${cell_mean},${cell_sd}" >> "$csv_file"
                fi
            else
                printf "%15s%%" "-"
            fi
        done

        local row_avg
        if [[ -n "$row_scores" ]]; then
            local rms
            rms=$(_ruler_mean_stdev "$row_scores")
            row_avg="${rms% *}±${rms#* }"
        else
            row_avg="-"
        fi
        printf "%15s%%\n" "$row_avg"
        ruler_task_scores_export[$task]="${row_avg%%±*}"
    done

    echo ""

    # ── summary ──────────────────────────────────────────────────────────────
    if (( global_count > 0 )); then
        ruler_global_score=$(awk "BEGIN { printf \"%.1f\", ($global_score_sum / $global_count) * 100 }")
    else
        ruler_global_score="N/A"
    fi

    ruler_task_scores=()
    if [[ ${#ruler_task_scores_export[@]} -gt 0 ]]; then
        for k in "${!ruler_task_scores_export[@]}"; do
            ruler_task_scores[$k]="${ruler_task_scores_export[$k]}"
        done
    fi

    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo "  Overall accuracy: ${ruler_global_score}%  (${global_count} retrievals)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

# ── auto-run when executed directly (not sourced) ─────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    ruler_bench "$@"
fi
