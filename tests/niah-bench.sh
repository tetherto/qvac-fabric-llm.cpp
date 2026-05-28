#!/usr/bin/env bash
#
# NIAH (Needle In A Haystack, gkamradt-style) benchmark for KV cache
# quantization quality.
#
# Sweeps a 2-D grid of (context_length, depth_percent) cells: a single
# "needle" sentence is inserted at a known depth into a long Paul Graham
# haystack, and the model is asked to retrieve it. Score per cell is
# binary recall (1 if the answer keyword appears in the model's output).
#
# Output is the canonical NIAH heatmap as a (ctx × depth) text grid plus
# a CSV row per cell. The orchestrator (test-kv-cache-niah.py) stacks
# these per-cell heatmaps across cache configs so you can see exactly
# where TBQ / PQ / q4_0 start to lose retrieval at each context length.
#
# Why a separate bench (not folded into RULER)? RULER's `niah_single_*`
# evaluate retrieval at one fixed depth (50% of the haystack) and vary
# along orthogonal axes (needle type, multi-key, multi-query); the
# canonical NIAH benchmark from the Google TurboQuant blog post explicitly
# sweeps depth × ctx and renders the heatmap that's commonly cited as
# "NIAH performance". Folding into RULER would lose the depth axis.
#
# Usage (direct):
#   ./tests/niah-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/niah-bench.sh
#   niah_bench -m model.gguf -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=kv_cache_bench_common.sh
source "${SCRIPT_DIR}/kv_cache_bench_common.sh"

NH_DIR="${SCRIPT_DIR}/niah"
NH_VENV="${NH_DIR}/.venv"
NH_PREPARE="${SCRIPT_DIR}/niah-prepare.py"
NH_SCORE="${SCRIPT_DIR}/niah-score.py"

# Re-use RULER's Paul Graham essay dump if it exists — saves a second
# network fetch and means NIAH inherits whatever fallback corpus RULER's
# PG-download step ended up with.
NH_RULER_HAYSTACK="${SCRIPT_DIR}/ruler/scripts/data/synthetic/json/PaulGrahamEssays.json"

NH_DATA_DIR=""

# Canonical gkamradt-style grid: 5 ctx × 5 depths = 25 cells. Override with
# --ctx-lengths / --depth-percents. The original gkamradt visualisation
# uses 7 depths (0/11/22/...); we use 5 to keep the heatmap legible in a
# fixed-width terminal and the wall time bounded.
NH_DEFAULT_CTX="2048 4096 8192 16384 32768"
NH_DEFAULT_DEPTHS="0 25 50 75 100"

niah_usage() {
    cat <<'EOF'
NIAH (Needle In A Haystack) benchmark for KV cache quantization quality

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                K cache type                        (default: f16)
  -ctv  TYPE                V cache type                        (default: f16)
  --ctx-lengths "N ..."     Context lengths to sweep            (default: "2048 4096 8192 16384 32768")
                            The largest must fit in the server's -c budget;
                            we set -c = max(ctx_lengths) + 256.
  --depth-percents "D ..."  Needle depths (0..100)              (default: "0 25 50 75 100")
  --tokenizer PATH          HF tokenizer name/path              (default: auto-detect)
  --server-bin PATH         Path to llama-server binary         (default: build/bin/llama-server)
  --cli-bin PATH            Deprecated alias for --server-bin.
  --needle "TEXT"           Override the needle sentence
                            (default: Greg Kamradt's San Francisco sandwich)
  --question "TEXT"         Override the retrieval question
  --answer-keys "K1|K2|..." Pipe-separated answer-key substrings (default: "dolores park|sandwich and sit")
  --haystack-path PATH      Use a custom haystack JSON/text file
  -q, --quiet               Suppress per-sample output
  --csv FILE                Write per-cell CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-server.

Example:
  ./tests/niah-bench.sh -m models/Llama-3.1-8B-Instruct-Q8_0.gguf \
      -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1
EOF
}

# ── venv + deps (numpy / transformers / jinja2 mirrors leval) ───────────────
_niah_ensure_deps() {
    _kv_ensure_venv "${NH_VENV}" "${NH_USE_UV:-0}" "${NH_DIR}/.uv"
    _kv_ensure_pip_deps "NIAH" \
        "numpy transformers jinja2" \
        "numpy transformers jinja2"
}

# Server lifecycle (_kv_pick_port / _kv_start_server / _kv_stop_server) and
# /completion inference (_kv_infer_completion) live in kv_cache_bench_common.sh.

# ── prep the (ctx × depth) grid one-shot ────────────────────────────────────
# niah-prepare.py is one-shot for the entire grid (vs. the per-task pattern in
# LongBench / ZeroSCROLLS / L-Eval) because NIAH's input is parameterised by
# (ctx, depth) on a single needle/haystack/question — there's no per-task
# divergence to amortise.
_niah_prepare_grid() {
    local tokenizer=$1 ctx_lengths=$2 depth_percents=$3
    local haystack_path=$4 needle=$5 question=$6
    shift 6
    local -a answer_keys=("$@")

    local prep_file="${NH_DATA_DIR}/cells.jsonl"
    if [[ -s "$prep_file" ]]; then
        return 0
    fi

    echo "Preparing NIAH (ctx × depth) cells ..."
    local -a cmd=(
        python3 "$NH_PREPARE"
        --tokenizer "$tokenizer"
        --ctx-lengths "$ctx_lengths"
        --depth-percents "$depth_percents"
        --output-file "$prep_file"
    )
    if [[ -n "$haystack_path" ]]; then
        cmd+=(--haystack-path "$haystack_path")
    fi
    if [[ -f "$NH_RULER_HAYSTACK" ]]; then
        cmd+=(--ruler-haystack-cache "$NH_RULER_HAYSTACK")
    fi
    if [[ -n "$needle" ]]; then
        cmd+=(--needle "$needle")
    fi
    if [[ -n "$question" ]]; then
        cmd+=(--question "$question")
    fi
    if (( ${#answer_keys[@]} > 0 )); then
        cmd+=(--answer-keys "${answer_keys[@]}")
    fi
    "${cmd[@]}" 2>&1 | tail -5
    [[ -s "$prep_file" ]]
}

# ── per-cell scoring (case-insensitive substring match) ─────────────────────
_niah_score() {
    local prediction=$1 references_json=$2
    local payload
    payload=$(python3 -c '
import json, sys
print(json.dumps({"prediction": sys.argv[1], "references": json.loads(sys.argv[2])}))' \
        "$prediction" "$references_json")
    python3 "$NH_SCORE" --stdin <<<"$payload"
}

niah_bench() {
    local server_bin="build/bin/llama-server"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local ctx_lengths="$NH_DEFAULT_CTX"
    local depth_percents="$NH_DEFAULT_DEPTHS"
    local tokenizer=""
    local needle=""
    local question=""
    local answer_keys="dolores park|sandwich and sit"
    local haystack_path=""
    local quiet=0
    local csv_file=""
    local log_file=""
    local -a extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        niah_usage; return 0 ;;
            -m|--model)       model="$2";          shift 2 ;;
            -ctk)             ctk="$2";            shift 2 ;;
            -ctv)             ctv="$2";            shift 2 ;;
            --ctx-lengths)    ctx_lengths="$2";    shift 2 ;;
            --depth-percents) depth_percents="$2"; shift 2 ;;
            --tokenizer)      tokenizer="$2";      shift 2 ;;
            --cli-bin|--server-bin) server_bin="$2"; shift 2 ;;
            --needle)         needle="$2";         shift 2 ;;
            --question)       question="$2";       shift 2 ;;
            --answer-keys)    answer_keys="$2";    shift 2 ;;
            --haystack-path)  haystack_path="$2";  shift 2 ;;
            -q|--quiet)       quiet=1;             shift ;;
            --csv)            csv_file="$2";       shift 2 ;;
            --log)            log_file="$2";       shift 2 ;;
            *)                extra_args+=("$1");  shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2; niah_usage >&2; return 1
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

    local -a answer_keys_arr=()
    IFS='|' read -r -a answer_keys_arr <<<"$answer_keys"

    # Per-(tokenizer × grid) data cache. Shared across cache-config runs so
    # the tokenisation/splice work is amortised, since the prompts depend
    # only on (tokenizer, grid, needle), not on (ctk, ctv).
    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    local grid_slug
    grid_slug=$(echo "${ctx_lengths}__${depth_percents}" | tr ' ' '_')
    NH_DATA_DIR="${NH_DIR}/_data/${tok_slug}/${grid_slug}"
    mkdir -p "$NH_DATA_DIR"

    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _niah_ensure_deps

    echo ""
    echo "=========================================="
    echo " NIAH (Needle In A Haystack) Benchmark"
    echo "=========================================="
    echo "  Model:        $(basename "$model")"
    echo "  K type:       $ctk"
    echo "  V type:       $ctv"
    echo "  ctx_lengths:  $ctx_lengths"
    echo "  depths (%):   $depth_percents"
    echo "  Tokenizer:    $tokenizer"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:        ${extra_args[*]}"
    echo "=========================================="
    echo ""

    _niah_prepare_grid "$tokenizer" "$ctx_lengths" "$depth_percents" \
        "$haystack_path" "$needle" "$question" "${answer_keys_arr[@]}" || {
        echo "ERROR: niah-prepare.py produced no output" >&2
        return 1
    }
    echo ""

    local prep_file="${NH_DATA_DIR}/cells.jsonl"

    # n_ctx = max(ctx_lengths) + slack (BOS / chat-template overhead).
    # 256 is generous given the chat templates we use top out at ~30 tokens.
    local max_ctx
    max_ctx=$(echo "$ctx_lengths" | tr ' ' '\n' | sort -n | tail -1)
    local n_ctx=$(( max_ctx + 256 ))

    trap '_kv_stop_server' EXIT INT TERM
    _kv_start_server "$server_bin" "$model" "$ctk" "$ctv" "$n_ctx" "${extra_args[@]}" || return 1

    # We store scores in an associative array keyed by "<ctx>_<depth>" so we
    # can render the heatmap at the end without re-reading the JSONL.
    declare -A cell_score
    local total=0 hits=0
    while IFS= read -r line; do
        local input_text references_json ctx depth max_gen
        input_text=$(jq -r '.input' <<<"$line")
        references_json=$(jq -c '.references' <<<"$line")
        ctx=$(jq -r '.ctx_len' <<<"$line")
        # niah-prepare.py writes depth_percent as a JSON float (e.g. 25.0).
        # The rendering loop iterates the user-supplied --depth-percents
        # string ("0 25 50 75 100"), which are bare ints. Canonicalise both
        # sides with awk '%g' so cell_score lookups match — otherwise the
        # heatmap shows "-" everywhere even though scoring succeeded.
        depth=$(awk -v d="$(jq -r '.depth_percent' <<<"$line")" 'BEGIN{printf "%g", d}')
        max_gen=$(jq -r '.max_gen' <<<"$line")
        if [[ -z "$input_text" ]]; then continue; fi

        local prediction
        prediction=$(_kv_infer_completion "$max_gen" "$input_text" "$KV_SERVER_URL") || {
            echo "ERROR: inference failed for ctx=$ctx depth=$depth" >&2
            continue
        }
        local score
        score=$(_niah_score "$prediction" "$references_json") || {
            echo "ERROR: scoring failed for ctx=$ctx depth=$depth" >&2
            continue
        }
        local key="${ctx}_${depth}"
        cell_score[$key]=$score
        total=$((total + 1))
        if [[ "$score" == "1.000000" ]]; then hits=$((hits + 1)); fi

        if (( quiet == 0 )); then
            local pct
            pct=$(awk "BEGIN { printf \"%.0f\", $score * 100 }")
            # UTF-8-safe 60-byte truncation — see longbench-bench.sh.
            local short_pred
            short_pred=$(printf '%s' "$prediction" | tr '\n' ' ' \
                | head -c 60 | iconv -c -f UTF-8 -t UTF-8 2>/dev/null || true)
            printf "  [ctx=%6d depth=%6s] %s%%  got=%s\n" \
                "$ctx" "${depth}%" "$pct" "$short_pred"
        fi
    done < "$prep_file"

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
    echo " NIAH heatmap: K=$ctk(${_bpw_k}bpw)  V=$ctv(${_bpw_v}bpw)  Model=$(basename "$model")"
    echo "=========================================="
    echo ""

    printf "%-8s" "ctx ↓"
    for d in $depth_percents; do
        printf " %5s%%" "$d"
    done
    printf "  %5s\n" "row-avg"

    if [[ -n "$csv_file" ]]; then
        echo "model,cache_k,cache_v,ctx_len,depth_percent,score" > "$csv_file"
    fi

    local model_base
    model_base=$(basename "$model")

    local grid_sum=0 grid_count=0
    for c in $ctx_lengths; do
        printf "%-8d" "$c"
        local row_sum=0 row_count=0
        for d in $depth_percents; do
            local d_canon
            d_canon=$(awk -v d="$d" 'BEGIN{printf "%g", d}')
            local key="${c}_${d_canon}"
            if [[ -n "${cell_score[$key]:-}" ]]; then
                local s=${cell_score[$key]}
                local s_pct
                s_pct=$(awk "BEGIN { printf \"%.0f\", $s * 100 }")
                printf " %5s%%" "$s_pct"
                row_sum=$(awk "BEGIN { printf \"%.6f\", $row_sum + $s }")
                row_count=$((row_count + 1))
                grid_sum=$(awk "BEGIN { printf \"%.6f\", $grid_sum + $s }")
                grid_count=$((grid_count + 1))
                if [[ -n "$csv_file" ]]; then
                    echo "${model_base},${ctk},${ctv},${c},${d},${s_pct}" >> "$csv_file"
                fi
            else
                printf " %6s" "-"
            fi
        done
        if (( row_count > 0 )); then
            local row_avg
            row_avg=$(awk "BEGIN { printf \"%.1f\", $row_sum / $row_count * 100 }")
            printf "  %5s%%\n" "$row_avg"
        else
            printf "  %5s\n" "-"
        fi
    done

    echo ""
    if (( grid_count > 0 )); then
        niah_global_score=$(awk "BEGIN { printf \"%.1f\", $grid_sum / $grid_count * 100 }")
    else
        niah_global_score="N/A"
    fi
    echo "  Grid average:    ${niah_global_score}%  (${hits}/${total} cells passed)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    niah_bench "$@"
fi
