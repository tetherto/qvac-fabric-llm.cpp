# shellcheck shell=bash
# Shared bash helpers for the KV-cache eval bench scripts.
#
# Sourced by tests/{longbench,zeroscrolls,leval,ruler}-bench.sh.
#
# Per-cell state lives in a small set of KV_SERVER_* globals — there is only
# one llama-server alive per cell, so a single namespace is enough.

KV_SERVER_PORT=""
KV_SERVER_PID=""
KV_SERVER_URL=""
KV_SERVER_LOG=""

# ── Pick a free TCP port in the high-ephemeral range ────────────────────────
# Random in [20000, 60000). Two parallel cells from the orchestrator each
# pick their own port; collision probability is ~2/40000 per pair so we
# don't bother with retry logic. If the bind fails the server log will say
# so and the calling code raises an error.
_kv_pick_port() {
    awk -v seed="$$$(date +%N)" 'BEGIN { srand(seed); printf "%d\n", 20000 + int(rand() * 40000) }'
}

# ── Start one persistent llama-server for this cell ─────────────────────────
# Loaded once per (model, ctk, ctv) cell; reused across all samples in the
# cell, saving the ~3s/sample model-reload overhead a per-sample llama-cli
# spawn would cost. The server takes the same -ngl/-fa/--split-mode args
# the orchestrator forwards.
#
# Caller computes n_ctx itself because each eval has a different slack
# formula on top of max_length (L-Eval +64+512, LongBench +512+1024,
# ZeroSCROLLS +1024+512). Keeping the math in the caller avoids encoding
# eval-specific knowledge in this lib.
#
# Args: $1=server_bin  $2=model  $3=ctk  $4=ctv  $5=n_ctx  [extra server args...]
# Sets KV_SERVER_PORT / KV_SERVER_URL / KV_SERVER_PID / KV_SERVER_LOG on
# success. Returns non-zero if the server never becomes healthy.
_kv_start_server() {
    local server_bin=$1 model=$2 ctk=$3 ctv=$4 n_ctx=$5
    shift 5
    local -a extra=("$@")

    KV_SERVER_PORT=$(_kv_pick_port)
    KV_SERVER_URL="http://127.0.0.1:${KV_SERVER_PORT}"
    KV_SERVER_LOG=$(mktemp)

    echo "Starting llama-server (port=${KV_SERVER_PORT}, n_ctx=${n_ctx}, K=${ctk}, V=${ctv}) ..."
    "$server_bin" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -c "$n_ctx" \
        --host 127.0.0.1 \
        --port "$KV_SERVER_PORT" \
        --no-webui \
        "${extra[@]}" \
        >"$KV_SERVER_LOG" 2>&1 &
    KV_SERVER_PID=$!

    # Poll /health up to 120s. Vulkan first-load on a cold cache can be
    # slow; the pipeline cache from earlier runs makes this ~5s in practice.
    local i
    for ((i=0; i<120; i++)); do
        if ! kill -0 "$KV_SERVER_PID" 2>/dev/null; then
            echo "ERROR: llama-server died during startup (see $KV_SERVER_LOG):" >&2
            tail -20 "$KV_SERVER_LOG" >&2
            return 1
        fi
        if curl -sf "${KV_SERVER_URL}/health" >/dev/null 2>&1; then
            echo "Server ready after ${i}s."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: llama-server did not become healthy within 120s." >&2
    tail -20 "$KV_SERVER_LOG" >&2
    return 1
}

# ── Stop the server (call from a trap) ──────────────────────────────────────
_kv_stop_server() {
    if [[ -n "$KV_SERVER_PID" ]] && kill -0 "$KV_SERVER_PID" 2>/dev/null; then
        kill "$KV_SERVER_PID" 2>/dev/null
        # Give it 5s to shut down gracefully, then SIGKILL.
        local i
        for ((i=0; i<5; i++)); do
            kill -0 "$KV_SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$KV_SERVER_PID" 2>/dev/null || true
    fi
    [[ -n "$KV_SERVER_LOG" && -f "$KV_SERVER_LOG" ]] && rm -f "$KV_SERVER_LOG"
}

# ── Run inference on a single prepared prompt via the persistent server ─────
# The prompt is already chat-template-wrapped by the caller's *-prepare.py
# helper and is sent as a raw completion request to llama-server's
# /completion endpoint. cache_prompt=false guarantees each sample is a
# fresh context — eval samples are independent, so we don't want the
# server's prefix cache to carry tokens across samples.
#
# Args: $1=tokens_to_gen  $2=prompt  $3=server_url
# Echoes the model's reply text on stdout, returns non-zero on curl failure.
_kv_infer_completion() {
    local tokens_to_gen=$1 prompt=$2 server_url=$3

    # Build the request body with jq so weird characters in `prompt` (quotes,
    # newlines, the chat template's literal "<|...|>" tokens) survive
    # serialisation intact. Inlining shell-quoted JSON would silently corrupt
    # the prompt on any task containing a quote, which several LongBench
    # tasks do.
    #
    # The prompt is fed to jq via --rawfile + stdin instead of --arg because
    # at ctx ≥ ~32k tokens under Llama-3.1 / Qwen tokenizers the decoded
    # prompt text crosses Linux's execve(2) per-arg ceiling (~128 KB) and
    # bash returns "Argument list too long" before jq runs. Piping via stdin
    # is unbounded. Pre-refactor evals capped max_length at 16k so never hit
    # this; NIAH at 32k is what surfaced it.
    #
    # add_special=false because the prepare script already runs the prompt
    # through tokenizer.apply_chat_template, which emits a leading
    # <|begin_of_text|>. With the server's default add_special=true we'd
    # double-BOS every request (verified via /tokenize), which is
    # technically off-distribution input to the model. HF's upstream
    # LongBench pred.py exhibits the same default-double-BOS, so cells
    # collected with this flag flipped are *not* directly comparable
    # against cells collected without it — keep the flag consistent across
    # all cells of any single comparison run.
    # `printf '%s'` (NOT bash <<<, which appends a trailing newline) so the
    # bytes jq sees are byte-identical to what --arg would have produced.
    # Without this the JSON would include a spurious "\n" at the end of the
    # prompt vs. the pre-fix path, breaking cross-comparability of cells.
    local body
    body=$(printf '%s' "$prompt" | jq -nc \
        --rawfile p /dev/stdin \
        --argjson n "$tokens_to_gen" \
        '{prompt: $p, n_predict: $n, temperature: 0, cache_prompt: false, add_special: false}')

    # --max-time covers worst-case: 16k prompt prefill on quantized KV at
    # ~80 t/s ≈ 200s, plus 512-token generation ≈ another 60s, plus
    # margin. 600s is comfortable; we'd rather time out than block forever
    # if the server hangs.
    #
    # `--data-binary @-` (NOT `-d "$body"`) for the same execve(2) ARG_MAX
    # reason as the jq stdin trick above: at 32k+ ctx the JSON body itself
    # crosses the per-argv limit and curl would fail with "Argument list
    # too long". `printf` is a bash builtin so the body never traverses
    # execve(2); it's piped to curl which reads it from stdin unbounded.
    # `--data-binary` (not `--data`) preserves the body bytes verbatim.
    local resp rc=0
    resp=$(printf '%s' "$body" | curl -sf --max-time 600 \
        -X POST "${server_url}/completion" \
        -H "Content-Type: application/json" \
        --data-binary @-) || rc=$?

    if (( rc != 0 )); then
        echo "ERROR: /completion request failed (curl rc=$rc)" >&2
        return 1
    fi
    # llama-server's native /completion returns {"content": "...", ...}.
    jq -r '.content' <<<"$resp"
}

# ── Auto-detect HF tokenizer from model filename ────────────────────────────
# Used by all three bench scripts when --tokenizer is not given on the CLI.
# The default fallback is the Llama-3.1-8B tokenizer because (a) it's the
# paper's primary model and (b) NousResearch's un-gated mirror works without
# HF auth.
#
# Args: $1 = model path
_kv_detect_tokenizer() {
    local model_path=$1
    local base
    base=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    case "$base" in
        *llama-3.1-8b*|*llama-3-8b*|*llama-3.1*|*meta-llama*)
            # NousResearch mirror is un-gated; identical tokenizer to the
            # official meta-llama repo (same SHA on tokenizer.json).
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
        *ministral*)
            echo "mistralai/Ministral-8B-Instruct-2410" ;;
        *mistral*)
            echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
        *qwen2.5*)
            echo "Qwen/Qwen2.5-7B-Instruct" ;;
        *qwen*)
            echo "Qwen/Qwen2-7B-Instruct" ;;
        *llama-2*|*llama2*)
            echo "meta-llama/Llama-2-7b-chat-hf" ;;
        *phi*)
            echo "microsoft/Phi-3-mini-128K-instruct" ;;
        *)
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
    esac
}

# ── Per-task mean / stdev (output as "MEAN_PCT STDEV_PCT") ──────────────────
# Args: $1 = whitespace-separated list of 0..1 scores (e.g. "0.42 0.51 0.38")
# Sample stdev (Bessel's correction). One score → stdev=0.
_kv_mean_stdev() {
    echo "$1" | awk '{
        n = NF; if (n == 0) { print "- -"; exit }
        sum = 0; for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        sumsq = 0; for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        sd = (n > 1) ? sqrt(sumsq / (n - 1)) : 0
        printf "%.1f %.1f", mean * 100, sd * 100
    }'
}

# ── Bootstrap a vendored copy of uv into uv_dir ─────────────────────────────
# Used as a fallback when python3-venv is unavailable on the host (some
# minimal Debian-derived runners lack the venv stdlib but have curl).
#
# Args: $1 = uv_dir (uv binary is dropped at ${uv_dir}/uv)
_kv_ensure_uv() {
    local uv_dir=$1
    local uv_bin="${uv_dir}/uv"
    if [[ -x "$uv_bin" ]]; then
        return 0
    fi
    echo "Installing uv into ${uv_dir} ..."
    UV_INSTALL_DIR="$uv_dir" curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    [[ -x "$uv_bin" ]]
}

# ── Create + source a per-eval Python venv ──────────────────────────────────
# Serialise creation with flock: the orchestrator spawns multiple cells in
# parallel, and on a fresh _data/ dir they all race to create the same
# venv. Without the lock the loser fails with "venv already exists". The
# TOCTOU check inside the lock (test bin/activate again) means only the
# first holder actually creates; the rest see the venv and just `source`.
#
# Args:
#   $1 = venv_dir (e.g. tests/longbench/.venv)
#   $2 = use_uv flag ("1" forces uv; anything else prefers python3 -m venv
#        and falls back to uv only if venv stdlib is unavailable)
#   $3 = uv_dir (where _kv_ensure_uv will drop a vendored uv binary if
#        neither system uv nor python3-venv are available)
_kv_ensure_venv() {
    local venv_dir=$1 use_uv=$2 uv_dir=$3
    local parent_dir
    parent_dir=$(dirname "$venv_dir")
    mkdir -p "$parent_dir"
    local lock="${parent_dir}/.venv.lock"
    (
        flock -x 9
        if [[ -f "${venv_dir}/bin/activate" ]]; then
            exit 0  # another cell already created it while we waited
        fi
        echo "Creating Python venv at ${venv_dir} ..."
        local uv_bin="${uv_dir}/uv"
        if [[ "$use_uv" == "1" ]]; then
            if command -v uv &>/dev/null; then
                uv venv "$venv_dir"
            elif _kv_ensure_uv "$uv_dir"; then
                "$uv_bin" venv "$venv_dir"
            else
                echo "ERROR: USE_UV=1 but failed to install uv" >&2
                exit 1
            fi
        elif python3 -m venv "$venv_dir" 2>/dev/null; then
            : # success
        elif command -v uv &>/dev/null; then
            echo "python3 -m venv unavailable, using uv ..."
            uv venv "$venv_dir"
        elif _kv_ensure_uv "$uv_dir"; then
            echo "python3 -m venv unavailable, using local uv ..."
            "$uv_bin" venv "$venv_dir"
        else
            echo "ERROR: cannot create a Python venv (no python3-venv, no uv)" >&2
            exit 1
        fi
    ) 9>"$lock"
    local rc=$?
    if (( rc != 0 )); then return $rc; fi
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
}

# ── Check importability of a set of packages; pip-install if any missing ───
# Kept separate from _kv_ensure_venv because some bench scripts may want to
# share a venv but have different dep sets (or vice versa). Callers usually
# invoke _kv_ensure_venv first.
#
# Args:
#   $1 = label (used in the "Installing X Python dependencies ..." log line)
#   $2 = importable names, space-separated (passed one-by-one to python3 -c "import N")
#   $3 = pip install names, space-separated (passed verbatim to pip install)
#
# Why two separate lists? Some PyPI packages have different install names
# than import names — e.g. python-Levenshtein installs the `Levenshtein`
# module, and `rouge` (the original package, not rouge-score) has
# importable name `rouge` but ships extra wheels we don't import directly.
_kv_ensure_pip_deps() {
    local label=$1 imports=$2 installs=$3
    local missing=0
    local pkg
    for pkg in $imports; do
        python3 -c "import ${pkg}" 2>/dev/null || missing=1
    done
    if (( missing )); then
        echo "Installing ${label} Python dependencies into venv ..."
        # shellcheck disable=SC2086
        python3 -m pip install --quiet $installs 2>&1 | tail -5
    fi
}
