#!/bin/bash
# usage: run_engine.sh <run dir> <engine up|fab|vllm|sglang> <mode ar|mtp|dflash2> <label> <job args...> [:: <label> <job args...>]...
#   env PROFILE=standard|heavy (server sizing), NP/CTX (llama.cpp slots and context, override the profile's),
#   DEPTHS="1024 4096 ..." (reference style: one depth sweep as the only job)
# Starts the engine's OpenAI-compatible server from the paths in engines.env, sends one unmeasured warm-up request,
# runs each job (jobs are separated by "::") against that one server, appends the results to <run dir>/results.jsonl
# and stops the server. A job is bench_client.py arguments, or "--depths 1024,4096,..." for a depth sweep: one request
# per prompt length (--mode conc --long-output --concurrency 1 --rounds 1 --gen 128; prefill t/s = prompt tokens / time
# to first token). Sharing a server across jobs saves vLLM's and SGLang's ~5-minute start.
# Server sizing:
#   standard: llama.cpp -np 1 -c 16384 for --mode spec, -np 32 -kvu -c 65536 for --mode conc, -np 1 -c 40960 for a
#             depth sweep; vLLM and SGLang 32 running requests, 40960-token context.
#   heavy:    tuned for throughput. llama.cpp -np 64 -kvu -c 262144 (DFlash2: -np 32 -c 131072); vLLM 64 sequences, CUDA graphs captured up to 512
#             tokens (64 sequences x 8 with DFlash2), 8192 batched tokens, 131072 context; SGLang 64 running requests,
#             decode CUDA graphs up to 64, 8192-token prefill chunks, 131072 context (env SGL_MAXREQ lowers the 64).
# For vLLM and SGLang the drafted and accepted token counters are read from /metrics before and after each job and
# appended as a "specstats" row. MAX_JOBS=4 caps the parallel nvcc jobs of their first-use kernel builds (FlashInfer
# JIT): with the default of one per core the Spark ran out of memory.
# Every failed client call (each failed depth step on its own), every request that failed inside a concurrency level
# and a failed warm-up add a row to <run dir>/errors.jsonl through client_check.py: label, step, reason ("error") and
# time, which summarize.py copies into summary.json.
set -u
RUN=$1; ENG=$2; MODE=$3; TAG=$4; shift 3
W=${BENCH_DIR:-$HOME/bench-weekly}; PORT=8098
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
source $W/engines.env
M=${MODELS_DIR:-$HOME/models/qwen38-27b}
GGUF=$M/gguf/Qwen3.8-27B-NVFP4-modelopt.gguf
DGGUF=$M/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf
HF=$M/nvfp4-hf
DHF=$M/dflash2-hf
LOG=$RUN/logs/$TAG.srv.log
PROFILE=${PROFILE:-standard}
if [ -n "${DEPTHS:-}" ]; then set -- "$TAG" --depths "$(echo $DEPTHS | tr ' ' ',')"; fi
MULTI=0; case " $* " in *" --mode conc "*) MULTI=1 ;; esac
DEPTHJOB=0; case " $* " in *" --depths "*) DEPTHJOB=1 ;; esac
METRICS=0
# check <label> <client log> <byte offset of this call's output> <exit code> <step> [key=value...] [-- <client args>]
check() { local tag=$1 log=$2 off=$3 rc=$4 step=$5; shift 5
  python3 $W/client_check.py $RUN/errors.jsonl $log $off $rc $tag "$step" engine=$ENG mode=$MODE srvlog=$LOG "$@"; }
case $ENG in
  up|fab)
    BIN=$UP_BIN; [ $ENG = fab ] && BIN=$FAB_BIN
    # -cram 0: no host-RAM prompt cache (or idle-slot save/clear); -ctxcp 0: no context checkpoints, which would also
    # split prompt batches. On from 2026-10-24; MTP/DFlash2 keep their own rollback state.
    CMD=($BIN/llama-server -m $GGUF -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-cache-prompt --no-webui -cram 0 -ctxcp 0)
    # heavy DFlash2: speculation keeps a rollback copy of each slot's 151 MB recurrent state per draft position (8 for
    # DFlash2), about 77 GB at 64 slots, which ran the Spark out of memory on 2026-10-21; 32 slots cover its C <= 32
    if [ $PROFILE = heavy ] && [ $MODE = dflash2 ]; then SIZE=(-np ${NP:-32} -kvu -c ${CTX:-131072})
    elif [ $PROFILE = heavy ]; then SIZE=(-np ${NP:-64} -kvu -c ${CTX:-262144})
    elif [ -n "${NP:-}" ]; then SIZE=(-np $NP -kvu -c ${CTX:-65536})
    elif [ $MULTI = 1 ]; then SIZE=(-np 32 -kvu -c 65536)
    elif [ $DEPTHJOB = 1 ]; then SIZE=(-np 1 -c 40960)
    else SIZE=(-np 1 -c 16384); fi
    CMD+=("${SIZE[@]}")
    case $MODE in
      mtp) CMD+=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
      dflash2) CMD+=(-md $DGGUF -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7) ;;
    esac
    ENVS=(); [ $ENG = fab ] && ENVS=(LLAMA_GDN_DEFERRED=${FAB_DEFERRED:-1}) ;;
  vllm)
    CMD=($(dirname $VLLM_PY)/vllm serve $HF --served-model-name q27 --host 127.0.0.1 --port $PORT --language-model-only
         --no-enable-prefix-caching --gpu-memory-utilization 0.45)
    if [ $PROFILE = heavy ]; then CMD+=(--max-num-seqs 64 --max-cudagraph-capture-size 512 --max-num-batched-tokens 8192 --max-model-len 131072)
    else CMD+=(--max-num-seqs 32 --max-model-len 40960); fi
    case $MODE in
      mtp) CMD+=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}') ;;
      dflash2) CMD+=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$DHF\", \"num_speculative_tokens\": 7}") ;;
    esac
    ENVS=(MAX_JOBS=4); METRICS=1 ;;
  sglang)
    CMD=($SGL_PY -m sglang.launch_server --model-path $HF --served-model-name q27 --host 127.0.0.1 --port $PORT
         --disable-radix-cache --mem-fraction-static 0.6 --enable-metrics)
    # SGL_MAXREQ lowers the heavy profile's 64 running requests (DFlash2 at 64 leaves no KV memory even at 0.68)
    if [ $PROFILE = heavy ]; then CMD+=(--max-running-requests ${SGL_MAXREQ:-64} --cuda-graph-max-bs-decode ${SGL_MAXREQ:-64} --chunked-prefill-size 8192 --context-length 131072)
      # 64 requests with speculation need more Mamba state than 0.6 leaves (seen 2026-10-21)
      [ $MODE != ar ] && for i in "${!CMD[@]}"; do [ "${CMD[$i]}" = "--mem-fraction-static" ] && CMD[$((i + 1))]=0.68; done
    else CMD+=(--max-running-requests 32 --context-length 40960); fi
    case $MODE in
      mtp) CMD+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4) ;;
      dflash2) CMD+=(--speculative-algorithm DFLASH --speculative-draft-model-path $DHF --speculative-dflash-block-size 8) ;;
    esac
    ENVS=(MAX_JOBS=4); METRICS=1 ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
MODEL_ARG=(); [ $ENG = vllm -o $ENG = sglang ] && MODEL_ARG=(--model q27)
{ printf '%s ' "${ENVS[@]}"; printf '%q ' "${CMD[@]}"; echo; } > $RUN/logs/$TAG.cmd
# own process group, so the whole server tree (vLLM's engine core, SGLang's scheduler) stops together
start_server() {
  env "${ENVS[@]}" setsid "${CMD[@]}" > $LOG 2>&1 &
  SPID=$!
  up=0
  for i in $(seq 1 1800); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
}
start_server
# SGLang sizes its Mamba state pool from the requests it admits; with 64 of them and speculation it may want more than
# the 0.6 static fraction leaves, so retry once at 0.68
if [ $up = 0 ] && [ $ENG = sglang ] && grep -q "leave no GPU memory for the KV cache" $LOG; then
  echo "$TAG: SGLang needs more static memory, retrying at --mem-fraction-static 0.68"
  for i in "${!CMD[@]}"; do [ "${CMD[$i]}" = "--mem-fraction-static" ] && CMD[$((i + 1))]=0.68; done
  mv $LOG $LOG.first; printf '%q ' "${CMD[@]}" > $RUN/logs/$TAG.cmd; echo >> $RUN/logs/$TAG.cmd
  sleep 10; start_server
fi

scrape() { # <out file>: the spec-decode counters as JSON
  curl -s -m 30 http://127.0.0.1:$PORT/metrics | python3 -c '
import json, re, sys
want = {"vllm:spec_decode_num_drafts_total": "drafts", "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted", "vllm:generation_tokens_total": "gen",
        "sglang:spec_verify_calls_total": "verify", "sglang:generation_tokens_total": "gen"}
out = {}
for line in sys.stdin:
    m = re.match(r"([a-z_:]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line.strip())
    if m and m.group(1) in want:
        out[want[m.group(1)]] = out.get(want[m.group(1)], 0) + float(m.group(3))
print(json.dumps(out))' > $1
}

run_job() { # <label> <bench_client args...>
  local tag=$1; shift
  local clog=$RUN/logs/$tag.client.log off rc i d dl
  echo "== $tag job $(date +%H:%M:%S)"
  [ $METRICS = 1 ] && scrape $RUN/logs/$tag.metrics-pre.json
  if [ "${1:-}" = "--depths" ]; then
    # one client call per depth; the first failure stops the sweep and its row lists the depths not run
    dl=(${2//,/ })
    for i in "${!dl[@]}"; do
      d=${dl[$i]}; off=$(cat $clog 2>/dev/null | wc -c)
      python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $tag --out $RUN/results.jsonl ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} \
        --mode conc --long-output --concurrency 1 --rounds 1 --prompt-tokens $d --gen 128 >> $clog 2>&1
      rc=$?
      check $tag $clog $off $rc "depth $d" not_run="${dl[*]:$((i + 1))}"
      [ $rc = 0 ] || { echo "client failed: $tag at depth $d"; tail -3 $clog; break; }
    done
  else
    off=$(cat $clog 2>/dev/null | wc -c)
    python3 ${CLIENT:-$W/bench_client.py} --url http://127.0.0.1:$PORT --label $tag --out $RUN/results.jsonl ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} "$@" \
      >> $clog 2>&1
    rc=$?
    check $tag $clog $off $rc - -- "$@"
    [ $rc = 0 ] || { echo "client failed: $tag"; tail -3 $clog; }
  fi
  if [ $METRICS = 1 ]; then
    scrape $RUN/logs/$tag.metrics-post.json
    python3 - $RUN/logs/$tag.metrics-pre.json $RUN/logs/$tag.metrics-post.json "$tag" >> $RUN/results.jsonl <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
print(json.dumps({"mode": "specstats", "label": sys.argv[3], **{k: round(v) for k, v in d.items()}}))
EOF
  fi
}

job=()
if [ $up = 1 ]; then
  echo "== $TAG server up $(date +%H:%M:%S)"
  # one unmeasured request first, so kernels an engine builds or compiles on first use are not timed
  M=""; [ ${#MODEL_ARG[@]} -gt 0 ] && M="\"model\": \"${MODEL_ARG[1]}\", "
  curl -s -m 900 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "{${M}\"prompt\": \"Here is a detailed history of the printing press, century by century:\\n\\n\", \"max_tokens\": 256, \"temperature\": 0}" \
    > $RUN/logs/$TAG.warmup.json 2>&1 \
    || { rc=$?; echo "warm-up request failed: $TAG"; check $TAG - 0 $rc warm-up "error=warm-up request failed (curl exit $rc)"; }
  for a in "$@" "::"; do
    if [ "$a" = "::" ]; then [ ${#job[@]} -gt 0 ] && run_job "${job[@]}"; job=(); else job+=("$a"); fi
  done
else
  echo "server died: $TAG"; tail -5 $LOG
  for a in "$@" "::"; do
    if [ "$a" = "::" ]; then
      [ ${#job[@]} -gt 0 ] && echo "{\"label\": \"${job[0]}\", \"engine\": \"$ENG\", \"mode\": \"$MODE\", \"step\": \"server start\", \"error\": \"server did not start\"}" >> $RUN/errors.jsonl
      job=()
    else job+=("$a"); fi
  done
fi
kill -TERM -- -$SPID 2>/dev/null
for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL -- -$SPID 2>/dev/null; wait $SPID 2>/dev/null
# our own leftovers only (pkill without root cannot touch other users' processes); the brackets keep the patterns
# from matching a shell whose command line merely mentions them
pkill -u $(id -u) -f "llama-server .*--port $PORT|vllm serve[ ]|sglang[.]launch_server|sglang::[a-z]" 2>/dev/null
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:$PORT/ >/dev/null 2>&1 || break; sleep 1; done
sleep 5
