#!/bin/bash
# usage: run_engine.sh <run dir> <engine up|fab|vllm|sglang> <mode ar|mtp|dflash2> <label> <job args...> [:: <label> <job args...>]...
#   env NP/CTX (llama.cpp slots and context), DEPTHS="1024 4096 ..." (reference style: one depth sweep as the only job),
#   BENCH_GPU (the one GPU the bench uses, nvidia-smi index, default 7: the last of the eight, as agreed)
# Starts the engine's OpenAI-compatible server from the paths in engines.env, sends unmeasured warm-up requests (one,
# then concurrent bursts), runs each job (jobs are separated by "::") against that one server, appends the results to
# <run dir>/results.jsonl and stops the server. A job is bench_client.py arguments, or "--depths 1024,4096,..." for a
# depth sweep: one request per prompt length (--mode conc --long-output --concurrency 1 --rounds 1 --gen 128; prefill
# t/s = prompt tokens / time to first token). Sharing a server across jobs saves vLLM's and SGLang's start (compile
# and CUDA graph capture).
# H100 copy of the DGX Spark script. Weights: vLLM and SGLang serve Qwen's FP8 checkpoint (native FP8 on Hopper),
# llama.cpp and qvac-fabric the Q8_0 GGUF of the same model. Server sizing (standard profile only):
#   llama.cpp: -np 1 -c 16384 for --mode spec, -np 32 -kvu -c 65536 for --mode conc, -np 1 -c 131072 for a depth sweep
#   vLLM: 32 sequences, 131072-token context, --gpu-memory-utilization 0.9 (the GPU is the bench's alone)
#   SGLang: 32 running requests (DFlash2: 8, env SGL_MAXREQ), 131072-token context, --mem-fraction-static 0.85
# For vLLM and SGLang the drafted and accepted token counters are read from /metrics before and after each job and
# appended as a "specstats" row.
# Every failed client call (each failed depth step on its own), every request that failed inside a concurrency level
# and a failed warm-up add a row to <run dir>/errors.jsonl through client_check.py: label, step, reason ("error") and
# time, which summarize.py copies into summary.json.
set -u
RUN=$1; ENG=$2; MODE=$3; TAG=$4; shift 3
# an uncommon port: other users of the box run their own servers
W=${BENCH_DIR:-$HOME/bench-weekly}; PORT=${BENCH_PORT:-18098}
export PATH=$HOME/.local/bin:$W/venvs/tools/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${BENCH_GPU:-7}
source ${ENGINES_ENV:-$W/engines.env}
M=${MODELS_DIR:-$HOME/models/qwen38-27b}
GGUF=$M/gguf/Qwen3.8-27B-Q8_0.gguf
DGGUF=$M/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf
HF=$M/fp8-hf
DHF=$M/dflash2-hf
LOG=$RUN/logs/$TAG.srv.log
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
    # split prompt batches. MTP/DFlash2 keep their own rollback state.
    CMD=($BIN/llama-server -m $GGUF -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-cache-prompt --no-webui -cram 0 -ctxcp 0)
    if [ -n "${NP:-}" ]; then SIZE=(-np $NP -kvu -c ${CTX:-65536})
    elif [ $MULTI = 1 ]; then SIZE=(-np 32 -kvu -c 65536)
    elif [ $DEPTHJOB = 1 ]; then SIZE=(-np 1 -c 131072)
    else SIZE=(-np 1 -c 16384); fi
    CMD+=("${SIZE[@]}")
    case $MODE in
      mtp) CMD+=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
      dflash2) CMD+=(-md $DGGUF -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7) ;;
    esac
    ENVS=(); [ $ENG = fab ] && ENVS=(LLAMA_GDN_DEFERRED=${FAB_DEFERRED:-1}) ;;
  vllm)
    CMD=($(dirname $VLLM_PY)/vllm serve $HF --served-model-name q27 --host 127.0.0.1 --port $PORT --language-model-only
         --no-enable-prefix-caching --gpu-memory-utilization 0.9 --max-num-seqs 32 --max-model-len 131072)
    case $MODE in
      mtp) CMD+=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}') ;;
      dflash2) CMD+=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$DHF\", \"num_speculative_tokens\": 7}") ;;
    esac
    ENVS=(MAX_JOBS=16); METRICS=1 ;;
  sglang)
    CMD=($SGL_PY -m sglang.launch_server --model-path $HF --served-model-name q27 --host 127.0.0.1 --port $PORT
         --disable-radix-cache --mem-fraction-static 0.85 --enable-metrics --context-length 131072)
    # DFlash2: without the radix cache SGLang reserves recurrent state for every running request plus a copy per draft
    # position (block 8), ~1.3 GB per request (42 GB at 32), so its server admits 8 and runs only spec and concspec (C <= 8)
    if [ $MODE = dflash2 ]; then CMD+=(--max-running-requests ${SGL_MAXREQ:-8})
    else CMD+=(--max-running-requests 32); fi
    case $MODE in
      mtp) CMD+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4) ;;
      dflash2) CMD+=(--speculative-algorithm DFLASH --speculative-draft-model-path $DHF --speculative-dflash-block-size 8) ;;
    esac
    ENVS=(MAX_JOBS=16); METRICS=1 ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
MODEL_ARG=(); [ $ENG = vllm -o $ENG = sglang ] && MODEL_ARG=(--model q27)
# vLLM's and SGLang's FP8 block-scaled GEMMs (DeepGEMM) compile kernels on first use with the nvcc under CUDA_HOME
# (default /usr/local/cuda, which has no compiler on this box). DeepGEMM needs nvcc >= 12.9 whose version matches the
# CUDA headers: the box's 12.8 is too old and the venvs' pip toolkits mix versions (nvcc 13.4, headers 13.2), so
# update.sh keeps a pip toolkit per torch CUDA version in cuda/<version> (CUDA 12.8 is only the fallback).
if [ $ENG = vllm -o $ENG = sglang ]; then
  PY=$VLLM_PY; [ $ENG = sglang ] && PY=$SGL_PY
  CU=$(ls -d $W/cuda/$($PY -c 'import torch; print(torch.version.cuda)' 2>/dev/null)/nvidia/cu* 2>/dev/null | head -1)
  [ -x "$CU/bin/nvcc" ] || CU=/usr/local/cuda-12.8
  ENVS+=(CUDA_HOME=$CU PATH=$CU/bin:$PATH)
fi
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
  # then unmeasured bursts of 2, 4, 8 and 16 concurrent requests with ~250-token prompts: SGLang pays a one-time cost
  # (2-5 s) the first time it prefills several sequences together, which on 2026-10-28 landed in its first 4-request
  # level and cut the one-round speculation totals at 4 requests about threefold
  WP=$(python3 -c 'import json; print(json.dumps("Summarize the history of the printing press in one page. " * 25))')
  for n in 2 4 8 16; do
    pids=()
    for i in $(seq 1 $n); do
      curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
        -d "{${M}\"prompt\": $WP, \"max_tokens\": 32, \"temperature\": 0}" > /dev/null 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}"
  done
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
# wait until the GPU's memory is back, so the next engine's memory profiling starts from an empty GPU
for i in $(seq 1 60); do
  [ "$(nvidia-smi -i ${BENCH_GPU:-7} --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" -lt 1024 ] 2>/dev/null && break
  sleep 2
done
sleep 3
