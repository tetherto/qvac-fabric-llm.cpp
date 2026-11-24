#!/bin/bash
# usage: run_engine.sh <run dir> <engine up|fab|vllm> <mode ar|mtp|dflash2> <label> <job args...> [:: <label> <job args...>]...
#   env NP/CTX (llama.cpp slots and context), DEPTHS="1024 4096 ..." (one depth sweep as the only job)
# 2x DGX Spark: the head node (this one, SELF) and the second node (PEER), joined by a direct ConnectX-7 cable on their
# second port (two RoCE functions, about 110 Gb/s each; addresses and interfaces in pair.conf). One model split over
# both GPUs:
#   up, fab: an ggml-rpc-server on each node (RDMA transport, local tensor cache) and llama-server here with
#            -dev RPC0,RPC1 -sm tensor: every weight matrix and the KV cache split across the two GPUs, the partial
#            results combined over RPC. All-RPC with two devices takes the direct all-reduce path.
#   vllm:    tensor parallel 2 over the two nodes without Ray (NCCL over both RoCE ports), rank 0 here serves the API
#            and rank 1 runs headless on the second node. The bench folder, its binaries, venvs and model paths are the
#            same on both nodes.
# Otherwise as the single Spark's run_engine.sh: one unmeasured warm-up request, each job (bench_client.py arguments,
# or "--depths 1024,4096,..." for a depth sweep) against one server, results in <run dir>/results.jsonl, failures in
# <run dir>/errors.jsonl through client_check.py, vLLM's speculation counters as "specstats" rows. Server sizing as
# the Spark's standard profile: llama.cpp -np 1 -c 16384 for --mode spec, -np 32 -kvu -c 65536 for --mode conc,
# -np 1 -c 40960 for a depth sweep; vLLM 32 sequences and a 40960-token context.
set -u
RUN=$1; ENG=$2; MODE=$3; TAG=$4; shift 3
W=${BENCH_DIR:-$HOME/bench-pair}; PORT=${PORT:-8097}
source $W/pair.conf
# bench_client.py takes its default corpus ($BENCH_DIR/corpus/docs) from here
export BENCH_DIR=$W
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
source $W/engines.env
M=${MODELS_DIR:-$W/models}
GGUF=$M/gguf/Qwen3.8-27B-NVFP4-modelopt.gguf
DGGUF=$M/dflash2-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf
HF=$M/nvfp4-hf
DHF=$M/dflash2-hf
LOG=$RUN/logs/$TAG.srv.log
if [ -n "${DEPTHS:-}" ]; then set -- "$TAG" --depths "$(echo $DEPTHS | tr ' ' ',')"; fi
MULTI=0; case " $* " in *" --mode conc "*) MULTI=1 ;; esac
DEPTHJOB=0; case " $* " in *" --depths "*) DEPTHJOB=1 ;; esac
METRICS=0
# check <label> <client log> <byte offset of this call's output> <exit code> <step> [key=value...] [-- <client args>]
check() { local tag=$1 log=$2 off=$3 rc=$4 step=$5; shift 5
  python3 $W/client_check.py $RUN/errors.jsonl $log $off $rc $tag "$step" engine=$ENG mode=$MODE srvlog=$LOG "$@"; }
# NCCL and Gloo for vLLM: both RoCE functions for the tensor-parallel all-reduce, the SELF/PEER link for bootstrap
# (NCCL_DEBUG=INFO for INIT and NET puts the transport it picked, IB/RoCE or sockets, in the server log)
NET_ENV=(NCCL_SOCKET_IFNAME=$NET_IF GLOO_SOCKET_IFNAME=$NET_IF NCCL_IB_HCA=$IB_HCA NCCL_IB_DISABLE=0
         NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET)
PEER_CMD=()
case $ENG in
  up|fab)
    BIN=$UP_BIN; [ $ENG = fab ] && BIN=$FAB_BIN
    # -cram 0: no host-RAM prompt cache; -ctxcp 0: no context checkpoints (they split prompt batches)
    CMD=($BIN/llama-server -m $GGUF -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-cache-prompt --no-webui -cram 0 -ctxcp 0
         --rpc $SELF:$RPC_PORT,$PEER:$RPC_PORT -dev RPC0,RPC1 -sm tensor)
    if [ -n "${NP:-}" ]; then SIZE=(-np $NP -kvu -c ${CTX:-65536})
    elif [ $MULTI = 1 ]; then SIZE=(-np 32 -kvu -c 65536)
    elif [ $DEPTHJOB = 1 ]; then SIZE=(-np 1 -c 40960)
    else SIZE=(-np 1 -c 16384); fi
    CMD+=("${SIZE[@]}")
    case $MODE in
      mtp) CMD+=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
      dflash2) CMD+=(-md $DGGUF -ngld 99 -devd ${DEVD:-RPC0,RPC1} --spec-type draft-dflash --spec-draft-n-max 7) ;;
    esac
    # qvac-fabric runs without its deferred gated delta net commit here: -sm tensor has no split rules for the
    # deferred ops yet (GDN_CONV_DEFERRED aborts in ggml-backend-meta.cpp)
    ENVS=(DUMMY_ENV=0); [ $ENG = fab ] && ENVS=(LLAMA_GDN_DEFERRED=${FAB_DEFERRED:-0}) ;;
  vllm)
    VCMD=($(dirname $VLLM_PY)/vllm serve $HF --served-model-name q27 --language-model-only --no-enable-prefix-caching
          --gpu-memory-utilization 0.45 --max-num-seqs 32 --max-model-len 40960
          --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 --master-addr $SELF --master-port 29517)
    case $MODE in
      mtp) VCMD+=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}') ;;
      dflash2) VCMD+=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$DHF\", \"num_speculative_tokens\": 7}") ;;
    esac
    CMD=("${VCMD[@]}" --node-rank 0 --host 127.0.0.1 --port $PORT)
    PEER_CMD=("${VCMD[@]}" --node-rank 1 --headless)
    # FlashInfer builds its kernels on first use (flashinfer-cubin is not installed): it needs nvcc (CUDA_HOME) and
    # ninja, which these nodes only have in the venv
    ENVS=(MAX_JOBS=4 PATH=$(dirname $VLLM_PY):$PATH CUDA_HOME=/usr/local/cuda VLLM_HOST_IP=$SELF "${NET_ENV[@]}"); METRICS=1 ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
MODEL_ARG=(); [ $ENG = vllm ] && MODEL_ARG=(--model q27)
{ printf '%s ' "${ENVS[@]}"; printf '%q ' "${CMD[@]}"; echo; } > $RUN/logs/$TAG.cmd
[ ${#PEER_CMD[@]} -gt 0 ] && { echo -n "peer: "; printf '%q ' "${PEER_CMD[@]}"; echo; } >> $RUN/logs/$TAG.cmd

# our own servers only (pkill without root cannot touch other users' processes; the brackets keep the patterns from
# matching a shell whose command line merely mentions them). Other ggml-rpc-servers on these nodes use other ports.
stop_all() {
  if [ $ENG = vllm ]; then
    pkill -u $(id -u) -f "vllm serve[ ].*--master-port 29517" 2>/dev/null
    peer "pkill -u \$(id -u) -f 'vllm serve[ ].*--master-port 29517'" 2>/dev/null
  else
    pkill -u $(id -u) -f "llama-server[ ].*--port $PORT" 2>/dev/null
    $W/rpc.sh stop
  fi
}
# an engine stopped just before can take a while to hand its memory back (unified memory: the rpc-servers and
# llama-server hold tens of GB), and vLLM aborts its start when free memory grows while it profiles ("Error in memory
# profiling", 2026-10-27): wait until MemAvailable on both nodes holds still for 10 s, at most 3 minutes
settle_memory() {
  local prev="" cur i
  for i in $(seq 1 36); do
    cur="$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)/$(peer "awk '/^MemAvailable:/ {print int(\$2 / 1048576)}' /proc/meminfo")"
    [ "$cur" = "$prev" ] && { echo "== $TAG memory settled at $cur GiB available"; return 0; }
    prev=$cur; sleep 10
  done
  echo "== $TAG memory still moving ($cur GiB available), starting anyway"
}
# own process group, so the whole server tree (vLLM's engine core and workers) stops together
start_server() {
  if [ $ENG = vllm ]; then
    settle_memory
    peer "mkdir -p $RUN/logs && cd $W && env ${ENVS[*]/VLLM_HOST_IP=$SELF/VLLM_HOST_IP=$PEER} setsid -f $(printf '%q ' "${PEER_CMD[@]}") > $RUN/logs/$TAG.srv1.log 2>&1 < /dev/null"
  else
    stop_all; $W/rpc.sh start $BIN $RUN/logs/$TAG || echo "rpc-servers did not start: $TAG"
  fi
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
        "vllm:spec_decode_num_accepted_tokens_total": "accepted", "vllm:generation_tokens_total": "gen"}
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
    python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $tag --out $RUN/results.jsonl ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} "$@" \
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
  MJ=""; [ ${#MODEL_ARG[@]} -gt 0 ] && MJ="\"model\": \"${MODEL_ARG[1]}\", "
  curl -s -m 900 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "{${MJ}\"prompt\": \"Here is a detailed history of the printing press, century by century:\\n\\n\", \"max_tokens\": 256, \"temperature\": 0}" \
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
stop_all
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:$PORT/ >/dev/null 2>&1 || break; sleep 1; done
sleep 5
