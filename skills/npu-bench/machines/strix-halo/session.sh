#!/bin/bash
# usage: session.sh <run dir> <model tag> <xdna|flm> <session> [bench_npu.py run args...]
# Starts one server for one model, runs one client session into <run>/results.jsonl, then stops the server.
# Failures go to <run>/errors.log as SESSION_FAIL lines.
set -u
W=${BENCH_DIR:-$HOME/npu-bench}
RUN=$1 TAG=$2 RT=$3 SESSION=$4; shift 4
SERVER_TIMEOUT=14400; BENCH_TIMEOUT=10800; READY_TRIES=300; STOP_TRIES=30
export PATH=/opt/xilinx/xrt/bin:$PATH
source $W/bench.conf; source $W/engines.env
LABEL="$TAG|$RT|$SESSION"; LOG=$RUN/logs/srv-$TAG-$RT-$SESSION.log

fail() { echo "SESSION_FAIL $LABEL $1" | tee -a $RUN/errors.log; }

wait_ready() { # <url>
  local i
  for i in $(seq $READY_TRIES); do
    curl -s -m 2 -o /dev/null -w '%{http_code}' "$1" | grep -q 200 && return 0
    kill -0 $pid 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

stop_server() {
  local i
  kill $pid 2>/dev/null
  for i in $(seq $STOP_TRIES); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -0 $pid 2>/dev/null && { fail "server did not stop"; return 1; }
  sleep 5
}

read -r _ GGUF _ FLM_TAG XDNA_ENV < <(awk -v t="$TAG" '$1 == t' $W/models.conf)
[ -n "${GGUF:-}" ] || { fail "model not in models.conf"; exit 1; }

case $RT in
  xdna)
    envs=(); [ "$XDNA_ENV" != - ] && envs=(${XDNA_ENV//+/ })
    env "${envs[@]}" timeout $SERVER_TIMEOUT $FAB_BIN/llama-server -m $MODELS_DIR/$GGUF "${SERVER_FLAGS[@]}" \
      --host 127.0.0.1 --port $XDNA_PORT > $LOG 2>&1 < /dev/null &
    pid=$!; URL=http://127.0.0.1:$XDNA_PORT; READY=$URL/health; MODEL=default
    EXTRA='{"ignore_eos": true, "cache_prompt": false}' ;;
  flm)
    FLM_MODEL_PATH=$W/flm-models timeout $SERVER_TIMEOUT $FLM_BIN serve $FLM_TAG --port $FLM_PORT > $LOG 2>&1 < /dev/null &
    pid=$!; URL=http://127.0.0.1:$FLM_PORT; READY=$URL/v1/models; MODEL=$FLM_TAG
    EXTRA='{"ignore_eos": true}' ;;
  *) fail "unknown runtime"; exit 1 ;;
esac

wait_ready $READY || { fail "server not ready, see $LOG"; stop_server; exit 1; }
timeout $BENCH_TIMEOUT python3 -I $W/bench_npu.py run --server $URL --model "$MODEL" --prompts $W/prompts.json \
  --label "$LABEL" --out $RUN/results.jsonl --extra "$EXTRA" "$@"
rc=$?
stop_server || exit 1
grep -h "xdna-runtime: kernel .* not found" $LOG | sort -u | sed "s/^/SESSION_FAIL $LABEL /" >> $RUN/errors.log
[ $rc = 0 ] || { fail "bench rc=$rc"; exit 1; }
