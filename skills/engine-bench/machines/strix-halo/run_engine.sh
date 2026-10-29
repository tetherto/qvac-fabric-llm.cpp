#!/bin/bash
# usage: run_engine.sh <run dir> <engine> <backend> <mode> <kind> <bench_client args...>
#   engine   up | strix | fab | nath | paoai | gufo
#   backend  vulkan | rocm            (gufo is always rocm)
#   mode     ar | mtp | dflash2
#   kind     spec (1 slot, the four spec prompts) | conc (32 slots, no speculation) | concspec (8 slots, MTP or DFlash2)
#            | depth (1 slot, 40K context; with env DEPTHS="1024 4096 ...", one request per prompt length from one server
#              start: prefill t/s = prompt tokens / time to first token, decode t/s at that depth; client args unused)
# Starts the engine's OpenAI-compatible server from the paths in engines.env, sends one unmeasured warm-up request,
# runs bench_client.py against it with label <kind>-<engine>-<mode>-<backend> (the shared layout's <kind>-<engine>-<mode>
# plus the backend, since two backends of one engine run here), appends the results to <run dir>/results.jsonl,
# records the exact command line in <run dir>/commands.txt and stops the server. Every failed client call (each failed
# depth step on its own), every request that failed inside a concurrency level and a failed warm-up add a row to
# <run dir>/errors.jsonl through client_check.py: label, step, reason ("error") and time, which summarize.py copies
# into summary.json.
set -u
RUN=$1; ENG=$2; BE=$3; MODE=$4; KIND=$5; shift 5
W=${BENCH_DIR:-$HOME/bench-weekly}; PORT=8098; ROCM=/opt/rocm-7.2.4
TAG=$KIND-$ENG-$MODE-$BE
source $W/engines.env
source ${OPTS_FILE:-$W/engine-opts.conf}
M=${MODELS_DIR:-$HOME/models/qwen38-27b}/Qwen3.8-27B-UD-Q4_K_XL.gguf
QD=${MODELS_DIR:-$HOME/models/qwen38-27b}/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=$RUN/logs/$TAG.srv.log
skip() { echo "skip $TAG: $1"; echo "{\"label\": \"$TAG\", \"engine\": \"$ENG\", \"backend\": \"$BE\", \"mode\": \"$MODE\", \"kind\": \"$KIND\", \"skipped\": \"$1\"}" >> $RUN/errors.jsonl; exit 0; }
# check <client log> <byte offset of this call's output> <exit code> <step> [key=value...] [-- <client args>]
check() { local log=$1 off=$2 rc=$3 step=$4; shift 4
  python3 $W/client_check.py $RUN/errors.jsonl $log $off $rc $TAG "$step" engine=$ENG backend=$BE mode=$MODE kind=$KIND srvlog=$LOG "$@"; }
CLOG=$RUN/logs/$TAG.client.log
MODEL_ARG=()
ENV=()
case $ENG in
  up|strix|fab|nath|paoai)
    P=$(echo $ENG | tr a-z A-Z); [ $ENG = up ] && P=UP
    V=${P}_$(echo $BE | tr a-z A-Z)_BIN; BIN=${!V:-}
    [ -x "$BIN/llama-server" ] || skip "no $BE build"
    ARGS=(-m $M -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-cache-prompt --no-webui)
    # -cram 0 turns off the host-RAM prompt cache (and the idle-slot saving that needs it), -ctxcp 0 the per-slot
    # context checkpoints, which split prompt batches and copy state; both would skew timings. A fork too old for a
    # flag runs without it, and the run's notes say so.
    HELP=$($BIN/llama-server --help 2>&1)
    for fl in "-cram:--cache-ram" "-ctxcp:--ctx-checkpoints"; do
      if grep -qE -- "${fl#*:}[ ,]" <<< "$HELP"; then ARGS+=(${fl%:*} 0)
      elif ! grep -qs "\"$ENG lacks ${fl%:*}\"" $RUN/errors.jsonl; then
        echo "{\"label\": \"$ENG-$BE\", \"note\": \"$ENG lacks ${fl%:*}\"}" >> $RUN/errors.jsonl
      fi
    done
    case $KIND in spec) ARGS+=(-np 1 -c 16384) ;; conc) ARGS+=(-np 32 -kvu -c 65536) ;; concspec) ARGS+=(-np 8 -kvu -c 32768) ;;
                  depth) ARGS+=(-np 1 -c 40960) ;; esac
    # extra speculative flags per engine and mode (engine-opts.conf), e.g. SPEC_EXTRA_STRIX_DFLASH2; a variable with
    # the kind appended (SPEC_EXTRA_PAOAI_DFLASH2_CONCSPEC), even an empty one, overrides it for that kind
    X=SPEC_EXTRA_${P}_$(echo $MODE | tr a-z A-Z); XK=${X}_$(echo $KIND | tr a-z A-Z)
    [ -n "${!XK+set}" ] && X=$XK
    EXTRA=(${!X:-})
    case $MODE in
      # the UD-Q4_K_XL GGUF carries the MTP layer itself (blk.64, nextn_predict_layers = 1): no sidecar needed
      mtp)     ARGS+=(--spec-type draft-mtp --spec-draft-n-max 3 "${EXTRA[@]}") ;;
      dflash2) ARGS+=(-md $QD -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7 "${EXTRA[@]}") ;;
    esac
    [ $ENG = fab ] && ENV+=(LLAMA_GDN_DEFERRED=1)
    # gfx1151 HIP async-execution bug: batched inference is wrong without serialized launches (strix-llama.cpp README)
    [ $BE = rocm ] && ENV+=(HIP_LAUNCH_BLOCKING=1)
    CMD=($BIN/llama-server "${ARGS[@]}") ;;
  gufo)
    [ -x "${GUFO_BIN:-}" ] || skip "no gufo build"
    [ $MODE = mtp ] && skip "not supported: gufo has no MTP for Qwen3.8-27B"
    # gufo has no switch to turn its snapshot cache off (--cache-ram-bytes 0 means automatic); the client's
    # cache_prompt: false skips every lookup, so no request reuses a cached prefix
    ARGS=(serve --host 127.0.0.1 --port $PORT llm --model $M --served-model-name q27)
    case $KIND in spec) ARGS+=(--sessions 1 --context 16384) ;; conc) ARGS+=(--sessions 16 --context 4096) ;; concspec) ARGS+=(--sessions 8 --context 4096) ;;
                  depth) ARGS+=(--sessions 1 --context 40960) ;; esac
    [ $MODE = dflash2 ] && ARGS+=(--speculative dflash2 --dflash-model $QD ${GUFO_SPEC_EXTRA:-})
    ENV+=(LD_LIBRARY_PATH=$ROCM/lib)
    CMD=($GUFO_BIN "${ARGS[@]}")
    MODEL_ARG=(--model q27) ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
echo "$TAG: ${ENV[*]} ${CMD[*]}" >> $RUN/commands.txt
env "${ENV[@]}" "${CMD[@]}" > $LOG 2>&1 &
SPID=$!
up=0
for i in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 1
done
if [ $up = 1 ]; then
  echo "== $TAG server up $(date +%H:%M:%S)"
  # one unmeasured request first, so kernels or graphs an engine builds on first use are not timed
  MJ=""; [ ${#MODEL_ARG[@]} -gt 0 ] && MJ="\"model\": \"${MODEL_ARG[1]}\", "
  curl -s -m 900 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "{${MJ}\"prompt\": \"Here is a detailed history of the printing press, century by century:\\n\\n\", \"max_tokens\": 128, \"temperature\": 0}" \
    > $RUN/logs/$TAG.warmup.json 2>&1 \
    || { rc=$?; echo "warm-up request failed: $TAG"; check - 0 $rc warm-up "error=warm-up request failed (curl exit $rc)"; }
  # results count only if the server is still alive afterwards: a crash mid-run leaves short, meaningless rows
  OUT=$RUN/logs/$TAG.results.jsonl; rm -f $OUT
  if [ -n "${DEPTHS:-}" ]; then
    # one client call per depth; the first failure stops the sweep and its row lists the depths not run
    : > $CLOG
    DL=($DEPTHS)
    for i in "${!DL[@]}"; do
      d=${DL[$i]}; off=$(wc -c < $CLOG)
      python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $TAG --out $OUT ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} \
        --mode conc --long-output --concurrency 1 --rounds 1 --prompt-tokens $d --gen 128 --corpus "$W/corpus/**/*.md" \
        >> $CLOG 2>&1
      rc=$?
      check $CLOG $off $rc "depth $d" not_run="${DL[*]:$((i + 1))}"
      [ $rc = 0 ] || { echo "client failed: $TAG at depth $d"; break; }
    done
  else
    python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $TAG --out $OUT ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} "$@" \
      > $CLOG 2>&1
    rc=$?
    check $CLOG 0 $rc - -- "$@"
    [ $rc = 0 ] || { echo "client failed: $TAG"; tail -3 $CLOG; }
  fi
  if ! kill -0 $SPID 2>/dev/null; then
    WHY=$( { grep -m1 GGML_ASSERT $LOG || grep -m1 -iE "terminate|abort|error" $LOG; } | tr -d '"\\' | cut -c1-200)
    echo "server crashed during the run: $TAG: $WHY"
    echo "{\"label\": \"$TAG\", \"engine\": \"$ENG\", \"backend\": \"$BE\", \"mode\": \"$MODE\", \"kind\": \"$KIND\", \"step\": \"server, results discarded\", \"error\": \"server crashed: $WHY\"}" >> $RUN/errors.jsonl
  elif [ -s $OUT ]; then
    cat $OUT >> $RUN/results.jsonl
  fi
else
  echo "server died: $TAG"; tail -5 $LOG
  echo "{\"label\": \"$TAG\", \"engine\": \"$ENG\", \"backend\": \"$BE\", \"mode\": \"$MODE\", \"kind\": \"$KIND\", \"step\": \"server start\", \"error\": \"server did not start\"}" >> $RUN/errors.jsonl
fi
kill $SPID 2>/dev/null
for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null
# only this user's servers on this port (pkill never reaches other users' processes)
pkill -u $(id -u) -f "[l]lama-server .*--port $PORT" 2>/dev/null
pkill -u $(id -u) -f "[g]ufo serve --host 127.0.0.1 --port $PORT" 2>/dev/null
sleep 8
