#!/bin/bash
# usage: run_engine.sh <run dir> <engine> <mode> <label> <bench_client args...>   env: NP CTX (llama.cpp slots and context)
#   env SGL_CTX: SGLang's context length (default 65536)
#   env DEPTHS="1024 4096 ...": instead of one client call, one request per prompt length (depth sweep: prefill t/s
#   = prompt tokens / time to first token, decode t/s at that depth), all from one server start
#   engine: up | fab | mlx | sglang | omlx | splash      mode: ar | mtp | dflash2
# Starts the engine's OpenAI-compatible server in that speculative mode from the paths in engines.env, sends one
# unmeasured warm-up request, runs bench_client.py against it, appends the results to <run dir>/results.jsonl
# (summarize.py reads engine and mode back from labels <kind>-<engine>-<mode>) and stops the server.
# A mode an engine does not support is an error. Every failed client call (each failed depth step on its own), every
# request that failed inside a concurrency level and a failed warm-up add a row to <run dir>/errors.jsonl through
# client_check.py: label, step, reason ("error") and time, which summarize.py copies into summary.json.
set -u
RUN=$1; ENG=$2; MODE=$3; TAG=$4; shift 4
W=${BENCH_DIR:-$HOME/bench-weekly}; PORT=8098
# ENGINES_ENV: use another engines.env (for example a past run's copy, to add results with the same builds)
source ${ENGINES_ENV:-$W/engines.env}
MODELS=${MODELS_DIR:-/Users/shared/models/mlx-vs-llama}
Q0=$MODELS/gguf/Qwen3.8-27B-Q4_0-pure.gguf
QD=$MODELS/gguf/Qwen3.8-27B-DFlash2-Q4_0.gguf
X=$MODELS/mlx/Qwen3.8-27B-4bit
XD=$MODELS/hf/Qwen3.8-27B-DFlash2
LOG=$RUN/logs/$TAG.srv.log
MODEL_ARG=()
unsupported() { echo "unsupported: $ENG $MODE"; echo "{\"label\": \"$TAG\", \"engine\": \"$ENG\", \"mode\": \"$MODE\", \"step\": \"server start\", \"error\": \"mode not supported\"}" >> $RUN/errors.jsonl; exit 1; }
# check <client log> <byte offset of this call's output> <exit code> <step> [key=value...] [-- <client args>]
check() { local log=$1 off=$2 rc=$3 step=$4; shift 4
  python3 $W/client_check.py $RUN/errors.jsonl $log $off $rc $TAG "$step" engine=$ENG mode=$MODE srvlog=$LOG "$@"; }
CLOG=$RUN/logs/$TAG.client.log
case $ENG in
  up|fab)
    BIN=$UP_BIN; [ $ENG = fab ] && BIN=$FAB_BIN
    SPEC=()
    case $MODE in
      ar) ;;
      mtp) SPEC=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
      dflash2) SPEC=(-md $QD -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7) ;;
      *) unsupported ;;
    esac
    # qvac-fabric: deferred recurrent-state commit for speculative decoding (no effect without speculation).
    # -cram 0 turns off the host-RAM prompt cache (and the idle-slot saving that needs it), -ctxcp 0 the per-slot
    # context checkpoints: both copy state around requests and would skew the timings
    LLAMA_GDN_DEFERRED=1 $BIN/llama-server -m $Q0 -ngl 99 -fa on --port $PORT --no-cache-prompt -cram 0 -ctxcp 0 --no-webui \
        -np ${NP:-32} -kvu -c ${CTX:-65536} ${SPEC[@]+"${SPEC[@]}"} > $LOG 2>&1 & ;;
  mlx)
    [ $MODE = ar ] || unsupported
    $MLX_PY -m mlx_lm.server --model $X --port $PORT --max-tokens 4096 > $LOG 2>&1 & ;;
  sglang)
    [ $MODE = ar ] || unsupported
    ( cd $SGL_DIR && SGLANG_USE_MLX=1 exec .venv/bin/python -m sglang.launch_server --model-path $X --served-model-name q27 \
        --disable-cuda-graph --disable-radix-cache --mamba-radix-cache-strategy no_buffer --context-length ${SGL_CTX:-65536} \
        --host 127.0.0.1 --port $PORT ) > $LOG 2>&1 &
    MODEL_ARG=(--model q27) ;;
  omlx)
    # each mode has its own base path; model_settings.json there enables DFlash2 or the MTP drafter
    case $MODE in ar) BASE=$W/omlx-base ;; mtp) BASE=$W/omlx-base-mtp ;; dflash2) BASE=$W/omlx-base-dflash ;; *) unsupported ;; esac
    mkdir -p $W/omlx-models $BASE && ln -sfn $X $W/omlx-models/Qwen3.8-27B-4bit
    if [ ! -f $BASE/model_settings.json ]; then
      case $MODE in
        mtp) S='"vlm_mtp_enabled": true, "vlm_mtp_draft_model": "'$MODELS/mlx/Qwen3.8-27B-MTP-4bit'", "vlm_mtp_draft_block_size": 3' ;;
        dflash2) S='"dflash_enabled": true, "dflash_draft_model": "'$XD'"' ;;
        *) S= ;;
      esac
      echo '{"version": 1, "models": {"Qwen3.8-27B-4bit": {'"$S"'}}}' > $BASE/model_settings.json
    fi
    $OMLX_DIR/.venv/bin/omlx serve --model-dir $W/omlx-models --base-path $BASE --port $PORT --no-cache \
        --max-concurrent-requests 16 > $LOG 2>&1 &
    MODEL_ARG=(--model Qwen3.8-27B-4bit) ;;
  splash)
    # Splash always decodes with its DFlash2 draft
    [ $MODE = dflash2 ] || unsupported
    # prepare_local_model is cheap after the first call and must be redone after an upgrade
    SM=$(cd $SPLASH_DIR && PYTHONPATH=$SPLASH_DIR python/bin/python3 -P -c "
from install.local import prepare_local_model
print(prepare_local_model('$X', draft='$XD', language_only=True))" 2> $RUN/logs/$TAG.prepare.err | tail -1)
    [ -d "$SM" ] || { echo "splash prepare failed: $TAG"; tail -3 $RUN/logs/$TAG.prepare.err
      check $RUN/logs/$TAG.prepare.err 0 1 "server start" "error=splash prepare failed: $(tail -1 $RUN/logs/$TAG.prepare.err | cut -c1-200)"; exit 1; }
    ( cd $SPLASH_DIR && PYTHONPATH=$SPLASH_DIR exec python/bin/python3 -P -m server.server "$SM" --model mlx-community/Qwen3.8-27B-4bit \
        --binary $SPLASH_DIR/engine/splash --port $PORT --kv-format bf16 --max-cache-disk 0 --idle-release off --no-webui ) > $LOG 2>&1 &
    MODEL_ARG=(--model mlx-community/Qwen3.8-27B-4bit) ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
SPID=$!
up=0
for i in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 1
done
[ $ENG = splash ] && [ $up = 1 ] && for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/ready >/dev/null 2>&1 && break; sleep 1; done
if [ $up = 1 ]; then
  echo "== $TAG server up $(date +%H:%M:%S)"
  # one unmeasured request first, so kernels an engine builds on first use (oMLX 0.7.1 does) are not timed
  M=""; [ ${#MODEL_ARG[@]} -gt 0 ] && M="\"model\": \"${MODEL_ARG[1]}\", "
  curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "{${M}\"prompt\": \"Here is a detailed history of the printing press, century by century:\\n\\n\", \"max_tokens\": 256, \"temperature\": 0}" \
    > $RUN/logs/$TAG.warmup.json 2>&1 \
    || { rc=$?; echo "warm-up request failed: $TAG"; check - 0 $rc warm-up "error=warm-up request failed (curl exit $rc)"; }
  if [ -n "${DEPTHS:-}" ]; then
    # one client call per depth; the first failure stops the sweep and its row lists the depths not run
    DL=($DEPTHS)
    for i in "${!DL[@]}"; do
      d=${DL[$i]}; off=$(cat $CLOG 2>/dev/null | wc -c)
      python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $TAG --out $RUN/results.jsonl ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} \
        --mode conc --long-output --concurrency 1 --rounds 1 --prompt-tokens $d --gen 128 >> $CLOG 2>&1
      rc=$?
      check $CLOG $off $rc "depth $d" not_run="${DL[*]:$((i + 1))}"
      [ $rc = 0 ] || { echo "client failed: $TAG at depth $d"; break; }
    done
  else
    python3 $W/bench_client.py --url http://127.0.0.1:$PORT --label $TAG --out $RUN/results.jsonl ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} "$@" \
      > $CLOG 2>&1
    rc=$?
    check $CLOG 0 $rc - -- "$@"
    [ $rc = 0 ] || echo "client failed: $TAG"
  fi
else
  echo "server died: $TAG"; tail -5 $LOG
  echo "{\"label\": \"$TAG\", \"engine\": \"$ENG\", \"mode\": \"$MODE\", \"step\": \"server start\", \"error\": \"server did not start\"}" >> $RUN/errors.jsonl
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
# oMLX renames its process to omlx-server, so match both names
pkill -f "sglang.launch_server|omlx serve|omlx-server|mlx_lm.server|splash-[0-9.]+-arm64-macos26/" 2>/dev/null
pkill -f "llama-server .*--port $PORT" 2>/dev/null
sleep 6
