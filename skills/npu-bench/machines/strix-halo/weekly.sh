#!/bin/bash
# Weekly xdna vs FastFlowLM benchmark on the Strix Halo NPU. Waits for the GPU lock and a quiet machine, updates both
# engines, runs every model through an xdna session, an flm session and a short xdna drift session, and writes
# runs/<date>/summary.json. Marks the run DONE or FAILED.
#   weekly.sh            full run (about 6 h)
#   weekly.sh --quick    pipeline check: no updates, 0.8B at 1k, short requests (about 5 min)
set -u
W=${BENCH_DIR:-$HOME/npu-bench}
source $W/bench.conf
export PATH=/opt/xilinx/xrt/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
DATE=$(TZ=Europe/Zurich date +%F); [ $QUICK = 1 ] && DATE="$DATE-quick"
RUN=$W/runs/$DATE
[ -e $RUN/results.jsonl ] && mv $RUN $RUN-$(date +%H%M%S)
mkdir -p $RUN/logs
exec >> $RUN/weekly.log 2>&1
echo "start $(date)"
rm -f $RUN/DONE $RUN/FAILED

WAIT_S=$((6 * 3600)); POLL_S=20; BUSY_CORES_MAX=1.0; GPU_BUSY_MAX=20; NPU_CHECK_TIMEOUT=30
# other inference, build and test jobs of any user; this run's own processes live under $W
OTHERS='^(llama|flm|cmake|ninja|make|cc1plus|ctest|test-|vllm|sglang|ollama|lemonade|gufo|rpc-server)'
if [ $QUICK = 1 ]; then
  MODELS=$QUICK_MODELS; CONTEXTS=$QUICK_CONTEXTS; MAX_TOKENS=$QUICK_MAX_TOKENS; ARGS=("${QUICK_ARGS[@]}"); DRIFT=("${QUICK_DRIFT[@]}")
else
  MODELS=$FULL_MODELS; CONTEXTS=$FULL_CONTEXTS; MAX_TOKENS=$FULL_MAX_TOKENS; ARGS=("${FULL_ARGS[@]}"); DRIFT=("${FULL_DRIFT[@]}")
fi

fail() { echo "FAILED: $1"; echo "$1" > $RUN/FAILED; python3 -I $W/summarize.py $RUN --status failed --reason "$1"; exit 1; }

others() { ps -eo comm=,args= | awk -v re="$OTHERS" '$1 ~ re' | grep -v -F "$W/" | head -1; }

busy_cores() { # CPUs busy over 2 s, from /proc/stat
  local u1 n1 s1 i1 w1 q1 sq1 st1 u2 n2 s2 i2 w2 q2 sq2 st2
  read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat; sleep 2
  read -r _ u2 n2 s2 i2 w2 q2 sq2 st2 _ < /proc/stat
  awk -v b=$(( (u2+n2+s2+q2+sq2+st2) - (u1+n1+s1+q1+sq1+st1) )) \
      -v t=$(( (u2+n2+s2+i2+w2+q2+sq2+st2) - (u1+n1+s1+i1+w1+q1+sq1+st1) )) -v c=$(nproc) 'BEGIN{printf "%.2f", b / t * c}'
}

gpu_busy() {
  local f
  for f in /sys/class/drm/card*/device/gpu_busy_percent; do [ "$(cat $f 2>/dev/null || echo 0)" -gt $GPU_BUSY_MAX ] && return 0; done
  return 1
}

machine_quiet() {
  [ -z "$(others)" ] && awk -v b=$(busy_cores) -v m=$BUSY_CORES_MAX 'BEGIN{exit !(b < m)}' && ! gpu_busy
}

quiet() { # <label>: wait until no other job runs on the machine
  local deadline=$(( $(date +%s) + WAIT_S )) n=0
  until machine_quiet; do
    [ $(date +%s) -gt $deadline ] && fail "machine busy for $((WAIT_S / 3600)) h before $1 ($(others))"
    [ $((n++ % 15)) = 0 ] && echo "QUIET_WAIT $1 other='$(others)' $(date +%T)"
    sleep $POLL_S
  done
}

take_lock() {
  local deadline=$(( $(date +%s) + WAIT_S ))
  while [ -e $GPU_LOCK ] || ! machine_quiet; do
    [ $(date +%s) -gt $deadline ] && fail "GPU lock or machine busy for $((WAIT_S / 3600)) h ($(cat $GPU_LOCK 2>/dev/null) $(others))"
    sleep $POLL_S
  done
  echo "$USER npu bench since $(date +%H:%M) (pid $$)" > $GPU_LOCK
  echo "lock taken $(date)"
}

watch_foreign() { # every change in the set of other jobs, with a timestamp
  local prev="" cur
  while true; do
    cur=$(ps -eo pid=,user=,comm=,args= | awk -v re="$OTHERS" '$3 ~ re' | grep -v -F "$W/" | cut -c1-200)
    if [ "$cur" != "$prev" ]; then echo "[$(date +%T)] ---"; [ -n "$cur" ] && echo "$cur"; prev=$cur; fi
    sleep 2
  done
}

model_field() { awk -v t="$1" -v f=$2 '$1 == t {print $f}' $W/models.conf; }

check_models() {
  local tag f
  for tag in $MODELS; do
    f=$MODELS_DIR/$(model_field $tag 2)
    [ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" = "$(model_field $tag 3)" ] || fail "model $f is missing or differs from models.conf"
    FLM_MODEL_PATH=$W/flm-models $FLM_BIN check $(model_field $tag 4) 2>&1 | grep -q "check completed successfully" \
      || fail "flm model $(model_field $tag 4) not installed; run update.sh"
  done
}

preflight() {
  timeout $NPU_CHECK_TIMEOUT xrt-smi examine > $RUN/npu.txt 2>&1; grep -q "NPU" $RUN/npu.txt || fail "no NPU in xrt-smi examine"
  python3 -I $W/bench_npu.py sensors > $RUN/sensors.json || fail "power or temperature sensor not found"
  [ "$(sha256sum $W/prompts.json 2>/dev/null | cut -d' ' -f1)" = "$PROMPTS_SHA256" ] || fail "prompts.json missing or changed; run setup.sh"
  check_models
}

write_run_json() {
  HW="$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'), $(awk -F'|' '/^\|\[/ {gsub(/^ +| +$/, "", $3); print $3; exit}' $RUN/npu.txt), XRT $(awk -F: '/^ *Version/ {gsub(/ /, "", $2); print $2; exit}' $RUN/npu.txt), NPU firmware $(awk -F: '/NPU Firmware Version/ {gsub(/ /, "", $2); print $2}' $RUN/npu.txt)" \
  SERVER="${SERVER_FLAGS[*]}" BENCH="--contexts $CONTEXTS --max-tokens $MAX_TOKENS ${ARGS[*]}" \
  MODELS="$MODELS" CONTEXTS=$CONTEXTS MAX_TOKENS=$MAX_TOKENS DATE=$DATE CONF=$W/models.conf MDIR=$MODELS_DIR \
  python3 -I - > $RUN/run.json <<'EOF'
import json, os, socket, time
e = os.environ
rows = {r[0]: r for r in (line.split() for line in open(e["CONF"])) if r and not r[0].startswith("#")}
models = e["MODELS"].split()
env = "; ".join(f"{m.upper()} {'unset' if rows[m][4] == '-' else rows[m][4]}" for m in models)
json.dump({"machine": socket.gethostname(), "date": e["DATE"], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "hardware": e["HW"], "versions": {"xdna": e["FAB_VER"], "flm": "v" + e["FLM_VER"]},
           "models": {m: {"file": rows[m][1], "sha256": rows[m][2]} for m in models},
           "server_flags": e["SERVER"], "bench_args": e["BENCH"], "omp_policy": f"OpenMP wait policy {env}",
           "max_tokens": int(e["MAX_TOKENS"]), "contexts": [int(c) for c in e["CONTEXTS"].split(",")],
           "models_order": models}, open(1, "w"), indent=1)
EOF
}

run_model() { # <tag>
  quiet "$1 xdna"; $W/session.sh $RUN $1 xdna main --contexts $CONTEXTS --max-tokens $MAX_TOKENS "${ARGS[@]}"
  quiet "$1 flm"; $W/session.sh $RUN $1 flm main --contexts $CONTEXTS --max-tokens $MAX_TOKENS "${ARGS[@]}"
  quiet "$1 drift"; $W/session.sh $RUN $1 xdna drift --max-tokens $MAX_TOKENS "${DRIFT[@]}"
  echo "MODEL_DONE $1 $(date +%T)"
}

run_matrix() {
  local tag
  for tag in $MODELS; do run_model $tag; done
}

take_lock
trap 'rm -f $GPU_LOCK; [ -n "${watcher:-}" ] && kill $watcher 2>/dev/null' EXIT
[ $QUICK = 0 ] && { $W/update.sh $RUN || echo "update.sh reported errors, see updates.txt"; }
[ -f $W/engines.env ] || fail "no engines.env; run update.sh once by hand"
source $W/engines.env; cp $W/engines.env $W/bench.conf $W/models.conf $RUN/
[ -x "$FAB_BIN/llama-server" ] && [ -x "$FLM_BIN" ] || fail "an engine is missing; run update.sh"
export FAB_VER FLM_VER
preflight
write_run_json || fail "run.json"
watch_foreign > $RUN/foreign.log 2>&1 &
watcher=$!
run_matrix
python3 -I $W/summarize.py $RUN --status ok || fail "summarize.py failed"
touch $RUN/DONE
echo "end $(date)"
