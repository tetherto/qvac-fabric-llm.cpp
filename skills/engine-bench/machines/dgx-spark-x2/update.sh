#!/bin/bash
# usage: update.sh <run dir> [engine ...]     engines: llama fabric vllm (default: all)
# 2x DGX Spark (the head node, this one, + the second node in pair.conf). llama.cpp master and qvac-fabric temp-11259
# are fetched and built here (build.sh) when their head moves and copied to the same path on the second node; vLLM's
# latest PyPI release is installed into $W/venvs on both nodes. Each new version only becomes current once it installs
# and passes a check on both nodes, so a failed update keeps the previous one. Writes engines.env and
# <run dir>/updates.txt.
set -u
RUN=$1; shift; W=${BENCH_DIR:-$HOME/bench-pair}
source $W/pair.conf
ENGINES=${*:-llama fabric vllm}
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p $W/src $W/bin $W/venvs $RUN/logs
UPD=$RUN/updates.txt; touch $UPD
[ -f $W/engines.env ] && source $W/engines.env
ERR=0
note() { echo "$*" >> $UPD; echo "$*"; }
pypi_latest() { curl -sSf https://pypi.org/pypi/$1/json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'; }

# usage: build_llama <name> <url> <branch>; on success sets NEW_BIN and NEW_VER
build_llama() {
  local name=$1 url=$2 br=$3 src=$W/src/$1 sha day dst n
  if [ ! -d $src/.git ]; then
    git clone -q --depth 1 --branch $br $url $src || return 1
  else
    git -C $src fetch -q --depth 1 origin $br && git -C $src reset -q --hard FETCH_HEAD || return 1
  fi
  sha=$(git -C $src rev-parse --short=9 HEAD); day=$(git -C $src log -1 --format=%cs)
  $W/build.sh $RUN/logs $name || return 1
  dst=$W/bin/$name-$sha
  for n in "" peer; do
    $n $dst/llama-server --list-devices 2>&1 | grep -q "CUDA0" || { echo "$name: no CUDA device ${n:+on the peer}"; return 1; }
    $n test -x $dst/ggml-rpc-server || return 1
  done
  NEW_BIN=$dst; NEW_VER="$br @ $sha ($day)"
}

update_llama() {
  build_llama llama https://github.com/ggml-org/llama.cpp.git master || return 1
  [ "${UP_VER:-}" != "$NEW_VER" ] && note "llama.cpp: ${UP_VER:-none} -> $NEW_VER" || note "llama.cpp: unchanged, $NEW_VER"
  UP_BIN=$NEW_BIN; UP_VER=$NEW_VER
}

# the qvac-fabric release branch: the deferred gated delta net stack (PRs #413/#414) was merged into it on 2026-10-25,
# the rpc-server graph cache fix -sm tensor needs (PR #440) on 2026-10-28
update_fabric() {
  build_llama fabric https://github.com/tetherto/qvac-fabric-llm.cpp.git temp-11259 || return 1
  [ "${FAB_VER:-}" != "$NEW_VER" ] && note "qvac-fabric: ${FAB_VER:-none} -> $NEW_VER" || note "qvac-fabric: unchanged, $NEW_VER"
  FAB_BIN=$NEW_BIN; FAB_VER=$NEW_VER
}

update_vllm() {
  local ver d n
  ver=$(pypi_latest vllm) || return 1
  d=$W/venvs/vllm-$ver
  for n in "" peer; do
    $n $W/install-vllm.sh $ver $d >> $RUN/logs/install-vllm.log 2>&1 || { echo "vLLM $ver: install FAILED ${n:+on the peer}"; return 1; }
  done
  [ "${VLLM_VER:-}" != "$ver" ] && note "vLLM: ${VLLM_VER:-none} -> $ver" || note "vLLM: unchanged, $ver"
  VLLM_PY=$d/bin/python; VLLM_VER=$ver
}

for e in $ENGINES; do
  update_$e || { note "$e: update FAILED, keeping the previous install"; ERR=1; }
done

cat > $W/engines.env.new <<EOT
UP_BIN='${UP_BIN:-}'
UP_VER='${UP_VER:-}'
FAB_BIN='${FAB_BIN:-}'
FAB_VER='${FAB_VER:-}'
VLLM_PY='${VLLM_PY:-}'
VLLM_VER='${VLLM_VER:-}'
EOT
mv $W/engines.env.new $W/engines.env

# keep the current install and the one before it in each folder, on both nodes
prune() { # <dir> <glob prefix> <current path>
  local n=0 p
  for p in $(ls -dt $1/$2* 2>/dev/null); do
    case "$3" in "$p"|"$p"/*) continue ;; esac
    n=$((n + 1)); [ $n -gt 1 ] && { rm -rf "$p"; peer "rm -rf '$p'"; }
  done
}
prune $W/bin llama- "${UP_BIN:-}"; prune $W/bin fabric- "${FAB_BIN:-}"; prune $W/venvs vllm- "${VLLM_PY:-}"
exit $ERR
