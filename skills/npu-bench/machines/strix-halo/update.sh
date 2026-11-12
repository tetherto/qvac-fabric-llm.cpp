#!/bin/bash
# usage: update.sh <run dir> [fabric] [flm]     (default: both)
# Builds qvac-fabric at the head of XDNA_BRANCH with the XDNA backend and its NPU kernels, and installs the newest
# FastFlowLM portable release plus the flm models in models.conf (FLM_PULL limits the tags). Each version goes into
# its own folder and becomes current only after it passes a check, so a failed update keeps the previous one.
# Writes engines.env and <run dir>/updates.txt.
set -u
RUN=$1; shift; W=${BENCH_DIR:-$HOME/npu-bench}
source $W/bench.conf
export PATH=/opt/xilinx/xrt/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
FABRIC_URL=https://github.com/tetherto/qvac-fabric-llm.cpp.git
FLM_REPO=ROCm/FastFlowLM
PULL_TIMEOUT=3600
JOBS=$(nproc)
mkdir -p $W/src $W/bin $W/engines $W/flm-models $RUN/logs
UPD=$RUN/updates.txt; touch $UPD
[ -f $W/engines.env ] && source $W/engines.env

note() { echo "$*" >> $UPD; echo "$*"; }

# fetch_fabric: shallow clone or fast-forward to the branch head; sets SHA and DAY
fetch_fabric() {
  local src=$W/src/fabric
  if [ ! -d $src/.git ]; then
    git clone -q --depth 1 --branch $XDNA_BRANCH $FABRIC_URL $src || return 1
  else
    git -C $src fetch -q --depth 1 origin $XDNA_BRANCH && git -C $src reset -q --hard FETCH_HEAD || return 1
  fi
  SHA=$(git -C $src rev-parse --short=9 HEAD); DAY=$(git -C $src log -1 --format=%cs)
}

# build_fabric <dst>: static llama-server plus the NPU kernel artifacts, which the backend finds next to the binary
build_fabric() {
  local src=$W/src/fabric dst=$1 n
  rm -rf $src/build
  cmake -S $src -B $src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON -DGGML_XDNA=ON -DGGML_OPENMP=ON \
    -DGGML_XDNA_BUILD_KERNELS=ON -DGGML_XDNA_GEMM_PYTHON=$AIE_PYTHON > $RUN/logs/cmake-fabric.log 2>&1 || return 1
  grep -q "skipping kernel build" $RUN/logs/cmake-fabric.log && { note "fabric: IRON not found in $AIE_PYTHON"; return 1; }
  cmake --build $src/build -j $JOBS --target llama-server ggml-xdna-kernels > $RUN/logs/build-fabric.log 2>&1 || return 1
  n=$(ls $src/build/bin/*.xclbin 2>/dev/null | wc -l)
  [ $n -gt 0 ] || { note "fabric: no NPU kernels built"; return 1; }
  rm -rf $dst.tmp && cp -a $src/build/bin $dst.tmp && mv $dst.tmp $dst || return 1
  rm -rf $src/build
  note "fabric: $n NPU kernels"
}

update_fabric() {
  local dst new
  fetch_fabric || return 1
  dst=$W/bin/fabric-$SHA; new="$XDNA_BRANCH @ $SHA ($DAY)"
  [ -x $dst/llama-server ] || build_fabric $dst || return 1
  $dst/llama-server --list-devices > $RUN/logs/devices-fabric.txt 2>&1
  grep -q XDNA0 $RUN/logs/devices-fabric.txt || { note "fabric: no XDNA0 device"; return 1; }
  [ "${FAB_VER:-none}" != "$new" ] && note "qvac-fabric: ${FAB_VER:-none} -> $new" || note "qvac-fabric: unchanged, $new"
  FAB_BIN=$dst; FAB_VER=$new
}

flm_tags() {
  awk '!/^#/ && NF {print $4}' $W/models.conf
}

# pull_flm_models <flm>: download (if missing) and check every model tag
pull_flm_models() {
  local flm=$1 tag
  for tag in ${FLM_PULL:-$(flm_tags)}; do
    FLM_MODEL_PATH=$W/flm-models timeout $PULL_TIMEOUT $flm pull $tag > $RUN/logs/flm-pull-${tag//:/_}.log 2>&1 \
      || { note "flm: pull $tag failed"; return 1; }
    FLM_MODEL_PATH=$W/flm-models $flm check $tag 2>&1 | grep -q "check completed successfully" \
      || { note "flm: check $tag failed"; return 1; }
  done
}

update_flm() {
  local tag ver d
  tag=$(curl -sSf https://api.github.com/repos/$FLM_REPO/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])') || return 1
  ver=${tag#v}; d=$W/engines/flm-$ver
  if [ ! -x $d/flm ]; then
    rm -rf $d.tmp && mkdir -p $d.tmp
    curl -sfL https://github.com/$FLM_REPO/releases/download/$tag/fastflowlm_${ver}_linux.tar.gz | tar xz -C $d.tmp || return 1
    mv $d.tmp $d
  fi
  $d/flm version 2>&1 | grep -q "v$ver" || { note "flm: $d/flm does not run"; return 1; }
  pull_flm_models $d/flm || return 1
  [ "${FLM_VER:-none}" != "$ver" ] && note "flm: ${FLM_VER:-none} -> $ver" || note "flm: unchanged, $ver"
  FLM_BIN=$d/flm; FLM_VER=$ver
}

# prune <dir> <prefix> <current>: keep the current install and the one before it
prune() {
  local n=0 p
  for p in $(ls -dt $1/$2* 2>/dev/null); do
    [ "$p" = "$3" ] && continue
    n=$((n + 1)); [ $n -gt 1 ] && rm -rf "$p"
  done
}

ERR=0
for e in ${*:-fabric flm}; do
  update_$e || { note "$e: update FAILED, keeping the previous install"; ERR=1; }
done
{ echo "FAB_BIN='${FAB_BIN:-}'"; echo "FAB_VER='${FAB_VER:-}'"; echo "FLM_BIN='${FLM_BIN:-}'"; echo "FLM_VER='${FLM_VER:-}'"; } > $W/engines.env.new
mv $W/engines.env.new $W/engines.env
prune $W/bin fabric- "${FAB_BIN:-}"
prune $W/engines flm- "$(dirname "${FLM_BIN:-x/x}")"
exit $ERR
