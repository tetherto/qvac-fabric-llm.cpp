#!/bin/bash
# Brings both TurboVec implementations to their latest version and writes engines.env:
#   - qvac-fabric: the newest temp-<n> branch of tetherto/qvac-fabric-llm.cpp (highest n), its
#     ggml-vector-index library built for this CPU (rebuilt only when the branch tip moves) plus bench_fabric linked
#     to it, in bin/fabric-<sha>. A build whose header has no thread/staged API (the 0.9 port, before the 1.1.2
#     stack merges) gets bench_fabric without those calls, so only its 1-thread full-scan rows run.
#   - turbovec: the newest stable PyPI release in venvs/turbovec-<ver> (with numpy; its python runs the drivers).
# A failed update keeps the previous build. Notes go to <run>/updates.txt.
#   update.sh <run dir>
set -u
T=${TV_DIR:-$HOME/bench-turbovec}
source $T/machine.conf
RUN=$1; mkdir -p $RUN/logs
note() { echo "$*" | tee -a $RUN/updates.txt; }
[ -f $T/engines.env ] && source $T/engines.env
REPO=https://github.com/tetherto/qvac-fabric-llm.cpp.git
ERR=0

update_fabric() {
  local br sha s d src b flags lib
  br=$(git ls-remote --heads $REPO 'temp-*' | sed 's#.*refs/heads/##' | grep -E '^temp-[0-9]+$' | sort -t- -k2 -n | tail -1)
  [ -n "$br" ] || { note "qvac-fabric: FAILED, no temp-<n> branch found"; return 1; }
  sha=$(git ls-remote $REPO refs/heads/$br | cut -c1-40); s=${sha:0:9}
  d=$T/bin/fabric-$s
  if [ ! -x $d/bench_fabric ]; then
    src=$T/src/fabric
    if [ ! -d $src/.git ]; then
      git clone -q --depth 1 --branch $br $REPO $src > $RUN/logs/fabric-clone.log 2>&1 || { note "qvac-fabric: FAILED to clone $br"; return 1; }
    else
      git -C $src fetch -q --depth 1 origin $br > $RUN/logs/fabric-fetch.log 2>&1 && git -C $src checkout -q -f FETCH_HEAD \
        || { note "qvac-fabric: FAILED to fetch $br"; return 1; }
    fi
    [ "$(git -C $src rev-parse HEAD)" = "$sha" ] || { note "qvac-fabric: FAILED, checkout is not $br @ $s"; return 1; }
    b=$src/build-$s
    { cmake -S $src -B $b -DCMAKE_BUILD_TYPE=Release -DGGML_VECTOR_INDEX=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      && cmake --build $b -j $BUILD_JOBS --target ggml-vector-index; } > $RUN/logs/fabric-build.log 2>&1 \
      || { note "qvac-fabric: FAILED to build $br @ $s (logs/fabric-build.log)"; return 1; }
    # the 1.1.2 stack adds the thread, staged-search and calibration calls; the 0.9 port has none of them
    flags=""
    grep -q ggml_vec_index_set_threads $src/ggml/include/ggml-vector-index.h \
      && grep -q ggml_vec_index_turbovec_set_staged $src/ggml/include/ggml-vector-index.h && flags="-DBENCH_HAS_THREADS"
    grep -q ggml_vec_index_turbovec_calibrate $src/ggml/include/ggml-vector-index.h && flags="$flags -DBENCH_HAS_CALIBRATE"
    rm -rf $d.tmp; mkdir -p $d.tmp
    for lib in $(find $b -name 'libggml-vector-index*' \( -name '*.so*' -o -name '*.dylib' \)); do cp -P $lib $d.tmp/; done
    ls $d.tmp/libggml-vector-index* > /dev/null 2>&1 || { note "qvac-fabric: FAILED, no libggml-vector-index in the build"; return 1; }
    c++ -O2 -std=c++17 $flags -I$src/ggml/include $T/bench/bench_fabric.cpp -L$d.tmp -lggml-vector-index \
        -Wl,-rpath,$d -o $d.tmp/bench_fabric > $RUN/logs/fabric-bench-build.log 2>&1 \
      || { note "qvac-fabric: FAILED to build bench_fabric (logs/fabric-bench-build.log)"; return 1; }
    echo "${flags# }" > $d.tmp/flags
    git -C $src log -1 --format=%cs > $d.tmp/date
    rm -rf $d; mv $d.tmp $d
    rm -rf $b
  fi
  local api=0; grep -q BENCH_HAS_THREADS $d/flags && api=1
  local new="$br @ $s ($(cat $d/date))"
  [ "${FAB_VER:-}" != "$new" ] && note "qvac-fabric: ${FAB_VER:-none} -> $new" || note "qvac-fabric: unchanged, $new"
  [ "${FAB_API:-}" != "$api" ] && [ -n "${FAB_API:-}" ] && note "qvac-fabric: thread/staged API $([ $api = 1 ] && echo appeared || echo disappeared)"
  FAB_VER=$new; FAB_BIN=$d; FAB_API=$api; FAB_BRANCH=$br; FAB_SHA=$s
}

update_turbovec() {
  local ver d
  ver=$(curl -sSf https://pypi.org/pypi/turbovec/json | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
ok = [v for v, files in d["releases"].items() if re.fullmatch(r"\d+(\.\d+)*", v) and files and not all(f.get("yanked") for f in files)]
print(max(ok, key=lambda v: [int(x) for x in v.split(".")]))') || { note "turbovec: FAILED to read PyPI"; return 1; }
  d=$T/venvs/turbovec-$ver
  if ! $d/bin/python -c 'import turbovec, numpy' > /dev/null 2>&1; then
    rm -rf $d
    if command -v uv > /dev/null; then
      uv venv -q --python python3 $d && uv pip install -q --python $d/bin/python "turbovec==$ver" numpy
    else
      python3 -m venv $d && $d/bin/pip install -q --upgrade pip && $d/bin/pip install -q "turbovec==$ver" numpy
    fi > $RUN/logs/turbovec-install.log 2>&1 && $d/bin/python -c 'import turbovec, numpy' \
      || { rm -rf $d; note "turbovec: FAILED to install $ver (logs/turbovec-install.log)"; return 1; }
  fi
  [ "${TV_VER:-}" != "$ver" ] && note "turbovec: ${TV_VER:-none} -> $ver" || note "turbovec: unchanged, $ver"
  TV_VER=$ver; TV_PY=$d/bin/python
}

update_fabric || ERR=1
update_turbovec || ERR=1
cat > $T/engines.env <<EOF
FAB_VER="${FAB_VER:-}"
FAB_BIN=${FAB_BIN:-}
FAB_API=${FAB_API:-}
FAB_BRANCH=${FAB_BRANCH:-}
FAB_SHA=${FAB_SHA:-}
TV_VER=${TV_VER:-}
TV_PY=${TV_PY:-}
EOF
# keep the current and one previous build or venv of each
prune() { ls -dt $1/$2* 2>/dev/null | grep -v -x "$3" | tail -n +2 | xargs rm -rf; }
prune $T/bin fabric- "${FAB_BIN:-x}"; prune $T/venvs turbovec- "$(dirname "$(dirname "${TV_PY:-x/x/x}")")"
exit $ERR
