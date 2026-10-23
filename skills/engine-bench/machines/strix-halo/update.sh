#!/bin/bash
# usage: update.sh <run dir> [engine ...]     engines: llama strix fabric nath paoai gufo (default: all)
# Brings every engine on qvac-dev-strix-0 to its latest version: llama.cpp master, halo-box/strix-llama.cpp master,
# the qvac-fabric PR stack branch and two Vulkan-only Strix forks (Nathanw1014/llama.cpp strix-halo-vulkan,
# guevae2/paoai-strix-engine strix-main), rebuilt when their head moves for each backend listed in backends.conf, and
# gufo from its latest GitHub release. Each new version is built into its own folder and only becomes current once it
# passes a quick check, so a failed update keeps the previous install. Writes engines.env and <run dir>/updates.txt.
set -u
RUN=$1; shift; W=${BENCH_DIR:-$HOME/bench-weekly}
ROCM=/opt/rocm-7.2.4; DEPS=$W/deps/root
export PATH=$HOME/.local/bin:$ROCM/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p $W/src $W/bin $W/engines $RUN/logs
UPD=$RUN/updates.txt; touch $UPD
[ -f $W/engines.env ] && source $W/engines.env
source $W/backends.conf
ERR=0
JOBS=24
note() { echo "$*" >> $UPD; echo "$*"; }
gh_latest() { curl -sSf https://api.github.com/repos/$1/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'; }

# fetch_src <name> <url> <branch>: shallow clone or fast-forward to the branch head; sets SHA and DAY
fetch_src() {
  local src=$W/src/$1
  if [ ! -d $src/.git ]; then
    git clone -q --depth 1 --branch $3 $2 $src || return 1
  else
    git -C $src fetch -q --depth 1 origin $3 && git -C $src reset -q --hard FETCH_HEAD || return 1
  fi
  SHA=$(git -C $src rev-parse --short=9 HEAD); DAY=$(git -C $src log -1 --format=%cs)
}

# build_backend <name> <vulkan|rocm>: static llama-server and llama-bench into bin/<name>-<backend>-<sha>; sets NEW_BIN
build_backend() {
  local name=$1 be=$2 src=$W/src/$1 dst=$W/bin/$1-$2-$SHA flags ev=()
  if [ ! -x $dst/llama-server ]; then
    flags=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
           -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
    case $be in
      vulkan) flags+=(-DGGML_VULKAN=ON) ;;
      rocm)   flags+=(-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_PREFIX_PATH=$ROCM)
              ev=(HIPCXX=$ROCM/lib/llvm/bin/clang HIP_PATH=$ROCM) ;;
    esac
    rm -rf $src/build-$be
    env "${ev[@]}" cmake -S $src -B $src/build-$be "${flags[@]}" > $RUN/logs/cmake-$name-$be.log 2>&1 || return 1
    cmake --build $src/build-$be -j $JOBS --target llama-server llama-bench > $RUN/logs/build-$name-$be.log 2>&1 || return 1
    mkdir -p $dst.tmp && cp $src/build-$be/bin/llama-server $src/build-$be/bin/llama-bench $dst.tmp/ && mv $dst.tmp $dst || return 1
    rm -rf $src/build-$be
  fi
  $dst/llama-server --list-devices > $RUN/logs/devices-$name-$be.txt 2>&1 || return 1
  grep -qiE "gfx1151|8060S|RADV" $RUN/logs/devices-$name-$be.txt || return 1
  NEW_BIN=$dst
}

# update_family <name> <url> <branch> <VAR prefix> <label>: one source tree, one build per backend in <PREFIX>_BACKENDS
update_family() {
  local name=$1 url=$2 br=$3 P=$4 label=$5 be ok=1 old new bvar
  fetch_src $name $url $br || return 1
  new="$br @ $SHA ($DAY)"; old=$(eval echo "\${${P}_VER:-none}")
  for be in $(eval echo "\${${P}_BACKENDS}"); do
    bvar=${P}_$(echo $be | tr a-z A-Z)_BIN
    if build_backend $name $be; then
      eval "$bvar=$NEW_BIN"
    else
      note "$label $be: build FAILED at $SHA, keeping $(eval echo "\${$bvar:-nothing}")"; ok=0
    fi
  done
  [ "$old" != "$new" ] && note "$label: $old -> $new" || note "$label: unchanged, $new"
  eval "${P}_VER=\"\$new\""
  [ $ok = 1 ]
}

update_llama()  { update_family llama  https://github.com/ggml-org/llama.cpp.git master UP llama.cpp; }
update_strix()  { update_family strix  https://github.com/halo-box/strix-llama.cpp.git master STRIX strix-llama.cpp; }
update_fabric() { update_family fabric https://github.com/tetherto/qvac-fabric-llm.cpp.git dflash-deferred-commit-temp-11259 FAB qvac-fabric; }
update_nath()   { update_family nath   https://github.com/Nathanw1014/llama.cpp.git strix-halo-vulkan NATH nathanw-strix-vulkan; }
update_paoai()  { update_family paoai  https://github.com/guevae2/paoai-strix-engine.git strix-main PAOAI paoai-strix-engine; }

update_gufo() {
  local tag ver d icu
  tag=$(gh_latest gufo-org/gufo) || return 1; ver=${tag#v}
  [ "${GUFO_VER:-}" = "$ver" ] && [ -x "${GUFO_BIN:-}" ] && { note "gufo: unchanged, $ver"; return 0; }
  d=$W/engines/gufo-$ver
  if [ ! -x $d/build/release/gufo ]; then
    rm -rf $d
    git clone -q --depth 1 --branch $tag https://github.com/gufo-org/gufo.git $d || return 1
    # ICU headers come from the extracted libicu-dev; point its .so links at the system's runtime libraries
    for icu in $DEPS/usr/lib/x86_64-linux-gnu/libicu*.so; do
      ln -sf /usr/lib/x86_64-linux-gnu/$(basename $(readlink -f $icu)) $icu 2>/dev/null
    done
    ( cd $d && cmake --preset release -DCMAKE_PREFIX_PATH="$ROCM;$DEPS$ROCM" -DICU_ROOT=$DEPS/usr \
          -DCMAKE_LIBRARY_PATH=$DEPS/usr/lib/x86_64-linux-gnu -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
          -DGUFO_RELEASE_VERSION=$ver -DGUFO_REVISION=$(git -C $d rev-parse --short=9 HEAD) \
          -DCMAKE_HIP_FLAGS="-I$DEPS$ROCM/include" -DCMAKE_CXX_FLAGS="-I$DEPS$ROCM/include" \
        && cmake --build --preset release --parallel $JOBS ) > $RUN/logs/build-gufo.log 2>&1 || return 1
  fi
  # the ROCm libraries are not on the loader path and gufo has no rpath to them
  LD_LIBRARY_PATH=$ROCM/lib $d/build/release/gufo diagnose > $RUN/logs/gufo-diagnose.txt 2>&1 || return 1
  rm -rf $d/build/release/CMakeFiles
  note "gufo: ${GUFO_VER:-none} -> $ver"
  GUFO_BIN=$d/build/release/gufo; GUFO_VER=$ver
}

ENGINES=${*:-llama strix fabric nath paoai gufo}
FAMS="UP STRIX FAB NATH PAOAI"
for e in $ENGINES; do
  update_$e || { note "$e: update FAILED, keeping the previous install"; ERR=1; }
done

# another update.sh may have written engines.env meanwhile: keep its values for the engines not updated here
declare -A KEEP
for e in $ENGINES; do
  case $e in llama) P=UP ;; strix) P=STRIX ;; fabric) P=FAB ;; nath) P=NATH ;; paoai) P=PAOAI ;; gufo) P=GUFO ;; esac
  vs="${P}_VULKAN_BIN ${P}_ROCM_BIN ${P}_VER"; [ $P = GUFO ] && vs="GUFO_BIN GUFO_VER"
  for v in $vs; do KEEP[$v]=${!v:-}; done
done
exec 9> $W/.engines.lock; flock 9
[ -f $W/engines.env ] && source $W/engines.env
for v in "${!KEEP[@]}"; do printf -v $v '%s' "${KEEP[$v]}"; done
{ for P in $FAMS; do for v in ${P}_VULKAN_BIN ${P}_ROCM_BIN ${P}_VER; do echo "$v='${!v:-}'"; done; done
  echo "GUFO_BIN='${GUFO_BIN:-}'"; echo "GUFO_VER='${GUFO_VER:-}'"; } > $W/engines.env.new
mv $W/engines.env.new $W/engines.env
flock -u 9

# keep the current install and the one before it for each engine and backend
prune() { # <dir> <glob prefix> <current path>
  local n=0 p
  for p in $(ls -dt $1/$2* 2>/dev/null); do
    case "$3" in "$p"|"$p"/*) continue ;; esac
    n=$((n + 1)); [ $n -gt 1 ] && rm -rf "$p"
  done
}
for n in llama:UP strix:STRIX fabric:FAB nath:NATH paoai:PAOAI; do
  for be in vulkan rocm; do
    v=${n#*:}_$(echo $be | tr a-z A-Z)_BIN
    prune $W/bin ${n%:*}-$be- "${!v:-}"
  done
done
prune $W/engines gufo- "$(dirname "$(dirname "$(dirname "${GUFO_BIN:-x/x/x/x}")")")"
exit $ERR
