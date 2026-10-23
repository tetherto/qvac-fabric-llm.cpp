#!/bin/bash
# usage: update.sh <run dir> [engine ...]     engines: llama fabric vllm sglang (default: all)
# Brings every engine to its latest version: llama.cpp master and the qvac-fabric PR stack branch (rebuilt for CUDA
# sm_121a when their head moves), vLLM and SGLang from their latest PyPI releases. Each new version is installed into
# its own folder and only becomes current once it installs and passes a quick check, so a failed update keeps the
# previous install. Writes engines.env (paths and versions) and <run dir>/updates.txt.
set -u
RUN=$1; shift; W=${BENCH_DIR:-$HOME/bench-weekly}
ENGINES=${*:-llama fabric vllm sglang}
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
CUDA=/usr/local/cuda-13.0
mkdir -p $W/src $W/bin $W/venvs $RUN/logs
UPD=$RUN/updates.txt; touch $UPD
[ -f $W/engines.env ] && source $W/engines.env
ERR=0
note() { echo "$*" >> $UPD; echo "$*"; }
pypi_latest() { curl -sSf https://pypi.org/pypi/$1/json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'; }

# qvac-fabric's NVFP4 prefill path (GGML_CUDA_CUTLASS, ~1.4x prefill on the GB10) is built once the branch has the fix
# that keeps empty MUL_MATs on repacked weights on the GPU: without it every ubatch without logits copies the 715 MB
# lm_head to the CPU and short prompts get slower than MMQ (measured 2026-10-22). FAB_CUTLASS=1 or 0 forces it.
fabric_cutlass_ready() { # <src>
  case ${FAB_CUTLASS:-auto} in 1) return 0 ;; 0) return 1 ;; esac
  awk '/^static bool ggml_backend_cuda_repacked_supports_op\(/,/^}/' $1/ggml/src/ggml-cuda/ggml-cuda.cu | grep -q 'ggml_is_empty(op)'
}

# shallow checkout of the CUTLASS tag the tree pins, so CMake does not clone the whole repository; prints its path
cutlass_src() { # <src>
  local tag d
  tag=$(awk '/FetchContent_Declare\(/ {getline; name = $1} name == "CUTLASS" && $1 == "GIT_TAG" {print $2; exit}' \
      $1/ggml/src/ggml-cuda/CMakeLists.txt)
  [ -n "$tag" ] || return 1
  d=$W/src/cutlass-$tag
  [ -d $d/include ] || git -c advice.detachedHead=false clone -q --depth 1 --branch $tag https://github.com/NVIDIA/cutlass.git $d || return 1
  echo $d
}

# llama.cpp-style tree: fetch the branch head, build static llama-server/llama-bench for CUDA, copy them to bin/<name>-<sha>
# usage: build_llama <name> <url> <branch>; on success sets NEW_BIN and NEW_VER
build_llama() {
  local name=$1 url=$2 br=$3 src=$W/src/$1 sha day dst cutlass extra=() suffix=""
  if [ ! -d $src/.git ]; then
    git clone -q --depth 1 --branch $br $url $src || return 1
  else
    git -C $src fetch -q --depth 1 origin $br && git -C $src reset -q --hard FETCH_HEAD || return 1
  fi
  sha=$(git -C $src rev-parse --short=9 HEAD); day=$(git -C $src log -1 --format=%cs)
  if [ $name = fabric ]; then
    extra=(-DGGML_CUDA_CUTLASS=OFF)
    if fabric_cutlass_ready $src; then
      extra=(-DGGML_CUDA_CUTLASS=ON); suffix=-cutlass
      cutlass=$(cutlass_src $src) && extra+=(-DFETCHCONTENT_SOURCE_DIR_CUTLASS=$cutlass)
    fi
  fi
  dst=$W/bin/$name-$sha$suffix
  if [ ! -x $dst/llama-server ]; then
    cmake -S $src -B $src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=121a-real -DGGML_CUDA_NCCL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF "${extra[@]}" > $RUN/logs/cmake-$name.log 2>&1 || return 1
    # at most 8 jobs: CUDA template compiles take 3-6 GB each and this box is shared
    cmake --build $src/build -j ${BUILD_JOBS:-8} --target llama-server llama-bench > $RUN/logs/build-$name.log 2>&1 || return 1
    mkdir -p $dst.tmp && cp $src/build/bin/llama-server $src/build/bin/llama-bench $dst.tmp/ && mv $dst.tmp $dst || return 1
  fi
  $dst/llama-server --version > /dev/null 2>&1 || return 1
  $dst/llama-server --list-devices 2>&1 | grep -q "CUDA0" || { echo "$name: no CUDA device"; return 1; }
  NEW_BIN=$dst; NEW_VER="$br @ $sha ($day)${suffix:+, CUTLASS}"
}

update_llama() {
  build_llama llama https://github.com/ggml-org/llama.cpp.git master || return 1
  [ "${UP_VER:-}" != "$NEW_VER" ] && note "llama.cpp: ${UP_VER:-none} -> $NEW_VER" || note "llama.cpp: unchanged, $NEW_VER"
  UP_BIN=$NEW_BIN; UP_VER=$NEW_VER
}

update_fabric() {
  build_llama fabric https://github.com/tetherto/qvac-fabric-llm.cpp.git dflash-deferred-commit-temp-11259 || return 1
  [ "${FAB_VER:-}" != "$NEW_VER" ] && note "qvac-fabric: ${FAB_VER:-none} -> $NEW_VER" || note "qvac-fabric: unchanged, $NEW_VER"
  FAB_BIN=$NEW_BIN; FAB_VER=$NEW_VER
}

# usage: pip_engine <label> <pypi name> <pip spec suffix> <import check>; sets NEW_PY and NEW_VER
pip_engine() {
  local label=$1 pkg=$2 extra=$3 check=$4 ver d
  ver=$(pypi_latest $pkg) || return 1
  d=$W/venvs/$pkg-$ver
  if [ -x $d/bin/python ] && $d/bin/python -c "$check" > /dev/null 2>&1; then
    NEW_PY=$d/bin/python; NEW_VER=$ver; return 0
  fi
  rm -rf $d
  { uv venv -q -p 3.12 $d && uv pip install --python $d/bin/python "$pkg$extra==$ver" --torch-backend=auto; } \
      > $RUN/logs/install-$pkg.log 2>&1 || return 1
  $d/bin/python -c "$check" >> $RUN/logs/install-$pkg.log 2>&1 || return 1
  NEW_PY=$d/bin/python; NEW_VER=$ver
}

update_vllm() {
  pip_engine vLLM vllm "" "import vllm, torch; assert torch.cuda.is_available()" || return 1
  [ "${VLLM_VER:-}" != "$NEW_VER" ] && note "vLLM: ${VLLM_VER:-none} -> $NEW_VER" || note "vLLM: unchanged, $NEW_VER"
  VLLM_PY=$NEW_PY; VLLM_VER=$NEW_VER
}

update_sglang() {
  pip_engine SGLang sglang "[all]" "import sglang, sgl_kernel, torch; assert torch.cuda.is_available()" || return 1
  [ "${SGL_VER:-}" != "$NEW_VER" ] && note "SGLang: ${SGL_VER:-none} -> $NEW_VER" || note "SGLang: unchanged, $NEW_VER"
  SGL_PY=$NEW_PY; SGL_VER=$NEW_VER
}

for e in $ENGINES; do
  update_$e || { note "$e: update FAILED, keeping the previous install"; ERR=1; }
done

cat > $W/engines.env.new <<EOF
UP_BIN='${UP_BIN:-}'
UP_VER='${UP_VER:-}'
FAB_BIN='${FAB_BIN:-}'
FAB_VER='${FAB_VER:-}'
VLLM_PY='${VLLM_PY:-}'
VLLM_VER='${VLLM_VER:-}'
SGL_PY='${SGL_PY:-}'
SGL_VER='${SGL_VER:-}'
EOF
mv $W/engines.env.new $W/engines.env

# keep the current install and the one before it in each folder
prune() { # <dir> <glob prefix> <current path>
  local n=0 p
  for p in $(ls -dt $1/$2* 2>/dev/null); do
    case "$3" in "$p"|"$p"/*) continue ;; esac
    n=$((n + 1)); [ $n -gt 1 ] && rm -rf "$p"
  done
}
prune $W/bin llama- "${UP_BIN:-}"; prune $W/bin fabric- "${FAB_BIN:-}"
prune $W/venvs vllm- "${VLLM_PY:-}"; prune $W/venvs sglang- "${SGL_PY:-}"
exit $ERR
