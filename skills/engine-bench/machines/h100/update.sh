#!/bin/bash
# usage: update.sh <run dir> [engine ...]     engines: llama fabric vllm sglang (default: all)
# Brings every engine to its latest version: llama.cpp master and qvac-fabric temp-11259 (rebuilt for CUDA sm_90a
# when their head moves), vLLM and SGLang from their latest PyPI releases. Each new version is installed into its own
# folder and only becomes current once it installs and passes a quick check, so a failed update keeps the previous
# install. Writes engines.env (paths and versions) and <run dir>/updates.txt.
# H100 copy of the DGX Spark script: CUDA 12.8 (the box's only nvcc), sm_90a, no CUTLASS (its fabric path is NVFP4 on
# Blackwell), cmake and ninja from venvs/tools, and the checks run on the bench's GPU (BENCH_GPU, default 7).
set -u
RUN=$1; shift; W=${BENCH_DIR:-$HOME/bench-weekly}
ENGINES=${*:-llama fabric vllm sglang}
export PATH=$HOME/.local/bin:$W/venvs/tools/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${BENCH_GPU:-7}
CUDA=/usr/local/cuda-12.8
mkdir -p $W/src $W/bin $W/venvs $RUN/logs
UPD=$RUN/updates.txt; touch $UPD
[ -f $W/engines.env ] && source $W/engines.env
ERR=0
note() { echo "$*" >> $UPD; echo "$*"; }
pypi_latest() { curl -sSf https://pypi.org/pypi/$1/json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'; }

# llama.cpp-style tree: fetch the branch head, build static llama-server/llama-bench for CUDA, copy them to bin/<name>-<sha>
# usage: build_llama <name> <url> <branch>; on success sets NEW_BIN and NEW_VER
build_llama() {
  local name=$1 url=$2 br=$3 src=$W/src/$1 sha day dst
  if [ ! -d $src/.git ]; then
    git clone -q --depth 1 --branch $br $url $src || return 1
  else
    git -C $src fetch -q --depth 1 origin $br && git -C $src reset -q --hard FETCH_HEAD || return 1
  fi
  sha=$(git -C $src rev-parse --short=9 HEAD); day=$(git -C $src log -1 --format=%cs)
  dst=$W/bin/$name-$sha
  if [ ! -x $dst/llama-server ]; then
    cmake -S $src -B $src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=90a-real -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_CUTLASS=OFF -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF > $RUN/logs/cmake-$name.log 2>&1 || return 1
    cmake --build $src/build -j ${BUILD_JOBS:-32} --target llama-server llama-bench > $RUN/logs/build-$name.log 2>&1 || return 1
    mkdir -p $dst.tmp && cp $src/build/bin/llama-server $src/build/bin/llama-bench $dst.tmp/ && mv $dst.tmp $dst || return 1
  fi
  $dst/llama-server --version > /dev/null 2>&1 || return 1
  $dst/llama-server --list-devices 2>&1 | grep -q "CUDA0" || { echo "$name: no CUDA device"; return 1; }
  NEW_BIN=$dst; NEW_VER="$br @ $sha ($day)"
}

update_llama() {
  build_llama llama https://github.com/ggml-org/llama.cpp.git master || return 1
  [ "${UP_VER:-}" != "$NEW_VER" ] && note "llama.cpp: ${UP_VER:-none} -> $NEW_VER" || note "llama.cpp: unchanged, $NEW_VER"
  UP_BIN=$NEW_BIN; UP_VER=$NEW_VER
}

update_fabric() {
  build_llama fabric https://github.com/tetherto/qvac-fabric-llm.cpp.git temp-11259 || return 1
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

# vLLM's and SGLang's FP8 GEMMs (DeepGEMM, FlashInfer) compile kernels on first use and need an nvcc >= 12.9 that
# matches its CUDA headers, plus cuBLAS and friends to build against; the box has only CUDA 12.8, and the venvs' pip
# toolkits mix versions (nvcc 13.4 with 13.2 headers). So keep a pip toolkit (NVIDIA's cuda-toolkit metapackage, the
# release matching torch's CUDA version) in cuda/<version>; run_engine.sh points CUDA_HOME at it.
ensure_cuda() { # <python>
  local ver d c tk f
  ver=$($1 -c 'import torch; print(torch.version.cuda)') || return 1
  d=$W/cuda/$ver
  ls $d/nvidia/cu*/bin/nvcc > /dev/null 2>&1 && return 0
  tk=$(curl -sSf https://pypi.org/pypi/cuda-toolkit/json | python3 -c "import json, sys; r = json.load(sys.stdin)['releases']
print(sorted((x for x in r if x.startswith('$ver.') and x.replace('.', '').isdigit()), key=lambda s: [int(t) for t in s.split('.')])[-1])") \
    || return 1
  rm -rf $d.tmp
  uv pip install -q --python $W/venvs/tools/bin/python --target $d.tmp \
      "cuda-toolkit[nvcc,cudart,cublas,cccl,crt,curand,cusparse,cusolver,cufft,nvjitlink]==$tk" > $RUN/logs/install-cuda-$ver.log 2>&1 \
    || return 1
  # the CUDA_HOME layout JIT builds expect: lib64 and unversioned library names for -l
  for c in $d.tmp/nvidia/cu*; do
    ln -s lib $c/lib64
    for f in $c/lib/lib*.so.*; do [ -e ${f%%.so.*}.so ] || ln -s $(basename $f) ${f%%.so.*}.so; done
  done
  mv $d.tmp $d && note "CUDA toolkit for JIT kernels: cuda-toolkit $tk (pip) in $d"
}

update_vllm() {
  pip_engine vLLM vllm "" "import vllm, torch; assert torch.cuda.is_available()" || return 1
  ensure_cuda $NEW_PY || return 1
  [ "${VLLM_VER:-}" != "$NEW_VER" ] && note "vLLM: ${VLLM_VER:-none} -> $NEW_VER" || note "vLLM: unchanged, $NEW_VER"
  VLLM_PY=$NEW_PY; VLLM_VER=$NEW_VER
}

update_sglang() {
  pip_engine SGLang sglang "[all]" "import sglang, sgl_kernel, torch; assert torch.cuda.is_available()" || return 1
  ensure_cuda $NEW_PY || return 1
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
