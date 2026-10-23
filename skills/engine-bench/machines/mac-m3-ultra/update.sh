#!/bin/bash
# usage: update.sh <run dir>
# Brings every engine to its latest version: llama.cpp master and the qvac-fabric PR stack branch (rebuilt when their
# head moves), mlx-lm and mlx from PyPI, oMLX, SGLang and Splash from their latest GitHub releases. Each new version is
# installed into its own folder and only becomes current once it installs and passes a quick check, so a failed
# update keeps the previous install. Writes engines.env (paths and versions) and <run dir>/updates.txt.
set -u
RUN=$1; W=${BENCH_DIR:-$HOME/bench-weekly}
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p $W/src $W/bin $W/engines $W/venvs $RUN/logs
UPD=$RUN/updates.txt; : > $UPD
[ -f $W/engines.env ] && source $W/engines.env
ERR=0
note() { echo "$*" >> $UPD; echo "$*"; }
gh_latest() { curl -sSf https://api.github.com/repos/$1/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'; }
pypi_latest() { curl -sSf https://pypi.org/pypi/$1/json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'; }

# llama.cpp-style tree: fetch the branch head, build static llama-server/llama-bench, copy them to bin/<name>-<sha>
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
    cmake -S $src -B $src/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > $RUN/logs/cmake-$name.log 2>&1 || return 1
    cmake --build $src/build -j 16 --target llama-server llama-bench > $RUN/logs/build-$name.log 2>&1 || return 1
    mkdir -p $dst.tmp && cp $src/build/bin/llama-server $src/build/bin/llama-bench $dst.tmp/ && mv $dst.tmp $dst || return 1
  fi
  $dst/llama-server --version > /dev/null 2>&1 || return 1
  NEW_BIN=$dst; NEW_VER="$br @ $sha ($day)"
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

update_mlx() {
  local lm mx ver d
  lm=$(pypi_latest mlx-lm) && mx=$(pypi_latest mlx) || return 1
  ver="mlx-lm $lm, mlx $mx"
  [ "${MLX_VER:-}" = "$ver" ] && [ -x "${MLX_PY:-}" ] && { note "mlx-lm: unchanged, $ver"; return 0; }
  d=$W/venvs/mlx-lm-$lm-mlx-$mx; rm -rf $d
  { uv venv -q -p 3.12 $d && uv pip install --python $d/bin/python "mlx-lm==$lm" "mlx==$mx"; } > $RUN/logs/install-mlx.log 2>&1 || return 1
  $d/bin/python -c "import mlx_lm, mlx.core" || return 1
  note "mlx-lm: ${MLX_VER:-none} -> $ver"
  MLX_PY=$d/bin/python; MLX_VER=$ver
}

update_omlx() {
  local tag ver d
  tag=$(gh_latest jundot/omlx) || return 1; ver=${tag#v}
  [ "${OMLX_VER:-}" = "$ver" ] && { note "oMLX: unchanged, $ver"; return 0; }
  d=$W/engines/omlx-$ver; rm -rf $d
  git clone -q --depth 1 --branch $tag https://github.com/jundot/omlx.git $d || return 1
  ( cd $d && uv venv -q -p 3.12 .venv && uv pip install --python .venv/bin/python pip \
      && OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/python -m pip install -e . ) > $RUN/logs/install-omlx.log 2>&1 || return 1
  $d/.venv/bin/omlx serve --help > /dev/null 2>&1 || return 1
  note "oMLX: ${OMLX_VER:-none} -> $ver"
  OMLX_DIR=$d; OMLX_VER=$ver
}

update_sglang() {
  local tag ver d
  tag=$(gh_latest sgl-project/sglang) || return 1; ver=${tag#v}
  [ "${SGL_VER:-}" = "$ver" ] && { note "SGLang: unchanged, $ver"; return 0; }
  d=$W/engines/sglang-$ver; rm -rf $d
  git clone -q --depth 1 --branch $tag https://github.com/sgl-project/sglang.git $d || return 1
  ( cd $d && uv venv -q -p 3.12 .venv && source .venv/bin/activate && uv pip install --upgrade pip \
      && { uv run --active python python/sglang/kernels/aot/setup_metal.py install || echo "Metal kernels failed (optional)"; } \
      && rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
      && SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python[all_mps]" ) > $RUN/logs/install-sglang.log 2>&1 || return 1
  $d/.venv/bin/python -c "import sglang" || return 1
  note "SGLang: ${SGL_VER:-none} -> $ver"
  SGL_DIR=$d; SGL_VER=$ver
}

update_splash() {
  local tag ver d a u
  tag=$(gh_latest incoai/splash) || return 1; ver=${tag#v}
  [ "${SPLASH_VER:-}" = "$ver" ] && { note "Splash: unchanged, $ver"; return 0; }
  d=$W/engines/splash-$ver; rm -rf $d; mkdir -p $d
  a=splash-$ver-arm64-macos26.tar.gz; u=https://github.com/incoai/splash/releases/download/$tag
  curl -sSfL -o $d/$a $u/$a && curl -sSfL -o $d/$a.sha256 $u/$a.sha256 || return 1
  [ "$(shasum -a 256 $d/$a | cut -d' ' -f1)" = "$(cut -d' ' -f1 $d/$a.sha256)" ] || { echo "Splash checksum mismatch"; return 1; }
  tar -xzf $d/$a -C $d && rm -f $d/$a || return 1
  [ -x $d/splash-$ver-arm64-macos26/engine/splash ] || return 1
  note "Splash: ${SPLASH_VER:-none} -> $ver"
  SPLASH_DIR=$d/splash-$ver-arm64-macos26; SPLASH_VER=$ver
}

for e in llama fabric mlx omlx sglang splash; do
  update_$e || { note "$e: update FAILED, keeping the previous install"; ERR=1; }
done

cat > $W/engines.env.new <<EOF
UP_BIN='${UP_BIN:-}'
UP_VER='${UP_VER:-}'
FAB_BIN='${FAB_BIN:-}'
FAB_VER='${FAB_VER:-}'
MLX_PY='${MLX_PY:-}'
MLX_VER='${MLX_VER:-}'
OMLX_DIR='${OMLX_DIR:-}'
OMLX_VER='${OMLX_VER:-}'
SGL_DIR='${SGL_DIR:-}'
SGL_VER='${SGL_VER:-}'
SPLASH_DIR='${SPLASH_DIR:-}'
SPLASH_VER='${SPLASH_VER:-}'
EOF
mv $W/engines.env.new $W/engines.env

# keep the current install and the one before it in each weekly folder; never touches ~/engines
prune() { # <dir> <glob prefix> <current path>
  local n=0 p
  for p in $(ls -dt $1/$2* 2>/dev/null); do
    case "$3" in "$p"|"$p"/*) continue ;; esac
    n=$((n + 1)); [ $n -gt 1 ] && rm -rf "$p"
  done
}
prune $W/bin llama- "$UP_BIN"; prune $W/bin fabric- "$FAB_BIN"; prune $W/venvs mlx-lm- "$(dirname "$(dirname "${MLX_PY:-x/x/x}")")"
prune $W/engines omlx- "${OMLX_DIR:-}"; prune $W/engines sglang- "${SGL_DIR:-}"; prune $W/engines splash- "${SPLASH_DIR:-}"
exit $ERR
