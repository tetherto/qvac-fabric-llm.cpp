#!/bin/bash
# usage: build.sh <log dir> <name> ...
# Static CUDA (sm_121a) builds of llama-server, llama-bench and ggml-rpc-server with the RPC backend from
# $W/src/<name> (its RDMA transport is auto-detected from libibverbs), into $W/bin/<name>-<sha>, then copied to the
# same path on the second node.
set -u
W=${BENCH_DIR:-$HOME/bench-pair}; CUDA=/usr/local/cuda-13.0; LOGS=$1; shift
source $W/pair.conf
export PATH=/usr/local/bin:/usr/bin:/bin
ERR=0
for name in "$@"; do
  src=$W/src/$name; sha=$(git -C $src rev-parse --short=9 HEAD); dst=$W/bin/$name-$sha
  if [ ! -x $dst/ggml-rpc-server ]; then
    rpc=$(awk '/set\(TARGET/ {gsub(/\)/, "", $2); print $2}' $src/tools/rpc/CMakeLists.txt)
    cmake -S $src -B $src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=121a-real -DGGML_CUDA_NCCL=OFF -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA_CUTLASS=OFF > $LOGS/cmake-$name.log 2>&1 \
      || { echo "$name: cmake FAILED"; ERR=1; continue; }
    grep -q "RDMA transport enabled" $LOGS/cmake-$name.log || echo "$name: WARNING, RPC built without RDMA"
    # at most 8 jobs: CUDA template compiles take 3-6 GB each and both Sparks are shared
    cmake --build $src/build -j ${BUILD_JOBS:-8} --target llama-server llama-bench $rpc > $LOGS/build-$name.log 2>&1 \
      || { echo "$name: build FAILED"; ERR=1; continue; }
    rm -rf $dst.tmp; mkdir -p $dst.tmp && cp $src/build/bin/llama-server $src/build/bin/llama-bench $src/build/bin/$rpc $dst.tmp/ \
      && { [ $rpc = ggml-rpc-server ] || ln -s $rpc $dst.tmp/ggml-rpc-server; } && mv $dst.tmp $dst || { ERR=1; continue; }
  fi
  peer "mkdir -p $W/bin" && rsync -a $dst $PEER_SSH:$W/bin/ || { echo "$name: copy to the peer FAILED"; ERR=1; continue; }
  echo "$name: built $dst"
done
exit $ERR
