#!/bin/bash
# After e2e.sh: build qvac-fabric and the Core ML server with the new prefill attention kernel, then run the
# prompt-length sweep for both through the engine-bench client into runs/2026-10-31-fa.
set -u
W=$HOME/bench-weekly; RUN=$W/runs/2026-10-31-fa
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p $RUN/logs
exec >> $RUN/run.log 2>&1
echo "start $(date)"
until grep -q "^end" ~/fa-work/e2e.log; do sleep 30; done
cd ~/fa-work
cmake -S . -B build-fa -DLLAMA_BUILD_SERVER=ON > $RUN/logs/cfg-fab.log 2>&1 && cmake --build build-fa -j 16 --target llama-server > $RUN/logs/build-fab.log 2>&1 || { echo "fab build failed"; touch $RUN/FAILED; exit 1; }
[ -d $W/src/fabric-coreml-fa ] || cp -Rc $W/src/fabric-coreml $W/src/fabric-coreml-fa
cp ~/fa-work/ggml/src/ggml-metal/kernels/fa_f16.metal $W/src/fabric-coreml-fa/ggml/src/ggml-metal/kernels/fa_f16.metal
cp ~/fa-work/ggml/src/ggml-metal/ggml-metal-ops.cpp $W/src/fabric-coreml-fa/ggml/src/ggml-metal/ggml-metal-ops.cpp
cd $W/src/fabric-coreml-fa && cmake -S examples/ffn-coreml -B build-cml-fa -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DFFN_COREML_DECODER_EXPERIMENT=ON -DFFN_COREML_SERVER=ON > $RUN/logs/cfg-cml.log 2>&1 && \
  cmake --build build-cml-fa -j 16 --target llama-server-coreml > $RUN/logs/build-cml.log 2>&1 || { echo "cml build failed"; touch $RUN/FAILED; exit 1; }
echo "built $(date)"
sed -e "s#^FAB_BIN=.*#FAB_BIN=$HOME/fa-work/build-fa/bin#" -e "s#^FABCML_BIN=.*#FABCML_BIN=$W/src/fabric-coreml-fa/build-cml-fa#" \
    -e "s#^FAB_VER=.*#FAB_VER=temp-11259 @ d3d019a3b + prefill attention kernel#" \
    -e "s#^FABCML_VER=.*#FABCML_VER=temp-11259 @ d3d019a3b + Core ML server (fbb340d) + prefill attention kernel#" \
    $W/runs/2026-10-31-x/engines.env > $RUN/engines.env
export ENGINES_ENV=$RUN/engines.env FABCML_BIN=$W/src/fabric-coreml-fa/build-cml-fa CML_ASSETS=$W/cml/sel-q40
LOCK=/tmp/gpu-bench.lock
BUSY="test-backend-ops|llama-(bench|server|batched|cli)|mlx_lm|sglang.launch|omlx serve|omlx-server|tensorfold-native|splash-[0-9.]+-arm64"
while [ -e $LOCK ] || ps -Ao command | grep -E "$BUSY" | grep -v grep >/dev/null; do sleep 60; done
echo "$USER attention kernel engine bench since $(date +%H:%M)" > $LOCK
trap "rm -f $LOCK" EXIT
caffeinate -i -w $$ &
DP="1024 4096 16384 32768 57344 104857"
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN fabcml ar depth-fabcml-ar
NP=1 CTX=163840 DEPTHS="$DP" $W/run_engine.sh $RUN fab ar depth-fab-ar
NP=1 CTX=16384 $W/run_engine.sh $RUN fab ar spec-fab-ar --mode spec --fixed-prefix --gen 512
python3 $W/summarize.py $RUN --status ok && echo "summary written"
touch $RUN/DONE
echo "end $(date)"
