#!/bin/bash
# usage: setup.sh       (once, after the first update.sh)
# Rebuilds the frozen prompt set: corpus.txt from the pinned qvac-fabric commit, then prompts.json sliced token-exact
# by a CPU-only llama-server with the 0.8B model. Both files must match the hashes in bench.conf.
set -u
W=${BENCH_DIR:-$HOME/npu-bench}
source $W/bench.conf; source $W/engines.env
PREP_PORT=18090; READY_TRIES=300
TOP=qvac-fabric-llm.cpp-$CORPUS_COMMIT

fail() { echo "SETUP_FAIL $1"; [ -n "${pid:-}" ] && kill $pid 2>/dev/null; exit 1; }
sha_ok() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

rm -rf $W/corpus && mkdir -p $W/corpus
curl -sfL https://codeload.github.com/tetherto/qvac-fabric-llm.cpp/tar.gz/$CORPUS_COMMIT \
  | tar xz -C $W/corpus --strip-components=1 --wildcards "$TOP/README.md" "$TOP/docs/*.md" || fail "corpus download"
( export LC_ALL=C; cd $W/corpus && cat README.md docs/*.md ) > $W/corpus.txt
sha_ok $W/corpus.txt $CORPUS_SHA256 || fail "corpus.txt hash"

GGUF=$(awk '$1 == "0.8b" {print $2}' $W/models.conf)
$FAB_BIN/llama-server -m $MODELS_DIR/$GGUF -ngl 0 --device none --host 127.0.0.1 --port $PREP_PORT > $W/corpus/prep-server.log 2>&1 < /dev/null &
pid=$!
for i in $(seq $READY_TRIES); do curl -sf -m 2 -o /dev/null http://127.0.0.1:$PREP_PORT/health && break; sleep 1; done
python3 -I $W/bench_npu.py prep --server http://127.0.0.1:$PREP_PORT --corpus $W/corpus.txt --contexts $PROMPT_CONTEXTS \
  --out $W/prompts.json || fail "prep"
kill $pid; pid=
sha_ok $W/prompts.json $PROMPTS_SHA256 || fail "prompts.json hash"
echo "SETUP_OK corpus and prompts match bench.conf"
