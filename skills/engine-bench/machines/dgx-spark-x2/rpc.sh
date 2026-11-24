#!/bin/bash
# usage: rpc.sh start <bin dir> <log prefix> | rpc.sh stop
# One ggml-rpc-server per node of the pair (SELF, this node, and PEER from pair.conf) on port RPC_PORT, each serving its
# node's GPU over the RDMA transport of the direct ConnectX-7 link. -c keeps a local copy of every tensor it receives,
# so later starts of the same model skip the transfer. The bin dir and the log prefix are the same paths on both nodes
# (build.sh copies the builds over). stop ends only this user's servers on that port: other experiments on these nodes
# run their own rpc-servers on other ports.
W=${BENCH_DIR:-$HOME/bench-pair}
source $W/pair.conf
P=$RPC_PORT
stop() {
  pkill -u $(id -u) -f "ggml-rpc-server[ ].*-p $P" 2>/dev/null
  peer "pkill -u \$(id -u) -f 'ggml-rpc-server[ ].*-p $P'" 2>/dev/null
  local i
  for i in $(seq 1 20); do
    (echo > /dev/tcp/$SELF/$P) 2>/dev/null || (echo > /dev/tcp/$PEER/$P) 2>/dev/null || return 0
    sleep 1
  done
}
case $1 in
  start)
    B=$2; L=$3; stop
    mkdir -p $(dirname $L); peer "mkdir -p $(dirname $L)"
    setsid -f $B/ggml-rpc-server -H $SELF -p $P -d CUDA0 -c > $L.rpc0.log 2>&1 < /dev/null
    peer "setsid -f $B/ggml-rpc-server -H $PEER -p $P -d CUDA0 -c > $L.rpc1.log 2>&1 < /dev/null"
    for i in $(seq 1 60); do
      # the probe connection is refused politely by the server ("Expected HELLO command")
      (echo > /dev/tcp/$SELF/$P) 2>/dev/null && (echo > /dev/tcp/$PEER/$P) 2>/dev/null && exit 0
      sleep 1
    done
    exit 1 ;;
  stop) stop ;;
  *) echo "usage: rpc.sh start <bin dir> <log prefix> | rpc.sh stop"; exit 2 ;;
esac
