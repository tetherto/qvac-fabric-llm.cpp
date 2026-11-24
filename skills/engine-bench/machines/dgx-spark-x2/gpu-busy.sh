#!/bin/bash
# usage: gpu-busy.sh <lock file>...; exit 0 when this node's GPU is taken: any of the lock files exists (the pair's own
# lock, or another bench's, e.g. a single-Spark bench with its own lock), a process holds more than 2 GiB (other
# experiments park ~400 MB ggml-rpc-servers here, which do not count), or the GPU is drawing real power (utilization
# alone can stick at 96% on an idle GB10 after a context goes away)
for l in "$@"; do [ -e "$l" ] && exit 0; done
nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 2048 {f = 1} END {exit !f}' && exit 0
read u p <<< "$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ,)"
[ "${u:-0}" -gt 10 ] 2>/dev/null && [ "${p%.*}" -gt 30 ] 2>/dev/null && exit 0
exit 1
