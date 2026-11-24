#!/bin/bash
# Lists the GPU devices each installed backend sees; used to check GPU access from the detached scheduler.
W=${BENCH_DIR:-$HOME/bench-weekly}; source $W/engines.env
for v in UP_VULKAN_BIN UP_ROCM_BIN; do
  [ -x "${!v}/llama-server" ] && { echo "== $v"; ${!v}/llama-server --list-devices 2>&1 | grep -iE "gfx|radv|8060|Vulkan|ROCm" | head -4; }
done
[ -x "$GUFO_BIN" ] && { echo "== gufo"; LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib $GUFO_BIN diagnose 2>&1 | grep -E "Overall|GPU Model"; }
id
