#!/bin/bash
# usage: install-vllm.sh <version> <venv dir>   (update.sh runs it on both nodes)
# installs vLLM <version> into <venv dir> unless a working install is already there, then checks that it imports and
# sees the GPU. The system Python 3.12 exists on both nodes; MAX_JOBS keeps any source build from flooding the box.
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ver=$1; d=$2; check="import vllm, torch; assert torch.cuda.is_available()"
if ! { [ -x $d/bin/python ] && $d/bin/python -c "$check" > /dev/null 2>&1; }; then
  rm -rf $d
  uv venv -q -p /usr/bin/python3.12 $d && MAX_JOBS=4 uv pip install --python $d/bin/python "vllm==$ver" --torch-backend=auto || exit 1
fi
$d/bin/python -c "$check"
