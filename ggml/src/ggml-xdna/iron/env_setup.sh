#!/bin/bash
# Local env setup for ggml-xdna/iron (adapted from Xilinx/mlir-aie utils/env_setup.sh).
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Usage (from any cwd, with ironenv activated):
#   source /opt/xilinx/xrt/setup.sh
#   source ggml/src/ggml-xdna/iron/env_setup.sh

if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$HOME/ironenv" ]; then
    # shellcheck disable=SC1091
    source "$HOME/ironenv/bin/activate"
fi

MLIR_AIE_INSTALL_DIR="$(python3 - <<'PY'
import pathlib
try:
    import mlir_aie
except ImportError:
    raise SystemExit("")
# namespace package: site-packages/mlir_aie
locs = list(getattr(mlir_aie, "__path__", []))
print(locs[0] if locs else "")
PY
)"

if [ -z "$MLIR_AIE_INSTALL_DIR" ]; then
    echo "ERROR: mlir_aie not found. Activate ~/ironenv first." >&2
    return 1 2>/dev/null || exit 1
fi
export MLIR_AIE_INSTALL_DIR

PEANO_INSTALL_DIR="$(python3 - <<'PY'
from importlib.metadata import distribution
import pathlib
dist = distribution("llvm-aie")
root = pathlib.Path(str(dist.locate_file("llvm-aie")))
print(root if root.exists() else "")
PY
)"

if [ -z "$PEANO_INSTALL_DIR" ] || [ ! -d "$PEANO_INSTALL_DIR" ]; then
    echo "ERROR: llvm-aie (Peano) not found in the active venv." >&2
    return 1 2>/dev/null || exit 1
fi
export PEANO_INSTALL_DIR

export PATH="${MLIR_AIE_INSTALL_DIR}/bin:${PATH}"
export PYTHONPATH="${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

if ! command -v xrt-smi >/dev/null 2>&1; then
    echo "ERROR: xrt-smi not found. Run: source /opt/xilinx/xrt/setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

NPUPAT='NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]'
NPU="$(xrt-smi examine 2>/dev/null | tr -d '\r' | grep -E "$NPUPAT" || true)"
if echo "$NPU" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
    export NPU2=1
else
    export NPU2=0
fi

if ! python3 -c "import pyxrt" 2>/dev/null; then
    echo "ERROR: pyxrt not importable. source /opt/xilinx/xrt/setup.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

echo "IRON env ready:"
echo "  MLIR_AIE_INSTALL_DIR=$MLIR_AIE_INSTALL_DIR"
echo "  PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR"
echo "  NPU2=$NPU2  ($NPU)"
