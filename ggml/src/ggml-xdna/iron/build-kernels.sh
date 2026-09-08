#!/usr/bin/env bash
# Build the ggml-xdna NPU kernel artifacts (xclbin + instruction stream).
#
#   build-kernels.sh <python> <out-dir> <tile> [dma-tile] [work-dir] [rms-row]
#                    [gemm-mb]
#
# By default, builds only the two GEMM artifact pairs. Set
# GGML_XDNA_EXPERIMENTAL=1 to additionally build:
#   ggml-xdna-add-npu2-f32-<tile>.{xclbin,insts.bin}
#   ggml-xdna-mul-npu2-f32-<tile>.{xclbin,insts.bin}
#   ggml-xdna-rms-norm-npu2-f32-<rms-row>.{xclbin,insts.bin}
#   ggml-xdna-add-rms-mul-npu2-f32-<rms-row>.{xclbin,insts.bin}
#   ggml-xdna-gdn-npu2-s128-r32-cs64[-w4].{xclbin,insts.bin}
#
# GEMM artifacts use ggml-xdna-gemm-npu2-bf16-<KT>-<MB>:
#   KT=16 (K=1024) staged as one MemTile chain per column, replayed MB times.
#   Host splits larger K (accumulate C) and loops N in blocks of 512 (NB=1).
#
# SPDX-License-Identifier: MIT

set -euo pipefail

PYTHON="${1:?usage: build-kernels.sh <python> <out-dir> <tile> [dma-tile] [work-dir]}"
OUT_DIR="${2:?missing out-dir}"
TILE="${3:?missing tile}"
DMA_TILE="${4:-256}"
WORK_DIR="${5:-$OUT_DIR}"
RMS_ROW="${6:-1024}"
GEMM_MB="${7:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$PYTHON" ]; then
    echo "ggml-xdna: IRON python not found: $PYTHON" >&2
    exit 1
fi

MLIR_AIE_INSTALL_DIR="$("$PYTHON" - <<'PY'
import mlir_aie
print(list(mlir_aie.__path__)[0])
PY
)"
PEANO_INSTALL_DIR="$("$PYTHON" - <<'PY'
from importlib.metadata import distribution
print(distribution("llvm-aie").locate_file("llvm-aie"))
PY
)"
export MLIR_AIE_INSTALL_DIR PEANO_INSTALL_DIR

XRT_DIR="${XILINX_XRT:-/opt/xilinx/xrt}"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$XRT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:$XRT_DIR/lib:${LD_LIBRARY_PATH:-}"

if ! command -v xclbinutil >/dev/null 2>&1; then
    echo "ggml-xdna: xclbinutil not found (need XRT at $XRT_DIR)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$WORK_DIR"

build_op() {
    local op="$1"
    local n="$2"
    local design="$SCRIPT_DIR/ggml-xdna-$op.py"
    local name="ggml-xdna-$op-npu2-f32-$n"
    local stem="$WORK_DIR/$name"

    "$PYTHON" "$design" \
        -d npu2 \
        -n "$n" \
        --tile-size "$DMA_TILE" \
        --dtype f32 \
        --xclbin-path "$stem.xclbin" \
        --insts-path  "$stem.insts.bin"

    if [ "$WORK_DIR" != "$OUT_DIR" ]; then
        cp -f "$stem.xclbin" "$stem.insts.bin" "$OUT_DIR/"
    fi

    echo "ggml-xdna: built $OUT_DIR/$name.xclbin + .insts.bin"
}

# One GEMM artifact: KT=16 (K=1024 per submit), MB row blocks. Host splits larger
# K and loops over N column blocks. Keep geometry in step with ggml-xdna.cpp.
build_gemm() {
    local kt="$1"
    local mb="$2"
    local name="ggml-xdna-gemm-npu2-bf16-$kt-$mb"
    local stem="$WORK_DIR/$name"

    "$PYTHON" "$SCRIPT_DIR/ggml-xdna-gemm.py" \
        -d npu2 \
        --KT "$kt" \
        --MB "$mb" \
        --grid 4 8 \
        --xclbin-path "$stem.xclbin" \
        --insts-path  "$stem.insts.bin"

    if [ "$WORK_DIR" != "$OUT_DIR" ]; then
        cp -f "$stem.xclbin" "$stem.insts.bin" "$OUT_DIR/"
    fi

    echo "ggml-xdna: built $OUT_DIR/$name.xclbin + .insts.bin (KT=$kt, MB=$mb)"
}

build_gemm 16 "$GEMM_MB"
# Also build MB=32 (M_block=4096) so long prefills can cover a full ubatch in
# one row-block pass. Weight layout is MB-independent; host picks at runtime.
if [ "$GEMM_MB" != "32" ]; then
    build_gemm 16 32
fi
if [ "$GEMM_MB" != "16" ]; then
    build_gemm 16 16
fi

# GDN strip kernel: S=128, ROWS=32, CS=64.
build_gdn() {
    local s="$1" rows="$2" cs="$3" workers="${4:-1}"
    local name="ggml-xdna-gdn-npu2-s${s}-r${rows}-cs${cs}"
    if [ "$workers" -gt 1 ]; then
        name="${name}-w${workers}"
    fi
    local stem="$WORK_DIR/$name"

    "$PYTHON" "$SCRIPT_DIR/ggml-xdna-gdn.py" \
        -d npu2 \
        --S "$s" \
        --ROWS "$rows" \
        --CS "$cs" \
        --workers "$workers" \
        --xclbin-path "$stem.xclbin" \
        --insts-path  "$stem.insts.bin"

    if [ "$WORK_DIR" != "$OUT_DIR" ]; then
        cp -f "$stem.xclbin" "$stem.insts.bin" "$OUT_DIR/"
    fi

    echo "ggml-xdna: built $OUT_DIR/$name.xclbin + .insts.bin (S=$s ROWS=$rows CS=$cs workers=$workers)"
}

if [ "${GGML_XDNA_EXPERIMENTAL:-0}" = "1" ]; then
    build_op add "$TILE"
    build_op mul "$TILE"

    # RMS_NORM normalises one whole row per submission, so its artifact is sized
    # by the model's ne0, not by the elementwise tile.
    build_op rms-norm "$RMS_ROW"
    build_op add-rms-mul "$RMS_ROW"

    build_gdn 128 32 64 1
    build_gdn 128 32 64 4
fi
