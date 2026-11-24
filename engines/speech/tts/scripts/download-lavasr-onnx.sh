#!/bin/bash
# Download the LavaSR ONNX release assets the two LavaSR converters read: the
# UL-UNAS denoiser core and the Vocos BWE enhancer backbone + spec head (with
# their external-data files), published by the LavaSRcpp project. The original
# torch weights live at huggingface.co/YatharthS/LavaSR; these ONNX exports
# are the layout convert-lavasr-{denoiser,enhancer}-to-gguf.py consume.
#
# Usage: download-lavasr-onnx.sh [--dir DIR]
#   --dir DIR  target directory (default: checkpoints/lavasr)

set -eu

DIR="checkpoints/lavasr"
RELEASE_URL="https://github.com/Topping1/LavaSRcpp/releases/download/Alpha-v.01"
FILES="denoiser_core_legacy_fixed63.onnx enhancer_backbone.onnx \
enhancer_backbone.onnx.data enhancer_spec_head.onnx enhancer_spec_head.onnx.data"

usage() {
    cat >&2 << 'EOF'
Usage: download-lavasr-onnx.sh [--dir DIR]
  --dir DIR  target directory (default: checkpoints/lavasr)
EOF
    exit 1
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir)
                [ $# -ge 2 ] || usage
                DIR="$2"
                shift 2
                ;;
            *)
                usage
                ;;
        esac
    done
}

download_asset() {
    local file="$1"
    local dest="$DIR/$file"
    local tmp="$dest.tmp"
    if [ -f "$dest" ]; then
        echo "[ok] $file"
        return
    fi
    echo "[download] $file <- $RELEASE_URL"
    curl -fL --retry 3 -o "$tmp" "$RELEASE_URL/$file"
    mv "$tmp" "$dest"
}

download_assets() {
    local file
    for file in $FILES; do
        download_asset "$file"
    done
}

parse_args "$@"
mkdir -p "$DIR"

download_assets

echo "[done] ONNX files ready in $DIR"
echo "[done] next: the LavaSR conversion steps in README.md"
