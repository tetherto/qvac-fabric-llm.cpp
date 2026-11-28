#!/bin/bash
# Download the MiniMax-Music3 safetensors checkpoint from Hugging Face into a
# local directory, ready for scripts/convert-minimax-music3-to-gguf.py. Fetches
# the Comfy-Org single-file repack (the converter's preferred source): the
# plain bf16 text encoder (the global LM; the _pruned_ and _int8_convrot
# repacks are refused by the converter), one DiT precision, and the DAC
# vocoder. Weights are governed by the MiniMax-Music3 Community License.
#
# Usage: download-minimax-music3.sh [--dir DIR] [--fp32]
#   --dir DIR  target directory (default: checkpoints/MiniMax-Music3)
#   --fp32     download the fp32 DiT instead of the fp16 one
#
# Requires the Hugging Face CLI: pip install -U huggingface_hub

set -eu

DIR="checkpoints/MiniMax-Music3"
REPO="Comfy-Org/MiniMax-Music-3"
DIT_FILE="diffusion_models/minimax_music3_dit_fp16.safetensors"
LM_FILE="text_encoders/minimax_music3_text_encoder_bf16.safetensors"
VOCODER_FILE="vae/minimax_music3_dav.safetensors"

usage() {
    cat >&2 << 'EOF'
Usage: download-minimax-music3.sh [--dir DIR] [--fp32]
  --dir DIR  target directory (default: checkpoints/MiniMax-Music3)
  --fp32     download the fp32 DiT instead of the fp16 one
Requires the Hugging Face CLI: pip install -U huggingface_hub
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
            --fp32)
                DIT_FILE="diffusion_models/minimax_music3_dit_fp32.safetensors"
                shift
                ;;
            *)
                usage
                ;;
        esac
    done
}

resolve_hf_cli() {
    if command -v hf > /dev/null 2>&1; then
        echo "hf"
    elif command -v huggingface-cli > /dev/null 2>&1; then
        echo "huggingface-cli"
    else
        echo "error: Hugging Face CLI not found; pip install -U huggingface_hub" >&2
        exit 1
    fi
}

download_file() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[ok] $file"
        return
    fi
    echo "[download] $file <- $REPO"
    "$HF_CLI" download --quiet "$REPO" "$file" --local-dir "$DIR"
}

remove_hf_cache_dirs() {
    find "$DIR" -name '.cache' -type d -exec rm -rf {} + 2> /dev/null || true
}

parse_args "$@"
HF_CLI="$(resolve_hf_cli)"
mkdir -p "$DIR"

download_file "$LM_FILE"
download_file "$DIT_FILE"
download_file "$VOCODER_FILE"

remove_hf_cache_dirs
echo "[done] checkpoint ready in $DIR"
echo "[done] next: python3 scripts/convert-minimax-music3-to-gguf.py --src $DIR --out models/minimax"
