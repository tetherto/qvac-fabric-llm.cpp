#!/bin/bash
# Download the Audio8 TTS checkpoint files the two Audio8 converters read:
# the DualAR safetensors, the DAC-style codec, the model config, and the
# Qwen2 tokenizer.
#
# Usage: download-audio8-checkpoint.sh [--dir DIR]
#   --dir DIR  target directory (default: checkpoints/Audio8-TTS-Preview-0.6b)
#
# Requires the Hugging Face CLI: pip install -U huggingface_hub

set -eu

DIR="checkpoints/Audio8-TTS-Preview-0.6b"
REPO="Audio8/Audio8-TTS-Preview-0.6b"
FILES="model.safetensors codec.pth config.json tokenizer.json"

usage() {
    cat >&2 << 'EOF'
Usage: download-audio8-checkpoint.sh [--dir DIR]
  --dir DIR  target directory (default: checkpoints/Audio8-TTS-Preview-0.6b)
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

# No existence-based skip: hf download tracks completed files in $DIR/.cache
# and re-fetches truncated or missing ones, which a plain -f test cannot tell
# apart from complete downloads.
download_checkpoint_file() {
    local file="$1"
    echo "[sync] $file <- $REPO"
    "$HF_CLI" download --quiet "$REPO" "$file" --local-dir "$DIR"
}

download_checkpoint_files() {
    local file
    for file in $FILES; do
        download_checkpoint_file "$file"
    done
}

parse_args "$@"
HF_CLI="$(resolve_hf_cli)"
mkdir -p "$DIR"

download_checkpoint_files

echo "[done] checkpoint ready in $DIR"
echo "[done] next: the Audio8 Convert steps in README.md"
