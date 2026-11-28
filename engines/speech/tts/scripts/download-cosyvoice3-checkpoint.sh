#!/bin/bash
# Download everything the CosyVoice3 converters read: the official
# Fun-CosyVoice3-0.5B checkpoint files from Hugging Face (LM, flow, HiFT,
# config, supervised speech tokenizer, Qwen2 BPE vocab + merges) plus the
# 3D-Speaker CAM++ torch checkpoint the cloning front-end converter needs
# (the campplus.onnx bundled with CosyVoice3 is BN-folded, so
# convert-campplus-to-gguf.py takes the original torch weights instead).
#
# Usage: download-cosyvoice3-checkpoint.sh [--dir DIR]
#   --dir DIR  target directory (default: checkpoints/cosyvoice3)
#
# Requires the Hugging Face CLI: pip install -U huggingface_hub

set -eu

DIR="checkpoints/cosyvoice3"
REPO="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
FILES="llm.pt flow.pt hift.pt cosyvoice3.yaml speech_tokenizer_v3.onnx \
CosyVoice-BlankEN/vocab.json CosyVoice-BlankEN/merges.txt"
CAMPPLUS_FILE="campplus_cn_common.bin"
CAMPPLUS_URL="https://huggingface.co/funasr/campplus/resolve/main/$CAMPPLUS_FILE"

usage() {
    cat >&2 << 'EOF'
Usage: download-cosyvoice3-checkpoint.sh [--dir DIR]
  --dir DIR  target directory (default: checkpoints/cosyvoice3)
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

download_campplus() {
    local dest="$DIR/$CAMPPLUS_FILE"
    local tmp="$dest.tmp"
    if [ -f "$dest" ]; then
        echo "[ok] $CAMPPLUS_FILE"
        return
    fi
    echo "[download] $CAMPPLUS_FILE <- $CAMPPLUS_URL"
    curl -fL --retry 3 -o "$tmp" "$CAMPPLUS_URL"
    mv "$tmp" "$dest"
}

parse_args "$@"
HF_CLI="$(resolve_hf_cli)"
mkdir -p "$DIR"

download_checkpoint_files
download_campplus

echo "[done] checkpoint ready in $DIR"
echo "[done] next: the CosyVoice3 conversion steps in README.md"
