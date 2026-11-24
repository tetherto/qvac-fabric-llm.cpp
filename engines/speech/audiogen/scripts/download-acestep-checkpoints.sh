#!/bin/bash
# Download the ACE-Step v1.5 safetensors checkpoints from Hugging Face into a
# local directory, ready for scripts/convert-acestep-to-gguf.py. The default
# set covers the stages of the validated four-file combinations documented in
# README.md: Qwen3-Embedding-0.6B (text encoder), acestep-5Hz-lm-0.6B (LM),
# acestep-v15-turbo (DiT), and vae. Derived from checkpoints.sh in acestep.cpp
# (github.com/ServeurpersoCom/acestep.cpp, MIT).
#
# Usage: download-acestep-checkpoints.sh [--dir DIR] [--sft]
#   --dir DIR  target directory (default: checkpoints)
#   --sft      also download the acestep-v15-sft DiT variant
#
# Requires the Hugging Face CLI: pip install -U huggingface_hub

set -eu

DIR="checkpoints"
WITH_SFT=0
MAIN_REPO="ACE-Step/Ace-Step1.5"

usage() {
    cat >&2 << 'EOF'
Usage: download-acestep-checkpoints.sh [--dir DIR] [--sft]
  --dir DIR  target directory (default: checkpoints)
  --sft      also download the acestep-v15-sft DiT variant
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
            --sft)
                WITH_SFT=1
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

checkpoint_present() {
    local target="$1"
    [ -d "$target" ] && [ "$(ls "$target"/*.safetensors 2> /dev/null | wc -l)" -gt 0 ]
}

dit_checkpoint_complete() {
    local target="$1"
    checkpoint_present "$target" && [ -f "$target/silence_latent.pt" ]
}

download_from_main_repo() {
    local name="$1"
    case "$name" in
        acestep-v15-*)
            if dit_checkpoint_complete "$DIR/$name"; then
                echo "[ok] $name"
                return
            fi
            ;;
        *)
            if checkpoint_present "$DIR/$name"; then
                echo "[ok] $name"
                return
            fi
            ;;
    esac
    echo "[download] $name <- $MAIN_REPO"
    "$HF_CLI" download --quiet "$MAIN_REPO" --include "$name/*" --local-dir "$DIR"
}

download_repo() {
    local name="$1" repo="$2"
    if checkpoint_present "$DIR/$name"; then
        echo "[ok] $name"
        return
    fi
    echo "[download] $name <- $repo"
    "$HF_CLI" download --quiet "$repo" --local-dir "$DIR/$name"
}

remove_hf_cache_dirs() {
    find "$DIR" -name '.cache' -type d -exec rm -rf {} + 2> /dev/null || true
}

parse_args "$@"
HF_CLI="$(resolve_hf_cli)"
mkdir -p "$DIR"

download_from_main_repo "Qwen3-Embedding-0.6B"
download_repo "acestep-5Hz-lm-0.6B" "ACE-Step/acestep-5Hz-lm-0.6B"
download_from_main_repo "acestep-v15-turbo"
download_from_main_repo "vae"

if [ "$WITH_SFT" -eq 1 ]; then
    download_repo "acestep-v15-sft" "ACE-Step/acestep-v15-sft"
fi

remove_hf_cache_dirs
echo "[done] checkpoints ready in $DIR"
echo "[done] next: python3 scripts/convert-acestep-to-gguf.py --checkpoints $DIR --out models/bf16"
