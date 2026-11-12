#!/bin/bash
# Downloads the pinned weights with resumable curl (hf download stalled on this box on 2026-10-21).
set -u
D=${MODELS_DIR:-$HOME/models/qwen38-27b}; mkdir -p $D/MTP
get() { # <repo> <rev> <path>
  local u=https://huggingface.co/$1/resolve/$2/$3
  for i in 1 2 3 4 5; do curl -sSfL -C - --retry 5 -o $D/$3 "$u" && return 0; sleep 10; done; return 1
}
U=unsloth/Qwen3.8-27B-GGUF; UR=4ca720788d1e01f1bff70c033e0d0028fd02e502
Z=z-lab/Qwen3.8-27B-DFlash2-GGUF; ZR=2d9571f8ce46e151f61c6499c99dee6079e1d610
get $Z $ZR Qwen3.8-27B-DFlash2-Q4_K_M.gguf && get $U $UR MTP/mtp-Qwen3.8-27B-Q4_0.gguf && get $U $UR Qwen3.8-27B-UD-Q4_K_XL.gguf && echo DLDONE
