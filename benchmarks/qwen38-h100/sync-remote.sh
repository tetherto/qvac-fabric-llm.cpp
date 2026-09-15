#!/usr/bin/env bash
# Sync the local campaign worktree to the bench host as one incremental patch.
# The snapshot is a git tree object of the last synced working state (no commit), so the patch only holds changes since the last sync.
# Usage: sync-remote.sh init            (record current tree as the synced snapshot)
#        sync-remote.sh push [name]     (apply changes since the snapshot on the remote, save patch copy, advance the snapshot)
set -euo pipefail

REMOTE=${REMOTE:-pratik@cosmicac-b4c09bd2}
REMOTE_REPO=${REMOTE_REPO:-/home/pratik/qwen38-bench/qvac-fabric-llm.cpp}
SNAP_FILE=${SNAP_FILE:-/tmp/fabric-campaign-snapshot-tree}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

snapshot_tree() {
    local idx
    idx=$(mktemp)
    rm -f "$idx"
    GIT_INDEX_FILE="$idx" git read-tree HEAD
    GIT_INDEX_FILE="$idx" git add -A -- ggml include src common tools tests gguf-py conversion convert_hf_to_gguf.py
    GIT_INDEX_FILE="$idx" git write-tree
    rm -f "$idx"
}

case "${1:-}" in
    init)
        snapshot_tree > "$SNAP_FILE"
        echo "snapshot $(cat "$SNAP_FILE")"
        ;;
    push)
        name=${2:-sync}
        base=$(cat "$SNAP_FILE")
        new=$(snapshot_tree)
        patch=$(mktemp)
        git diff "$base" "$new" > "$patch"
        if [ ! -s "$patch" ]; then
            echo "nothing to sync"
            rm -f "$patch"
            exit 0
        fi
        ssh "$REMOTE" "cd $REMOTE_REPO && git apply --check" < "$patch"
        ssh "$REMOTE" "cd $REMOTE_REPO && git apply" < "$patch"
        mkdir -p benchmarks/qwen38-h100/patches
        cp "$patch" "benchmarks/qwen38-h100/patches/$name.patch"
        echo "$new" > "$SNAP_FILE"
        echo "applied $(git diff --stat "$base" "$new" | tail -1) as patches/$name.patch"
        rm -f "$patch"
        ;;
    *)
        echo "usage: $0 init|push [name]" >&2
        exit 2
        ;;
esac
