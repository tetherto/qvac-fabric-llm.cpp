#!/usr/bin/env bash
# Stage the fabric ggml for the standalone parakeet build. Idempotent.
#
# The shared ggml of the QVAC speech stack is the ggml/ tree of
# qvac-fabric-llm.cpp (the same ggml the LLM stack ships; published to vcpkg
# as the qvac-fabric port). Backend filename-prefix loading, non-Adreno OpenCL
# support and the OpenCL program-binary cache are commits in that tree, so no
# local patches are applied here.
#
# GGML_REF is the single source of truth for which commit parakeet builds
# against; keep it equal to GGML_REF in .github/workflows/parakeet-ci.yml.

set -euo pipefail

GGML_URL="https://github.com/tetherto/qvac-fabric-llm.cpp.git"
GGML_REF="3cefcca8107194a372f015b13f27cd1aac301b67"   # qvac-fabric-llm.cpp v11739.3.0 (temp-11739)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "parakeet.cpp: setting up ggml from ${GGML_URL} @ ${GGML_REF:0:10}"

fetch_pin() {
    git -C ggml-src fetch --depth 1 origin "${GGML_REF}"
    git -C ggml-src checkout -q --detach FETCH_HEAD
}

if [ -d ggml-src/.git ]; then
    echo "  -> existing clone in ggml-src; fetching the pin"
    fetch_pin
else
    echo "  -> cloning into ggml-src"
    rm -rf ggml-src
    git init -q ggml-src
    git -C ggml-src remote add origin "$GGML_URL"
    fetch_pin
fi

if [ -e ggml ] && [ ! -L ggml ]; then
    echo "  ERROR: ./ggml exists and is not a symlink; remove it and re-run this script." >&2
    exit 1
fi
ln -sfn ggml-src/ggml ggml

echo "  -> ok, ggml -> ggml-src/ggml at $(git -C ggml-src rev-parse --short=8 HEAD)"

echo
echo "ggml is ready. Next:"
echo "    cmake -S \"${REPO_ROOT}\" -B \"${REPO_ROOT}/build\" -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build \"${REPO_ROOT}/build\" -j"
