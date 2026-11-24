#!/usr/bin/env bash
#
# setup-ggml.sh: stage the fabric ggml for the bundled-ggml dev build of tts-cpp.
#
# The shared ggml of the QVAC speech stack is the ggml/ tree of
# qvac-fabric-llm.cpp (the same ggml the LLM stack ships; published to vcpkg
# as the qvac-fabric port). This script fetches that repository at the pinned
# commit into tts/ggml-src/ and exposes its ggml/ tree as
# tts/ggml/, which TTS_CPP_USE_SYSTEM_GGML=OFF adds as a subdirectory.
#
# Usage:
#   bash tts/scripts/setup-ggml.sh
#   cmake -S tts -B tts/build -DTTS_CPP_USE_SYSTEM_GGML=OFF
#   cmake --build tts/build -j
#
# To move the pin: bump GGML_REF below (keep it equal to GGML_REF in
# .github/workflows/tts-ci.yml) and re-run. The script is idempotent: an
# existing clone is fetched and checked out at the pin without re-cloning.

set -euo pipefail

GGML_REPO_URL="https://github.com/tetherto/qvac-fabric-llm.cpp.git"
GGML_REF="3cefcca8107194a372f015b13f27cd1aac301b67"   # qvac-fabric-llm.cpp v11739.3.0 (temp-11739)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_SRC_DIR="${TTS_CPP_DIR}/ggml-src"
GGML_DIR="${TTS_CPP_DIR}/ggml"

fetch_pin() {
    git -C "${GGML_SRC_DIR}" fetch --depth 1 origin "${GGML_REF}"
    git -C "${GGML_SRC_DIR}" checkout -q --detach FETCH_HEAD
}

if [ -d "${GGML_SRC_DIR}/.git" ]; then
    echo "setup-ggml: existing clone at ${GGML_SRC_DIR}; fetching and checking out pin ${GGML_REF:0:10}"
    fetch_pin
else
    echo "setup-ggml: cloning qvac-fabric-llm.cpp @ ${GGML_REF:0:10} into ${GGML_SRC_DIR}"
    rm -rf "${GGML_SRC_DIR}"
    git init -q "${GGML_SRC_DIR}"
    git -C "${GGML_SRC_DIR}" remote add origin "${GGML_REPO_URL}"
    fetch_pin
fi

if [ -e "${GGML_DIR}" ] && [ ! -L "${GGML_DIR}" ]; then
    echo "setup-ggml: ${GGML_DIR} exists and is not a symlink; remove it and re-run" >&2
    exit 1
fi
ln -sfn "ggml-src/ggml" "${GGML_DIR}"

echo "setup-ggml: tts/ggml -> ggml-src/ggml at $(git -C "${GGML_SRC_DIR}" rev-parse --short HEAD)"
echo "setup-ggml: next: cmake -S tts -B tts/build -DTTS_CPP_USE_SYSTEM_GGML=OFF"
