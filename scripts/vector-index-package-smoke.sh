#!/usr/bin/env bash

set -euo pipefail

: "${PREFIX:?}"
: "${CONSUMER_DIR:=vector-index-consumer}"
: "${LIBDIR:=lib}"
: "${RUN_CMAKE:=1}"
: "${RUN_PKG_CONFIG:=1}"
: "${PKG_CONFIG_STATIC:=0}"

mkdir -p "$CONSUMER_DIR"

cat > "$CONSUMER_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(vector_index_consumer C CXX)
find_package(ggml CONFIG REQUIRED)
add_executable(vector_index_consumer main.c)
target_link_libraries(vector_index_consumer PRIVATE ggml::vector-index)
EOF

cat > "$CONSUMER_DIR/main.c" <<'EOF'
#include "ggml-vector-index.h"

int main(void) {
    ggml_vec_index_t * idx = ggml_vec_index_create(2, 32);
    if (idx == 0) {
        return 1;
    }
    ggml_vec_index_free(idx);
    return 0;
}
EOF

if [[ "$RUN_CMAKE" == "1" ]]; then
    cmake -S "$CONSUMER_DIR" -B "$CONSUMER_DIR/build" \
        -DCMAKE_PREFIX_PATH="$PREFIX/share"
    cmake --build "$CONSUMER_DIR/build"
    "$CONSUMER_DIR/build/vector_index_consumer"
fi

if [[ "$RUN_PKG_CONFIG" == "1" ]]; then
    export PKG_CONFIG_PATH="$PREFIX/$LIBDIR/pkgconfig"

    pkg_config_args=()
    if [[ "$PKG_CONFIG_STATIC" == "1" ]]; then
        pkg_config_args+=(--static)
    fi

    cc "$CONSUMER_DIR/main.c" \
        $(pkg-config --cflags --libs "${pkg_config_args[@]}" ggml-vector-index) \
        -o "$CONSUMER_DIR/vector-index-pkg-config"

    if [[ "$PKG_CONFIG_STATIC" == "1" ]]; then
        "$CONSUMER_DIR/vector-index-pkg-config"
    else
        DYLD_LIBRARY_PATH="$PREFIX/$LIBDIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
            LD_LIBRARY_PATH="$PREFIX/$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$CONSUMER_DIR/vector-index-pkg-config"
    fi
fi
