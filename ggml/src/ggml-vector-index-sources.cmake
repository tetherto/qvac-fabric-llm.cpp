set(GGML_VECTOR_INDEX_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(GGML_VECTOR_INDEX_IMPL_SOURCES
    "${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index.cpp"
    "${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-turbovec.cpp"
    "${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-search.cpp"
    "${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-persistence.cpp")

set(GGML_VECTOR_INDEX_INTERNAL_HEADERS
    "${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-impl.h")
