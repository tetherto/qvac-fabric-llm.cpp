if (TARGET ggml-vector-index)
    set(TARGET llama-vector-index-bench)

    add_executable(${TARGET} bench-vector-index.cpp)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TARGET} PRIVATE ggml-vector-index)

    get_target_property(
        GGML_VECTOR_INDEX_HAS_AVX2
        ggml-vector-index
        GGML_VECTOR_INDEX_HAS_AVX2)
    if (GGML_VECTOR_INDEX_HAS_AVX2)
        target_compile_definitions(${TARGET} PRIVATE GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    endif()

    if (LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
endif()
