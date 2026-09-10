llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common llama-common-test)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common llama-common-test)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

# POC vector-index smoke test: no model required, no llama runtime touched.
if (TARGET ggml-vector-index)
    add_executable(test-vector-index test-vector-index.cpp)
    target_compile_features(test-vector-index PRIVATE cxx_std_17)
    target_include_directories(test-vector-index PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-vector-index PRIVATE ggml-vector-index)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-vector-index RUNTIME)
    endif()
    llama_test(test-vector-index)

    add_executable(test-vector-index-hooks test-vector-index.cpp)
    target_compile_features(test-vector-index-hooks PRIVATE cxx_std_17)
    target_include_directories(test-vector-index-hooks PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-vector-index-hooks PRIVATE GGML_VEC_INDEX_TEST_HOOKS)
    target_link_libraries(test-vector-index-hooks PRIVATE ggml-vector-index-test)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-vector-index-hooks RUNTIME)
    endif()
    llama_test(test-vector-index-hooks ARGS --internal-hooks)

    add_executable(test-vector-index-faults test-vector-index-faults.cpp)
    target_compile_features(test-vector-index-faults PRIVATE cxx_std_17)
    target_link_libraries(test-vector-index-faults PRIVATE ggml-vector-index-test)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-vector-index-faults RUNTIME)
    endif()
    llama_test(test-vector-index-faults)
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # WebGPU excludes QWEN35 in arch_supported() (test-llama-archs.cpp)
    if (NOT GGML_WEBGPU)
        llama_test(
            test-recurrent-state-rollback
            LABEL main
            ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
        )
        set_tests_properties(test-recurrent-state-rollback PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endif()
    llama_build_and_test(test-kv-cells.cpp)
    target_include_directories(test-kv-cells PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-kv-cache.cpp)
    target_include_directories(test-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)


llama_build(test-multi-instance-repro.cpp)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)
llama_build_and_test(test-unified-memory-props.cpp)
if (GGML_METAL)
    # only a Metal build can enumerate a Metal device
    target_compile_definitions(test-unified-memory-props PRIVATE LLAMA_TEST_EXPECT_METAL)
endif()
llama_build_and_test(test-fit-params.cpp)
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # This test uses internal symbols that are not exported from llama.dll.
    llama_build(test-moe-cache.cpp)
    target_include_directories(test-moe-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-moe-cache)
endif()
if (GGML_RPC)
    llama_build_and_test(test-rpc.cpp)
    set_tests_properties(test-rpc PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

# Vulkan TBQ/PQ copy-to-quant subgroup-size sweep. Spawns itself with different
# GGML_VK_TBQ_COPY_SG_SIZE values to exercise the shader's cross-subgroup stitch
# path on hardware that only exposes one native subgroup size. Auto-skips when no
# GPU backend is available.
llama_build_and_test(test-copy-tbq-subgroups.cpp)

# QVAC-21914: ggml-opencl flash-attention q-chunking parity vs the CPU
# backend. Unconditional and self-skipping at runtime when no OpenCL device is
# present (like test-copy-tbq-subgroups above), so it exercises chunking
# wherever OpenCL exists (Adreno devices, desktop OpenCL, PoCL). Does not link
# mtmd, so it is intentionally outside the LLAMA_MTMD block below.
llama_build_and_test(test-opencl-fa-chunking.cpp)

# QVAC-23763: the CUDA arch-availability registration guard. Self-skips with 77
# when no CUDA registry is present, which today is every runner - there is no
# CUDA hardware in CI, so both cases below skip and this is a fixture for manual
# runs, not a gate. The second registration runs the same binary with virtual
# devices on, which exercises the group-skip path; see the comment in the test
# for what that check can and cannot detect (it catches a gross split, not the
# one-device case).
llama_build_and_test(test-cuda-arch-registration.cpp)
set_tests_properties(test-cuda-arch-registration PROPERTIES SKIP_RETURN_CODE 77)
llama_test(test-cuda-arch-registration NAME test-cuda-arch-registration-virtual)
set_tests_properties(test-cuda-arch-registration-virtual PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "GGML_CUDA_DEVICES=4")

llama_build_and_test(test-model-load-cancel.cpp       LABEL "model")
llama_build_and_test(test-model-load-disk.cpp         LABEL "model")
llama_build_and_test(test-model-load-memory.cpp       LABEL "model")
target_include_directories(test-model-load-memory PRIVATE ${PROJECT_SOURCE_DIR}/common_test)
llama_build_and_test(test-model-load-memory-split.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

# Test sequence token-count memory API
llama_build_and_test(test-memory-seq-token-count.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-memory-seq-token-count PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    target_include_directories(test-quantize-fns PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
if (LLAMA_MTMD)
    set(LLAMA_TEST_NAME test-mtmd-c-api)
    llama_build_and_test(test-mtmd-c-api.c)
    target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
    unset(LLAMA_TEST_NAME)

    # QVAC-21914: clip flash-attention AUTO budget arithmetic (pure CPU).
    set(LLAMA_TEST_NAME test-clip-fa-cutoff)
    llama_build_and_test(test-clip-fa-cutoff.cpp)
    target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
    target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
    unset(LLAMA_TEST_NAME)

    # QVAC-23075: legacy projector alias resolution, which must stay gated on general.name.
    # Resolution is inline in clip-impl.h, so this one links on every platform.
    set(LLAMA_TEST_NAME test-clip-projector-alias)
    llama_build_and_test(test-clip-projector-alias.cpp)
    target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
    target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
    unset(LLAMA_TEST_NAME)

    if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
        # Same reason as the llama tests above, and as the note at clip.h's
        # clip_fa_effective_min_kv: these reach mtmd_calc_idefics3_sizing, clip_init and
        # clip_log_set_callback, which carry no MTMD_API decoration and so are not exported
        # across the Windows mtmd.dll boundary. They still run on every other platform, and on
        # Windows itself in a static build.

        # QVAC-23075: idefics3-family refined size and slice grid against the reference rule
        # (pure CPU, no model).
        set(LLAMA_TEST_NAME test-mtmd-preproc-sizing)
        llama_build_and_test(test-mtmd-preproc-sizing.cpp)
        target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
        target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        unset(LLAMA_TEST_NAME)

        # QVAC-23075: preprocessing metadata rejected at load, on generated metadata-only mmprojs.
        set(LLAMA_TEST_NAME test-clip-preproc-metadata)
        llama_build_and_test(test-clip-preproc-metadata.cpp)
        target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
        target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        unset(LLAMA_TEST_NAME)
    endif()
endif()

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
