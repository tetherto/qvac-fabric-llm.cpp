llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common + llama-common-test, which also bring llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common llama-common-test)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
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
# - LINK: libraries to link against (defaults to llama-common + llama-common-test, which also bring llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common llama-common-test)
    endif()

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
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

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

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp LINK llama)
    llama_build_and_test(test-sampling.cpp LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp LINK llama)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp LINK llama)
    llama_build_and_test(test-batch-alloc.cpp LINK llama)
    llama_build_and_test(test-dflash-draft-vocab.cpp)
    target_include_directories(test-dflash-draft-vocab PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-dflash-loader.cpp)
    target_include_directories(test-dflash-loader PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-server-checkpoint-pool.cpp)
    target_include_directories(test-server-checkpoint-pool PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-checkpoint-pool PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

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

    # TODO: after `test-llama-archs` is refactored to not link internal symbols, we can move all related tests out of
    #       this branch and run them on all platforms
    llama_build(test-llama-archs.cpp)
    llama_test(test-llama-archs NAME test-glm5-kpool-sequences ARGS --glm5-kpool-sequences)
    llama_test(test-llama-archs NAME test-glm5-invalid-metadata ARGS --glm5-invalid-metadata)
    llama_test(test-llama-archs NAME test-mtp-shared-cpu ARGS --mtp-shared-cpu)
    llama_test(test-llama-archs NAME test-hadamard-contracts ARGS --hadamard-contracts)
    llama_test(test-llama-archs NAME test-layer-inp-pos-min ARGS --layer-inp-pos-min)
    llama_test(test-llama-archs NAME test-fit-borrowing-draft ARGS --fit-borrowing-draft)
    llama_test(test-llama-archs NAME test-moe-no-output-ubatches ARGS --moe-no-output-ubatches)
    set_tests_properties(test-moe-no-output-ubatches PROPERTIES ENVIRONMENT GGML_SCHED_DEBUG_REALLOC=1)
    if (GGML_CUDA)
        llama_test(test-llama-archs NAME test-mtp-shared ARGS --mtp-shared)
    endif()
    if (NOT GGML_WEBGPU)
        llama_test(test-llama-archs NAME test-qsa-unified-multiseq ARGS --qsa-unified-multiseq --seed 1)
    endif()

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

    # The dummy-model decode tests below get no benefit from many threads, and a
    # high thread count makes them collapse on shared CI runners: every graph node
    # is a threadpool barrier that needs all n_threads scheduled, so under external
    # CPU contention the wall time degrades superlinearly (measured: the save/load
    # sweep at 32 threads goes from 41 s idle to ~50 s per model with one competing
    # 32-thread load — >90 min for the sweep). The CI workflows export
    # LLAMA_ARG_THREADS=$(nproc); the explicit -t below overrides the env var.

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL main
        ARGS --models "${MODEL_DIR}" -t 4
    )
    # abort on graph reallocations that the initial reserve should have covered
    # TIMEOUT is belt and braces for badly contended runners; the property also
    # overrides the CTest default and the ctest --timeout values passed by the
    # workflow jobs.
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
        TIMEOUT 3600
    )

    # the recurrent-state cases with the deferred gated delta net commit (qwen35, qwen35moe)
    # its own working directory: the per-model logs would collide with test-llama-context under ctest -j
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-llama-context-gdn-deferred")
    llama_test(
        test-llama-context
        NAME test-llama-context-gdn-deferred
        LABEL main
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-llama-context-gdn-deferred"
        ARGS --models "${MODEL_DIR}" -t 4 --test "rollback|replay|shared|swap"
    )
    set_tests_properties(test-llama-context-gdn-deferred PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_GDN_DEFERRED=1;GGML_SCHED_DEBUG_REALLOC=1"
        TIMEOUT 3600
    )

    llama_build(test-fit-measured.cpp)
    llama_test(
        test-fit-measured
        LABEL main
        ARGS "${MODEL_DIR}/llama-dense.gguf"
    )
    set_tests_properties(test-fit-measured PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)

    llama_build_and_test(test-kv-cells.cpp)
    target_include_directories(test-kv-cells PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-kv-cache.cpp)
    target_include_directories(test-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
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
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context NAME test-llama-context-model LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)


llama_build(test-multi-instance-repro.cpp)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
llama_build_and_test(test-model-load-meta-only.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test-mmap-direct-io test-mmap-direct-io.cpp ../src/llama-mmap.cpp ../src/llama-impl.cpp)
    target_include_directories(test-mmap-direct-io PRIVATE ../src ../include)
    target_link_libraries(test-mmap-direct-io PRIVATE ggml)
    target_compile_features(test-mmap-direct-io PRIVATE cxx_std_17)
    target_link_options(test-mmap-direct-io PRIVATE -Wl,--wrap=pread -Wl,--wrap=read -Wl,--wrap=posix_memalign)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-mmap-direct-io RUNTIME)
    endif()
    llama_test(test-mmap-direct-io)

    llama_build_and_test(test-model-load-direct-io.cpp)
    target_include_directories(test-model-load-direct-io PRIVATE ../src ../ggml/src)
    set_tests_properties(test-mmap-direct-io test-model-load-direct-io PROPERTIES SKIP_RETURN_CODE 77)
endif()
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

if (GGML_WEBGPU AND NOT EMSCRIPTEN)
    llama_build_and_test(test-webgpu-trig-range.cpp)
    set_tests_properties(test-webgpu-trig-range PROPERTIES SKIP_RETURN_CODE 77)
    llama_build_and_test(test-webgpu-mul-mat-prec-f32.cpp)
    set_tests_properties(test-webgpu-mul-mat-prec-f32 PROPERTIES SKIP_RETURN_CODE 77)
    llama_build(test-webgpu-registry.cpp)
    if (GGML_BACKEND_DL)
        target_compile_definitions(test-webgpu-registry PRIVATE TEST_WEBGPU_DL)
        add_dependencies(test-webgpu-registry ggml-webgpu)
        llama_test(test-webgpu-registry
            ARGS "$<TARGET_FILE_DIR:ggml-webgpu>" "$<TARGET_FILE:ggml-webgpu>")
        set_tests_properties(test-webgpu-registry PROPERTIES ENVIRONMENT "GGML_WEBGPU_AUTOLOAD=1")
        llama_test(test-webgpu-registry NAME test-webgpu-registry-no-autoload
            ARGS "$<TARGET_FILE_DIR:ggml-webgpu>" "$<TARGET_FILE:ggml-webgpu>" --no-autoload)
        set_tests_properties(test-webgpu-registry-no-autoload PROPERTIES ENVIRONMENT "GGML_WEBGPU_AUTOLOAD=0")
    else()
        target_link_libraries(test-webgpu-registry PRIVATE ggml-webgpu)
        llama_test(test-webgpu-registry)
    endif()
    set_tests_properties(test-webgpu-registry PROPERTIES SKIP_RETURN_CODE 77)
endif()
llama_build_and_test(test-unified-memory-props.cpp)
if (GGML_METAL)
    # only a Metal build can enumerate a Metal device
    target_compile_definitions(test-unified-memory-props PRIVATE LLAMA_TEST_EXPECT_METAL)
endif()
llama_build_and_test(test-fit-params.cpp)
llama_build_and_test(test-fit-logger.cpp)
if (GGML_RPC)
    llama_build_and_test(test-rpc.cpp LINK ggml)
    set_tests_properties(test-rpc PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

    add_executable(test-rpc-transport test-rpc-transport.cpp ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc/transport.cpp ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc/log.cpp)
    target_link_libraries(test-rpc-transport PRIVATE ggml-base)
    target_include_directories(test-rpc-transport PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc)
    target_compile_features(test-rpc-transport PRIVATE cxx_std_17)
    if (WIN32)
        target_link_libraries(test-rpc-transport PRIVATE bcrypt ws2_32)
    endif()
    if (GGML_RPC_RDMA AND NOT APPLE AND NOT WIN32)
        # build transport.cpp the way ggml-rpc does: RDMA on, libibverbs loaded at runtime
        target_sources(test-rpc-transport PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc/ibv-drv.cpp)
        target_compile_definitions(test-rpc-transport PRIVATE GGML_RPC_RDMA GGML_RPC_RDMA_DLOPEN)
        target_link_libraries(test-rpc-transport PRIVATE ${CMAKE_DL_LIBS})
    endif()
    llama_test(test-rpc-transport)

    add_executable(test-rpc-server-port test-rpc-server-port.cpp)
    target_link_libraries(test-rpc-server-port PRIVATE ggml)
    target_compile_features(test-rpc-server-port PRIVATE cxx_std_17)
    if (WIN32)
        target_link_libraries(test-rpc-server-port PRIVATE ws2_32)
    endif()
    llama_test(test-rpc-server-port)
    set_tests_properties(test-rpc-server-port PROPERTIES SKIP_RETURN_CODE 77)

    if (NOT GGML_BACKEND_DL AND GGML_CPU AND NOT LLAMA_USE_SYSTEM_GGML)
        # calls the CPU and RPC backends directly and needs the non-public ggml-backend-impl.h
        add_executable(test-rpc-extras test-rpc-extras.cpp)
        target_link_libraries(test-rpc-extras PRIVATE ggml)
        target_include_directories(test-rpc-extras PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
        target_compile_features(test-rpc-extras PRIVATE cxx_std_17)
        llama_test(test-rpc-extras)
    endif()

    add_executable(test-rpc-allreduce test-rpc-allreduce.cpp)
    target_link_libraries(test-rpc-allreduce PRIVATE ggml)
    target_compile_features(test-rpc-allreduce PRIVATE cxx_std_17)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND AND LLAMA_BUILD_TOOLS AND GGML_CPU AND NOT EMSCRIPTEN)
        function(rpc_allreduce_test name)
            add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_rpc_allreduce.py
                --server $<TARGET_FILE:ggml-rpc-server> ${ARGN})
            set_tests_properties(${name} PROPERTIES TIMEOUT 150 LABELS "rpc")
        endfunction()
        foreach(ranks 2 4 8)
            foreach(wire f32 bf16)
                rpc_allreduce_test(test-rpc-allreduce-${ranks}-${wire}
                    --client $<TARGET_FILE:test-rpc-allreduce> --ranks ${ranks} --wire ${wire})
            endforeach()
        endforeach()
        rpc_allreduce_test(test-rpc-allreduce-port-base
            --client $<TARGET_FILE:test-rpc-allreduce> --ranks 4 --port-base)
        rpc_allreduce_test(test-rpc-allreduce-init-failure
            --client $<TARGET_FILE:test-rpc-allreduce> --ranks 4 --init-failure)
        rpc_allreduce_test(test-rpc-allreduce-disabled
            --client $<TARGET_FILE:test-rpc-allreduce> --ranks 4 --disabled)
        rpc_allreduce_test(test-rpc-local
            --client $<TARGET_FILE:test-rpc> --ranks 2 --wire bf16 --legacy)
    endif()
endif()

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
add_executable(test-hexagon-dma-reader test-hexagon-dma-reader.cpp)
llama_test(test-hexagon-dma-reader)

llama_build(test-backend-ops.cpp LINK ggml)

llama_build_and_test(test-tiled-mulmat.cpp LINK ggml)

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

# QVAC-26299: PNG zlib decode in vendor/stb/stb_image.h stays bounded by the
# IHDR size (decompression bomb) and still decodes valid and Adam7 PNGs. Pure
# CPU, compiles its own stb copy with tracking allocators, so it does not need
# mtmd and sits outside the LLAMA_MTMD block below.
llama_build_and_test(test-stb-png-decode-bound.cpp)
target_include_directories(test-stb-png-decode-bound PRIVATE ${PROJECT_SOURCE_DIR}/vendor)

# Readback of views of OpenCL Q1_0 / PQ2_0 weights, which the backend stores converted. Self-skipping like the above.
llama_build_and_test(test-opencl-quant-views.cpp)

# QVAC-23763: the CUDA arch-availability registration guard. Self-skips with 77
# when no CUDA registry is present, which covers non-CUDA builds and DL builds
# whose module scores 0. A static CUDA build without a GPU runs the
# empty-registry check, and CUDA-equipped runners exercise the kept-device path. The virtual-device case catches a
# gross group split, not the one-device case.
llama_build_and_test(test-cuda-arch-registration.cpp)
set_tests_properties(test-cuda-arch-registration PROPERTIES SKIP_RETURN_CODE 77)
llama_test(test-cuda-arch-registration NAME test-cuda-arch-registration-virtual)
set_tests_properties(test-cuda-arch-registration-virtual PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "GGML_CUDA_DEVICES=4")
# Forces physical device 0 to be skipped: all of a one-GPU host, the mixed
# skipped-then-kept case on a two-GPU host.
llama_test(test-cuda-arch-registration NAME test-cuda-arch-registration-skip)
set_tests_properties(test-cuda-arch-registration-skip PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "GGML_CUDA_TEST_SKIP_DEVICES=0")

# QVAC-23763: backend loader checks that need no GPU. Two stub modules stand in
# for CUDA modules: zero-score rejection, attempted-path dedupe and the
# GGML_DISABLE_CUDA guard. Without GGML_BACKEND_DL the loader is compiled out.
if (GGML_BACKEND_DL AND NOT EMSCRIPTEN)
    foreach (stub IN ITEMS zero ok)
        string(TOUPPER ${stub} stub_upper)
        add_library(test-backend-loader-stub-${stub} MODULE test-backend-loader-stub.cpp)
        target_include_directories(test-backend-loader-stub-${stub} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
        target_link_libraries(test-backend-loader-stub-${stub} PRIVATE ggml-base)
        target_compile_definitions(test-backend-loader-stub-${stub} PRIVATE STUB_NAME="STUB_${stub_upper}")
        # Like the real backend modules (ggml_add_backend_library), so the log forwarding checks also run on Windows.
        if (COMMAND ggml_backend_export_log_functions)
            ggml_backend_export_log_functions(test-backend-loader-stub-${stub})
        endif()
    endforeach()
    target_compile_definitions(test-backend-loader-stub-zero PRIVATE STUB_SCORE=0)
    target_compile_definitions(test-backend-loader-stub-ok   PRIVATE STUB_SCORE=5)

    llama_build_and_test(test-backend-loader.cpp)
    add_dependencies(test-backend-loader test-backend-loader-stub-zero test-backend-loader-stub-ok)
    target_compile_definitions(test-backend-loader PRIVATE
        STUB_ZERO_PATH="$<TARGET_FILE:test-backend-loader-stub-zero>"
        STUB_OK_PATH="$<TARGET_FILE:test-backend-loader-stub-ok>")
    set_tests_properties(test-backend-loader PROPERTIES SKIP_RETURN_CODE 77)

    # Windows only: a third stub imports a DLL built outside the executable's
    # directory, so it loads only when the loader adds %CUDA_PATH%\bin\x64.
    if (WIN32)
        add_library(test-backend-loader-dep SHARED test-backend-loader-dep.cpp)
        set_target_properties(test-backend-loader-dep PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/loader-dep)
        add_library(test-backend-loader-stub-dep MODULE test-backend-loader-stub.cpp)
        target_include_directories(test-backend-loader-stub-dep PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
        target_link_libraries(test-backend-loader-stub-dep PRIVATE ggml-base test-backend-loader-dep)
        target_compile_definitions(test-backend-loader-stub-dep PRIVATE STUB_NAME="STUB_DEP" STUB_SCORE_FROM_DEP)
        add_dependencies(test-backend-loader test-backend-loader-stub-dep)
        target_compile_definitions(test-backend-loader PRIVATE
            STUB_DEP_PATH="$<TARGET_FILE:test-backend-loader-stub-dep>"
            DEP_DLL_PATH="$<TARGET_FILE:test-backend-loader-dep>")
    endif()
endif()

llama_build_and_test(test-model-load-cancel.cpp       LABEL "model")
llama_build_and_test(test-model-load-disk.cpp         LABEL "model")
llama_build_and_test(test-model-load-memory.cpp       LABEL "model")
target_include_directories(test-model-load-memory PRIVATE ${PROJECT_SOURCE_DIR}/common_test)
llama_build_and_test(test-model-load-memory-split.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")
llama_build_and_test(test-laya.cpp)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

# Test sequence token-count memory API
llama_build_and_test(test-memory-seq-token-count.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-memory-seq-token-count PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp LINK ggml)
    target_include_directories(test-quantize-fns PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    llama_build_and_test(test-quantize-perf.cpp LINK ggml)
    llama_build_and_test(test-rope.cpp LINK ggml)
    llama_build_and_test(test-gdn-deferred.cpp LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp LINK ggml)
    llama_build_and_test(test-conv-1d.cpp LINK ggml)
endif()

# libmtmd
if (LLAMA_MTMD)
    llama_build_and_test(test-mtmd-c-api.c LINK mtmd)

    # QVAC-21914: clip flash-attention AUTO budget arithmetic (pure CPU).
    llama_build_and_test(test-clip-fa-cutoff.cpp LINK mtmd)
    target_include_directories(test-clip-fa-cutoff PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)

    # QVAC-23075: legacy projector alias resolution, which must stay gated on general.name.
    # Resolution is inline in clip-impl.h, so this one links on every platform.
    llama_build_and_test(test-clip-projector-alias.cpp LINK mtmd)
    target_include_directories(test-clip-projector-alias PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)

    if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
        # Same reason as the llama tests above, and as the note at clip.h's
        # clip_fa_effective_min_kv: these reach mtmd_calc_idefics3_sizing, clip_init and
        # clip_log_set_callback, which carry no MTMD_API decoration and so are not exported
        # across the Windows mtmd.dll boundary. They still run on every other platform, and on
        # Windows itself in a static build.

        # QVAC-23075: idefics3-family refined size and slice grid against the reference rule
        # (pure CPU, no model).
        llama_build_and_test(test-mtmd-preproc-sizing.cpp LINK mtmd)
        target_include_directories(test-mtmd-preproc-sizing PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)

        # QVAC-23075: preprocessing metadata rejected at load, on generated metadata-only mmprojs.
        llama_build_and_test(test-clip-preproc-metadata.cpp LINK mtmd)
        target_include_directories(test-clip-preproc-metadata PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
    endif()
endif()

llama_build_and_test(test-mtmd-impl.cpp LINK mtmd)

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
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model" LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-tensor-name-length.cpp)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

if (GGML_OPENCL)
    llama_build_and_test(test-opencl-mul-mat-sum-range.cpp)
    set_tests_properties(test-opencl-mul-mat-sum-range PROPERTIES SKIP_RETURN_CODE 77)
endif()
