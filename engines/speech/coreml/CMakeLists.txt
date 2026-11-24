# Standalone build of the shared Core ML runtime's portable tests, for working
# on coreml without configuring an engine (no ggml needed):
#   cmake -S coreml -B build && cmake --build build && ctest --test-dir build
# Engines fold the runtime into their own libraries via qvac_coreml.cmake.
cmake_minimum_required(VERSION 3.20)
project(qvac-coreml LANGUAGES C CXX)

if (APPLE)
    enable_language(OBJC OBJCXX)
endif()

enable_testing()
# Optional: with a ggml install on CMAKE_PREFIX_PATH (e.g. the ggml-speech
# port), qvac-coreml-tool and the GGUF cases of the generator tests build too.
if (NOT TARGET ggml)
    find_package(ggml CONFIG QUIET)
    if (ggml_FOUND)
        add_library(ggml ALIAS ggml::ggml)
    endif()
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/qvac_coreml.cmake)
qvac_coreml_add_tests()
