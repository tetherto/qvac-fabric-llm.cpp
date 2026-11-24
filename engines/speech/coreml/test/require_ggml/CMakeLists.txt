# Configure-time check driven by test-coreml-require-ggml: an engine's
# qvac_coreml_add_tests(REQUIRE_GGML) without a `ggml` target must fail
# configure (the test passes on that error message).
cmake_minimum_required(VERSION 3.20)
project(qvac-coreml-require-ggml LANGUAGES NONE)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../qvac_coreml.cmake)
qvac_coreml_add_tests(REQUIRE_GGML)
