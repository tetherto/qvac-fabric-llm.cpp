execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env GGML_METAL_DEVICES=0
        "${TEST_BINARY}" MTL 512 64 512 f32 1
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 30)
if (NOT "${result}" STREQUAL "1" OR NOT error MATCHES "FAIL: requested backend has no devices")
    message(FATAL_ERROR "Expected clean no-device failure, got ${result}: ${output}${error}")
endif()
