function(tts_cpp_set_test_runtime_environment test_name)
    if (WIN32)
        # Shared ggml DLLs live in bin or bin/<Config>, outside the harness directory.
        set(_tts_test_path "${CMAKE_BINARY_DIR}/bin/$<CONFIG>;${CMAKE_BINARY_DIR}/bin;$ENV{PATH}")
        string(REPLACE ";" "\\;" _tts_test_path "${_tts_test_path}")
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PATH=${_tts_test_path}")
    endif()
endfunction()
