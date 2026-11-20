# Exporting the derived class must not export its MSVC STL template base.
if (NOT EXISTS "${IMPORT_LIBRARY}")
    message(FATAL_ERROR "Missing ggml-base import library: ${IMPORT_LIBRARY}")
endif()
execute_process(COMMAND "${NM}" --defined-only --extern-only "${IMPORT_LIBRARY}"
    OUTPUT_VARIABLE exports
    COMMAND_ERROR_IS_FATAL ANY)
if (exports MATCHES "[?][$]basic_streambuf@")
    message(FATAL_ERROR "ggml-base exports std::basic_streambuf; export stream members instead of the class")
endif()
