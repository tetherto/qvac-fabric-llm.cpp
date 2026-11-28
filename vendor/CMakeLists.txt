# mtmd needs these even when common is not built
add_subdirectory(hash)
add_subdirectory(miniaudio)
add_subdirectory(nlohmann)
add_subdirectory(sheredom)
add_subdirectory(stb)

# only used by common
if (LLAMA_BUILD_COMMON)
    add_subdirectory(cpp-httplib)
    # cpp-httplib's PARENT_SCOPE set only reaches this directory; forward it to
    # the top level so llama-config.cmake.in can emit find_dependency(OpenSSL)
    set(LLAMA_CPP_HTTPLIB_SYSTEM_OPENSSL ${LLAMA_CPP_HTTPLIB_SYSTEM_OPENSSL} PARENT_SCOPE)
endif()
