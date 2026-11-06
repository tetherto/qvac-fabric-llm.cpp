# qvac-fabric: build and install upstream's server-context library on its own.
#
# QVAC hosts upstream's decision execution (server-context: decision parsing,
# task scheduling, the response queue and the post_systemone handler) inside the
# shared @qvac/fabric runtime, behind a QVAC-owned C bridge. Enabling
# LLAMA_BUILD_SERVER for that would also build the web UI, the CLI, the HTTP
# listener (cpp-httplib) and the llama-server executable, none of which the
# runtime may contain. This file builds only the server-context target, with
# the same sources, include directories and link libraries as upstream's
# tools/server/CMakeLists.txt, and installs its archive and headers.
#
# The source list is read from upstream's own tools/server/CMakeLists.txt, so a
# rebase that adds or renames a server-context source is picked up without a
# fork-side edit. If upstream restructures that block, configure fails here
# with a message instead of building a different library silently.

set(_qvac_sc_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools/server")

if (TARGET server-context)
    # The full tools + server build already defined it (LLAMA_BUILD_SERVER=ON).
    # Reuse that target; only the install rules below are added.
    set(QVAC_FABRIC_SERVER_CONTEXT_TARGET server-context)
else()
    if (NOT TARGET llama-common)
        message(FATAL_ERROR "QVAC_FABRIC_SERVER_CONTEXT requires LLAMA_BUILD_COMMON=ON")
    endif()
    if (NOT TARGET mtmd)
        message(FATAL_ERROR "QVAC_FABRIC_SERVER_CONTEXT requires LLAMA_MTMD=ON")
    endif()

    file(READ "${_qvac_sc_dir}/CMakeLists.txt" _qvac_sc_cmake)
    string(REGEX MATCH
        "set\\(TARGET server-context\\)[ \t\r\n]+add_library\\(\\$\\{TARGET\\} STATIC([^)]*)\\)"
        _qvac_sc_match "${_qvac_sc_cmake}")
    if (NOT _qvac_sc_match)
        message(FATAL_ERROR
            "QVAC_FABRIC_SERVER_CONTEXT: could not find the server-context source list in "
            "tools/server/CMakeLists.txt. Upstream changed how that target is declared; update "
            "cmake/qvac-server-context.cmake to match.")
    endif()

    string(REGEX REPLACE "[ \t\r\n]+" ";" _qvac_sc_files "${CMAKE_MATCH_1}")
    list(FILTER _qvac_sc_files EXCLUDE REGEX "^$")

    set(_qvac_sc_sources)
    foreach(_f IN LISTS _qvac_sc_files)
        if (NOT EXISTS "${_qvac_sc_dir}/${_f}")
            message(FATAL_ERROR "QVAC_FABRIC_SERVER_CONTEXT: listed source ${_f} does not exist")
        endif()
        # Guard the property that makes this target safe to ship: none of the
        # HTTP server or model-router implementation files may be compiled in.
        if (_f MATCHES "^(server|server-http|server-models)\\.cpp$")
            message(FATAL_ERROR
                "QVAC_FABRIC_SERVER_CONTEXT: upstream's server-context now lists ${_f}, "
                "which belongs to the HTTP server. Review before shipping it in the runtime.")
        endif()
        list(APPEND _qvac_sc_sources "${_qvac_sc_dir}/${_f}")
    endforeach()

    add_library(server-context STATIC ${_qvac_sc_sources})
    llama_add_compile_flags()
    target_compile_features(server-context PRIVATE cxx_std_17)
    # Linked into the shared @qvac/fabric module, so position-independent
    # regardless of BUILD_SHARED_LIBS (upstream only sets it for shared builds).
    set_target_properties(server-context PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(server-context PRIVATE
        "${_qvac_sc_dir}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/mtmd"
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(server-context PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

    set(QVAC_FABRIC_SERVER_CONTEXT_TARGET server-context)
endif()

# Every server-*.h, not only the ones in the source list: server-context.h
# includes server-http.h (header-only request/response structs), which upstream
# lists under the HTTP target. Headers carry no code into the runtime.
file(GLOB QVAC_FABRIC_SERVER_CONTEXT_HEADERS "${_qvac_sc_dir}/server-*.h")

# Install rules. The headers go to a directory of their own, not next to the
# public llama/common/mtmd headers: they are upstream internals, consumed only by
# the QVAC bridge inside @qvac/fabric, and the npm package must not ship them.
#
# The exported llama::server-context target carries the include directories its
# headers need: they include "common.h", "mtmd.h" and their sibling server
# headers without a path, so a consumer only links the target.
function(qvac_fabric_install_server_context include_dir)
    install(
        TARGETS ${QVAC_FABRIC_SERVER_CONTEXT_TARGET}
        EXPORT llama-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION
            ${include_dir}/server-context
            ${include_dir}/common
            ${include_dir}/mtmd)
    install(FILES ${QVAC_FABRIC_SERVER_CONTEXT_HEADERS}
        DESTINATION ${include_dir}/server-context)
endfunction()
