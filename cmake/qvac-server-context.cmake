# qvac: build and install upstream's server-context library on its own.
#
# QVAC's decision addon (@qvac/decision-ggml) runs upstream's decision execution
# (server-context: decision parsing, task scheduling, the response queue and the
# post_systemone handler) in-process. It builds this port with the
# server-context feature and links the archive statically; its llama, common
# and mtmd references resolve from the shared @qvac/fabric runtime. Enabling
# LLAMA_BUILD_SERVER for that would also build the web UI, the CLI, the HTTP
# listener (cpp-httplib) and the llama-server executable, none of which the
# addon may contain.
#
# This file adds upstream's tools/server directory, which stops right after
# defining the server-context target when QVAC_FABRIC_SERVER_CONTEXT is on (a
# guarded return() in tools/server/CMakeLists.txt). So the target is upstream's
# own definition, with its sources, include directories, link libraries and
# definitions as a rebase changes them; this file only makes it
# position-independent and hidden, and installs its archive and headers.

set(_qvac_sc_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools/server")

# The library this option ships must be position-independent and hidden (see
# below). With LLAMA_BUILD_SERVER=ON the tools build already defines the target,
# for upstream's own server libraries, and making those hidden would change
# what they export. The two options build different things; refuse the mix.
if (TARGET server-context)
    message(FATAL_ERROR
        "QVAC_FABRIC_SERVER_CONTEXT cannot be combined with LLAMA_BUILD_SERVER: it builds "
        "a position-independent, hidden-visibility server-context for the QVAC decision "
        "addon. Turn one of them off (the qvac-fabric port sets LLAMA_BUILD_SERVER=OFF).")
endif()

if (NOT TARGET llama-common)
    message(FATAL_ERROR "QVAC_FABRIC_SERVER_CONTEXT requires LLAMA_BUILD_COMMON=ON")
endif()
if (NOT LLAMA_MTMD OR NOT TARGET mtmd)
    message(FATAL_ERROR "QVAC_FABRIC_SERVER_CONTEXT requires LLAMA_MTMD=ON")
endif()

# Before the directory: llama_add_compile_flags() sets directory-scope options
# (warnings, LLAMA_FATAL_WARNINGS, LLAMA_SANITIZE_*), which seed the targets
# created after it, as upstream calls it in tools/CMakeLists.txt before
# add_subdirectory(server). No other target is created after this file.
llama_add_compile_flags()
add_subdirectory(tools/server)

if (NOT TARGET server-context)
    message(FATAL_ERROR
        "QVAC_FABRIC_SERVER_CONTEXT: tools/server/CMakeLists.txt no longer defines "
        "server-context before its qvac-fabric return(); update the guard there.")
endif()

# Guard the property that makes this target safe to ship: none of the HTTP
# server or model-router implementation files may be compiled in.
get_target_property(_qvac_sc_sources server-context SOURCES)
foreach(_f IN LISTS _qvac_sc_sources)
    get_filename_component(_name "${_f}" NAME)
    if (_name MATCHES "^(server|server-http|server-connect|server-models|server-models-sched)\\.cpp$")
        message(FATAL_ERROR
            "QVAC_FABRIC_SERVER_CONTEXT: upstream's server-context now compiles ${_name}, "
            "which belongs to the HTTP server. Review before shipping it in the decision addon.")
    endif()
endforeach()

# Linked into a shared addon module, so position-independent regardless of
# BUILD_SHARED_LIBS (upstream only sets it for shared builds).
#
# Hidden visibility: the server classes are internals of the addon that links
# this archive, and the addon exports only its module entry points. A hidden
# symbol is never exported from the addon, whatever its export list or a
# platform's default export behaviour would match.
set_target_properties(server-context PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# Install rules. The headers go to a directory of their own, not next to the
# public llama/common/mtmd headers: they are upstream internals, consumed only by
# the decision addon that links the archive. They are the target's own headers
# plus server-http.h, which server-context.h includes (header-only
# request/response structs) and upstream lists under the HTTP target; together
# they are everything server-context.h includes. The HTTP-only headers
# (server-connect.h, server-cors-proxy.h, server-models*.h) are left out.
#
# The exported llama::server-context target carries the include directories its
# headers need: they include "common.h", "mtmd.h" and their sibling server
# headers without a path, so a consumer only links the target.
set(_qvac_sc_headers "${_qvac_sc_dir}/server-http.h")
foreach(_f IN LISTS _qvac_sc_sources)
    if (_f MATCHES "\\.h$")
        if (NOT IS_ABSOLUTE "${_f}")
            set(_f "${_qvac_sc_dir}/${_f}")
        endif()
        list(APPEND _qvac_sc_headers "${_f}")
    endif()
endforeach()

install(
    TARGETS server-context
    EXPORT llama-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION
        ${LLAMA_INCLUDE_INSTALL_DIR}/server-context
        ${LLAMA_INCLUDE_INSTALL_DIR}/common
        ${LLAMA_INCLUDE_INSTALL_DIR}/mtmd)
install(FILES ${_qvac_sc_headers}
    DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}/server-context)
