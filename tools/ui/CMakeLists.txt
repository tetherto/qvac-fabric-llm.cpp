set(TARGET llama-ui)

set(LLAMA_UI_HF_BUCKET "ggml-org/llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt UI assets")
set(LLAMA_UI_GZIP      ON                  CACHE BOOL   "Apply gzip compress to assets to save bandwidth")

# Backward compat: forward old var to new one
if(DEFINED LLAMA_BUILD_WEBUI)
    set(LLAMA_BUILD_UI ${LLAMA_BUILD_WEBUI})
    message(DEPRECATION "LLAMA_BUILD_WEBUI is deprecated, use LLAMA_BUILD_UI instead")
endif()
if(DEFINED LLAMA_USE_PREBUILT_WEBUI)
    set(LLAMA_USE_PREBUILT_UI ${LLAMA_USE_PREBUILT_WEBUI})
    message(DEPRECATION "LLAMA_USE_PREBUILT_WEBUI is deprecated, use LLAMA_USE_PREBUILT_UI instead")
endif()
if(DEFINED LLAMA_WEBUI_HF_BUCKET)
    set(LLAMA_UI_HF_BUCKET ${LLAMA_WEBUI_HF_BUCKET})
    message(DEPRECATION "LLAMA_WEBUI_HF_BUCKET is deprecated, use LLAMA_UI_HF_BUCKET instead")
endif()

# Resolve HF asset version: explicit env var > upstream base tag > derived from build number > unset
if(DEFINED ENV{HF_WEBUI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_WEBUI_VERSION}")
    message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
elseif(DEFINED ENV{HF_UI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
else()
    # In this fork LLAMA_BUILD_NUMBER counts the downstream commits too, so
    # "b${LLAMA_BUILD_NUMBER}" names a bucket tag upstream never published; the download
    # then 404s and falls back to 'latest', whose newer-era dist is missing assets this
    # tree's llama-ui-embed requires (loading.html) and the build fails. Pin to the
    # nearest reachable upstream release tag instead; the build-number derivation stays
    # as the fallback for tarball/tagless checkouts.
    execute_process(
        COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" describe --tags --abbrev=0 --match "b[0-9]*"
        OUTPUT_VARIABLE UPSTREAM_BASE_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT "${UPSTREAM_BASE_TAG}" STREQUAL "")
        set(HF_UI_VERSION "${UPSTREAM_BASE_TAG}")
    elseif(DEFINED LLAMA_BUILD_NUMBER)
        set(HF_UI_VERSION "b${LLAMA_BUILD_NUMBER}")
    else()
        set(HF_UI_VERSION "")
    endif()
endif()

if(NOT "${HF_UI_VERSION}" STREQUAL "" AND NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "UI: invalid HF_UI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

if(CMAKE_CROSSCOMPILING)
    find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    if(NOT HOST_CXX_COMPILER)
        message(FATAL_ERROR "UI: no host C++ compiler (g++/clang++) found to build llama-ui-embed; set -DHOST_CXX_COMPILER=<path>")
    endif()
    message(STATUS "UI: building llama-ui-embed with host compiler ${HOST_CXX_COMPILER}")

    if(CMAKE_HOST_WIN32)
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host.exe")
    else()
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host")
    endif()

    add_custom_command(
        OUTPUT  "${LLAMA_UI_EMBED_EXE}"
        COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
                -o "${LLAMA_UI_EMBED_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        COMMENT "Building llama-ui-embed (host)"
        VERBATIM
    )

    # phony target to tie it into the dependency graph
    add_custom_target(llama-ui-embed DEPENDS "${LLAMA_UI_EMBED_EXE}")
else()
    # exclude llama-ui-embed from sanitizer flags,
    # it's a build-time-only tool, no need to instrument it
    # this is to fix TSan "memory layout is incompatible" error on CI
    get_directory_property(_llama_ui_dir_co COMPILE_OPTIONS)
    get_directory_property(_llama_ui_dir_ll LINK_LIBRARIES)
    set(_llama_ui_embed_co ${_llama_ui_dir_co})
    set(_llama_ui_embed_ll ${_llama_ui_dir_ll})
    list(FILTER _llama_ui_embed_co EXCLUDE REGEX ".*-fsanitize=.*")
    list(FILTER _llama_ui_embed_ll EXCLUDE REGEX ".*-fsanitize=.*")
    set_directory_properties(PROPERTIES
        COMPILE_OPTIONS "${_llama_ui_embed_co}"
        LINK_LIBRARIES  "${_llama_ui_embed_ll}")

    add_executable(llama-ui-embed embed.cpp)
    target_compile_features(llama-ui-embed PRIVATE cxx_std_17)
    set_target_properties(llama-ui-embed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set(LLAMA_UI_EMBED_EXE "$<TARGET_FILE:llama-ui-embed>")

    # restore so the llama-ui library below keeps sanitizer instrumentation
    set_directory_properties(PROPERTIES
        COMPILE_OPTIONS "${_llama_ui_dir_co}"
        LINK_LIBRARIES  "${_llama_ui_dir_ll}")
endif()

# Run the provisioning script every build so source changes in tools/ui/ are
# always picked up. The script uses copy_if_different for ui.cpp/ui.h, so the
# library only recompiles when contents actually change.
add_custom_target(llama-ui-assets ALL
    BYPRODUCTS ${UI_CPP} ${UI_H}
    COMMAND ${CMAKE_COMMAND}
        "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DLLAMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DLLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
        "-DHF_BUCKET=${LLAMA_UI_HF_BUCKET}"
        "-DHF_VERSION=${HF_UI_VERSION}"
        "-DHF_ENABLED=${LLAMA_USE_PREBUILT_UI}"
        "-DBUILD_UI=${LLAMA_BUILD_UI}"
        "-DLLAMA_UI_EMBED=${LLAMA_UI_EMBED_EXE}"
        "-DLLAMA_UI_GZIP=${LLAMA_UI_GZIP}"
        -P "${PROJECT_SOURCE_DIR}/scripts/ui-assets.cmake"
    COMMENT "Provisioning UI assets"
    VERBATIM
)

add_dependencies(llama-ui-assets llama-ui-embed)

set_source_files_properties(${UI_CPP} ${UI_H} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${UI_CPP} ${UI_H})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} llama-ui-assets)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
